# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Heightfields (terrain): a grid of points with a 'height' that the HeightField SOPs edit."""

import numpy as np

from .base import SOP, Parm, register_sop, CookError
from .generators import grid_quads
from ..core.geometry import Geometry, POINT, DETAIL
from ..core import noise as _noise


def _in(inputs, i=0):
    if len(inputs) > i and inputs[i] is not None:
        return inputs[i]
    return Geometry()


def hf_grid(geo):
    """(rows, cols) of a heightfield geometry."""
    res = geo.attribs[DETAIL].get('hf_res')
    if res is None:
        raise CookError("Input is not a heightfield (use a HeightField SOP)")
    rows, cols = (int(x) for x in np.asarray(res).ravel()[:2])
    if rows * cols != geo.num_points:
        raise CookError("Heightfield resolution doesn't match the point count")
    return rows, cols


def hf_heights(geo):
    rows, cols = hf_grid(geo)
    return geo.P[:, 1].reshape(rows, cols).copy()


def hf_set(geo, H):
    P = geo.P.copy()
    P[:, 1] = H.ravel()
    geo.P = P
    geo.attribs[POINT]['height'] = H.ravel().copy()
    return geo


@register_sop
class HeightFieldSOP(SOP):
    id = 'heightfield'
    label = "HeightField"
    category = 'Primitive'
    icon = 'RNDCURVE'
    parms = (
        Parm('size', "Size", 'vector2', default=(20.0, 20.0)),
        Parm('gridspacing', "Grid Spacing", 'float', default=0.1, min=0.005, soft_max=1),
        Parm('initval', "Initial Height", 'float', default=0.0),
    )

    def cook(self, inputs, p, ctx):
        cols = max(int(round(p.size[0] / p.gridspacing)) + 1, 2)
        rows = max(int(round(p.size[1] / p.gridspacing)) + 1, 2)
        if rows * cols > 4_000_000:
            raise CookError("Heightfield too large (%d x %d)" % (cols, rows))
        x = np.linspace(-p.size[0] / 2, p.size[0] / 2, cols)
        z = np.linspace(-p.size[1] / 2, p.size[1] / 2, rows)
        X, Z = np.meshgrid(x, z)
        P = np.stack([X.ravel(), np.full(X.size, p.initval), Z.ravel()], axis=1)
        geo = Geometry.from_polys(P)
        quads = grid_quads(cols, rows)
        geo.add_prims_flat(np.full(len(quads), 4), quads.ravel())
        geo.attribs[DETAIL]['hf_res'] = np.array([[rows, cols]], dtype=np.int64)
        geo.attribs[DETAIL]['hf_spacing'] = np.array([p.gridspacing])
        geo.attribs[POINT]['height'] = np.full(geo.num_points, p.initval)
        return geo


@register_sop
class HeightFieldNoiseSOP(SOP):
    id = 'hfnoise'
    label = "HeightField Noise"
    category = 'Manipulate'
    inputs = 1
    icon = 'MOD_NOISE'
    parms = (
        Parm('combine', "Combine Method", 'enum', items=[('add', "Add", ""), ('max', "Maximum", ""), ('mult', "Multiply", ""),
                                                        ('set', "Replace", "")]),
        Parm('amp', "Amplitude", 'float', default=3.0, soft_min=0, soft_max=20),
        Parm('elementsize', "Element Size", 'float', default=8.0, min=0.01, soft_max=50),
        Parm('offset', "Offset", 'vector'),
        Parm('octaves', "Octaves", 'int', default=6, min=1, max=12),
        Parm('roughness', "Roughness", 'float', default=0.5, min=0, max=1),
        Parm('ridged', "Ridged", 'bool', default=False, description="Ridged multifractal (sharp mountain crests)"),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        H = hf_heights(geo)
        pos = geo.P.copy()
        pos[:, 1] = 0
        pos = pos / max(p.elementsize, 1e-6) + np.asarray(p.offset)
        if p.ridged:
            total = np.zeros(len(pos))
            amp, freq, norm = 1.0, 1.0, 0.0
            for _ in range(p.octaves):
                n = 1.0 - np.abs(_noise.snoise(pos * freq))
                total += n * n * amp
                norm += amp
                amp *= p.roughness
                freq *= 2.0
            n = total / norm
        else:
            n = np.clip(_noise.fbm(pos, p.octaves, 2.0, p.roughness) * 1.6 + 0.5, 0.0, None)
        n = (n * p.amp).reshape(H.shape)
        if p.combine == 'add':
            H = H + n
        elif p.combine == 'max':
            H = np.maximum(H, n)
        elif p.combine == 'mult':
            H = H * n
        else:
            H = n
        return hf_set(geo, H)


def _neighbour_diffs(H):
    """Height differences to the 4 neighbours (positive = neighbour is lower), padded at borders."""
    Hp = np.pad(H, 1, mode='edge')
    return [H - Hp[1:-1, :-2], H - Hp[1:-1, 2:], H - Hp[:-2, 1:-1], H - Hp[2:, 1:-1]]


def _shift_add(arr, out, k):
    """Add arr into out at the neighbour k (inverse of _neighbour_diffs ordering)."""
    if k == 0:
        out[:, :-1] += arr[:, 1:]
    elif k == 1:
        out[:, 1:] += arr[:, :-1]
    elif k == 2:
        out[:-1, :] += arr[1:, :]
    else:
        out[1:, :] += arr[:-1, :]


def flow_accumulation(H):
    """D8-lite (4 neighbour) flow accumulation: how much upstream area drains through each cell."""
    rows, cols = H.shape
    order = np.argsort(H.ravel())[::-1]
    acc = np.ones(rows * cols)
    Hf = H.ravel()
    # Steepest lower neighbour per cell.
    Hp = np.pad(H, 1, mode='edge')
    neigh = np.stack([Hp[1:-1, :-2], Hp[1:-1, 2:], Hp[:-2, 1:-1], Hp[2:, 1:-1]], axis=0).reshape(4, -1)
    drop = Hf[None, :] - neigh
    best = np.argmax(drop, axis=0)
    has = drop[best, np.arange(rows * cols)] > 0
    idx = np.arange(rows * cols)
    r, c = idx // cols, idx % cols
    tr = np.where(best == 2, r - 1, np.where(best == 3, r + 1, r))
    tc = np.where(best == 0, c - 1, np.where(best == 1, c + 1, c))
    target = np.clip(tr, 0, rows - 1) * cols + np.clip(tc, 0, cols - 1)
    target = np.where(has, target, -1)
    for i in order.tolist():
        t = target[i]
        if t >= 0:
            acc[t] += acc[i]
    return acc.reshape(rows, cols)


@register_sop
class HeightFieldErodeSOP(SOP):
    id = 'hferode'
    label = "HeightField Erode"
    category = 'Manipulate'
    inputs = 1
    icon = 'FORCE_TURBULENCE'
    parms = (
        Parm('iterations', "Iterations", 'int', default=30, min=0, soft_max=200),
        Parm('talus', "Thermal Repose Angle", 'float', default=35.0, min=0, max=89),
        Parm('thermal', "Thermal Rate", 'float', default=0.25, min=0, max=0.5),
        Parm('hydro', "Hydraulic Erosion", 'float', default=0.004, min=0, soft_max=0.05),
        Parm('flowexp', "Flow Exponent", 'float', default=0.5, min=0, max=2),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        H = hf_heights(geo)
        spacing = float(np.asarray(geo.attribs[DETAIL].get('hf_spacing', [[0.1]])).ravel()[0])
        talus = np.tan(np.radians(p.talus)) * spacing
        flow = np.zeros_like(H)
        for it in range(max(p.iterations, 0)):
            # Thermal: material above the repose angle slides to lower neighbours.
            diffs = _neighbour_diffs(H)
            excess = [np.maximum(d - talus, 0) for d in diffs]
            total = sum(excess)
            safe = np.where(total == 0, 1, total)
            maxd = np.maximum.reduce(excess)
            moved = maxd * p.thermal
            out = np.zeros_like(H)
            for k, e in enumerate(excess):
                share = moved * e / safe
                out -= share
                _shift_add(share, out, k)
            H = H + out
            # Hydraulic (stream power): carve where lots of water flows over steep ground.
            if p.hydro > 0 and (it % 5 == 0 or it == p.iterations - 1):
                flow = flow_accumulation(H)
            if p.hydro > 0:
                slope = np.maximum.reduce([np.maximum(d, 0) for d in _neighbour_diffs(H)]) / spacing
                H = H - p.hydro * (flow ** p.flowexp) * slope * spacing
        geo = hf_set(geo, H)
        if p.hydro > 0:
            geo.attribs[POINT]['flow'] = np.log1p(flow.ravel())
        return geo
