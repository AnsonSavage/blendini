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
    category = 'Terrain'
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
    category = 'Terrain'
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
    category = 'Terrain'
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


def hf_slope_normal(geo):
    """Per point slope (degrees) and normals of a heightfield."""
    rows, cols = hf_grid(geo)
    H = hf_heights(geo)
    P = geo.P.reshape(rows, cols, 3)
    dx = max(abs(P[0, 1, 0] - P[0, 0, 0]), 1e-9) if cols > 1 else 1.0
    dz = max(abs(P[1, 0, 2] - P[0, 0, 2]), 1e-9) if rows > 1 else 1.0
    gz, gx = np.gradient(H, dz, dx)
    N = np.stack([-gx, np.ones_like(H), -gz], axis=-1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    slope = np.degrees(np.arccos(np.clip(N[..., 1], -1, 1)))
    return slope, N


def _blur2d(M, iterations):
    for _ in range(max(int(iterations), 0)):
        Pd = np.pad(M, 1, mode='edge')
        M = (Pd[:-2, 1:-1] + Pd[2:, 1:-1] + Pd[1:-1, :-2] + Pd[1:-1, 2:] + 4 * M) / 8.0
    return M


def _ramp(x, lo, hi, soft):
    """1 inside [lo, hi], fading to 0 over 'soft' outside."""
    soft = max(soft, 1e-9)
    return np.clip(np.minimum((x - lo) / soft + 1, (hi - x) / soft + 1), 0, 1)


@register_sop
class HeightFieldMaskByFeatureSOP(SOP):
    id = 'hfmaskbyfeature'
    label = "HeightField Mask by Feature"
    category = 'Terrain'
    inputs = 1
    icon = 'MOD_MASK'
    description = "Mask (point float 'mask') from slope, height, direction and noise; shown in red"
    parms = (
        Parm('combine', "Combine with Existing Mask", 'enum', items=[('replace', "Replace", ""), ('mult', "Multiply", ""),
                                                                     ('max', "Maximum", ""), ('add', "Add", "")]),
        Parm('useslope', "Mask by Slope", 'bool', default=True),
        Parm('slopemin', "Min Slope", 'float', default=0.0, min=0, max=90, hide_when={'useslope': False}),
        Parm('slopemax', "Max Slope", 'float', default=30.0, min=0, max=90, hide_when={'useslope': False}),
        Parm('slopesoft', "Slope Falloff", 'float', default=8.0, min=0, max=90, hide_when={'useslope': False}),
        Parm('useheight', "Mask by Height", 'bool', default=False),
        Parm('heightmin', "Min Height", 'float', default=0.0, hide_when={'useheight': False}),
        Parm('heightmax', "Max Height", 'float', default=2.0, hide_when={'useheight': False}),
        Parm('heightsoft', "Height Falloff", 'float', default=0.3, min=0, hide_when={'useheight': False}),
        Parm('usedir', "Mask by Direction", 'bool', default=False),
        Parm('dir', "Facing Direction", 'vector', default=(1.0, 0.0, 0.0), hide_when={'usedir': False}),
        Parm('dirspread', "Direction Spread", 'float', default=60.0, min=0, max=180, hide_when={'usedir': False}),
        Parm('noise', "Noise Amount", 'float', default=0.0, min=0, max=1),
        Parm('noisesize', "Noise Size", 'float', default=2.0, min=0.01, hide_when={'noise': 0.0}),
        Parm('blur', "Blur Iterations", 'int', default=1, min=0, max=50),
        Parm('invert', "Invert", 'bool', default=False),
        Parm('visualize', "Visualize as Cd", 'bool', default=True),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        rows, cols = hf_grid(geo)
        slope, N = hf_slope_normal(geo)
        H = hf_heights(geo)
        M = np.ones((rows, cols))
        if p.useslope:
            M *= _ramp(slope, p.slopemin, p.slopemax, p.slopesoft)
        if p.useheight:
            M *= _ramp(H, p.heightmin, p.heightmax, p.heightsoft)
        if p.usedir:
            d = np.asarray(p.dir, dtype=np.float64)
            d = np.array([d[0], 0.0, d[2]])
            ln = np.linalg.norm(d)
            if ln > 1e-9:
                flat = N.copy()
                flat[..., 1] = 0
                fl = np.linalg.norm(flat, axis=-1)
                cosang = np.einsum('ijk,k->ij', flat, d / ln) / np.maximum(fl, 1e-9)
                ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                M *= np.where(fl > 1e-4, _ramp(ang, -1, p.dirspread / 2, 10.0), 0.0)
        if p.noise > 0:
            nz = _noise.noise(geo.P / max(p.noisesize, 1e-3)).reshape(rows, cols)
            M = M * (1 - p.noise) + M * nz * p.noise * 2
        M = np.clip(_blur2d(M, p.blur), 0, 1)
        if p.invert:
            M = 1 - M
        old = geo.attribs[POINT].get('mask')
        if old is not None and old.dtype != object and p.combine != 'replace':
            o = np.asarray(old, dtype=np.float64).reshape(rows, cols)
            M = {'mult': o * M, 'max': np.maximum(o, M), 'add': np.clip(o + M, 0, 1)}[p.combine]
        geo.attribs[POINT]['mask'] = M.ravel()
        if p.visualize:
            base = np.full((rows * cols, 3), 0.75)
            geo.attribs[POINT]['Cd'] = base * (1 - M.ravel()[:, None]) + np.array([0.9, 0.1, 0.05]) * M.ravel()[:, None]
        return geo


@register_sop
class HeightFieldScatterSOP(SOP):
    id = 'hfscatter'
    label = "HeightField Scatter"
    category = 'Terrain'
    inputs = 1
    icon = 'OUTLINER_OB_POINTCLOUD'
    description = "Scatter points on a heightfield, weighted by its mask, with pscale, N and up for Copy to Points"
    parms = (
        Parm('count', "Point Count", 'int', default=500, min=0, soft_max=20000),
        Parm('usemask', "Use Mask", 'bool', default=True),
        Parm('seed', "Seed", 'int', default=0),
        Parm('pscalemin', "Min Scale", 'float', default=0.6, min=0.0),
        Parm('pscalemax', "Max Scale", 'float', default=1.2, min=0.0),
        Parm('alignnormal', "Align to Terrain Normal", 'float', default=0.2, min=0.0, max=1.0,
             description="0: upright, 1: along the terrain normal"),
        Parm('randomyaw', "Random Rotation", 'bool', default=True),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        rows, cols = hf_grid(geo)
        out = Geometry()
        if p.count <= 0:
            return out
        P = geo.P.reshape(rows, cols, 3)
        _slope, N = hf_slope_normal(geo)
        M = np.ones((rows, cols))
        if p.usemask and 'mask' in geo.attribs[POINT]:
            M = np.clip(np.asarray(geo.attribs[POINT]['mask'], dtype=np.float64).reshape(rows, cols), 0, None)
        # Cells weighted by the average mask of their corners.
        W = (M[:-1, :-1] + M[1:, :-1] + M[:-1, 1:] + M[1:, 1:]).ravel() / 4
        if W.sum() <= 0:
            return out
        rng = np.random.default_rng(int(p.seed) & 0x7FFFFFFF)
        cells = rng.choice(len(W), size=p.count, p=W / W.sum())
        r, c = np.divmod(cells, cols - 1)
        u, v = rng.random(p.count), rng.random(p.count)

        def bil(A):
            a00, a01, a10, a11 = A[r, c], A[r, c + 1], A[r + 1, c], A[r + 1, c + 1]
            uu, vv = u[:, None] if A.ndim == 3 else u, v[:, None] if A.ndim == 3 else v
            return (a00 * (1 - uu) + a01 * uu) * (1 - vv) + (a10 * (1 - uu) + a11 * uu) * vv
        out.P = bil(P)
        n = bil(N)
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        upright = np.array([0.0, 1.0, 0.0])
        up = upright * (1 - p.alignnormal) + n * p.alignnormal
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        yaw = rng.random(p.count) * 2 * np.pi if p.randomyaw else np.zeros(p.count)
        fwd = np.stack([np.cos(yaw), np.zeros(p.count), np.sin(yaw)], axis=1)
        fwd -= up * np.einsum('ij,ij->i', fwd, up)[:, None]
        fwd /= np.maximum(np.linalg.norm(fwd, axis=1, keepdims=True), 1e-9)
        # Copy to Points convention: N is the forward (Z) axis, up the Y axis.
        out.attribs[POINT]['N'] = fwd
        out.attribs[POINT]['up'] = up
        out.attribs[POINT]['pscale'] = p.pscalemin + rng.random(p.count) * (p.pscalemax - p.pscalemin)
        out.attribs[POINT]['mask'] = bil(M)
        return out


@register_sop
class HeightFieldTerraceSOP(SOP):
    id = 'hfterrace'
    label = "HeightField Terrace"
    category = 'Terrain'
    inputs = 1
    icon = 'IPO_CONSTANT'
    description = "Step the terrain into terraces (optionally only where the mask is)"
    parms = (
        Parm('spacing', "Step Height", 'float', default=0.5, min=0.01, soft_max=5.0),
        Parm('sharpness', "Sharpness", 'float', default=0.8, min=0.0, max=1.0),
        Parm('offset', "Offset", 'float', default=0.0),
        Parm('usemask', "Use Mask", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        rows, cols = hf_grid(geo)
        H = hf_heights(geo)
        x = (H - p.offset) / p.spacing
        f = np.floor(x)
        t = x - f
        # Smooth staircase: flat treads, steep risers controlled by sharpness.
        k = 1 + p.sharpness * 20
        step = (np.tanh((t - 0.5) * k) / np.tanh(0.5 * k) + 1) / 2 if p.sharpness > 0 else t
        Hn = (f + step) * p.spacing + p.offset
        if p.usemask and 'mask' in geo.attribs[POINT]:
            m = np.clip(np.asarray(geo.attribs[POINT]['mask'], dtype=np.float64).reshape(rows, cols), 0, 1)
            Hn = H * (1 - m) + Hn * m
        return hf_set(geo, Hn)


@register_sop
class HeightFieldBlurSOP(SOP):
    id = 'hfblur'
    label = "HeightField Blur"
    category = 'Terrain'
    inputs = 1
    icon = 'MOD_SMOOTH'
    parms = (
        Parm('layer', "Layer", 'enum', items=[('height', "Height", ""), ('mask', "Mask", "")]),
        Parm('iterations', "Iterations", 'int', default=5, min=0, max=200),
        Parm('usemask', "Only Where Masked", 'bool', default=False, hide_when={'layer': 'mask'}),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        rows, cols = hf_grid(geo)
        if p.layer == 'mask':
            m = geo.attribs[POINT].get('mask')
            if m is None:
                return geo
            geo.attribs[POINT]['mask'] = _blur2d(np.asarray(m, dtype=np.float64).reshape(rows, cols), p.iterations).ravel()
            return geo
        H = hf_heights(geo)
        Hb = _blur2d(H, p.iterations)
        if p.usemask and 'mask' in geo.attribs[POINT]:
            m = np.clip(np.asarray(geo.attribs[POINT]['mask'], dtype=np.float64).reshape(rows, cols), 0, 1)
            Hb = H * (1 - m) + Hb * m
        return hf_set(geo, Hb)
