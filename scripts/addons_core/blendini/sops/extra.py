# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""More everyday SOPs: Enumerate, Group by Range, Switch-If, Revolve, Skin, Path Deform,
Distance from Geometry, Lattice."""

import numpy as np

from .base import SOP, Parm, register_sop, parse_group_pattern, CookError
from ..core.geometry import Geometry, POINT, DETAIL
from .modify import _in, curve_frames, CLASS_PP

AXES = [('x', "X", ""), ('y', "Y", ""), ('z', "Z", "")]
AXIS_INDEX = {'x': 0, 'y': 1, 'z': 2}


@register_sop
class EnumerateSOP(SOP):
    id = 'enumerate'
    label = "Enumerate"
    category = 'Attribute'
    inputs = 1
    icon = 'LINENUMBERS_ON'
    description = "Number elements (or pieces) into an integer or string attribute"
    parms = (
        Parm('class', "Run Over", 'enum', items=CLASS_PP),
        Parm('group', "Group", 'string'),
        Parm('attrib', "Attribute", 'string', default='index'),
        Parm('mode', "Mode", 'enum', items=[
            ('element', "Elements", "0, 1, 2 ... over the selected elements"),
            ('piece', "Pieces", "Index of the piece (by first occurrence of the piece attribute value)"),
            ('inpiece', "Elements Within Piece", "Restart the count for every piece"),
        ]),
        Parm('pieceattrib', "Piece Attribute", 'string', default='name', hide_when={'mode': 'element'}),
        Parm('start', "Start Value", 'int', default=0),
        Parm('prefix', "String Prefix", 'string',
             description="Non-empty: create a string attribute such as piece0, piece1 ..."),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        cls = p['class']
        n = geo.count(cls)
        mask = parse_group_pattern(geo, cls, p.group)
        sel = np.nonzero(mask)[0]
        if p.mode == 'element':
            idx = np.arange(len(sel))
        else:
            key = geo.attribs[cls].get(p.pieceattrib)
            if key is None:
                raise CookError("Enumerate: no %s attribute '%s'" % (cls, p.pieceattrib))
            k = key[sel]
            if k.ndim > 1:
                k = [tuple(r) for r in k.tolist()]
            else:
                k = k.tolist()
            ids = {}
            piece = np.array([ids.setdefault(v, len(ids)) for v in k], dtype=np.int64)
            if p.mode == 'piece':
                idx = piece
            else:
                idx = np.zeros(len(sel), dtype=np.int64)
                counter = {}
                for i, pc in enumerate(piece.tolist()):
                    idx[i] = counter.get(pc, 0)
                    counter[pc] = idx[i] + 1
        idx = idx + p.start
        name = p.attrib.strip() or 'index'
        if p.prefix:
            arr = geo.attribs[cls].get(name)
            if arr is None or arr.dtype != object:
                arr = np.full(n, '', dtype=object)
            arr = arr.copy()
            arr[sel] = [p.prefix + str(v) for v in idx.tolist()]
        else:
            arr = geo.attribs[cls].get(name)
            if arr is None or arr.dtype.kind not in 'iu' or arr.ndim != 1:
                arr = np.zeros(n, dtype=np.int64)
            arr = arr.copy()
            arr[sel] = idx
        geo.attribs[cls][name] = arr
        return geo


@register_sop
class GroupRangeSOP(SOP):
    id = 'grouprange'
    label = "Group by Range"
    category = 'Group'
    inputs = 1
    icon = 'GROUP'
    description = "Select every Nth element in a range (\"select 1 of 3\")"
    parms = (
        Parm('groupname', "Group Name", 'string', default='range1'),
        Parm('class', "Group Type", 'enum', items=CLASS_PP),
        Parm('start', "Range Start", 'int', default=0),
        Parm('end', "Range End", 'int', default=-1, description="Negative values count from the end (-1 is the last)"),
        Parm('select', "Select", 'int', default=1, min=0),
        Parm('of', "Of", 'int', default=2, min=1),
        Parm('offset', "Offset", 'int', default=0),
        Parm('invert', "Invert", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        cls = p['class']
        n = geo.count(cls)
        end = p.end if p.end >= 0 else n + p.end
        i = np.arange(n)
        inside = (i >= p.start) & (i <= end)
        of = max(p.of, 1)
        pattern = ((i - p.start - p.offset) % of) < p.select
        mask = inside & pattern
        if p.invert:
            mask = ~mask
        geo.groups[cls][p.groupname.strip() or 'range1'] = mask
        return geo


@register_sop
class SwitchIfSOP(SOP):
    id = 'switchif'
    label = "Switch-If"
    category = 'Utility'
    inputs = 2
    input_labels = ("If True", "If False")
    icon = 'QUESTION'
    description = "Pass the first input when the test is true, otherwise the second"
    parms = (
        Parm('test', "Test", 'enum', items=[
            ('expression', "VEX Expression", "Evaluated once (detail) with the inputs bound to 0 and 1"),
            ('attrib', "Attribute Exists", ""),
            ('empty', "Input Is Empty", "True when the first input has no points"),
        ]),
        Parm('expr', "Expression", 'string', default='npoints(0) > 0', hide_when={'test': ('attrib', 'empty')}),
        Parm('attribclass', "Attribute Class", 'enum', items=[('point', "Point", ""), ('prim', "Primitive", ""),
                                                             ('vertex', "Vertex", ""), ('detail', "Detail", "")],
             hide_when={'test': ('expression', 'empty')}),
        Parm('attrib', "Attribute", 'string', default='Cd', hide_when={'test': ('expression', 'empty')}),
    )

    def condition(self, a, b, p):
        if p.test == 'empty':
            return a.num_points == 0
        if p.test == 'attrib':
            return p.attrib.strip() in a.attribs[p.attribclass]
        from ..vex.wrangle import run_wrangle
        code = "i@__cond = (%s) ? 1 : 0;" % (p.expr.strip().rstrip(';') or '0')
        out, _ = run_wrangle(code, [a.copy(), b], run_over=DETAIL)
        cond = out.attribs[DETAIL].get('__cond')
        return bool(cond is not None and np.asarray(cond).ravel()[0])

    def cook(self, inputs, p, ctx):
        a, b = _in(inputs, 0), _in(inputs, 1)
        return (a if self.condition(a, b, p) else b).copy()


def rotate_about_axis(P, origin, axis, angle):
    """Rotate points (n, 3) around an axis by angle (radians), Rodrigues formula."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / max(np.linalg.norm(k), 1e-12)
    v = P - origin
    c, s = np.cos(angle), np.sin(angle)
    return origin + v * c + np.cross(k, v) * s + np.outer(v @ k, k) * (1 - c)


def _prim_lists(geo):
    """(points, closed) for every primitive, or the points as one curve when there are no primitives."""
    if geo.num_prims == 0:
        return [(np.arange(geo.num_points), False)] if geo.num_points > 1 else []
    return [(geo.prim_points(i), bool(geo.prim_closed[i])) for i in range(geo.num_prims)]


def _copy_point_attribs(src, dst, idx, skip=('P',)):
    for name, arr in src.attribs[POINT].items():
        if name not in skip:
            dst.attribs[POINT][name] = arr[idx]


@register_sop
class RevolveSOP(SOP):
    id = 'revolve'
    label = "Revolve"
    category = 'Polygon'
    inputs = 1
    input_labels = ("Profile Curves",)
    icon = 'MOD_SCREW'
    description = "Spin profile curves around an axis into a surface (lathe)"
    parms = (
        Parm('origin', "Origin", 'vector'),
        Parm('dir', "Axis Direction", 'vector', default=(0.0, 1.0, 0.0)),
        Parm('divs', "Divisions", 'int', default=16, min=1, max=512),
        Parm('angle_start', "Start Angle", 'float', default=0.0),
        Parm('angle_end', "End Angle", 'float', default=360.0),
        Parm('endcaps', "End Caps", 'bool', default=False, description="Close partial sweeps of closed profiles"),
    )

    def cook(self, inputs, p, ctx):
        src = _in(inputs)
        a0, a1 = np.radians(p.angle_start), np.radians(p.angle_end)
        full = abs(abs(p.angle_end - p.angle_start) - 360.0) < 1e-6
        divs = max(p.divs, 1)
        rings = divs if full else divs + 1
        out = Geometry()
        origin = np.asarray(p.origin, dtype=np.float64)
        allP, idx_all, polys, caps, u_attr, v_attr = [], [], [], [], [], []
        offset = 0
        for pts, closed in _prim_lists(src):
            m = len(pts)
            if m < 2:
                continue
            base = src.P[pts]
            for r in range(rings):
                ang = a0 + (a1 - a0) * r / divs
                allP.append(rotate_about_axis(base, origin, p.dir, ang))
                idx_all.append(pts)
                u_attr.append(np.full(m, r / divs))
                v_attr.append(np.linspace(0, 1, m))
            cols = m if closed else m - 1
            for r in range(divs):
                r1 = (r + 1) % rings
                for j in range(cols):
                    j1 = (j + 1) % m
                    polys.append([offset + r * m + j, offset + r * m + j1, offset + r1 * m + j1, offset + r1 * m + j])
            if p.endcaps and closed and not full:
                caps.append([offset + j for j in range(m)])
                caps.append([offset + (rings - 1) * m + j for j in range(m)][::-1])
            offset += rings * m
        if not allP:
            return out
        out.P = np.concatenate(allP)
        _copy_point_attribs(src, out, np.concatenate(idx_all))
        out.attribs[POINT]['revolveu'] = np.concatenate(u_attr)
        out.attribs[POINT]['revolvev'] = np.concatenate(v_attr)
        out.add_prims(polys)
        if caps:
            out.add_prims(caps)
        return out


@register_sop
class SkinSOP(SOP):
    id = 'skin'
    label = "Skin"
    category = 'Polygon'
    inputs = 2
    input_labels = ("Cross Sections", "Second Cross Sections (optional)")
    icon = 'MOD_SKIN'
    description = "Connect a series of curves with the same number of points into a surface"
    parms = (
        Parm('closev', "Close (connect last to first)", 'bool', default=False),
        Parm('keepprims', "Keep Input Curves", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        a, b = _in(inputs, 0), _in(inputs, 1)
        if b.num_points:
            # Two inputs: skin each curve of the first input to the matching curve of the second.
            from ..core.geometry import merge
            combined = merge([a, b])
            na = len(_prim_lists(a))
            sections = _prim_lists(combined)
            pairs = [(i, na + i) for i in range(min(na, len(sections) - na))]
            src = combined
        else:
            src = a
            sections = _prim_lists(src)
            pairs = [(i, i + 1) for i in range(len(sections) - 1)]
            if p.closev and len(sections) > 2:
                pairs.append((len(sections) - 1, 0))
        out = src.copy()
        if not p.keepprims:
            out.delete_prims(np.ones(out.num_prims, dtype=bool), keep_points=True)
        quads = []
        for i, j in pairs:
            pi, ci = sections[i]
            pj, _cj = sections[j]
            if len(pi) != len(pj):
                raise CookError("Skin: cross sections %d and %d have different point counts (%d, %d)"
                                % (i, j, len(pi), len(pj)))
            m = len(pi)
            cols = m if ci else m - 1
            for k in range(cols):
                k1 = (k + 1) % m
                quads.append([pi[k], pi[k1], pj[k1], pj[k]])
        out.add_prims(quads)
        return out


def _resample_frames(P, closed):
    """Arc length table and frames of a polyline."""
    if closed:
        P = np.concatenate([P, P[:1]])
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    T, N, B = curve_frames(P, False)
    return P, s, T, N, B


@register_sop
class PathDeformSOP(SOP):
    id = 'pathdeform'
    label = "Path Deform"
    category = 'Manipulate'
    inputs = 2
    input_labels = ("Geometry", "Path Curve")
    icon = 'MOD_CURVE'
    description = "Bend geometry along a curve (the chosen axis follows the path)"
    parms = (
        Parm('axis', "Deform Axis", 'enum', items=[('z', "Z", ""), ('x', "X", ""), ('y', "Y", "")]),
        Parm('stretch', "Stretch to Fit", 'bool', default=False,
             description="Scale the geometry along the axis so it covers the whole path"),
        Parm('offset', "Offset Along Path", 'float', default=0.0),
        Parm('scale', "Scale", 'float', default=1.0, min=0.0),
        Parm('twist', "Twist (degrees)", 'float', default=0.0),
        Parm('rollN', "Roll (degrees)", 'float', default=0.0),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs, 0).copy()
        path = _in(inputs, 1)
        curves = _prim_lists(path)
        if not curves or geo.num_points == 0:
            if not curves:
                ctx.warn("Path Deform: connect a curve to the second input")
            return geo
        pts, closed = curves[0]
        CP, s, T, N, B = _resample_frames(path.P[pts], closed)
        L = s[-1]
        if L <= 0:
            return geo
        a = AXIS_INDEX[p.axis]
        # Lateral axes mapped to the frame (normal follows Blendini's Y-up convention).
        lat = {0: ((1, 'N'), (2, 'B')), 1: ((0, 'N'), (2, '-B')), 2: ((0, '-B'), (1, 'N'))}[a]
        P = geo.P
        lo, hi = P[:, a].min(), P[:, a].max()
        along = P[:, a] - lo
        if p.stretch and hi > lo:
            along = along / (hi - lo) * L
        else:
            along = along * p.scale
        along = along + p.offset
        if closed:
            along = np.mod(along, L)
        clamped = np.clip(along, 0, L)
        i = np.clip(np.searchsorted(s, clamped, side='right') - 1, 0, len(s) - 2)
        t = ((clamped - s[i]) / np.maximum(s[i + 1] - s[i], 1e-12))[:, None]

        def interp(arr, norm=True):
            v = arr[i] * (1 - t) + arr[i + 1] * t
            if norm:
                v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
            return v
        base = interp(CP, False)
        Ti, Ni, Bi = interp(T), interp(N), interp(B)
        # Extrapolate past the ends along the end tangents.
        base = base + Ti * (along - clamped)[:, None]
        ang = np.radians(p.rollN + p.twist * clamped / L)[:, None]
        c, sn = np.cos(ang), np.sin(ang)
        Nr = Ni * c + Bi * sn
        Br = -Ni * sn + Bi * c
        frame = {'N': Nr, 'B': Br, '-B': -Br}
        k = p.scale if not p.stretch else 1.0
        newP = base.copy()
        for comp, vec in lat:
            newP += frame[vec] * (P[:, comp] * k)[:, None]
        geo.P = newP
        if 'N' in geo.attribs[POINT]:
            geo.attribs[POINT]['N'] = geo.point_normals() if geo.num_prims else geo.attribs[POINT]['N']
        return geo


@register_sop
class DistanceFromGeometrySOP(SOP):
    id = 'distancefromgeo'
    label = "Distance from Geometry"
    category = 'Attribute'
    inputs = 2
    input_labels = ("Geometry", "Target")
    icon = 'DRIVER_DISTANCE'
    description = "Distance from each point to the target surface or points, plus a 0-1 falloff mask"
    parms = (
        Parm('mode', "Distance To", 'enum', items=[('surface', "Surface", ""), ('points', "Points", "")]),
        Parm('distattrib', "Distance Attribute", 'string', default='dist'),
        Parm('maskattrib', "Mask Attribute", 'string', default='mask'),
        Parm('maxdist', "Max Distance", 'float', default=1.0, min=1e-6, soft_max=10),
        Parm('invert', "Invert Mask", 'bool', default=False),
        Parm('tocolor', "Visualize as Cd", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        from ..core import spatial
        geo = _in(inputs, 0).copy()
        tgt = _in(inputs, 1)
        n = geo.num_points
        if tgt.num_points == 0:
            ctx.warn("Distance from Geometry: the target (second input) is empty")
            d = np.full(n, np.inf)
        elif p.mode == 'points' or tgt.num_prims == 0:
            _idx, d = spatial.nearest(tgt.P, geo.P)
        else:
            _loc, _nrm, _prim, d = spatial.closest(tgt, geo.P)
        mask = 1.0 - np.clip(d / p.maxdist, 0.0, 1.0)
        if p.invert:
            mask = 1.0 - mask
        if p.distattrib.strip():
            geo.attribs[POINT][p.distattrib.strip()] = np.where(np.isfinite(d), d, 1e30)
        if p.maskattrib.strip():
            geo.attribs[POINT][p.maskattrib.strip()] = mask
        if p.tocolor:
            from .attribs import _infra_red
            geo.attribs[POINT]['Cd'] = _infra_red(mask)
        return geo


@register_sop
class LatticeSOP(SOP):
    id = 'lattice'
    label = "Lattice"
    category = 'Manipulate'
    inputs = 3
    input_labels = ("Geometry", "Rest Lattice", "Deformed Lattice")
    icon = 'LATTICE_DATA'
    description = ("Free-form deformation: the rest lattice points (e.g. a Bound with divisions, or the "
                   "lattice this node builds) are moved like the deformed lattice")
    parms = (
        Parm('divs', "Divisions (no rest input)", 'vector', default=(2, 2, 2)),
        Parm('falloff', "Falloff (scattered lattices)", 'float', default=2.0, min=0.5, max=8),
    )

    @staticmethod
    def grid(lo, hi, divs):
        axes = [np.linspace(lo[k], hi[k], max(int(round(divs[k])), 1) + 1) for k in range(3)]
        X, Y, Z = np.meshgrid(*axes, indexing='ij')
        return axes, np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    def cook(self, inputs, p, ctx):
        geo = _in(inputs, 0).copy()
        rest = _in(inputs, 1)
        deformed = _in(inputs, 2)
        if geo.num_points == 0:
            return geo
        if rest.num_points == 0:
            lo, hi = geo.bbox()
            hi = np.maximum(hi, lo + 1e-6)
            axes, restP = self.grid(lo, hi, p.divs)
        else:
            restP = rest.P
            axes = [np.unique(np.round(restP[:, k], 9)) for k in range(3)]
            if len(axes[0]) * len(axes[1]) * len(axes[2]) != len(restP):
                axes = None
        if deformed.num_points != len(restP):
            ctx.warn("Lattice: the deformed lattice needs %d points (it has %d)" % (len(restP), deformed.num_points))
            return geo
        offsets = deformed.P - restP
        P = geo.P
        if axes is not None:
            # Regular grid: trilinear interpolation of the lattice offsets.
            shape = tuple(len(a) for a in axes)
            if rest.num_points:
                order = np.lexsort((restP[:, 2], restP[:, 1], restP[:, 0]))
                grid_off = offsets[order].reshape(shape + (3,))
            else:
                grid_off = offsets.reshape(shape + (3,))
            idx, frac = [], []
            for k in range(3):
                a = axes[k]
                if len(a) == 1:
                    idx.append(np.zeros(len(P), dtype=np.int64))
                    frac.append(np.zeros(len(P)))
                    continue
                i = np.clip(np.searchsorted(a, P[:, k], side='right') - 1, 0, len(a) - 2)
                idx.append(i)
                frac.append(np.clip((P[:, k] - a[i]) / np.maximum(a[i + 1] - a[i], 1e-12), 0, 1))
            disp = np.zeros_like(P)
            for dx in (0, 1):
                for dy in (0, 1):
                    for dz in (0, 1):
                        w = ((frac[0] if dx else 1 - frac[0]) * (frac[1] if dy else 1 - frac[1])
                             * (frac[2] if dz else 1 - frac[2]))
                        ii = [np.minimum(idx[k] + d, shape[k] - 1) for k, d in enumerate((dx, dy, dz))]
                        disp += w[:, None] * grid_off[ii[0], ii[1], ii[2]]
        else:
            # Scattered lattice points: inverse distance weighting.
            d2 = np.sum((P[:, None, :] - restP[None]) ** 2, axis=2)
            w = 1.0 / np.maximum(d2, 1e-12) ** (p.falloff / 2)
            w /= w.sum(axis=1, keepdims=True)
            disp = w @ offsets
        geo.P = P + disp
        return geo

