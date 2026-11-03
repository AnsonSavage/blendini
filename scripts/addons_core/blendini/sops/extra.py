# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""More everyday SOPs: Enumerate, Group by Range, Switch-If, Revolve, Skin, Path Deform,
Distance from Geometry, Lattice, Connect Adjacent Pieces."""

import numpy as np

from .base import SOP, Parm, register_sop, parse_group_pattern, CookError
from ..core.geometry import Geometry, POINT, DETAIL
from .modify import _in, curve_frames, CLASS_PP, split_pieces, piece_names

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
            r = np.arange(divs)[:, None]
            r1 = (r + 1) % rings
            j = np.arange(cols)[None, :]
            j1 = (j + 1) % m
            quads = np.stack(np.broadcast_arrays(offset + r * m + j, offset + r * m + j1, offset + r1 * m + j1,
                                                 offset + r1 * m + j), axis=-1).reshape(-1, 4)
            polys.append(quads)
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
        if polys:
            q = np.concatenate(polys)
            out.add_prims_flat(np.full(len(q), 4), q.ravel())
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
            k = np.arange(cols)
            k1 = (k + 1) % m
            quads.append(np.stack([pi[k], pi[k1], pj[k1], pj[k]], axis=1))
        if quads:
            q = np.concatenate(quads)
            out.add_prims_flat(np.full(len(q), 4), q.ravel())
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



CONSTRAINT_ITEMS = [
    ('glue', "Glue (breakable fixed)", ""), ('fixed', "Hard / Fixed", ""), ('point', "Point (ball joint)", ""),
    ('hinge', "Hinge", ""), ('slider', "Slider", ""), ('spring', "Spring", ""),
]


@register_sop
class ConnectAdjacentPiecesSOP(SOP):
    id = 'connectadjacentpieces'
    label = "Connect Adjacent Pieces"
    category = 'Simulation'
    inputs = 1
    input_labels = ("Pieces",)
    icon = 'CONSTRAINT'
    description = ("Constraint network for the RBD solver: a point at every piece center (with the piece "
                   "'name') and a polyline between neighbouring pieces")
    parms = (
        Parm('pieceattrib', "Piece Attribute", 'string', default='name'),
        Parm('radius', "Search Radius", 'float', default=0.0, min=0.0, soft_max=10.0,
             description="0: automatic (1.6 x the median distance to the nearest piece)"),
        Parm('maxpartners', "Max Partners", 'int', default=6, min=1, max=50),
        Parm('mode', "Connect", 'enum', items=[('neighbours', "Neighbours", ""),
                                               ('chain', "Chain (in piece order)", "")]),
        Parm('constraint_type', "Constraint Type", 'enum', items=CONSTRAINT_ITEMS),
        Parm('strength', "Strength", 'float', default=20.0, min=-1.0, soft_max=500.0,
             description="Impulse that breaks the constraint (0 or less: unbreakable; glue defaults to 20)"),
        Parm('stiffness', "Spring Stiffness", 'float', default=50.0, min=0.0, hide_when={'constraint_type': (
            'glue', 'fixed', 'point', 'hinge', 'slider')}),
        Parm('damping', "Spring Damping", 'float', default=0.5, min=0.0, max=1.0, hide_when={'constraint_type': (
            'glue', 'fixed', 'point', 'hinge', 'slider')}),
        Parm('axis', "Hinge Axis", 'vector', default=(0.0, 0.0, 1.0), hide_when={'constraint_type': (
            'glue', 'fixed', 'point', 'slider', 'spring')}),
    )

    def cook(self, inputs, p, ctx):
        src = _in(inputs)
        out = Geometry()
        if src.num_prims == 0:
            return out
        pieces = split_pieces(src, p.pieceattrib.strip())
        names = piece_names(pieces, p.pieceattrib.strip())
        centers = np.array([pc.P.mean(axis=0) if pc.num_points else np.zeros(3) for pc in pieces])
        n = len(pieces)
        pairs = []
        if p.mode == 'chain':
            pairs = [(i, i + 1) for i in range(n - 1)]
        elif n > 1:
            d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            radius = p.radius if p.radius > 0 else 1.6 * float(np.median(d.min(axis=1)))
            k = min(max(int(p.maxpartners), 1), n - 1)
            seen = set()
            for i in range(n):
                for j in np.argsort(d[i])[:k]:
                    j = int(j)
                    if d[i, j] <= radius and (min(i, j), max(i, j)) not in seen:
                        seen.add((min(i, j), max(i, j)))
                        pairs.append((min(i, j), max(i, j)))
        out.P = centers
        out.attribs[POINT]['name'] = np.array(names, dtype=object)
        if not pairs:
            return out
        out.add_prims([list(pr) for pr in pairs], closed=False)
        m = len(pairs)
        out.attribs['prim']['constraint_type'] = np.array([p.constraint_type] * m, dtype=object)
        out.attribs['prim']['constraint_name'] = np.array([p.constraint_type] * m, dtype=object)
        out.attribs['prim']['strength'] = np.full(m, float(p.strength))
        out.attribs['prim']['stiffness'] = np.full(m, float(p.stiffness))
        out.attribs['prim']['damping'] = np.full(m, float(p.damping))
        out.attribs['prim']['axis'] = np.tile(np.asarray(p.axis, dtype=np.float64), (m, 1))
        return out


@register_sop
class UVTransformSOP(SOP):
    id = 'uvtransform'
    label = "UV Transform"
    category = 'Attribute'
    inputs = 1
    icon = 'UV_DATA'
    description = "Translate, rotate and scale texture coordinates"
    parms = (
        Parm('group', "Group (primitives)", 'string'),
        Parm('attrib', "Attribute", 'string', default='uv'),
        Parm('t', "Translate", 'vector'),
        Parm('r', "Rotate (around W)", 'float', default=0.0),
        Parm('s', "Scale", 'vector', default=(1.0, 1.0, 1.0)),
        Parm('p', "Pivot", 'vector'),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        name = p.attrib.strip() or 'uv'
        cls = 'vertex' if name in geo.attribs['vertex'] else 'point' if name in geo.attribs['point'] else None
        if cls is None:
            ctx.warn("UV Transform: no '%s' attribute" % name)
            return geo
        uv = np.array(geo.attribs[cls][name], dtype=np.float64, copy=True)
        if uv.ndim == 1:
            return geo
        prim_mask = parse_group_pattern(geo, 'prim', p.group)
        if cls == 'vertex':
            sel = np.repeat(prim_mask, geo.prim_sizes())
        else:
            sel = np.zeros(geo.num_points, dtype=bool)
            sel[geo.vtx_point[np.repeat(prim_mask, geo.prim_sizes())]] = True
            if geo.num_prims == 0:
                sel[:] = True
        k = uv.shape[1]
        piv = np.asarray(p.p, dtype=np.float64)[:k]
        d = uv[sel] - piv
        d = d * np.asarray(p.s, dtype=np.float64)[:k]
        a = np.radians(p.r)
        c, s_ = np.cos(a), np.sin(a)
        x, y = d[:, 0].copy(), d[:, 1].copy()
        d[:, 0], d[:, 1] = x * c - y * s_, x * s_ + y * c
        uv[sel] = d + piv + np.asarray(p.t, dtype=np.float64)[:k]
        geo.attribs[cls][name] = uv
        return geo


def _adjust_values(geo, cls, n, size, p, ctx):
    """Values for Attribute Adjust: constant, random or noise (n x size)."""
    from ..core import noise as _noise
    if p.pattern == 'constant':
        v = np.asarray(p.value if size > 1 else [p.fvalue], dtype=np.float64)[:size]
        return np.broadcast_to(v, (n, size)).copy()
    if p.pattern == 'random':
        idx = np.arange(n, dtype=np.float64)
        r = _noise.rand(np.stack([idx, np.full(n, float(p.seed))], axis=1), size)
        r = r.reshape(n, size)
        lo = np.asarray(p.minv if size > 1 else [p.fmin], dtype=np.float64)[:size]
        hi = np.asarray(p.maxv if size > 1 else [p.fmax], dtype=np.float64)[:size]
        return lo + r * (hi - lo)
    # noise
    if cls == 'point':
        P = geo.P
    elif cls == 'prim':
        P = geo.prim_centers()
    elif cls == 'vertex':
        P = geo.P[geo.vtx_point]
    else:
        P = np.zeros((1, 3))
    pos = P / max(p.elementsize, 1e-9) + np.asarray(p.offset, dtype=np.float64)
    nz = _noise.snoise(pos, size).reshape(n, size)
    return nz * p.amplitude + p.center


class _AttribAdjustBase(SOP):
    category = 'Attribute'
    inputs = 1
    size = 1
    common = (
        Parm('class', "Class", 'enum', items=[('point', "Point", ""), ('prim', "Primitive", ""),
                                             ('vertex', "Vertex", ""), ('detail', "Detail", "")]),
        Parm('group', "Group", 'string'),
        Parm('op', "Operation", 'enum', items=[('set', "Set", ""), ('add', "Add", ""), ('sub', "Subtract", ""),
                                               ('mult', "Multiply", ""), ('min', "Minimum", ""),
                                               ('max', "Maximum", "")]),
        Parm('pattern', "Pattern", 'enum', items=[('constant', "Constant", ""), ('random', "Random", ""),
                                                  ('noise', "Noise", "")]),
        Parm('seed', "Seed", 'int', default=0, hide_when={'pattern': ('constant', 'noise')}),
        Parm('elementsize', "Element Size", 'float', default=1.0, min=1e-4, hide_when={'pattern': ('constant', 'random')}),
        Parm('offset', "Noise Offset", 'vector', hide_when={'pattern': ('constant', 'random')}),
        Parm('amplitude', "Amplitude", 'float', default=1.0, hide_when={'pattern': ('constant', 'random')}),
        Parm('center', "Center Value", 'float', default=0.0, hide_when={'pattern': ('constant', 'random')}),
        Parm('maskattrib', "Mask Attribute", 'string', default='',
             description="Optional float attribute (0-1) blending the adjustment"),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        cls = p['class']
        name = p.attrib.strip()
        if not name:
            return geo
        n = geo.count(cls)
        size = self.size
        old = geo.attribs[cls].get(name) if name != 'P' else geo.P
        if old is None or old.dtype == object:
            from ..core.geometry import ATTRIB_DEFAULTS
            dflt = ATTRIB_DEFAULTS.get(name, 0.0)
            old = np.full((n, size), dflt if np.ndim(dflt) == 0 else 0.0, dtype=np.float64)
            if size == 1:
                old = old[:, 0]
        if n == 0:
            return geo
        cur = np.array(old, dtype=np.float64, copy=True).reshape(n, -1)
        if cur.shape[1] != size:
            raise CookError("%s: '%s' has %d components, expected %d" % (self.label, name, cur.shape[1], size))
        vals = _adjust_values(geo, cls, n, size, p, ctx)
        new = {'set': lambda a, b: b, 'add': np.add, 'sub': np.subtract, 'mult': np.multiply,
               'min': np.minimum, 'max': np.maximum}[p.op](cur, vals)
        mask = parse_group_pattern(geo, cls, p.group) if cls != 'detail' else np.ones(1, dtype=bool)
        w = mask.astype(np.float64)
        if p.maskattrib.strip():
            m = geo.attribs[cls].get(p.maskattrib.strip())
            if m is not None and m.dtype != object:
                w = w * np.clip(np.asarray(m, dtype=np.float64).reshape(n, -1)[:, 0], 0, 1)
        out = cur + (new - cur) * w[:, None]
        if size == 1:
            out = out[:, 0]
        if name == 'P':
            geo.P = out
        else:
            geo.attribs[cls][name] = out
        return geo


@register_sop
class AttribAdjustFloatSOP(_AttribAdjustBase):
    id = 'attribadjustfloat'
    label = "Attribute Adjust Float"
    icon = 'MOD_HUE_SATURATION'
    description = "Set / add / multiply a float attribute by a constant, random or noise pattern"
    size = 1
    parms = (Parm('attrib', "Attribute", 'string', default='pscale'),) + _AttribAdjustBase.common[:4] + (
        Parm('fvalue', "Value", 'float', default=1.0, hide_when={'pattern': ('random', 'noise')}),
        Parm('fmin', "Min Value", 'float', default=0.0, hide_when={'pattern': ('constant', 'noise')}),
        Parm('fmax', "Max Value", 'float', default=1.0, hide_when={'pattern': ('constant', 'noise')}),
    ) + _AttribAdjustBase.common[4:]


@register_sop
class AttribAdjustVectorSOP(_AttribAdjustBase):
    id = 'attribadjustvector'
    label = "Attribute Adjust Vector"
    icon = 'ORIENTATION_GLOBAL'
    description = "Set / add / multiply a vector attribute by a constant, random or noise pattern"
    size = 3
    parms = (Parm('attrib', "Attribute", 'string', default='v'),) + _AttribAdjustBase.common[:4] + (
        Parm('value', "Value", 'vector', default=(0.0, 1.0, 0.0), hide_when={'pattern': ('random', 'noise')}),
        Parm('minv', "Min Value", 'vector', default=(-1.0, -1.0, -1.0), hide_when={'pattern': ('constant', 'noise')}),
        Parm('maxv', "Max Value", 'vector', default=(1.0, 1.0, 1.0), hide_when={'pattern': ('constant', 'noise')}),
    ) + _AttribAdjustBase.common[4:]


@register_sop
class ExtractCentroidSOP(SOP):
    id = 'extractcentroid'
    label = "Extract Centroid"
    category = 'Utility'
    inputs = 1
    icon = 'PIVOT_MEDIAN'
    description = "One point per piece at its centroid (or bounding box center), keeping the piece name"
    parms = (
        Parm('pieceattrib', "Piece Attribute", 'string', default='name'),
        Parm('method', "Method", 'enum', items=[('centroid', "Centroid (average)", ""),
                                                ('bbox', "Bounding Box Center", "")]),
    )

    def cook(self, inputs, p, ctx):
        src = _in(inputs)
        out = Geometry()
        if src.num_points == 0:
            return out
        if src.num_prims == 0:
            pieces = [src]
        else:
            pieces = split_pieces(src, p.pieceattrib.strip())
        names = piece_names(pieces, p.pieceattrib.strip())
        cents = []
        for pc in pieces:
            if p.method == 'bbox':
                lo, hi = pc.bbox()
                cents.append((lo + hi) / 2)
            else:
                cents.append(pc.P.mean(axis=0))
        out.P = np.array(cents)
        attrib = p.pieceattrib.strip() or 'name'
        out.attribs[POINT][attrib] = np.array(names, dtype=object)
        return out


@register_sop
class GroupFromAttribBoundarySOP(SOP):
    id = 'groupfromattribboundary'
    label = "Group from Attribute Boundary"
    category = 'Group'
    inputs = 1
    icon = 'GROUP_VERTEX'
    description = "Points on the borders between primitives with different attribute values"
    parms = (
        Parm('attrib', "Attribute", 'string', default='name'),
        Parm('groupname', "Group Name", 'string', default='boundary'),
        Parm('includeunshared', "Include Unshared Edges", 'bool', default=False,
             description="Also include points on open (mesh border) edges"),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        vals = geo.attribs['prim'].get(p.attrib.strip())
        mask = np.zeros(geo.num_points, dtype=bool)
        if vals is None:
            ctx.warn("Group from Attribute Boundary: no primitive attribute '%s'" % p.attrib)
        else:
            keyvals = [tuple(v) if np.ndim(v) else v for v in vals.tolist()]
            edges = {}
            for prim in range(geo.num_prims):
                pts = geo.prim_points(prim).tolist()
                m = len(pts)
                count = m if geo.prim_closed[prim] else m - 1
                for k in range(count):
                    a, b = pts[k], pts[(k + 1) % m]
                    edges.setdefault((min(a, b), max(a, b)), []).append(prim)
            for (a, b), prims in edges.items():
                distinct = len({keyvals[q] for q in prims}) > 1
                if distinct or (p.includeunshared and len(prims) == 1):
                    mask[a] = mask[b] = True
        geo.groups['point'][p.groupname.strip() or 'boundary'] = mask
        return geo


@register_sop
class OcclusionSOP(SOP):
    id = 'occlusion'
    label = "Calculate Occlusion"
    category = 'Attribute'
    inputs = 2
    input_labels = ("Geometry", "Occluders (optional)")
    icon = 'SHADING_RENDERED'
    description = "Ambient occlusion per point by casting rays around the normal (0 open, 1 fully occluded)"
    parms = (
        Parm('rays', "Rays", 'int', default=16, min=1, max=256),
        Parm('maxdist', "Max Distance", 'float', default=1.0, min=1e-4, soft_max=10),
        Parm('attrib', "Attribute", 'string', default='occlusion'),
        Parm('tocolor', "Darken Cd", 'bool', default=True),
        Parm('seed', "Seed", 'int', default=0),
    )

    def cook(self, inputs, p, ctx):
        from ..core import spatial
        geo = _in(inputs, 0).copy()
        occ_geo = _in(inputs, 1)
        target = geo if occ_geo.num_prims == 0 else occ_geo
        n = geo.num_points
        if n == 0 or target.num_prims == 0:
            return geo
        N = geo.attribs[POINT].get('N')
        if N is None or N.dtype == object:
            N = geo.point_normals()
        N = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        k = max(int(p.rays), 1)
        rng = np.random.default_rng(int(p.seed) & 0x7FFFFFFF)
        # Cosine weighted hemisphere directions (shared pattern, rotated per point).
        u = rng.random((k, 2))
        r = np.sqrt(u[:, 0])
        a = 2 * np.pi * u[:, 1]
        local = np.stack([r * np.cos(a), r * np.sin(a), np.sqrt(np.maximum(0, 1 - u[:, 0]))], axis=1)
        helper = np.where(np.abs(N[:, 0:1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
        X = np.cross(helper, N)
        X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        Y = np.cross(N, X)
        dirs = (local[None, :, 0:1] * X[:, None] + local[None, :, 1:2] * Y[:, None]
                + local[None, :, 2:3] * N[:, None]).reshape(-1, 3)
        eps = 1e-4 * max(float(np.linalg.norm(geo.bbox()[1] - geo.bbox()[0])), 1.0)
        origins = np.repeat(geo.P + N * eps, k, axis=0)
        hit, _loc, _nrm, _prim, _dist = spatial.ray_cast(target, origins, dirs, p.maxdist)
        occ = hit.reshape(n, k).mean(axis=1)
        geo.attribs[POINT][p.attrib.strip() or 'occlusion'] = occ
        if p.tocolor:
            cd = geo.attribs[POINT].get('Cd')
            base = np.ones((n, 3)) if cd is None or cd.dtype == object else cd
            geo.attribs[POINT]['Cd'] = base * (1.0 - occ)[:, None]
        return geo


def cache_path(pattern, ctx):
    """Absolute cache file path for the current frame ('//' is relative to the .blend file)."""
    import os
    from ..core.geoio import expand_path
    node = (ctx.node_path or 'filecache').split('/')[-1]
    blend_dir = ''
    try:
        import bpy
        blend_dir = os.path.dirname(bpy.data.filepath) if bpy.data.filepath else bpy.app.tempdir.rstrip('/\\')
        pattern = pattern.replace('//', '$HIP/', 1) if pattern.startswith('//') else pattern
    except ImportError:
        pass
    return os.path.abspath(expand_path(pattern, ctx.frame, node, blend_dir))


@register_sop
class FileCacheSOP(SOP):
    id = 'filecache'
    label = "File Cache"
    category = 'Utility'
    inputs = 1
    icon = 'FILE_CACHE'
    time_dependent = True
    description = ("Write geometry (with volumes and packed instances) to disk per frame with 'Save to Disk', "
                   "then read it back with 'Load from Disk' (fast playback of simulations)")
    parms = (
        Parm('file', "Geometry File", 'string', default='//cache/$OS.$F4.npz',
             description="$F / $F4 frame, $OS node name, '//' or $HIP: folder of the .blend file"),
        Parm('load', "Load from Disk", 'bool', default=False),
        Parm('startframe', "Start Frame", 'int', default=1),
        Parm('endframe', "End Frame", 'int', default=48),
        Parm('missing', "Missing Frame", 'enum', items=[('input', "Cook the Input", ""), ('empty', "No Geometry", ""),
                                                        ('error', "Report Error", "")],
             hide_when={'load': False}),
    )

    def cook(self, inputs, p, ctx):
        import os
        from ..core.geoio import load_geometry
        if not p.load:
            return _in(inputs).copy()
        path = cache_path(p.file, ctx)
        if os.path.isfile(path):
            try:
                return load_geometry(path)
            except (OSError, ValueError, KeyError) as ex:
                raise CookError("Can't read %s: %s" % (path, ex))
        if p.missing == 'error':
            raise CookError("No cache file %s" % path)
        if p.missing == 'empty':
            return Geometry()
        ctx.warn("No cache file for this frame: %s" % path)
        return _in(inputs).copy()


@register_sop
class MaterialSOP(SOP):
    id = 'material'
    label = "Material"
    category = 'Utility'
    inputs = 1
    icon = 'MATERIAL'
    description = "Assign a Blender material to primitives (writes the shop_materialpath primitive attribute)"
    parms = (
        Parm('group', "Group", 'string'),
        Parm('material', "Material", 'material'),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        n = geo.num_prims
        arr = geo.attribs['prim'].get('shop_materialpath')
        if arr is None or arr.dtype != object:
            arr = np.full(n, '', dtype=object)
        else:
            arr = arr.copy()
        arr[parse_group_pattern(geo, 'prim', p.group)] = p.material.strip()
        geo.attribs['prim']['shop_materialpath'] = arr
        return geo


@register_sop
class BlendShapesSOP(SOP):
    id = 'blendshapes'
    label = "Blend Shapes"
    category = 'Manipulate'
    inputs = 4
    input_labels = ("Base", "Target 1", "Target 2", "Target 3")
    icon = 'SHAPEKEY_DATA'
    description = "Blend point positions (and Cd / N) toward up to three targets with matching point counts"
    parms = (
        Parm('w1', "Blend 1", 'float', default=0.5, soft_min=0.0, soft_max=1.0),
        Parm('w2', "Blend 2", 'float', default=0.0, soft_min=0.0, soft_max=1.0),
        Parm('w3', "Blend 3", 'float', default=0.0, soft_min=0.0, soft_max=1.0),
        Parm('attribs', "Attributes", 'string', default='P Cd N', description="Point attributes to blend"),
    )

    def cook(self, inputs, p, ctx):
        base = _in(inputs, 0)
        out = base.copy()
        n = base.num_points
        for k, w in ((1, p.w1), (2, p.w2), (3, p.w3)):
            tgt = inputs[k] if len(inputs) > k else None
            if tgt is None or w == 0:
                continue
            if tgt.num_points != n:
                ctx.warn("Blend Shapes: target %d has %d points (base has %d)" % (k, tgt.num_points, n))
                continue
            for name in p.attribs.split():
                a = out.attribs[POINT].get(name)
                b0 = base.attribs[POINT].get(name)
                t = tgt.attribs[POINT].get(name)
                if a is None or b0 is None or t is None or a.dtype == object or t.shape != b0.shape:
                    continue
                out.attribs[POINT][name] = a + (t - b0) * w
        if 'N' in out.attribs[POINT] and out.attribs[POINT]['N'].ndim == 2:
            N = out.attribs[POINT]['N']
            out.attribs[POINT]['N'] = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        return out


@register_sop
class ExplodedViewSOP(SOP):
    id = 'explodedview'
    label = "Exploded View"
    category = 'Manipulate'
    inputs = 1
    icon = 'FULLSCREEN_EXIT'
    description = "Push pieces away from the center (inspect fractures and assemblies)"
    parms = (
        Parm('pieceattrib', "Piece Attribute", 'string', default='name'),
        Parm('scale', "Scale", 'float', default=0.5, soft_min=0.0, soft_max=3.0),
        Parm('center', "Center", 'enum', items=[('bbox', "Bounding Box Center", ""), ('origin', "Origin", "")]),
    )

    def cook(self, inputs, p, ctx):
        from ..core.geometry import merge
        src = _in(inputs)
        if src.num_prims == 0:
            return src.copy()
        pieces = split_pieces(src, p.pieceattrib.strip())
        lo, hi = src.bbox()
        c = (lo + hi) / 2 if p.center == 'bbox' else np.zeros(3)
        moved = []
        for pc in pieces:
            g = pc.copy()
            if g.num_points:
                g.P = g.P + (g.P.mean(axis=0) - c) * p.scale
            moved.append(g)
        return merge(moved)


def boundary_loops(geo):
    """Closed loops of boundary edges (edges used by one polygon), oriented opposite to their faces."""
    from collections import defaultdict
    count = defaultdict(int)
    directed = {}
    for prim in range(geo.num_prims):
        if not geo.prim_closed[prim]:
            continue
        pts = geo.prim_points(prim).tolist()
        m = len(pts)
        for k in range(m):
            a, b = pts[k], pts[(k + 1) % m]
            count[(min(a, b), max(a, b))] += 1
            directed[(a, b)] = True
    nxt = {}
    for (a, b) in directed:
        if count[(min(a, b), max(a, b))] == 1:
            nxt[b] = a       # reversed: the cap faces the other way
    loops = []
    seen = set()
    for start in list(nxt):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        cur = nxt[start]
        while cur != start and cur in nxt and cur not in seen:
            loop.append(cur)
            seen.add(cur)
            cur = nxt[cur]
        if cur == start and len(loop) >= 3:
            loops.append(loop)
    return loops


@register_sop
class PolyCapSOP(SOP):
    id = 'polycap'
    label = "PolyCap"
    category = 'Polygon'
    inputs = 1
    icon = 'MESH_CIRCLE'
    description = "Close holes: one polygon per boundary loop (or a triangle fan around its center)"
    parms = (
        Parm('mode', "Cap Type", 'enum', items=[('ngon', "Single Polygon", ""), ('fan', "Triangle Fan", "")]),
        Parm('groupname', "Output Group", 'string', default='caps'),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        loops = boundary_loops(geo)
        n0 = geo.num_prims
        for loop in loops:
            if p.mode == 'ngon':
                geo.add_prims([loop])
            else:
                c = geo.add_points(geo.P[loop].mean(axis=0)[None])[0]
                m = len(loop)
                geo.add_prims([[loop[k], loop[(k + 1) % m], c] for k in range(m)])
        if p.groupname.strip():
            g = np.zeros(geo.num_prims, dtype=bool)
            g[n0:] = True
            geo.groups['prim'][p.groupname.strip()] = g
        return geo


@register_sop
class PointReplicateSOP(SOP):
    id = 'pointreplicate'
    label = "Point Replicate"
    category = 'Copy'
    inputs = 1
    icon = 'PARTICLE_POINT'
    description = "Several jittered copies of every point (e.g. more particles before rasterizing)"
    parms = (
        Parm('count', "Copies per Point", 'int', default=5, min=1, soft_max=100),
        Parm('radius', "Jitter Radius", 'float', default=0.1, min=0.0, soft_max=2.0),
        Parm('usepscale', "Scale by pscale", 'bool', default=True),
        Parm('seed', "Seed", 'int', default=0),
    )

    def cook(self, inputs, p, ctx):
        src = _in(inputs)
        n = src.num_points
        out = Geometry()
        if n == 0:
            return out
        k = max(int(p.count), 1)
        idx = np.repeat(np.arange(n), k)
        rng = np.random.default_rng(int(p.seed) & 0x7FFFFFFF)
        d = rng.normal(size=(n * k, 3))
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        d *= np.cbrt(rng.random(n * k))[:, None] * p.radius
        if p.usepscale and 'pscale' in src.attribs[POINT]:
            d *= np.asarray(src.attribs[POINT]['pscale'], dtype=np.float64)[idx][:, None]
        out.P = src.P[idx] + d
        for name, arr in src.attribs[POINT].items():
            if name != 'P':
                out.attribs[POINT][name] = arr[idx]
        out.attribs[POINT]['sourceptnum'] = idx
        return out


@register_sop
class GroupTransferSOP(SOP):
    id = 'grouptransfer'
    label = "Group Transfer"
    category = 'Group'
    inputs = 2
    input_labels = ("Geometry", "Groups From")
    icon = 'GROUP'
    description = "Copy point / primitive groups from the second input by nearest position"
    parms = (
        Parm('class', "Group Type", 'enum', items=CLASS_PP),
        Parm('groups', "Groups", 'string', default='*'),
        Parm('maxdist', "Distance Threshold", 'float', default=0.1, min=0.0, soft_max=10.0),
    )

    def cook(self, inputs, p, ctx):
        import fnmatch
        from ..core import spatial
        geo = _in(inputs, 0).copy()
        src = _in(inputs, 1)
        cls = p['class']
        names = [g for g in src.groups[cls] if any(fnmatch.fnmatchcase(g, pat) for pat in (p.groups or '*').split())]
        if not names:
            return geo
        if cls == 'point':
            qp, sp = geo.P, src.P
        else:
            qp, sp = geo.prim_centers(), src.prim_centers()
        idx, dist = spatial.nearest(sp, qp, p.maxdist if p.maxdist > 0 else np.inf)
        ok = idx >= 0
        for name in names:
            g = np.zeros(len(qp), dtype=bool)
            g[ok] = src.groups[cls][name][idx[ok]]
            geo.groups[cls][name] = g
        return geo


@register_sop
class AttribPaintSOP(SOP):
    id = 'attribpaint'
    label = "Attribute Paint"
    category = 'Attribute'
    inputs = 1
    icon = 'BRUSH_DATA'
    description = ("Paint a float point attribute (e.g. a mask) with Blender's Weight Paint tools; "
                   "the painted values are stored on the node")
    parms = (
        Parm('attrib', "Attribute", 'string', default='mask'),
        Parm('default', "Unpainted Value", 'float', default=0.0, min=0.0, max=1.0),
        Parm('visualize', "Visualize as Cd", 'bool', default=True),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        n = geo.num_points
        vals = getattr(ctx, 'paint_values', None)
        out = np.full(n, float(p.default))
        if vals is not None and len(vals):
            if len(vals) == n:
                out = np.asarray(vals, dtype=np.float64)
            else:
                ctx.warn("Attribute Paint: painted %d points but the input has %d (repaint or clear)" % (len(vals), n))
        name = p.attrib.strip() or 'mask'
        geo.attribs[POINT][name] = out
        if p.visualize:
            cd = geo.attribs[POINT].get('Cd')
            base = np.full((n, 3), 0.75) if cd is None or cd.dtype == object else cd
            geo.attribs[POINT]['Cd'] = base * (1 - out[:, None]) + np.array([0.9, 0.1, 0.05]) * out[:, None]
        return geo


FALLOFFS = [('smooth', "Smooth", ""), ('linear', "Linear", ""), ('sharp', "Sharp", ""), ('constant', "Constant", "")]


def falloff(d, radius, kind):
    """Weight 1 at distance 0 fading to 0 at the radius."""
    x = np.clip(d / max(radius, 1e-9), 0, 1)
    if kind == 'linear':
        return 1 - x
    if kind == 'sharp':
        return (1 - x) ** 3
    if kind == 'constant':
        return (x < 1).astype(np.float64)
    return 1 - x * x * (3 - 2 * x)


@register_sop
class SoftTransformSOP(SOP):
    id = 'softxform'
    label = "Soft Transform"
    category = 'Manipulate'
    inputs = 1
    icon = 'PROP_ON'
    description = "Transform a group of points, with neighbours within the radius following by a falloff"
    parms = (
        Parm('group', "Group", 'string', default='0', description="Points that move fully"),
        Parm('t', "Translate", 'vector', default=(0.0, 0.5, 0.0)),
        Parm('r', "Rotate", 'vector'),
        Parm('s', "Scale", 'vector', default=(1.0, 1.0, 1.0)),
        Parm('radius', "Soft Radius", 'float', default=1.0, min=0.0, soft_max=10.0),
        Parm('falloff', "Falloff", 'enum', items=FALLOFFS),
        Parm('distmetric', "Distance", 'enum', items=[('dist', "Straight Line", ""), ('edges', "Along Edges", "")]),
    )

    def cook(self, inputs, p, ctx):
        from .base import transform_matrix
        geo = _in(inputs).copy()
        n = geo.num_points
        if n == 0:
            return geo
        sel = parse_group_pattern(geo, 'point', p.group)
        if not sel.any():
            return geo
        P = geo.P
        if p.distmetric == 'edges' and geo.num_prims:
            dist = _edge_distance(geo, sel, p.radius)
        else:
            from ..core import spatial
            _idx, dist = spatial.nearest(P[sel], P)
        w = falloff(dist, p.radius, p.falloff) if p.radius > 0 else sel.astype(np.float64)
        w[sel] = 1.0
        pivot = P[sel].mean(axis=0)
        M = transform_matrix(t=p.t, r=p.r, s=p.s, pivot=pivot)
        moved = P @ M[:3, :3].T + M[:3, 3]
        geo.P = P + (moved - P) * w[:, None]
        geo.attribs[POINT]['softweight'] = w
        return geo


def _edge_distance(geo, sources, limit):
    """Dijkstra distance along edges from the source points (inf beyond the limit)."""
    import heapq
    n = geo.num_points
    E = geo.edges()
    lengths = np.linalg.norm(geo.P[E[:, 0]] - geo.P[E[:, 1]], axis=1)
    adj = [[] for _ in range(n)]
    for (a, b), l in zip(E.tolist(), lengths.tolist()):
        adj[a].append((b, l))
        adj[b].append((a, l))
    dist = np.full(n, np.inf)
    heap = [(0.0, int(i)) for i in np.nonzero(sources)[0]]
    for _d, i in heap:
        dist[i] = 0.0
    heapq.heapify(heap)
    while heap:
        d, i = heapq.heappop(heap)
        if d > dist[i] or d > limit:
            continue
        for j, l in adj[i]:
            nd = d + l
            if nd < dist[j]:
                dist[j] = nd
                heapq.heappush(heap, (nd, j))
    return dist


@register_sop
class MagnetSOP(SOP):
    id = 'magnet'
    label = "Magnet"
    category = 'Manipulate'
    inputs = 2
    input_labels = ("Geometry", "Magnets (points, pscale = radius)")
    icon = 'SNAP_ON'
    description = "Push, pull or scale points near the magnet points of the second input, with a falloff"
    parms = (
        Parm('mode', "Effect", 'enum', items=[('translate', "Translate", ""), ('push', "Push Out (radial)", ""),
                                              ('scale', "Scale Around Magnet", "")]),
        Parm('t', "Translate", 'vector', default=(0.0, 0.5, 0.0), hide_when={'mode': ('push', 'scale')}),
        Parm('amount', "Amount", 'float', default=0.3, soft_min=-2.0, soft_max=2.0, hide_when={'mode': 'translate'}),
        Parm('radius', "Radius", 'float', default=1.0, min=0.0, soft_max=10.0, description="Multiplied by pscale"),
        Parm('falloff', "Falloff", 'enum', items=FALLOFFS),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs, 0).copy()
        mags = _in(inputs, 1)
        if geo.num_points == 0 or mags.num_points == 0:
            if mags.num_points == 0:
                ctx.warn("Magnet: connect magnet points to the second input")
            return geo
        P = geo.P.copy()
        ps = mags.attribs[POINT].get('pscale')
        rad = (np.asarray(ps, dtype=np.float64) if ps is not None and ps.dtype != object
               else np.ones(mags.num_points)) * p.radius
        disp = np.zeros_like(P)
        for c, r in zip(mags.P, rad):
            d = P - c
            dist = np.linalg.norm(d, axis=1)
            w = falloff(dist, r, p.falloff)
            if p.mode == 'translate':
                disp += w[:, None] * np.asarray(p.t, dtype=np.float64)
            elif p.mode == 'push':
                disp += w[:, None] * d / np.maximum(dist, 1e-9)[:, None] * p.amount
            else:
                disp += w[:, None] * d * p.amount
        geo.P = P + disp
        return geo
