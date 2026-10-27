# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Manipulation, polygon, copy and utility SOPs."""

import numpy as np

from .base import SOP, Parm, register_sop, transform_matrix, parse_group_pattern, orient_matrices, CookError
from ..core.geometry import Geometry, merge, POINT, PRIM, VERTEX, DETAIL
from ..core import noise as _noise

CLASS_PP = [('prim', "Primitives", ""), ('point', "Points", "")]
CLASS_ALL = [('point', "Point", ""), ('prim', "Primitive", ""), ('vertex', "Vertex", ""), ('detail', "Detail", "")]
ROT_ORDERS = [(o, o.upper(), "") for o in ('xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx')]
XFORM_ORDERS = [(o, o.upper(), "") for o in ('srt', 'str', 'rst', 'rts', 'tsr', 'trs')]


def _in(inputs, i=0):
    if len(inputs) > i and inputs[i] is not None:
        return inputs[i]
    return Geometry()


# ----------------------------------------------------------------------------
# Manipulate


@register_sop
class TransformSOP(SOP):
    id = 'xform'
    label = "Transform"
    category = 'Manipulate'
    inputs = 1
    icon = 'OBJECT_ORIGIN'
    parms = (
        Parm('group', "Group", 'string'),
        Parm('grouptype', "Group Type", 'enum', items=CLASS_PP),
        Parm('xord', "Transform Order", 'enum', items=XFORM_ORDERS),
        Parm('rord', "Rotate Order", 'enum', items=ROT_ORDERS),
        Parm('t', "Translate", 'vector'),
        Parm('r', "Rotate", 'vector'),
        Parm('s', "Scale", 'vector', default=(1.0, 1.0, 1.0)),
        Parm('p', "Pivot Translate", 'vector'),
        Parm('scale', "Uniform Scale", 'float', default=1.0, soft_min=0, soft_max=10),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        m = transform_matrix(p.t, p.r, p.s, p.scale, p.p, p.rord, p.xord)
        if p.group.strip():
            if p.grouptype == 'prim':
                pm = parse_group_pattern(geo, PRIM, p.group)
                mask = np.zeros(geo.num_points, dtype=bool)
                mask[geo.vtx_point[np.repeat(pm, geo.prim_sizes())]] = True
            else:
                mask = parse_group_pattern(geo, POINT, p.group)
            P = geo.P.copy()
            P[mask] = P[mask] @ m[:3, :3].T + m[:3, 3]
            geo.P = P
            return geo
        return geo.transform(m)


@register_sop
class MergeSOP(SOP):
    id = 'merge'
    label = "Merge"
    category = 'Utility'
    inputs = 1
    multi_input = True
    icon = 'SELECT_EXTEND'

    def cook(self, inputs, p, ctx):
        return merge([g for g in inputs if g is not None])


@register_sop
class NullSOP(SOP):
    id = 'null'
    label = "Null"
    category = 'Utility'
    inputs = 1
    icon = 'X'

    def cook(self, inputs, p, ctx):
        return _in(inputs)


@register_sop
class OutputSOP(SOP):
    id = 'output'
    label = "Output"
    category = 'Utility'
    inputs = 1
    icon = 'EXPORT'
    parms = (Parm('outputidx', "Output Index", 'int', default=0, min=0),)

    def cook(self, inputs, p, ctx):
        return _in(inputs)


@register_sop
class SwitchSOP(SOP):
    id = 'switch'
    label = "Switch"
    category = 'Utility'
    inputs = 1
    multi_input = True
    icon = 'ARROW_LEFTRIGHT'
    parms = (Parm('input', "Select Input", 'int', default=0, min=0, soft_max=10),)

    def cook(self, inputs, p, ctx):
        if not inputs:
            return Geometry()
        i = int(np.clip(p.input, 0, len(inputs) - 1))
        return _in(inputs, i)


def _point_mask_from_group(geo, pattern, cls='point'):
    if cls == 'prim':
        pm = parse_group_pattern(geo, PRIM, pattern)
        m = np.zeros(geo.num_points, dtype=bool)
        m[geo.vtx_point[np.repeat(pm, geo.prim_sizes())]] = True
        return m
    return parse_group_pattern(geo, POINT, pattern)


@register_sop
class MountainSOP(SOP):
    id = 'mountain'
    label = "Mountain"
    category = 'Manipulate'
    inputs = 1
    icon = 'RNDCURVE'
    time_dependent = False
    parms = (
        Parm('group', "Group", 'string'),
        Parm('height', "Height", 'float', default=1.0, soft_min=-5, soft_max=5),
        Parm('elementsize', "Element Size", 'float', default=1.0, min=0.0001, soft_max=10),
        Parm('offset', "Offset", 'vector'),
        Parm('octaves', "Max Octaves", 'int', default=3, min=1, max=12),
        Parm('lacunarity', "Lacunarity", 'float', default=2.01, soft_min=1, soft_max=4),
        Parm('roughness', "Roughness", 'float', default=0.5, min=0, max=1),
        Parm('mode', "Displace Along", 'enum', items=[('N', "Normal", ""), ('Y', "Y Axis", ""), ('XYZ', "Each Axis", "")]),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        if geo.num_points == 0:
            return geo
        mask = _point_mask_from_group(geo, p.group)
        pos = geo.P[mask] / max(p.elementsize, 1e-6) + np.asarray(p.offset)
        P = geo.P.copy()
        if p.mode == 'XYZ':
            d = _noise.fbm(pos, p.octaves, p.lacunarity, p.roughness, dims=3)
            P[mask] += d * p.height
        else:
            d = _noise.fbm(pos, p.octaves, p.lacunarity, p.roughness)
            if p.mode == 'N':
                N = geo.attribs[POINT]['N'] if geo.has_attrib(POINT, 'N') else geo.point_normals()
                P[mask] += N[mask] * (d * p.height)[:, None]
            else:
                P[mask, 1] += d * p.height
        geo.P = P
        return geo


@register_sop
class PointJitterSOP(SOP):
    id = 'pointjitter'
    label = "Point Jitter"
    category = 'Manipulate'
    inputs = 1
    icon = 'PARTICLES'
    parms = (
        Parm('group', "Group", 'string'),
        Parm('scale', "Scale", 'float', default=1.0, soft_min=0, soft_max=10),
        Parm('amount', "Amount", 'vector', default=(0.1, 0.1, 0.1)),
        Parm('seed', "Seed", 'int', default=0),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        mask = _point_mask_from_group(geo, p.group)
        ids = geo.attribs[POINT].get('id')
        seed_src = ids if ids is not None else np.arange(geo.num_points)
        r = _noise.rand(np.stack([seed_src.astype(np.float64), np.full(len(seed_src), float(p.seed))], axis=1), 3) - 0.5
        P = geo.P.copy()
        P[mask] += (r * np.asarray(p.amount) * p.scale)[mask]
        geo.P = P
        return geo


@register_sop
class PeakSOP(SOP):
    id = 'peak'
    label = "Peak"
    category = 'Manipulate'
    inputs = 1
    icon = 'NORMALS_VERTEX'
    parms = (
        Parm('group', "Group", 'string'),
        Parm('dist', "Distance", 'float', default=0.1, soft_min=-1, soft_max=1),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        mask = _point_mask_from_group(geo, p.group)
        N = geo.attribs[POINT]['N'] if geo.has_attrib(POINT, 'N') else geo.point_normals()
        P = geo.P.copy()
        P[mask] += N[mask] * p.dist
        geo.P = P
        return geo


def _point_neighbours_sum(geo, values):
    E = geo.edges()
    acc = np.zeros_like(values)
    cnt = np.zeros(len(values))
    if len(E):
        np.add.at(acc, E[:, 0], values[E[:, 1]])
        np.add.at(acc, E[:, 1], values[E[:, 0]])
        np.add.at(cnt, E[:, 0], 1)
        np.add.at(cnt, E[:, 1], 1)
    return acc, cnt


@register_sop
class SmoothSOP(SOP):
    id = 'smooth'
    label = "Smooth"
    category = 'Manipulate'
    inputs = 1
    icon = 'MOD_SMOOTH'
    parms = (
        Parm('group', "Group", 'string'),
        Parm('attribs', "Attributes", 'string', default='P'),
        Parm('strength', "Strength", 'float', default=0.5, min=0, soft_max=1),
        Parm('iterations', "Iterations", 'int', default=10, min=0, soft_max=100),
        Parm('pinborder', "Pin Border Points", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        mask = _point_mask_from_group(geo, p.group)
        E = geo.edges()
        if p.pinborder and len(E):
            border = _border_points(geo)
            mask &= ~border
        for name in p.attribs.split():
            arr = geo.attribs[POINT].get(name)
            if arr is None or arr.dtype == object:
                continue
            vals = arr.astype(np.float64).copy()
            for _ in range(max(p.iterations, 0)):
                acc, cnt = _point_neighbours_sum(geo, vals)
                shape = (-1,) + (1,) * (vals.ndim - 1)
                avg = np.where(cnt.reshape(shape) > 0, acc / np.maximum(cnt, 1).reshape(shape), vals)
                vals[mask] = vals[mask] + (avg[mask] - vals[mask]) * p.strength
            geo.attribs[POINT][name] = vals.astype(arr.dtype)
        return geo


def _border_points(geo):
    """Points on boundary edges (used by exactly one primitive)."""
    idx = np.arange(geo.num_vertices)
    nxt = idx + 1
    last = geo.prim_start[1:] - 1
    sizes = geo.prim_sizes()
    nz = sizes > 0
    nxt[last[nz]] = geo.prim_start[:-1][nz]
    a = geo.vtx_point
    b = geo.vtx_point[nxt]
    e = np.sort(np.stack([a, b], axis=1), axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    border = np.zeros(geo.num_points, dtype=bool)
    if len(uniq):
        be = uniq[counts == 1]
        border[be.ravel()] = True
    return border


# ----------------------------------------------------------------------------
# Polygon


def fuse_points(geo, dist):
    """Snap points closer than dist together (grid based) and remove degenerate prims."""
    if geo.num_points == 0:
        return geo
    dist = max(dist, 1e-9)
    key = np.round(geo.P / dist).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    remap = rank[inverse]
    keep_pts = first[order]
    # Average positions of fused points.
    newP = np.zeros((len(keep_pts), 3))
    cnt = np.zeros(len(keep_pts))
    np.add.at(newP, remap, geo.P)
    np.add.at(cnt, remap, 1)
    newP /= cnt[:, None]
    for name, arr in list(geo.attribs[POINT].items()):
        geo.attribs[POINT][name] = arr[keep_pts]
    geo.attribs[POINT]['P'] = newP
    for name, g in list(geo.groups[POINT].items()):
        geo.groups[POINT][name] = g[keep_pts]
    geo.vtx_point = remap[geo.vtx_point]
    _remove_degenerate(geo)
    return geo


def _remove_degenerate(geo):
    """Remove repeated consecutive vertices and degenerate primitives."""
    if geo.num_vertices == 0:
        return
    idx = np.arange(geo.num_vertices)
    prv = idx - 1
    first = geo.prim_start[:-1]
    sizes = geo.prim_sizes()
    nz = sizes > 0
    prv[first[nz]] = (geo.prim_start[1:] - 1)[nz]
    dup = geo.vtx_point == geo.vtx_point[prv]
    # Keep the first vertex of open polylines regardless of wrap.
    open_first = first[nz & ~geo.prim_closed]
    dup[open_first] = False
    if dup.any():
        keepv = ~dup
        vp = geo.vertex_prim()
        new_sizes = np.bincount(vp[keepv], minlength=geo.num_prims)
        geo.vtx_point = geo.vtx_point[keepv]
        for name, arr in list(geo.attribs[VERTEX].items()):
            geo.attribs[VERTEX][name] = arr[keepv]
        geo.prim_start = np.concatenate([[0], np.cumsum(new_sizes)]).astype(np.int64)
    sizes = geo.prim_sizes()
    bad = np.where(geo.prim_closed, sizes < 3, sizes < 2)
    if bad.any():
        geo.delete_prims(bad, keep_points=True)


@register_sop
class FuseSOP(SOP):
    id = 'fuse'
    label = "Fuse"
    category = 'Polygon'
    inputs = 1
    icon = 'AUTOMERGE_ON'
    parms = (Parm('dist', "Snap Distance", 'float', default=0.001, min=0, soft_max=1),)

    def cook(self, inputs, p, ctx):
        return fuse_points(_in(inputs).copy(), p.dist)


@register_sop
class FacetSOP(SOP):
    id = 'facet'
    label = "Facet"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_EDGESPLIT'
    parms = (
        Parm('unique', "Unique Points", 'bool', default=False),
        Parm('cons', "Consolidate Points", 'bool', default=False),
        Parm('dist', "Distance", 'float', default=0.001, hide_when={'cons': False}),
        Parm('prenml', "Pre-Compute Normals", 'bool', default=False),
        Parm('postnml', "Post-Compute Normals", 'bool', default=False),
        Parm('removeinline', "Remove Inline Points", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        if p.prenml:
            geo.attribs[POINT]['N'] = geo.point_normals()
        if p.cons:
            fuse_points(geo, p.dist)
        if p.unique:
            geo.unique_points()
        if p.postnml:
            geo.attribs[POINT]['N'] = geo.point_normals()
        return geo


@register_sop
class NormalSOP(SOP):
    id = 'normal'
    label = "Normal"
    category = 'Attribute'
    inputs = 1
    icon = 'NORMALS_FACE'
    parms = (
        Parm('type', "Add Normals To", 'enum', items=[
            ('point', "Points", ""), ('vertex', "Vertices", ""), ('prim', "Primitives", ""),
        ]),
        Parm('cuspangle', "Cusp Angle", 'float', default=60.0, min=0, max=180, hide_when={'type': ('point', 'prim')}),
        Parm('reverse', "Reverse Normals", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        for cls in (POINT, VERTEX, PRIM):
            geo.remove_attrib(cls, 'N')
        sign = -1.0 if p.reverse else 1.0
        if p.type == 'prim':
            geo.attribs[PRIM]['N'] = geo.prim_normals() * sign
        elif p.type == 'point':
            geo.attribs[POINT]['N'] = geo.point_normals() * sign
        else:
            fn = geo.prim_normals()
            vp = geo.vertex_prim()
            pn = geo.point_normals()
            vfn = fn[vp]
            vpn = pn[geo.vtx_point]
            # Smooth where the face normal is within the cusp angle of the averaged normal.
            cos_cusp = np.cos(np.radians(p.cuspangle))
            smooth = np.sum(vfn * vpn, axis=1) >= cos_cusp
            if p.cuspangle < 179.99:
                # Average only faces within cusp of each other around the point.
                acc = np.zeros((geo.num_vertices, 3))
                N = np.where(smooth[:, None], vpn, vfn)
                acc = N
            else:
                acc = vpn
            geo.attribs[VERTEX]['N'] = acc * sign
        return geo


@register_sop
class ReverseSOP(SOP):
    id = 'reverse'
    label = "Reverse"
    category = 'Polygon'
    inputs = 1
    icon = 'ARROW_LEFTRIGHT'
    parms = (Parm('group', "Group", 'string'),)

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        sel = parse_group_pattern(geo, PRIM, p.group)
        order = np.arange(geo.num_vertices)
        for i in np.nonzero(sel)[0]:
            s, e = geo.prim_start[i], geo.prim_start[i + 1]
            order[s:e] = order[s:e][::-1]
        geo.vtx_point = geo.vtx_point[order]
        for name, arr in list(geo.attribs[VERTEX].items()):
            geo.attribs[VERTEX][name] = arr[order]
        for cls in (POINT, PRIM, VERTEX):
            if 'N' in geo.attribs[cls]:
                geo.attribs[cls]['N'] = -geo.attribs[cls]['N']
        return geo


@register_sop
class DivideSOP(SOP):
    id = 'divide'
    label = "Divide"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_TRIANGULATE'
    description = "Triangulate polygons (fan triangulation)"
    parms = (Parm('group', "Group", 'string'),)

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        sel = parse_group_pattern(geo, PRIM, p.group)
        tris, tri_prim = geo.triangulate_flat()
        keep = ~sel | ~geo.prim_closed | (geo.prim_sizes() <= 3)
        out = geo.copy()
        out._delete_prims_only(~keep)
        take = sel[tri_prim] & (geo.prim_sizes()[tri_prim] > 3)
        tris = tris[take]
        src_prim = tri_prim[take]
        start = out.num_prims
        out.add_prims_flat(np.full(len(tris), 3), tris.ravel())
        for name, arr in geo.attribs[PRIM].items():
            out.attribs[PRIM][name][start:] = arr[src_prim]
        for name, g in geo.groups[PRIM].items():
            out.groups[PRIM][name][start:] = g[src_prim]
        return out


@register_sop
class PolyExtrudeSOP(SOP):
    id = 'polyextrude'
    label = "PolyExtrude"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_SOLIDIFY'
    parms = (
        Parm('group', "Group", 'string'),
        Parm('splittype', "Extrusion Mode", 'enum', items=[
            ('connected', "Connected Components", ""), ('individual', "Individual Elements", ""),
        ]),
        Parm('dist', "Distance", 'float', default=0.1, soft_min=-1, soft_max=1),
        Parm('inset', "Inset", 'float', default=0.0, soft_min=0, soft_max=1),
        Parm('outputfront', "Output Front", 'bool', default=True),
        Parm('outputback', "Output Back", 'bool', default=False),
        Parm('outputside', "Output Side", 'bool', default=True),
        Parm('frontgroup', "Front Group", 'string', default='extrudeFront'),
        Parm('sidegroup', "Side Group", 'string', default='extrudeSide'),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        sel = parse_group_pattern(geo, PRIM, p.group) & geo.prim_closed
        if not sel.any():
            return geo
        sizes = geo.prim_sizes()
        if p.splittype == 'individual':
            # Give the selected faces their own points.
            vsel = np.repeat(sel, sizes)
            vids = np.nonzero(vsel)[0]
            new_idx = geo.add_points(geo.P[geo.vtx_point[vids]])
            for name, arr in geo.attribs[POINT].items():
                if name != 'P':
                    arr[new_idx] = arr[geo.vtx_point[vids]]
            old_points = geo.vtx_point[vids].copy()
            geo.vtx_point[vids] = new_idx
        fn = geo.prim_normals()
        vp = geo.vertex_prim()
        vsel = sel[vp]
        sel_pts = np.zeros(geo.num_points, dtype=bool)
        sel_pts[geo.vtx_point[vsel]] = True
        # Extrusion direction per point: average of selected face normals.
        dirs = np.zeros((geo.num_points, 3))
        np.add.at(dirs, geo.vtx_point[vsel], fn[vp[vsel]])
        ln = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs = dirs / np.where(ln == 0, 1, ln)
        # Inset: move towards face centre (per point average).
        centers = geo.prim_centers()
        inset_vec = np.zeros((geo.num_points, 3))
        if p.inset:
            cnt = np.zeros(geo.num_points)
            np.add.at(inset_vec, geo.vtx_point[vsel], centers[vp[vsel]] - geo.P[geo.vtx_point[vsel]])
            np.add.at(cnt, geo.vtx_point[vsel], 1)
            inset_vec /= np.maximum(cnt, 1)[:, None]
        pts = np.nonzero(sel_pts)[0]
        new_pts = geo.add_points(geo.P[pts] + dirs[pts] * p.dist + inset_vec[pts] * p.inset)
        for name, arr in geo.attribs[POINT].items():
            if name != 'P':
                arr[new_pts] = arr[pts]
        remap = np.arange(geo.num_points)
        remap[pts] = new_pts
        # Boundary edges of the selection: directed edges whose reverse isn't selected.
        idx = np.arange(geo.num_vertices)
        nxt = idx + 1
        last = geo.prim_start[1:] - 1
        nz = sizes > 0
        nxt[last[nz]] = geo.prim_start[:-1][nz]
        a = geo.vtx_point[vsel]
        b = geo.vtx_point[nxt][vsel]
        src_prim = vp[vsel]
        fwd = set(zip(a.tolist(), b.tolist()))
        side = []
        side_src = []
        for x, y, sp in zip(a.tolist(), b.tolist(), src_prim.tolist()):
            if (y, x) not in fwd:
                side.append((x, y))
                side_src.append(sp)
        back_prims = None
        if p.outputback:
            back_prims = [geo.prim_points(i)[::-1].tolist() for i in np.nonzero(sel)[0]]
        # Move front faces to new points.
        geo.vtx_point[vsel] = remap[geo.vtx_point[vsel]]
        front_group = np.zeros(geo.num_prims, dtype=bool)
        front_group[sel] = True
        n_before = geo.num_prims
        sel_idx = np.nonzero(sel)[0]
        if p.outputside and side:
            quads = [[x, y, remap[y], remap[x]] for x, y in side]
            new = geo.add_prims(quads)
            for name, arr in geo.attribs[PRIM].items():
                arr[new] = arr[np.asarray(side_src)]
        if back_prims:
            new = geo.add_prims(back_prims)
            for name, arr in geo.attribs[PRIM].items():
                arr[new] = arr[sel_idx]
        if not p.outputfront:
            geo.delete_prims(np.concatenate([front_group, np.zeros(geo.num_prims - n_before, dtype=bool)]),
                             keep_points=True)
            n_before -= int(front_group.sum())
            front_group = front_group[~front_group]
        if p.frontgroup:
            g = np.zeros(geo.num_prims, dtype=bool)
            g[:len(front_group)] = front_group
            geo.groups[PRIM][p.frontgroup] = g
        if p.sidegroup and p.outputside:
            g = np.zeros(geo.num_prims, dtype=bool)
            g[n_before:n_before + len(side)] = True
            geo.groups[PRIM][p.sidegroup] = g
        return geo


@register_sop
class ResampleSOP(SOP):
    id = 'resample'
    label = "Resample"
    category = 'Polygon'
    inputs = 1
    icon = 'IPO_EASE_IN_OUT'
    parms = (
        Parm('mode', "Mode", 'enum', items=[('length', "Maximum Segment Length", ""), ('segs', "Segment Count", "")]),
        Parm('length', "Length", 'float', default=0.1, min=0.0001, soft_max=1, hide_when={'mode': 'segs'}),
        Parm('segs', "Segments", 'int', default=10, min=1, soft_max=100, hide_when={'mode': 'length'}),
        Parm('curveu', "Curve U Attribute", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        out = Geometry()
        polys_closed = []
        polys_open = []
        all_P = []
        all_u = []
        offset = 0
        for i in range(geo.num_prims):
            pts = geo.prim_points(i)
            closed = bool(geo.prim_closed[i])
            P = geo.P[pts]
            if closed:
                P = np.concatenate([P, P[:1]])
            seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
            total = seg.sum()
            if len(P) < 2 or total == 0:
                continue
            if p.mode == 'segs':
                n = max(p.segs, 1)
            else:
                n = max(int(np.ceil(total / p.length)), 1)
            cum = np.concatenate([[0], np.cumsum(seg)])
            t = np.linspace(0, total, n + 1)
            if closed:
                t = t[:-1]
            newP = np.stack([np.interp(t, cum, P[:, k]) for k in range(3)], axis=1)
            all_P.append(newP)
            all_u.append(t / total)
            idx = list(range(offset, offset + len(newP)))
            (polys_closed if closed else polys_open).append(idx)
            offset += len(newP)
        if all_P:
            out.P = np.concatenate(all_P)
            out.add_prims(polys_closed, closed=True)
            out.add_prims(polys_open, closed=False)
            if p.curveu:
                out.attribs[POINT]['curveu'] = np.concatenate(all_u)
        return out


@register_sop
class SortSOP(SOP):
    id = 'sort'
    label = "Sort"
    category = 'Utility'
    inputs = 1
    icon = 'SORTSIZE'
    parms = (
        Parm('ptsort', "Point Sort", 'enum', items=[
            ('none', "No Change", ""), ('x', "By X", ""), ('y', "By Y", ""), ('z', "By Z", ""), ('random', "Random", ""),
            ('reverse', "Reverse", ""),
        ]),
        Parm('primsort', "Primitive Sort", 'enum', items=[
            ('none', "No Change", ""), ('x', "By X", ""), ('y', "By Y", ""), ('z', "By Z", ""), ('random', "Random", ""),
            ('reverse', "Reverse", ""),
        ]),
        Parm('seed', "Seed", 'int', default=0),
    )

    def _order(self, kind, pos, seed):
        n = len(pos)
        if kind == 'none':
            return None
        if kind in ('x', 'y', 'z'):
            return np.argsort(pos[:, 'xyz'.index(kind)], kind='stable')
        if kind == 'random':
            return np.random.RandomState(seed).permutation(n)
        return np.arange(n)[::-1]

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        order = self._order(p.ptsort, geo.P, p.seed)
        if order is not None:
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            for name, arr in list(geo.attribs[POINT].items()):
                geo.attribs[POINT][name] = arr[order]
            for name, g in list(geo.groups[POINT].items()):
                geo.groups[POINT][name] = g[order]
            geo.vtx_point = inv[geo.vtx_point]
        order = self._order(p.primsort, geo.prim_centers(), p.seed)
        if order is not None and geo.num_prims:
            sizes = geo.prim_sizes()
            vorder = np.concatenate([np.arange(geo.prim_start[i], geo.prim_start[i + 1]) for i in order])
            geo.vtx_point = geo.vtx_point[vorder]
            for name, arr in list(geo.attribs[VERTEX].items()):
                geo.attribs[VERTEX][name] = arr[vorder]
            geo.prim_start = np.concatenate([[0], np.cumsum(sizes[order])]).astype(np.int64)
            geo.prim_closed = geo.prim_closed[order]
            for name, arr in list(geo.attribs[PRIM].items()):
                geo.attribs[PRIM][name] = arr[order]
            for name, g in list(geo.groups[PRIM].items()):
                geo.groups[PRIM][name] = g[order]
        return geo


# ----------------------------------------------------------------------------
# Groups


@register_sop
class GroupCreateSOP(SOP):
    id = 'groupcreate'
    label = "Group Create"
    category = 'Group'
    inputs = 1
    icon = 'GROUP_VERTEX'
    parms = (
        Parm('groupname', "Group Name", 'string', default='group1'),
        Parm('grouptype', "Group Type", 'enum', items=CLASS_PP),
        Parm('mode', "Base Group", 'enum', items=[
            ('all', "Entire Geometry", ""), ('pattern', "Pattern", "Numbers, groups or @expressions"),
            ('bbox', "Bounding Box", ""), ('normal', "Normal Direction", ""), ('random', "Random", ""),
        ]),
        Parm('pattern', "Pattern", 'string', default='@P.y > 0', hide_when={'mode': ('all', 'bbox', 'normal', 'random')}),
        Parm('size', "Box Size", 'vector', default=(1.0, 1.0, 1.0), hide_when={'mode': ('all', 'pattern', 'normal', 'random')}),
        Parm('t', "Box Center", 'vector', hide_when={'mode': ('all', 'pattern', 'normal', 'random')}),
        Parm('dir', "Direction", 'vector', default=(0.0, 1.0, 0.0), hide_when={'mode': ('all', 'pattern', 'bbox', 'random')}),
        Parm('angle', "Spread Angle", 'float', default=45.0, min=0, max=180, hide_when={'mode': ('all', 'pattern', 'bbox', 'random')}),
        Parm('percent', "Percentage", 'float', default=50.0, min=0, max=100, hide_when={'mode': ('all', 'pattern', 'bbox', 'normal')}),
        Parm('seed', "Seed", 'int', default=0, hide_when={'mode': ('all', 'pattern', 'bbox', 'normal')}),
        Parm('invert', "Invert", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        cls = p.grouptype
        n = geo.count(cls)
        pos = geo.P if cls == POINT else geo.prim_centers()
        if p.mode == 'all':
            mask = np.ones(n, dtype=bool)
        elif p.mode == 'pattern':
            mask = parse_group_pattern(geo, cls, p.pattern)
        elif p.mode == 'bbox':
            lo = np.asarray(p.t) - np.asarray(p.size) / 2
            hi = np.asarray(p.t) + np.asarray(p.size) / 2
            mask = np.all((pos >= lo) & (pos <= hi), axis=1)
        elif p.mode == 'normal':
            N = geo.point_normals() if cls == POINT else geo.prim_normals()
            if cls == POINT and geo.has_attrib(POINT, 'N'):
                N = geo.attribs[POINT]['N']
            d = np.asarray(p.dir, dtype=np.float64)
            d = d / max(np.linalg.norm(d), 1e-12)
            mask = N @ d >= np.cos(np.radians(p.angle))
        else:
            mask = _noise.rand(np.stack([np.arange(n, dtype=np.float64), np.full(n, float(p.seed))], 1)) < p.percent / 100
        if p.invert:
            mask = ~mask
        geo.groups[cls][p.groupname or 'group1'] = mask
        return geo


@register_sop
class BlastSOP(SOP):
    id = 'blast'
    label = "Blast"
    category = 'Group'
    inputs = 1
    icon = 'TRASH'
    parms = (
        Parm('group', "Group", 'string', description="Group name, numbers (0-10), or an @expression like @P.y<0"),
        Parm('grouptype', "Group Type", 'enum', items=CLASS_PP),
        Parm('negate', "Delete Non Selected", 'bool', default=False),
        Parm('removegrp', "Delete Unused Groups", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        if not p.group.strip():
            return geo
        mask = parse_group_pattern(geo, p.grouptype, p.group)
        if p.negate:
            mask = ~mask
        if p.grouptype == PRIM:
            geo.delete_prims(mask)
        else:
            geo.delete_points(mask)
        if p.removegrp:
            for cls in (POINT, PRIM):
                for name in list(geo.groups[cls]):
                    if not geo.groups[cls][name].any():
                        del geo.groups[cls][name]
        return geo


@register_sop
class ConnectivitySOP(SOP):
    id = 'connectivity'
    label = "Connectivity"
    category = 'Attribute'
    inputs = 1
    icon = 'LINKED'
    parms = (
        Parm('connecttype', "Connectivity Type", 'enum', items=CLASS_PP),
        Parm('attribname', "Attribute", 'string', default='class'),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        n = geo.num_points
        parent = np.arange(n)

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root
        for a, b in geo.edges().tolist():
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        roots = np.array([find(i) for i in range(n)], dtype=np.int64)
        _, labels = np.unique(roots, return_inverse=True)
        labels = labels.ravel().astype(np.int64)
        if p.connecttype == POINT:
            geo.attribs[POINT][p.attribname] = labels
        else:
            first = geo.vtx_point[geo.prim_start[:-1][geo.prim_sizes() > 0]]
            vals = np.zeros(geo.num_prims, dtype=np.int64)
            vals[geo.prim_sizes() > 0] = labels[first]
            geo.attribs[PRIM][p.attribname] = vals
        return geo


@register_sop
class MeasureSOP(SOP):
    id = 'measure'
    label = "Measure"
    category = 'Attribute'
    inputs = 1
    icon = 'DRIVER_DISTANCE'
    parms = (
        Parm('measure', "Measure", 'enum', items=[
            ('area', "Area", ""), ('perimeter', "Perimeter", ""), ('centroid', "Centroid", ""),
        ]),
        Parm('attribname', "Attribute", 'string', default='area'),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        if p.measure == 'area':
            geo.attribs[PRIM][p.attribname] = geo.prim_areas()
        elif p.measure == 'centroid':
            geo.attribs[PRIM][p.attribname] = geo.prim_centers()
        else:
            idx = np.arange(geo.num_vertices)
            nxt = idx + 1
            last = geo.prim_start[1:] - 1
            nz = geo.prim_sizes() > 0
            nxt[last[nz]] = geo.prim_start[:-1][nz]
            seg = np.linalg.norm(geo.P[geo.vtx_point[nxt]] - geo.P[geo.vtx_point], axis=1)
            seg[last[nz & ~geo.prim_closed]] = 0
            per = np.zeros(geo.num_prims)
            np.add.at(per, geo.vertex_prim(), seg)
            geo.attribs[PRIM][p.attribname] = per
        return geo


# ----------------------------------------------------------------------------
# Copy / scatter


@register_sop
class CopyToPointsSOP(SOP):
    id = 'copytopoints'
    label = "Copy to Points"
    category = 'Copy'
    inputs = 2
    input_labels = ("Geometry to Copy", "Target Points")
    icon = 'PARTICLE_POINT'
    parms = (
        Parm('targetgroup', "Target Points", 'string'),
        Parm('transform', "Transform Using Target Point Orientations", 'bool', default=True),
        Parm('copyattribs', "Copy Point Attributes", 'string', default='*',
             description="Template point attributes to copy onto the copies (wildcards)"),
    )

    def cook(self, inputs, p, ctx):
        src = _in(inputs, 0)
        tgt = _in(inputs, 1)
        sel = parse_group_pattern(tgt, POINT, p.targetgroup)
        tidx = np.nonzero(sel)[0]
        K = len(tidx)
        if K == 0 or src.num_points == 0:
            return Geometry()
        pa = tgt.attribs[POINT]
        if p.transform:
            R = orient_matrices(tgt)[tidx]
            scale = np.ones((K, 3))
            if 'pscale' in pa:
                scale *= pa['pscale'][tidx][:, None]
            if 'scale' in pa:
                scale *= pa['scale'][tidx]
            M = R * scale[:, None, :]
        else:
            M = np.broadcast_to(np.eye(3), (K, 3, 3))
        T = tgt.P[tidx]
        if 'trans' in pa and p.transform:
            T = T + pa['trans'][tidx]
        if 'pivot' in pa and p.transform:
            T = T - np.einsum('kij,kj->ki', M, pa['pivot'][tidx])
        import fnmatch
        copy_names = [n for n in pa if n not in ('P', 'N', 'orient', 'up', 'pscale', 'scale', 'trans', 'pivot', 'rot')
                      and any(fnmatch.fnmatchcase(n, pat) for pat in p.copyattribs.split())]
        S = src.num_points
        out = Geometry()
        P = np.einsum('kij,sj->ksi', M, src.P) + T[:, None, :]
        out.P = P.reshape(-1, 3)
        for name, arr in src.attribs[POINT].items():
            if name == 'P':
                continue
            data = np.tile(arr, (K,) + (1,) * (arr.ndim - 1))
            if name == 'N' and arr.ndim == 2:
                normal_m = np.linalg.inv(np.where(np.abs(np.linalg.det(M))[:, None, None] < 1e-12, np.eye(3), M))
                normal_m = np.swapaxes(normal_m, 1, 2)
                d = np.einsum('kij,sj->ksi', normal_m, arr).reshape(-1, 3)
                ln = np.linalg.norm(d, axis=1, keepdims=True)
                data = d / np.where(ln == 0, 1, ln)
            elif name in ('v', 'up') and arr.ndim == 2:
                data = np.einsum('kij,sj->ksi', M, arr).reshape(-1, 3)
            out.attribs[POINT][name] = data
        for n in copy_names:
            out.attribs[POINT][n] = np.repeat(pa[n][tidx], S, axis=0)
        sizes = src.prim_sizes()
        offsets = (np.arange(K) * S)[:, None]
        flat = (src.vtx_point[None, :] + offsets).ravel()
        out.add_prims_flat(np.tile(sizes, K), flat, np.tile(src.prim_closed, K))
        for name, arr in src.attribs[PRIM].items():
            out.attribs[PRIM][name] = np.tile(arr, (K,) + (1,) * (arr.ndim - 1))
        for name, arr in src.attribs[VERTEX].items():
            out.attribs[VERTEX][name] = np.tile(arr, (K,) + (1,) * (arr.ndim - 1))
        for name, arr in src.attribs[DETAIL].items():
            out.attribs[DETAIL][name] = arr.copy()
        out.attribs[PRIM]['copynum'] = np.repeat(np.arange(K), src.num_prims)
        out.attribs[POINT]['copynum'] = np.repeat(np.arange(K), S)
        for cls in (POINT, PRIM):
            for name, g in src.groups[cls].items():
                out.groups[cls][name] = np.tile(g, K)
        return out


@register_sop
class CopyXformSOP(SOP):
    id = 'copyxform'
    label = "Copy and Transform"
    category = 'Copy'
    inputs = 1
    icon = 'MOD_ARRAY'
    parms = (
        Parm('ncy', "Total Number", 'int', default=2, min=1, soft_max=100),
        Parm('t', "Translate", 'vector', default=(1.0, 0.0, 0.0)),
        Parm('r', "Rotate", 'vector'),
        Parm('s', "Scale", 'vector', default=(1.0, 1.0, 1.0)),
        Parm('p', "Pivot", 'vector'),
        Parm('scale', "Uniform Scale", 'float', default=1.0),
    )

    def cook(self, inputs, p, ctx):
        src = _in(inputs)
        step = transform_matrix(p.t, p.r, p.s, p.scale, p.p)
        copies = []
        m = np.eye(4)
        for i in range(max(p.ncy, 1)):
            g = src.copy().transform(m)
            g.attribs[PRIM]['copynum'] = np.full(g.num_prims, i, dtype=np.int64)
            g.attribs[POINT]['copynum'] = np.full(g.num_points, i, dtype=np.int64)
            copies.append(g)
            m = step @ m
        return merge(copies)


@register_sop
class ScatterSOP(SOP):
    id = 'scatter'
    label = "Scatter"
    category = 'Copy'
    inputs = 1
    icon = 'STICKY_UVS_DISABLE'
    parms = (
        Parm('group', "Group", 'string'),
        Parm('npts', "Force Total Count", 'int', default=1000, min=0, soft_max=100000),
        Parm('densityattrib', "Density Attribute", 'string', description="Optional prim or point float attribute scaling density"),
        Parm('seed', "Global Seed", 'int', default=0),
        Parm('relax', "Relax Iterations", 'int', default=0, min=0, soft_max=20),
        Parm('outputsourceprim', "Output Source Primitive Attribute", 'bool', default=True),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        sel = parse_group_pattern(geo, PRIM, p.group)
        tris, tri_prim = geo.triangulate_flat()
        take = sel[tri_prim]
        tris, tri_prim = tris[take], tri_prim[take]
        out = Geometry()
        if len(tris) == 0 or p.npts <= 0:
            return out
        A, B, C = geo.P[tris[:, 0]], geo.P[tris[:, 1]], geo.P[tris[:, 2]]
        area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=1)
        w = area.copy()
        if p.densityattrib:
            if geo.has_attrib(PRIM, p.densityattrib):
                w *= np.maximum(geo.attribs[PRIM][p.densityattrib][tri_prim].astype(np.float64), 0)
            elif geo.has_attrib(POINT, p.densityattrib):
                d = geo.attribs[POINT][p.densityattrib].astype(np.float64)
                w *= np.maximum(d[tris].mean(axis=1), 0)
        total = w.sum()
        if total <= 0:
            return out
        rng = np.random.RandomState(p.seed & 0x7FFFFFFF)
        n = p.npts
        cdf = np.cumsum(w) / total
        ti = np.minimum(np.searchsorted(cdf, rng.random_sample(n)), len(cdf) - 1)
        u = rng.random_sample(n)
        v = rng.random_sample(n)
        flip = u + v > 1
        u[flip] = 1 - u[flip]
        v[flip] = 1 - v[flip]
        P = A[ti] + (B[ti] - A[ti]) * u[:, None] + (C[ti] - A[ti]) * v[:, None]
        fn = geo.prim_normals()
        N = fn[tri_prim[ti]]
        for _ in range(max(p.relax, 0)):
            P = _relax_points(P, N, total, n)
        out.P = P
        out.attribs[POINT]['N'] = N
        if p.outputsourceprim:
            out.attribs[POINT]['sourceprim'] = tri_prim[ti].astype(np.int64)
        return out


def _relax_points(P, N, area, n):
    """One step of point repulsion constrained to the tangent plane."""
    try:
        from mathutils.kdtree import KDTree
    except ImportError:
        return P
    r = 2.0 * np.sqrt(area / max(n, 1) / np.pi)
    tree = KDTree(len(P))
    for i, co in enumerate(P.tolist()):
        tree.insert(co, i)
    tree.balance()
    disp = np.zeros_like(P)
    for i, co in enumerate(P.tolist()):
        for (_co, j, d) in tree.find_range(co, r):
            if j == i or d == 0:
                continue
            disp[i] += (P[i] - P[j]) / d * (r - d) * 0.5
    disp -= N * np.sum(disp * N, axis=1, keepdims=True)
    return P + disp


@register_sop
class PolyWireSOP(SOP):
    id = 'polywire'
    label = "PolyWire"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_WIREFRAME'
    parms = (
        Parm('radius', "Wire Radius", 'float', default=0.02, min=0, soft_max=1),
        Parm('divs', "Divisions", 'int', default=6, min=3, max=64),
        Parm('usepscale', "Scale Radius by pscale", 'bool', default=True),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        E = geo.edges()
        out = Geometry()
        if len(E) == 0:
            return out
        k = max(p.divs, 3)
        A = geo.P[E[:, 0]]
        B = geo.P[E[:, 1]]
        d = B - A
        ln = np.linalg.norm(d, axis=1, keepdims=True)
        d = d / np.where(ln == 0, 1, ln)
        ref = np.where(np.abs(d[:, 1:2]) < 0.9, [[0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0]])
        u = np.cross(d, ref)
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        v = np.cross(d, u)
        ang = 2 * np.pi * np.arange(k) / k
        ring = np.cos(ang)[None, :, None] * u[:, None, :] + np.sin(ang)[None, :, None] * v[:, None, :]
        rad = np.full((len(E), 2), p.radius)
        pscale = geo.attribs[POINT].get('pscale')
        if p.usepscale and pscale is not None:
            rad = rad * pscale[E]
        PA = A[:, None, :] + ring * rad[:, 0, None, None]
        PB = B[:, None, :] + ring * rad[:, 1, None, None]
        out.P = np.concatenate([PA, PB], axis=1).reshape(-1, 3)
        base = (np.arange(len(E)) * 2 * k)[:, None]
        i = np.arange(k)[None, :]
        j = (i + 1) % k
        quads = np.stack([base + i, base + j, base + k + j, base + k + i], axis=2).reshape(-1, 4)
        out.add_prims_flat(np.full(len(quads), 4), quads.ravel())
        # Carry point attributes from the wire's end points.
        for name, arr in geo.attribs[POINT].items():
            if name in ('P', 'N') or arr.dtype == object:
                continue
            src = np.concatenate([np.repeat(E[:, :1], k, axis=1), np.repeat(E[:, 1:], k, axis=1)], axis=1).ravel()
            out.attribs[POINT][name] = arr[src]
        return out


@register_sop
class AttribTransferSOP(SOP):
    id = 'attribtransfer'
    label = "Attribute Transfer"
    category = 'Attribute'
    inputs = 2
    input_labels = ("Destination", "Source")
    icon = 'UV_SYNC_SELECT'
    parms = (
        Parm('pointattriblist', "Point Attributes", 'string', default='Cd'),
        Parm('thresholddist', "Distance Threshold", 'float', default=0.1, min=0, soft_max=10),
        Parm('blendwidth', "Blend Width", 'float', default=0.2, min=0, soft_max=10),
    )

    def cook(self, inputs, p, ctx):
        import fnmatch
        from ..core.spatial import nearest
        dst = _in(inputs, 0).copy()
        src = _in(inputs, 1)
        if dst.num_points == 0 or src.num_points == 0:
            return dst
        maxd = p.thresholddist + p.blendwidth
        idx, dist = nearest(src.P, dst.P, maxd)
        hit = idx >= 0
        if p.blendwidth > 0:
            w = np.clip(1.0 - (dist - p.thresholddist) / p.blendwidth, 0, 1)
        else:
            w = (dist <= p.thresholddist).astype(np.float64)
        w[~hit] = 0
        pats = p.pointattriblist.split()
        for name, arr in src.attribs[POINT].items():
            if name == 'P' or arr.dtype == object or not any(fnmatch.fnmatchcase(name, pt) for pt in pats):
                continue
            cur = dst.attribs[POINT].get(name)
            if cur is None or cur.shape[1:] != arr.shape[1:]:
                cur = np.zeros((dst.num_points,) + arr.shape[1:])
            cur = cur.astype(np.float64)
            vals = arr[np.clip(idx, 0, None)].astype(np.float64)
            ww = w.reshape((-1,) + (1,) * (arr.ndim - 1))
            dst.attribs[POINT][name] = (cur + (vals - cur) * ww).astype(arr.dtype if arr.dtype.kind == 'f' else np.float64)
        return dst


@register_sop
class RaySOP(SOP):
    id = 'ray'
    label = "Ray"
    category = 'Manipulate'
    inputs = 2
    input_labels = ("Geometry to Project", "Collision Geometry")
    icon = 'LIGHT_SUN'
    parms = (
        Parm('group', "Group", 'string'),
        Parm('method', "Method", 'enum', items=[('minimum', "Minimum Distance", ""), ('project', "Projection", "")]),
        Parm('dirmethod', "Direction From", 'enum', items=[('vector', "Vector", ""), ('normal', "Normal (N)", "")],
             hide_when={'method': 'minimum'}),
        Parm('dir', "Ray Direction", 'vector', default=(0.0, -1.0, 0.0), hide_when={'method': 'minimum'}),
        Parm('scale', "Scale", 'float', default=1.0, min=0, max=1),
        Parm('putnml', "Point Intersection Normal", 'bool', default=False),
        Parm('putdist', "Point Intersection Distance", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        from ..core.spatial import ray_cast, closest
        geo = _in(inputs, 0).copy()
        col = _in(inputs, 1)
        mask = _point_mask_from_group(geo, p.group)
        idx = np.nonzero(mask)[0]
        P = geo.P[idx]
        if p.method == 'minimum':
            loc, nrm, prim, dist = closest(col, P)
            hit = prim >= 0
        else:
            if p.dirmethod == 'normal':
                N = geo.attribs[POINT]['N'] if geo.has_attrib(POINT, 'N') else geo.point_normals()
                D = N[idx]
            else:
                D = np.broadcast_to(np.asarray(p.dir, dtype=np.float64), P.shape)
            hit, loc, nrm, prim, dist = ray_cast(col, P, D)
        newP = geo.P.copy()
        newP[idx] = np.where(hit[:, None], P + (loc - P) * p.scale, P)
        geo.P = newP
        if p.putnml:
            N = geo.add_attrib(POINT, 'N', shape=(3,))
            N[idx[hit]] = nrm[hit]
        if p.putdist:
            d = geo.add_attrib(POINT, 'dist', shape=())
            d[idx] = dist
        geo.groups[POINT]['rayHitGroup'] = np.zeros(geo.num_points, dtype=bool)
        geo.groups[POINT]['rayHitGroup'][idx[hit]] = True
        return geo


@register_sop
class TimeShiftSOP(SOP):
    id = 'timeshift'
    label = "Time Shift"
    category = 'Utility'
    inputs = 1
    icon = 'TIME'
    description = "Cook the input at a different frame"
    parms = (
        Parm('mode', "Method", 'enum', items=[('frame', "Fixed Frame", ""), ('offset', "Frame Offset", "")]),
        Parm('frame', "Frame", 'float', default=1.0),
    )

    def cook(self, inputs, p, ctx):
        # The actual time change happens in the cook engine (inputs are cooked at another frame).
        return _in(inputs)


@register_sop
class BoundSOP(SOP):
    id = 'bound'
    label = "Bound"
    category = 'Primitive'
    inputs = 1
    icon = 'SHADING_BBOX'
    parms = (
        Parm('minpad', "Lower Padding", 'vector'),
        Parm('maxpad', "Upper Padding", 'vector'),
        Parm('divs', "Divisions", 'vector', default=(1, 1, 1)),
    )

    def cook(self, inputs, p, ctx):
        from .generators import make_box
        geo = _in(inputs)
        lo, hi = geo.bbox()
        lo = lo - np.asarray(p.minpad)
        hi = hi + np.asarray(p.maxpad)
        size = np.maximum(hi - lo, 1e-6)
        box = make_box(size, [round(d) for d in p.divs])
        box.P = box.P + (lo + hi) / 2
        return box


@register_sop
class UVTextureSOP(SOP):
    id = 'uvtexture'
    label = "UV Texture"
    category = 'Attribute'
    inputs = 1
    icon = 'UV'
    parms = (
        Parm('type', "Texture Type", 'enum', items=[
            ('ortho', "Orthographic", ""), ('cyl', "Cylindrical", ""), ('sph', "Polar (Spherical)", ""),
        ]),
        Parm('axis', "Axis", 'enum', items=[('Y', "Y", ""), ('X', "X", ""), ('Z', "Z", "")]),
        Parm('class', "Attribute Class", 'enum', items=[('vertex', "Vertices", ""), ('point', "Points", "")]),
        Parm('scale', "Scale", 'vector2', default=(1.0, 1.0)),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        lo, hi = geo.bbox()
        center = (lo + hi) / 2
        size = np.where(hi - lo == 0, 1, hi - lo)
        a = 'XYZ'.index(p.axis)
        others = [i for i in range(3) if i != a]
        P = geo.P if p['class'] == 'point' else geo.P[geo.vtx_point]
        rel = (P - center) / size
        uv = np.zeros((len(P), 3))
        if p.type == 'ortho':
            uv[:, 0] = rel[:, others[0]] + 0.5
            uv[:, 1] = rel[:, others[1]] + 0.5
        else:
            d = P - center
            ang = np.arctan2(d[:, others[0]], d[:, others[1]])
            uv[:, 0] = (ang / (2 * np.pi)) + 0.5
            if p.type == 'cyl':
                uv[:, 1] = rel[:, a] + 0.5
            else:
                r = np.linalg.norm(d, axis=1)
                uv[:, 1] = np.arccos(np.clip(d[:, a] / np.where(r == 0, 1, r), -1, 1)) / np.pi
                uv[:, 1] = 1 - uv[:, 1]
        uv[:, 0] *= p.scale[0]
        uv[:, 1] *= p.scale[1]
        geo.attribs[p['class']]['uv'] = uv
        return geo


@register_sop
class AttribNoiseSOP(SOP):
    id = 'attribnoise'
    label = "Attribute Noise"
    category = 'Attribute'
    inputs = 1
    icon = 'MOD_NOISE'
    time_dependent = False
    parms = (
        Parm('group', "Group", 'string'),
        Parm('attribs', "Attribute Name", 'string', default='P'),
        Parm('mode', "Operation", 'enum', items=[('add', "Add", ""), ('set', "Set", ""), ('mult', "Multiply", "")]),
        Parm('amplitude', "Amplitude", 'float', default=0.25, soft_min=0, soft_max=2),
        Parm('elementsize', "Element Size", 'float', default=1.0, min=0.0001, soft_max=10),
        Parm('offset', "Offset", 'vector'),
        Parm('octaves', "Fractal Octaves", 'int', default=3, min=1, max=10),
        Parm('animated', "Animate (offset by time)", 'bool', default=False),
        Parm('speed', "Pulse Speed", 'float', default=1.0, hide_when={'animated': False}),
        Parm('signed', "Center Noise Around Zero", 'bool', default=True),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        mask = _point_mask_from_group(geo, p.group)
        off = np.asarray(p.offset, dtype=np.float64)
        if p.animated:
            off = off + np.array([0.0, 0.0, ctx.time * p.speed])
        pos = geo.P[mask] / max(p.elementsize, 1e-6) + off
        name = p.attribs.strip() or 'P'
        arr = geo.attribs[POINT].get(name)
        dims = arr.shape[1] if arr is not None and arr.ndim == 2 else (3 if name in ('P', 'Cd', 'N', 'v') else 1)
        nz = _noise.fbm(pos, p.octaves, 2.0, 0.5, dims=dims)
        if not p.signed:
            nz = nz * 0.5 + 0.5
        nz = nz * p.amplitude
        if arr is None:
            arr = np.zeros((geo.num_points,) + ((dims,) if dims > 1 else ()))
        arr = arr.astype(np.float64).copy()
        if p.mode == 'add':
            arr[mask] += nz
        elif p.mode == 'set':
            arr[mask] = nz
        else:
            arr[mask] *= nz
        geo.attribs[POINT][name] = arr
        return geo

    time_dependent = True
