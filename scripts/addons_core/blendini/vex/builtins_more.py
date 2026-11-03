# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""More VEX builtins: half-edges, topology queries, unique values, regular expressions,
matrix helpers, sampling, point cloud iteration, ray intersection, curve lengths."""

import fnmatch
import re

import numpy as np

from .builtins import (
    builtin, OUTARGS, _n, _f, _uniform_str, _geo, _point_prims_table, _axis_angle_matrix, _matrix3_from_quat,
    _quat_from_matrix3, _pc_handle, _noise_pos, _result_dims,
)
from .values import V, VexError, SCALARS, VECTORS, MATRICES, VEC_BY_SIZE, const, full, convert, type_of_array
from ..core import noise as _noise


def _ints(I, v, n=None):
    return full(convert(v, 'int'), n or _n(I, v))


def _int_array(rows):
    out = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        out[i] = np.asarray(r, dtype=np.int64)
    return V('int[]', out)


# ----------------------------------------------------------------------------
# Half-edges. A half-edge is identified by its source vertex (linear vertex number) and
# runs to the next vertex of the primitive (closed primitives wrap, open ones don't).


def _hedge_tables(I, geo):
    key = ('hedges', id(geo))
    cache = I.ctx.cache
    if key not in cache:
        nv = geo.num_vertices
        vp = geo.vertex_prim()
        start = geo.prim_start[vp] if nv else np.zeros(0, np.int64)
        end = geo.prim_start[vp + 1] if nv else np.zeros(0, np.int64)
        nxt = np.arange(nv) + 1
        closed = geo.prim_closed[vp] if nv else np.zeros(0, bool)
        wrap = nxt >= end
        nxt = np.where(wrap, np.where(closed, start, -1), nxt)
        prv = np.arange(nv) - 1
        pwrap = prv < start
        prv = np.where(pwrap, np.where(closed, end - 1, -1), prv)
        src = geo.vtx_point
        dst = np.where(nxt >= 0, src[np.clip(nxt, 0, max(nv - 1, 0))] if nv else -1, -1)
        # Edges (unordered point pairs) -> list of half-edges, for equivalence queries.
        edges = {}
        for h in range(nv):
            if nxt[h] < 0:
                continue
            a, b = int(src[h]), int(dst[h])
            edges.setdefault((min(a, b), max(a, b)), []).append(h)
        by_src = {}
        for h in range(nv):
            if nxt[h] >= 0:
                by_src.setdefault(int(src[h]), []).append(h)
        cache[key] = dict(nxt=nxt, prv=prv, src=src, dst=dst, vp=vp, edges=edges, by_src=by_src)
    return cache[key]


def _valid(t, h):
    return (h >= 0) & (h < len(t['nxt'])) & (t['nxt'][np.clip(h, 0, max(len(t['nxt']) - 1, 0))] >= 0 if len(t['nxt'])
                                              else np.zeros_like(h, dtype=bool))


def _hedge_op(fn):
    def impl(I, args, mask, hint):
        geo = _geo(I, args[0])
        t = _hedge_tables(I, geo)
        h = _ints(I, args[1])
        ok = _valid(t, h)
        hc = np.clip(h, 0, max(len(t['nxt']) - 1, 0))
        if not len(t['nxt']):
            return V('int', np.full(len(h), -1, dtype=np.int64))
        return V('int', np.where(ok, fn(t, hc), -1).astype(np.int64))
    return impl


builtin('hedge_srcpoint')(_hedge_op(lambda t, h: t['src'][h]))
builtin('hedge_dstpoint')(_hedge_op(lambda t, h: t['dst'][h]))
builtin('hedge_srcvertex')(_hedge_op(lambda t, h: h))
builtin('hedge_dstvertex')(_hedge_op(lambda t, h: t['nxt'][h]))
builtin('hedge_next')(_hedge_op(lambda t, h: t['nxt'][h]))
builtin('hedge_prim')(_hedge_op(lambda t, h: t['vp'][h]))
builtin('hedge_prev')(_hedge_op(lambda t, h: np.where(t['prv'][h] >= 0, t['prv'][h], -1)))
builtin('hedge_presrcpoint')(_hedge_op(
    lambda t, h: np.where(t['prv'][h] >= 0, t['src'][np.clip(t['prv'][h], 0, None)], -1)))
builtin('hedge_postdstpoint')(_hedge_op(
    lambda t, h: np.where(t['nxt'][np.clip(t['nxt'][h], 0, None)] >= 0,
                          t['dst'][np.clip(t['nxt'][h], 0, None)], -1)))


@builtin('hedge_isvalid')
def _hedge_isvalid(I, args, mask, hint):
    t = _hedge_tables(I, _geo(I, args[0]))
    return V('int', _valid(t, _ints(I, args[1])).astype(np.int64))


@builtin('hedge_isprimary')
def _hedge_isprimary(I, args, mask, hint):
    t = _hedge_tables(I, _geo(I, args[0]))
    h = _ints(I, args[1])
    ok = _valid(t, h)
    out = np.zeros(len(h), dtype=np.int64)
    for i in np.nonzero(ok)[0]:
        a, b = int(t['src'][h[i]]), int(t['dst'][h[i]])
        out[i] = int(t['edges'][(min(a, b), max(a, b))][0] == h[i])
    return V('int', out)


@builtin('hedge_equivcount')
def _hedge_equivcount(I, args, mask, hint):
    t = _hedge_tables(I, _geo(I, args[0]))
    h = _ints(I, args[1])
    ok = _valid(t, h)
    out = np.full(len(h), -1, dtype=np.int64)
    for i in np.nonzero(ok)[0]:
        a, b = int(t['src'][h[i]]), int(t['dst'][h[i]])
        out[i] = len(t['edges'][(min(a, b), max(a, b))])
    return V('int', out)


@builtin('hedge_nextequiv')
def _hedge_nextequiv(I, args, mask, hint):
    t = _hedge_tables(I, _geo(I, args[0]))
    h = _ints(I, args[1])
    ok = _valid(t, h)
    out = np.full(len(h), -1, dtype=np.int64)
    for i in np.nonzero(ok)[0]:
        a, b = int(t['src'][h[i]]), int(t['dst'][h[i]])
        eq = t['edges'][(min(a, b), max(a, b))]
        out[i] = eq[(eq.index(int(h[i])) + 1) % len(eq)]
    return V('int', out)


@builtin('primhedge')
def _primhedge(I, args, mask, hint):
    geo = _geo(I, args[0])
    t = _hedge_tables(I, geo)
    prim = _ints(I, args[1])
    ok = (prim >= 0) & (prim < geo.num_prims)
    first = geo.prim_start[np.clip(prim, 0, max(geo.num_prims - 1, 0))] if geo.num_prims else np.zeros_like(prim)
    valid = ok & _valid(t, first)
    return V('int', np.where(valid, first, -1).astype(np.int64))


@builtin('pointhedge')
def _pointhedge(I, args, mask, hint):
    """pointhedge(geo, srcpt) / pointhedge(geo, srcpt, dstpt): a half-edge leaving srcpt (to dstpt)."""
    t = _hedge_tables(I, _geo(I, args[0]))
    n = _n(I, *args[1:])
    a = _ints(I, args[1], n)
    b = _ints(I, args[2], n) if len(args) > 2 else None
    out = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        for h in t['by_src'].get(int(a[i]), ()):
            if b is None or t['dst'][h] == b[i]:
                out[i] = h
                break
    return V('int', out)


@builtin('pointedge')
def _pointedge(I, args, mask, hint):
    """A half-edge between two points, in either direction (-1 when they aren't connected)."""
    t = _hedge_tables(I, _geo(I, args[0]))
    n = _n(I, args[1], args[2])
    a, b = _ints(I, args[1], n), _ints(I, args[2], n)
    out = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        eq = t['edges'].get((min(int(a[i]), int(b[i])), max(int(a[i]), int(b[i]))))
        if eq:
            out[i] = eq[0]
    return V('int', out)


@builtin('hedge_count', 'pointhedgecount')
def _pointhedgecount(I, args, mask, hint):
    t = _hedge_tables(I, _geo(I, args[0]))
    a = _ints(I, args[1])
    return V('int', np.array([len(t['by_src'].get(int(x), ())) for x in a], dtype=np.int64))


# ----------------------------------------------------------------------------
# Topology


@builtin('primvertices')
def _primvertices(I, args, mask, hint):
    geo = _geo(I, args[0])
    prim = _ints(I, args[1])
    s = geo.prim_start
    return _int_array([np.arange(s[p], s[p + 1]) if 0 <= p < geo.num_prims else [] for p in prim])


@builtin('pointvertices')
def _pointvertices(I, args, mask, hint):
    geo = _geo(I, args[0])
    key = ('pointvertices', id(geo))
    table = I.ctx.cache.get(key)
    if table is None:
        table = [[] for _ in range(geo.num_points)]
        for v, p in enumerate(geo.vtx_point.tolist()):
            table[p].append(v)
        I.ctx.cache[key] = table
    pts = _ints(I, args[1])
    return _int_array([table[p] if 0 <= p < len(table) else [] for p in pts])


@builtin('pointvertex')
def _pointvertex(I, args, mask, hint):
    res = _pointvertices(I, args, mask, hint)
    return V('int', np.array([r[0] if len(r) else -1 for r in res.d], dtype=np.int64))


@builtin('vertexindex')
def _vertexindex(I, args, mask, hint):
    geo = _geo(I, args[0])
    n = _n(I, args[1], args[2])
    prim, k = _ints(I, args[1], n), _ints(I, args[2], n)
    sizes = geo.prim_sizes()
    ok = (prim >= 0) & (prim < geo.num_prims)
    pc = np.clip(prim, 0, max(geo.num_prims - 1, 0))
    ok &= (k >= 0) & (k < (sizes[pc] if len(sizes) else 0))
    return V('int', np.where(ok, (geo.prim_start[pc] if geo.num_prims else 0) + k, -1).astype(np.int64))


@builtin('vertexprimindex')
def _vertexprimindex(I, args, mask, hint):
    geo = _geo(I, args[0])
    v = _ints(I, args[1])
    ok = (v >= 0) & (v < geo.num_vertices)
    vp = geo.vertex_prim()
    vc = np.clip(v, 0, max(geo.num_vertices - 1, 0))
    return V('int', np.where(ok, v - geo.prim_start[vp[vc]] if geo.num_vertices else -1, -1).astype(np.int64))


@builtin('expandprimgroup', 'getprimgroup')
def _expandprimgroup(I, args, mask, hint):
    geo = _geo(I, args[0])
    name = _uniform_str(args[1])
    g = geo.groups['prim'].get(name)
    idx = np.nonzero(g)[0] if g is not None else (np.arange(geo.num_prims) if name in ('', '*') else [])
    return _int_array([idx] * I.n)


@builtin('nprimsgroup')
def _nprimsgroup2(I, args, mask, hint):
    geo = _geo(I, args[0])
    g = geo.groups['prim'].get(_uniform_str(args[1]))
    return const('int', int(g.sum()) if g is not None else 0)


def _make_pointbbox(which):
    def impl(I, args, mask, hint):
        geo = _geo(I, args[0])
        P = geo.P
        if len(args) > 1:
            from ..sops.base import parse_group_pattern
            P = P[parse_group_pattern(geo, 'point', _uniform_str(args[1]))]
        if len(P) == 0:
            return V('vector', np.zeros((1, 3)))
        lo, hi = P.min(axis=0), P.max(axis=0)
        val = {'center': (lo + hi) / 2, 'size': hi - lo, 'min': lo, 'max': hi}[which]
        return V('vector', val[None])
    return impl


for _w in ('center', 'size', 'min', 'max'):
    builtin('getpointbbox_' + _w)(_make_pointbbox(_w))


# ----------------------------------------------------------------------------
# Attribute queries


def _attr_array(geo, cls, name):
    cls = {'primitive': 'prim', 'global': 'detail', 'points': 'point'}.get(cls, cls)
    if name == 'P' and cls == 'point':
        return geo.P
    return geo.attribs.get(cls, {}).get(name)


def _unique(arr):
    if arr is None:
        return []
    if arr.dtype == object:
        return sorted(set(arr.tolist()))
    if arr.ndim > 1:
        return [tuple(r) for r in np.unique(arr, axis=0).tolist()]
    return np.unique(arr).tolist()


@builtin('nuniqueval')
def _nuniqueval(I, args, mask, hint):
    geo = _geo(I, args[0])
    return const('int', len(_unique(_attr_array(geo, _uniform_str(args[1]), _uniform_str(args[2])))))


@builtin('uniqueval')
def _uniqueval(I, args, mask, hint):
    geo = _geo(I, args[0])
    arr = _attr_array(geo, _uniform_str(args[1]), _uniform_str(args[2]))
    vals = _unique(arr)
    idx = _ints(I, args[3])
    if arr is None:
        return const('int', 0)
    t = type_of_array(arr)
    if t == 'string':
        return V('string', np.array([vals[i] if 0 <= i < len(vals) else '' for i in idx], dtype=object))
    shape = arr.shape[1:]
    out = np.zeros((len(idx),) + shape, dtype=arr.dtype)
    for j, i in enumerate(idx):
        if 0 <= i < len(vals):
            out[j] = vals[i]
    return V(t, out)


@builtin('uniquevals')
def _uniquevals(I, args, mask, hint):
    from .interp import _make_array
    geo = _geo(I, args[0])
    arr = _attr_array(geo, _uniform_str(args[1]), _uniform_str(args[2]))
    if arr is None:
        return _make_array('int', [], I.n)
    t = type_of_array(arr)
    vals = _unique(arr)
    return _make_array(t, [const(t, v) for v in vals], I.n)


@builtin('findattribvalcount')
def _findattribvalcount(I, args, mask, hint):
    geo = _geo(I, args[0])
    arr = _attr_array(geo, _uniform_str(args[1]), _uniform_str(args[2]))
    vals = full(args[3], _n(I, args[3]))
    if arr is None:
        return const('int', 0)
    out = np.zeros(len(vals), dtype=np.int64)
    for i, v in enumerate(vals):
        out[i] = int(np.sum(arr == v)) if arr.ndim == 1 else int(np.sum(np.all(arr == v, axis=1)))
    return V('int', out)


def _name_lookup(I, cls):
    def impl(I, args, mask, hint):
        geo = _geo(I, args[0])
        arr = geo.attribs[cls].get('name')
        names = full(args[1], _n(I, args[1]))
        if arr is None:
            return V('int', np.full(len(names), -1, dtype=np.int64))
        first = {}
        for i, v in enumerate(arr.tolist()):
            first.setdefault(v, i)
        return V('int', np.array([first.get(nm, -1) for nm in names], dtype=np.int64))
    return impl


builtin('nametopoint')(_name_lookup(None, 'point'))
builtin('nametoprim')(_name_lookup(None, 'prim'))

_TYPE_CODES = {'int': 0, 'float': 1, 'string': 2}


def _make_attribtype(cls_fixed):
    def impl(I, args, mask, hint):
        geo = _geo(I, args[0])
        if cls_fixed is None:
            cls, name = _uniform_str(args[1]), _uniform_str(args[2])
        else:
            cls, name = cls_fixed, _uniform_str(args[1])
        arr = _attr_array(geo, cls, name)
        if arr is None:
            return const('int', -1)
        t = type_of_array(arr)
        return const('int', _TYPE_CODES.get(t, 1))
    return impl


def _make_attribsize(cls_fixed):
    def impl(I, args, mask, hint):
        geo = _geo(I, args[0])
        if cls_fixed is None:
            cls, name = _uniform_str(args[1]), _uniform_str(args[2])
        else:
            cls, name = cls_fixed, _uniform_str(args[1])
        arr = _attr_array(geo, cls, name)
        if arr is None:
            return const('int', 0)
        return const('int', int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1)
    return impl


builtin('attribtype')(_make_attribtype(None))
builtin('attribsize')(_make_attribsize(None))
for _c in ('point', 'prim', 'vertex', 'detail'):
    builtin(_c + 'attribtype')(_make_attribtype(_c))
    builtin(_c + 'attribsize')(_make_attribsize(_c))


# ----------------------------------------------------------------------------
# Math


@builtin('fmod')
def _fmod(I, args, mask, hint):
    a, b = args
    t = a.t if a.t != 'int' else 'float'
    return V(t, np.fmod(_f(convert(a, t)), _f(convert(b, t)) if b.t in SCALARS else _f(b)))


@builtin('hypot')
def _hypot(I, args, mask, hint):
    return V('float', np.hypot(_f(args[0]), _f(args[1])))


@builtin('cbrt')
def _cbrt(I, args, mask, hint):
    t = 'float' if args[0].t == 'int' else args[0].t
    return V(t, np.cbrt(_f(args[0])))


@builtin('isnan')
def _isnan(I, args, mask, hint):
    d = _f(args[0])
    return V('int', (np.isnan(d) if d.ndim == 1 else np.isnan(d).any(axis=tuple(range(1, d.ndim)))).astype(np.int64))


@builtin('isfinite')
def _isfinite(I, args, mask, hint):
    d = _f(args[0])
    return V('int', (np.isfinite(d) if d.ndim == 1 else np.isfinite(d).all(axis=tuple(range(1, d.ndim)))).astype(np.int64))


@builtin('outerproduct')
def _outerproduct(I, args, mask, hint):
    a, b = _f(convert(args[0], 'vector')), _f(convert(args[1], 'vector'))
    a, b = np.broadcast_arrays(a, b)
    return V('matrix3', np.einsum('ni,nj->nij', a, b))


def _expand(m, R3):
    n = len(R3)
    if m.t == 'matrix':
        R = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        R[:, :3, :3] = R3
        return R
    return R3


@builtin('prerotate', inplace=True)
def _prerotate(I, args, mask, hint):
    m, angle, axis = args
    n = _n(I, m, angle, axis)
    R = _expand(m, _axis_angle_matrix(full(convert(angle, 'float'), n), full(convert(axis, 'vector'), n)))
    return V(m.t, np.matmul(R, full(m, n))), None


@builtin('prescale', inplace=True)
def _prescale(I, args, mask, hint):
    m, s = args
    n = _n(I, m, s)
    k = MATRICES[m.t]
    S = np.broadcast_to(np.eye(k), (n, k, k)).copy()
    sv = full(convert(s, 'vector'), n)
    for i in range(3):
        S[:, i, i] = sv[:, i]
    return V(m.t, np.matmul(S, full(m, n))), None


@builtin('pretranslate', inplace=True)
def _pretranslate(I, args, mask, hint):
    m, t = args
    n = _n(I, m, t)
    T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    T[:, 3, :3] = full(convert(t, 'vector'), n)
    return V('matrix', np.matmul(T, full(convert(m, 'matrix'), n))), None


def _euler_from_row_matrix(M):
    """XYZ Euler angles (radians) of row-vector rotation matrices (Houdini's default order)."""
    C = np.swapaxes(M, 1, 2)   # column-vector form: C = Rz Ry Rx
    sy = np.clip(-C[:, 2, 0], -1.0, 1.0)
    beta = np.arcsin(sy)
    alpha = np.arctan2(C[:, 2, 1], C[:, 2, 2])
    gamma = np.arctan2(C[:, 1, 0], C[:, 0, 0])
    # Gimbal lock: put everything in X.
    lock = np.abs(sy) > 1 - 1e-9
    alpha = np.where(lock, np.arctan2(-C[:, 1, 2], C[:, 1, 1]), alpha)
    gamma = np.where(lock, 0.0, gamma)
    return np.stack([alpha, beta, gamma], axis=1)


@builtin('quaterniontoeuler')
def _q2euler(I, args, mask, hint):
    q = _f(convert(args[0], 'vector4'))
    return V('vector', _euler_from_row_matrix(_matrix3_from_quat(np.array(q, copy=True))))


@builtin('qinvert')
def _qinvert(I, args, mask, hint):
    q = np.array(_f(convert(args[0], 'vector4')), copy=True)
    q[:, :3] *= -1
    ln2 = np.maximum(np.sum(q * q, axis=1, keepdims=True), 1e-24)
    return V('vector4', q / ln2)


@builtin('cracktransform')
def _cracktransform(I, args, mask, hint):
    """cracktransform(trs, xyz, c, pivot, matrix): c = 0 translate, 1 rotate (degrees), 2 scale."""
    c = int(full(convert(args[2], 'int'), 1)[0])
    m = full(convert(args[4], 'matrix'), _n(I, args[4]))
    if c == 0:
        return V('vector', m[:, 3, :3].copy())
    rs = m[:, :3, :3]
    scale = np.linalg.norm(rs, axis=2)
    if c == 2:
        return V('vector', scale)
    R = rs / np.maximum(scale[:, :, None], 1e-12)
    return V('vector', np.degrees(_euler_from_row_matrix(R)))


@builtin('spline')
def _spline(I, args, mask, hint):
    """spline(basis, t, v0, v1, ...) with 'linear' / 'catrom' / 'constant' bases (also a single array)."""
    basis = _uniform_str(args[0]).lower()
    n = _n(I, *args[1:])
    t = np.clip(full(convert(args[1], 'float'), n), 0.0, 1.0)
    keys = args[2:]
    if len(keys) == 1 and keys[0].t.endswith('[]'):
        arr = full(keys[0], n)
        et = keys[0].t[:-2]
        vals = [V(et, np.array([a[min(k, len(a) - 1)] if len(a) else 0 for a in arr]))
                for k in range(max((len(a) for a in arr), default=0))]
        keys = vals
    if not keys:
        return const('float', 0.0)
    rt = 'float'
    for k in keys:
        if k.t in VECTORS:
            rt = k.t
    ks = np.stack([full(convert(k, rt), n).astype(np.float64) for k in keys], axis=1)  # (n, K[, d])
    K = ks.shape[1]
    if K == 1:
        return V(rt, ks[:, 0])
    if basis.startswith('const'):
        idx = np.minimum((t * K).astype(np.int64), K - 1)
        return V(rt, ks[np.arange(n), idx])
    if basis.startswith('cat') and K >= 4:
        # Catmull-Rom through the inner keys (first and last are tangent controls, as in Houdini).
        segs = K - 3
        x = t * segs
        i = np.minimum(x.astype(np.int64), segs - 1)
        u = x - i
        r = np.arange(n)
        p0, p1, p2, p3 = ks[r, i], ks[r, i + 1], ks[r, i + 2], ks[r, i + 3]
        if ks.ndim == 3:
            u = u[:, None]
        val = 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                     + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3)
        return V(rt, val)
    segs = K - 1
    x = t * segs
    i = np.minimum(x.astype(np.int64), segs - 1)
    u = x - i
    if ks.ndim == 3:
        u = u[:, None]
    r = np.arange(n)
    return V(rt, ks[r, i] * (1 - u) + ks[r, i + 1] * u)


@builtin('planepointdistance')
def _planepointdistance(I, args, mask, hint):
    o = _f(convert(args[0], 'vector'))
    nrm = _f(convert(args[1], 'vector'))
    p = _f(convert(args[2], 'vector'))
    o, nrm, p = np.broadcast_arrays(o, nrm, p)
    nn = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    return V('float', np.abs(np.einsum('ij,ij->i', p - o, nn)))


@builtin('ptlined')
def _ptlined(I, args, mask, hint):
    a, b, q = (_f(convert(x, 'vector')) for x in args[:3])
    a, b, q = np.broadcast_arrays(a, b, q)
    ab = b - a
    t = np.clip(np.einsum('ij,ij->i', q - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-24), 0, 1)
    return V('float', np.linalg.norm(a + ab * t[:, None] - q, axis=1))


@builtin('luminance')
def _luminance(I, args, mask, hint):
    c = _f(convert(args[0], 'vector'))
    return V('float', c @ np.array([0.2126, 0.7152, 0.0722]))


@builtin('blackbody')
def _blackbody(I, args, mask, hint):
    """Approximate RGB of a black body at a temperature in Kelvin (Tanner Helland's fit)."""
    temp = np.clip(_f(convert(args[0], 'float')), 1000, 40000) / 100.0
    r = np.where(temp <= 66, 255.0, 329.698727446 * np.power(np.maximum(temp - 60, 1e-6), -0.1332047592))
    g = np.where(temp <= 66, 99.4708025861 * np.log(temp) - 161.1195681661,
                 288.1221695283 * np.power(np.maximum(temp - 60, 1e-6), -0.0755148492))
    b = np.where(temp >= 66, 255.0, np.where(temp <= 19, 0.0, 138.5177312231 * np.log(np.maximum(temp - 10, 1e-6))
                                             - 305.0447927307))
    rgb = np.clip(np.stack([r, g, b], axis=1) / 255.0, 0, 1) ** 2.2
    if len(args) > 1:
        rgb = rgb * _f(convert(args[1], 'float'))[:, None]
    return V('vector', rgb)


@builtin('xyztorgb')
def _xyztorgb(I, args, mask, hint):
    m = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    return V('vector', _f(convert(args[0], 'vector')) @ m.T)


# ----------------------------------------------------------------------------
# Arrays


@builtin('slice')
def _slice(I, args, mask, hint):
    """slice(array_or_string, start, end[, step]) with Python semantics."""
    a = args[0]
    n = _n(I, *args)
    data = full(a, n)
    st = full(convert(args[1], 'int'), n) if len(args) > 1 else np.zeros(n, np.int64)
    en = full(convert(args[2], 'int'), n) if len(args) > 2 else None
    sp = full(convert(args[3], 'int'), n) if len(args) > 3 else np.ones(n, np.int64)
    out = np.empty(n, dtype=object)
    for i in range(n):
        s = slice(int(st[i]), int(en[i]) if en is not None else None, int(sp[i]) or 1)
        out[i] = data[i][s]
    return V(a.t, out)


@builtin('setcomp', inplace=True)
def _setcomp(I, args, mask, hint):
    m, val = args[0], args[1]
    n = _n(I, *args)
    d = np.array(full(m, n), dtype=np.float64, copy=True)
    v = full(convert(val, 'float'), n)
    i = full(convert(args[2], 'int'), n)
    r = np.arange(n)
    if m.t in MATRICES:
        j = full(convert(args[3], 'int'), n)
        d[r, i, j] = v
    else:
        d[r, np.clip(i, 0, d.shape[1] - 1)] = v
    return V(m.t, d), None


@builtin('removevalue', inplace=True)
def _removevalue(I, args, mask, hint):
    arr, val = args
    n = _n(I, arr, val)
    d = np.array(full(arr, n), dtype=object, copy=True)
    et = arr.t[:-2]
    vals = full(convert(val, et), n)
    found = np.zeros(n, dtype=np.int64)
    for i in np.nonzero(mask)[0]:
        a = d[i]
        for k in range(len(a)):
            same = a[k] == vals[i] if np.ndim(a[k]) == 0 else np.all(a[k] == vals[i])
            if same:
                d[i] = (list(a[:k]) + list(a[k + 1:])) if isinstance(a, list) else np.delete(a, k, axis=0)
                found[i] = 1
                break
    return V(arr.t, d), V('int', found)


@builtin('upush', inplace=True)
def _upush(I, args, mask, hint):
    arr, val = args
    n = _n(I, arr, val)
    d = np.array(full(arr, n), dtype=object, copy=True)
    et = arr.t[:-2]
    vals = full(convert(val, et), n)
    for i in np.nonzero(mask)[0]:
        a = d[i]
        present = any((x == vals[i]) if np.ndim(x) == 0 else np.all(x == vals[i]) for x in a)
        if not present:
            d[i] = list(a) + [vals[i]] if et == 'string' else np.concatenate([a, np.asarray(vals[i], dtype=a.dtype)[None]])
    return V(arr.t, d), None


# ----------------------------------------------------------------------------
# Strings


def _str_args(I, args):
    n = _n(I, *args)
    return n, [full(a, n) for a in args]


@builtin('re_match')
def _re_match(I, args, mask, hint):
    n, (pat, s) = _str_args(I, args[:2])
    return V('int', np.array([1 if re.match(p + r'\Z', x) else 0 for p, x in zip(pat, s)], dtype=np.int64))


@builtin('re_find')
def _re_find(I, args, mask, hint):
    """re_find(regex, input) -> the first match (a string)."""
    n, (pat, s) = _str_args(I, args[:2])
    out = []
    for p, x in zip(pat, s):
        m = re.search(p, x)
        out.append(m.group(0) if m else '')
    return V('string', np.array(out, dtype=object))


@builtin('re_findall')
def _re_findall(I, args, mask, hint):
    n, (pat, s) = _str_args(I, args[:2])
    out = np.empty(n, dtype=object)
    for i, (p, x) in enumerate(zip(pat, s)):
        out[i] = [m.group(0) for m in re.finditer(p, x)]
    return V('string[]', out)


@builtin('re_replace')
def _re_replace(I, args, mask, hint):
    """re_replace(regex, replacement, input[, maxreplace]); \\1 style back references."""
    n, (pat, rep, s) = _str_args(I, args[:3])
    count = int(full(convert(args[3], 'int'), 1)[0]) if len(args) > 3 else 0
    return V('string', np.array([re.sub(p, r, x, count=max(count, 0)) for p, r, x in zip(pat, rep, s)], dtype=object))


@builtin('re_split')
def _re_split(I, args, mask, hint):
    n, (pat, s) = _str_args(I, args[:2])
    out = np.empty(n, dtype=object)
    for i, (p, x) in enumerate(zip(pat, s)):
        out[i] = [t for t in re.split(p, x) if t != '']
    return V('string[]', out)


@builtin('match')
def _match(I, args, mask, hint):
    """match(pattern, string): glob style ('*', '?', '[abc]')."""
    n, (pat, s) = _str_args(I, args[:2])
    return V('int', np.array([1 if fnmatch.fnmatchcase(x, p) else 0 for p, x in zip(pat, s)], dtype=np.int64))


@builtin('lstrip')
def _lstrip(I, args, mask, hint):
    chars = _uniform_str(args[1]) if len(args) > 1 else None
    return V('string', np.array([x.lstrip(chars) for x in args[0].d], dtype=object))


@builtin('rstrip')
def _rstrip(I, args, mask, hint):
    chars = _uniform_str(args[1]) if len(args) > 1 else None
    return V('string', np.array([x.rstrip(chars) for x in args[0].d], dtype=object))


@builtin('isdigit')
def _isdigit(I, args, mask, hint):
    return V('int', np.array([1 if x.isdigit() else 0 for x in args[0].d], dtype=np.int64))


@builtin('isalpha')
def _isalpha(I, args, mask, hint):
    return V('int', np.array([1 if x.isalpha() else 0 for x in args[0].d], dtype=np.int64))


# ----------------------------------------------------------------------------
# Sampling / random


def _u2(I, args):
    u = _f(convert(args[0], 'vector2')) if args[0].t != 'float' else np.stack([_f(args[0]), np.zeros(len(args[0].d))], 1)
    return u


@builtin('sample_circle_uniform')
def _sample_circle(I, args, mask, hint):
    u = _f(convert(args[0], 'float'))
    a = 2 * np.pi * u
    return V('vector2', np.stack([np.cos(a), np.sin(a)], axis=1))


@builtin('sample_disk_uniform', 'sample_circle_area_uniform')
def _sample_disk(I, args, mask, hint):
    u = _u2(I, args)
    r = np.sqrt(u[:, 0])
    a = 2 * np.pi * u[:, 1]
    return V('vector2', np.stack([r * np.cos(a), r * np.sin(a)], axis=1))


@builtin('sample_hemisphere')
def _sample_hemisphere(I, args, mask, hint):
    """sample_hemisphere([center,] u): uniform directions around +Z (or around center)."""
    if len(args) > 1:
        center = _f(convert(args[0], 'vector'))
        u = _u2(I, args[1:])
    else:
        center = np.array([[0.0, 0.0, 1.0]])
        u = _u2(I, args)
    z = u[:, 0]
    r = np.sqrt(np.maximum(0, 1 - z * z))
    a = 2 * np.pi * u[:, 1]
    d = np.stack([r * np.cos(a), r * np.sin(a), z], axis=1)
    return V('vector', _orient_to(d, center))


@builtin('sample_direction_cone')
def _sample_cone(I, args, mask, hint):
    """sample_direction_cone(center, maxangle, u)."""
    center = _f(convert(args[0], 'vector'))
    maxang = _f(convert(args[1], 'float'))
    u = _u2(I, args[2:])
    n = max(len(center), len(maxang), len(u))
    maxang = np.broadcast_to(maxang, (n,))
    u = np.broadcast_to(u, (n, 2))
    z = 1 - u[:, 0] * (1 - np.cos(maxang))
    r = np.sqrt(np.maximum(0, 1 - z * z))
    a = 2 * np.pi * u[:, 1]
    d = np.stack([r * np.cos(a), r * np.sin(a), z], axis=1)
    return V('vector', _orient_to(d, np.broadcast_to(center, (n, 3))))


def _orient_to(d, axis):
    n = max(len(d), len(axis))
    d = np.broadcast_to(d, (n, 3))
    zax = np.broadcast_to(axis, (n, 3)).astype(np.float64)
    zax = zax / np.maximum(np.linalg.norm(zax, axis=1, keepdims=True), 1e-12)
    helper = np.where(np.abs(zax[:, 0:1]) < 0.9, [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]])
    xax = np.cross(helper, zax)
    xax /= np.maximum(np.linalg.norm(xax, axis=1, keepdims=True), 1e-12)
    yax = np.cross(zax, xax)
    return d[:, 0:1] * xax + d[:, 1:2] * yax + d[:, 2:3] * zax


@builtin('nrandom')
def _nrandom(I, args, mask, hint):
    """Non-deterministic random numbers (differ on every cook, as in Houdini)."""
    rng = np.random.default_rng()
    dims = _result_dims(hint)
    d = rng.random((I.n, dims)) if dims > 1 else rng.random(I.n)
    return V(VEC_BY_SIZE[dims], d)


@builtin('vnoise', 'flownoise', 'wnoise', 'curlxnoise')
def _noise_aliases(I, args, mask, hint):
    dims = _result_dims(hint)
    return V(VEC_BY_SIZE[dims], _noise.noise(_noise_pos(I, args), dims))


# ----------------------------------------------------------------------------
# Point cloud iteration (pcopen ... while (pciterate(h)) pcimport(h, "P", pos);)


@builtin('pciterate')
def _pciterate(I, args, mask, hint):
    geo, res, pos, radius = _pc_handle(I, args[0])
    key = ('pciter', int(args[0].d.flat[0]))
    it = I.ctx.cache.get(key)
    if it is None:
        it = I.ctx.cache[key] = np.full(I.n, -1, dtype=np.int64)
    lens = np.array([len(r) for r in res], dtype=np.int64)
    it[mask] += 1
    more = (it < lens) & mask
    it[mask & ~more] = -1   # finished: the next loop over this handle starts again
    return V('int', more.astype(np.int64))


@builtin('pcimport')
def _pcimport(I, args, mask, hint):
    geo, res, pos, radius = _pc_handle(I, args[0])
    key = ('pciter', int(args[0].d.flat[0]))
    it = I.ctx.cache.get(key, np.zeros(I.n, dtype=np.int64))
    name = _uniform_str(args[1])
    cur = np.full(I.n, -1, dtype=np.int64)
    for i in np.nonzero(mask)[0]:
        if 0 <= it[i] < len(res[i]):
            cur[i] = res[i][it[i]]
    ok = cur >= 0
    cc = np.clip(cur, 0, max(geo.num_points - 1, 0))
    if name == 'point.number':
        val = V('int', cur)
    elif name == 'point.distance':
        val = V('float', np.where(ok, np.linalg.norm(geo.P[cc] - pos, axis=1) if geo.num_points else 0.0, 0.0))
    else:
        arr = geo.P if name == 'P' else geo.attribs['point'].get(name)
        if arr is None:
            return const('int', 0), {}
        t = type_of_array(arr)
        val = V(t, arr[cc] if geo.num_points else arr[:0])
    return V('int', ok.astype(np.int64)), {2: val}


OUTARGS.add('pcimport')


# ----------------------------------------------------------------------------
# Geometry queries


@builtin('intersect')
def _intersect(I, args, mask, hint):
    """intersect(geo, orig, dir, &pos, &uvw): first hit prim along dir (its length is the max distance)."""
    from ..core.spatial import ray_cast
    geo = _geo(I, args[0])
    n = I.n
    orig = full(convert(args[1], 'vector'), n)
    d = full(convert(args[2], 'vector'), n)
    length = np.linalg.norm(d, axis=1)
    dirs = d / np.maximum(length[:, None], 1e-12)
    prim = np.full(n, -1, dtype=np.int64)
    pos = orig.copy()
    lanes = np.nonzero(mask & (length > 0))[0]
    if len(lanes) and geo.num_prims:
        hit, loc, _nrm, pr, dist = ray_cast(geo, orig[lanes], dirs[lanes])
        ok = hit & (dist <= length[lanes])
        prim[lanes[ok]] = pr[ok]
        pos[lanes[ok]] = loc[ok]
    uvw = np.zeros((n, 3))
    if (prim >= 0).any():
        from .builtins import _solve_uv
        uvw = _solve_uv(geo, prim, pos)
    return V('int', prim), {3: V('vector', pos), 4: V('vector', uvw)}


OUTARGS.add('intersect')


@builtin('primarclen', 'curvearclen')
def _primarclen(I, args, mask, hint):
    geo = _geo(I, args[0])
    prim = _ints(I, args[1])
    out = np.zeros(len(prim))
    for i, p in enumerate(prim):
        if 0 <= p < geo.num_prims:
            pts = geo.prim_points(p)
            P = geo.P[pts]
            if geo.prim_closed[p] and len(P) > 2:
                P = np.concatenate([P, P[:1]])
            out[i] = np.linalg.norm(np.diff(P, axis=0), axis=1).sum() if len(P) > 1 else 0.0
    return V('float', out)


_ = (SCALARS, _quat_from_matrix3, _point_prims_table)


# ----------------------------------------------------------------------------
# Volumes (native voxel grids, see core/volume.py)


def _volume(I, geo_arg, name_arg):
    from ..core.volume import volume_by_name
    geo = _geo(I, geo_arg)
    key = int(name_arg.d.flat[0]) if name_arg.t in SCALARS else _uniform_str(name_arg)
    return volume_by_name(geo, key)


@builtin('volumesample', 'volumesamplev')
def _volumesample(I, args, mask, hint):
    vol = _volume(I, args[0], args[1])
    pos = full(convert(args[2], 'vector'), I.n)
    if vol is None:
        return const('vector' if hint == 'vector' else 'float', 0.0) if hint != 'vector' else V('vector', np.zeros((1, 3)))
    val = vol.sample(pos)
    return V('vector', val) if vol.is_vector else V('float', val)


@builtin('volumegradient')
def _volumegradient(I, args, mask, hint):
    vol = _volume(I, args[0], args[1])
    if vol is None or vol.is_vector:
        return V('vector', np.zeros((1, 3)))
    return V('vector', vol.gradient(full(convert(args[2], 'vector'), I.n)))


@builtin('volumeres')
def _volumeres(I, args, mask, hint):
    vol = _volume(I, args[0], args[1])
    return V('vector', (vol.res.astype(np.float64) if vol is not None else np.zeros(3))[None])


@builtin('volumevoxeldiameter')
def _volumevoxeldiameter(I, args, mask, hint):
    vol = _volume(I, args[0], args[1])
    return const('float', vol.voxelsize * np.sqrt(3) if vol is not None else 0.0)


@builtin('volumepostoindex')
def _volumepostoindex(I, args, mask, hint):
    vol = _volume(I, args[0], args[1])
    pos = full(convert(args[2], 'vector'), I.n)
    if vol is None:
        return V('vector', np.zeros((1, 3)))
    return V('vector', vol.pos_to_index(pos).astype(np.float64))


@builtin('volumeindextopos')
def _volumeindextopos(I, args, mask, hint):
    vol = _volume(I, args[0], args[1])
    idx = full(convert(args[2], 'vector'), I.n)
    if vol is None:
        return V('vector', np.zeros((1, 3)))
    return V('vector', vol.index_to_pos(idx))


@builtin('volumeindex', 'volumeindexv')
def _volumeindex(I, args, mask, hint):
    vol = _volume(I, args[0], args[1])
    idx = full(convert(args[2], 'vector'), I.n).astype(np.int64)
    if vol is None:
        return const('float', 0.0)
    r = vol.res
    ok = np.all((idx >= 0) & (idx < r), axis=1)
    ic = np.clip(idx, 0, r - 1)
    val = vol.data[ic[:, 0], ic[:, 1], ic[:, 2]]
    if vol.is_vector:
        return V('vector', np.where(ok[:, None], val, 0.0))
    return V('float', np.where(ok, val, 0.0))


# ----------------------------------------------------------------------------
# Dictionaries and generic attribute access


@builtin('keys')
def _keys(I, args, mask, hint):
    d = full(args[0], I.n)
    out = np.empty(I.n, dtype=object)
    for i in range(I.n):
        out[i] = sorted(d[i].keys()) if isinstance(d[i], dict) else []
    return V('string[]', out)


@builtin('getattrib')
def _getattrib(I, args, mask, hint):
    """getattrib(geo, class, name, elemnum, &success)"""
    from .builtins import FUNCTIONS
    cls = _uniform_str(args[1])
    fn = {'point': 'point', 'prim': 'prim', 'primitive': 'prim', 'vertex': 'vertex', 'detail': 'detail',
          'global': 'detail'}.get(cls)
    if fn is None:
        raise VexError("getattrib: unknown attribute class '%s'" % cls)
    call = [args[0], args[2]] + ([args[3]] if fn != 'detail' else [])
    val = FUNCTIONS[fn](I, call, mask, hint)
    geo = _geo(I, args[0])
    name = _uniform_str(args[2])
    c = {'primitive': 'prim', 'global': 'detail'}.get(cls, cls)
    ok = name in geo.attribs.get(c, {}) or (name == 'P' and c == 'point')
    return val, {4: const('int', 1 if ok else 0)}


OUTARGS.add('getattrib')


# ----------------------------------------------------------------------------
# Topology edits, ray queries, sampling


@builtin('setvertexpoint', 'setprimvertex')
def _setvertexpoint(I, args, mask, hint):
    """setvertexpoint(geo, prim, vtx, pt): rewire a vertex to another point (prim -1: vtx is a linear vertex)."""
    geo = I.ctx.input(0)
    n = _n(I, *args[1:])
    prim, vtx, pt = (_ints(I, a, n) for a in args[1:4])
    for i in np.nonzero(mask)[0]:
        pr, v = int(prim[i]), int(vtx[i])
        if pr >= 0:
            if pr >= geo.num_prims or v < 0 or v >= geo.prim_start[pr + 1] - geo.prim_start[pr]:
                continue
            v = int(geo.prim_start[pr]) + v
        I.ctx.rewire.append((v, int(pt[i])))
    return const('int', 0)


@builtin('intersect_all')
def _intersect_all(I, args, mask, hint):
    """intersect_all(geo, orig, dir, &pos[], &prim[], &uvw[], tol=0.01, ttol=0.01): all hits along the ray."""
    from ..core.spatial import ray_cast
    from .interp import _make_array
    geo = _geo(I, args[0])
    n = I.n
    orig = full(convert(args[1], 'vector'), n)
    d = full(convert(args[2], 'vector'), n)
    length = np.linalg.norm(d, axis=1)
    dirs = d / np.maximum(length[:, None], 1e-12)
    pos_out = np.empty(n, dtype=object)
    prim_out = np.empty(n, dtype=object)
    uvw_out = np.empty(n, dtype=object)
    count = np.zeros(n, dtype=np.int64)
    for i in range(n):
        pos_out[i], prim_out[i], uvw_out[i] = np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros((0, 3))
    if geo.num_prims:
        for i in np.nonzero(mask & (length > 0))[0]:
            o, travelled, hits_p, hits_pr = orig[i].copy(), 0.0, [], []
            for _ in range(64):
                hit, loc, _nrm, pr, dist = ray_cast(geo, o[None], dirs[i][None], length[i] - travelled)
                if not hit[0]:
                    break
                hits_p.append(loc[0])
                hits_pr.append(int(pr[0]))
                travelled += dist[0] + 1e-5
                o = loc[0] + dirs[i] * 1e-5
                if travelled >= length[i]:
                    break
            if hits_p:
                pos_out[i] = np.array(hits_p)
                prim_out[i] = np.array(hits_pr, dtype=np.int64)
                uvw_out[i] = np.zeros((len(hits_p), 3))
                count[i] = len(hits_p)
    _ = _make_array
    return V('int', count), {3: V('vector[]', pos_out), 4: V('int[]', prim_out), 5: V('vector[]', uvw_out)}


OUTARGS.add('intersect_all')


@builtin('pcfind_radius')
def _pcfind_radius(I, args, mask, hint):
    """pcfind_radius(geo, Pchannel, radchannel, radscale, pos, radius, maxpts): points whose own radius
    (pscale * radscale) plus the search radius reaches pos."""
    geo = _geo(I, args[0])
    n = I.n
    radname = _uniform_str(args[2])
    rs = full(convert(args[3], 'float'), n)
    pos = full(convert(args[4], 'vector'), n)
    radius = full(convert(args[5], 'float'), n)
    maxpts = full(convert(args[6], 'int'), n) if len(args) > 6 else np.full(n, 1 << 30)
    prad = geo.attribs['point'].get(radname)
    prad = np.asarray(prad, dtype=np.float64) if prad is not None and prad.dtype != object else np.zeros(geo.num_points)
    out = np.empty(n, dtype=object)
    for i in range(n):
        if not mask[i] or geo.num_points == 0:
            out[i] = np.zeros(0, np.int64)
            continue
        dist = np.linalg.norm(geo.P - pos[i], axis=1)
        ok = np.nonzero(dist <= radius[i] + prad * rs[i])[0]
        out[i] = ok[np.argsort(dist[ok])][:max(int(maxpts[i]), 0)]
    return V('int[]', out)


@builtin('isbound')
def _isbound(I, args, mask, hint):
    """isbound(name): 1 when the attribute exists on the geometry being processed."""
    name = _uniform_str(args[0])
    geo = I.ctx.input(0)
    cls = getattr(I.ctx, 'run_over', 'point')
    return const('int', 1 if name in geo.attribs.get(cls, {}) or (name == 'P' and cls == 'point') else 0)


@builtin('uvsample')
def _uvsample(I, args, mask, hint):
    """uvsample(geo, attrib, uvname, uv): an attribute at a texture coordinate (barycentric on the uv triangles)."""
    from .values import type_of_array as _toa
    geo = _geo(I, args[0])
    name = _uniform_str(args[1])
    uvname = _uniform_str(args[2])
    uv = full(convert(args[3], 'vector'), I.n)[:, :2]
    tris, tri_prim = geo.triangulate_flat()
    uvv = geo.attribs['vertex'].get(uvname)
    uvp = geo.attribs['point'].get(uvname)
    src = geo.P if name == 'P' else geo.attribs['point'].get(name)
    if src is None or not len(tris) or (uvv is None and uvp is None):
        return const('vector' if hint == 'vector' else 'float', 0.0) if hint != 'vector' else V('vector', np.zeros((1, 3)))
    # Point uv per triangle corner (vertex uvs looked up through the first vertex using each point).
    if uvp is not None:
        tuv = uvp[tris][:, :, :2]
    else:
        first = np.full(geo.num_points, 0, dtype=np.int64)
        first[geo.vtx_point[::-1]] = np.arange(geo.num_vertices)[::-1]
        tuv = uvv[first[tris]][:, :, :2]
    vals = np.asarray(src, dtype=np.float64)[tris]
    t = _toa(src)
    out = np.zeros((I.n,) + vals.shape[2:])
    a, b, c = tuv[:, 0], tuv[:, 1], tuv[:, 2]
    v0, v1 = b - a, c - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    for i in np.nonzero(mask)[0]:
        v2 = uv[i] - a
        with np.errstate(divide='ignore', invalid='ignore'):
            bb = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
            cc = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        inside = np.nonzero((bb >= -1e-6) & (cc >= -1e-6) & (bb + cc <= 1 + 1e-6))[0]
        if len(inside):
            k = inside[0]
            w = np.array([1 - bb[k] - cc[k], bb[k], cc[k]])
            out[i] = np.tensordot(w, vals[k], axes=1)
    return V(t, out)
