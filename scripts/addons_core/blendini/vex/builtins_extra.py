# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Further VEX builtins: Worley / Voronoi cellular noise, array reordering, attribute type info."""

import numpy as np

from .builtins import builtin, OUTARGS, FUNCTIONS, _n, _f, _uniform_str, _geo
from .values import V, VexError, SCALARS, VECTORS, is_array, elem_type, full, convert, type_of_array
from ..core import noise as _noise


# ----------------------------------------------------------------------------
# Cellular noise: one jittered feature point per unit cell, distances to the nearest two.


def _cell_pos(I, v):
    """Position as (n, k) floats, k = 1..4 (the dimension of the noise)."""
    n = _n(I, v)
    if v.t in SCALARS:
        return full(convert(v, 'float'), n)[:, None]
    if v.t in VECTORS:
        return np.asarray(full(v, n), dtype=np.float64)
    raise VexError("cellular noise needs a float or vector position, got %s" % v.t)


def _worley(p, jitter, period=None, metric='euclidean'):
    """Vectorized Worley noise over the 3^k neighbouring cells.
    Returns f1, f2 (distances), seed (int of the nearest cell), pos1, pos2 (feature points)."""
    n, k = p.shape
    base = np.floor(p)
    f1 = np.full(n, np.inf)
    f2 = np.full(n, np.inf)
    seed = np.zeros(n, dtype=np.int64)
    p1 = np.zeros((n, k))
    p2 = np.zeros((n, k))
    offsets = np.stack(np.meshgrid(*([np.arange(-1, 2)] * k), indexing='ij'), -1).reshape(-1, k)
    for off in offsets:
        cell = base + off
        key = cell if period is None else np.mod(cell, period)
        h = _noise.hash_seed(key)
        rnd = _noise.rand(key, k).reshape(n, k)
        feat = cell + 0.5 + (rnd - 0.5) * jitter
        diff = feat - p
        if metric == 'manhattan':
            d = np.abs(diff).sum(axis=1)
        elif metric == 'chebyshev':
            d = np.abs(diff).max(axis=1)
        else:
            d = np.sqrt((diff * diff).sum(axis=1))
        nearer = d < f1
        second = ~nearer & (d < f2)
        f2 = np.where(nearer, f1, np.where(second, d, f2))
        p2 = np.where(nearer[:, None], p1, np.where(second[:, None], feat, p2))
        f1 = np.where(nearer, d, f1)
        p1 = np.where(nearer[:, None], feat, p1)
        seed = np.where(nearer, (h & np.uint64(0x7FFFFFFF)).astype(np.int64), seed)
    return f1, f2, seed, p1, p2


def _periods(I, args, k):
    if not args:
        return None
    vals = [float(np.asarray(_f(convert(a, 'float'))).ravel()[0]) for a in args]
    if len(vals) == 1 and args[0].t in VECTORS:
        vals = list(np.asarray(_f(args[0])).ravel()[:k])
    vals = (vals + [vals[-1]] * k)[:k]
    return np.maximum(np.array(vals), 1.0)


def _wnoise_impl(metric):
    def impl(I, args, mask, hint):
        # wnoise(pos, &seed, &f1, &f2[, &f3, &f4]) or wnoise(pos, &seed, &f1, &f2, periodx, ...)
        p = _cell_pos(I, args[0])
        period = None
        if len(args) > 4 and args[4].t == 'int':
            period = _periods(I, args[4:], p.shape[1])
        f1, f2, seed, _p1, _p2 = _worley(p, 1.0, period, metric)
        outs = {1: V('int', seed), 2: V('float', f1), 3: V('float', f2)}
        if len(args) > 4 and period is None:
            # f3 / f4 are approximated by f2 (only the nearest two are tracked).
            outs[4] = V('float', f2)
            if len(args) > 5:
                outs[5] = V('float', f2)
        return V('float', f1), outs
    return impl


for _name, _metric in (('wnoise', 'euclidean'), ('mwnoise', 'manhattan'), ('cwnoise', 'chebyshev')):
    builtin(_name)(_wnoise_impl(_metric))
    OUTARGS.add(_name)


@builtin('vnoise')
def _vnoise(I, args, mask, hint):
    """vnoise(pos, jitter, &seed, &f1, &f2, &pos1, &pos2[, period ...]): Voronoi noise."""
    p = _cell_pos(I, args[0])
    k = p.shape[1]
    if len(args) < 7:
        # Short form (not in VEX, kept for convenience): the distance to the nearest feature point.
        return V('float', _worley(p, 1.0)[0]), {}
    jit = _f(convert(args[1], 'float' if k == 1 else args[0].t if args[0].t in VECTORS else 'float'))
    jit = np.asarray(jit, dtype=np.float64)
    jit = jit.reshape(jit.shape[0], -1)
    period = _periods(I, args[7:], k) if len(args) > 7 else None
    f1, f2, seed, p1, p2 = _worley(p, jit, period)
    pt = args[0].t if args[0].t in VECTORS else 'float'
    pos1 = V(pt, p1 if k > 1 else p1[:, 0])
    pos2 = V(pt, p2 if k > 1 else p2[:, 0])
    return None, {2: V('int', seed), 3: V('float', f1), 4: V('float', f2), 5: pos1, 6: pos2}


OUTARGS.add('vnoise')


# ----------------------------------------------------------------------------
# Arrays


@builtin('reorder')
def _reorder(I, args, mask, hint):
    """reorder(values, indices): values[indices[0]], values[indices[1]] ..."""
    arr, idx = args
    if not is_array(arr.t):
        raise VexError("reorder() needs an array, got %s" % arr.t)
    n = _n(I, arr, idx)
    d = full(arr, n)
    ii = full(convert(idx, 'int[]'), n)
    out = np.empty(n, dtype=object)
    for i in range(n):
        a = d[i]
        sel = np.asarray(ii[i], dtype=np.int64)
        sel = sel[(sel >= -len(a)) & (sel < len(a))]
        out[i] = [a[j] for j in sel] if isinstance(a, list) else np.asarray(a)[sel]
    return V(arr.t, out)


@builtin('product')
def _product(I, args, mask, hint):
    a = args[0]
    if is_array(a.t):
        et = elem_type(a.t)
        d = full(a, I.n)
        shape = np.asarray(d[0]).shape[1:] if len(d) else ()
        out = np.ones((I.n,) + shape, dtype=np.int64 if et == 'int' else np.float64)
        for i in range(I.n):
            if len(d[i]):
                out[i] = np.prod(np.asarray(d[i]), axis=0)
        return V(et, out)
    return V('float', np.prod(_f(a), axis=1))


# ----------------------------------------------------------------------------
# Attribute type info. Blendini doesn't store type info on geometry, so attribtypeinfo()
# reports Houdini's defaults, overridden by setattribtypeinfo() calls in the same snippet.

_DEFAULT_TYPEINFO = {'P': 'point', 'rest': 'point', 'pivot': 'point', 'N': 'normal', 'up': 'vector',
                     'v': 'vector', 'force': 'vector', 'accel': 'vector', 'Cd': 'color', 'Cs': 'color',
                     'orient': 'quaternion', 'rot': 'quaternion', 'uv': 'texturecoord', 'transform': 'matrix'}


def _cls(s):
    return {'primitive': 'prim', 'global': 'detail', 'points': 'point'}.get(s, s)


@builtin('setattribtypeinfo')
def _setattribtypeinfo(I, args, mask, hint):
    info = I.ctx.cache.setdefault('typeinfo', {})
    info[(_cls(_uniform_str(args[1])), _uniform_str(args[2]))] = _uniform_str(args[3])
    return V('string', np.array([_uniform_str(args[2])], dtype=object))


@builtin('attribtypeinfo')
def _attribtypeinfo(I, args, mask, hint):
    geo = _geo(I, args[0])
    cls, name = _cls(_uniform_str(args[1])), _uniform_str(args[2])
    info = I.ctx.cache.get('typeinfo', {}).get((cls, name))
    if info is None:
        arr = geo.P if (cls, name) == ('point', 'P') else geo.attribs.get(cls, {}).get(name)
        info = '' if arr is None else _DEFAULT_TYPEINFO.get(name, 'none')
    return V('string', np.array([info], dtype=object))


# attribtype(): 0 int, 1 float, 2 string, 3 int array, 4 float array, 5 string array, 6 dict.
_TYPE_CODES = {'int': 0, 'float': 1, 'string': 2, 'int[]': 3, 'string[]': 5, 'dict': 6}


def _make_attribtype(cls_fixed):
    def impl(I, args, mask, hint):
        geo = _geo(I, args[0])
        if cls_fixed is None:
            cls, name = _cls(_uniform_str(args[1])), _uniform_str(args[2])
        else:
            cls, name = cls_fixed, _uniform_str(args[1])
        arr = geo.P if (cls, name) == ('point', 'P') else geo.attribs.get(cls, {}).get(name)
        if arr is None:
            return V('int', np.array([-1], dtype=np.int64))
        t = type_of_array(arr)
        code = _TYPE_CODES.get(t, 4 if is_array(t) else 1)
        return V('int', np.array([code], dtype=np.int64))
    return impl


FUNCTIONS['attribtype'] = _make_attribtype(None)
for _c in ('point', 'prim', 'vertex', 'detail'):
    FUNCTIONS[_c + 'attribtype'] = _make_attribtype(_c)
