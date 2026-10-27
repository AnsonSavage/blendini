# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
VEX builtin functions.

Each function has the signature ``fn(interp, args, mask, hint) -> V`` where ``args`` is a
list of evaluated lane values, ``mask`` the active lanes and ``hint`` the type expected by
the caller (VEX overloads some functions on return type, e.g. ``vector r = rand(1);``).
"""

import math
import numpy as np

from .values import (
    V, VexError, SCALARS, VECTORS, MATRICES, VEC_BY_SIZE, SHAPES,
    is_array, elem_type, const, zeros, full, convert, truthy, type_of_array, dtype_for, _empty_array, _fmt_value,
)
from ..core import noise as _noise

FUNCTIONS = {}
# Functions that modify their first argument: they return (new_first_arg, result).
INPLACE = set()


def builtin(*names, inplace=False):
    def deco(fn):
        for n in names:
            FUNCTIONS[n] = fn
            if inplace:
                INPLACE.add(n)
        return fn
    return deco


def _n(I, *vals):
    return max([I.n] + [v.d.shape[0] for v in vals])


def _f(v):
    """As float data (scalars, vectors, matrices)."""
    if v.t == 'int':
        return v.d.astype(np.float64)
    if v.t == 'string' or is_array(v.t):
        raise VexError("Expected a number, got %s" % v.t)
    return v.d


def _uniform_int(v):
    return int(v.d.flat[0]) if v.t in SCALARS else 0


def _uniform_str(v):
    if v.t != 'string':
        raise VexError("Expected a string, got %s" % v.t)
    return v.d[0]


def _float_type(t):
    return 'float' if t == 'int' else t


# ----------------------------------------------------------------------------
# Math


def _unary_math(np_fn, keep_int=False):
    def fn(I, args, mask, hint):
        a = args[0]
        if keep_int and a.t == 'int':
            return V('int', np_fn(a.d).astype(np.int64))
        return V(_float_type(a.t), np_fn(_f(a)))
    return fn


for _name, _fn in {
    'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'asin': np.arcsin, 'acos': np.arccos,
    'sinh': np.sinh, 'cosh': np.cosh, 'tanh': np.tanh, 'exp': np.exp,
    'log': lambda x: np.log(np.where(x > 0, x, 1e-300)), 'log10': lambda x: np.log10(np.where(x > 0, x, 1e-300)),
    'sqrt': lambda x: np.sqrt(np.maximum(x, 0)), 'floor': np.floor, 'ceil': np.ceil, 'frac': lambda x: x - np.floor(x),
    'rint': np.rint, 'trunc': np.trunc, 'degrees': np.degrees, 'radians': np.radians,
}.items():
    FUNCTIONS[_name] = _unary_math(_fn)

FUNCTIONS['abs'] = _unary_math(np.abs, keep_int=True)
FUNCTIONS['sign'] = _unary_math(np.sign, keep_int=True)


@builtin('round')
def _round(I, args, mask, hint):
    d = _f(args[0])
    return V(_float_type(args[0].t), np.floor(d + 0.5))


@builtin('atan', 'atan2')
def _atan(I, args, mask, hint):
    if len(args) == 1:
        return V('float', np.arctan(_f(args[0])))
    return V('float', np.arctan2(_f(args[0]), _f(args[1])))


@builtin('pow')
def _pow(I, args, mask, hint):
    a, b = args
    t = _float_type(a.t)
    db = _f(b)
    if a.t in VECTORS and b.t in SCALARS:
        db = db[:, None]
    return V(t, np.power(_f(a), db))


def _same_type(args):
    t = args[0].t
    for a in args[1:]:
        if a.t != t:
            if 'float' in (t, a.t) and t in SCALARS and a.t in SCALARS:
                t = 'float'
            elif a.t in VECTORS:
                t = a.t
    return t, [convert(a, t) for a in args]


@builtin('min')
def _min(I, args, mask, hint):
    if len(args) == 1 and is_array(args[0].t):
        return _array_reduce(I, args[0], np.min)
    t, vals = _same_type(args)
    d = vals[0].d
    for v in vals[1:]:
        d = np.minimum(d, v.d)
    return V(t, d)


@builtin('max')
def _max(I, args, mask, hint):
    if len(args) == 1 and is_array(args[0].t):
        return _array_reduce(I, args[0], np.max)
    t, vals = _same_type(args)
    d = vals[0].d
    for v in vals[1:]:
        d = np.maximum(d, v.d)
    return V(t, d)


@builtin('sum')
def _sum(I, args, mask, hint):
    if is_array(args[0].t):
        return _array_reduce(I, args[0], np.sum)
    return V('float', np.sum(_f(args[0]), axis=1))


def _array_reduce(I, arr, fn):
    et = elem_type(arr.t)
    d = full(arr, I.n)
    out = zeros(et, I.n).d
    for i in range(I.n):
        if len(d[i]):
            out[i] = fn(np.asarray(d[i]), axis=0)
    return V(et, out)


def _bc(x, like):
    """Broadcast scalar data to the trailing shape of like."""
    while x.ndim < like.ndim:
        x = x[..., None]
    return x


@builtin('clamp')
def _clamp(I, args, mask, hint):
    v, lo, hi = args
    d = _f(v)
    return V(_float_type(v.t) if v.t != 'int' else 'int', np.clip(v.d, _bc(lo.d, v.d), _bc(hi.d, v.d)))


@builtin('fit')
def _fit(I, args, mask, hint):
    x, omin, omax, nmin, nmax = args
    t = _float_type(x.t)
    for a in (nmin, nmax):
        if a.t in VECTORS:
            t = a.t
    dx = _f(convert(x, t) if t in VECTORS and x.t in SCALARS else x)
    o0, o1 = _bc(_f(omin), dx), _bc(_f(omax), dx)
    n0, n1 = _bc(_f(convert(nmin, t)), dx), _bc(_f(convert(nmax, t)), dx)
    rng = o1 - o0
    u = np.clip(np.where(rng == 0, 0.0, (dx - o0) / np.where(rng == 0, 1, rng)), 0, 1)
    u = np.where(rng < 0, np.clip((dx - o0) / np.where(rng == 0, 1, rng), 0, 1), u)
    return V(t, n0 + (n1 - n0) * u)


@builtin('efit')
def _efit(I, args, mask, hint):
    x, omin, omax, nmin, nmax = args
    dx = _f(x)
    o0, o1, n0, n1 = (_bc(_f(a), dx) for a in (omin, omax, nmin, nmax))
    rng = np.where(o1 - o0 == 0, 1, o1 - o0)
    return V(_float_type(x.t), n0 + (n1 - n0) * (dx - o0) / rng)


@builtin('fit01')
def _fit01(I, args, mask, hint):
    return _fit(I, [args[0], const('float', 0.0), const('float', 1.0), args[1], args[2]], mask, hint)


@builtin('fit10')
def _fit10(I, args, mask, hint):
    return _fit(I, [args[0], const('float', 1.0), const('float', 0.0), args[1], args[2]], mask, hint)


@builtin('fit11')
def _fit11(I, args, mask, hint):
    return _fit(I, [args[0], const('float', -1.0), const('float', 1.0), args[1], args[2]], mask, hint)


@builtin('lerp', 'mix')
def _lerp(I, args, mask, hint):
    a, b, t = args
    rt, (a, b) = _same_type([a, b])
    rt = _float_type(rt)
    da, db = _f(convert(a, rt)), _f(convert(b, rt))
    dt = _f(t)
    if t.t in SCALARS:
        dt = _bc(dt, da if da.ndim >= db.ndim else db)
    return V(rt, da + (db - da) * dt)


@builtin('smooth')
def _smooth(I, args, mask, hint):
    lo, hi, x = args[:3]
    d0, d1, dx = _f(lo), _f(hi), _f(x)
    rng = np.where(d1 - d0 == 0, 1, d1 - d0)
    u = np.clip((dx - d0) / rng, 0, 1)
    return V('float', u * u * (3 - 2 * u))


@builtin('length')
def _length(I, args, mask, hint):
    a = args[0]
    if a.t in SCALARS:
        return V('float', np.abs(_f(a)))
    return V('float', np.linalg.norm(a.d, axis=1))


@builtin('length2')
def _length2(I, args, mask, hint):
    a = args[0]
    if a.t in SCALARS:
        return V('float', _f(a) ** 2)
    return V('float', np.sum(a.d * a.d, axis=1))


@builtin('normalize')
def _normalize(I, args, mask, hint):
    a = args[0]
    if a.t in SCALARS:
        return V('float', np.sign(_f(a)))
    ln = np.linalg.norm(a.d, axis=1, keepdims=True)
    return V(a.t, a.d / np.where(ln == 0, 1, ln))


@builtin('dot')
def _dot(I, args, mask, hint):
    t, (a, b) = _same_type(args)
    return V('float', np.sum(a.d * b.d, axis=-1))


@builtin('cross')
def _cross(I, args, mask, hint):
    a, b = convert(args[0], 'vector'), convert(args[1], 'vector')
    n = _n(I, a, b)
    return V('vector', np.cross(full(a, n), full(b, n)))


@builtin('distance')
def _distance(I, args, mask, hint):
    t, (a, b) = _same_type(args)
    d = a.d - b.d
    if t in SCALARS:
        return V('float', np.abs(d).astype(np.float64))
    return V('float', np.linalg.norm(d, axis=1))


@builtin('distance2')
def _distance2(I, args, mask, hint):
    t, (a, b) = _same_type(args)
    d = a.d - b.d
    if t in SCALARS:
        return V('float', (d * d).astype(np.float64))
    return V('float', np.sum(d * d, axis=1))


@builtin('avg')
def _avg(I, args, mask, hint):
    if len(args) == 1:
        a = args[0]
        if a.t in VECTORS:
            return V('float', np.mean(a.d, axis=1))
        if is_array(a.t):
            return _array_reduce(I, a, np.mean)
        return convert(a, 'float')
    t, vals = _same_type(args)
    t = _float_type(t)
    return V(t, sum(_f(convert(v, t)) for v in vals) / len(vals))


@builtin('reflect')
def _reflect(I, args, mask, hint):
    v, n = _f(args[0]), _f(args[1])
    return V('vector', v - 2 * np.sum(v * n, axis=1, keepdims=True) * n)


@builtin('set')
def _set(I, args, mask, hint):
    from .interp import _assemble
    if len(args) == 1:
        return convert(args[0], hint or 'vector')
    t = {2: 'vector2', 3: 'vector', 4: 'vector4', 9: 'matrix3', 16: 'matrix'}.get(len(args), 'vector')
    return _assemble(t, args, I.n)


@builtin('getcomp')
def _getcomp(I, args, mask, hint):
    v, i = args[0], convert(args[1], 'int')
    n = _n(I, v, i)
    d = full(v, n).reshape(n, -1)
    return V('float', d[np.arange(n), np.clip(full(i, n), 0, d.shape[1] - 1)])


@builtin('ident')
def _ident(I, args, mask, hint):
    t = hint if hint in MATRICES else 'matrix'
    return zeros(t, 1)


@builtin('transpose')
def _transpose(I, args, mask, hint):
    return V(args[0].t, np.swapaxes(args[0].d, 1, 2))


@builtin('invert')
def _invert(I, args, mask, hint):
    a = args[0]
    if a.t in MATRICES:
        try:
            return V(a.t, np.linalg.inv(a.d))
        except np.linalg.LinAlgError:
            return V(a.t, np.linalg.pinv(a.d))
    return V('float', np.where(_f(a) == 0, 0, 1.0 / np.where(_f(a) == 0, 1, _f(a))))


@builtin('determinant')
def _det(I, args, mask, hint):
    return V('float', np.linalg.det(args[0].d))


def _axis_angle_matrix(angle, axis):
    """Rotation matrices (N, 3, 3) for row vectors (VEX convention)."""
    axis = np.asarray(axis, dtype=np.float64)
    ln = np.linalg.norm(axis, axis=1, keepdims=True)
    axis = axis / np.where(ln == 0, 1, ln)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    # Column-vector rotation matrix, transposed for row vectors.
    m = np.stack([
        np.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], axis=-1),
        np.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], axis=-1),
        np.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], axis=-1),
    ], axis=1)
    return np.swapaxes(m, 1, 2)


@builtin('rotate', inplace=True)
def _rotate(I, args, mask, hint):
    m, angle, axis = args
    n = _n(I, m, angle, axis)
    ang = full(convert(angle, 'float'), n)
    ax = full(convert(axis, 'vector'), n)
    R = _axis_angle_matrix(ang, ax)
    md = full(m, n)
    if m.t == 'matrix':
        R4 = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        R4[:, :3, :3] = R
        R = R4
    return V(m.t, np.matmul(md, R)), None


@builtin('scale', inplace=True)
def _scale(I, args, mask, hint):
    m, s = args
    n = _n(I, m, s)
    sv = full(convert(s, 'vector'), n)
    k = MATRICES[m.t]
    S = np.broadcast_to(np.eye(k), (n, k, k)).copy()
    for i in range(3):
        S[:, i, i] = sv[:, i]
    return V(m.t, np.matmul(full(m, n), S)), None


@builtin('translate', inplace=True)
def _translate(I, args, mask, hint):
    m, t = args
    n = _n(I, m, t)
    T = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    T[:, 3, :3] = full(convert(t, 'vector'), n)
    return V('matrix', np.matmul(full(convert(m, 'matrix'), n), T)), None


@builtin('maketransform')
def _maketransform(I, args, mask, hint):
    # maketransform(zaxis, yaxis[, translate]) -> matrix3 / matrix
    z = _f(convert(args[0], 'vector'))
    y = _f(convert(args[1], 'vector'))
    n = max(len(z), len(y), I.n)
    z = np.broadcast_to(z, (n, 3))
    y = np.broadcast_to(y, (n, 3))
    zn = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    x = np.cross(y, zn)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    yn = np.cross(zn, x)
    m3 = np.stack([x, yn, zn], axis=1)
    if len(args) >= 3 or hint == 'matrix':
        m4 = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
        m4[:, :3, :3] = m3
        if len(args) >= 3:
            m4[:, 3, :3] = np.broadcast_to(_f(convert(args[2], 'vector')), (n, 3))
        return V('matrix', m4)
    return V('matrix3', m3)


@builtin('dihedral')
def _dihedral(I, args, mask, hint):
    a = _f(convert(args[0], 'vector'))
    b = _f(convert(args[1], 'vector'))
    n = max(len(a), len(b))
    a = np.broadcast_to(a, (n, 3))
    b = np.broadcast_to(b, (n, 3))
    an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    bn = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    axis = np.cross(an, bn)
    s = np.linalg.norm(axis, axis=1)
    c = np.sum(an * bn, axis=1)
    ang = np.arctan2(s, c)
    fallback = np.where(np.abs(an[:, 0:1]) < 0.9, [[1, 0, 0]], [[0, 1, 0]])
    axis = np.where((s < 1e-9)[:, None], np.cross(an, fallback), axis)
    return V('matrix3', _axis_angle_matrix(ang, axis))


def _quat_from_matrix3(m):
    # m is (N,3,3) row-vector matrix; convert column-vector form.
    R = np.swapaxes(m, 1, 2)
    n = len(R)
    q = np.zeros((n, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    s = np.sqrt(np.maximum(tr + 1.0, 1e-12)) * 2
    q[:, 3] = 0.25 * s
    q[:, 0] = (R[:, 2, 1] - R[:, 1, 2]) / s
    q[:, 1] = (R[:, 0, 2] - R[:, 2, 0]) / s
    q[:, 2] = (R[:, 1, 0] - R[:, 0, 1]) / s
    ln = np.linalg.norm(q, axis=1, keepdims=True)
    return q / np.where(ln == 0, 1, ln)


def _matrix3_from_quat(q):
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)
    return np.swapaxes(R, 1, 2)


@builtin('quaternion')
def _quaternion(I, args, mask, hint):
    if len(args) == 1:
        a = args[0]
        if a.t in MATRICES:
            return V('vector4', _quat_from_matrix3(convert(a, 'matrix3').d))
        # angle * axis vector
        v = _f(convert(a, 'vector'))
        ang = np.linalg.norm(v, axis=1)
        return V('vector4', _quat_from_matrix3(_axis_angle_matrix(ang, v)))
    ang = _f(convert(args[0], 'float'))
    ax = _f(convert(args[1], 'vector'))
    n = max(len(ang), len(ax))
    return V('vector4', _quat_from_matrix3(_axis_angle_matrix(np.broadcast_to(ang, (n,)), np.broadcast_to(ax, (n, 3)))))


@builtin('qconvert')
def _qconvert(I, args, mask, hint):
    return V('matrix3', _matrix3_from_quat(_f(convert(args[0], 'vector4'))))


@builtin('qrotate')
def _qrotate(I, args, mask, hint):
    q = _f(convert(args[0], 'vector4'))
    v = _f(convert(args[1], 'vector'))
    n = max(len(q), len(v))
    m = _matrix3_from_quat(np.broadcast_to(q, (n, 4)).copy())
    return V('vector', np.einsum('ni,nij->nj', np.broadcast_to(v, (n, 3)), m))


@builtin('qmultiply')
def _qmultiply(I, args, mask, hint):
    a = _f(convert(args[0], 'vector4'))
    b = _f(convert(args[1], 'vector4'))
    a, b = np.broadcast_arrays(a, b)
    x1, y1, z1, w1 = a.T
    x2, y2, z2, w2 = b.T
    return V('vector4', np.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], axis=1))


@builtin('slerp')
def _slerp(I, args, mask, hint):
    a = _f(convert(args[0], 'vector4'))
    b = _f(convert(args[1], 'vector4'))
    t = _f(convert(args[2], 'float'))
    a, b = np.broadcast_arrays(a, b)
    n = len(a)
    t = np.broadcast_to(t, (n,))[:, None]
    d = np.sum(a * b, axis=1, keepdims=True)
    b = np.where(d < 0, -b, b)
    d = np.abs(d)
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    small = s < 1e-6
    wa = np.where(small, 1 - t, np.sin((1 - t) * th) / np.where(small, 1, s))
    wb = np.where(small, t, np.sin(t * th) / np.where(small, 1, s))
    r = wa * a + wb * b
    return V('vector4', r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-12))


@builtin('eulertoquaternion')
def _euler2q(I, args, mask, hint):
    e = _f(convert(args[0], 'vector'))
    n = len(e)
    mats = None
    for axis_i in range(3):
        ax = np.zeros((n, 3))
        ax[:, axis_i] = 1
        R = _axis_angle_matrix(e[:, axis_i], ax)
        mats = R if mats is None else np.matmul(mats, R)
    return V('vector4', _quat_from_matrix3(mats))


# ----------------------------------------------------------------------------
# Random / noise


def _result_dims(hint, default=1):
    if hint in VECTORS:
        return VECTORS[hint]
    return default


@builtin('rand', 'random')
def _rand(I, args, mask, hint):
    seeds = [full(convert(a, 'float') if a.t in SCALARS else a, _n(I, *args)) for a in args]
    seeds = [s.reshape(len(s), -1) for s in seeds]
    seed = np.concatenate(seeds, axis=1)
    dims = _result_dims(hint)
    d = _noise.rand(seed, dims)
    return V(VEC_BY_SIZE[dims], d)


def _noise_pos(I, args):
    a = args[0]
    n = _n(I, a)
    if a.t in SCALARS:
        p = np.zeros((n, 3))
        p[:, 0] = full(convert(a, 'float'), n)
    else:
        d = full(a, n)
        p = np.zeros((n, 3))
        k = min(3, d.shape[1])
        p[:, :k] = d[:, :k]
        if d.shape[1] == 4:
            p += d[:, 3:4] * 0.618
    return p


@builtin('noise', 'onoise', 'pnoise', 'anoise', 'xnoise')
def _noisefn(I, args, mask, hint):
    dims = _result_dims(hint)
    p = _noise_pos(I, args)
    return V(VEC_BY_SIZE[dims], _noise.noise(p, dims))


@builtin('snoise', 'sxnoise')
def _snoisefn(I, args, mask, hint):
    dims = _result_dims(hint)
    p = _noise_pos(I, args)
    return V(VEC_BY_SIZE[dims], _noise.snoise(p, dims))


@builtin('curlnoise')
def _curlnoise(I, args, mask, hint):
    return V('vector', _noise.curlnoise(_noise_pos(I, args)))


# ----------------------------------------------------------------------------
# Strings / output


@builtin('sprintf')
def _sprintf(I, args, mask, hint):
    fmt = args[0]
    n = _n(I, *args)
    out = np.empty(n, dtype=object)
    fd = full(fmt, n)
    datas = [full(a, n) for a in args[1:]]
    for i in range(n):
        out[i] = _vex_format(fd[i], [(a.t, d[i]) for a, d in zip(args[1:], datas)])
    return V('string', out)


def _vex_format(fmt, values):
    import re
    it = iter(values)

    def rep(m):
        spec = m.group(0)
        if spec == '%%':
            return '%'
        try:
            t, val = next(it)
        except StopIteration:
            return spec
        conv = spec[-1]
        if t in VECTORS or t in MATRICES or is_array(t):
            return _fmt_value(val, t if not is_array(t) else 'vector')
        if conv in 'dixXoc':
            return (spec[:-1] + 'd') % int(val) if conv in 'di' else spec % int(val)
        if conv in 'feEgG':
            return spec % float(val)
        if conv == 's':
            return _fmt_value(val, t)
        if conv == 'p':
            return _fmt_value(val, t)
        return spec % val
    return re.sub(r'%[-+ #0]*\d*(?:\.\d+)?[a-zA-Z%]', rep, fmt)


@builtin('printf', 'print')
def _printf(I, args, mask, hint):
    s = _sprintf(I, args, mask, hint)
    lanes = np.nonzero(mask)[0]
    d = full(s, I.n)
    for i in lanes[:200]:
        I.ctx.log(d[i])
    if len(lanes) > 200:
        I.ctx.log("... (%d more lines)" % (len(lanes) - 200))
    return None


@builtin('error', 'warning')
def _error(I, args, mask, hint):
    if mask.any():
        s = _sprintf(I, args, mask, hint)
        I.ctx.log("Error: " + full(s, I.n)[np.argmax(mask)])
    return None


@builtin('itoa')
def _itoa(I, args, mask, hint):
    return convert(convert(args[0], 'int'), 'string')


@builtin('atof')
def _atof(I, args, mask, hint):
    return convert(args[0], 'float')


@builtin('atoi')
def _atoi(I, args, mask, hint):
    return convert(convert(args[0], 'float'), 'int')


@builtin('strlen')
def _strlen(I, args, mask, hint):
    return V('int', np.array([len(s) for s in args[0].d], dtype=np.int64))


@builtin('concat')
def _concat(I, args, mask, hint):
    n = _n(I, *args)
    parts = [full(convert(a, 'string'), n) for a in args]
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = ''.join(p[i] for p in parts)
    return V('string', out)


@builtin('startswith')
def _startswith(I, args, mask, hint):
    n = _n(I, *args)
    a, b = full(args[0], n), full(args[1], n)
    return V('int', np.array([x.startswith(y) for x, y in zip(a, b)], dtype=np.int64))


@builtin('endswith')
def _endswith(I, args, mask, hint):
    n = _n(I, *args)
    a, b = full(args[0], n), full(args[1], n)
    return V('int', np.array([x.endswith(y) for x, y in zip(a, b)], dtype=np.int64))


@builtin('toupper')
def _upper(I, args, mask, hint):
    return V('string', np.array([s.upper() for s in args[0].d], dtype=object))


@builtin('tolower')
def _lower(I, args, mask, hint):
    return V('string', np.array([s.lower() for s in args[0].d], dtype=object))


# ----------------------------------------------------------------------------
# Arrays


def _array_lanes(I, v):
    return np.array(full(v, I.n), dtype=object, copy=True)


@builtin('len')
def _len(I, args, mask, hint):
    a = args[0]
    if is_array(a.t) or a.t == 'string':
        return V('int', np.array([len(x) for x in a.d], dtype=np.int64))
    if a.t in VECTORS:
        return const('int', VECTORS[a.t])
    if a.t in MATRICES:
        return const('int', MATRICES[a.t] ** 2)
    return const('int', 1)


@builtin('array')
def _array(I, args, mask, hint):
    from .interp import _make_array
    if hint and is_array(hint):
        et = elem_type(hint)
    elif args:
        et = args[0].t
    else:
        et = 'float'
    return _make_array(et, [convert(a, et) for a in args], I.n)


@builtin('append', 'push', inplace=True)
def _append(I, args, mask, hint):
    arr, val = args
    if not is_array(arr.t):
        raise VexError("append() needs an array")
    et = elem_type(arr.t)
    d = _array_lanes(I, arr)
    if is_array(val.t):
        vals = full(convert(val, arr.t), I.n)
        for i in np.nonzero(mask)[0]:
            d[i] = (list(d[i]) + list(vals[i])) if et == 'string' else np.concatenate([d[i], vals[i]])
    else:
        vals = full(convert(val, et), I.n)
        for i in np.nonzero(mask)[0]:
            if et == 'string':
                d[i] = list(d[i]) + [vals[i]]
            else:
                d[i] = np.concatenate([d[i], np.asarray(vals[i], dtype=d[i].dtype)[None]])
    return V(arr.t, d), None


@builtin('pop', inplace=True)
def _pop(I, args, mask, hint):
    arr = args[0]
    et = elem_type(arr.t)
    d = _array_lanes(I, arr)
    out = zeros(et, I.n).d
    for i in np.nonzero(mask)[0]:
        if len(d[i]):
            out[i] = d[i][-1]
            d[i] = d[i][:-1]
    return V(arr.t, d), V(et, out)


@builtin('removeindex', inplace=True)
def _removeindex(I, args, mask, hint):
    arr, idx = args
    d = _array_lanes(I, arr)
    ii = full(convert(idx, 'int'), I.n)
    for i in np.nonzero(mask)[0]:
        k = int(ii[i])
        if -len(d[i]) <= k < len(d[i]):
            if isinstance(d[i], list):
                d[i] = d[i][:k] + d[i][k + 1:] if k >= 0 else d[i][:k] + d[i][len(d[i]) + k + 1:]
            else:
                d[i] = np.delete(d[i], k, axis=0)
    return V(arr.t, d), None


@builtin('resize', inplace=True)
def _resize(I, args, mask, hint):
    arr, size = args
    et = elem_type(arr.t)
    d = _array_lanes(I, arr)
    ss = full(convert(size, 'int'), I.n)
    for i in np.nonzero(mask)[0]:
        k = max(0, int(ss[i]))
        cur = d[i]
        if et == 'string':
            d[i] = (list(cur) + [''] * k)[:k]
        else:
            if len(cur) >= k:
                d[i] = cur[:k]
            else:
                d[i] = np.concatenate([cur, np.zeros((k - len(cur),) + cur.shape[1:], dtype=cur.dtype)])
    return V(arr.t, d), None


@builtin('find')
def _find(I, args, mask, hint):
    arr, val = args
    n = I.n
    d = full(arr, n)
    if arr.t == 'string':
        vals = full(val, n)
        return V('int', np.array([s.find(v) for s, v in zip(d, vals)], dtype=np.int64))
    et = elem_type(arr.t)
    vals = full(convert(val, et), n)
    out = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        a = d[i]
        for j in range(len(a)):
            if np.all(np.asarray(a[j]) == vals[i]):
                out[i] = j
                break
    return V('int', out)


@builtin('sort')
def _sort(I, args, mask, hint):
    arr = args[0]
    d = _array_lanes(I, arr)
    for i in range(I.n):
        d[i] = sorted(d[i]) if isinstance(d[i], list) else np.sort(d[i], axis=0)
    return V(arr.t, d)


@builtin('argsort')
def _argsort(I, args, mask, hint):
    arr = args[0]
    d = full(arr, I.n)
    out = np.empty(I.n, dtype=object)
    for i in range(I.n):
        out[i] = np.argsort(np.asarray(d[i]), kind='stable').astype(np.int64)
    return V('int[]', out)


@builtin('reverse')
def _reverse(I, args, mask, hint):
    arr = args[0]
    if arr.t == 'string':
        return V('string', np.array([s[::-1] for s in arr.d], dtype=object))
    d = _array_lanes(I, arr)
    for i in range(I.n):
        d[i] = d[i][::-1]
    return V(arr.t, d)


@builtin('isvalidindex')
def _isvalidindex(I, args, mask, hint):
    arr, idx = args
    d = full(arr, I.n)
    ii = full(convert(idx, 'int'), I.n)
    return V('int', np.array([0 <= ii[i] < len(d[i]) for i in range(I.n)], dtype=np.int64))


# ----------------------------------------------------------------------------
# Channels (spare parameters)


@builtin('ch', 'chf', 'chi', 'chv', 'chs', 'chu', 'chp', 'ch3', 'ch4')
def _ch(I, args, mask, hint):
    name = _uniform_str(args[0])
    val = I.ctx.channel(name)
    if val is None:
        val = 0.0
    if isinstance(val, str):
        return const('string', val)
    arr = np.asarray(val, dtype=np.float64)
    if arr.ndim == 0:
        v = const('float', float(arr))
    else:
        v = V(VEC_BY_SIZE.get(arr.size, 'vector'), arr.reshape(1, -1))
    fname = I._current_call if hasattr(I, '_current_call') else None
    return v


def _make_ch_typed(t):
    def fn(I, args, mask, hint):
        return convert(_ch(I, args, mask, hint), t)
    return fn


FUNCTIONS['chf'] = _make_ch_typed('float')
FUNCTIONS['chi'] = _make_ch_typed('int')
FUNCTIONS['chv'] = _make_ch_typed('vector')
FUNCTIONS['ch3'] = _make_ch_typed('matrix3')
FUNCTIONS['ch4'] = _make_ch_typed('matrix')
FUNCTIONS['chp'] = _make_ch_typed('vector4')
FUNCTIONS['chu'] = _make_ch_typed('vector2')
FUNCTIONS['chs'] = _make_ch_typed('string')


@builtin('chramp')
def _chramp(I, args, mask, hint):
    name = _uniform_str(args[0])
    x = _f(convert(args[1], 'float'))
    return V('float', I.ctx.ramp(name, x))


# ----------------------------------------------------------------------------
# Geometry access


def _geo(I, v, write=False):
    if v.t == 'string':
        s = v.d[0]
        idx = int(s.split(':')[-1]) if s.startswith('opinput:') else 0
    else:
        idx = _uniform_int(v)
    return I.ctx.input(idx)


def _attr_read(I, geo, cls, name, idx, hint):
    from ..core.geometry import POINT, PRIM, DETAIL, VERTEX
    n = max(I.n, len(idx))
    idx = np.broadcast_to(idx, (n,))
    arr = I.ctx.read_attrib(geo, cls, name)
    if arr is None:
        t = hint if hint and hint in SHAPES else 'float'
        return zeros(t, n)
    t = type_of_array(arr)
    count = len(arr)
    if count == 0:
        return zeros(t, n)
    safe = np.clip(idx, 0, count - 1)
    d = arr[safe]
    if d.dtype != object:
        bad = (idx < 0) | (idx >= count)
        if bad.any():
            d = d.copy()
            d[bad] = 0
    return V(t, d)


@builtin('point')
def _point(I, args, mask, hint):
    from ..core.geometry import POINT
    geo = _geo(I, args[0])
    name = _uniform_str(args[1])
    idx = full(convert(args[2], 'int'), _n(I, args[2]))
    return _attr_read(I, geo, POINT, name, idx, hint)


@builtin('prim')
def _prim(I, args, mask, hint):
    from ..core.geometry import PRIM
    geo = _geo(I, args[0])
    name = _uniform_str(args[1])
    idx = full(convert(args[2], 'int'), _n(I, args[2]))
    if name == 'P' and geo is not None:
        c = geo.prim_centers()
        safe = np.clip(idx, 0, max(len(c) - 1, 0))
        return V('vector', c[safe] if len(c) else np.zeros((len(idx), 3)))
    return _attr_read(I, geo, PRIM, name, idx, hint)


@builtin('vertex')
def _vertex(I, args, mask, hint):
    from ..core.geometry import VERTEX
    geo = _geo(I, args[0])
    name = _uniform_str(args[1])
    idx = full(convert(args[-1], 'int'), _n(I, args[-1]))
    if len(args) == 4 and geo is not None:
        prim = full(convert(args[2], 'int'), _n(I, args[2]))
        idx = geo.prim_start[np.clip(prim, 0, geo.num_prims - 1)] + idx
    return _attr_read(I, geo, VERTEX, name, idx, hint)


@builtin('detail')
def _detail(I, args, mask, hint):
    from ..core.geometry import DETAIL
    geo = _geo(I, args[0])
    name = _uniform_str(args[1])
    return _attr_read(I, geo, DETAIL, name, np.zeros(1, dtype=np.int64), hint)


@builtin('pointattrib', 'primattrib')
def _pointattrib(I, args, mask, hint):
    return _point(I, args[:3], mask, hint)


def _has(cls):
    def fn(I, args, mask, hint):
        geo = _geo(I, args[0])
        name = _uniform_str(args[1])
        return const('int', int(geo is not None and I.ctx.read_attrib(geo, cls, name) is not None))
    return fn


FUNCTIONS['haspointattrib'] = _has('point')
FUNCTIONS['hasprimattrib'] = _has('prim')
FUNCTIONS['hasdetailattrib'] = _has('detail')
FUNCTIONS['hasvertexattrib'] = _has('vertex')


@builtin('npoints')
def _npoints(I, args, mask, hint):
    geo = _geo(I, args[0])
    return const('int', geo.num_points if geo is not None else 0)


@builtin('nprimitives', 'nprims')
def _nprims(I, args, mask, hint):
    geo = _geo(I, args[0])
    return const('int', geo.num_prims if geo is not None else 0)


@builtin('nvertices')
def _nvertices(I, args, mask, hint):
    geo = _geo(I, args[0])
    return const('int', geo.num_vertices if geo is not None else 0)


@builtin('primpoints')
def _primpoints(I, args, mask, hint):
    geo = _geo(I, args[0])
    prim = full(convert(args[1], 'int'), _n(I, args[1]))
    out = np.empty(len(prim), dtype=object)
    for i, p in enumerate(prim):
        out[i] = geo.prim_points(p).copy() if geo is not None and 0 <= p < geo.num_prims else np.zeros(0, np.int64)
    return V('int[]', out)


@builtin('primvertexcount')
def _primvertexcount(I, args, mask, hint):
    geo = _geo(I, args[0])
    prim = full(convert(args[1], 'int'), _n(I, args[1]))
    sizes = geo.prim_sizes()
    ok = (prim >= 0) & (prim < len(sizes))
    return V('int', np.where(ok, sizes[np.clip(prim, 0, max(len(sizes) - 1, 0))] if len(sizes) else 0, 0))


@builtin('primpoint')
def _primpoint(I, args, mask, hint):
    geo = _geo(I, args[0])
    n = _n(I, args[1], args[2])
    prim = full(convert(args[1], 'int'), n)
    k = full(convert(args[2], 'int'), n)
    if geo is None or geo.num_prims == 0:
        return const('int', -1)
    p = np.clip(prim, 0, geo.num_prims - 1)
    sizes = geo.prim_sizes()[p]
    ok = (prim >= 0) & (prim < geo.num_prims) & (k >= 0) & (k < sizes)
    v = geo.vtx_point[np.clip(geo.prim_start[p] + k, 0, max(geo.num_vertices - 1, 0))]
    return V('int', np.where(ok, v, -1))


def _point_prims_table(I, geo):
    key = ('pointprims', id(geo))
    cache = I.ctx.cache
    if key not in cache:
        table = [[] for _ in range(geo.num_points)]
        vp = geo.vertex_prim()
        for pt, pr in zip(geo.vtx_point.tolist(), vp.tolist()):
            if not table[pt] or table[pt][-1] != pr:
                table[pt].append(pr)
        cache[key] = table
    return cache[key]


def _neighbours_table(I, geo):
    key = ('neighbours', id(geo))
    cache = I.ctx.cache
    if key not in cache:
        table = [set() for _ in range(geo.num_points)]
        for a, b in geo.edges().tolist():
            table[a].add(b)
            table[b].add(a)
        cache[key] = [np.array(sorted(s), dtype=np.int64) for s in table]
    return cache[key]


@builtin('pointprims')
def _pointprims(I, args, mask, hint):
    geo = _geo(I, args[0])
    pts = full(convert(args[1], 'int'), _n(I, args[1]))
    table = _point_prims_table(I, geo)
    out = np.empty(len(pts), dtype=object)
    for i, p in enumerate(pts):
        out[i] = np.array(table[p], dtype=np.int64) if 0 <= p < len(table) else np.zeros(0, np.int64)
    return V('int[]', out)


@builtin('neighbours')
def _neighbours(I, args, mask, hint):
    geo = _geo(I, args[0])
    pts = full(convert(args[1], 'int'), _n(I, args[1]))
    table = _neighbours_table(I, geo)
    out = np.empty(len(pts), dtype=object)
    for i, p in enumerate(pts):
        out[i] = table[p] if 0 <= p < len(table) else np.zeros(0, np.int64)
    return V('int[]', out)


@builtin('neighbourcount')
def _neighbourcount(I, args, mask, hint):
    geo = _geo(I, args[0])
    pts = full(convert(args[1], 'int'), _n(I, args[1]))
    table = _neighbours_table(I, geo)
    return V('int', np.array([len(table[p]) if 0 <= p < len(table) else 0 for p in pts], dtype=np.int64))


@builtin('neighbour')
def _neighbour(I, args, mask, hint):
    geo = _geo(I, args[0])
    n = _n(I, args[1], args[2])
    pts = full(convert(args[1], 'int'), n)
    k = full(convert(args[2], 'int'), n)
    table = _neighbours_table(I, geo)
    out = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        p = pts[i]
        if 0 <= p < len(table) and 0 <= k[i] < len(table[p]):
            out[i] = table[p][k[i]]
    return V('int', out)


def _kdtree(I, P):
    key = ('kdtree', id(P), len(P))
    tree = I.ctx.cache.get(key)
    if tree is None:
        try:
            from mathutils.kdtree import KDTree
        except ImportError:
            return None
        tree = KDTree(len(P))
        for i, co in enumerate(P.tolist()):
            tree.insert(co, i)
        tree.balance()
        I.ctx.cache[key] = tree
    return tree


def _near_search(P, query, maxdist, maxpts, lanes, tree=None):
    """Nearest neighbour search. Returns list of sorted index arrays per lane."""
    results = [np.zeros(0, dtype=np.int64)] * len(query)
    if len(P) == 0:
        return results
    if tree is not None:
        q = query.tolist()
        for lane in np.asarray(lanes).tolist():
            md = float(maxdist[lane])
            k = int(maxpts[lane])
            if k <= 0:
                continue
            if md < 1e17:
                found = tree.find_range(q[lane], md)
                found.sort(key=lambda f: f[2])
                found = found[:k]
            else:
                found = tree.find_n(q[lane], k)
            results[lane] = np.array([f[1] for f in found], dtype=np.int64)
        return results
    chunk = max(1, int(4_000_000 // max(len(P), 1)))
    lanes = np.asarray(lanes)
    for s in range(0, len(lanes), chunk):
        ln = lanes[s:s + chunk]
        q = query[ln]
        d2 = np.sum((q[:, None, :] - P[None, :, :]) ** 2, axis=2)
        for row, lane in enumerate(ln):
            md = maxdist[lane]
            dr = d2[row]
            k = maxpts[lane]
            if k < len(dr):
                cand = np.argpartition(dr, k - 1)[:k] if k > 0 else np.zeros(0, np.int64)
            else:
                cand = np.arange(len(dr))
            cand = cand[dr[cand] <= md * md]
            cand = cand[np.argsort(dr[cand], kind='stable')]
            results[lane] = cand.astype(np.int64)
    return results


@builtin('nearpoints', 'pcfind')
def _nearpoints(I, args, mask, hint):
    geo = _geo(I, args[0])
    if args[1].t == 'string':
        args = [args[0]] + args[2:]
    n = I.n
    pos = full(convert(args[1], 'vector'), n)
    maxdist = full(convert(args[2], 'float'), n) if len(args) > 2 else np.full(n, 1e18)
    maxpts = full(convert(args[3], 'int'), n) if len(args) > 3 else np.full(n, 1 << 30)
    res = _near_search(geo.P, pos, maxdist, np.minimum(maxpts, geo.num_points), np.nonzero(mask)[0], _kdtree(I, geo.P))
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = res[i]
    return V('int[]', out)


@builtin('nearpoint')
def _nearpoint(I, args, mask, hint):
    geo = _geo(I, args[0])
    if args[1].t == 'string':
        args = [args[0]] + args[2:]
    n = I.n
    pos = full(convert(args[1], 'vector'), n)
    maxdist = full(convert(args[2], 'float'), n) if len(args) > 2 else np.full(n, 1e18)
    res = _near_search(geo.P, pos, maxdist, np.ones(n, dtype=np.int64), np.nonzero(mask)[0], _kdtree(I, geo.P))
    return V('int', np.array([r[0] if len(r) else -1 for r in res], dtype=np.int64))


@builtin('getbbox_min')
def _bbmin(I, args, mask, hint):
    return V('vector', _geo(I, args[0]).bbox()[0][None])


@builtin('getbbox_max')
def _bbmax(I, args, mask, hint):
    return V('vector', _geo(I, args[0]).bbox()[1][None])


@builtin('getbbox_center')
def _bbcenter(I, args, mask, hint):
    lo, hi = _geo(I, args[0]).bbox()
    return V('vector', ((lo + hi) * 0.5)[None])


@builtin('getbbox_size')
def _bbsize(I, args, mask, hint):
    lo, hi = _geo(I, args[0]).bbox()
    return V('vector', (hi - lo)[None])


@builtin('relbbox', 'relpointbbox')
def _relbbox(I, args, mask, hint):
    lo, hi = _geo(I, args[0]).bbox()
    p = _f(convert(args[1], 'vector'))
    size = np.where(hi - lo == 0, 1, hi - lo)
    return V('vector', (p - lo) / size)


@builtin('inpointgroup')
def _inpointgroup(I, args, mask, hint):
    geo = _geo(I, args[0])
    name = _uniform_str(args[1])
    pts = full(convert(args[2], 'int'), _n(I, args[2]))
    g = geo.groups['point'].get(name)
    if g is None:
        return const('int', 0)
    ok = (pts >= 0) & (pts < len(g))
    return V('int', (ok & g[np.clip(pts, 0, max(len(g) - 1, 0))]).astype(np.int64))


@builtin('inprimgroup')
def _inprimgroup(I, args, mask, hint):
    geo = _geo(I, args[0])
    name = _uniform_str(args[1])
    prims = full(convert(args[2], 'int'), _n(I, args[2]))
    g = geo.groups['prim'].get(name)
    if g is None:
        return const('int', 0)
    ok = (prims >= 0) & (prims < len(g))
    return V('int', (ok & g[np.clip(prims, 0, max(len(g) - 1, 0))]).astype(np.int64))


# ----------------------------------------------------------------------------
# Geometry creation / modification (deferred until the snippet finishes)


@builtin('addpoint')
def _addpoint(I, args, mask, hint):
    n = I.n
    src = args[1]
    lanes = np.nonzero(mask)[0]
    if src.t == 'int':
        pts = full(src, n)[lanes]
        idx = I.ctx.add_points(None, copy_from=pts)
    else:
        pos = full(convert(src, 'vector'), n)[lanes]
        idx = I.ctx.add_points(pos)
    out = np.full(n, -1, dtype=np.int64)
    out[lanes] = idx
    return V('int', out)


@builtin('addprim')
def _addprim(I, args, mask, hint):
    n = I.n
    ptype = _uniform_str(args[1])
    closed = ptype not in ('polyline', 'polyline_open')
    lanes = np.nonzero(mask)[0]
    pt_lists = [[] for _ in lanes]
    for a in args[2:]:
        if is_array(a.t):
            d = full(a, n)
            for j, lane in enumerate(lanes):
                pt_lists[j].extend(int(x) for x in d[lane])
        else:
            d = full(convert(a, 'int'), n)
            for j, lane in enumerate(lanes):
                pt_lists[j].append(int(d[lane]))
    idx = I.ctx.add_prims(pt_lists, closed)
    out = np.full(n, -1, dtype=np.int64)
    out[lanes] = idx
    return V('int', out)


@builtin('addvertex')
def _addvertex(I, args, mask, hint):
    n = I.n
    prim = full(convert(args[1], 'int'), n)
    pt = full(convert(args[2], 'int'), n)
    for lane in np.nonzero(mask)[0]:
        I.ctx.add_vertex(int(prim[lane]), int(pt[lane]))
    return const('int', 0)


@builtin('removepoint')
def _removepoint(I, args, mask, hint):
    pts = full(convert(args[1], 'int'), I.n)
    I.ctx.remove('point', pts[mask])
    return const('int', 1)


@builtin('removeprim')
def _removeprim(I, args, mask, hint):
    prims = full(convert(args[1], 'int'), I.n)
    andpoints = len(args) > 2 and bool(truthy(args[2]).any())
    I.ctx.remove('prim', prims[mask], andpoints)
    return const('int', 1)


def _set_attrib_fn(cls):
    def fn(I, args, mask, hint):
        n = I.n
        name = _uniform_str(args[1])
        if cls == 'detail':
            idx = np.zeros(n, dtype=np.int64)
            value = args[2]
            mode = args[3] if len(args) > 3 else None
        else:
            idx = full(convert(args[2], 'int'), n)
            value = args[3]
            mode = args[4] if len(args) > 4 else None
            if cls == 'vertex' and len(args) > 4 and args[4].t != 'string':
                # setvertexattrib(geo, name, prim, vtx, value)
                prim = idx
                vtx = full(convert(args[3], 'int'), n)
                value = args[4]
                idx = I.ctx.vertex_index(prim, vtx)
                mode = args[5] if len(args) > 5 else None
        mode = _uniform_str(mode) if mode is not None else 'set'
        I.ctx.queue_set(cls, name, idx[mask], value.t, full(value, n)[mask], mode)
        return const('int', 1)
    return fn


FUNCTIONS['setpointattrib'] = _set_attrib_fn('point')
FUNCTIONS['setprimattrib'] = _set_attrib_fn('prim')
FUNCTIONS['setvertexattrib'] = _set_attrib_fn('vertex')
FUNCTIONS['setdetailattrib'] = _set_attrib_fn('detail')


def _set_group_fn(cls):
    def fn(I, args, mask, hint):
        n = I.n
        name = _uniform_str(args[1])
        idx = full(convert(args[2], 'int'), n)
        val = full(convert(args[3], 'int'), n) if len(args) > 3 else np.ones(n, dtype=np.int64)
        I.ctx.queue_group(cls, name, idx[mask], val[mask] != 0)
        return const('int', 1)
    return fn


FUNCTIONS['setpointgroup'] = _set_group_fn('point')
FUNCTIONS['setprimgroup'] = _set_group_fn('prim')


@builtin('getpointgroup', 'expandpointgroup')
def _expandpointgroup(I, args, mask, hint):
    geo = _geo(I, args[0])
    name = _uniform_str(args[1])
    g = geo.groups['point'].get(name)
    idx = np.nonzero(g)[0] if g is not None else (np.arange(geo.num_points) if name in ('', '*') else np.zeros(0, np.int64))
    out = np.empty(I.n, dtype=object)
    for i in range(I.n):
        out[i] = idx
    return V('int[]', out)
