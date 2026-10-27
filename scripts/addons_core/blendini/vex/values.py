# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
SIMD values for the VEX interpreter.

Every value holds the data for *all* lanes (elements being processed), with the lane
axis first. Constants may use a lane axis of length 1 and broadcast.
"""

import numpy as np

SHAPES = {
    'int': (), 'float': (), 'string': (),
    'vector2': (2,), 'vector': (3,), 'vector4': (4,),
    'matrix2': (2, 2), 'matrix3': (3, 3), 'matrix': (4, 4),
}
SCALARS = {'int', 'float'}
VECTORS = {'vector2': 2, 'vector': 3, 'vector4': 4}
MATRICES = {'matrix2': 2, 'matrix3': 3, 'matrix': 4}
VEC_BY_SIZE = {1: 'float', 2: 'vector2', 3: 'vector', 4: 'vector4'}


class VexError(Exception):
    pass


class V:
    """A typed lane-parallel value."""
    __slots__ = ('t', 'd')

    def __init__(self, t, d):
        self.t = t
        self.d = d

    @property
    def n(self):
        return self.d.shape[0]

    def __repr__(self):
        return "V(%s, %r)" % (self.t, self.d)


def is_array(t):
    return t.endswith('[]')


def elem_type(t):
    return t[:-2] if t.endswith('[]') else t


def dtype_for(t):
    if t == 'int':
        return np.int64
    if t == 'string' or is_array(t):
        return object
    return np.float64


def const(t, value, n=1):
    if t == 'string':
        d = np.empty(n, dtype=object)
        d[:] = value
        return V(t, d)
    shape = SHAPES[t]
    d = np.empty((n,) + shape, dtype=dtype_for(t))
    d[...] = value
    return V(t, d)


def zeros(t, n):
    if is_array(t):
        d = np.empty(n, dtype=object)
        et = elem_type(t)
        for i in range(n):
            d[i] = _empty_array(et)
        return V(t, d)
    if t == 'string':
        d = np.empty(n, dtype=object)
        d[:] = ''
        return V(t, d)
    if t in MATRICES:
        k = MATRICES[t]
        return V(t, np.broadcast_to(np.eye(k), (n, k, k)).copy())
    return V(t, np.zeros((n,) + SHAPES.get(t, ()), dtype=dtype_for(t)))


def _empty_array(et):
    if et == 'string':
        return []
    return np.zeros((0,) + SHAPES.get(et, ()), dtype=dtype_for(et))


def full(v, n):
    """Return data broadcast to n lanes (read-only view if broadcast)."""
    d = v.d
    if d.shape[0] == n:
        return d
    return np.broadcast_to(d, (n,) + d.shape[1:])


def lane_mask(mask, d):
    """Reshape a (N,) mask so it broadcasts against data d."""
    return mask.reshape(mask.shape + (1,) * (d.ndim - 1))


def mwhere(mask, new, old):
    """Masked select new where mask else old. Returns full width array."""
    if old is None:
        return new
    n = mask.shape[0]
    if mask.all():
        out = new
        if out.shape[0] != n:
            out = np.broadcast_to(out, (n,) + out.shape[1:])
        return np.array(out, copy=True) if not out.flags.writeable else out
    old_full = full(V(None, old), n)
    new_full = full(V(None, new), n)
    if old_full.dtype == object or new_full.dtype == object:
        out = np.array(old_full, dtype=object, copy=True)
        out[mask] = new_full[mask]
        return out
    return np.where(lane_mask(mask, old_full), new_full, old_full)


def type_of_array(arr):
    """VEX type matching a geometry attribute array."""
    if arr.dtype == object:
        if len(arr) and not isinstance(arr[0], str):
            return 'float[]'
        return 'string'
    shape = arr.shape[1:]
    if shape == ():
        return 'int' if arr.dtype.kind in 'iub' else 'float'
    if shape == (2,):
        return 'vector2'
    if shape == (3,):
        return 'vector'
    if shape == (4,):
        return 'vector4'
    if shape == (3, 3):
        return 'matrix3'
    if shape == (4, 4):
        return 'matrix'
    if shape == (2, 2):
        return 'matrix2'
    return 'float'


def convert(v, t):
    """Convert value v to type t following VEX's implicit conversions."""
    st = v.t
    if st == t:
        return v
    d = v.d
    if is_array(t):
        et = elem_type(t)
        if is_array(st):
            out = np.empty(d.shape[0], dtype=object)
            for i in range(d.shape[0]):
                out[i] = convert_elem_array(d[i], elem_type(st), et)
            return V(t, out)
        # scalar -> array of one element
        out = np.empty(d.shape[0], dtype=object)
        for i in range(d.shape[0]):
            out[i] = np.asarray([d[i]], dtype=dtype_for(et)) if et != 'string' else [d[i]]
        return V(t, out)
    if is_array(st):
        raise VexError("Can't convert %s to %s" % (st, t))
    if t == 'string':
        out = np.empty(d.shape[0], dtype=object)
        for i in range(d.shape[0]):
            out[i] = _fmt_value(d[i], st)
        return V(t, out)
    if st == 'string':
        nums = np.array([_atof(x) for x in d], dtype=np.float64)
        return convert(V('float', nums), t)
    if t == 'int':
        if st in SCALARS:
            return V(t, np.trunc(d).astype(np.int64) if st == 'float' else d.astype(np.int64))
        flat = d.reshape(d.shape[0], -1)
        return V(t, np.trunc(flat[:, 0]).astype(np.int64))
    if t == 'float':
        if st in SCALARS:
            return V(t, d.astype(np.float64))
        flat = d.reshape(d.shape[0], -1)
        return V(t, flat[:, 0].astype(np.float64))
    if t in VECTORS:
        k = VECTORS[t]
        if st in SCALARS:
            return V(t, np.repeat(d.astype(np.float64)[:, None], k, axis=1))
        if st in VECTORS:
            sk = VECTORS[st]
            if sk > k:
                return V(t, d[:, :k].copy())
            out = np.zeros((d.shape[0], k))
            out[:, :sk] = d
            if t == 'vector4' and st == 'vector':
                out[:, 3] = 0.0
            return V(t, out)
        if st in MATRICES:
            flat = d.reshape(d.shape[0], -1)
            return V(t, flat[:, :k].astype(np.float64))
    if t in MATRICES:
        k = MATRICES[t]
        if st in SCALARS:
            return V(t, np.eye(k)[None] * d.astype(np.float64)[:, None, None])
        if st in MATRICES:
            sk = MATRICES[st]
            if sk > k:
                return V(t, d[:, :k, :k].copy())
            out = np.broadcast_to(np.eye(k), (d.shape[0], k, k)).copy()
            out[:, :sk, :sk] = d
            return V(t, out)
        if st in VECTORS:
            sk = VECTORS[st]
            out = np.zeros((d.shape[0], k * k))
            out[:, :sk] = d
            return V(t, out.reshape(-1, k, k))
    raise VexError("Can't convert %s to %s" % (st, t))


def convert_elem_array(arr, st, et):
    if st == et:
        return arr
    if et == 'string':
        return [_fmt_value(x, st) for x in arr]
    if st == 'string':
        arr = np.array([_atof(x) for x in arr])
        st = 'float'
    n = len(arr)
    if n == 0:
        return _empty_array(et)
    return convert(V(st, np.asarray(arr)), et).d


def _atof(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0


def _fmt_value(x, t):
    if t == 'string':
        return x
    if t == 'int':
        return str(int(x))
    if t == 'float':
        return _fmt_float(x)
    if isinstance(x, np.ndarray):
        return '{' + ', '.join(_fmt_float(c) for c in x.ravel()) + '}'
    return str(x)


def _fmt_float(x):
    x = float(x)
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return '%g' % x


def truthy(v):
    d = v.d
    if is_array(v.t):
        return np.array([len(x) > 0 for x in d], dtype=bool)
    if v.t == 'string':
        return np.array([bool(x) for x in d], dtype=bool)
    if d.ndim == 1:
        return d != 0
    return np.any(d.reshape(d.shape[0], -1) != 0, axis=1)
