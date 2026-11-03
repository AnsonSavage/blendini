# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Parametric (u, v) coordinates on primitives, vectorized.

Triangles: P = P0 + u (P1 - P0) + v (P2 - P0).
Quads: bilinear, P = (1-u)(1-v) P0 + u(1-v) P1 + uv P2 + (1-u)v P3.
Other polygons fall back to the average of their points.
"""

import numpy as np


def _solve2(J, r):
    """Least squares solve of J (n,3,2) x = r (n,3) via the normal equations."""
    JtJ = np.einsum('nki,nkj->nij', J, J)
    Jtr = np.einsum('nki,nk->ni', J, r)
    det = JtJ[:, 0, 0] * JtJ[:, 1, 1] - JtJ[:, 0, 1] * JtJ[:, 1, 0]
    det = np.where(np.abs(det) < 1e-20, 1e-20, det)
    x = (JtJ[:, 1, 1] * Jtr[:, 0] - JtJ[:, 0, 1] * Jtr[:, 1]) / det
    y = (JtJ[:, 0, 0] * Jtr[:, 1] - JtJ[:, 1, 0] * Jtr[:, 0]) / det
    return np.stack([x, y], axis=1)


def _corners(geo, prims, k):
    start = geo.prim_start[prims]
    return [geo.P[geo.vtx_point[start + i]] for i in range(k)]


def prim_uv(geo, prims, P):
    """(n, 3) parametric uv of points P on primitives prims."""
    prims = np.asarray(prims, dtype=np.int64)
    P = np.asarray(P, dtype=np.float64)
    uv = np.zeros((len(prims), 3))
    if len(prims) == 0 or geo.num_prims == 0:
        return uv
    valid = (prims >= 0) & (prims < geo.num_prims)
    sizes = np.where(valid, geo.prim_sizes()[np.clip(prims, 0, geo.num_prims - 1)], 0)
    tri = np.nonzero(sizes == 3)[0]
    if len(tri):
        p0, p1, p2 = _corners(geo, prims[tri], 3)
        J = np.stack([p1 - p0, p2 - p0], axis=2)
        uv[tri, :2] = _solve2(J, P[tri] - p0)
    quad = np.nonzero(sizes == 4)[0]
    if len(quad):
        q0, q1, q2, q3 = _corners(geo, prims[quad], 4)
        x = P[quad]
        u = np.full(len(quad), 0.5)
        v = np.full(len(quad), 0.5)
        for _ in range(10):
            a, b = (1 - u)[:, None], (1 - v)[:, None]
            uu, vv = u[:, None], v[:, None]
            Pp = a * b * q0 + uu * b * q1 + uu * vv * q2 + a * vv * q3
            du = -b * q0 + b * q1 + vv * q2 - vv * q3
            dv = -a * q0 - uu * q1 + uu * q2 + a * q3
            step = _solve2(np.stack([du, dv], axis=2), x - Pp)
            u, v = u + step[:, 0], v + step[:, 1]
        uv[quad, 0], uv[quad, 1] = u, v
    return uv


def interpolate(geo, cls, values, prims, uv):
    """Evaluate a point or vertex attribute array at (prims, uv). Returns (n, ...)."""
    prims = np.asarray(prims, dtype=np.int64)
    n = len(prims)
    out = np.zeros((n,) + values.shape[1:])
    if n == 0 or geo.num_prims == 0:
        return out
    valid = (prims >= 0) & (prims < geo.num_prims)
    safe = np.clip(prims, 0, geo.num_prims - 1)
    sizes = np.where(valid, geo.prim_sizes()[safe], 0)
    start = geo.prim_start[safe]
    u, v = uv[:, 0], uv[:, 1]

    def val(idx_rows, k):
        vi = start[idx_rows] + k
        src = geo.vtx_point[vi] if cls == 'point' else vi
        return values[src].astype(np.float64)

    def bc(w, like):
        return w.reshape((-1,) + (1,) * (like.ndim - 1))
    tri = np.nonzero(sizes == 3)[0]
    if len(tri):
        a, b, c = (val(tri, k) for k in range(3))
        out[tri] = bc(1 - u[tri] - v[tri], a) * a + bc(u[tri], a) * b + bc(v[tri], a) * c
    quad = np.nonzero(sizes == 4)[0]
    if len(quad):
        a, b, c, d = (val(quad, k) for k in range(4))
        uq, vq = u[quad], v[quad]
        out[quad] = (bc((1 - uq) * (1 - vq), a) * a + bc(uq * (1 - vq), a) * b + bc(uq * vq, a) * c
                     + bc((1 - uq) * vq, a) * d)
    other = np.nonzero((sizes > 4) | ((sizes > 0) & (sizes < 3)))[0]
    for i in other:
        k = sizes[i]
        rows = np.full(k, i)
        vals = np.stack([val(rows[:1], j)[0] for j in range(k)])
        out[i] = vals.mean(axis=0)
    return out
