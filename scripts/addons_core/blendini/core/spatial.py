# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Spatial queries. Uses Blender's mathutils KD/BVH trees when available."""

import numpy as np


def nearest(src, queries, maxdist=np.inf):
    """Index of and distance to the nearest src point for each query (-1 when none)."""
    src = np.asarray(src, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    n = len(queries)
    idx = np.full(n, -1, dtype=np.int64)
    dist = np.full(n, np.inf)
    if len(src) == 0 or n == 0:
        return idx, dist
    try:
        from mathutils.kdtree import KDTree
    except ImportError:
        KDTree = None
    if KDTree is not None:
        tree = KDTree(len(src))
        for i, co in enumerate(src.tolist()):
            tree.insert(co, i)
        tree.balance()
        for i, q in enumerate(queries.tolist()):
            co, j, d = tree.find(q)
            if j is not None and d <= maxdist:
                idx[i] = j
                dist[i] = d
        return idx, dist
    chunk = max(1, 2_000_000 // len(src))
    for s in range(0, n, chunk):
        q = queries[s:s + chunk]
        d2 = np.sum((q[:, None, :] - src[None]) ** 2, axis=2)
        j = np.argmin(d2, axis=1)
        d = np.sqrt(d2[np.arange(len(q)), j])
        ok = d <= maxdist
        idx[s:s + chunk] = np.where(ok, j, -1)
        dist[s:s + chunk] = np.where(ok, d, np.inf)
    return idx, dist


def ray_cast(geo, origins, directions, maxdist=1e30):
    """Cast rays against geometry. Returns (hit (n,), location (n,3), normal (n,3), prim (n,), dist (n,))."""
    n = len(origins)
    hit = np.zeros(n, dtype=bool)
    loc = np.array(origins, dtype=np.float64, copy=True)
    nrm = np.zeros((n, 3))
    prim = np.full(n, -1, dtype=np.int64)
    dist = np.zeros(n)
    tris, tri_prim = geo.triangulate_flat()
    if len(tris) == 0:
        return hit, loc, nrm, prim, dist
    try:
        from mathutils.bvhtree import BVHTree
    except ImportError:
        return hit, loc, nrm, prim, dist
    tree = BVHTree.FromPolygons(geo.P.tolist(), tris.tolist(), all_triangles=True)
    for i, (o, d) in enumerate(zip(np.asarray(origins).tolist(), np.asarray(directions).tolist())):
        res = tree.ray_cast(o, d, maxdist)
        if res[0] is not None:
            hit[i] = True
            loc[i] = res[0]
            nrm[i] = res[1]
            prim[i] = tri_prim[res[2]]
            dist[i] = res[3]
    return hit, loc, nrm, prim, dist


def closest(geo, points):
    """Closest surface location for each point. Returns (location, normal, prim, dist)."""
    n = len(points)
    loc = np.array(points, dtype=np.float64, copy=True)
    nrm = np.zeros((n, 3))
    prim = np.full(n, -1, dtype=np.int64)
    dist = np.zeros(n)
    tris, tri_prim = geo.triangulate_flat()
    if len(tris) == 0:
        return loc, nrm, prim, dist
    try:
        from mathutils.bvhtree import BVHTree
    except ImportError:
        return closest_numpy(geo.P, tris, tri_prim, points)
    tree = BVHTree.FromPolygons(geo.P.tolist(), tris.tolist(), all_triangles=True)
    for i, p in enumerate(np.asarray(points).tolist()):
        res = tree.find_nearest(p)
        if res[0] is not None:
            loc[i] = res[0]
            nrm[i] = res[1]
            prim[i] = tri_prim[res[2]]
            dist[i] = res[3]
    return loc, nrm, prim, dist


def closest_point_triangles(p, a, b, c):
    """Closest points on triangles (a, b, c) to points p, all (n, 3) (Ericson, Real-Time Collision Detection)."""
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum('ij,ij->i', ab, ap), np.einsum('ij,ij->i', ac, ap)
    bp = p - b
    d3, d4 = np.einsum('ij,ij->i', ab, bp), np.einsum('ij,ij->i', ac, bp)
    cp = p - c
    d5, d6 = np.einsum('ij,ij->i', ab, cp), np.einsum('ij,ij->i', ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    denom = va + vb + vc
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    v = vb / denom
    w = vc / denom
    out = a + ab * v[:, None] + ac * w[:, None]
    # Edge regions.
    sel = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3)
    out = np.where(sel[:, None], a + ab * t[:, None], out)
    sel = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6)
    out = np.where(sel[:, None], a + ac * t[:, None], out)
    sel = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    den = (d4 - d3) + (d5 - d6)
    t = (d4 - d3) / np.where(np.abs(den) < 1e-30, 1e-30, den)
    out = np.where(sel[:, None], b + (c - b) * t[:, None], out)
    # Vertex regions.
    out = np.where(((d6 >= 0) & (d5 <= d6))[:, None], c, out)
    out = np.where(((d3 >= 0) & (d4 <= d3))[:, None], b, out)
    out = np.where(((d1 <= 0) & (d2 <= 0))[:, None], a, out)
    return out


def closest_numpy(P, tris, tri_prim, points):
    """Brute force closest surface point (used when mathutils is not available)."""
    points = np.asarray(points, dtype=np.float64)
    n, m = len(points), len(tris)
    loc = points.copy()
    nrm = np.zeros((n, 3))
    prim = np.full(n, -1, dtype=np.int64)
    dist = np.zeros(n)
    A, B, C = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    fn = np.cross(B - A, C - A)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)
    chunk = max(1, 1_000_000 // max(m, 1))
    for s in range(0, n, chunk):
        q = points[s:s + chunk]
        k = len(q)
        qq = np.repeat(q, m, axis=0)
        cp = closest_point_triangles(qq, np.tile(A, (k, 1)), np.tile(B, (k, 1)), np.tile(C, (k, 1)))
        d = np.linalg.norm(cp - qq, axis=1).reshape(k, m)
        j = np.argmin(d, axis=1)
        rows = np.arange(k)
        loc[s:s + chunk] = cp.reshape(k, m, 3)[rows, j]
        dist[s:s + chunk] = d[rows, j]
        nrm[s:s + chunk] = fn[j]
        prim[s:s + chunk] = tri_prim[j]
    return loc, nrm, prim, dist
