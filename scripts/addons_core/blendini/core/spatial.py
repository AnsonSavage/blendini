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
        return loc, nrm, prim, dist
    tree = BVHTree.FromPolygons(geo.P.tolist(), tris.tolist(), all_triangles=True)
    for i, p in enumerate(np.asarray(points).tolist()):
        res = tree.find_nearest(p)
        if res[0] is not None:
            loc[i] = res[0]
            nrm[i] = res[1]
            prim[i] = tri_prim[res[2]]
            dist[i] = res[3]
    return loc, nrm, prim, dist
