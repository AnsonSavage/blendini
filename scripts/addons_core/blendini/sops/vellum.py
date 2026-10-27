# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""A small position based dynamics cloth solver, in the spirit of Houdini's Vellum."""

import numpy as np

from .base import SOP, Parm, register_sop, parse_group_pattern
from ..core.geometry import Geometry, POINT


def _in(inputs, i=0):
    if len(inputs) > i and inputs[i] is not None:
        return inputs[i]
    return Geometry()


def _bend_pairs(geo):
    """Point pairs across shared edges (opposite vertices of adjacent triangles/quads)."""
    from collections import defaultdict
    edge_faces = defaultdict(list)
    for prim in range(geo.num_prims):
        pts = geo.prim_points(prim).tolist()
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            # The point "opposite" this edge inside the face.
            opp = pts[(i + n // 2 + 1) % n] if n > 3 else pts[(i + 2) % n]
            edge_faces[(min(a, b), max(a, b))].append(opp)
    pairs = [(f[0], f[1]) for f in edge_faces.values() if len(f) == 2 and f[0] != f[1]]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


class _Collider:
    def __init__(self, geo):
        self.tree = None
        if geo is None or geo.num_prims == 0:
            return
        tris, _ = geo.triangulate_flat()
        if len(tris) == 0:
            return
        try:
            from mathutils.bvhtree import BVHTree
        except ImportError:
            return
        self.tree = BVHTree.FromPolygons(geo.P.tolist(), tris.tolist(), all_triangles=True)
        self.lo, self.hi = geo.bbox()

    def push_out(self, P, thickness):
        if self.tree is None:
            return P
        out = P.copy()
        near = np.all((P >= self.lo - thickness * 2) & (P <= self.hi + thickness * 2), axis=1)
        idx = np.nonzero(near)[0]
        for i, p in zip(idx.tolist(), P[idx].tolist()):
            loc, nrm, _idx, dist = self.tree.find_nearest(p)
            if loc is None:
                continue
            d = np.asarray(p) - np.asarray(loc)
            n = np.asarray(nrm)
            side = float(np.dot(d, n))
            if side < thickness:
                out[i] = np.asarray(loc) + n * thickness
        return out


def pbd_step(state, dt, p, collider):
    P, V = state['P'], state['V']
    pinned = state['pinned']
    free = ~pinned
    gravity = np.array([0.0, -p.gravity, 0.0])
    V = V + gravity * dt
    V *= (1.0 - p.drag)
    X = P + V * dt
    X[pinned] = state['pin_pos']
    E, rest = state['edges'], state['rest']
    B, brest = state['bend'], state['bend_rest']
    n = len(X)
    for _ in range(max(int(p.iterations), 1)):
        for pairs, lengths, k in ((E, rest, p.stiffness), (B, brest, p.bendstiffness)):
            if len(pairs) == 0 or k <= 0:
                continue
            a, b = pairs[:, 0], pairs[:, 1]
            d = X[b] - X[a]
            ln = np.linalg.norm(d, axis=1)
            safe = np.where(ln == 0, 1, ln)
            wa = free[a].astype(np.float64)
            wb = free[b].astype(np.float64)
            wsum = np.where(wa + wb == 0, 1, wa + wb)
            corr = (d * ((ln - lengths) / safe / wsum * k)[:, None])
            acc = np.zeros((n, 3))
            cnt = np.zeros(n)
            for axis in range(3):
                acc[:, axis] += np.bincount(a, weights=corr[:, axis] * wa, minlength=n)
                acc[:, axis] -= np.bincount(b, weights=corr[:, axis] * wb, minlength=n)
            cnt += np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
            X += acc / np.maximum(cnt, 1)[:, None] * 1.5
        if p.ground:
            below = X[:, 1] < p.groundy + p.thickness
            X[below, 1] = p.groundy + p.thickness
        X[pinned] = state['pin_pos']
    # Collisions once per substep (substeps keep it stable).
    X = collider.push_out(X, p.thickness)
    X[pinned] = state['pin_pos']
    V = (X - P) / dt
    if p.ground:
        on_ground = X[:, 1] <= p.groundy + p.thickness + 1e-6
        V[on_ground] *= (1.0 - p.friction)
    state['P'], state['V'] = X, V


@register_sop
class VellumSOP(SOP):
    id = 'vellum'
    label = "Vellum Cloth (PBD)"
    category = 'Manipulate'
    inputs = 2
    input_labels = ("Cloth Geometry", "Collision Geometry")
    icon = 'MOD_CLOTH'
    time_dependent = True
    description = "Position based cloth simulation: stretch and bend constraints, pins, ground and colliders"
    parms = (
        Parm('pingroup', "Pin Points", 'string', default='', description="Points to pin, e.g. '0 9' or '@P.z<-4.9'"),
        Parm('startframe', "Start Frame", 'int', default=1),
        Parm('substeps', "Substeps", 'int', default=4, min=1, max=50),
        Parm('iterations', "Constraint Iterations", 'int', default=20, min=1, max=200),
        Parm('stiffness', "Stretch Stiffness", 'float', default=1.0, min=0, max=1),
        Parm('bendstiffness', "Bend Stiffness", 'float', default=0.05, min=0, max=1),
        Parm('gravity', "Gravity", 'float', default=9.81),
        Parm('drag', "Drag", 'float', default=0.002, min=0, max=1),
        Parm('ground', "Ground Plane", 'bool', default=True),
        Parm('groundy', "Ground Height", 'float', default=0.0),
        Parm('friction', "Ground Friction", 'float', default=0.2, min=0, max=1),
        Parm('thickness', "Thickness", 'float', default=0.02, min=0, soft_max=0.5),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs, 0)
        col_geo = _in(inputs, 1)
        state = ctx.state
        frame = int(round(ctx.frame))
        start = int(p.startframe)
        key = (tuple(sorted((k, v if not isinstance(v, list) else tuple(v)) for k, v in p.items())),
               geo.num_points, geo.num_prims, float(geo.P.sum()) if geo.num_points else 0.0,
               col_geo.num_points, float(col_geo.P.sum()) if col_geo.num_points else 0.0)
        if state.get('key') != key or frame <= start or 'sim' not in state:
            state.clear()
            state['key'] = key
            E = geo.edges()
            pinned = parse_group_pattern(geo, POINT, p.pingroup) if p.pingroup.strip() else np.zeros(geo.num_points, bool)
            B = _bend_pairs(geo)
            state['sim'] = {
                'P': geo.P.copy(), 'V': np.zeros_like(geo.P), 'pinned': pinned, 'pin_pos': geo.P[pinned].copy(),
                'edges': E, 'rest': np.linalg.norm(geo.P[E[:, 1]] - geo.P[E[:, 0]], axis=1) if len(E) else np.zeros(0),
                'bend': B, 'bend_rest': np.linalg.norm(geo.P[B[:, 1]] - geo.P[B[:, 0]], axis=1) if len(B) else np.zeros(0),
            }
            state['frames'] = {start: geo.P.copy()}
            state['last'] = start
        frames = state['frames']
        if frame <= start:
            return geo.copy()
        if frame in frames:
            out = geo.copy()
            out.P = frames[frame]
            return out
        if frame < state['last']:
            # Scrubbed backwards past the cache: restart.
            state.clear()
            return self.cook(inputs, p, ctx)
        collider = _Collider(col_geo)
        sim = state['sim']
        sub = max(int(p.substeps), 1)
        dt = 1.0 / ctx.fps / sub
        for f in range(state['last'] + 1, frame + 1):
            for _ in range(sub):
                pbd_step(sim, dt, p, collider)
            frames[f] = sim['P'].copy()
            state['last'] = f
        out = geo.copy()
        out.P = frames[frame]
        out.attribs[POINT]['v'] = sim['V'].copy()
        return out
