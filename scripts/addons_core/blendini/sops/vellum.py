# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""A small position based dynamics cloth solver, in the spirit of Houdini's Vellum."""

import numpy as np

from .base import SOP, Parm, register_sop, upstream_changed, parse_group_pattern
from ..core.geometry import Geometry, POINT, PRIM


def _in(inputs, i=0):
    if len(inputs) > i and inputs[i] is not None:
        return inputs[i]
    return Geometry()


def _bend_pairs(geo):
    """Point pairs across shared edges (opposite vertices of adjacent triangles/quads)."""
    from collections import defaultdict
    edge_faces = defaultdict(list)
    strand = []
    for prim in range(geo.num_prims):
        pts = geo.prim_points(prim).tolist()
        n = len(pts)
        if not geo.prim_closed[prim] or n < 3:
            # Curves (hair, strands): bend constraints between points two apart along the curve.
            strand.extend((pts[i], pts[i + 2]) for i in range(n - 2))
            continue
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            # The point "opposite" this edge inside the face.
            opp = pts[(i + n // 2 + 1) % n] if n > 3 else pts[(i + 2) % n]
            edge_faces[(min(a, b), max(a, b))].append(opp)
    pairs = [(f[0], f[1]) for f in edge_faces.values() if len(f) == 2 and f[0] != f[1]] + strand
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


def mesh_volume(X, tris):
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)


def _volume_constraint(X, tris, target, free, comp_tri=None, comp_pt=None):
    """PBD volume (balloon) constraint per connected piece: move points along the volume gradient
    toward each piece's target volume (target: array per piece, or a scalar for one piece)."""
    n = len(X)
    a, b, c = X[tris[:, 0]], X[tris[:, 1]], X[tris[:, 2]]
    triple = np.einsum('ij,ij->i', a, np.cross(b, c)) / 6.0
    if comp_tri is None:
        comp_tri = np.zeros(len(tris), dtype=np.int64)
        comp_pt = np.zeros(n, dtype=np.int64)
    k = int(comp_pt.max()) + 1 if n else 1
    vol = np.bincount(comp_tri, weights=triple, minlength=k)
    grad = np.zeros((n, 3))
    for idx, g in ((0, np.cross(b, c)), (1, np.cross(c, a)), (2, np.cross(a, b))):
        for axis in range(3):
            grad[:, axis] += np.bincount(tris[:, idx], weights=g[:, axis], minlength=n)
    grad /= 6.0
    grad[~free] = 0.0
    denom = np.bincount(comp_pt, weights=np.einsum('ij,ij->i', grad, grad), minlength=k)
    lam = np.where(denom > 1e-18, (np.broadcast_to(target, (k,)) - vol) / np.maximum(denom, 1e-18), 0.0)
    return X + grad * lam[comp_pt][:, None]


def _components(n, edges):
    """Connected component id per point (union-find over edges)."""
    parent = np.arange(n)

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root
    for a_, b_ in edges.tolist():
        ra, rb = find(a_), find(b_)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    _, comp = np.unique(roots, return_inverse=True)
    return comp


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
        if p.pressure > 0 and state.get('tris') is not None and len(state['tris']):
            X = _volume_constraint(X, state['tris'], state['rest_volume'] * p.pressure, free,
                                   state['comp_tri'], state['comp_pt'])
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


# ----------------------------------------------------------------------------
# Tearing: constraints are rebuilt from a vertex level topology, so points can be duplicated where
# the cloth separates. Torn state lives on half-edges (vertex -> next vertex of its primitive),
# which survive point splits.


def _tear_init(geo, sim):
    n_v = geo.num_vertices
    starts, sizes = geo.prim_start, geo.prim_sizes()
    prim = geo.vertex_prim()
    local = np.arange(n_v) - starts[:-1][prim]
    nxt = np.arange(n_v) + 1
    nz = sizes > 0
    nxt[(starts[1:] - 1)[nz]] = starts[:-1][nz]
    closed_v = geo.prim_closed[prim] & (sizes[prim] >= 3)
    valid = closed_v | (local < sizes[prim] - 1)
    n = sizes[prim]
    opp = starts[:-1][prim] + (local + np.where(n > 3, n // 2 + 1, 2)) % np.maximum(n, 1)
    vp = geo.vtx_point.copy()
    sim.update({
        'vp': vp, 'nxt': nxt, 'closed_v': closed_v, 'valid': valid, 'opp': opp, 'local': local,
        'vsize': sizes[prim], 'vtorn': np.zeros(n_v, dtype=bool),
        'vrest': np.linalg.norm(geo.P[vp[nxt]] - geo.P[vp], axis=1),
        'orig': np.arange(geo.num_points), 'P0': geo.P.copy(),
    })
    _tear_build(sim)


def _tear_build(sim):
    """Stretch and bend constraints from the current topology and torn half-edges."""
    vp, nxt, valid, closed_v, vtorn = sim['vp'], sim['nxt'], sim['valid'], sim['closed_v'], sim['vtorn']
    N = max(len(sim['P']), 1)
    hv = np.nonzero(valid)[0]
    a, b = vp[hv], vp[nxt[hv]]
    key = np.minimum(a, b) * N + np.maximum(a, b)
    if len(hv):
        uniq, first, inv = np.unique(key, return_index=True, return_inverse=True)
    else:
        uniq, first, inv = np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64)
    m = len(uniq)
    c_closed = np.bincount(inv, weights=closed_v[hv], minlength=m)
    c_open = np.bincount(inv, weights=~closed_v[hv], minlength=m)
    torn = np.bincount(inv, weights=vtorn[hv], minlength=m) > 0
    # Torn edges disappear, except boundaries left by a split (each side relaxes to its rest length).
    active = (~torn | ((c_closed == 1) & (c_open == 0))) & (uniq // N != uniq % N)
    eligible = ~torn & ((c_closed == 2) | ((c_open >= 1) & (c_closed == 0)))
    E = np.stack([uniq // N, uniq % N], axis=1)
    sim['edges'] = E[active]
    sim['rest'] = sim['vrest'][hv[first]][active]
    sim['tear_ok'] = eligible[active]
    sim['edge_mesh'] = ((c_closed == 2) & (c_open == 0))[active]
    sim['edge_keys'] = uniq[active]
    sim['h_key'], sim['h_v'] = key, hv
    # Bend pairs: opposite points of the two faces sharing an untorn edge.
    ok = closed_v[hv] & (c_closed[inv] == 2) & ~torn[inv]
    hk, hvv = key[ok], hv[ok]
    order = np.argsort(hk, kind='stable')
    hk, hvv = hk[order], hvv[order]
    pair = np.nonzero(hk[1:] == hk[:-1])[0] if len(hk) > 1 else np.zeros(0, np.int64)
    B = np.stack([vp[sim['opp'][hvv[pair]]], vp[sim['opp'][hvv[pair + 1]]]], axis=1) if len(pair) \
        else np.zeros((0, 2), np.int64)
    # Strands (open curves): points two apart, unless a segment between them is torn.
    sv = np.nonzero(~closed_v & (sim['local'] <= sim['vsize'] - 3))[0]
    sv = sv[~vtorn[sv] & ~vtorn[sv + 1]]
    B = np.concatenate([B, np.stack([vp[sv], vp[sv + 2]], axis=1)]) if len(sv) else B
    B = B[B[:, 0] != B[:, 1]]
    sim['bend'] = B
    P0, orig = sim['P0'], sim['orig']
    sim['bend_rest'] = np.linalg.norm(P0[orig[B[:, 1]]] - P0[orig[B[:, 0]]], axis=1) if len(B) else np.zeros(0)


def _tear_split(sim):
    """Duplicate points whose surrounding faces are no longer connected through untorn edges."""
    vp, nxt, valid, closed_v, vtorn = sim['vp'], sim['nxt'], sim['valid'], sim['closed_v'], sim['vtorn']
    n_v, N = len(vp), len(sim['P'])
    if n_v == 0:
        return False
    th = np.nonzero(vtorn & valid)[0]
    touched = np.zeros(N, dtype=bool)
    touched[vp[th]] = True
    touched[vp[nxt[th]]] = True
    first_v = np.full(N, n_v, dtype=np.int64)
    np.minimum.at(first_v, vp, np.arange(n_v))
    # Vertices that must stay together: at untouched points, on curves, or across untorn shared edges.
    stay = ~touched[vp] | ~closed_v
    U = [np.nonzero(stay)[0]]
    W = [first_v[vp[U[0]]]]
    hv = np.nonzero(valid & closed_v & ~vtorn)[0]
    a, b = vp[hv], vp[nxt[hv]]
    key = np.minimum(a, b) * N + np.maximum(a, b)
    order = np.argsort(key, kind='stable')
    k, h = key[order], hv[order]
    if len(k) > 1:
        pr = np.nonzero(k[1:] == k[:-1])[0]
        h1, h2 = h[pr], h[pr + 1]
        n1, n2 = nxt[h1], nxt[h2]
        same = vp[h1] == vp[h2]
        U += [h1, n1]
        W += [np.where(same, h2, n2), np.where(same, n2, h2)]
    u, w = np.concatenate(U), np.concatenate(W)
    lab = np.arange(n_v)
    for _ in range(n_v + 1):
        m = np.minimum(lab[u], lab[w])
        new = lab.copy()
        np.minimum.at(new, u, m)
        np.minimum.at(new, w, m)
        new = new[new]
        if np.array_equal(new, lab):
            break
        lab = new
    primary = lab[first_v[vp]]
    extra = lab != primary
    if not extra.any():
        return False
    labels, rank = np.unique(lab[extra], return_inverse=True)
    new_vp = vp.copy()
    new_vp[extra] = N + rank
    src = vp[labels]
    sim['vp'] = new_vp
    sim['P'] = np.concatenate([sim['P'], sim['P'][src]])
    sim['V'] = np.concatenate([sim['V'], sim['V'][src]])
    sim['pinned'] = np.concatenate([sim['pinned'], sim['pinned'][src]])
    sim['orig'] = np.concatenate([sim['orig'], sim['orig'][src]])
    sim['pin_pos'] = sim['P'][sim['pinned']].copy()
    # A torn balloon loses its pressure.
    sim['tris'] = None
    return True


def _crack_edges(sim, broken):
    """For each over-stretched mesh edge (a, b), the crack runs through a across the edges most
    perpendicular to ab (one on each side), so the faces on b's side separate from the others."""
    E, X = sim['edges'], sim['P']
    cand = sim['tear_ok'] & sim['edge_mesh']
    bi = np.nonzero(broken & sim['edge_mesh'])[0]
    if len(bi) == 0:
        return np.zeros(0, np.int64), bi
    a, b = E[bi, 0].copy(), E[bi, 1].copy()
    pin = sim['pinned']
    swap = pin[a] & ~pin[b]
    a[swap], b[swap] = b[swap], a[swap]
    # Incidence of candidate edges: (point, edge, other point), sorted by point.
    ce = np.nonzero(cand)[0]
    pts = np.concatenate([E[ce, 0], E[ce, 1]])
    oth = np.concatenate([E[ce, 1], E[ce, 0]])
    eid = np.concatenate([ce, ce])
    order = np.argsort(pts, kind='stable')
    pts, oth, eid = pts[order], oth[order], eid[order]
    lo = np.searchsorted(pts, a, 'left')
    cnt = np.searchsorted(pts, a, 'right') - lo
    k = np.repeat(np.arange(len(a)), cnt)
    rows = np.repeat(lo, cnt) + (np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    o, e = oth[rows], eid[rows]
    keep = o != b[k]
    k, o, e = k[keep], o[keep], e[keep]
    ab = X[b] - X[a]
    ab /= np.maximum(np.linalg.norm(ab, axis=1, keepdims=True), 1e-12)
    d = X[o] - X[a[k]]
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    cos = np.abs(np.einsum('ij,ij->i', d, ab[k]))
    chosen = []
    first_dir = np.zeros((len(a), 3))
    has_first = np.zeros(len(a), dtype=bool)
    for pass_ in range(2):
        if pass_ == 1:
            ok = has_first[k] & (np.einsum('ij,ij->i', d, first_dir[k]) < 0)
        else:
            ok = np.ones(len(k), dtype=bool)
        kk, cc, ee, dd = k[ok], cos[ok], e[ok], d[ok]
        if len(kk) == 0:
            break
        srt = np.lexsort((cc, kk))
        firsts = srt[np.concatenate([[True], kk[srt][1:] != kk[srt][:-1]])]
        chosen.append(ee[firsts])
        if pass_ == 0:
            first_dir[kk[firsts]] = dd[firsts]
            has_first[kk[firsts]] = True
    crack = np.unique(np.concatenate(chosen)) if chosen else np.zeros(0, np.int64)
    # Over-stretched edges with nowhere to crack (no candidate at a) just break themselves.
    lone = bi[~has_first]
    return crack, lone


def _tear(sim, threshold):
    """Break stretch constraints longer than rest * threshold. Returns True when the topology changed."""
    E = sim['edges']
    if len(E) == 0:
        return False
    X = sim['P']
    ln = np.linalg.norm(X[E[:, 1]] - X[E[:, 0]], axis=1)
    broken = sim['tear_ok'] & (ln > sim['rest'] * threshold)
    if not broken.any():
        return False
    crack, lone = _crack_edges(sim, broken)
    # Curve segments break directly; mesh edges open a crack across them.
    tear_idx = np.concatenate([np.nonzero(broken & ~sim['edge_mesh'])[0], crack, lone])
    vtorn = sim['vtorn'].copy()
    vtorn[sim['h_v'][np.isin(sim['h_key'], sim['edge_keys'][tear_idx])]] = True
    sim['vtorn'] = vtorn
    _tear_split(sim)
    _tear_build(sim)
    return True


def _tear_output(geo, data):
    """Input geometry with the torn topology: points duplicated along splits, torn curves cut."""
    vp, orig, vtorn = data['vp'], data['orig'], data['vtorn']
    out = geo.copy()
    for name, arr in geo.attribs[POINT].items():
        out.attribs[POINT][name] = arr[orig]
    for name, g in geo.groups[POINT].items():
        out.groups[POINT][name] = g[orig]
    out.P = data['P']
    out.attribs[POINT]['v'] = data['V'].copy()
    out.vtx_point = vp.copy()
    tp = np.zeros(len(orig), dtype=bool)
    tv = np.nonzero(vtorn)[0]
    if len(tv):
        nxt = np.arange(len(vp)) + 1
        sizes = out.prim_sizes()
        nz = sizes > 0
        nxt[(out.prim_start[1:] - 1)[nz]] = out.prim_start[:-1][nz]
        tp[vp[tv]] = True
        tp[vp[nxt[tv]]] = True
        # Open curves: cut at their torn segments.
        prim = out.vertex_prim()
        cut = tv[~out.prim_closed[prim[tv]] & (tv + 1 < out.prim_start[1:][prim[tv]])]
        if len(cut):
            starts = np.unique(np.concatenate([out.prim_start[:-1][sizes > 0], cut + 1]))
            src_prim = prim[starts]
            new_sizes = np.diff(np.concatenate([starts, [len(vp)]]))
            out.prim_start = np.concatenate([[0], np.cumsum(new_sizes)]).astype(np.int64)
            out.prim_closed = geo.prim_closed[src_prim]
            for name, arr in geo.attribs[PRIM].items():
                out.attribs[PRIM][name] = arr[src_prim]
            for name, g in geo.groups[PRIM].items():
                out.groups[PRIM][name] = g[src_prim]
    out.attribs[POINT]['torn'] = tp.astype(np.int64)
    out.groups[POINT]['torn'] = tp
    return out


@register_sop
class VellumSOP(SOP):
    id = 'vellum'
    label = "Vellum Cloth (PBD)"
    category = 'Simulation'
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
        Parm('pressure', "Pressure (Balloon)", 'float', default=0.0, min=0.0, soft_max=3.0,
             description="Closed meshes keep pressure x their rest volume (0 = off, 1 = keep volume, >1 inflate)"),
        Parm('tearing', "Tearing", 'bool', default=False,
             description="Stretch constraints break past the threshold and the mesh splits there (points 'torn')"),
        Parm('tearthreshold', "Tear Threshold", 'float', default=1.5, min=1.0, soft_max=4.0,
             hide_when={'tearing': False}, description="Stretch ratio (length / rest length) that breaks an edge"),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs, 0)
        col_geo = _in(inputs, 1)
        state = ctx.state
        frame = int(round(ctx.frame))
        start = int(p.startframe)
        key = (tuple(sorted((k, v if not isinstance(v, list) else tuple(v)) for k, v in p.items())),
               geo.num_points, geo.num_prims, col_geo.num_points, col_geo.num_prims)
        sig = (float(geo.P.sum()) if geo.num_points else 0.0, float(col_geo.P.sum()) if col_geo.num_points else 0.0)
        edited = state.get('key') == key and upstream_changed(state, frame, sig)
        if state.get('key') != key or frame <= start or 'sim' not in state or edited:
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
            tris, _ = geo.triangulate_flat()
            sim = state['sim']
            if len(tris) and p.pressure > 0:
                comp_pt = _components(geo.num_points, E) if len(E) else np.zeros(geo.num_points, dtype=np.int64)
                comp_tri = comp_pt[tris[:, 0]]
                k = int(comp_pt.max()) + 1
                A, Bv, C = geo.P[tris[:, 0]], geo.P[tris[:, 1]], geo.P[tris[:, 2]]
                vols = np.bincount(comp_tri, weights=np.einsum('ij,ij->i', A, np.cross(Bv, C)) / 6.0, minlength=k)
                # Pieces wound inside out: flip their triangles so every rest volume is positive.
                flip = vols[comp_tri] < 0
                tris = tris.copy()
                tris[flip] = tris[flip][:, ::-1]
                sim['tris'], sim['comp_tri'], sim['comp_pt'] = tris, comp_tri, comp_pt
                sim['rest_volume'] = np.abs(vols)
            else:
                sim['tris'] = None
            if p.tearing:
                _tear_init(geo, sim)
            state['frames'] = {start: geo.P.copy()}
            state['last'] = start
        frames = state['frames']
        if frame <= start:
            return geo.copy()
        if frame in frames:
            if isinstance(frames[frame], dict):
                return _tear_output(geo, frames[frame])
            out = geo.copy()
            out.P = frames[frame]
            return out
        if frame < state['last']:
            # Scrubbed backwards past the cache: restart.
            state.clear()
            return self.cook(inputs, p, ctx)
        collider = _Collider(col_geo)
        sim = state['sim']
        if p.tearing and sim['pinned'].any() and 'orig' in sim:
            sim['pin_pos'] = geo.P[sim['orig'][sim['pinned']]].copy()
        elif sim['pinned'].any() and geo.num_points == len(sim['P']):
            # Pins follow the (possibly animated) input.
            sim['pin_pos'] = geo.P[sim['pinned']].copy()
        sub = max(int(p.substeps), 1)
        dt = 1.0 / ctx.fps / sub
        for f in range(state['last'] + 1, frame + 1):
            for _ in range(sub):
                pbd_step(sim, dt, p, collider)
                if p.tearing:
                    _tear(sim, max(float(p.tearthreshold), 1.0))
            if p.tearing:
                frames[f] = {'P': sim['P'].copy(), 'V': sim['V'].copy(), 'vp': sim['vp'], 'orig': sim['orig'],
                             'vtorn': sim['vtorn']}
            else:
                frames[f] = sim['P'].copy()
            state['last'] = f
            if len(frames) > 500:
                del frames[min(fr for fr in frames if fr != start)]
        if p.tearing:
            return _tear_output(geo, frames[frame])
        out = geo.copy()
        out.P = frames[frame]
        out.attribs[POINT]['v'] = sim['V'].copy()
        return out
