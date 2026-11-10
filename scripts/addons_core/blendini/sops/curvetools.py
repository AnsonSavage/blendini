# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Curve and polyline SOPs: Join, PolyCut, Orientation Along Curve, Fillet, Convert Curve
(NURBS / Bezier / Catmull-Rom evaluation), PolyExpand 2D, Curve Attributes, Spiral and Arc."""

from collections import defaultdict
from math import comb

import numpy as np

from .base import SOP, Parm, register_sop, parse_group_pattern, CookError
from ..core.geometry import Geometry, POINT, PRIM, DETAIL, merge
from .modify import _in
from .generators import ORIENT_ITEMS, AXIS_ITEMS, orient_plane, orient_axis, catmull_rom

EPS = 1e-12


# ----------------------------------------------------------------------------
# Shared helpers


def _normalize(v, fallback=None):
    ln = np.linalg.norm(v, axis=-1, keepdims=True)
    out = v / np.where(ln < EPS, 1.0, ln)
    if fallback is not None:
        out = np.where(ln < EPS, fallback, out)
    return out


def _vertex_topology(geo):
    """Per vertex: (prim, local index, prim size, previous vertex, next vertex, closed).

    Previous / next wrap around on closed primitives and clamp to the vertex itself at the ends of
    open ones."""
    nv = geo.num_vertices
    sizes = geo.prim_sizes()
    starts = geo.prim_start[:-1]
    vp = geo.vertex_prim()
    local = np.arange(nv) - starts[vp]
    size = sizes[vp]
    closed = geo.prim_closed[vp]
    idx = np.arange(nv)
    prv = np.where(local == 0, np.where(closed, idx + size - 1, idx), idx - 1)
    nxt = np.where(local == size - 1, np.where(closed, starts[vp], idx), idx + 1)
    return vp, local, size, prv, nxt, closed


def _segmented_cumsum(values, vp, starts):
    """Inclusive cumulative sum restarting at each primitive (vertices of a prim are contiguous)."""
    if len(values) == 0:
        return values.copy()
    c = np.cumsum(values)
    before = np.concatenate([[0.0], c])[starts]
    return c - before[vp]


def _curves_geo(src, point_attribs, sizes, closed, prim_src, point_groups=None):
    """Build polyline / polygon geometry whose primitives use consecutive new points."""
    out = Geometry()
    out.attribs[POINT] = point_attribs
    if point_groups:
        out.groups[POINT] = point_groups
    sizes = np.asarray(sizes, dtype=np.int64)
    keep = sizes > 0
    sizes = sizes[keep]
    prim_src = np.asarray(prim_src, dtype=np.int64)[keep]
    closed = np.asarray(closed, dtype=bool)[keep]
    if len(sizes):
        out.add_prims_flat(sizes, np.arange(int(sizes.sum())), closed)
        for name, arr in src.attribs[PRIM].items():
            out.attribs[PRIM][name] = arr[prim_src]
        for name, g in src.groups[PRIM].items():
            out.groups[PRIM][name] = g[prim_src]
    out.attribs[DETAIL] = {k: v.copy() for k, v in src.attribs[DETAIL].items()}
    return out


def _sampled_curves(src, A, sizes, closed, prim_src, P=None, B=None, t=None):
    """Curves made from source points A (optionally interpolated towards B by t)."""
    A = np.asarray(A, dtype=np.int64)
    attribs = {}
    groups = {}
    if B is not None:
        B = np.asarray(B, dtype=np.int64)
        t = np.asarray(t, dtype=np.float64)
        near_b = t >= 0.5
    for name, arr in src.attribs[POINT].items():
        if B is None or arr.dtype == object:
            attribs[name] = arr[A]
        elif arr.dtype.kind == 'f':
            tt = t.reshape((-1,) + (1,) * (arr.ndim - 1))
            attribs[name] = arr[A] * (1 - tt) + arr[B] * tt
        else:
            attribs[name] = np.where(near_b.reshape((-1,) + (1,) * (arr.ndim - 1)), arr[B], arr[A])
    for name, g in src.groups[POINT].items():
        groups[name] = g[A] if B is None else np.where(near_b, g[B], g[A])
    if P is not None:
        attribs['P'] = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    return _curves_geo(src, attribs, sizes, closed, prim_src, groups)


def quat_from_matrices(m):
    """Unit quaternions (x, y, z, w) from (n, 3, 3) column-vector rotation matrices."""
    m = np.asarray(m, dtype=np.float64)
    n = len(m)
    q = np.zeros((n, 4))
    if n == 0:
        return q
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    tr = m00 + m11 + m22
    cand = np.stack([tr, m00, m11, m22], axis=1)
    case = np.argmax(cand, axis=1)
    # w largest
    s = np.sqrt(np.maximum(1.0 + tr, EPS)) * 2
    qa = np.stack([(m[:, 2, 1] - m[:, 1, 2]) / s, (m[:, 0, 2] - m[:, 2, 0]) / s,
                   (m[:, 1, 0] - m[:, 0, 1]) / s, 0.25 * s], axis=1)
    s = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, EPS)) * 2
    qb = np.stack([0.25 * s, (m[:, 0, 1] + m[:, 1, 0]) / s, (m[:, 0, 2] + m[:, 2, 0]) / s,
                   (m[:, 2, 1] - m[:, 1, 2]) / s], axis=1)
    s = np.sqrt(np.maximum(1.0 - m00 + m11 - m22, EPS)) * 2
    qc = np.stack([(m[:, 0, 1] + m[:, 1, 0]) / s, 0.25 * s, (m[:, 1, 2] + m[:, 2, 1]) / s,
                   (m[:, 0, 2] - m[:, 2, 0]) / s], axis=1)
    s = np.sqrt(np.maximum(1.0 - m00 - m11 + m22, EPS)) * 2
    qd = np.stack([(m[:, 0, 2] + m[:, 2, 0]) / s, (m[:, 1, 2] + m[:, 2, 1]) / s, 0.25 * s,
                   (m[:, 1, 0] - m[:, 0, 1]) / s], axis=1)
    q = np.choose(case[:, None], [qa, qb, qc, qd])
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), EPS)
    # Keep w positive for stable output.
    q[q[:, 3] < 0] *= -1
    return q


def _cluster_points(P, tol):
    """Cluster id (index of a representative) for every point, merging points within tol."""
    n = len(P)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if tol <= 0:
        _, first, inv = np.unique(P, axis=0, return_index=True, return_inverse=True)
        return first[inv.ravel()]
    ids = np.empty(n, dtype=np.int64)
    chunk = max(1, 4_000_000 // n)
    for s in range(0, n, chunk):
        d = np.linalg.norm(P[s:s + chunk, None, :] - P[None], axis=2)
        ids[s:s + chunk] = np.argmax(d <= tol, axis=1)
    for _ in range(8):
        nxt = ids[ids]
        if np.array_equal(nxt, ids):
            break
        ids = nxt
    return ids


# ----------------------------------------------------------------------------
# Curve evaluation (pure numpy)


def bspline_basis(knots, degree, u, span):
    """Non-zero B-spline basis functions (Cox-de Boor, Piegl & Tiller A2.2), vectorized over u.

    Returns an (m, degree + 1) array: column j weights control point span - degree + j."""
    p = degree
    m = len(u)
    N = np.zeros((m, p + 1))
    N[:, 0] = 1.0
    left = np.zeros((m, p + 1))
    right = np.zeros((m, p + 1))
    for j in range(1, p + 1):
        left[:, j] = u - knots[span + 1 - j]
        right[:, j] = knots[span + j] - u
        saved = np.zeros(m)
        for r in range(j):
            den = right[:, r + 1] + left[:, j - r]
            temp = np.where(den != 0, N[:, r] / np.where(den == 0, 1.0, den), 0.0)
            N[:, r] = saved + right[:, r + 1] * temp
            saved = left[:, j - r] * temp
        N[:, j] = saved
    return N


def nurbs_curve(C, degree=3, closed=False, knots='clamped', weights=None, divisions=8):
    """Evaluate a NURBS curve with uniform knots at `divisions` samples per knot span.

    C: (n, k) control values (positions and any other attributes, interpolated alike).
    knots: 'clamped' (open, passes through the end points) or 'uniform' (open, unclamped).
    closed: periodic curve (the first `degree` control points are wrapped around).
    weights: optional (n,) rational weights.
    """
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    n = len(C)
    if n < 2:
        return C.copy()
    divisions = max(int(divisions), 1)
    w = np.ones(n) if weights is None else np.maximum(np.asarray(weights, dtype=np.float64).reshape(n), 1e-9)
    if closed and n >= 3:
        p = max(1, min(int(degree), n - 1))
        Cc = np.concatenate([C, C[:p]])
        wc = np.concatenate([w, w[:p]])
        count = n + p
        U = np.arange(count + p + 1, dtype=np.float64)
        samples = n * divisions
        u = p + n * np.arange(samples) / samples
    else:
        p = max(1, min(int(degree), n - 1))
        Cc, wc, count = C, w, n
        if knots == 'clamped':
            U = np.concatenate([np.zeros(p + 1), np.arange(1, n - p) / (n - p), np.ones(p + 1)])
            lo, hi = 0.0, 1.0
        else:
            U = np.arange(n + p + 1, dtype=np.float64)
            lo, hi = float(p), float(n)
        u = np.linspace(lo, hi, (n - p) * divisions + 1)
    span = np.clip(np.searchsorted(U, u, side='right') - 1, p, count - 1)
    Nb = bspline_basis(U, p, u, span)
    idx = span[:, None] - p + np.arange(p + 1)
    Wb = Nb * wc[idx]
    den = Wb.sum(axis=1, keepdims=True)
    return np.einsum('mj,mjk->mk', Wb, Cc[idx]) / np.where(den == 0, 1.0, den)


def bezier_curve(C, degree=3, closed=False, divisions=8):
    """Piecewise Bezier: every `degree`-th control point lies on the curve (a shorter last
    segment uses a lower degree)."""
    C = np.asarray(C, dtype=np.float64)
    if C.ndim == 1:
        C = C[:, None]
    if closed and len(C) >= 3:
        C = np.concatenate([C, C[:1]])
    n = len(C)
    if n < 2:
        return C.copy()
    divisions = max(int(divisions), 1)
    d = max(1, min(int(degree), n - 1))
    t = np.arange(divisions) / divisions

    def bernstein(dd):
        j = np.arange(dd + 1)
        coef = np.array([comb(dd, k) for k in j], dtype=np.float64)
        return coef * t[:, None] ** j * (1 - t[:, None]) ** (dd - j)

    parts = []
    nfull = (n - 1) // d
    if nfull:
        idx = np.arange(nfull)[:, None] * d + np.arange(d + 1)
        parts.append(np.einsum('tj,sjk->stk', bernstein(d), C[idx]).reshape(-1, C.shape[1]))
    rest = n - 1 - nfull * d
    if rest:
        parts.append(bernstein(rest) @ C[nfull * d:])
    if not (closed and len(C) >= 4):
        parts.append(C[-1:])
    return np.concatenate(parts)


def _stack_point_attribs(geo):
    """(names, shapes, dtypes, (n, k) float matrix) for numeric point attributes, P first."""
    names, shapes, dtypes, cols = [], [], [], []
    n = geo.num_points
    for name, arr in sorted(geo.attribs[POINT].items(), key=lambda kv: kv[0] != 'P'):
        if arr.dtype == object or arr.dtype.kind not in 'biuf':
            continue
        names.append(name)
        shapes.append(arr.shape[1:])
        dtypes.append(arr.dtype)
        cols.append(arr.reshape(n, -1).astype(np.float64))
    return names, shapes, dtypes, np.concatenate(cols, axis=1) if cols else np.zeros((n, 0))


def _unstack(names, shapes, dtypes, M):
    out = {}
    col = 0
    for name, shape, dtype in zip(names, shapes, dtypes):
        w = int(np.prod(shape)) if shape else 1
        vals = M[:, col:col + w].reshape((len(M),) + tuple(shape))
        col += w
        if dtype.kind == 'b':
            vals = vals > 0.5
        elif dtype.kind in 'iu':
            vals = np.rint(vals).astype(dtype)
        out[name] = vals
    return out


# ----------------------------------------------------------------------------
# SOPs


@register_sop
class JoinSOP(SOP):
    id = 'join'
    label = "Join"
    category = 'Polygon'
    inputs = 1
    icon = 'AUTOMERGE_ON'
    description = "Connect consecutive open curves into single curves (or only those whose ends touch)"
    parms = (
        Parm('group', "Group", 'string', description="Primitives to join (open curves only)"),
        Parm('onlyconnected', "Only Connected", 'bool', default=False,
             description="Join curves whose end points touch (any order or direction) instead of in sequence"),
        Parm('joincount', "Join Count", 'int', default=0, min=0, soft_max=10, hide_when={'onlyconnected': True},
             description="Join every N consecutive curves (0 joins all of them)"),
        Parm('closestends', "Connect Closest Ends", 'bool', default=True, hide_when={'onlyconnected': True},
             description="Reverse curves so the nearest end points are connected"),
        Parm('fuse', "Fuse Touching Ends", 'bool', default=True,
             description="Use one point where the joined ends are within the tolerance"),
        Parm('tolerance', "Tolerance", 'float', default=0.001, min=0.0, soft_max=0.1),
        Parm('wrap', "Wrap (Close Result)", 'bool', default=False, hide_when={'onlyconnected': True}),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        out = geo.copy()
        if geo.num_prims == 0:
            return out
        sizes = geo.prim_sizes()
        sel = parse_group_pattern(geo, PRIM, p.group) & ~geo.prim_closed & (sizes >= 1)
        ids = np.nonzero(sel)[0]
        if len(ids) == 0:
            return out
        tol = max(float(p.tolerance), 0.0)
        P = geo.P
        curves = [geo.prim_points(i) for i in ids]
        if p.onlyconnected:
            groups = self._connected(curves, P, tol)
        else:
            groups = self._sequential(curves, P, p)
        new_lists, new_closed, firsts = [], [], []
        for seq, closed in groups:
            parts = []
            for k, (c, rev) in enumerate(seq):
                pts = curves[c][::-1] if rev else curves[c]
                if parts and p.fuse and len(pts) and np.linalg.norm(P[parts[-1][-1]] - P[pts[0]]) <= tol:
                    pts = pts[1:]
                if len(pts):
                    parts.append(pts)
            chain = np.concatenate(parts)
            if closed and p.fuse and len(chain) > 2 and np.linalg.norm(P[chain[0]] - P[chain[-1]]) <= tol:
                chain = chain[:-1]
            new_lists.append(chain)
            new_closed.append(bool(closed) and len(chain) >= 3)
            firsts.append(ids[seq[0][0]])
        used_before = np.zeros(geo.num_points, dtype=bool)
        used_before[geo.vtx_point] = True
        out._delete_prims_only(sel)
        first_new = out.num_prims
        out.add_prims_flat([len(c) for c in new_lists], np.concatenate(new_lists), np.array(new_closed, dtype=bool))
        new = np.arange(first_new, out.num_prims)
        firsts = np.array(firsts, dtype=np.int64)
        for name, arr in geo.attribs[PRIM].items():
            out.attribs[PRIM][name][new] = arr[firsts]
        for name, g in geo.groups[PRIM].items():
            out.groups[PRIM][name][new] = g[firsts]
        used_after = np.zeros(out.num_points, dtype=bool)
        used_after[out.vtx_point] = True
        out.delete_points(used_before & ~used_after)
        return out

    @staticmethod
    def _sequential(curves, P, p):
        n = len(curves)
        count = p.joincount if p.joincount > 0 else n
        groups = []
        for s in range(0, n, count):
            members = list(range(s, min(s + count, n)))
            seq = [[members[0], False]]
            for c in members[1:]:
                rev = False
                if p.closestends:
                    prev_c, prev_rev = seq[-1]
                    pc = curves[prev_c]
                    end = P[pc[0] if prev_rev else pc[-1]]
                    cc = curves[c]
                    d_start = np.linalg.norm(end - P[cc[0]])
                    d_end = np.linalg.norm(end - P[cc[-1]])
                    if len(seq) == 1:
                        start = P[pc[-1] if prev_rev else pc[0]]
                        # The first curve may be flipped too.
                        opts = [(d_start, False, False), (d_end, False, True),
                                (np.linalg.norm(start - P[cc[0]]), True, False),
                                (np.linalg.norm(start - P[cc[-1]]), True, True)]
                        _, flip_first, rev = min(opts, key=lambda o: o[0])
                        if flip_first:
                            seq[0][1] = not seq[0][1]
                    else:
                        rev = d_end < d_start
                seq.append([c, rev])
            groups.append(([tuple(x) for x in seq], bool(p.wrap)))
        return groups

    @staticmethod
    def _connected(curves, P, tol):
        ends = np.array([[c[0], c[-1]] for c in curves], dtype=np.int64)
        cid = _cluster_points(P[ends.ravel()], tol).reshape(-1, 2)
        ns, ne = cid[:, 0].tolist(), cid[:, 1].tolist()
        adj = defaultdict(list)
        for c, (a, b) in enumerate(zip(ns, ne)):
            adj[a].append((c, 0))
            adj[b].append((c, 1))
        used = [False] * len(curves)

        def walk(c, rev):
            seq = [(c, rev)]
            used[c] = True
            node = ns[c] if rev else ne[c]
            while len(adj[node]) == 2:
                nxt = [(cc, e) for cc, e in adj[node] if not used[cc]]
                if not nxt:
                    break
                cc, e = nxt[0]
                used[cc] = True
                r = e == 1
                seq.append((cc, r))
                node = ns[cc] if r else ne[cc]
            return seq, node

        groups = []
        for node, links in list(adj.items()):
            if len(links) != 2:
                for c, e in links:
                    if not used[c]:
                        seq, _ = walk(c, e == 1)
                        groups.append((seq, False))
        for c in range(len(curves)):
            if not used[c]:
                seq, end = walk(c, False)
                groups.append((seq, end == ns[c]))
        return groups


@register_sop
class PolyCutSOP(SOP):
    id = 'polycut'
    label = "PolyCut"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_EDGESPLIT'
    description = "Break curves at a point group, where an attribute changes, or where it crosses a threshold"
    parms = (
        Parm('mode', "Cut Where", 'enum', items=[
            ('group', "Points in Group", "Cut (or remove) the points in the group"),
            ('change', "Attribute Changes", "Cut edges whose end points have different attribute values"),
            ('threshold', "Attribute Crosses Threshold", "Cut at the interpolated position where the attribute "
                                                          "crosses the threshold"),
        ]),
        Parm('strategy', "Strategy", 'enum', items=[
            ('cut', "Cut", "Split the curve, both pieces keep the cut point"),
            ('remove', "Remove", "Group: delete the points. Change: delete the changing edges. "
                                 "Threshold: also delete the parts below the threshold"),
        ]),
        Parm('group', "Cut Points", 'string', hide_when={'mode': ('change', 'threshold')},
             description="Point group pattern (empty cuts nothing)"),
        Parm('attrib', "Attribute", 'string', default='Cd', hide_when={'mode': 'group'},
             description="Point attribute (vectors use their length for the threshold)"),
        Parm('threshold', "Threshold", 'float', default=0.5, hide_when={'mode': ('group', 'change')}),
        Parm('tolerance', "Tolerance", 'float', default=1e-5, min=0.0, hide_when={'mode': ('group', 'threshold')}),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        if geo.num_prims == 0:
            return geo.copy()
        remove = p.strategy == 'remove'
        npts = geo.num_points
        vmark = np.zeros(npts, dtype=bool)
        vals = None
        if p.mode == 'group':
            if p.group.strip():
                vmark = parse_group_pattern(geo, POINT, p.group)
        else:
            name = p.attrib.strip()
            arr = geo.attribs[POINT].get(name)
            if arr is None or arr.dtype == object:
                raise CookError("PolyCut: no numeric point attribute '%s'" % name)
            vals = arr.reshape(npts, -1).astype(np.float64)
            if p.mode == 'threshold':
                vals = vals[:, 0] if vals.shape[1] == 1 else np.linalg.norm(vals, axis=1)
                vals = vals - float(p.threshold)
                if remove:
                    vmark = vals < 0
        A_parts, B_parts, t_parts, sizes, closed_out, prim_src = [], [], [], [], [], []

        def emit(A, B, t, closed, prim):
            A_parts.append(A)
            B_parts.append(B)
            t_parts.append(t)
            sizes.append(len(A))
            closed_out.append(closed)
            prim_src.append(prim)

        for i in range(geo.num_prims):
            pts = geo.prim_points(i)
            n = len(pts)
            if n == 0:
                continue
            closed = bool(geo.prim_closed[i]) and n >= 3
            m = n if closed else n - 1
            a, b = pts[:m], pts[(np.arange(m) + 1) % n]
            vdel = vmark[pts] if (remove and p.mode != 'change') else np.zeros(n, dtype=bool)
            vcut = vmark[pts] if (not remove and p.mode == 'group') else np.zeros(n, dtype=bool)
            edel = np.zeros(n, dtype=bool)
            ins_t = np.full(m, np.nan)
            if p.mode == 'change' and m:
                diff = np.any(np.abs(vals[a] - vals[b]) > p.tolerance, axis=1)
                if remove:
                    edel[:m] = diff
                else:
                    ins_t[diff] = 0.5
            elif p.mode == 'threshold' and m:
                va, vb = vals[a], vals[b]
                cross = (va < 0) != (vb < 0)
                ins_t[cross] = np.clip(va[cross] / (va[cross] - vb[cross]), 0.0, 1.0)
            ins = np.nonzero(~np.isnan(ins_t))[0]
            if not (vdel.any() or vcut.any() or edel.any() or len(ins)):
                emit(pts, pts, np.zeros(n), closed, i)
                continue
            # Refined sample sequence: vertex j has key 2j, a sample inserted on edge j has key 2j+1.
            keys = np.concatenate([2 * np.arange(n), 2 * ins + 1])
            order = np.argsort(keys, kind='stable')
            sA = np.concatenate([pts, a[ins]])[order]
            sB = np.concatenate([pts, b[ins]])[order]
            st = np.concatenate([np.zeros(n), ins_t[ins]])[order]
            scut = np.concatenate([vcut, np.ones(len(ins), dtype=bool)])[order]
            sdel = np.concatenate([vdel, np.zeros(len(ins), dtype=bool)])[order]
            sedel = np.concatenate([edel, np.zeros(len(ins), dtype=bool)])[order]   # edge after the sample is gone
            K = len(sA)
            if closed:
                brk_before = scut | sdel | np.roll(sedel, 1) | np.roll(sdel, 1)
                r = int(np.argmax(brk_before))
                roll = (np.arange(K + 1) + r) % K
                sA, sB, st, scut, sdel, sedel = (x[roll] for x in (sA, sB, st, scut, sdel, sedel))
            rep = np.where(scut, 2, 1)
            src = np.repeat(np.arange(len(sA)), rep)
            brk = np.zeros(len(src), dtype=bool)
            same = src[1:] == src[:-1]
            brk[1:] = same | (~same & sedel[src[:-1]]) | sdel[src[1:]] | sdel[src[:-1]]
            piece = np.cumsum(brk)
            keep = ~sdel[src]
            src, piece = src[keep], piece[keep]
            if len(src) == 0:
                continue
            bounds = np.nonzero(np.diff(piece))[0] + 1
            for seg in np.split(src, bounds):
                if len(seg) >= 2:
                    emit(sA[seg], sB[seg], st[seg], False, i)
        if not sizes:
            out = Geometry()
            out.attribs[POINT] = {k: v[:0] for k, v in geo.attribs[POINT].items()}
            return out
        return _sampled_curves(geo, np.concatenate(A_parts), sizes, closed_out, prim_src,
                               B=np.concatenate(B_parts), t=np.concatenate(t_parts))


@register_sop
class OrientAlongCurveSOP(SOP):
    id = 'orientalongcurve'
    label = "Orientation Along Curve"
    category = 'Attribute'
    inputs = 1
    icon = 'ORIENTATION_GIMBAL'
    description = "Per point frames along curves (parallel transport): N (tangent), up, orient"
    parms = (
        Parm('tangenttype', "Tangent Type", 'enum', items=[
            ('avg', "Average of Edges", ""), ('next', "Next Edge", ""), ('prev', "Previous Edge", ""),
        ]),
        Parm('uptype', "Up Vector Type", 'enum', items=[
            ('transport', "Parallel Transport", "Start from the up vector and carry it along without twisting"),
            ('constant', "Constant Up Vector", "Project the up vector at every point"),
        ]),
        Parm('upvector', "Up Vector", 'vector', default=(0.0, 1.0, 0.0)),
        Parm('continuity', "Enforce Closed Curve Continuity", 'bool', default=True,
             hide_when={'uptype': 'constant'},
             description="Spread the transport mismatch of closed curves along their length"),
        Parm('roll', "Roll", 'float', default=0.0, soft_min=-180, soft_max=180),
        Parm('fulltwists', "Full Twists", 'int', default=0, soft_min=-10, soft_max=10),
        Parm('outputN', "Output N (Tangent)", 'bool', default=True),
        Parm('outputup', "Output up", 'bool', default=True),
        Parm('outputorient', "Output orient", 'bool', default=True),
        Parm('outputtangentu', "Output tangentu", 'bool', default=False),
        Parm('outputout', "Output out (X Axis)", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        nv = geo.num_vertices
        if nv == 0:
            return geo
        frames = curve_frames_all(geo, p.tangenttype, p.uptype, p.upvector, p.continuity, p.roll, p.fulltwists)
        T, U, X = frames
        vtx = geo.vtx_point
        pa = geo.attribs[POINT]
        npts = geo.num_points

        def put(name, vals):
            arr = pa.get(name)
            if arr is None or arr.shape[1:] != vals.shape[1:] or arr.dtype.kind != 'f':
                arr = np.zeros((npts,) + vals.shape[1:])
            else:
                arr = arr.copy()
            arr[vtx] = vals
            pa[name] = arr

        if p.outputN:
            put('N', T)
        if p.outputup:
            put('up', U)
        if p.outputtangentu:
            put('tangentu', T)
        if p.outputout:
            put('out', X)
        if p.outputorient:
            put('orient', quat_from_matrices(np.stack([X, U, T], axis=2)))
        return geo


def curve_frames_all(geo, tangenttype='avg', uptype='transport', upvector=(0, 1, 0), continuity=True, roll=0.0,
                     fulltwists=0):
    """Frames for every vertex of every primitive, vectorized over all curves.

    Returns (T, U, X) per vertex: tangent, up (perpendicular to T) and X = U x T, so that
    [X, U, T] is a right handed rotation (Z along the curve, like Copy to Points)."""
    vp, local, size, prv, nxt, closed = _vertex_topology(geo)
    starts = geo.prim_start[:-1]
    Pv = geo.P[geo.vtx_point]
    e_next = Pv[nxt] - Pv
    e_prev = Pv - Pv[prv]
    zn = np.linalg.norm(e_next, axis=1, keepdims=True) < EPS
    zp = np.linalg.norm(e_prev, axis=1, keepdims=True) < EPS
    if tangenttype == 'next':
        T = np.where(zn, e_prev, e_next)
    elif tangenttype == 'prev':
        T = np.where(zp, e_next, e_prev)
    else:
        T = _normalize(e_next) + _normalize(e_prev)
        T = np.where(np.linalg.norm(T, axis=1, keepdims=True) < 1e-9, np.where(zn, e_prev, e_next), T)
    T = _normalize(T, fallback=np.array([0.0, 0.0, 1.0]))
    up = np.asarray(upvector, dtype=np.float64)
    if np.linalg.norm(up) < EPS:
        up = np.array([0.0, 1.0, 0.0])
    up = up / np.linalg.norm(up)

    def perpendicular(axis):
        r = axis - T * (T @ axis)[:, None]
        bad = np.linalg.norm(r, axis=1) < 1e-6
        if bad.any():
            alt = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
            r2 = alt - T * (T @ alt)[:, None]
            bad2 = np.linalg.norm(r2, axis=1) < 1e-6
            r2[bad2] = np.cross(T[bad2], [0.0, 1.0, 0.0])
            r[bad] = r2[bad]
        return _normalize(r)

    R = perpendicular(up)          # Reference direction per vertex
    S = np.cross(T, R)             # R rotated by 90 degrees about T
    seg = np.linalg.norm(e_next, axis=1)
    seg_out = np.where(closed | (local < size - 1), seg, 0.0)
    dist = _segmented_cumsum(seg_out, vp, starts) - seg_out
    total = np.zeros(geo.num_prims)
    np.add.at(total, vp, seg_out)
    u = dist / np.where(total[vp] < EPS, 1.0, total[vp])
    if uptype == 'constant':
        theta = np.zeros(len(T))
    else:
        # Transport R[prev] onto T and measure its angle against R: the per-edge twist.
        def transport_angle(a_idx, b_idx):
            a, b = T[a_idx], T[b_idx]
            r = R[a_idx]
            axis = np.cross(a, b)
            s = np.linalg.norm(axis, axis=1)
            c = np.einsum('ij,ij->i', a, b)
            k = axis / np.where(s < EPS, 1.0, s)[:, None]
            flip = (s < 1e-9) & (c < 0)
            k[flip] = r[flip]
            s = np.where(flip, 0.0, s)
            kr = np.einsum('ij,ij->i', k, r)
            moved = r * c[:, None] + np.cross(k, r) * s[:, None] + k * (kr * (1 - c))[:, None]
            moved[(s < 1e-9) & ~flip] = r[(s < 1e-9) & ~flip]
            rb, sb = R[b_idx], S[b_idx]
            return np.arctan2(np.einsum('ij,ij->i', moved, sb), np.einsum('ij,ij->i', moved, rb))

        idx = np.arange(len(T))
        delta = np.where(local > 0, transport_angle(np.maximum(idx - 1, 0), idx), 0.0)
        theta = _segmented_cumsum(delta, vp, starts)
        if continuity:
            sizes = geo.prim_sizes()
            cl = geo.prim_closed & (sizes >= 3)
            if cl.any():
                last = (geo.prim_start[1:] - 1)[cl]
                first = starts[cl]
                mismatch = theta[last] + transport_angle(last, first)
                mismatch = (mismatch + np.pi) % (2 * np.pi) - np.pi
                corr = np.zeros(geo.num_prims)
                corr[cl] = mismatch
                theta = theta - corr[vp] * u
    theta = theta + np.radians(roll) + 2 * np.pi * fulltwists * u
    U = R * np.cos(theta)[:, None] + S * np.sin(theta)[:, None]
    X = np.cross(U, T)
    return T, U, X


@register_sop
class FilletSOP(SOP):
    id = 'fillet'
    label = "Fillet (Round Corners)"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_BEVEL'
    description = "Replace sharp curve / polygon corners with circular arcs of a radius"
    parms = (
        Parm('group', "Corner Points", 'string', description="Point group pattern (empty rounds every corner)"),
        Parm('radius', "Radius", 'float', default=0.1, min=0.0, soft_max=1.0),
        Parm('divs', "Divisions", 'int', default=6, min=1, soft_max=32,
             description="Segments per arc (1 gives a chamfer)"),
        Parm('minangle', "Minimum Corner Angle", 'float', default=1.0, min=0.0, max=180.0,
             description="Corners that turn less than this (degrees) stay sharp"),
        Parm('clamp', "Clamp Radius", 'bool', default=True,
             description="Shrink arcs that would use more than half of an adjacent edge"),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        if geo.num_vertices == 0:
            return geo.copy()
        vp, local, size, prv, nxt, closed = _vertex_topology(geo)
        vtx = geo.vtx_point
        Pv = geo.P[vtx]
        a = Pv[prv] - Pv
        b = Pv[nxt] - Pv
        la = np.linalg.norm(a, axis=1)
        lb = np.linalg.norm(b, axis=1)
        ua = a / np.where(la < EPS, 1.0, la)[:, None]
        ub = b / np.where(lb < EPS, 1.0, lb)[:, None]
        cos_t = np.clip(np.einsum('ij,ij->i', ua, ub), -1.0, 1.0)
        theta = np.arccos(cos_t)          # interior angle at the corner
        turn = np.pi - theta
        corner = np.where(closed, size >= 3, (local > 0) & (local < size - 1))
        ok = corner & (la > 1e-9) & (lb > 1e-9) & (turn > np.radians(max(p.minangle, 1e-3))) & (theta > 1e-6)
        if p.group.strip():
            ok &= parse_group_pattern(geo, POINT, p.group)[vtx]
        if p.radius <= 0:
            ok[:] = False
        half_tan = np.tan(np.where(ok, theta, 1.0) / 2)
        d = p.radius / half_tan
        if p.clamp:
            d = np.minimum(d, 0.5 * np.minimum(la, lb))
        divs = max(int(p.divs), 1)
        counts = np.where(ok, divs + 1, 1)
        A = np.repeat(vtx, counts)
        owner = np.repeat(np.arange(len(vtx)), counts)
        k = np.arange(len(A)) - np.repeat(np.cumsum(counts) - counts, counts)
        P = geo.P[A].copy()
        sel = ok[owner]
        if sel.any():
            o = owner[sel]
            bis = _normalize(ua[o] + ub[o])
            r_eff = d[o] * half_tan[o]
            center = Pv[o] + bis * (d[o] / np.cos(theta[o] / 2))[:, None]
            t1 = Pv[o] + ua[o] * d[o][:, None] - center
            t2 = Pv[o] + ub[o] * d[o][:, None] - center
            phi = turn[o]
            s = (k[sel] / divs)
            w1 = np.sin((1 - s) * phi) / np.sin(phi)
            w2 = np.sin(s * phi) / np.sin(phi)
            arc = t1 * w1[:, None] + t2 * w2[:, None]
            # Guard against rounding: keep exactly on the circle.
            arc = _normalize(arc) * r_eff[:, None]
            P[sel] = center + arc
        prim_sizes = np.zeros(geo.num_prims, dtype=np.int64)
        np.add.at(prim_sizes, vp, counts)
        return _sampled_curves(geo, A, prim_sizes, geo.prim_closed, np.arange(geo.num_prims), P=P)


@register_sop
class ConvertCurveSOP(SOP):
    id = 'convertcurve'
    label = "Convert Curve"
    category = 'Polygon'
    inputs = 1
    icon = 'CURVE_BEZCURVE'
    description = "Treat polylines as NURBS / Bezier / Catmull-Rom control hulls and output evaluated polylines"
    parms = (
        Parm('type', "Curve Type", 'enum', items=[
            ('nurbs', "NURBS / B-Spline", "Uniform knots, optional weights"),
            ('bezier', "Bezier", "Every Nth point (N = degree) lies on the curve"),
            ('catmull', "Catmull-Rom", "Smooth curve through every point"),
        ]),
        Parm('degree', "Degree", 'int', default=3, min=1, max=10, hide_when={'type': 'catmull'}),
        Parm('knots', "Knots", 'enum', items=[
            ('clamped', "Clamped (End Interpolating)", "Open curves start and end at their end points"),
            ('uniform', "Uniform (Unclamped)", ""),
        ], hide_when={'type': ('bezier', 'catmull')}),
        Parm('weightattrib', "Weight Attribute", 'string', default='Pw', hide_when={'type': ('bezier', 'catmull')},
             description="Point float attribute with rational weights (falls back to 'weight')"),
        Parm('divisions', "Divisions per Span", 'int', default=8, min=1, soft_max=64),
        Parm('group', "Group", 'string', description="Primitives to convert (others pass through)"),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        if geo.num_prims == 0:
            return geo.copy()
        sel = parse_group_pattern(geo, PRIM, p.group) & (geo.prim_sizes() >= 2)
        names, shapes, dtypes, M = _stack_point_attribs(geo)
        w_all = None
        if p.type == 'nurbs':
            for wn in (p.weightattrib.strip(), 'weight'):
                arr = geo.attribs[POINT].get(wn)
                if wn and arr is not None and arr.dtype.kind in 'biuf':
                    w_all = arr.reshape(geo.num_points, -1)[:, -1].astype(np.float64)
                    break
        divs = max(int(p.divisions), 1)
        rows, first_pt, sizes, closed_out, prim_src = [], [], [], [], []
        for i in range(geo.num_prims):
            pts = geo.prim_points(i)
            closed = bool(geo.prim_closed[i])
            if not sel[i]:
                if len(pts) == 0:
                    continue
                rows.append(M[pts])
                first_pt.append(pts)
            else:
                C = M[pts]
                if p.type == 'nurbs':
                    E = nurbs_curve(C, p.degree, closed, p.knots, None if w_all is None else w_all[pts], divs)
                elif p.type == 'bezier':
                    E = bezier_curve(C, p.degree, closed, divs)
                else:
                    E = catmull_rom(C, divs, closed and len(C) >= 3) if len(C) >= 3 else C.copy()
                rows.append(E)
                first_pt.append(np.full(len(E), pts[0]))
            sizes.append(len(rows[-1]))
            closed_out.append(closed and len(rows[-1]) >= 3)
            prim_src.append(i)
        if not rows:
            return Geometry()
        attribs = _unstack(names, shapes, dtypes, np.concatenate(rows))
        src_pt = np.concatenate(first_pt)
        for name, arr in geo.attribs[POINT].items():
            if name not in attribs:
                attribs[name] = arr[src_pt]
        return _curves_geo(geo, attribs, sizes, closed_out, prim_src)


@register_sop
class PolyExpand2DSOP(SOP):
    id = 'polyexpand2d'
    label = "PolyExpand 2D"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_SOLIDIFY'
    description = "Offset planar curves / polygons in their plane by a distance (mitered corners)"
    parms = (
        Parm('offset', "Offset", 'float', default=0.1, min=0.0, soft_max=1.0),
        Parm('numoffsets', "Number of Offsets", 'int', default=1, min=1, soft_max=10),
        Parm('side', "Output", 'enum', items=[
            ('outside', "Outside", ""), ('inside', "Inside", ""), ('both', "Inside and Outside", ""),
        ]),
        Parm('miterlimit', "Miter Limit", 'float', default=4.0, min=1.0, soft_max=10.0,
             description="Longest corner extension, in multiples of the offset"),
        Parm('keepinput', "Keep Input", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        if geo.num_vertices == 0:
            return geo.copy() if p.keepinput else Geometry()
        vp, local, size, prv, nxt, closed = _vertex_topology(geo)
        vtx = geo.vtx_point
        Pv = geo.P[vtx]
        Nprim = geo.prim_normals()
        Nv = Nprim[vp]
        e_out = Pv[nxt] - Pv           # edge leaving the vertex
        e_in = Pv - Pv[prv]            # edge entering the vertex
        # Straight open curves have no plane: use the up axis (or Z) as the plane normal.
        flat = np.linalg.norm(Nv, axis=1) < 0.5
        if flat.any():
            e = np.where(np.linalg.norm(e_out[flat], axis=1, keepdims=True) > EPS, e_out[flat], e_in[flat])
            alt = np.where(np.abs(_normalize(e)[:, 1:2]) > 0.99, [[0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0]])
            Nv[flat] = alt
        n_out = _normalize(np.cross(e_out, Nv))
        n_in = _normalize(np.cross(e_in, Nv))
        has_out = np.linalg.norm(n_out, axis=1) > 0.5
        has_in = np.linalg.norm(n_in, axis=1) > 0.5
        n_out = np.where(has_out[:, None], n_out, n_in)
        n_in = np.where(has_in[:, None], n_in, n_out)
        dot = np.einsum('ij,ij->i', n_in, n_out)
        miter = (n_in + n_out) / np.maximum(1.0 + dot, 1e-9)[:, None]
        ml = np.linalg.norm(miter, axis=1)
        limit = max(float(p.miterlimit), 1.0)
        miter *= np.where(ml > limit, limit / np.maximum(ml, EPS), 1.0)[:, None]
        # Edges that exist (open curves have no closing edge) for the collapse test.
        edge_ok = (closed | (local < size - 1)) & (np.linalg.norm(e_out, axis=1) > 1e-9)
        sides = {'outside': [1.0], 'inside': [-1.0], 'both': [1.0, -1.0]}[p.side]
        parts = [geo.copy()] if p.keepinput else []
        nprims = geo.num_prims
        for k in range(1, max(int(p.numoffsets), 1) + 1):
            for sgn in sides:
                d = sgn * p.offset * k
                newP = Pv + miter * d
                new_e = newP[nxt] - newP
                flipped = edge_ok & (np.einsum('ij,ij->i', new_e, e_out) <= 0)
                bad = np.zeros(nprims, dtype=bool)
                bad[vp[flipped]] = True
                part = _sampled_curves(geo, vtx, geo.prim_sizes(), geo.prim_closed, np.arange(nprims), P=newP)
                if bad.any():
                    part.delete_prims(bad)
                if part.num_prims == 0:
                    continue
                gname = 'outside' if sgn > 0 else 'inside'
                part.groups[PRIM][gname] = np.ones(part.num_prims, dtype=bool)
                part.attribs[PRIM]['offsetindex'] = np.full(part.num_prims, k, dtype=np.int64)
                parts.append(part)
        if not parts:
            return Geometry()
        out = merge(parts)
        for gname in ('inside', 'outside'):
            out.groups[PRIM].setdefault(gname, np.zeros(out.num_prims, dtype=bool))
        return out


@register_sop
class CurveAttribsSOP(SOP):
    id = 'curveattribs'
    label = "Curve Attributes"
    category = 'Attribute'
    inputs = 1
    icon = 'DRIVER_DISTANCE'
    description = "Per point curveu (0-1), arc length, curve number and index along the curve"
    parms = (
        Parm('curveu', "Curve U Attribute", 'string', default='curveu', description="Empty to skip"),
        Parm('dist', "Arc Length Attribute", 'string', default='dist'),
        Parm('curvenum', "Curve Number Attribute", 'string', default='curvenum'),
        Parm('ptidx', "Index Along Curve Attribute", 'string', default=''),
        Parm('length', "Curve Length (Primitive)", 'string', default='perimeter'),
        Parm('reverse', "Reverse Direction", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        if geo.num_vertices == 0:
            return geo
        vp, local, size, prv, nxt, closed = _vertex_topology(geo)
        Pv = geo.P[geo.vtx_point]
        seg = np.linalg.norm(Pv[nxt] - Pv, axis=1)
        seg = np.where(closed | (local < size - 1), seg, 0.0)
        starts = geo.prim_start[:-1]
        dist = _segmented_cumsum(seg, vp, starts) - seg
        total = np.zeros(geo.num_prims)
        np.add.at(total, vp, seg)
        idx = local.copy()
        if p.reverse:
            # Open: measured from the last point. Closed: same start point, other direction.
            dist = np.where(closed, np.where(local == 0, 0.0, total[vp] - dist), total[vp] - dist)
            idx = np.where(closed, (size - local) % size, size - 1 - local)
        u = dist / np.where(total[vp] < EPS, 1.0, total[vp])
        pa = geo.attribs[POINT]
        npts = geo.num_points
        vtx = geo.vtx_point
        for name, vals, dtype in ((p.curveu, u, np.float64), (p.dist, dist, np.float64),
                                  (p.curvenum, vp, np.int64), (p.ptidx, idx, np.int64)):
            name = name.strip()
            if name:
                arr = np.zeros(npts, dtype=dtype)
                arr[vtx] = vals
                pa[name] = arr
        if p.length.strip():
            geo.attribs[PRIM][p.length.strip()] = total
        return geo


@register_sop
class SpiralSOP(SOP):
    id = 'spiral'
    label = "Spiral / Helix"
    category = 'Primitive'
    icon = 'MOD_SCREW'
    description = "Helix (constant radius) or spiral (changing radius) polyline"
    parms = (
        Parm('axis', "Axis", 'enum', items=AXIS_ITEMS),
        Parm('turns', "Turns", 'float', default=3.0, min=0.0, soft_max=20.0),
        Parm('height', "Height", 'float', default=1.0, soft_min=-10, soft_max=10),
        Parm('radstart', "Start Radius", 'float', default=0.5, min=0.0, soft_max=5.0),
        Parm('radend', "End Radius", 'float', default=0.5, min=0.0, soft_max=5.0),
        Parm('radiusinterp', "Radius Interpolation", 'enum', items=[
            ('linear', "Linear (Archimedean)", ""), ('log', "Logarithmic", "Constant ratio per turn"),
        ]),
        Parm('divs', "Points per Turn", 'int', default=32, min=3, soft_max=200),
        Parm('startangle', "Start Angle", 'float', default=0.0),
        Parm('clockwise', "Clockwise", 'bool', default=False),
        Parm('t', "Center", 'vector'),
        Parm('type', "Primitive Type", 'enum', items=[('poly', "Polyline", ""), ('points', "Points Only", "")]),
    )

    def cook(self, inputs, p, ctx):
        n = max(int(np.ceil(max(p.turns, 0.0) * max(p.divs, 3))), 1) + 1
        s = np.linspace(0.0, 1.0, n)
        ang = np.radians(p.startangle) + 2 * np.pi * p.turns * s * (-1 if p.clockwise else 1)
        r0, r1 = max(p.radstart, 0.0), max(p.radend, 0.0)
        if p.radiusinterp == 'log' and r0 > 0 and r1 > 0:
            r = r0 * (r1 / r0) ** s
        else:
            r = r0 + (r1 - r0) * s
        P = np.stack([r * np.cos(ang), p.height * s, -r * np.sin(ang)], axis=1)
        P = orient_axis(P, p.axis) + np.asarray(p.t, dtype=np.float64)
        geo = Geometry.from_polys(P)
        if p.type == 'poly' and n > 1:
            geo.add_prims_flat([n], np.arange(n), False)
        return geo


@register_sop
class ArcSOP(SOP):
    id = 'arc'
    label = "Arc"
    category = 'Primitive'
    icon = 'MESH_CIRCLE'
    description = "Circular / elliptical arc: open, closed (chord) or sliced (pie)"
    parms = (
        Parm('type', "Arc Type", 'enum', items=[
            ('open', "Open", "Open polyline"), ('closed', "Closed", "Closed with a straight chord"),
            ('sliced', "Sliced", "Pie slice through the center"),
        ]),
        Parm('orient', "Orientation", 'enum', items=ORIENT_ITEMS),
        Parm('rad', "Radius", 'vector2', default=(1.0, 1.0)),
        Parm('t', "Center", 'vector'),
        Parm('anglestart', "Start Angle", 'float', default=0.0),
        Parm('angleend', "End Angle", 'float', default=90.0),
        Parm('divs', "Divisions", 'int', default=12, min=1, soft_max=200),
    )

    def cook(self, inputs, p, ctx):
        n = max(int(p.divs), 1) + 1
        ang = np.radians(np.linspace(p.anglestart, p.angleend, n))
        P = np.stack([p.rad[0] * np.cos(ang), np.zeros(n), -p.rad[1] * np.sin(ang)], axis=1)
        if p.type == 'sliced':
            P = np.concatenate([P, np.zeros((1, 3))])
        P = orient_plane(P, p.orient) + np.asarray(p.t, dtype=np.float64)
        geo = Geometry.from_polys(P)
        m = len(P)
        geo.add_prims_flat([m], np.arange(m), p.type != 'open' and m >= 3)
        return geo
