# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Vellum-style grains: position based granular material (Macklin et al. 2014, "Unified particle
physics for real-time applications"). Particle contacts with static/kinetic friction, a ground
plane, a box container and an optional collider. Friction gives sand piles an angle of repose."""

import numpy as np

from .base import SOP, Parm, register_sop, CookError, upstream_changed
from ..core.geometry import Geometry, POINT
from .fluid import neighbour_pairs
from .particles import parm_key, geo_sig, cache_store

MAX_GRAINS = 200_000
MAX_SUBSTEPS = 20


def _in(inputs, i=0):
    if len(inputs) > i and inputs[i] is not None:
        return inputs[i]
    return Geometry()


def _scatter_add(n, idx, vec):
    out = np.zeros((n, 3))
    for k in range(3):
        out[:, k] = np.bincount(idx, weights=vec[:, k], minlength=n)
    return out


def _friction(dx, normal, pen, mu_s, mu_k):
    """Positional friction (Macklin 2014 eq. 24): tangential displacement to cancel, per contact."""
    dn = np.einsum('ij,ij->i', dx, normal)
    dt = dx - dn[:, None] * normal
    lt = np.linalg.norm(dt, axis=1)
    static = lt < mu_s * pen
    scale = np.where(static, 1.0, np.minimum(mu_k * pen / np.maximum(lt, 1e-12), 1.0))
    return dt * scale[:, None]


def _contacts(X, Xprev, I, J, rsum, fI, mu_s, mu_k, n, omega):
    """One Jacobi pass over particle contacts (with positional friction). Returns the correction."""
    d = X[I] - X[J]
    dist = np.linalg.norm(d, axis=1)
    pen = rsum - dist
    c = pen > 0
    if not c.any():
        return None
    Ic, Jc, pc = I[c], J[c], pen[c]
    nrm = d[c] / np.maximum(dist[c], 1e-12)[:, None]
    corr = nrm * pc[:, None]
    if mu_s > 0 and Xprev is not None:
        rel = (X[Ic] - Xprev[Ic]) - (X[Jc] - Xprev[Jc])
        corr = corr - _friction(rel, nrm, pc, mu_s, mu_k)
    fi = fI[c][:, None]
    cnt = np.bincount(Ic, minlength=n) + np.bincount(Jc, minlength=n)
    dX = _scatter_add(n, Ic, corr * fi) - _scatter_add(n, Jc, corr * (1.0 - fi))
    return dX * (omega / np.maximum(cnt, 1))[:, None]


def grain_step(st, dt, p, collider):
    P0, V, r = st['P'].copy(), st['V'], st['r']
    n = len(P0)
    g = np.asarray(p.gravity, dtype=np.float64)
    V = V + g * dt
    if p.drag > 0:
        V *= np.exp(-p.drag * dt)
    # Clamp the motion per substep (no tunnelling through neighbours).
    sp = np.linalg.norm(V, axis=1)
    V = V * np.minimum(1.0, (r / dt) / np.maximum(sp, 1e-12))[:, None]
    X = P0 + V * dt
    lo = np.asarray(p.boxmin, dtype=np.float64)
    hi = np.asarray(p.boxmax, dtype=np.float64)
    rmax = float(r.max())
    # Contact candidates once per substep, with a margin for the motion during the iterations.
    I, J = neighbour_pairs(X, 2.0 * rmax * 1.2)
    keep = I < J
    I, J = I[keep], J[keep]
    rsum = r[I] + r[J]
    mu_s, mu_k = p.friction, p.friction * 0.8
    # Shock propagation (mass scaling): lower grains act heavier so stacks don't sink into each other.
    hgt = (P0[:, 1] - P0[:, 1].min()) / (2.0 * rmax)
    w = np.exp(np.minimum(hgt * p.stackscale, 4.0))
    fI = w[I] / (w[I] + w[J])
    if len(I):
        # Stabilization: resolve penetration left at the start positions, moving both the start and
        # predicted positions, so fixing old overlaps adds no velocity.
        for _ in range(2):
            dX = _contacts(P0, None, I, J, rsum, fI, 0.0, 0.0, n, 1.0)
            if dX is None:
                break
            P0 += dX
            X += dX
        P0 = _boundaries(P0, P0, r, lo, hi, p, 0.0, 0.0)
    iters = max(int(p.iterations), 1)
    for _ in range(iters):
        if len(I):
            dX = _contacts(X, P0, I, J, rsum, fI, mu_s, mu_k, n, 1.0)
            if dX is not None:
                X += dX
        X = _boundaries(X, P0, r, lo, hi, p, mu_s, mu_k)
    if collider is not None and collider.tree is not None:
        X = collider.push_out(X, float(np.mean(r)))
        X = _boundaries(X, P0, r, lo, hi, p, mu_s, mu_k)
    # Sleeping: grains that barely move keep their position (piles come to rest).
    if p.sleep > 0:
        slow = np.einsum('ij,ij->i', X - P0, X - P0) < (p.sleep * dt) ** 2
        X[slow] = P0[slow]
    V = (X - P0) / dt
    st['P'], st['V'] = X, V


def _boundaries(X, P0, r, lo, hi, p, mu_s, mu_k):
    """Ground plane and box container with friction."""
    planes = []
    if p.ground:
        planes.append((1, p.groundy, 1.0))
    if p.container:
        for k in range(3):
            planes.append((k, lo[k], 1.0))
            planes.append((k, hi[k], -1.0))
    for axis, offset, sign in planes:
        # Signed distance of the particle surface to the plane (positive inside).
        sd = (X[:, axis] - offset) * sign - r
        hit = sd < 0
        if not hit.any():
            continue
        pen = -sd[hit]
        X[hit, axis] += pen * sign
        if mu_s > 0:
            nrm = np.zeros((int(hit.sum()), 3))
            nrm[:, axis] = sign
            dx = X[hit] - P0[hit]
            X[hit] -= _friction(dx, nrm, pen, mu_s, mu_k)
    return X


@register_sop
class VellumGrainsSOP(SOP):
    id = 'vellumgrains'
    label = "Vellum Grains"
    category = 'Simulation'
    inputs = 2
    input_labels = ("Grain Points", "Collision Geometry")
    icon = 'PARTICLES'
    time_dependent = True
    description = ("Sand-like grains (position based dynamics): particle contacts with friction, ground, box "
                   "container and collider. pscale is the grain radius")
    parms = (
        Parm('startframe', "Start Frame", 'int', default=1),
        Parm('radius', "Particle Radius", 'float', default=0.05, min=0.001, soft_max=0.5,
             description="Used for points without pscale"),
        Parm('usepscale', "Use pscale", 'bool', default=True, description="Points' pscale attribute is the radius"),
        Parm('substeps', "Substeps", 'int', default=2, min=1, max=20),
        Parm('iterations', "Constraint Iterations", 'int', default=15, min=1, max=100),
        Parm('gravity', "Gravity", 'vector', default=(0.0, -9.81, 0.0)),
        Parm('friction', "Friction", 'float', default=0.6, min=0.0, soft_max=2.0,
             description="Static friction coefficient (kinetic is 80%); higher piles steeper"),
        Parm('stackscale', "Stacking Stiffness", 'float', default=0.5, min=0.0, soft_max=2.0,
             description="Mass scaling with height (shock propagation): higher keeps tall piles from sinking"),
        Parm('drag', "Drag", 'float', default=0.0, min=0.0, soft_max=5.0),
        Parm('sleep', "Sleep Velocity", 'float', default=0.05, min=0.0, soft_max=0.5,
             description="Grains slower than this stop (helps piles come to rest)"),
        Parm('ground', "Ground Plane", 'bool', default=True),
        Parm('groundy', "Ground Height", 'float', default=0.0),
        Parm('container', "Box Container", 'bool', default=True),
        Parm('boxmin', "Container Min", 'vector', default=(-2.0, -1.0, -2.0)),
        Parm('boxmax', "Container Max", 'vector', default=(2.0, 6.0, 2.0)),
    )

    def cook(self, inputs, p, ctx):
        from .vellum import _Collider
        geo = _in(inputs, 0)
        col = inputs[1] if len(inputs) > 1 and inputs[1] is not None else None
        state = ctx.state
        frame = int(round(ctx.frame))
        start = int(p.startframe)
        if geo.num_points > MAX_GRAINS:
            raise CookError("Too many grains (%d, max %d)" % (geo.num_points, MAX_GRAINS))
        key = (parm_key(p), geo.num_points)
        sig = (geo_sig(geo), geo_sig(col))
        edited = state.get('key') == key and upstream_changed(state, frame, sig)
        if state.get('key') != key or frame <= start or 'sim' not in state or edited:
            state.clear()
            state['key'] = key
            state['start'] = start
            upstream_changed(state, frame, sig)
            n = geo.num_points
            ps = geo.attribs[POINT].get('pscale')
            if p.usepscale and ps is not None and ps.shape == (n,):
                r = np.maximum(np.asarray(ps, dtype=np.float64), 1e-4)
            else:
                r = np.full(n, max(float(p.radius), 1e-4))
            v0 = geo.attribs[POINT].get('v')
            V = np.array(v0, dtype=np.float64) if v0 is not None and v0.shape == (n, 3) else np.zeros((n, 3))
            state['sim'] = {'P': geo.P.copy(), 'V': V, 'r': r}
            state['frames'] = {}
            state['last'] = start
            cache_store(state, start, {'P': geo.P.copy(), 'V': V.copy()})
        frames = state['frames']
        if frame <= start or geo.num_points == 0:
            return self._output(geo, frames[start], state['sim']['r'])
        if frame not in frames:
            if frame < state['last']:
                state.clear()
                return self.cook(inputs, p, ctx)
            st = state['sim']
            collider = _Collider(col) if col is not None and col.num_prims else None
            frame_dt = 1.0 / ctx.fps
            rmin = float(st['r'].min())
            for f in range(state['last'] + 1, frame + 1):
                # Adaptive substeps: a grain moves at most about its radius per substep.
                vmax = float(np.sqrt(np.einsum('ij,ij->i', st['V'], st['V']).max()))
                sub = int(min(max(int(p.substeps), np.ceil(vmax * frame_dt / rmin)), MAX_SUBSTEPS))
                dt = frame_dt / sub
                for _ in range(sub):
                    grain_step(st, dt, p, collider)
                cache_store(state, f, {'P': st['P'].copy(), 'V': st['V'].copy()})
                state['last'] = f
        return self._output(geo, frames[frame], state['sim']['r'])

    @staticmethod
    def _output(geo, data, r):
        out = Geometry()
        # Packed grains stay packed: the __inst* attributes index the input's instances.
        out.instances = list(geo.instances)
        for name, arr in geo.attribs[POINT].items():
            out.attribs[POINT][name] = arr.copy()
        for name, g in geo.groups[POINT].items():
            out.groups[POINT][name] = g.copy()
        out.P = data['P']
        out.attribs[POINT]['v'] = data['V'].copy()
        out.attribs[POINT]['pscale'] = r.copy()
        return out
