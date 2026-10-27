# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Position Based Fluids (Macklin & Mueller 2013) in numpy: a small FLIP-like particle fluid."""

import numpy as np

from .base import SOP, Parm, register_sop
from ..core.geometry import Geometry, POINT


def _in(inputs, i=0):
    if len(inputs) > i and inputs[i] is not None:
        return inputs[i]
    return Geometry()


def neighbour_pairs(P, h):
    """All ordered pairs (i, j), i != j, closer than h. Uses a sorted spatial hash, fully vectorized."""
    n = len(P)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    cell = np.floor(P / h).astype(np.int64)
    cell -= cell.min(axis=0)
    dims = cell.max(axis=0) + 3
    def key_of(c):
        return ((c[:, 0] + 1) * dims[1] + (c[:, 1] + 1)) * dims[2] + (c[:, 2] + 1)
    keys = key_of(cell)
    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    I_parts, J_parts = [], []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                nk = key_of(cell + np.array([dx, dy, dz]))
                start = np.searchsorted(sorted_keys, nk, side='left')
                end = np.searchsorted(sorted_keys, nk, side='right')
                counts = end - start
                total = int(counts.sum())
                if total == 0:
                    continue
                i_idx = np.repeat(np.arange(n), counts)
                offs = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
                j_idx = order[np.repeat(start, counts) + offs]
                I_parts.append(i_idx)
                J_parts.append(j_idx)
    I = np.concatenate(I_parts)
    J = np.concatenate(J_parts)
    d = P[I] - P[J]
    close = (I != J) & (np.einsum('ij,ij->i', d, d) < h * h)
    return I[close], J[close]


def _poly6(r2, h):
    return np.where(r2 < h * h, (315.0 / (64.0 * np.pi * h ** 9)) * (h * h - r2) ** 3, 0.0)


def _spiky_grad(d, r, h):
    coef = np.where((r > 1e-9) & (r < h), -(45.0 / (np.pi * h ** 6)) * (h - r) ** 2 / np.maximum(r, 1e-9), 0.0)
    return d * coef[:, None]


def pbf_step(st, dt, p):
    P, V = st['P'], st['V']
    n = len(P)
    h = st['h']
    rho0 = st['rho0']
    V = V + np.array([0.0, -p.gravity, 0.0]) * dt
    X = P + V * dt
    lo = np.asarray(p.boxmin, dtype=np.float64)
    hi = np.asarray(p.boxmax, dtype=np.float64)
    I, J = neighbour_pairs(X, h)
    for _ in range(max(int(p.iterations), 1)):
        d = X[I] - X[J]
        r2 = np.einsum('ij,ij->i', d, d)
        r = np.sqrt(r2)
        rho = np.bincount(I, weights=_poly6(r2, h), minlength=n) + _poly6(np.zeros(1), h)[0]
        # Near the container walls part of the kernel is outside: lower the rest density there
        # so particles don't over-compress against the walls.
        frac = np.ones(n)
        for k in range(3):
            frac *= np.interp(X[:, k] - lo[k], st['wall_d'], st['wall_f'])
            frac *= np.interp(hi[k] - X[:, k], st['wall_d'], st['wall_f'])
        frac = np.maximum(frac, 0.45)
        C = np.maximum(rho / (rho0 * frac) - 1.0, 0.0)  # one-sided: only push apart
        grad = _spiky_grad(d, r, h) / rho0
        grad_i = np.zeros((n, 3))
        for k in range(3):
            grad_i[:, k] = np.bincount(I, weights=grad[:, k], minlength=n)
        sum_grad2 = np.bincount(I, weights=np.einsum('ij,ij->i', grad, grad), minlength=n) \
            + np.einsum('ij,ij->i', grad_i, grad_i)
        lam = -C / (sum_grad2 + p.relaxation * st['grad2_rest'])
        # The one-sided constraint already avoids tensile clumping, so no artificial pressure term.
        coef = lam[I] + lam[J]
        dP = np.zeros((n, 3))
        contrib = grad * coef[:, None]
        for k in range(3):
            dP[:, k] = np.bincount(I, weights=contrib[:, k], minlength=n)
        X = X + dP
        X = np.clip(X, lo, hi)
    newV = (X - P) / dt
    # XSPH viscosity.
    if p.viscosity > 0 and len(I):
        d = X[I] - X[J]
        w = _poly6(np.einsum('ij,ij->i', d, d), h)
        dv = (newV[J] - newV[I]) * w[:, None]
        acc = np.zeros((n, 3))
        for k in range(3):
            acc[:, k] = np.bincount(I, weights=dv[:, k], minlength=n)
        newV = newV + p.viscosity * acc / rho0
    # Walls: kill velocity into the box sides.
    for k in range(3):
        newV[:, k] = np.where((X[:, k] <= lo[k]) & (newV[:, k] < 0), 0.0, newV[:, k])
        newV[:, k] = np.where((X[:, k] >= hi[k]) & (newV[:, k] > 0), 0.0, newV[:, k])
    st['P'], st['V'] = X, newV


@register_sop
class FluidSolverSOP(SOP):
    id = 'fluidsolver'
    label = "Fluid Solver (PBF)"
    category = 'Manipulate'
    inputs = 1
    input_labels = ("Fluid Particles",)
    icon = 'MOD_FLUIDSIM'
    time_dependent = True
    description = "Particle fluid in a box (Position Based Fluids). Feed it Points from Volume"
    parms = (
        Parm('startframe', "Start Frame", 'int', default=1),
        Parm('particlesep', "Particle Separation", 'float', default=0.1, min=0.005, soft_max=1),
        Parm('substeps', "Substeps", 'int', default=2, min=1, max=20),
        Parm('iterations', "Density Iterations", 'int', default=4, min=1, max=20),
        Parm('gravity', "Gravity", 'float', default=9.81),
        Parm('viscosity', "Viscosity (XSPH)", 'float', default=0.01, min=0, max=1),
        Parm('relaxation', "Constraint Relaxation", 'float', default=0.1, min=0.0, soft_max=2.0,
             description="Softens the density constraint (relative to the kernel's gradient at rest)"),
        Parm('boxmin', "Container Min", 'vector', default=(-2.0, 0.0, -1.0)),
        Parm('boxmax', "Container Max", 'vector', default=(2.0, 4.0, 1.0)),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        state = ctx.state
        frame = int(round(ctx.frame))
        start = int(p.startframe)
        key = (tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in p.items())),
               geo.num_points, float(geo.P.sum()) if geo.num_points else 0.0)
        if state.get('key') != key or frame <= start or 'sim' not in state:
            state.clear()
            state['key'] = key
            h = p.particlesep * 2.0
            st = {'P': geo.P.copy(), 'V': np.zeros_like(geo.P), 'h': h}
            # Rest density from a perfect lattice at the particle separation.
            g = np.arange(-3, 4) * p.particlesep
            lattice = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
            st['rho0'] = float(np.sum(_poly6(np.einsum('ij,ij->i', lattice, lattice), h)))
            w = _poly6(np.einsum('ij,ij->i', lattice, lattice), h)
            # Fraction of the rest density present at distance d from a wall (lattice behind the wall missing).
            st['wall_d'] = np.linspace(0.0, h, 33)
            st['wall_f'] = np.array([w[lattice[:, 1] >= -d - 1e-9].sum() / w.sum() for d in st['wall_d']])
            r = np.linalg.norm(lattice, axis=1)
            gr = _spiky_grad(-lattice, r, h) / st['rho0']
            st['grad2_rest'] = float(np.sum(np.einsum('ij,ij->i', gr, gr)) + np.sum(np.sum(gr, axis=0) ** 2))
            state['sim'] = st
            state['frames'] = {start: geo.P.copy()}
            state['vel'] = {start: np.zeros_like(geo.P)}
            state['last'] = start
        if frame <= start or geo.num_points == 0:
            return geo.copy()
        frames = state['frames']
        if frame not in frames:
            if frame < state['last']:
                state.clear()
                return self.cook(inputs, p, ctx)
            st = state['sim']
            sub = max(int(p.substeps), 1)
            dt = 1.0 / ctx.fps / sub
            for f in range(state['last'] + 1, frame + 1):
                for _ in range(sub):
                    pbf_step(st, dt, p)
                frames[f] = st['P'].copy()
                state['vel'][f] = st['V'].copy()
                state['last'] = f
        out = Geometry()
        out.P = frames[frame]
        out.attribs[POINT]['v'] = state['vel'][frame]
        out.attribs[POINT]['pscale'] = np.full(out.num_points, p.particlesep * 0.5)
        return out
