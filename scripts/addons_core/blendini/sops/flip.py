# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""FLIP liquid (Zhu & Bridson 2005): particles carry velocity, a grid enforces incompressibility.

Collocated grid with trilinear particle/grid transfers, a free-surface Jacobi pressure solve
(pressure 0 in air cells, solid container walls), a FLIP/PIC blend and a particle separation
pass that keeps the particle spacing (volume) stable.
"""

import numpy as np

from .base import SOP, Parm, register_sop, CookError, upstream_changed
from ..core.geometry import Geometry, POINT
from .fluid import neighbour_pairs
from .modify import _in


def _corners(x, res):
    """Trilinear stencil for index space positions: (8 flat indices, 8 weights), each (n,)."""
    x = np.clip(x, 0, res - 1.000001)
    i0 = np.floor(x).astype(np.int64)
    t = x - i0
    out = []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                ix = np.minimum(i0[:, 0] + dx, res[0] - 1)
                iy = np.minimum(i0[:, 1] + dy, res[1] - 1)
                iz = np.minimum(i0[:, 2] + dz, res[2] - 1)
                w = ((t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1])
                     * (t[:, 2] if dz else 1 - t[:, 2]))
                out.append(((ix * res[1] + iy) * res[2] + iz, w))
    return out


def p2g(x, v, res):
    n_cells = int(np.prod(res))
    num = np.zeros((n_cells, 3))
    den = np.zeros(n_cells)
    for idx, w in _corners(x, res):
        den += np.bincount(idx, weights=w, minlength=n_cells)
        for k in range(3):
            num[:, k] += np.bincount(idx, weights=w * v[:, k], minlength=n_cells)
    vel = num / np.maximum(den, 1e-9)[:, None]
    return vel.reshape(tuple(res) + (3,)), den.reshape(tuple(res))


def g2p(field, x, res):
    flat = field.reshape(-1, field.shape[-1]) if field.ndim == 4 else field.reshape(-1)
    out = 0.0
    for idx, w in _corners(x, res):
        out = out + (w[:, None] * flat[idx] if flat.ndim == 2 else w * flat[idx])
    return out


def _walls(vel):
    """Solid container: no flow through the walls (normal velocity zero in the border cells)."""
    vel[0, ..., 0] = 0
    vel[-1, ..., 0] = 0
    vel[:, 0, :, 1] = np.maximum(vel[:, 0, :, 1], 0)
    vel[:, -1, :, 1] = np.minimum(vel[:, -1, :, 1], 0)
    vel[:, :, 0, 2] = 0
    vel[:, :, -1, 2] = 0


def pressure_project(vel, fluid, h, iterations):
    div = np.zeros(fluid.shape)
    for a in range(3):
        div += np.gradient(vel[..., a], h, axis=a)
    div *= fluid
    p = np.zeros(fluid.shape)
    f = fluid.astype(np.float64)
    for _ in range(iterations):
        # Walls (outside the grid) are Neumann: they mirror the cell's own pressure.
        pp = np.pad(p, 1, mode='edge')
        p = (pp[2:, 1:-1, 1:-1] + pp[:-2, 1:-1, 1:-1] + pp[1:-1, 2:, 1:-1] + pp[1:-1, :-2, 1:-1]
             + pp[1:-1, 1:-1, 2:] + pp[1:-1, 1:-1, :-2] - div * h * h) / 6.0
        p *= f    # free surface: air cells have zero pressure
    for a in range(3):
        vel[..., a] -= np.gradient(p, h, axis=a)
    return vel


@register_sop
class FLIPSolverSOP(SOP):
    id = 'flipsolver'
    label = "FLIP Solver"
    category = 'Simulation'
    inputs = 2
    input_labels = ("Fluid Particles", "Colliders (optional)")
    icon = 'MOD_FLUIDSIM'
    time_dependent = True
    description = "FLIP liquid in a box: feed it Points from Volume (particle separation should match)"
    parms = (
        Parm('startframe', "Start Frame", 'int', default=1),
        Parm('particlesep', "Particle Separation", 'float', default=0.08, min=0.005, soft_max=1.0),
        Parm('gridscale', "Grid Scale", 'float', default=2.0, min=1.0, max=4.0,
             description="Voxel size in particle separations"),
        Parm('flipratio', "FLIP / PIC Blend", 'float', default=0.95, min=0.0, max=1.0,
             description="1: lively (FLIP), 0: damped (PIC)"),
        Parm('gravity', "Gravity", 'float', default=9.81),
        Parm('substeps', "Max Substeps", 'int', default=4, min=1, max=30),
        Parm('cfl', "CFL Condition", 'float', default=1.0, min=0.1, max=5.0,
             description="Maximum voxels a particle may travel per substep"),
        Parm('pressureiters', "Pressure Iterations", 'int', default=100, min=1, max=400),
        Parm('separation', "Separation Strength", 'float', default=1.0, min=0.0, max=1.0),
        Parm('boxmin', "Container Min", 'vector', default=(-2.0, 0.0, -1.0)),
        Parm('boxmax', "Container Max", 'vector', default=(2.0, 3.0, 1.0)),
    )

    def step(self, st, dt, p):
        x, v = st['x'], st['v']
        h, origin, res = st['h'], st['origin'], st['res']
        xi = (x - origin) / h - 0.5
        vel, den = p2g(xi, v, res)
        fluid = den > 0.3
        old = vel.copy()
        vel[..., 1] -= p.gravity * dt * (den > 1e-6)
        _walls(vel)
        vel = pressure_project(vel, fluid, h, p.pressureiters)
        _walls(vel)
        v_pic = g2p(vel, xi, res)
        v_flip = v + g2p(vel - old, xi, res)
        v = p.flipratio * v_flip + (1 - p.flipratio) * v_pic
        x = x + v * dt
        # Particle separation: push apart particles closer than the separation.
        sep = p.particlesep
        if p.separation > 0 and len(x) > 1:
            I, J = neighbour_pairs(x, sep)
            if len(I):
                d = x[I] - x[J]
                dist = np.linalg.norm(d, axis=1, keepdims=True)
                push = d / np.maximum(dist, 1e-9) * (sep - dist) * 0.5 * p.separation
                corr = np.stack([np.bincount(I, weights=push[:, k], minlength=len(x)) for k in range(3)], axis=1)
                x = x + corr
        if st.get('collider') is not None:
            x = st['collider'].push_out(x, sep * 0.5)
        lo = np.asarray(p.boxmin, dtype=np.float64) + sep * 0.5
        hi = np.asarray(p.boxmax, dtype=np.float64) - sep * 0.5
        out_lo, out_hi = x < lo, x > hi
        x = np.clip(x, lo, hi)
        v = np.where(out_lo, np.maximum(v, 0), v)
        v = np.where(out_hi, np.minimum(v, 0), v)
        st['x'], st['v'] = x, v

    def cook(self, inputs, p, ctx):
        from .vellum import _Collider
        geo = _in(inputs, 0)
        col = inputs[1] if len(inputs) > 1 else None
        state = ctx.state
        frame = int(round(ctx.frame))
        start = int(p.startframe)
        key = (tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in p.items())),
               geo.num_points, (col.num_points, col.num_prims) if col is not None and col.num_points else None)
        sig = (float(geo.P.sum()) if geo.num_points else 0.0,
               float(col.P.sum()) if col is not None and col.num_points else 0.0)
        edited = state.get('key') == key and upstream_changed(state, frame, sig)
        if state.get('key') != key or frame <= start or 'sim' not in state or edited:
            state.clear()
            state['key'] = key
            lo = np.asarray(p.boxmin, dtype=np.float64)
            hi = np.asarray(p.boxmax, dtype=np.float64)
            h = p.particlesep * p.gridscale
            res = np.maximum(np.ceil((hi - lo) / h).astype(np.int64), 2)
            if np.prod(res) > 8_000_000:
                raise CookError("FLIP grid too large (%s cells): increase the particle separation or grid scale, "
                                "or shrink the container" % 'x'.join(str(int(r)) for r in res))
            v0 = geo.attribs[POINT].get('v')
            st = {'x': geo.P.copy(), 'h': h, 'origin': lo, 'res': res,
                  'v': np.array(v0, dtype=np.float64) if v0 is not None and v0.shape == geo.P.shape
                  else np.zeros_like(geo.P),
                  'collider': _Collider(col) if col is not None and col.num_prims else None}
            state['sim'] = st
            state['frames'] = {start: (geo.P.copy(), st['v'].copy())}
            state['last'] = start
        out = geo.copy()
        if frame <= start or geo.num_points == 0:
            if 'v' not in out.attribs[POINT]:
                out.attribs[POINT]['v'] = np.zeros((out.num_points, 3))
            out.attribs[POINT]['pscale'] = np.full(out.num_points, p.particlesep * 0.5)
            return out
        frames = state['frames']
        if frame not in frames:
            if frame < state['last']:
                state.clear()
                return self.cook(inputs, p, ctx)
            st = state['sim']
            if col is not None and col.num_prims:
                colsig = float(col.P.sum())
                if st.get('colsig') != colsig:
                    # Animated collider: follow its current position.
                    st['collider'] = _Collider(col)
                    st['colsig'] = colsig
            frame_dt = 1.0 / ctx.fps
            for f in range(state['last'] + 1, frame + 1):
                t = 0.0
                for _ in range(max(int(p.substeps), 1)):
                    vmax = float(np.abs(st['v']).max()) if len(st['v']) else 0.0
                    dt = min(frame_dt - t, p.cfl * st['h'] / max(vmax, 1e-6))
                    remaining_steps = max(int(p.substeps), 1) - _
                    dt = max(dt, (frame_dt - t) / remaining_steps)
                    self.step(st, dt, p)
                    t += dt
                    if t >= frame_dt - 1e-9:
                        break
                frames[f] = (st['x'].copy(), st['v'].copy())
                state['last'] = f
                if len(frames) > 400:
                    del frames[min(frames)]
        if frame not in frames:
            state.clear()
            return self.cook(inputs, p, ctx)
        P, V = frames[frame]
        out = Geometry()
        out.P = P
        out.attribs[POINT]['v'] = V
        out.attribs[POINT]['pscale'] = np.full(len(P), p.particlesep * 0.5)
        return out
