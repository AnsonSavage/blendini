# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Pyro-style smoke: a grid fluid solver (Stam's stable fluids) on Blendini's native volumes.

Collocated grid, semi-Lagrangian advection, buoyancy from temperature, vorticity confinement,
curl noise turbulence and a Jacobi pressure projection. Outputs density, temperature and vel
volumes (fog volumes render as smoke).
"""

import numpy as np

from .base import SOP, Parm, register_sop, CookError, upstream_changed
from ..core.geometry import Geometry, POINT
from ..core.volume import Volume, winding_numbers
from .modify import _in


def _sample(field, x):
    """Trilinear sample of a grid field at index space positions x (N, 3) (clamped)."""
    res = np.array(field.shape[:3])
    x = np.clip(x, 0, res - 1.000001)
    i0 = np.floor(x).astype(np.int64)
    t = x - i0
    i1 = np.minimum(i0 + 1, res - 1)
    out = 0.0
    for dx, ix in ((0, i0[:, 0]), (1, i1[:, 0])):
        wx = t[:, 0] if dx else 1 - t[:, 0]
        for dy, iy in ((0, i0[:, 1]), (1, i1[:, 1])):
            wy = t[:, 1] if dy else 1 - t[:, 1]
            for dz, iz in ((0, i0[:, 2]), (1, i1[:, 2])):
                wz = t[:, 2] if dz else 1 - t[:, 2]
                w = wx * wy * wz
                v = field[ix, iy, iz]
                out = out + (w[:, None] * v if v.ndim == 2 else w * v)
    return out


def _grad(f, h):
    g = np.zeros(f.shape + (3,))
    for a in range(3):
        g[..., a] = np.gradient(f, h, axis=a)
    return g


def _curl(v, h):
    dvz_dy = np.gradient(v[..., 2], h, axis=1)
    dvy_dz = np.gradient(v[..., 1], h, axis=2)
    dvx_dz = np.gradient(v[..., 0], h, axis=2)
    dvz_dx = np.gradient(v[..., 2], h, axis=0)
    dvy_dx = np.gradient(v[..., 1], h, axis=0)
    dvx_dy = np.gradient(v[..., 0], h, axis=1)
    return np.stack([dvz_dy - dvy_dz, dvx_dz - dvz_dx, dvy_dx - dvx_dy], axis=-1)


def _set_walls(v):
    v[0], v[-1] = 0.0, 0.0
    v[:, 0], v[:, -1] = 0.0, 0.0
    v[:, :, 0], v[:, :, -1] = 0.0, 0.0


def project(vel, h, iterations, solid=None):
    """Make the velocity divergence free (Jacobi iterations on the pressure Poisson equation)."""
    div = (np.gradient(vel[..., 0], h, axis=0) + np.gradient(vel[..., 1], h, axis=1)
           + np.gradient(vel[..., 2], h, axis=2))
    p = np.zeros(div.shape)
    for _ in range(iterations):
        pp = np.pad(p, 1, mode='edge')
        p = (pp[2:, 1:-1, 1:-1] + pp[:-2, 1:-1, 1:-1] + pp[1:-1, 2:, 1:-1] + pp[1:-1, :-2, 1:-1]
             + pp[1:-1, 1:-1, 2:] + pp[1:-1, 1:-1, :-2] - div * h * h) / 6.0
    vel = vel - _grad(p, h)
    if solid is not None:
        vel[solid] = 0.0
    _set_walls(vel)
    return vel


@register_sop
class PyroSolverSOP(SOP):
    id = 'pyrosolver'
    label = "Pyro Solver (Smoke)"
    category = 'Simulation'
    inputs = 2
    input_labels = ("Source (points, polygons or density volume)", "Colliders (optional)")
    icon = 'FORCE_FLUIDFLOW'
    time_dependent = True
    description = "Grid smoke solver: buoyant density and temperature with vorticity and turbulence (renders as smoke)"
    parms = (
        Parm('startframe', "Start Frame", 'int', default=1),
        Parm('size', "Container Size", 'vector', default=(2.0, 3.0, 2.0)),
        Parm('t', "Container Center", 'vector', default=(0.0, 1.5, 0.0)),
        Parm('voxelsize', "Voxel Size", 'float', default=0.06, min=0.01, soft_max=0.5),
        Parm('substeps', "Substeps", 'int', default=1, min=1, max=10),
        Parm('pressureiters', "Pressure Iterations", 'int', default=25, min=1, max=200),
        Parm('sourcedensity', "Source Density", 'float', default=4.0, min=0.0, soft_max=20.0),
        Parm('sourcetemp', "Source Temperature", 'float', default=2.0, min=0.0, soft_max=10.0),
        Parm('sourceradius', "Point Source Radius", 'float', default=0.15, min=0.001, soft_max=1.0,
             description="Multiplied by pscale, for point sources"),
        Parm('buoyancy', "Buoyancy Lift", 'float', default=2.5, soft_min=0.0, soft_max=10.0),
        Parm('weight', "Density Weight", 'float', default=0.1, soft_min=0.0, soft_max=2.0),
        Parm('cooling', "Cooling Rate", 'float', default=0.6, min=0.0, soft_max=5.0),
        Parm('dissipation', "Dissipation", 'float', default=0.05, min=0.0, max=1.0,
             description="Fraction of density lost per second"),
        Parm('vorticity', "Vorticity Confinement", 'float', default=0.3, min=0.0, soft_max=2.0),
        Parm('turbulence', "Turbulence", 'float', default=0.5, min=0.0, soft_max=5.0),
        Parm('turbscale', "Turbulence Swirl Size", 'float', default=0.4, min=0.01, soft_max=2.0),
        Parm('wind', "Wind", 'vector'),
        Parm('opentop', "Open Top", 'bool', default=True,
             description="Smoke leaves through the top of the container instead of collecting there"),
        Parm('fire', "Combustion (Fire)", 'bool', default=False,
             description="Sources emit fuel that burns into heat, flame and smoke"),
        Parm('fuel', "Source Fuel", 'float', default=3.0, min=0.0, soft_max=10.0, hide_when={'fire': False}),
        Parm('burnrate', "Burn Rate", 'float', default=25.0, min=0.0, soft_max=60.0, hide_when={'fire': False}),
        Parm('ignition', "Ignition Temperature", 'float', default=0.3, min=0.0, soft_max=2.0, hide_when={'fire': False}),
        Parm('heat', "Heat Release", 'float', default=0.6, min=0.0, soft_max=10.0, hide_when={'fire': False}),
        Parm('smoke', "Smoke from Burning", 'float', default=0.6, min=0.0, soft_max=5.0, hide_when={'fire': False}),
        Parm('flamecooling', "Flame Height (cooling)", 'float', default=15.0, min=0.1, soft_max=50.0,
             hide_when={'fire': False}, description="Higher values give shorter flames"),
    )

    def _grid(self, p):
        half = np.abs(np.asarray(p.size, dtype=np.float64)) / 2
        c = np.asarray(p.t, dtype=np.float64)
        return Volume.from_bounds('density', c - half, c + half, p.voxelsize)

    def _source(self, geo, grid, p, state):
        """Per-voxel source strength (0-1), cached while the source geometry doesn't change."""
        key = (geo.num_points, geo.num_prims, float(geo.P.sum()) if geo.num_points else 0.0, len(geo.volumes),
               p.sourceradius, tuple(grid.res), tuple(np.round(grid.origin, 9)))
        if state.get('src_key') == key:
            return state['src']
        res = tuple(grid.res)
        src = np.zeros(res)
        dens = next((v for v in geo.volumes if v.name == 'density' and not v.is_vector), None)
        if dens is not None:
            src = np.clip(dens.sample(grid.centers()).reshape(res), 0, None)
        elif geo.num_prims and np.any(geo.prim_closed):
            tris, _ = geo.triangulate_flat()
            if len(tris):
                lo, hi = geo.bbox()
                C = grid.centers()
                near = np.all((C >= lo - grid.voxelsize) & (C <= hi + grid.voxelsize), axis=1)
                inside = np.zeros(len(C), dtype=bool)
                if near.any():
                    inside[near] = winding_numbers(geo.P, tris, C[near]) > 0.5
                src = inside.reshape(res).astype(np.float64)
        elif geo.num_points:
            ps = geo.attribs[POINT].get('pscale')
            rad = (ps if ps is not None and ps.dtype != object else np.ones(geo.num_points)) * p.sourceradius
            vs = grid.voxelsize
            for c, r in zip(geo.P, rad):
                i0 = np.maximum(np.floor((c - r - grid.origin) / vs).astype(np.int64), 0)
                i1 = np.minimum(np.ceil((c + r - grid.origin) / vs).astype(np.int64), grid.res)
                if np.any(i1 <= i0):
                    continue
                g = np.meshgrid(*[np.arange(i0[k], i1[k]) for k in range(3)], indexing='ij')
                pos = grid.origin + (np.stack(g, axis=-1) + 0.5) * vs
                w = np.clip(1 - np.sum((pos - c) ** 2, axis=-1) / (r * r), 0, 1)
                sl = tuple(slice(i0[k], i1[k]) for k in range(3))
                src[sl] = np.maximum(src[sl], w)
        state['src_key'] = key
        state['src'] = src
        state['src_vel'] = self._source_velocity(geo, grid, src)
        return src

    @staticmethod
    def _source_velocity(geo, grid, src):
        """Velocity injected where the source is: a 'vel' volume, else the mean point 'v'."""
        vel = next((v for v in geo.volumes if v.name == 'vel' and v.is_vector), None)
        if vel is not None:
            return vel.sample(grid.centers()).reshape(tuple(grid.res) + (3,))
        v = geo.attribs[POINT].get('v')
        if v is not None and v.dtype != object and np.asarray(v).ndim == 2 and len(v):
            mean = np.asarray(v, dtype=np.float64).mean(axis=0)
            if np.linalg.norm(mean) > 1e-9:
                return np.broadcast_to(mean, tuple(grid.res) + (3,)).copy()
        return None

    def _solid(self, geo, grid):
        if geo is None or geo.num_prims == 0:
            return None
        tris, _ = geo.triangulate_flat()
        if not len(tris):
            return None
        C = grid.centers()
        lo, hi = geo.bbox()
        near = np.all((C >= lo - grid.voxelsize) & (C <= hi + grid.voxelsize), axis=1)
        solid = np.zeros(len(C), dtype=bool)
        if near.any():
            solid[near] = winding_numbers(geo.P, tris, C[near]) > 0.5
        return solid.reshape(tuple(grid.res))

    def step(self, st, src, dt, p, src_vel=None):
        h = st['h']
        d, T, v = st['density'], st['temperature'], st['vel']
        if src_vel is not None:
            w = np.clip(src, 0, 1)[..., None]
            v[:] = v * (1 - w) + src_vel * w
        solid = st.get('solid')
        # Sources.
        if p.fire:
            fuel, flame = st['fuel'], st['flame']
            fuel = np.maximum(fuel, src * p.fuel)
            T = np.maximum(T, src * p.ignition * 1.5)
            # Combustion: hot fuel burns into heat, flame and smoke.
            burn = np.where(T > p.ignition, np.minimum(fuel, fuel * p.burnrate * dt), 0.0)
            fuel = fuel - burn
            T = T + burn * p.heat
            d = d + burn * p.smoke
            flame = np.maximum(flame * max(0.0, 1.0 - p.flamecooling * dt), np.clip(burn / max(dt, 1e-9) * 0.25, 0, 1))
        else:
            d = np.maximum(d, src * p.sourcedensity) if p.sourcedensity > 0 else d
            T = np.maximum(T, src * p.sourcetemp)
        # Forces.
        v[..., 1] += (p.buoyancy * T - p.weight * d) * dt
        wind = np.asarray(p.wind, dtype=np.float64)
        if np.any(wind):
            v += (wind - v) * min(1.0, 0.5 * dt) * (d[..., None] > 1e-3)
        if p.vorticity > 0:
            w = _curl(v, h)
            mag = np.linalg.norm(w, axis=-1)
            N = _grad(mag, h)
            N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-9)
            v += p.vorticity * h * np.cross(N, w) * dt
        if p.fire:
            mask_src = d + st['flame']
        else:
            mask_src = d
        if p.turbulence > 0:
            from ..core import noise as _noise
            # The swirls are much bigger than a voxel: evaluate the curl noise on a coarser grid
            # (only where there is smoke) and interpolate.
            k = max(1, int(p.turbscale / (3.0 * h)))
            res = np.array(d.shape)
            cres = tuple(np.maximum((res + k - 1) // k, 1))
            cidx = np.indices(cres).reshape(3, -1).T.astype(np.float64)
            cpos = st['origin'] + (cidx * k + 0.5) * h
            cd = mask_src[::k, ::k, ::k].reshape(-1) > 1e-4
            coarse = np.zeros((len(cidx), 3))
            if cd.any():
                coarse[cd] = _noise.curlnoise(cpos[cd] / max(p.turbscale, 1e-3)
                                              + np.array([0.0, -0.3, 0.0]) * st['time'])
            curl = _sample(coarse.reshape(cres + (3,)), st['idx'] / k)
            mask = (mask_src > 1e-3).reshape(-1, 1)
            v += (curl * mask).reshape(v.shape) * p.turbulence * dt
        if solid is not None:
            v[solid] = 0.0
        _set_walls(v)
        v = project(v, h, p.pressureiters, solid)
        # Advection (semi-Lagrangian, index space).
        idx = st['idx']
        back = idx - v.reshape(-1, 3) * (dt / h)
        v = _sample(v, back).reshape(v.shape)
        d = _sample(d, back).reshape(d.shape)
        T = _sample(T, back).reshape(T.shape)
        if p.fire:
            st['fuel'] = _sample(fuel, back).reshape(fuel.shape)
            st['flame'] = _sample(flame, back).reshape(flame.shape)
        v = project(v, h, max(p.pressureiters // 2, 1), solid)
        d *= max(0.0, 1.0 - p.dissipation * dt)
        T *= max(0.0, 1.0 - p.cooling * dt)
        if solid is not None:
            d[solid] = 0.0
        if p.opentop:
            # Outflow: fade density and heat in the top cells.
            fade = np.linspace(1.0, 0.0, 4)[None, :, None]
            d[:, -4:, :] *= fade
            T[:, -4:, :] *= fade
            if p.fire:
                st['flame'][:, -4:, :] *= fade
                st['fuel'][:, -4:, :] *= fade
        st['density'], st['temperature'], st['vel'] = d, T, v
        st['time'] += dt

    def cook(self, inputs, p, ctx):
        src_geo = _in(inputs, 0)
        col_geo = inputs[1] if len(inputs) > 1 else None
        state = ctx.state
        frame = int(round(ctx.frame))
        start = int(p.startframe)
        key = tuple(sorted((k, tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in p.items()))
        # Collider topology only: animated colliders move without restarting the simulation.
        colkey = (col_geo.num_points, col_geo.num_prims) if col_geo is not None and col_geo.num_points else None
        try:
            grid = self._grid(p)
        except ValueError as ex:
            raise CookError(str(ex))
        sig = (src_geo.num_points, float(src_geo.P.sum()) if src_geo.num_points else 0.0,
               sum(float(v.data.sum()) for v in src_geo.volumes),
               float(col_geo.P.sum()) if col_geo is not None and col_geo.num_points else 0.0)
        edited = state.get('key') == (key, colkey) and upstream_changed(state, frame, sig)
        if state.get('key') != (key, colkey) or frame <= start or 'sim' not in state or edited:
            state.clear()
            state['key'] = (key, colkey)
            res = tuple(grid.res)
            st = {'h': grid.voxelsize, 'density': np.zeros(res), 'temperature': np.zeros(res),
                  'vel': np.zeros(res + (3,)), 'time': 0.0, 'origin': grid.origin.copy(),
                  'idx': np.indices(res).reshape(3, -1).T.astype(np.float64),
                  'fuel': np.zeros(res), 'flame': np.zeros(res)}
            st['solid'] = self._solid(col_geo, grid)
            st['colsig'] = float(col_geo.P.sum()) if col_geo is not None and col_geo.num_points else None
            state['sim'] = st
            state['frames'] = {}
            state['last'] = start
        out = Geometry()
        if frame <= start:
            return out
        frames = state['frames']
        if frame not in frames:
            if frame < state['last']:
                state.clear()
                return self.cook(inputs, p, ctx)
            st = state['sim']
            colsig = float(col_geo.P.sum()) if col_geo is not None and col_geo.num_points else None
            if colsig != st.get('colsig'):
                st['solid'] = self._solid(col_geo, grid)
                st['colsig'] = colsig
            src = self._source(src_geo, grid, p, state)
            sub = max(int(p.substeps), 1)
            dt = 1.0 / ctx.fps / sub
            for f in range(state['last'] + 1, frame + 1):
                for _ in range(sub):
                    self.step(st, src, dt, p, state.get('src_vel'))
                frames[f] = (st['density'].astype(np.float32), st['temperature'].astype(np.float32),
                             st['vel'].astype(np.float32), st['flame'].astype(np.float32) if p.fire else None)
                state['last'] = f
                # Keep memory bounded: drop the oldest cached frames.
                if len(frames) > 240:
                    del frames[min(frames)]
        if frame not in frames:
            state.clear()
            return self.cook(inputs, p, ctx)
        d, T, v, fl = frames[frame]
        out.volumes = [Volume('density', d.astype(np.float64), grid.origin, grid.voxelsize),
                       Volume('temperature', T.astype(np.float64), grid.origin, grid.voxelsize),
                       Volume('vel', v.astype(np.float64), grid.origin, grid.voxelsize)]
        if fl is not None:
            out.volumes.append(Volume('flame', fl.astype(np.float64), grid.origin, grid.voxelsize))
        return out
