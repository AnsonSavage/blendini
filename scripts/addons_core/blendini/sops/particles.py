# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""POP-style particle solver: emission from points or surfaces, forces, curl noise turbulence,
ground and collider bounces. Fully vectorized with numpy.

Shared helpers for stateful particle SOPs live here too (frame cache with a memory budget,
parameter keys, uniform sphere sampling).
"""

import numpy as np

from .base import SOP, Parm, register_sop, CookError, upstream_changed
from ..core.geometry import Geometry, POINT, PRIM, DETAIL
from ..core.noise import curlnoise

# Upper bound on the number of floats kept in a solver's frame cache (~160 MB).
CACHE_BUDGET = 20_000_000
MAX_PARTICLES_LIMIT = 5_000_000


def _in(inputs, i=0):
    if len(inputs) > i and inputs[i] is not None:
        return inputs[i]
    return Geometry()


def parm_key(p):
    """Hashable key of all parameter values (restart the simulation when it changes)."""
    def norm(v):
        if isinstance(v, (list, tuple)):
            return tuple(norm(x) for x in v)
        if isinstance(v, np.ndarray):
            return tuple(v.ravel().tolist())
        return v
    return tuple(sorted((k, norm(v)) for k, v in p.items()))


def geo_sig(geo):
    """Cheap signature of a geometry for upstream_changed (counts + position sum)."""
    if geo is None or geo.num_points == 0:
        return (0, 0, 0.0)
    return (geo.num_points, geo.num_prims, float(geo.P.sum()))


def cache_store(state, frame, data):
    """Store a frame (dict of arrays) and drop the oldest frames past the memory budget.
    The start frame is never dropped."""
    frames = state['frames']
    frames[frame] = data
    size = state.get('cache_size', 0) + sum(int(np.size(a)) for a in data.values() if isinstance(a, np.ndarray))
    start = state.get('start')
    while size > CACHE_BUDGET and len(frames) > 2:
        old = min(f for f in frames if f != start and f != frame)
        size -= sum(int(np.size(a)) for a in frames.pop(old).values() if isinstance(a, np.ndarray))
    state['cache_size'] = size


def random_in_sphere(rng, n):
    """n uniform samples inside the unit ball (n, 3)."""
    d = rng.normal(size=(n, 3))
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
    return d * rng.random(n)[:, None] ** (1.0 / 3.0)


def collide_with(collider, P, V, radius, bounce, friction):
    """Push points out of a _Collider and reflect their velocity along the push direction."""
    if collider is None or collider.tree is None or len(P) == 0:
        return P, V
    newP = collider.push_out(P, radius)
    moved = newP - P
    dist = np.linalg.norm(moved, axis=1)
    hit = dist > 1e-9
    if hit.any():
        n = moved[hit] / dist[hit, None]
        v = V[hit]
        vn = np.einsum('ij,ij->i', v, n)
        vt = v - vn[:, None] * n
        vn_new = np.where(vn < 0, -vn * bounce, vn)
        V = V.copy()
        V[hit] = vt * (1.0 - friction) + vn_new[:, None] * n
    return newP, V


class Emitter:
    """Samples birth positions (and interpolated attributes) on points or polygon surfaces."""

    def __init__(self, geo, vel=None):
        self.geo = geo
        pa = geo.attribs[POINT]
        n = geo.num_points
        self.v = vel if vel is not None else (pa['v'] if 'v' in pa and pa['v'].shape == (n, 3) else None)
        self.N = pa['N'] if 'N' in pa and pa['N'].shape == (n, 3) else None
        self.Cd_point = pa['Cd'] if 'Cd' in pa and pa['Cd'].shape == (n, 3) else None
        ca = geo.attribs[PRIM].get('Cd')
        self.Cd_prim = ca if ca is not None and ca.shape == (geo.num_prims, 3) else None
        cd = geo.attribs[DETAIL].get('Cd')
        self.Cd_detail = np.asarray(cd, dtype=np.float64).reshape(-1)[:3] if cd is not None and np.size(cd) >= 3 else None
        self.tris = np.zeros((0, 3), np.int64)
        if geo.num_prims:
            tris, tri_prim = geo.triangulate_flat()
            if len(tris):
                P = geo.P
                a, b, c = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
                cr = np.cross(b - a, c - a)
                area = np.linalg.norm(cr, axis=1)
                if area.sum() > 0:
                    self.tris, self.tri_prim = tris, tri_prim
                    self.cdf = np.cumsum(area) / area.sum()
                    self.tri_n = cr / np.maximum(area, 1e-20)[:, None]

    @property
    def has_colour(self):
        return self.Cd_point is not None or self.Cd_prim is not None or self.Cd_detail is not None

    def sample(self, rng, count):
        """Returns positions, emitter velocity, emitter normal, colour (or None) for count births."""
        geo = self.geo
        if len(self.tris):
            t = np.minimum(np.searchsorted(self.cdf, rng.random(count) * self.cdf[-1]), len(self.tris) - 1)
            u, w = rng.random(count), rng.random(count)
            flip = u + w > 1
            u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
            bary = np.stack([1 - u - w, u, w], axis=1)
            idx = self.tris[t]

            def interp(arr):
                return np.einsum('ij,ijk->ik', bary, arr[idx])
            P = interp(geo.P)
            v = interp(self.v) if self.v is not None else np.zeros((count, 3))
            N = interp(self.N) if self.N is not None else self.tri_n[t]
            if self.Cd_point is not None:
                cd = interp(self.Cd_point)
            elif self.Cd_prim is not None:
                cd = self.Cd_prim[self.tri_prim[t]]
            else:
                cd = None
        else:
            i = rng.integers(0, geo.num_points, count)
            P = geo.P[i]
            v = self.v[i] if self.v is not None else np.zeros((count, 3))
            N = self.N[i] if self.N is not None else np.zeros((count, 3))
            cd = self.Cd_point[i] if self.Cd_point is not None else None
        if cd is None and self.Cd_detail is not None:
            cd = np.broadcast_to(self.Cd_detail, (count, 3)).copy()
        return P, v, N, cd


_FIELDS = ('P', 'v', 'age', 'life', 'id', 'pscale', 'Cd')


@register_sop
class POPSolverSOP(SOP):
    id = 'popsolver'
    label = "POP Solver"
    category = 'Simulation'
    inputs = 2
    input_labels = ("Emitter", "Colliders (optional)")
    icon = 'PARTICLES'
    time_dependent = True
    description = ("Particle simulation: births from the emitter's points (or scattered on its surface), gravity, "
                   "drag, wind, curl noise, ground and collider bounces. Outputs points with v, age, life, id")
    parms = (
        Parm('startframe', "Start Frame", 'int', default=1),
        Parm('substeps', "Substeps", 'int', default=1, min=1, max=20),
        Parm('seed', "Seed", 'int', default=0),
        Parm('birthrate', "Birth Rate (per second)", 'float', default=500.0, min=0.0, soft_max=10000.0),
        Parm('maxparticles', "Max Particles", 'int', default=100000, min=0, max=MAX_PARTICLES_LIMIT),
        Parm('life', "Life Expectancy", 'float', default=3.0, min=0.0, soft_max=20.0, description="Seconds"),
        Parm('lifevar', "Life Variance", 'float', default=0.5, min=0.0, soft_max=10.0),
        Parm('pscale', "Particle Scale", 'float', default=0.02, min=0.0, soft_max=0.5),
        Parm('initvel', "Initial Velocity", 'vector', default=(0.0, 0.0, 0.0)),
        Parm('velvar', "Velocity Variance", 'vector', default=(0.2, 0.2, 0.2),
             description="Random velocity added per axis (uniform in +- variance)"),
        Parm('inheritv', "Inherit Velocity", 'float', default=1.0, soft_min=0.0, soft_max=2.0,
             description="Scale of the emitter's v (or its motion when it has no v)"),
        Parm('normalvel', "Normal Velocity", 'float', default=1.0, soft_min=-5.0, soft_max=5.0,
             description="Speed along the emitter normal N (surface normal when scattering)"),
        Parm('gravity', "Gravity", 'vector', default=(0.0, -9.81, 0.0)),
        Parm('drag', "Air Drag", 'float', default=0.1, min=0.0, soft_max=10.0,
             description="Velocity relaxes toward the wind at this rate (per second)"),
        Parm('wind', "Wind", 'vector', default=(0.0, 0.0, 0.0)),
        Parm('turbamp', "Turbulence Amplitude", 'float', default=0.0, min=0.0, soft_max=20.0),
        Parm('turbsize', "Swirl Size", 'float', default=1.0, min=0.001, soft_max=10.0),
        Parm('turbpulse', "Turbulence Pulse", 'float', default=0.2, min=0.0, soft_max=5.0,
             description="How fast the noise field evolves"),
        Parm('ground', "Ground Plane", 'bool', default=True),
        Parm('groundy', "Ground Height", 'float', default=0.0),
        Parm('bounce', "Bounce", 'float', default=0.5, min=0.0, max=1.0),
        Parm('friction', "Friction", 'float', default=0.2, min=0.0, max=1.0),
        Parm('killground', "Kill on Ground Hit", 'bool', default=False),
    )

    # -- simulation ------------------------------------------------------

    def _forces(self, s, dt, p, t):
        P, V = s['P'], s['v']
        if len(P) == 0:
            return
        g = np.asarray(p.gravity, dtype=np.float64)
        wind = np.asarray(p.wind, dtype=np.float64)
        V += g * dt
        if p.drag > 0:
            V += (wind - V) * (1.0 - np.exp(-p.drag * dt))
        if p.turbamp > 0:
            q = P / max(p.turbsize, 1e-6) + np.array([0.0, 0.0, t * p.turbpulse])
            V += curlnoise(q) * (p.turbamp * dt)
        P += V * dt

    def _collide(self, s, p, collider, dt):
        P, V = s['P'], s['v']
        if len(P) == 0:
            return None
        killed = None
        if p.ground:
            below = P[:, 1] < p.groundy
            if below.any():
                if p.killground:
                    killed = below
                P[below, 1] = p.groundy
                vy = V[below, 1]
                vy = np.where(vy < 0, -vy * p.bounce, vy)
                # Resting contact: no micro bounces under gravity.
                g = abs(float(np.asarray(p.gravity, dtype=np.float64)[1]))
                vy = np.where(vy < 2.0 * g * dt, 0.0, vy)
                V[below, 1] = vy
                V[below, 0] *= (1.0 - p.friction)
                V[below, 2] *= (1.0 - p.friction)
        if collider is not None:
            s['P'], s['v'] = collide_with(collider, P, V, 0.0, p.bounce, p.friction)
        return killed

    def _emit(self, s, state, emitter, p, frame, frame_dt):
        acc = state.get('birth_acc', 0.0) + max(p.birthrate, 0.0) * frame_dt
        count = int(np.floor(acc))
        state['birth_acc'] = acc - count
        alive = len(s['P'])
        count = max(min(count, int(p.maxparticles) - alive, MAX_PARTICLES_LIMIT - alive), 0)
        if count == 0 or emitter is None or emitter.geo.num_points == 0:
            return
        rng = np.random.default_rng([int(p.seed) & 0x7FFFFFFF, frame & 0x7FFFFFFF, 7])
        P, ev, N, cd = emitter.sample(rng, count)
        vel = (np.asarray(p.initvel, dtype=np.float64) + ev * p.inheritv + N * p.normalvel
               + (rng.random((count, 3)) * 2 - 1) * np.asarray(p.velvar, dtype=np.float64))
        # Born during the frame: spread over the sub-frame to avoid bands of particles.
        age = rng.random(count) * frame_dt
        g = np.asarray(p.gravity, dtype=np.float64)
        P = P + vel * age[:, None] + 0.5 * g * (age ** 2)[:, None]
        vel = vel + g * age[:, None]
        life = np.maximum(p.life + (rng.random(count) * 2 - 1) * p.lifevar, 1e-3)
        ids = np.arange(state['next_id'], state['next_id'] + count, dtype=np.int64)
        state['next_id'] += count
        new = {'P': P, 'v': vel, 'age': age, 'life': life, 'id': ids, 'pscale': np.full(count, float(p.pscale))}
        if s.get('Cd') is not None:
            new['Cd'] = cd if cd is not None else np.ones((count, 3))
        if p.ground:
            new['P'][:, 1] = np.maximum(new['P'][:, 1], p.groundy)
        for k, a in new.items():
            s[k] = np.concatenate([s[k], a])

    def cook(self, inputs, p, ctx):
        from .vellum import _Collider
        geo = _in(inputs, 0)
        col = inputs[1] if len(inputs) > 1 and inputs[1] is not None else None
        state = ctx.state
        frame = int(round(ctx.frame))
        start = int(p.startframe)
        if int(p.maxparticles) > MAX_PARTICLES_LIMIT:
            raise CookError("Max Particles is limited to %d" % MAX_PARTICLES_LIMIT)
        key = parm_key(p)
        sig = (geo_sig(geo), geo_sig(col))
        edited = state.get('key') == key and upstream_changed(state, frame, sig)
        if state.get('key') != key or frame <= start or 'sim' not in state or edited:
            state.clear()
            state['key'] = key
            state['start'] = start
            upstream_changed(state, frame, sig)
            has_cd = Emitter(geo).has_colour if geo.num_points else False
            s = {'P': np.zeros((0, 3)), 'v': np.zeros((0, 3)), 'age': np.zeros(0), 'life': np.zeros(0),
                 'id': np.zeros(0, np.int64), 'pscale': np.zeros(0), 'Cd': np.zeros((0, 3)) if has_cd else None}
            state['sim'] = s
            state['next_id'] = 0
            state['birth_acc'] = 0.0
            state['frames'] = {}
            state['last'] = start
            cache_store(state, start, {k: v.copy() for k, v in s.items() if v is not None})
        frames = state['frames']
        if frame <= start:
            return self._output(frames[start])
        if frame in frames:
            return self._output(frames[frame])
        if frame < state['last']:
            # Scrubbed back past the cache (or the frame was dropped): restart.
            state.clear()
            return self.cook(inputs, p, ctx)
        s = state['sim']
        # Emitter velocity from its motion when it carries no v.
        vel = None
        prev = state.get('prev_emit')
        if geo.num_points and 'v' not in geo.attribs[POINT] and prev is not None and prev[0] == frame - 1 \
                and prev[1].shape == geo.P.shape:
            vel = (geo.P - prev[1]) * ctx.fps
        state['prev_emit'] = (frame, geo.P.copy())
        emitter = Emitter(geo, vel) if geo.num_points else None
        collider = _Collider(col) if col is not None and col.num_prims else None
        frame_dt = 1.0 / ctx.fps
        sub = max(int(p.substeps), 1)
        dt = frame_dt / sub
        for f in range(state['last'] + 1, frame + 1):
            t0 = (f - 1 - start) * frame_dt
            for k in range(sub):
                self._forces(s, dt, p, t0 + k * dt)
                killed = self._collide(s, p, collider, dt)
                if killed is not None and killed.any():
                    keep = ~killed
                    for name in _FIELDS:
                        if s.get(name) is not None:
                            s[name] = s[name][keep]
            s['age'] = s['age'] + frame_dt
            self._emit(s, state, emitter, p, f, frame_dt)
            keep = s['age'] < s['life']
            if not keep.all():
                for name in _FIELDS:
                    if s.get(name) is not None:
                        s[name] = s[name][keep]
            cache_store(state, f, {k: v.copy() for k, v in s.items() if v is not None})
            state['last'] = f
        return self._output(frames[frame])

    @staticmethod
    def _output(data):
        out = Geometry()
        out.P = data['P']
        pa = out.attribs[POINT]
        for name in ('v', 'age', 'life', 'id', 'pscale', 'Cd'):
            if name in data:
                pa[name] = data[name].copy()
        return out
