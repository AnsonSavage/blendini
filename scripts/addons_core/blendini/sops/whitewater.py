# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Whitewater for FLIP liquids: spray, foam and bubble points emitted where the liquid is fast and
near its surface (few neighbours) or accelerating strongly. Spray flies ballistically with drag,
foam rides the surface velocity, bubbles rise inside the liquid."""

import numpy as np

from .base import SOP, Parm, register_sop, CookError, upstream_changed
from ..core.geometry import Geometry, POINT
from .particles import parm_key, geo_sig, cache_store, random_in_sphere

MAX_WHITEWATER = 2_000_000
SPRAY, FOAM, BUBBLE = 0, 1, 2
_COLOURS = np.array([[0.85, 0.92, 1.0], [1.0, 1.0, 1.0], [0.6, 0.8, 1.0]])
_FIELDS = ('P', 'v', 'age', 'life', 'id', 'type')


def _in(inputs, i=0):
    if len(inputs) > i and inputs[i] is not None:
        return inputs[i]
    return Geometry()


class _CellGrid:
    """Sparse grid of liquid particles: per cell count and velocity sum. Neighbourhood queries sum
    the 3x3x3 cells around a position (about a 1.5 cell radius)."""

    def __init__(self, P, V, cell):
        self.cell = cell
        if len(P) == 0:
            self.keys = np.zeros(0, np.int64)
            return
        c = np.floor(P / cell).astype(np.int64)
        self.keys, inv = np.unique(self._key(c), return_inverse=True)
        self.count = np.bincount(inv, minlength=len(self.keys)).astype(np.float64)
        self.vsum = np.stack([np.bincount(inv, weights=V[:, k], minlength=len(self.keys)) for k in range(3)], 1)

    @staticmethod
    def _key(c):
        c = c + (1 << 20)
        return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]

    def query(self, X):
        n = len(X)
        cnt, vel = np.zeros(n), np.zeros((n, 3))
        if n == 0 or len(self.keys) == 0:
            return cnt, vel
        c = np.floor(X / self.cell).astype(np.int64)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    k = self._key(c + np.array([dx, dy, dz]))
                    i = np.minimum(np.searchsorted(self.keys, k), len(self.keys) - 1)
                    hit = self.keys[i] == k
                    cnt[hit] += self.count[i[hit]]
                    vel[hit] += self.vsum[i[hit]]
        vel /= np.maximum(cnt, 1)[:, None]
        return cnt, vel


@register_sop
class WhitewaterSOP(SOP):
    id = 'whitewater'
    label = "Whitewater"
    category = 'Simulation'
    inputs = 1
    input_labels = ("FLIP Particles (with v)",)
    icon = 'MOD_FLUIDSIM'
    time_dependent = True
    description = ("Spray / foam / bubble points from a FLIP simulation: emitted where the liquid is fast near its "
                   "surface or accelerates, then advected (type: 0 spray, 1 foam, 2 bubble)")
    parms = (
        Parm('startframe', "Start Frame", 'int', default=1),
        Parm('seed', "Seed", 'int', default=0),
        Parm('particlesep', "Particle Separation", 'float', default=0.08, min=0.005, soft_max=1.0,
             description="Of the FLIP particles; sets the neighbour search radius"),
        Parm('emitrate', "Emission Rate", 'float', default=20.0, min=0.0, soft_max=200.0,
             description="Whitewater points per second from a fully eligible liquid particle"),
        Parm('minspeed', "Min Speed", 'float', default=1.5, min=0.0, soft_max=10.0),
        Parm('maxspeed', "Max Speed", 'float', default=4.0, min=0.0, soft_max=20.0,
             description="Speed at which emission reaches the full rate"),
        Parm('maxneighbours', "Surface Neighbours", 'int', default=18, min=0, max=100,
             description="Liquid particles with fewer neighbours than this count as surface (about 27 inside)"),
        Parm('accelthreshold', "Acceleration Threshold", 'float', default=60.0, min=0.0, soft_max=500.0,
             description="Liquid particles accelerating more than this also emit (0 = off)"),
        Parm('life', "Life", 'float', default=1.5, min=0.0, soft_max=10.0),
        Parm('lifevar', "Life Variance", 'float', default=0.5, min=0.0, soft_max=5.0),
        Parm('jitter', "Velocity Jitter", 'float', default=0.3, min=0.0, soft_max=3.0),
        Parm('gravity', "Gravity", 'vector', default=(0.0, -9.81, 0.0)),
        Parm('drag', "Spray Drag", 'float', default=0.5, min=0.0, soft_max=10.0),
        Parm('foamdrag', "Foam Advection", 'float', default=5.0, min=0.0, soft_max=50.0,
             description="How fast foam and bubbles follow the liquid velocity (per second)"),
        Parm('buoyancy', "Bubble Buoyancy", 'float', default=6.0, min=0.0, soft_max=30.0),
        Parm('foamneighbours', "Bubble Neighbours", 'int', default=14, min=1, max=100,
             description="Whitewater with at least this many liquid neighbours is a bubble (some = foam, none = spray)"),
        Parm('pscale', "Particle Scale", 'float', default=0.01, min=0.0, soft_max=0.2),
        Parm('killground', "Kill Below Ground", 'bool', default=True),
        Parm('groundy', "Ground Height", 'float', default=0.0),
        Parm('maxparticles', "Max Particles", 'int', default=200000, min=0, max=MAX_WHITEWATER),
    )

    def _emit(self, s, state, fluid_P, fluid_V, nb, accel, p, frame, frame_dt):
        speed = np.linalg.norm(fluid_V, axis=1)
        span = max(p.maxspeed - p.minspeed, 1e-6)
        w = np.clip((speed - p.minspeed) / span, 0.0, 1.0) * (nb < p.maxneighbours)
        if accel is not None and p.accelthreshold > 0:
            w = np.maximum(w, np.clip(accel / p.accelthreshold - 1.0, 0.0, 1.0))
        expected = w * p.emitrate * frame_dt
        rng = np.random.default_rng([int(p.seed) & 0x7FFFFFFF, frame & 0x7FFFFFFF, 11])
        count = np.floor(expected + rng.random(len(expected))).astype(np.int64)
        total = int(count.sum())
        room = max(min(int(p.maxparticles), MAX_WHITEWATER) - len(s['P']), 0)
        if total == 0 or room == 0:
            return
        src = np.repeat(np.arange(len(fluid_P)), count)
        if total > room:
            src = np.sort(rng.choice(src, room, replace=False))
            total = room
        P = fluid_P[src] + random_in_sphere(rng, total) * p.particlesep * 0.5
        V = fluid_V[src] * (1.0 + random_in_sphere(rng, total) * p.jitter)
        life = np.maximum(p.life + (rng.random(total) * 2 - 1) * p.lifevar, 1e-3)
        ids = np.arange(state['next_id'], state['next_id'] + total, dtype=np.int64)
        state['next_id'] += total
        new = {'P': P, 'v': V, 'age': rng.random(total) * frame_dt, 'life': life, 'id': ids,
               'type': np.full(total, SPRAY, np.int64)}
        for k, a in new.items():
            s[k] = np.concatenate([s[k], a])

    def _advect(self, s, grid, p, dt):
        P, V = s['P'], s['v']
        n = len(P)
        if n == 0:
            return
        cnt, fluid_v = grid.query(P)
        typ = np.where(cnt == 0, SPRAY, np.where(cnt >= p.foamneighbours, BUBBLE, FOAM))
        g = np.asarray(p.gravity, dtype=np.float64)
        spray = typ == SPRAY
        V[spray] = (V[spray] + g * dt) * np.exp(-p.drag * dt)
        follow = 1.0 - np.exp(-p.foamdrag * dt)
        wet = ~spray
        V[wet] += (fluid_v[wet] - V[wet]) * follow
        bub = typ == BUBBLE
        gl = np.linalg.norm(g)
        up = -g / gl if gl > 0 else np.array([0.0, 1.0, 0.0])
        V[bub] += up * p.buoyancy * dt
        P += V * dt
        s['type'] = typ

    def cook(self, inputs, p, ctx):
        geo = _in(inputs, 0)
        state = ctx.state
        frame = int(round(ctx.frame))
        start = int(p.startframe)
        if int(p.maxparticles) > MAX_WHITEWATER:
            raise CookError("Max Particles is limited to %d" % MAX_WHITEWATER)
        key = parm_key(p)
        sig = geo_sig(geo)
        edited = state.get('key') == key and upstream_changed(state, frame, sig)
        if state.get('key') != key or frame <= start or 'sim' not in state or edited:
            state.clear()
            state['key'] = key
            state['start'] = start
            upstream_changed(state, frame, sig)
            s = {'P': np.zeros((0, 3)), 'v': np.zeros((0, 3)), 'age': np.zeros(0), 'life': np.zeros(0),
                 'id': np.zeros(0, np.int64), 'type': np.zeros(0, np.int64)}
            state.update({'sim': s, 'next_id': 0, 'frames': {}, 'last': start})
            cache_store(state, start, {k: v.copy() for k, v in s.items()})
        frames = state['frames']
        if frame <= start:
            state['prev_fluid'] = (frame, self._fluid_v(geo))
            return self._output(frames[start], p)
        if frame in frames:
            return self._output(frames[frame], p)
        if frame < state['last']:
            state.clear()
            return self.cook(inputs, p, ctx)
        s = state['sim']
        fluid_P = geo.P
        fluid_V = self._fluid_v(geo)
        frame_dt = 1.0 / ctx.fps
        accel = None
        prev = state.get('prev_fluid')
        if prev is not None and prev[1].shape == fluid_V.shape:
            accel = np.linalg.norm(fluid_V - prev[1], axis=1) / (frame_dt * max(frame - prev[0], 1))
        state['prev_fluid'] = (frame, fluid_V.copy())
        grid = _CellGrid(fluid_P, fluid_V, p.particlesep)
        nb = grid.query(fluid_P)[0] if len(fluid_P) else None
        for f in range(state['last'] + 1, frame + 1):
            self._advect(s, grid, p, frame_dt)
            s['age'] = s['age'] + frame_dt
            keep = s['age'] < s['life']
            if p.killground:
                keep &= s['P'][:, 1] >= p.groundy
            if not keep.all():
                for k in _FIELDS:
                    s[k] = s[k][keep]
            # The input only shows the liquid at this frame: skipped frames advect but don't emit.
            if f == frame and nb is not None:
                self._emit(s, state, fluid_P, fluid_V, nb, accel, p, f, frame_dt)
            cache_store(state, f, {k: v.copy() for k, v in s.items()})
            state['last'] = f
        return self._output(frames[frame], p)

    @staticmethod
    def _fluid_v(geo):
        v = geo.attribs[POINT].get('v')
        if v is not None and v.shape == geo.P.shape:
            return np.asarray(v, dtype=np.float64)
        return np.zeros_like(geo.P)

    @staticmethod
    def _output(data, p):
        out = Geometry()
        out.P = data['P']
        pa = out.attribs[POINT]
        for name in ('v', 'age', 'life', 'id', 'type'):
            pa[name] = data[name].copy()
        pa['pscale'] = np.full(len(data['P']), float(p.pscale))
        pa['Cd'] = _COLOURS[data['type']] if len(data['P']) else np.zeros((0, 3))
        return out
