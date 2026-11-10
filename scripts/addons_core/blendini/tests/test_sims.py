# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Physical behaviour tests for the simulation SOPs (particles, grains, tearing, whitewater).

Run from ``scripts/addons_core``:  python -m unittest blendini.tests.test_sims
"""

import unittest

import numpy as np

from blendini.core.geometry import Geometry
from blendini.sops import REGISTRY, default_parms, CookContext, cook_sop


def _run(sop_id, inputs, p, frames, state=None, fps=24.0):
    state = {} if state is None else state
    out = None
    for f in frames:
        out = REGISTRY[sop_id].cook(list(inputs), p, CookContext(frame=f, time=(f - 1) / fps, fps=fps, state=state))
        if hasattr(out, 'validate'):
            assert out.validate() == [], out.validate()
    return out


def _points(P):
    g = Geometry()
    g.P = np.asarray(P, dtype=np.float64)
    return g


class TestPOPSolver(unittest.TestCase):
    def _parms(self, **kw):
        p = default_parms('popsolver')
        p.update(dict(birthrate=240.0, velvar=(0.0, 0.0, 0.0), normalvel=0.0, drag=0.0))
        p.update(kw)
        return p

    def test_particles_fall_and_have_attributes(self):
        emitter = _points([[0.0, 5.0, 0.0]])
        out = _run('popsolver', [emitter], self._parms(ground=False), range(1, 13))
        pa = out.attribs['point']
        for name in ('v', 'age', 'life', 'id', 'pscale'):
            self.assertIn(name, pa)
        self.assertGreater(out.num_points, 50)
        # Older particles fell further: y decreases with age, v.y is negative.
        old = pa['age'] > 0.3
        self.assertTrue(np.all(out.P[old, 1] < 5.0))
        self.assertTrue(np.all(pa['v'][old, 1] < 0))
        a = pa['age'][old][0]
        self.assertAlmostEqual(out.P[old][0, 1], 5.0 - 0.5 * 9.81 * a * a, delta=0.15)

    def test_ground_bounce(self):
        emitter = _points([[0.0, 1.0, 0.0]])
        p = self._parms(birthrate=24.0, bounce=0.8, life=10.0, lifevar=0.0)
        state = {}
        miny, rising = [], False
        for f in range(1, 30):
            out = _run('popsolver', [emitter], p, [f], state)
            first = out.attribs['point']['id'] == 0
            if first.any():
                miny.append(out.P[first, 1][0])
                rising |= bool(out.attribs['point']['v'][first, 1][0] > 1.0 and out.P[first, 1][0] < 0.9)
        self.assertGreaterEqual(min(miny), 0.0)
        self.assertTrue(rising, "particle should bounce back up from the ground")

    def test_age_out_and_stable_ids(self):
        emitter = _points([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
        p = self._parms(life=0.5, lifevar=0.0, gravity=(0.0, 0.0, 0.0))
        state = {}
        prev = None
        for f in range(1, 40):
            out = _run('popsolver', [emitter], p, [f], state)
            pa = out.attribs['point']
            self.assertTrue(np.all(pa['age'] < pa['life'] + 1e-9))
            ids = pa['id']
            self.assertEqual(len(np.unique(ids)), len(ids))
            if prev is not None and len(prev[0]):
                # Particles keep their id and position ordering from frame to frame.
                common = np.intersect1d(prev[0], ids)
                self.assertGreater(len(common), 0)
                np.testing.assert_array_equal(ids[np.isin(ids, common)], common)
            prev = (ids, out.P)
        # Steady state: rate * life particles alive.
        self.assertAlmostEqual(out.num_points, 240 * 0.5, delta=15)
        self.assertGreater(ids.min(), 0, "the first particles must have died")

    def test_deterministic_and_seed(self):
        sphere = cook_sop('sphere')
        p = self._parms(velvar=(1.0, 1.0, 1.0), turbamp=2.0)
        a = _run('popsolver', [sphere], p, range(1, 10))
        b = _run('popsolver', [sphere], p, range(1, 10))
        np.testing.assert_array_equal(a.P, b.P)
        p2 = self._parms(velvar=(1.0, 1.0, 1.0), turbamp=2.0, seed=3)
        c = _run('popsolver', [sphere], p2, range(1, 10))
        self.assertFalse(a.num_points == c.num_points and np.allclose(a.P, c.P))

    def test_surface_emission_and_colour(self):
        grid = cook_sop('grid', size=(2.0, 2.0))
        grid.attribs['point']['Cd'] = np.tile([1.0, 0.0, 0.0], (grid.num_points, 1))
        p = self._parms(gravity=(0.0, 0.0, 0.0), normalvel=1.0)
        out = _run('popsolver', [grid], p, range(1, 6))
        self.assertGreater(out.num_points, 0)
        np.testing.assert_allclose(out.attribs['point']['Cd'], [[1.0, 0.0, 0.0]] * out.num_points)
        lo, hi = out.P.min(axis=0), out.P.max(axis=0)
        self.assertLess(hi[0] - lo[0], 2.0 + 1e-6)
        self.assertGreater(hi[0] - lo[0], 1.0)   # scattered over the surface, not one point
        self.assertTrue(np.all(out.attribs['point']['v'][:, 1] > 0.5))   # along the grid normal

    def test_restart_and_scrub(self):
        emitter = _points([[0.0, 1.0, 0.0]])
        p = self._parms()
        state = {}
        a = _run('popsolver', [emitter], p, range(1, 11), state)
        back = _run('popsolver', [emitter], p, [5], state)
        again = _run('popsolver', [emitter], p, [10], state)
        np.testing.assert_array_equal(a.P, again.P)
        self.assertLess(back.num_points, a.num_points)
        # Upstream edit: emitter moved, the same frame re-cooks from scratch.
        moved = _points([[0.0, 3.0, 0.0]])
        c = _run('popsolver', [moved], p, [10], state)
        self.assertGreater(c.P[:, 1].max(), 1.5)

    def test_max_particles_and_empty(self):
        p = self._parms(birthrate=100000.0, maxparticles=500)
        out = _run('popsolver', [cook_sop('box')], p, range(1, 4))
        self.assertLessEqual(out.num_points, 500)
        out = _run('popsolver', [Geometry()], p, range(1, 4))
        self.assertEqual(out.num_points, 0)
        out = _run('popsolver', [], p, range(1, 4))
        self.assertEqual(out.num_points, 0)


class TestVellumGrains(unittest.TestCase):
    def test_column_forms_pile(self):
        r = 0.05
        g = np.arange(-3, 3) * 2 * r + r
        x, y, z = np.meshgrid(g, np.arange(12) * 2 * r + r, g, indexing='ij')
        P = np.stack([x.ravel(), y.ravel(), z.ravel()], 1)
        P += np.random.default_rng(0).normal(size=P.shape) * 0.002
        p = default_parms('vellumgrains')
        p.update(radius=r, boxmin=(-1.0, -1.0, -1.0), boxmax=(1.0, 3.0, 1.0))
        out = _run('vellumgrains', [_points(P)], p, range(1, 61))
        Q = out.P
        h0 = P[:, 1].max()
        spread0 = np.abs(P[:, [0, 2]]).max()
        # Spreads, but keeps a heap (angle of repose), not a single layer.
        self.assertLess(Q[:, 1].max(), h0)
        self.assertGreater(Q[:, 1].max(), 4 * r)
        self.assertGreater(np.abs(Q[:, [0, 2]]).max(), spread0 + r)
        # Within the container and above the ground.
        self.assertTrue(np.all(Q[:, 1] >= r - 1e-6))
        self.assertTrue(np.all(np.abs(Q[:, [0, 2]]) <= 1.0 - r + 1e-6))
        # No significant interpenetration.
        d = np.linalg.norm(Q[:, None] - Q[None], axis=2)
        np.fill_diagonal(d, np.inf)
        self.assertGreater(d.min(), 2 * r * 0.7)
        self.assertGreater(np.median(d.min(axis=1)), 2 * r * 0.9)
        # Comes (nearly) to rest.
        self.assertLess(np.median(np.linalg.norm(out.attribs['point']['v'], axis=1)), 0.2)

    def test_pscale_and_empty(self):
        pts = _points([[0.0, 1.0, 0.0], [0.0, 1.5, 0.0]])
        pts.attribs['point']['pscale'] = np.array([0.1, 0.2])
        out = _run('vellumgrains', [pts], default_parms('vellumgrains'), range(1, 40))
        np.testing.assert_allclose(out.attribs['point']['pscale'], [0.1, 0.2])
        self.assertAlmostEqual(out.P[:, 1].min(), 0.1, delta=0.02)
        self.assertGreater(np.linalg.norm(out.P[1] - out.P[0]), 0.3 * 0.9)
        out = _run('vellumgrains', [Geometry()], default_parms('vellumgrains'), range(1, 4))
        self.assertEqual(out.num_points, 0)


    def test_packed_input_stays_valid(self):
        packed = cook_sop('pack', [cook_sop('box')])
        out = _run('vellumgrains', [packed], default_parms('vellumgrains'), range(1, 4))
        self.assertEqual(out.validate(), [])
        self.assertEqual(len(out.instances), len(packed.instances))


class TestVellumTearing(unittest.TestCase):
    def _pulled(self, tearing, frames=30):
        from blendini.sops.vellum import _components
        grid = cook_sop('grid', rows=11, cols=11, size=(2.0, 2.0))
        grid.P = grid.P + [0.0, 2.0, 0.0]
        p = default_parms('vellum')
        p.update(tearing=tearing, pingroup='@P.x<-0.99 || @P.x>0.99', ground=False)
        state = {}
        for f in range(1, frames + 1):
            g = grid.copy()
            g.P[g.P[:, 0] > 0.99, 0] += 0.1 * (f - 1)   # animated pins pull the cloth apart
            out = _run('vellum', [g], p, [f], state)
        return grid, out, _components(out.num_points, out.edges())

    def test_off_by_default_keeps_topology(self):
        self.assertFalse(default_parms('vellum')['tearing'])
        grid, out, comp = self._pulled(False, 12)
        self.assertEqual(out.num_points, grid.num_points)
        np.testing.assert_array_equal(out.vtx_point, grid.vtx_point)
        self.assertEqual(comp.max(), 0)

    def test_tearing_splits_mesh(self):
        grid, out, comp = self._pulled(True)
        self.assertGreater(out.num_points, grid.num_points)      # points duplicated along the tear
        self.assertEqual(out.num_prims, grid.num_prims)          # no faces lost
        self.assertGreaterEqual(comp.max(), 1)                   # at least two pieces
        self.assertGreater(out.attribs['point']['torn'].sum(), 0)
        self.assertTrue(out.groups['point']['torn'].any())
        # The pieces really separate: the piece held by the left pins stays left of the right one.
        left = comp[np.argmin(out.P[:, 0])]
        right = comp[np.argmax(out.P[:, 0])]
        self.assertNotEqual(left, right)
        self.assertLess(out.P[comp == left, 0].max() + 0.5, out.P[comp == right, 0].min())

    def test_hanging_cloth_does_not_tear_with_high_threshold(self):
        grid = cook_sop('grid', rows=8, cols=8, size=(2.0, 2.0))
        grid.P = grid.P + [0.0, 2.0, 0.0]
        p = default_parms('vellum')
        p.update(tearing=True, tearthreshold=3.0, pingroup='0 7', ground=False)
        out = _run('vellum', [grid], p, range(1, 13))
        self.assertEqual(out.num_points, grid.num_points)


class TestWhitewater(unittest.TestCase):
    def _fluid(self, speed):
        g = np.arange(0, 0.8, 0.08)
        x, y, z = np.meshgrid(g, g + 1.0, g, indexing='ij')
        fl = _points(np.stack([x.ravel(), y.ravel(), z.ravel()], 1))
        fl.attribs['point']['v'] = np.tile([speed, 2.0 * speed, 0.0], (fl.num_points, 1))
        return fl

    def test_fast_surface_emits_and_spray_falls(self):
        fl = self._fluid(3.0)
        p = default_parms('whitewater')
        state = {}
        out = _run('whitewater', [fl], p, range(1, 4), state)
        self.assertGreater(out.num_points, 0)
        pa = out.attribs['point']
        for name in ('v', 'age', 'life', 'id', 'type', 'pscale', 'Cd'):
            self.assertIn(name, pa)
        self.assertEqual(len(np.unique(pa['id'])), out.num_points)
        # Remove the liquid: spray flies ballistically, falls and dies.
        empty = _points(np.zeros((0, 3)))
        state2 = {}
        _run('whitewater', [fl], p, range(1, 4), state2)
        ids0 = _run('whitewater', [fl], p, [4], state2).attribs['point']['id']
        after = _run('whitewater', [empty], p, range(5, 9), state2)
        self.assertTrue(np.all(after.attribs['point']['type'] == 0))
        self.assertTrue(np.all(after.attribs['point']['v'][:, 1] < 6.0 + 1e-9))
        self.assertTrue(np.isin(after.attribs['point']['id'], ids0).all())
        gone = _run('whitewater', [empty], p, range(9, 120), state2)
        self.assertEqual(gone.num_points, 0)

    def test_slow_liquid_emits_nothing_and_deterministic(self):
        p = default_parms('whitewater')
        out = _run('whitewater', [self._fluid(0.1)], p, range(1, 6))
        self.assertEqual(out.num_points, 0)
        a = _run('whitewater', [self._fluid(3.0)], p, range(1, 6))
        b = _run('whitewater', [self._fluid(3.0)], p, range(1, 6))
        np.testing.assert_array_equal(a.P, b.P)
        out = _run('whitewater', [Geometry()], p, range(1, 4))
        self.assertEqual(out.num_points, 0)


if __name__ == '__main__':
    unittest.main()
