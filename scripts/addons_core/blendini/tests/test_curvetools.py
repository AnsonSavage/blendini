# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Tests for the curve SOPs in sops/curvetools.py."""

import unittest

import numpy as np

from blendini.sops import cook_sop
from blendini.core.geometry import Geometry, merge
from blendini.sops.base import orient_matrices
from blendini.sops.curvetools import nurbs_curve, bezier_curve


def polyline(points, closed=False):
    pts = np.asarray(points, dtype=np.float64)
    return Geometry.from_polys(pts, [list(range(len(pts)))], closed=closed)


def square(side=2.0):
    h = side / 2
    return polyline([(-h, 0, -h), (h, 0, -h), (h, 0, h), (-h, 0, h)], closed=True)


def curve_length(geo, prim=0):
    P = geo.P[geo.prim_points(prim)]
    if geo.prim_closed[prim]:
        P = np.concatenate([P, P[:1]])
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def signed_area_y(geo, prim=0):
    P = geo.P[geo.prim_points(prim)]
    x, z = P[:, 0], P[:, 2]
    return 0.5 * float(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


class TestJoin(unittest.TestCase):
    def test_join_reverses_and_fuses(self):
        a = polyline([(0, 0, 0), (1, 0, 0)])
        b = polyline([(2, 0, 0), (1, 0, 0)])   # reversed, touching a's end
        out = cook_sop('join', [merge([a, b])])
        self.assertEqual(out.num_prims, 1)
        self.assertEqual(out.num_points, 3)
        self.assertAlmostEqual(curve_length(out), 2.0)
        self.assertTrue(np.allclose(out.P[out.prim_points(0)][:, 0], [0, 1, 2]))

    def test_join_gap_and_wrap(self):
        a = polyline([(0, 0, 0), (1, 0, 0)])
        b = polyline([(1, 0, 1), (0, 0, 1)])
        out = cook_sop('join', [merge([a, b])], wrap=True)
        self.assertEqual(out.num_prims, 1)
        self.assertTrue(out.prim_closed[0])
        self.assertAlmostEqual(curve_length(out), 4.0)

    def test_join_count(self):
        lines = [polyline([(i, 0, 0), (i + 1, 0, 0)]) for i in range(6)]
        out = cook_sop('join', [merge(lines)], joincount=2)
        self.assertEqual(out.num_prims, 3)
        self.assertTrue(all(len(out.prim_points(i)) == 3 for i in range(3)))

    def test_only_connected(self):
        # Two separate chains in shuffled order and direction plus a closed loop of 3 segments.
        segs = [[(1, 0, 0), (2, 0, 0)], [(5, 0, 0), (6, 0, 0)], [(1, 0, 0), (0, 0, 0)], [(7, 0, 0), (6, 0, 0)],
                [(0, 5, 0), (1, 5, 0)], [(1, 5, 0), (0, 6, 0)], [(0, 6, 0), (0, 5, 0)]]
        out = cook_sop('join', [merge([polyline(s) for s in segs])], onlyconnected=True)
        self.assertEqual(out.num_prims, 3)
        lengths = sorted(curve_length(out, i) for i in range(3))
        self.assertAlmostEqual(lengths[0], 2.0)
        self.assertAlmostEqual(lengths[1], 2.0)
        self.assertEqual(int(out.prim_closed.sum()), 1)

    def test_empty(self):
        self.assertEqual(cook_sop('join', [Geometry()]).num_points, 0)


class TestPolyCut(unittest.TestCase):
    def line10(self):
        return cook_sop('line', dir=(1, 0, 0), dist=9.0, points=10)

    def test_group_cut_and_remove(self):
        out = cook_sop('polycut', [self.line10()], group='4')
        self.assertEqual(out.num_prims, 2)
        self.assertEqual(sorted(len(out.prim_points(i)) for i in range(2)), [5, 6])
        self.assertAlmostEqual(curve_length(out, 0) + curve_length(out, 1), 9.0)
        out = cook_sop('polycut', [self.line10()], group='4', strategy='remove')
        self.assertEqual(sorted(len(out.prim_points(i)) for i in range(2)), [4, 5])

    def test_threshold(self):
        g = cook_sop('curveattribs', [self.line10()])
        out = cook_sop('polycut', [g], mode='threshold', attrib='curveu', threshold=0.5)
        self.assertEqual(out.num_prims, 2)
        ends = [out.P[out.prim_points(i)][[0, -1], 0] for i in range(2)]
        self.assertTrue(any(np.isclose(e, 4.5).any() for e in ends))
        self.assertAlmostEqual(curve_length(out, 0) + curve_length(out, 1), 9.0)
        out = cook_sop('polycut', [g], mode='threshold', attrib='curveu', threshold=0.5, strategy='remove')
        self.assertEqual(out.num_prims, 1)
        xs = out.P[:, 0]
        self.assertAlmostEqual(xs.min(), 4.5)
        self.assertAlmostEqual(xs.max(), 9.0)
        self.assertTrue(np.allclose(out.attribs['point']['curveu'], xs / 9.0))

    def test_attribute_change_on_closed(self):
        g = square()
        g.attribs['point']['side'] = np.array([0, 0, 1, 1])
        out = cook_sop('polycut', [g], mode='change', attrib='side')
        self.assertEqual(out.num_prims, 2)
        self.assertFalse(out.prim_closed.any())
        self.assertAlmostEqual(sum(curve_length(out, i) for i in range(2)), 8.0)
        out = cook_sop('polycut', [g], mode='change', attrib='side', strategy='remove')
        self.assertEqual(out.num_prims, 2)
        self.assertAlmostEqual(sum(curve_length(out, i) for i in range(2)), 4.0)

    def test_no_cut_passthrough(self):
        out = cook_sop('polycut', [square()])
        self.assertEqual(out.num_prims, 1)
        self.assertTrue(out.prim_closed[0])


class TestOrientAlongCurve(unittest.TestCase):
    def test_circle_frames(self):
        c = cook_sop('circle', divs=32)
        out = cook_sop('orientalongcurve', [c], outputout=True)
        N, up, X = (out.attribs['point'][k] for k in ('N', 'up', 'out'))
        self.assertTrue(np.allclose(np.abs(up[:, 1]), 1.0, atol=1e-6))   # planar: up stays the plane normal
        self.assertTrue(np.allclose(np.einsum('ij,ij->i', N, up), 0, atol=1e-9))
        self.assertTrue(np.allclose(np.linalg.norm(N, axis=1), 1.0))
        # The orient quaternion encodes the same frame as N + up.
        m = orient_matrices(out)
        self.assertTrue(np.allclose(m[:, :, 2], N, atol=1e-6))
        self.assertTrue(np.allclose(m[:, :, 1], up, atol=1e-6))
        self.assertTrue(np.allclose(m[:, :, 0], X, atol=1e-6))

    def test_helix_transport_has_no_twist(self):
        def twist(divs, uptype):
            h = cook_sop('spiral', turns=2, height=3.0, divs=divs)
            out = cook_sop('orientalongcurve', [h], outputout=True, uptype=uptype, upvector=(1, 0, 0))
            up, X = out.attribs['point']['up'], out.attribs['point']['out']
            return np.abs(np.einsum('ij,ij->i', np.diff(up, axis=0), X[:-1])).max()
        # Rotation minimizing: the twist per step is second order in the step (vanishes 16x for
        # 4x the points) while a projected constant up twists to first order.
        self.assertLess(twist(64, 'transport'), 0.02 * twist(64, 'constant'))
        self.assertLess(twist(256, 'transport'), twist(64, 'transport') / 10)

    def test_closed_continuity(self):
        t = np.linspace(0, 2 * np.pi, 80, endpoint=False)
        P = np.stack([np.cos(t), 0.6 * np.sin(2 * t), np.sin(t)], axis=1)   # non-planar closed loop
        out = cook_sop('orientalongcurve', [polyline(P, closed=True)])
        up = out.attribs['point']['up']
        steps = np.einsum('ij,ij->i', up, np.roll(up, -1, axis=0))
        self.assertGreater(steps.min(), 0.95)        # includes the closing edge

    def test_roll_and_constant(self):
        line = cook_sop('line', dir=(1, 0, 0), points=5)
        out = cook_sop('orientalongcurve', [line], roll=90.0)
        up = out.attribs['point']['up']
        self.assertTrue(np.allclose(np.abs(up[:, 2]), 1.0))
        out = cook_sop('orientalongcurve', [line], uptype='constant', upvector=(0, 0, 1))
        self.assertTrue(np.allclose(out.attribs['point']['up'], [0, 0, 1]))


class TestFillet(unittest.TestCase):
    def test_square_corners(self):
        out = cook_sop('fillet', [square(2.0)], radius=0.5, divs=64)
        self.assertEqual(out.num_points, 4 * 65)
        self.assertTrue(out.prim_closed[0])
        self.assertAlmostEqual(curve_length(out), 4.0 + np.pi, places=3)
        # Arc points lie at the fillet radius from the inset corner centers.
        P = out.P
        centers = np.sign(P[:, [0, 2]]) * 0.5
        r = np.linalg.norm(P[:, [0, 2]] - centers, axis=1)
        self.assertTrue(np.allclose(r, 0.5))

    def test_clamp_and_open(self):
        line = polyline([(0, 0, 0), (1, 0, 0), (1, 0, 1)])
        out = cook_sop('fillet', [line], radius=10.0, divs=8)
        self.assertEqual(out.num_points, 2 + 9)
        P = out.P
        self.assertTrue(np.allclose(P[0], [0, 0, 0]) and np.allclose(P[-1], [1, 0, 1]))
        self.assertTrue(np.allclose(P[1], [0.5, 0, 0]))     # clamped to half the edge
        out = cook_sop('fillet', [line], radius=10.0, divs=8, clamp=False)
        self.assertEqual(out.num_points, 11)

    def test_straight_points_untouched(self):
        line = cook_sop('line', points=5)
        self.assertEqual(cook_sop('fillet', [line], radius=0.3).num_points, 5)


class TestConvertCurve(unittest.TestCase):
    def test_rational_quarter_circle(self):
        C = np.array([[1.0, 0, 0], [1, 1, 0], [0, 1, 0]])
        E = nurbs_curve(C, 2, weights=[1.0, np.sqrt(0.5), 1.0], divisions=20)
        self.assertTrue(np.allclose(np.linalg.norm(E, axis=1), 1.0))
        self.assertTrue(np.allclose(E[0], C[0]) and np.allclose(E[-1], C[-1]))

    def test_weight_attribute_via_sop(self):
        g = polyline([(1.0, 0, 0), (1, 1, 0), (0, 1, 0)])
        g.attribs['point']['Pw'] = np.array([1.0, np.sqrt(0.5), 1.0])
        out = cook_sop('convertcurve', [g], degree=2, divisions=16)
        self.assertEqual(out.num_points, 17)
        self.assertTrue(np.allclose(np.linalg.norm(out.P, axis=1), 1.0))

    def test_degree_one_reproduces_polygon(self):
        C = np.array([[0.0, 0, 0], [1, 0, 0], [1, 2, 0], [3, 2, 0]])
        E = nurbs_curve(C, 1, divisions=4)
        self.assertEqual(len(E), 13)
        self.assertTrue(np.allclose(E[::4], C))

    def test_uniform_cubic_and_closed(self):
        C = np.array([[0.0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
        E = nurbs_curve(C, 3, knots='uniform', divisions=6)
        # Uniform cubic B-spline of 4 collinear equally spaced points: one span from 1 to 2.
        self.assertTrue(np.allclose(E[[0, -1], 0], [1, 2]))
        sq = square(2.0)
        out = cook_sop('convertcurve', [sq], divisions=10)
        self.assertTrue(out.prim_closed[0])
        self.assertEqual(out.num_points, 40)
        r = np.linalg.norm(out.P[:, [0, 2]], axis=1)
        self.assertTrue(np.all(r < np.sqrt(2)) and np.all(r > 0.5))
        steps = np.linalg.norm(out.P - np.roll(out.P, 1, axis=0), axis=1)
        self.assertLess(steps.max() / steps.min(), 3.0)       # continuous across the seam

    def test_bezier_and_catmull(self):
        C = np.array([[0.0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, -1, 0], [2, -1, 0], [2, 0, 0]])
        E = bezier_curve(C, 3, divisions=8)
        self.assertEqual(len(E), 17)
        self.assertTrue(np.allclose(E[[0, 8, 16]], C[[0, 3, 6]]))
        out = cook_sop('convertcurve', [polyline(C)], type='catmull', divisions=5)
        self.assertTrue(np.allclose(out.P[::5], C))

    def test_attributes_interpolated(self):
        g = cook_sop('line', dir=(1, 0, 0), points=5, dist=4.0)
        g.attribs['point']['val'] = np.arange(5, dtype=np.float64)
        out = cook_sop('convertcurve', [g], degree=1, divisions=2)
        self.assertTrue(np.allclose(out.attribs['point']['val'], out.P[:, 0]))


class TestPolyExpand2D(unittest.TestCase):
    def test_square_offsets(self):
        out = cook_sop('polyexpand2d', [square(2.0)], offset=0.5)
        self.assertAlmostEqual(abs(signed_area_y(out)), 9.0)
        self.assertAlmostEqual(curve_length(out), 12.0)
        out = cook_sop('polyexpand2d', [square(2.0)], offset=0.5, side='inside')
        self.assertAlmostEqual(abs(signed_area_y(out)), 1.0)
        self.assertTrue(out.groups['prim']['inside'].all())

    def test_collapse_and_multiple(self):
        out = cook_sop('polyexpand2d', [square(2.0)], offset=0.4, numoffsets=3, side='both', keepinput=True)
        # Outside: 3 rings, inside: 0.4 and 0.8 survive, 1.2 collapses; plus the input.
        self.assertEqual(out.num_prims, 1 + 3 + 2)
        areas = sorted(abs(signed_area_y(out, i)) for i in range(out.num_prims))
        self.assertTrue(np.allclose(areas, sorted([4.0, 2.8 ** 2, 3.6 ** 2, 4.4 ** 2, 1.2 ** 2, 0.4 ** 2])))


class TestCurveAttribs(unittest.TestCase):
    def test_line_and_closed(self):
        g = merge([cook_sop('line', dir=(1, 0, 0), dist=4.0, points=5), square(2.0)])
        out = cook_sop('curveattribs', [g], ptidx='i')
        pa = out.attribs['point']
        self.assertTrue(np.allclose(pa['curveu'][:5], [0, .25, .5, .75, 1]))
        self.assertTrue(np.allclose(pa['dist'][:5], [0, 1, 2, 3, 4]))
        self.assertTrue(np.allclose(pa['curveu'][5:], [0, .25, .5, .75]))
        self.assertEqual(pa['curvenum'].tolist(), [0] * 5 + [1] * 4)
        self.assertTrue(np.allclose(out.attribs['prim']['perimeter'], [4, 8]))
        out = cook_sop('curveattribs', [g], reverse=True)
        self.assertTrue(np.allclose(out.attribs['point']['curveu'][:5], [1, .75, .5, .25, 0]))


class TestGenerators(unittest.TestCase):
    def test_helix(self):
        h = cook_sop('spiral', turns=3, height=2.0, radstart=0.5, radend=0.5, divs=100)
        self.assertEqual(h.num_points, 301)
        self.assertTrue(np.allclose(np.linalg.norm(h.P[:, [0, 2]], axis=1), 0.5))
        self.assertAlmostEqual(h.P[:, 1].max() - h.P[:, 1].min(), 2.0)
        expected = 3 * np.hypot(2 * np.pi * 0.5, 2.0 / 3)
        self.assertAlmostEqual(curve_length(h), expected, delta=expected * 1e-3)

    def test_spiral_radius(self):
        s = cook_sop('spiral', height=0.0, radstart=0.0, radend=2.0, axis='Z')
        r = np.linalg.norm(s.P[:, :2], axis=1)
        self.assertAlmostEqual(r[0], 0.0)
        self.assertAlmostEqual(r[-1], 2.0)
        self.assertTrue(np.all(np.diff(r) > 0))
        self.assertTrue(np.allclose(s.P[:, 2], 0.0))

    def test_arc_types(self):
        a = cook_sop('arc', divs=8)
        self.assertEqual((a.num_points, bool(a.prim_closed[0])), (9, False))
        self.assertTrue(np.allclose(np.linalg.norm(a.P, axis=1), 1.0))
        self.assertAlmostEqual(curve_length(a), np.pi / 2, places=2)
        s = cook_sop('arc', type='sliced', divs=8)
        self.assertEqual(s.num_points, 10)
        self.assertTrue(s.prim_closed[0])
        self.assertAlmostEqual(abs(signed_area_y(s)), np.pi / 4, places=1)
        c = cook_sop('arc', type='closed', anglestart=0, angleend=180, divs=16)
        self.assertTrue(c.prim_closed[0])
        self.assertAlmostEqual(curve_length(c), np.pi + 2, delta=0.01)


class TestRobustness(unittest.TestCase):
    def test_empty_and_meshes(self):
        ids = ('join', 'polycut', 'orientalongcurve', 'fillet', 'convertcurve', 'polyexpand2d', 'curveattribs')
        inputs = [Geometry(), cook_sop('box'), cook_sop('sphere'), cook_sop('scatter', [cook_sop('sphere')], npts=20)]
        for sop_id in ids:
            for g in inputs:
                with self.subTest(sop=sop_id, geo=g):
                    out = cook_sop(sop_id, [g], attrib='P') if sop_id == 'polycut' else cook_sop(sop_id, [g])
                    self.assertIsInstance(out, Geometry)


if __name__ == '__main__':
    unittest.main()
