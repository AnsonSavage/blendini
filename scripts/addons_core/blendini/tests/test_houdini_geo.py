# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Tests for the Houdini ASCII .geo reader / writer (core.houdini_geo).

Run from ``scripts/addons_core``:  python -m unittest blendini.tests.test_houdini_geo

The fixtures in tests/data/*.geo are hand-written from the documented Houdini JSON geometry
layout; they were not produced by Houdini.
"""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from blendini.core.geometry import Geometry, merge
from blendini.core.houdini_geo import read_geo, write_geo, parse_geo
from blendini.sops import cook_sop

DATA = os.path.join(os.path.dirname(__file__), 'data')


class RoundTrip(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def roundtrip(self, geo):
        path = os.path.join(self.dir, 'g.geo')
        write_geo(geo, path)
        with open(path) as fh:
            self.assertEqual(json.load(fh)[0], 'fileversion')
        return read_geo(path)

    def assertSameGeo(self, a, b):
        self.assertEqual((a.num_points, a.num_prims, a.num_vertices), (b.num_points, b.num_prims, b.num_vertices))
        np.testing.assert_allclose(a.P, b.P)
        np.testing.assert_array_equal(a.vtx_point, b.vtx_point)
        np.testing.assert_array_equal(a.prim_start, b.prim_start)
        np.testing.assert_array_equal(a.prim_closed, b.prim_closed)
        for cls in ('point', 'vertex', 'prim', 'detail'):
            self.assertEqual(set(a.attribs[cls]), set(b.attribs[cls]), cls)
            for name, arr in a.attribs[cls].items():
                other = b.attribs[cls][name]
                self.assertEqual(arr.shape, other.shape, name)
                if arr.dtype == object:
                    self.assertEqual(arr.tolist(), other.tolist(), name)
                else:
                    np.testing.assert_allclose(arr, other, err_msg=name)
        for cls in ('point', 'prim'):
            self.assertEqual(set(a.groups[cls]), set(b.groups[cls]))
            for name, g in a.groups[cls].items():
                np.testing.assert_array_equal(g, b.groups[cls][name])

    def test_box(self):
        box = cook_sop('box')
        self.assertSameGeo(box, self.roundtrip(box))

    def test_grid_attributes_groups_strings(self):
        g = cook_sop('grid', rows=6, cols=5, size=(3.0, 2.0))
        n, npr, nv = g.num_points, g.num_prims, g.num_vertices
        rng = np.random.default_rng(1)
        g.attribs['point']['Cd'] = rng.random((n, 3))
        g.attribs['point']['id'] = np.arange(n, dtype=np.int64)
        g.attribs['point']['orient'] = rng.random((n, 4))
        g.attribs['point']['uv2'] = rng.random((n, 2))
        g.attribs['point']['xf'] = rng.random((n, 3, 3))
        g.attribs['point']['m4'] = rng.random((n, 4, 4))
        g.attribs['point']['name'] = np.array(['p%d' % (i % 3) for i in range(n)], dtype=object)
        g.attribs['vertex']['uv'] = rng.random((nv, 3))
        g.attribs['prim']['area'] = rng.random(npr)
        g.attribs['prim']['path'] = np.array(['/a' if i % 2 else '/b/c' for i in range(npr)], dtype=object)
        g.attribs['detail']['frame'] = np.array([12], dtype=np.int64)
        g.attribs['detail']['center'] = np.array([[1.0, 2.0, 3.0]])
        g.attribs['detail']['label'] = np.array(['hello "world"'], dtype=object)
        g.groups['point']['odd'] = np.arange(n) % 2 == 1
        g.groups['prim']['first'] = np.arange(npr) < 3
        self.assertSameGeo(g, self.roundtrip(g))

    def test_polylines_and_mixed(self):
        line = Geometry.from_polys([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 2, 1]], [[0, 1, 2, 3], [3, 1]], closed=False)
        self.assertSameGeo(line, self.roundtrip(line))
        mixed = merge([cook_sop('box'), line, cook_sop('grid', rows=3, cols=3)])
        mixed.attribs['prim']['kind'] = np.arange(mixed.num_prims, dtype=np.int64)
        back = self.roundtrip(mixed)
        self.assertSameGeo(mixed, back)
        self.assertEqual(back.prim_closed.tolist().count(False), 2)

    def test_points_only_and_empty(self):
        pts = Geometry.from_polys(np.arange(12, dtype=float).reshape(4, 3))
        pts.attribs['point']['pscale'] = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertSameGeo(pts, self.roundtrip(pts))
        self.assertSameGeo(Geometry(), self.roundtrip(Geometry()))


class Fixtures(unittest.TestCase):
    """Hand-written fixtures that follow the Houdini layout (not produced by Houdini)."""

    def test_run_rawpagedata(self):
        g = read_geo(os.path.join(DATA, 'grid_run_rawpage.geo'))
        self.assertEqual((g.num_points, g.num_prims, g.num_vertices), (9, 4, 16))
        self.assertTrue(g.prim_closed.all())
        np.testing.assert_allclose(g.P[4], [0, 0.25, 0])
        np.testing.assert_allclose(g.P[8], [1, 0, 1])
        self.assertEqual(g.prims_as_lists()[3], [4, 5, 8, 7])
        # Constant page: every point gets the single stored tuple.
        np.testing.assert_allclose(g.attribs['point']['Cd'], np.tile([1, 0.5, 0.25], (9, 1)))
        uv = g.attribs['vertex']['uv']
        self.assertEqual(uv.shape, (16, 3))
        np.testing.assert_allclose(uv[2], [0.5, 0.5, 0])
        np.testing.assert_allclose(uv[15], [0.5, 1, 0])
        np.testing.assert_allclose(uv[:, 2], 0)
        self.assertEqual(g.attribs['prim']['id'].tolist(), [10, 11, 12, 13])
        self.assertEqual(g.attribs['detail']['varmap'].tolist(), ['uv -> UV'])

    def test_poly_boolrle_groups(self):
        g = read_geo(os.path.join(DATA, 'box_poly_groups.geo'))
        # The Sphere primitive is skipped, its vertex is not used.
        self.assertEqual((g.num_points, g.num_prims, g.num_vertices), (8, 6, 24))
        self.assertEqual(g.P.shape, (8, 3))
        np.testing.assert_allclose(g.bbox()[0], [-0.5, -0.5, -0.5])
        np.testing.assert_allclose(g.attribs['point']['pscale'], 0.25)
        self.assertEqual(g.groups['point']['top'].tolist(), [False] * 4 + [True] * 4)
        self.assertEqual(g.groups['prim']['sides'].tolist(), [False, True, False, True, True, False])
        self.assertEqual(g.groups['prim']['firsttwo'].tolist(), [True, True, False, False, False, False])
        N = g.attribs['prim']['N']
        np.testing.assert_allclose(N[0], [0, 0, -1])
        np.testing.assert_allclose(N[5], [0, 1, 0])
        self.assertEqual(g.attribs['prim']['xform'].shape, (6, 3, 3))
        self.assertGreater(g.prim_areas().min(), 0.99)

    def test_curves_strings(self):
        g = read_geo(os.path.join(DATA, 'curves_strings.geo'))
        self.assertEqual((g.num_points, g.num_prims, g.num_vertices), (7, 3, 8))
        self.assertFalse(g.prim_closed.any())
        self.assertEqual(g.prims_as_lists(), [[0, 1, 2, 3], [4, 5, 6], [0]])
        self.assertEqual(g.attribs['point']['tag'].tolist(), ['a', 'a', 'b', 'b', '', 'a', 'b'])
        self.assertEqual(g.attribs['prim']['name'].tolist(), ['curve_A', 'curve_B', 'dot'])
        np.testing.assert_allclose(g.attribs['prim']['width'], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(g.attribs['vertex']['curveu'][:4], [0, 0.333333, 0.666667, 1])
        self.assertEqual(g.attribs['detail']['frame'].tolist(), [24])
        self.assertEqual(g.groups['prim']['curves'].tolist(), [True, True, False])

    def test_fixture_rewrite(self):
        for name in ('grid_run_rawpage.geo', 'box_poly_groups.geo', 'curves_strings.geo'):
            g = read_geo(os.path.join(DATA, name))
            d = tempfile.mkdtemp()
            try:
                path = os.path.join(d, name)
                write_geo(g, path)
                back = read_geo(path)
                RoundTrip.assertSameGeo(self, g, back)
            finally:
                shutil.rmtree(d, ignore_errors=True)


class Robustness(unittest.TestCase):
    def test_not_houdini(self):
        d = tempfile.mkdtemp()
        try:
            for i, text in enumerate(('{"a": 1}', '[1, 2, 3]', 'v 0 0 0\nf 1 2 3\n', '["foo", 1]', '')):
                path = os.path.join(d, '%d.geo' % i)
                with open(path, 'w') as fh:
                    fh.write(text)
                with self.assertRaises(ValueError):
                    read_geo(path)
            path = os.path.join(d, 'bin.geo')
            with open(path, 'wb') as fh:
                fh.write(b'\x7fNSJb\x00\x01')
            with self.assertRaisesRegex(ValueError, 'binary'):
                read_geo(path)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_unknown_keys_and_dict_maps(self):
        data = ["fileversion", "20.0", "somethingnew", {"x": 1}, "pointcount", 3, "vertexcount", 3,
                "primitivecount", 2,
                "topology", {"pointref": {"indices": [0, 1, 2]}},
                "attributes", {"pointattributes": [[{"name": "P", "type": "numeric"},
                                                    {"size": 3, "storage": "fpreal64",
                                                     "values": {"size": 3, "storage": "fpreal64",
                                                                "tuples": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]}}],
                                                   [{"name": "weird", "type": "dictionary"}, {}]],
                               "primitiveattributes": [[{"name": "k", "type": "numeric"},
                                                        {"size": 1, "storage": "int32",
                                                         "values": {"size": 1, "storage": "int32",
                                                                    "arrays": [[5, 7]]}}]]},
                "primitives", [[{"type": "NURBCurve"}, {"vertex": [0, 1]}],
                               [["type", "Poly"], ["vertex", [0, 1, 2]]]]]
        g = parse_geo(data)
        self.assertEqual((g.num_points, g.num_prims), (3, 1))
        self.assertTrue(g.prim_closed[0])
        self.assertEqual(g.attribs['prim']['k'].tolist(), [7])
        self.assertNotIn('weird', g.attribs['point'])

    def test_file_sop_reads_geo(self):
        d = tempfile.mkdtemp()
        try:
            path = os.path.join(d, 'box.geo')
            write_geo(cook_sop('box'), path)
            g = cook_sop('file', file=path)
            self.assertEqual((g.num_points, g.num_prims), (8, 6))
            g = cook_sop('file', file=os.path.join(DATA, 'curves_strings.geo'))
            self.assertEqual(g.num_prims, 3)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
