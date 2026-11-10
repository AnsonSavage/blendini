# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Property-based tests for the geometry core: random sequences of operations must keep the
topology arrays, attributes and groups consistent (``Geometry.validate() == []``).

Run from ``scripts/addons_core``:  python -m unittest blendini.tests.test_geometry_props
"""

import random
import unittest

import numpy as np

from blendini.core.geometry import Geometry, merge, POINT, PRIM, VERTEX, DETAIL
from blendini.sops import cook_sop


def random_geo(rng, max_points=30, max_prims=12):
    npts = rng.randint(0, max_points)
    P = np.array([[rng.uniform(-1, 1) for _ in range(3)] for _ in range(npts)]).reshape(-1, 3)
    geo = Geometry.from_polys(P)
    if npts:
        polys = [[rng.randrange(npts) for _ in range(rng.randint(1, 6))] for _ in range(rng.randint(0, max_prims))]
        for poly in polys:
            geo.add_prims([poly], closed=rng.random() < 0.7)
    if rng.random() < 0.7:
        geo.add_attrib(POINT, 'Cd')[...] = rng.random()
    if rng.random() < 0.5:
        geo.set_attrib(POINT, 'id', np.arange(geo.num_points))
    if rng.random() < 0.5:
        geo.set_attrib(POINT, 'N', geo.point_normals())
    if rng.random() < 0.5:
        geo.add_attrib(POINT, 'name')[...] = 'piece%d' % rng.randint(0, 3)
    if rng.random() < 0.5:
        geo.set_attrib(PRIM, 'pv', np.arange(geo.num_prims, dtype=np.float64))
    if rng.random() < 0.5:
        geo.set_attrib(VERTEX, 'uv', np.full((geo.num_vertices, 3), rng.random()))
    if rng.random() < 0.3:
        geo.set_attrib(DETAIL, 'scalar', np.array([rng.random()]))
    for cls in (POINT, PRIM):
        if rng.random() < 0.5:
            geo.groups[cls]['g%s' % cls] = np.array([rng.random() < 0.5 for _ in range(geo.count(cls))], dtype=bool)
    return geo


def _mask(rng, n, p=0.3):
    return np.array([rng.random() < p for _ in range(n)], dtype=bool)


def _rand_matrix(rng):
    a = rng.uniform(0, 6.28)
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4)
    m[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) * rng.choice([1.0, 0.5, 2.0])
    m[0, 0] *= rng.choice([1.0, -1.0, 3.0])
    m[:3, 3] = [rng.uniform(-2, 2) for _ in range(3)]
    return m


def op_add_points(rng, geo):
    geo.add_points(np.array([[rng.uniform(-1, 1) for _ in range(3)] for _ in range(rng.randint(0, 5))]).reshape(-1, 3))
    return geo


def op_add_prims(rng, geo):
    if geo.num_points:
        polys = [[rng.randrange(geo.num_points) for _ in range(rng.randint(1, 5))] for _ in range(rng.randint(1, 4))]
        geo.add_prims(polys, closed=rng.random() < 0.5)
    return geo


def op_delete_prims(rng, geo):
    return geo.delete_prims(_mask(rng, geo.num_prims), keep_points=rng.random() < 0.5)


def op_delete_points(rng, geo):
    return geo.delete_points(_mask(rng, geo.num_points, 0.2))


def op_delete_unused(rng, geo):
    return geo.delete_unused_points()


def op_unique(rng, geo):
    return geo.unique_points()


def op_copy(rng, geo):
    return geo.copy()


def op_merge(rng, geo):
    return merge([geo, random_geo(rng)] if rng.random() < 0.5 else [random_geo(rng), geo, geo])


def op_transform(rng, geo):
    return geo.transform(_rand_matrix(rng))


def op_sort(rng, geo):
    modes = ['none', 'x', 'y', 'z', 'random', 'reverse', 'proximity', 'vector', 'shift']
    return cook_sop('sort', [geo], ptsort=rng.choice(modes), primsort=rng.choice(modes), seed=rng.randint(0, 9),
                    offset=rng.randint(-3, 3))


def op_fuse(rng, geo):
    if rng.random() < 0.5:
        return cook_sop('fuse', [geo], dist=rng.choice([0.001, 0.2, 0.5]))
    return cook_sop('fuse', [geo], mode='grid', gridsize=(0.5, 0.5, 0.5), consolidate=rng.random() < 0.5)


def op_blast(rng, geo):
    return cook_sop('blast', [geo], group=rng.choice(['0-3', '@P.y<0', '1 4 5', 'gpoint', 'gprim']),
                    grouptype=rng.choice(['point', 'prim']), negate=rng.random() < 0.3)


def op_delete_sop(rng, geo):
    return cook_sop('delete', [geo], entity=rng.choice(['point', 'prim']),
                    mode=rng.choice(['pattern', 'bbox', 'normal', 'random']))


def op_groupcreate(rng, geo):
    return cook_sop('groupcreate', [geo], groupname=rng.choice(['gpoint', 'gprim', 'other']),
                    grouptype=rng.choice(['point', 'prim']), mode=rng.choice(['all', 'pattern', 'bbox', 'random']))


def op_facet(rng, geo):
    return cook_sop('facet', [geo], unique=rng.random() < 0.5, cons=rng.random() < 0.5)


def op_clean(rng, geo):
    return cook_sop('clean', [geo], fusepts=rng.random() < 0.5)


def op_reverse(rng, geo):
    return cook_sop('reverse', [geo])


OPS = [op_add_points, op_add_prims, op_delete_prims, op_delete_points, op_delete_unused, op_unique, op_copy,
       op_merge, op_transform, op_sort, op_fuse, op_blast, op_delete_sop, op_groupcreate, op_facet, op_clean,
       op_reverse]


class TestValidate(unittest.TestCase):
    def test_valid_basics(self):
        self.assertEqual(Geometry().validate(), [])
        for sid in ('box', 'grid', 'sphere', 'line'):
            self.assertEqual(cook_sop(sid).validate(), [], sid)
        self.assertEqual(cook_sop('pack', [cook_sop('box')]).validate(), [])

    def test_detects_problems(self):
        def broken(fn):
            geo = cook_sop('box')
            fn(geo)
            return geo.validate()
        self.assertTrue(broken(lambda g: setattr(g, 'prim_start', g.prim_start + 1)))
        self.assertTrue(broken(lambda g: setattr(g, 'prim_start', g.prim_start[::-1].copy())))
        self.assertTrue(broken(lambda g: setattr(g, 'vtx_point', g.vtx_point + 100)))
        self.assertTrue(broken(lambda g: setattr(g, 'prim_closed', g.prim_closed[:-1])))
        self.assertTrue(broken(lambda g: g.attribs[POINT].__setitem__('Cd', np.zeros((3, 3)))))
        self.assertTrue(broken(lambda g: g.attribs[DETAIL].__setitem__('d', np.zeros(2))))
        self.assertTrue(broken(lambda g: g.attribs[POINT].__setitem__('P', np.zeros((8, 2)))))
        self.assertTrue(broken(lambda g: g.attribs[POINT]['P'].__setitem__((0, 0), np.nan)))
        self.assertTrue(broken(lambda g: g.groups[PRIM].__setitem__('x', np.zeros(2, dtype=bool))))
        self.assertTrue(broken(lambda g: g.groups[POINT].__setitem__('x', np.zeros(8))))
        self.assertTrue(broken(lambda g: g.attribs[POINT].__setitem__('__inst', np.zeros(8, dtype=np.int64))))


class TestRandomSequences(unittest.TestCase):
    def test_random_geometry_is_valid(self):
        rng = random.Random(1)
        for _ in range(50):
            self.assertEqual(random_geo(rng).validate(), [])

    def test_random_op_sequences(self):
        for seed in range(100):
            rng = random.Random(seed)
            geo = random_geo(rng)
            trace = []
            for _ in range(8):
                op = rng.choice(OPS)
                trace.append(op.__name__)
                geo = op(rng, geo)
                probs = geo.validate()
                self.assertEqual(probs, [], "seed %d after %s" % (seed, ' > '.join(trace)))

    def test_packed_copy_sequences(self):
        rng = random.Random(7)
        packed = cook_sop('pack', [cook_sop('box')])
        for _ in range(10):
            geo = merge([packed, random_geo(rng), packed])
            for op in (op_transform, op_sort, op_delete_points, op_copy):
                geo = op(rng, geo)
                self.assertEqual(geo.validate(), [], op.__name__)
            for sid, ins in (('copytopoints', [geo, cook_sop('box')]), ('copytopoints', [cook_sop('box'), geo]),
                             ('pointreplicate', [geo])):
                self.assertEqual(cook_sop(sid, ins).validate(), [], sid)


def _yrot(angle, scale=1.0, t=(0.0, 0.0, 0.0)):
    c, s = np.cos(angle), np.sin(angle)
    m = np.eye(4)
    m[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) * scale
    m[:3, 3] = t
    return m


class TestTypeInfo(unittest.TestCase):
    def _geo(self):
        geo = cook_sop('box')
        n = geo.num_points
        rng = np.random.default_rng(3)
        for name in ('dir', 'N', 'v', 'up', 'force', 'Cd', 'rest'):
            geo.set_attrib(POINT, name, rng.normal(size=(n, 3)))
        geo.set_attrib(PRIM, 'pdir', rng.normal(size=(geo.num_prims, 3)))
        q = rng.normal(size=(n, 4))
        geo.set_attrib(POINT, 'orient', q / np.linalg.norm(q, axis=1, keepdims=True))
        return geo

    def test_defaults(self):
        geo = Geometry()
        expect = {'P': 'point', 'N': 'normal', 'v': 'vector', 'up': 'vector', 'force': 'vector', 'accel': 'vector',
                  'orient': 'quaternion', 'Cd': 'color', 'dir': 'none', 'rest': 'none'}
        for name, t in expect.items():
            self.assertEqual(geo.attrib_typeinfo(POINT, name), t, name)
        geo.set_attrib_typeinfo(POINT, 'Cd', 'vector')
        self.assertEqual(geo.attrib_typeinfo(POINT, 'Cd'), 'vector')
        self.assertEqual(geo.attrib_typeinfo(VERTEX, 'Cd'), 'color')
        with self.assertRaises(ValueError):
            geo.set_attrib_typeinfo(POINT, 'Cd', 'matrix9')

    def test_transform_default_names_unchanged(self):
        geo = self._geo()
        m = _yrot(0.7, 2.0, (1, 2, 3))
        m[0, 0] *= 3.0
        rot = m[:3, :3]
        nm = np.linalg.inv(rot).T
        before = {k: v.copy() for k, v in geo.attribs[POINT].items()}
        pdir = geo.attribs[PRIM]['pdir'].copy()
        geo.transform(m)
        pa = geo.attribs[POINT]
        np.testing.assert_allclose(pa['P'], before['P'] @ rot.T + m[:3, 3])
        for name in ('v', 'up', 'force'):
            np.testing.assert_allclose(pa[name], before[name] @ rot.T)
        n = before['N'] @ nm.T
        np.testing.assert_allclose(pa['N'], n / np.linalg.norm(n, axis=1, keepdims=True))
        for name in ('dir', 'Cd', 'rest'):
            np.testing.assert_array_equal(pa[name], before[name])
        np.testing.assert_array_equal(geo.attribs[PRIM]['pdir'], pdir)

    def test_transform_custom_types(self):
        geo = self._geo()
        m = _yrot(0.5, 1.0, (1.0, 0.0, 0.0))
        rot = m[:3, :3]
        d0 = geo.attribs[POINT]['dir'].copy()
        p0 = geo.attribs[PRIM]['pdir'].copy()
        geo.set_attrib_typeinfo(POINT, 'dir', 'vector')
        geo.set_attrib_typeinfo(PRIM, 'pdir', 'point')
        geo.transform(m)
        np.testing.assert_allclose(geo.attribs[POINT]['dir'], d0 @ rot.T)
        np.testing.assert_allclose(geo.attribs[PRIM]['pdir'], p0 @ rot.T + m[:3, 3])
        geo.set_attrib_typeinfo(POINT, 'dir', 'normal')
        geo.transform(np.diag([2.0, 1.0, 1.0, 1.0]))
        self.assertTrue(np.allclose(np.linalg.norm(geo.attribs[POINT]['dir'], axis=1), 1.0))
        geo.set_attrib_typeinfo(POINT, 'dir', 'none')
        d1 = geo.attribs[POINT]['dir'].copy()
        geo.transform(m)
        np.testing.assert_array_equal(geo.attribs[POINT]['dir'], d1)
        # transform(vectors=False) only moves P.
        geo.set_attrib_typeinfo(POINT, 'dir', 'vector')
        geo.transform(m, vectors=False)
        np.testing.assert_array_equal(geo.attribs[POINT]['dir'], d1)

    def test_transform_quaternion(self):
        from blendini.core.instances import quat_to_matrix
        geo = self._geo()
        q0 = geo.attribs[POINT]['orient'].copy()
        m = _yrot(1.1, 3.0, (0, 5, 0))
        geo.transform(m)
        R = m[:3, :3] / 3.0
        np.testing.assert_allclose(quat_to_matrix(geo.attribs[POINT]['orient']),
                                   np.einsum('ij,njk->nik', R, quat_to_matrix(q0)), atol=1e-9)

    def test_preserved(self):
        import os
        import tempfile
        from blendini.core.geoio import save_geometry, load_geometry
        geo = self._geo()
        geo.set_attrib_typeinfo(POINT, 'dir', 'vector')
        geo.set_attrib_typeinfo(PRIM, 'pdir', 'normal')
        expect = dict(geo.typeinfo)
        self.assertEqual(geo.copy().typeinfo, expect)
        self.assertEqual(merge([cook_sop('grid'), geo]).typeinfo, expect)
        g2 = cook_sop('box')
        g2.set_attrib_typeinfo(POINT, 'dir', 'point')
        self.assertEqual(merge([g2, geo]).attrib_typeinfo(POINT, 'dir'), 'point')
        g = geo.copy()
        g.delete_points(np.arange(g.num_points) < 3)
        g.delete_prims(np.arange(g.num_prims) < 1)
        g.unique_points()
        self.assertEqual(g.typeinfo, expect)
        for sid in ('blast', 'sort', 'fuse', 'groupcreate'):
            self.assertEqual(cook_sop(sid, [geo]).typeinfo, expect, sid)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'g.npz')
            save_geometry(geo, path)
            self.assertEqual(load_geometry(path).typeinfo, expect)
            save_geometry(cook_sop('box'), path)
            self.assertEqual(load_geometry(path).typeinfo, {})
        g.remove_attrib(POINT, 'dir')
        self.assertNotIn((POINT, 'dir'), g.typeinfo)

    def test_random_sequences_keep_typeinfo(self):
        for seed in range(30):
            rng = random.Random(1000 + seed)
            geo = random_geo(rng)
            geo.set_attrib(POINT, 'dir', np.ones((geo.num_points, 3)))
            geo.set_attrib_typeinfo(POINT, 'dir', 'vector')
            for _ in range(6):
                op = rng.choice([op_delete_prims, op_delete_points, op_delete_unused, op_unique, op_copy, op_merge,
                                 op_transform, op_sort, op_fuse, op_blast, op_delete_sop, op_groupcreate])
                geo = op(rng, geo)
                self.assertEqual(geo.validate(), [])
                self.assertEqual(geo.attrib_typeinfo(POINT, 'dir'), 'vector', op.__name__)


if __name__ == '__main__':
    unittest.main()
