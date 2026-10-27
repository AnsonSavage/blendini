# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Tests for the Blender independent parts of Blendini (geometry, VEX, SOPs).

Run from ``scripts/addons_core``:  python -m unittest blendini.tests.test_blendini
"""

import unittest

import numpy as np

from blendini.core.geometry import Geometry, merge
from blendini.sops import cook_sop
from blendini.vex.parser import parse, VexSyntaxError
from blendini.vex.wrangle import run_wrangle, find_channels


def _grid(n=5):
    return cook_sop('grid', rows=n, cols=n, size=(2.0, 2.0))


def _wrangle(code, geo=None, run_over='point', **kw):
    geo = geo if geo is not None else _grid()
    out, msgs = run_wrangle(code, [geo], run_over=run_over, **kw)
    return out, msgs


class TestParser(unittest.TestCase):
    def test_bindings(self):
        prog = parse('v@foo = @P; i[]@list = {1,2}; @Cd.r = 1;')
        self.assertEqual(prog.bindings['foo'], 'vector')
        self.assertEqual(prog.bindings['list'], 'int[]')
        self.assertIsNone(prog.bindings['Cd'])

    def test_function_definition(self):
        prog = parse('float sq(float x) { return x * x; } @P.y = sq(@P.x);')
        self.assertIn('sq', prog.functions)

    def test_syntax_error_has_line(self):
        with self.assertRaises(VexSyntaxError) as cm:
            parse('@P.y = 1;\n@P.x = ;')
        self.assertEqual(cm.exception.line, 2)

    def test_find_channels(self):
        chans = find_channels('float a = ch("amp"); vector c = chv("col"); int n = chi(\'num\'); ch("amp");')
        self.assertEqual(chans, [('amp', 'float'), ('col', 'vector'), ('num', 'int')])


class TestVex(unittest.TestCase):
    def test_simple_assignment(self):
        out, _ = _wrangle('@P.y = @P.x * 2;')
        np.testing.assert_allclose(out.P[:, 1], out.P[:, 0] * 2)

    def test_types_and_casts(self):
        out, _ = _wrangle('i@a = 7 / 2; f@b = 7.0 / 2; i@c = int(3.9); f@d = float(3) / 2; v@e = 1;')
        pa = out.attribs['point']
        self.assertTrue(np.all(pa['a'] == 3))
        self.assertTrue(np.all(pa['b'] == 3.5))
        self.assertTrue(np.all(pa['c'] == 3))
        self.assertTrue(np.all(pa['d'] == 1.5))
        np.testing.assert_allclose(pa['e'], 1.0)

    def test_if_else_masks(self):
        out, _ = _wrangle('if (@ptnum % 2 == 0) { i@even = 1; } else { i@even = -1; }')
        even = out.attribs['point']['even']
        self.assertTrue(np.all(even[::2] == 1))
        self.assertTrue(np.all(even[1::2] == -1))

    def test_loops_break_continue(self):
        code = '''
        int total = 0;
        for (int i = 0; i < 100; i++) {
            if (i % 2 == 1) continue;
            if (i > @ptnum) break;
            total += i;
        }
        i@total = total;
        int k = 0;
        while (k < @ptnum) { k += 3; }
        i@k = k;
        '''
        out, _ = _wrangle(code)
        pa = out.attribs['point']
        for pt in range(out.num_points):
            expect = sum(i for i in range(0, pt + 1, 2))
            self.assertEqual(pa['total'][pt], expect)
            k = 0
            while k < pt:
                k += 3
            self.assertEqual(pa['k'][pt], k)

    def test_user_function_by_reference(self):
        code = '''
        void bump(vector p; float amount) { p.y += amount; }
        float twice(float x) { if (x > 0) return x * 2; return 0; }
        vector q = @P;
        bump(q, 1.5);
        @P = q;
        f@t = twice(@P.x);
        '''
        g = _grid()
        out, _ = _wrangle(code, g)
        np.testing.assert_allclose(out.P[:, 1], g.P[:, 1] + 1.5)
        np.testing.assert_allclose(out.attribs['point']['t'], np.where(g.P[:, 0] > 0, g.P[:, 0] * 2, 0))

    def test_ternary_and_logic(self):
        out, _ = _wrangle('i@x = (@P.x > 0 && @P.z > 0) ? 1 : (@P.x < 0 || @P.z < 0) ? 2 : 3;')
        g = _grid()
        expect = np.where((g.P[:, 0] > 0) & (g.P[:, 2] > 0), 1, np.where((g.P[:, 0] < 0) | (g.P[:, 2] < 0), 2, 3))
        np.testing.assert_array_equal(out.attribs['point']['x'], expect)

    def test_vector_math(self):
        out, _ = _wrangle('v@n = normalize(set(3, 0, 4)); f@l = length({3, 4, 0}); v@c = cross({1,0,0}, {0,1,0}); '
                          'f@d = dot({1,2,3}, {4,5,6}); v@s = {1,2,3}.zyx;')
        pa = out.attribs['point']
        np.testing.assert_allclose(pa['n'][0], [0.6, 0, 0.8])
        np.testing.assert_allclose(pa['l'][0], 5)
        np.testing.assert_allclose(pa['c'][0], [0, 0, 1])
        np.testing.assert_allclose(pa['d'][0], 32)
        np.testing.assert_allclose(pa['s'][0], [3, 2, 1])

    def test_matrix_rotation(self):
        out, _ = _wrangle('matrix3 m = ident(); rotate(m, radians(90), {0, 1, 0}); v@r = {1, 0, 0} * m;')
        np.testing.assert_allclose(out.attribs['point']['r'][0], [0, 0, -1], atol=1e-9)

    def test_arrays(self):
        code = '''
        int a[] = array(3, 1, 2);
        append(a, @ptnum);
        i@n = len(a);
        i@first = a[0];
        i@last = a[-1];
        int s = 0;
        foreach (int x; a) s += x;
        i@sum = s;
        f@mx = max(a);
        '''
        out, _ = _wrangle(code)
        pa = out.attribs['point']
        self.assertTrue(np.all(pa['n'] == 4))
        self.assertTrue(np.all(pa['first'] == 3))
        np.testing.assert_array_equal(pa['last'], np.arange(out.num_points))
        np.testing.assert_array_equal(pa['sum'], 6 + np.arange(out.num_points))

    def test_strings(self):
        out, msgs = _wrangle('s@name = sprintf("piece%d", @ptnum); printf("%s\\n", s@name);')
        self.assertEqual(out.attribs['point']['name'][3], 'piece3')
        self.assertEqual(msgs[0], 'piece0\n')

    def test_point_lookup_reads_input(self):
        # point() always reads the unmodified input, like Houdini.
        out, _ = _wrangle('@P.y = 10; f@other = point(0, "P", (@ptnum + 1) % @numpt).y;')
        self.assertTrue(np.all(out.attribs['point']['other'] == 0))

    def test_nearpoints(self):
        out, _ = _wrangle('i@n = len(nearpoints(0, @P, 0.51));')
        g = _grid()
        d = np.linalg.norm(g.P[:, None] - g.P[None], axis=2)
        np.testing.assert_array_equal(out.attribs['point']['n'], (d <= 0.51).sum(axis=1))

    def test_add_and_remove_geometry(self):
        g = _grid(3)
        out, _ = _wrangle('int p = addpoint(0, @P + {0, 1, 0}); addprim(0, "polyline", @ptnum, p);'
                          'if (@ptnum == 0) removepoint(0, @ptnum);', g)
        # 9 original points, 9 new ones, minus the removed one.
        self.assertEqual(out.num_points, 17)

    def test_detail_wrangle_creates_geometry(self):
        out, _ = run_wrangle('for (int i = 0; i < 10; i++) addpoint(0, set(i, 0, 0));', [Geometry()],
                             run_over='detail')
        self.assertEqual(out.num_points, 10)

    def test_prim_wrangle(self):
        g = _grid()
        out, _ = _wrangle('f@area = primarea(0, @primnum); v@c = @P;', g, run_over='prim')
        np.testing.assert_allclose(out.attribs['prim']['area'], 0.25)
        np.testing.assert_allclose(out.attribs['prim']['c'], g.prim_centers())

    def test_groups(self):
        out, _ = _wrangle('if (@P.x > 0) i@group_right = 1;')
        g = _grid()
        np.testing.assert_array_equal(out.groups['point']['right'], g.P[:, 0] > 0)

    def test_setpointattrib_add_mode(self):
        out, _ = _wrangle('setpointattrib(0, "count", 0, 1, "add");')
        self.assertEqual(out.attribs['point']['count'][0], out.num_points)

    def test_channels(self):
        out, _ = _wrangle('@P.y = ch("amp");', channels={'amp': 2.5})
        self.assertTrue(np.all(out.P[:, 1] == 2.5))

    def test_rand_is_deterministic(self):
        a, _ = _wrangle('f@r = rand(@ptnum); v@v = rand(@ptnum);')
        b, _ = _wrangle('f@r = rand(@ptnum);')
        np.testing.assert_array_equal(a.attribs['point']['r'], b.attribs['point']['r'])
        self.assertEqual(a.attribs['point']['v'].shape[1], 3)
        self.assertTrue(np.all((a.attribs['point']['r'] >= 0) & (a.attribs['point']['r'] < 1)))


class TestSops(unittest.TestCase):
    def test_generators(self):
        self.assertEqual(cook_sop('box').num_points, 8)
        self.assertEqual(cook_sop('grid', rows=3, cols=4).num_prims, 6)
        self.assertEqual(cook_sop('line', points=7).num_points, 7)
        sphere = cook_sop('sphere')
        outward = np.sum(sphere.prim_normals() * sphere.prim_centers(), axis=1) > 0
        self.assertTrue(outward.all())

    def test_merge_attributes(self):
        a = cook_sop('color', [cook_sop('box')], color=(1.0, 0.0, 0.0))
        b = cook_sop('box')
        m = merge([a, b])
        self.assertEqual(m.num_points, 16)
        np.testing.assert_allclose(m.attribs['point']['Cd'][8:], 0)

    def test_copy_to_points(self):
        pts = cook_sop('pointwrangle', [_grid(3)], snippet='f@pscale = 2;')
        out = cook_sop('copytopoints', [cook_sop('box'), pts])
        self.assertEqual(out.num_points, 8 * 9)
        size = out.P[:8].max(axis=0) - out.P[:8].min(axis=0)
        np.testing.assert_allclose(size, 2)

    def test_scatter_count(self):
        self.assertEqual(cook_sop('scatter', [cook_sop('sphere')], npts=123).num_points, 123)

    def test_blast_expression(self):
        g = _grid()
        out = cook_sop('blast', [g], group='@P.x < 0', grouptype='point')
        self.assertTrue(np.all(out.P[:, 0] >= 0))

    def test_fuse(self):
        two = cook_sop('copyxform', [cook_sop('box')], ncy=2, t=(1.0, 0.0, 0.0))
        self.assertEqual(cook_sop('fuse', [two]).num_points, 12)

    def test_polyextrude(self):
        out = cook_sop('polyextrude', [cook_sop('grid', rows=2, cols=2)], dist=1.0)
        self.assertEqual(out.num_prims, 5)
        self.assertAlmostEqual(out.P[:, 1].max(), 1.0)

    def test_attrib_promote(self):
        g = cook_sop('pointwrangle', [_grid()], snippet='f@v = @ptnum;')
        out = cook_sop('attribpromote', [g], names='v', inclass='point', outclass='detail')
        self.assertAlmostEqual(float(out.attribs['detail']['v'][0]), np.mean(np.arange(g.num_points)))

    def test_connectivity(self):
        two = cook_sop('copyxform', [cook_sop('box')], ncy=3, t=(3.0, 0.0, 0.0))
        out = cook_sop('connectivity', [two])
        self.assertEqual(len(np.unique(out.attribs['prim']['class'])), 3)


TUTORIAL_SNIPPETS = {
 'sin': '@P.y += sin(@Time * 2 + @P.x) * chf("amp");',
 'dist': 'float dist = distance(@P, {0,0,0}); f@d = dist;',
 'curl': 'v@v = curlnoise(@P * 0.5 + @Time);',
 'avg': 'int npts[] = nearpoints(0, @P, 0.5); vector avg = 0; foreach(int pt; npts){ avg += point(0, "P", pt); } avg /= len(npts); @P = avg;',
 'remove': 'if(@P.y < 0) removepoint(0, @ptnum);',
 'pscale': '@pscale = fit01(rand(@ptnum + ch("seed")), 0.1, 0.3);',
 'orient': 'p@orient = quaternion(radians(ch("angle")), {0,1,0});',
 'xform': 'matrix m = maketransform(@N, {0,1,0}); 4@xform = m;',
 'name': 's@name = "piece" + itoa(@ptnum);',
 'dowhile': 'int i = 0; do { i++; } while (i < 5); i@i = i;',
 'hsv': '@Cd = hsvtorgb(set(rand(@ptnum), 1, 1));',
 'normalize': '@N = normalize(@P);',
 'opinput': '@P = lerp(@P, @opinput1_P, 0.5);',
 'xyz2': 'f@d = xyzdist(1, @P);',
 'getbbox': 'vector mn, mx; getbbox(0, mn, mx); v@size = mx - mn;',
 'addprim_arr': 'if (@ptnum == 0) { int pts[] = array(0, 1, 2); int prim = addprim(0, "polyline", pts); }',
 'setattr': 'setpointattrib(0, "Cd", @ptnum, {1,0,0}, "set");',
 'mulnoise': '@P *= 1 + noise(@P*5)*0.1;',
 'member': 'vector n = @N; n.y = 0; @N = n;',
 'atan2': 'float a = atan2(@P.z, @P.x); f@a = a;',
 'grpstr': 'string grp = "group_" + itoa(1); s@g = grp;',
 'primpts': 'int pts[] = primpoints(0, 0); vector pos = point(0, "P", pts[0]); v@p0 = pos;',
 'frame': 'float t = @Frame / 24.0; f@t = t;',
 'clamp': '@P.x = clamp(@P.x, -0.5, 0.5);',
 'npoints': 'int count = npoints(0); i@c = count;',
 'euler': 'vector4 q = eulertoquaternion(radians({0,90,0}), 0); p@orient = q;',
 'qmult': '@orient = qmultiply(@orient, quaternion(0.5, {0,1,0}));',
 'floatarr': 'float foo[] = {1.0, 2.0}; f@s = foo[1];',
 'vecarr': 'vector arr[]; arr[2] = {1,1,1}; i@l = len(arr);',
 'nbrs': 'i[]@nbrs = neighbours(0, @ptnum); i@nn = len(i[]@nbrs);',
 'grpwrite': '@group_top = @P.y > 0.5;',
 'smooth': 'float s = smooth(0, 1, @P.y); f@s = s;',
 'set2': 'vector2 uv = set(@P.x, @P.z); u@uv2 = uv;',
 'cmpwrite': '@Cd.r = @P.x > 0;',
 'emptyarr': 'int a[] = array(); push(a, 1); i@l = len(a);',
 'swz': '@P.xz = set(1, 2);',
 'castvec': 'vector v = vector(@P.x); v@v = v;',
 'intcast': 'int k = int(rand(@ptnum) * 10); i@k = k;',
 'primintr': 'f@area = primintrinsic(0, "measuredarea", 0);',
 'lookat': 'matrix3 m = lookat({0,0,0}, @N, {0,1,0}); v@x = {0,0,1} * m;',
 'abs_vec': 'v@a = abs(@P);',
 'min_vec': 'v@m = min(@P, {0,0,0});',
 'printf_f': 'if (@ptnum == 0) printf("%f %d %s %g\\n", @P.x, @ptnum, "hi", 1.5);',
 'chramp_vec': 'vector c = chramp("color", @P.y); @Cd = c;',
 'ternary_vec': '@Cd = @P.y > 0 ? {1,0,0} : {0,0,1};',
 'nested_fn': 'float sq(float x) { return x*x; } float cube(float x) { return sq(x) * x; } f@c = cube(@P.y);',
 'recursion': 'int fact(int n) { if (n <= 1) return 1; return n * fact(n - 1); } i@f = fact(5);',
 'while_break': 'int i = 0; while (1) { i++; if (i > 3) break; } i@i = i;',
 'string_cmp': 'string s = "abc"; if (s == "abc") i@ok = 1;',
 'vec_compare': 'if (@P == {0,1,0}) i@top = 1;',
 'int_div': 'i@d = @ptnum / 2;',
 'neg_mod': 'i@m = -7 % 3;',
 'sprintf_vec': 's@s = sprintf("%g", @P);',
 'relbbox': 'vector r = relbbox(0, @P); @Cd = r;',
 'removeprim': 'if (@ptnum == 0) removeprim(0, 0, 1);',
 'detail_read': 'f@x = detail(0, "missing");',
 'dot_dot': 'f@d = dot(@N, {0,1,0});',
 'const_pi': 'f@x = sin(@P.x * M_PI);',
 'iter_vec_fe': 'vector ps[] = {{1,2,3},{4,5,6}}; vector s = 0; foreach (vector p; ps) s += p; v@s = s;',
}


class TestTutorialSnippets(unittest.TestCase):
    """Typical wrangle snippets from Houdini tutorials must all run."""

    def test_snippets(self):
        g = cook_sop('normal', [cook_sop('sphere')])
        g2 = cook_sop('mountain', [g])
        for name, code in TUTORIAL_SNIPPETS.items():
            with self.subTest(name=name):
                run_wrangle(code, [g, g2], channels={'amp': 0.5, 'seed': 1, 'angle': 45})


class TestMoreSops(unittest.TestCase):
    def test_pcfilter_smooths(self):
        g = cook_sop('pointwrangle', [_grid(9)], snippet='f@v = @ptnum % 2;')
        out, _ = _wrangle('int h = pcopen(0, "P", @P, 0.6, 20); f@s = pcfilter(h, "v"); i@n = pcnumfound(h);', g)
        s = out.attribs['point']['s']
        self.assertTrue(np.all((s > 0) & (s < 1)))
        self.assertTrue(np.all(out.attribs['point']['n'] > 1))

    def test_sweep_outward(self):
        s = cook_sop('sweep', [cook_sop('line', points=5)], endcaps=False)
        c = s.prim_centers()
        c[:, 1] = 0
        self.assertTrue(np.all(np.sum(s.prim_normals() * c, axis=1) > 0))

    def test_lsystem(self):
        geo = cook_sop('lsystem', generations=3)
        self.assertGreater(geo.num_prims, 1)
        self.assertGreater(geo.P[:, 1].max(), 0)

    def test_pack_unpack_matches_copy(self):
        pts = cook_sop('pointwrangle', [_grid(3)], snippet='f@pscale = 0.5 + @ptnum * 0.1;')
        packed = cook_sop('copytopoints', [cook_sop('box'), pts], pack=True)
        self.assertEqual(packed.num_points, 9)
        moved = cook_sop('xform', [packed], r=(0.0, 45.0, 0.0), t=(1.0, 0.0, 0.0))
        real = cook_sop('xform', [cook_sop('copytopoints', [cook_sop('box'), pts])], r=(0.0, 45.0, 0.0),
                        t=(1.0, 0.0, 0.0))
        unpacked = cook_sop('unpack', [moved])
        np.testing.assert_allclose(np.sort(unpacked.P, axis=0), np.sort(real.P, axis=0), atol=1e-9)

    def test_split_pieces(self):
        from blendini.sops.modify import split_pieces
        three = cook_sop('copyxform', [cook_sop('box')], ncy=3, t=(3.0, 0.0, 0.0))
        pieces = split_pieces(three, 'class')
        self.assertEqual(len(pieces), 3)
        self.assertTrue(all(p.num_points == 8 for p in pieces))

    def test_obj_roundtrip(self):
        import os
        import tempfile
        from blendini.sops.modify import write_obj, read_obj
        geo = merge([cook_sop('sphere'), cook_sop('line', points=4)])
        path = os.path.join(tempfile.mkdtemp(), 'x.obj')
        write_obj(geo, path)
        back = read_obj(path)
        self.assertEqual((back.num_points, back.num_prims), (geo.num_points, geo.num_prims))
        np.testing.assert_allclose(back.P, geo.P, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
