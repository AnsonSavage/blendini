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

    def test_channel_reference_paths(self):
        from blendini.bl.channels import referenced_paths
        code = 'f@a = ch("../grid1/sizex"); f@b = chf("amp"); v@c = chv(\'box1/t\'); f@d = ch("../grid1/sizex");'
        self.assertEqual(referenced_paths(code), ['../grid1/sizex', 'box1/t'])

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

    def test_structs(self):
        code = """
        struct Particle { vector pos; float mass = 2.0; vector vel = {0, 1, 0}; }
        Particle advance(Particle p; float dt) { p.pos += p.vel * dt; return p; }
        Particle a = Particle(@P);
        if (@ptnum % 2 == 0) a.mass = 10;
        Particle b = advance(a, 2.0);
        @P = b.pos;
        f@m = a.mass;
        Particle d = {@P, 3.0};
        d.vel.y = 5;
        f@dv = d.vel.y;
        f@dm = d.mass;
        """
        g = _grid()
        out, _ = _wrangle(code, g)
        np.testing.assert_allclose(out.P[:, 1], g.P[:, 1] + 2.0)
        np.testing.assert_array_equal(out.attribs['point']['m'], np.where(np.arange(g.num_points) % 2 == 0, 10.0, 2.0))
        self.assertTrue(np.all(out.attribs['point']['dv'] == 5) and np.all(out.attribs['point']['dm'] == 3))

    def test_struct_arrays_and_methods(self):
        code = """
        struct Inner { float w = 1; }
        struct Hit {
            int prim; Inner in;
            float score() { return in.w * 2; }
            void bump(float k) { in.w += k; }
        }
        Hit hits[];
        for (int i = 0; i < @ptnum; i++) {
            Hit h; h.prim = i; h.in.w = i * 10;
            if (i % 2 == 0) push(hits, h);
        }
        i@n = len(hits);
        float s = 0;
        foreach (int k; Hit h; hits) s += h->score() + k;
        f@s = s;
        if (len(hits) > 0) hits[0]->bump(7);
        f@w0 = len(hits) > 0 ? hits[0].in.w : -1;
        Hit last = pop(hits);
        i@lastprim = last.prim;
        Hit lit[] = { Hit(3), Hit(4) };
        insert(lit, 0, Hit(9));
        i@lit = lit[0].prim * 100 + len(lit);
        """
        out, msgs = _wrangle(code, _grid(3))
        pa = out.attribs['point']
        pt = np.arange(9)
        evens = [[i for i in range(p) if i % 2 == 0] for p in pt]
        np.testing.assert_array_equal(pa['n'], [len(e) for e in evens])
        np.testing.assert_allclose(pa['s'], [sum(20 * i + k for k, i in enumerate(e)) for e in evens])
        np.testing.assert_allclose(pa['w0'], np.where(pt > 0, 7.0, -1.0))
        np.testing.assert_array_equal(pa['lastprim'], [e[-1] if e else 0 for e in evens])
        self.assertTrue(np.all(pa['lit'] == 903))

    def test_more_builtins(self):
        code = r"""
        int h = primhedge(0, 0);
        i@dst = hedge_dstpoint(0, h);
        i@eq = hedge_equivcount(0, pointedge(0, 1, 4));
        i@npv = len(pointvertices(0, 4));
        s@re = re_replace("(\\d+)", "<\\1>", "piece12_a3");
        i@m = match("pie*", "piece") + re_match("piece[0-9]+", "piece7");
        vector e = {0.3, -0.7, 1.1};
        v@back = quaterniontoeuler(eulertoquaternion(e, 0), 0);
        matrix mm = ident(); scale(mm, {2, 3, 4}); translate(mm, {1, 2, 3});
        v@cs = cracktransform(0, 0, 2, {0, 0, 0}, mm);
        f@sp = spline("linear", 0.75, 0.0, 10.0, 20.0);
        int a[] = {1, 2, 3}; upush(a, 3); upush(a, 9); removevalue(a, 1); i@up = len(a);
        int pc = pcopen(0, "P", {0, 0, 0}, 1.1, 10);
        int c = 0;
        while (pciterate(pc)) c++;
        while (pciterate(pc)) c++;
        i@c = c;
        """
        g = cook_sop('grid', rows=3, cols=3, size=(2.0, 2.0))
        out, msgs = _wrangle(code, g, run_over='detail')
        d = out.attribs['detail']
        self.assertEqual((int(d['dst'][0]), int(d['eq'][0]), int(d['npv'][0])), (3, 2, 4))
        self.assertEqual(d['re'][0], 'piece<12>_a<3>')
        self.assertEqual(int(d['m'][0]), 2)
        np.testing.assert_allclose(d['back'][0], [0.3, -0.7, 1.1], atol=1e-9)
        np.testing.assert_allclose(d['cs'][0], [2, 3, 4])
        self.assertAlmostEqual(float(d['sp'][0]), 15.0)
        self.assertEqual(int(d['up'][0]), 3)
        self.assertEqual(int(d['c'][0]), 10)   # 5 points found, iterated twice

    def test_intersect(self):
        out, _ = run_wrangle('vector hp, uvw; i@hit = intersect(1, @P + {0, 5, 0}, {0, -10, 0}, hp, uvw); f@hy = hp.y;',
                             [cook_sop('grid', rows=5, cols=5, size=(4.0, 4.0)), cook_sop('box')], run_over='point')
        hits = out.attribs['point']['hit']
        self.assertEqual(int((hits >= 0).sum()), 1)
        self.assertAlmostEqual(float(out.attribs['point']['hy'][hits >= 0][0]), 0.5)

    def test_dict_hex_getattrib(self):
        code = r"""
        dict d; d["a"] = 1; d["name"] = "piece"; d["pos"] = {1, 2, 3};
        if (@ptnum % 2) d["a"] = @ptnum;
        i@a = d["a"]; s@nm = d["name"]; v@pos = d["pos"]; i@n = len(d);
        i@has = isvalidindex(d, "pos") * 10 + isvalidindex(d, "zz");
        i@h = 0xff;
        int ok; vector p = getattrib(0, "point", "P", 1, ok); i@ok = ok;
        """
        g = _grid(3)
        out, _ = _wrangle(code, g)
        pa = out.attribs['point']
        np.testing.assert_array_equal(pa['a'], np.where(np.arange(9) % 2, np.arange(9), 1))
        self.assertEqual((pa['nm'][0], int(pa['n'][0]), int(pa['has'][0]), int(pa['h'][0]), int(pa['ok'][0])),
                         ('piece', 3, 10, 255, 1))
        np.testing.assert_allclose(pa['pos'][0], [1, 2, 3])

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
        # Like Houdini, geometry without Cd gets the attribute's default (white).
        np.testing.assert_allclose(m.attribs['point']['Cd'][8:], 1)

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
 'overload': 'float twice(float x) { return x * 2; } vector twice(vector v) { return v * 3; } f@a = twice(1.5); v@b = twice({1,1,1});',
 'func_macro': '#define SQ(x) ((x) * (x))\n#define LERP(a, b, t) ((a) + ((b) - (a)) * (t))\nf@s = SQ(@P.x + 1); f@l = LERP(0, 10, 0.25);',
 'const_ternary': 'v@c = 1 ? {1,0,0} : {0,0,1}; f@f = (2 > 3) ? 1.0 : 0.5;',
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


class TestSmoke(unittest.TestCase):
    """Cook every SOP with its defaults and each enum / toggle variant."""

    def test_all_sops(self):
        from blendini.sops import REGISTRY, default_parms, CookContext, CookError
        box, grid, sphere = cook_sop('box'), cook_sop('grid'), cook_sop('sphere')
        line = cook_sop('line', points=10)
        pts = cook_sop('scatter', [sphere], npts=50)
        hf = cook_sop('heightfield', size=(2.0, 2.0), gridspacing=0.2)
        special = {'hfnoise': [hf], 'hferode': [hf], 'sweep': [line], 'polyframe': [line], 'resample': [line],
                   'copytopoints': [box, pts], 'attribtransfer': [grid, sphere], 'vellum': [grid, sphere],
                   'fluidsolver': [pts], 'ends': [cook_sop('circle')]}
        for sop_id, sop in sorted(REGISTRY.items()):
            inputs = special.get(sop_id, [sphere, box, grid, pts][:sop.inputs])
            variants = [{}]
            for parm in sop.parms:
                if parm.type == 'enum':
                    variants += [{parm.name: item[0]} for item in parm.items[1:]]
                elif parm.type == 'bool':
                    variants.append({parm.name: not parm.default})
            for v in variants:
                with self.subTest(sop=sop_id, variant=v):
                    p = default_parms(sop_id)
                    p.update(v)
                    ctx = CookContext(frame=3, time=2 / 24, state={})
                    ctx.channels, ctx.ramps = {}, {}
                    try:
                        self.assertIsNotNone(sop.cook(list(inputs), p, ctx))
                    except CookError:
                        pass


class TestPresets(unittest.TestCase):
    def test_presets_run(self):
        from blendini.vex.presets import PRESETS
        g = cook_sop('normal', [cook_sop('sphere')])
        channels = {'amp': 0.3, 'freq': 2.0, 'seed': 0.0, 'min_scale': 0.05, 'max_scale': 0.2, 'keep': 0.5,
                    'radius': 0.5, 'maxpts': 5, 'speed': 1.0, 'degrees_per_unit': 45.0, 'count': 50, 'turns': 4.0,
                    'height': 3.0, 'distance': 0.1}
        for name, code in PRESETS:
            with self.subTest(preset=name):
                run_over = 'prim' if 'Primitive' in name else 'detail' if '(Detail)' in name else 'point'
                run_wrangle(code, [g], run_over=run_over, channels=channels, time=1.0, frame=25)


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

    def test_vellum_pins_and_ground(self):
        from blendini.sops import REGISTRY, default_parms, CookContext
        cloth = cook_sop('xform', [cook_sop('grid', rows=8, cols=8, size=(2.0, 2.0))], t=(0.0, 1.0, 0.0))
        sop = REGISTRY['vellum']
        p = default_parms('vellum')
        p['pingroup'] = '0'
        state = {}
        for f in range(1, 40):
            out = sop.cook([cloth, None], p, CookContext(time=(f - 1) / 24, frame=f, fps=24, state=state))
        np.testing.assert_allclose(out.P[0], cloth.P[0])
        self.assertGreaterEqual(out.P[:, 1].min(), p.groundy + p.thickness - 1e-6)
        self.assertLess(out.P[:, 1].mean(), 1.0)

    def test_fluid_neighbours_and_containment(self):
        from blendini.sops.fluid import neighbour_pairs
        from blendini.sops import REGISTRY, default_parms, CookContext
        rng = np.random.RandomState(0)
        P = rng.random_sample((300, 3))
        I, J = neighbour_pairs(P, 0.2)
        d = np.linalg.norm(P[:, None] - P[None], axis=2)
        self.assertEqual(len(I), int(((d < 0.2) & ~np.eye(300, dtype=bool)).sum()))
        g = np.arange(0, 0.5, 0.1) + 0.05
        blk = Geometry.from_polys(np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3))
        p = default_parms('fluidsolver')
        p['boxmin'], p['boxmax'] = (0.0, 0.0, 0.0), (1.0, 2.0, 1.0)
        state = {}
        for f in range(1, 20):
            out = REGISTRY['fluidsolver'].cook([blk], p, CookContext(time=(f - 1) / 24, frame=f, fps=24, state=state))
        self.assertTrue(np.all(out.P >= -1e-9) and np.all(out.P <= np.array([1.0, 2.0, 1.0]) + 1e-9))

    def test_heightfield(self):
        hf = cook_sop('heightfield', size=(4.0, 4.0), gridspacing=0.1)
        n = cook_sop('hfnoise', [hf])
        e = cook_sop('hferode', [n], iterations=5)
        self.assertEqual(e.num_points, hf.num_points)
        self.assertGreater(np.ptp(n.P[:, 1]), 0.1)

    def test_attribute_interpolate_sticks_to_deforming_surface(self):
        from blendini.core.primuv import prim_uv
        src = cook_sop('grid', rows=6, cols=6)
        pts = cook_sop('scatter', [src], npts=200)
        np.testing.assert_allclose(cook_sop('attribinterpolate', [pts, src]).P, pts.P, atol=1e-9)
        deformed = cook_sop('mountain', [src], height=0.5)
        moved = cook_sop('attribinterpolate', [pts, deformed])
        uv = prim_uv(deformed, pts.attribs['point']['sourceprim'], moved.P)
        np.testing.assert_allclose(uv[:, :2], pts.attribs['point']['sourceprimuv'][:, :2], atol=1e-6)

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


class TestEverydaySops(unittest.TestCase):
    def test_enumerate_and_group_range(self):
        g = _grid()
        pieces = cook_sop('connectivity', [merge([g, cook_sop('box')])], class_='prim')
        e = cook_sop('enumerate', [pieces], mode='inpiece', pieceattrib='class')
        idx = e.attribs['prim']['index']
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[g.num_prims], 0)
        r = cook_sop('grouprange', [g], select=1, of=3, offset=1)
        self.assertEqual(np.nonzero(r.groups['prim']['range1'])[0][:3].tolist(), [1, 4, 7])

    def test_switch_if(self):
        g = _grid()
        self.assertEqual(cook_sop('switchif', [g, Geometry()], expr='npoints(0) == %d' % g.num_points).num_points,
                         g.num_points)
        self.assertEqual(cook_sop('switchif', [g, Geometry()], test='attrib', attrib='nope').num_points, 0)

    def test_revolve_radius(self):
        prof = Geometry.from_polys([[1, 0, 0], [2, 1, 0], [1, 2, 0]], [[0, 1, 2]], closed=False)
        rv = cook_sop('revolve', [prof], divs=8)
        r = np.hypot(rv.P[:, 0], rv.P[:, 2])
        self.assertTrue(np.allclose(np.sort(np.unique(np.round(r, 6))), [1, 2]))
        self.assertEqual(rv.num_prims, 16)

    def test_path_deform_follows_arc(self):
        box = cook_sop('box', size=(0.2, 0.2, 4.0))
        arc = Geometry.from_polys([[np.cos(a) * 2, 0, np.sin(a) * 2] for a in np.linspace(0, np.pi, 64)],
                                  [list(range(64))], closed=False)
        out = cook_sop('pathdeform', [box, arc], stretch=True)
        r = np.hypot(out.P[:, 0], out.P[:, 2])
        self.assertTrue(np.all(np.abs(r - 2) < 0.12))

    def test_distance_and_lattice(self):
        g = _grid()
        d = cook_sop('distancefromgeo', [g, cook_sop('box')], maxdist=2.0)
        dist = d.attribs['point']['dist']
        centre = int(np.argmin(np.linalg.norm(g.P, axis=1)))
        self.assertAlmostEqual(float(dist[centre]), 0.5, places=5)       # inside: distance to the faces
        self.assertAlmostEqual(float(dist.max()), np.sqrt(0.5), places=5)  # grid corner to box edge
        from blendini.sops.extra import LatticeSOP
        s = cook_sop('sphere')
        _axes, rp = LatticeSOP.grid(*s.bbox(), (2, 2, 2))
        out = cook_sop('lattice', [s, Geometry.from_polys(rp), Geometry.from_polys(rp * [2, 1, 1])])
        self.assertTrue(np.allclose(out.P, s.P * [2, 1, 1], atol=1e-6))

    def test_connect_adjacent_pieces(self):
        boxes = merge([cook_sop('box', t=(i * 1.1, 0, 0)) for i in range(4)] + [cook_sop('box', t=(20, 0, 0))])
        net = cook_sop('connectadjacentpieces', [boxes], constraint_type='hinge')
        self.assertEqual(net.num_points, 5)
        self.assertEqual(net.prims_as_lists(), [[0, 1], [1, 2], [2, 3]])   # the far box stays free
        self.assertEqual(list(net.attribs['prim']['constraint_type']), ['hinge'] * 3)
        chain = cook_sop('connectadjacentpieces', [boxes], mode='chain')
        self.assertEqual(chain.num_prims, 4)

    def test_sop_batch_two(self):
        from blendini.vex.wrangle import run_wrangle as rw
        g = cook_sop('grid', rows=3, cols=3)
        t = cook_sop('uvtransform', [cook_sop('uvtexture', [g])], t=(1, 0, 0), s=(2, 2, 1), r=90)
        np.testing.assert_allclose(t.attribs['vertex']['uv'][1], [0, 0, 0], atol=1e-9)
        a = cook_sop('attribadjustfloat', [g], pattern='random', fmin=2.0, fmax=3.0)
        ps = a.attribs['point']['pscale']
        self.assertTrue(np.all((ps >= 2) & (ps <= 3)) and ps.std() > 0)
        m = cook_sop('attribadjustvector', [g], attrib='v', op='add', value=(0, 2, 0))
        np.testing.assert_allclose(m.attribs['point']['v'][:, 1], 2.0)
        boxes = merge([cook_sop('box', t=(i * 2, 0, 0)) for i in range(3)])
        c = cook_sop('extractcentroid', [boxes])
        np.testing.assert_allclose(c.P[:, 0], [0, 2, 4], atol=1e-9)
        sided, _ = rw('i@side = @P.x > 0;', [g], run_over='prim')
        b = cook_sop('groupfromattribboundary', [sided], attrib='side')
        np.testing.assert_allclose(np.abs(g.P[b.groups['point']['boundary'], 0]), 0, atol=1e-9)
        o = cook_sop('occlusion', [merge([cook_sop('grid', rows=6, cols=6, size=(4.0, 4.0)),
                                          cook_sop('box', t=(0, 0.5, 0))])], tocolor=False)
        occ = o.attribs['point']['occlusion'][:36].reshape(6, 6)
        self.assertGreater(occ[2:4, 2:4].mean(), 0.5)
        self.assertEqual(occ[0].sum() + occ[-1].sum(), 0)

    def test_volumes(self):
        from blendini.core.volume import surface_nets
        v = cook_sop('volume', size=(2.0, 2.0, 2.0), voxelsize=0.1, usebounds=False)
        w = cook_sop('volumewrangle', [v], snippet='@density = 0.8 - length(@P); v@vel = set(0, 1, 0);')
        names = [x.name for x in w.volumes]
        self.assertEqual(names, ['density', 'vel'])
        self.assertTrue(w.volumes[1].is_vector)
        pts, quads = surface_nets(w.volumes[0], 0.0, inside_below=False)
        r = np.linalg.norm(pts, axis=1)
        self.assertLess(abs(r.mean() - 0.8), 0.02)
        sdf = cook_sop('volumefrompolygons', [cook_sop('box')], voxelsize=0.1)
        out, _ = run_wrangle('f@d = volumesample(1, "surface", @P); v@g = volumegradient(1, 0, @P + {0.2, 0, 0});',
                             [Geometry.from_polys([[0, 0, 0], [0.7, 0, 0]]), sdf], run_over='point')
        d = out.attribs['point']['d']
        self.assertLess(d[0], -0.4)                     # box center is inside
        self.assertAlmostEqual(float(d[1]), 0.2, delta=0.05)
        self.assertGreater(out.attribs['point']['g'][0][0], 0.5)
        conv = cook_sop('convertvolume', [sdf])
        self.assertTrue(np.allclose(np.abs(conv.P).max(axis=1), 0.5, atol=0.06))
        moved = cook_sop('xform', [sdf], t=(2.0, 0, 0))
        np.testing.assert_allclose(moved.volumes[0].origin, sdf.volumes[0].origin + [2, 0, 0])
        blurred = cook_sop('volumeblur', [w], group='density')
        self.assertLess(blurred.volumes[0].data.max(), w.volumes[0].data.max())
        pts = Geometry.from_polys([[0, 0, 0], [1, 0, 0]])
        r = cook_sop('volumerasterizepoints', [pts], radius=0.3)
        self.assertGreater(r.volumes[0].sample([[0, 0, 0]])[0], 0.5)
        self.assertEqual(merge([w, sdf]).volumes[-1].name, 'surface')

    def test_pyro_smoke_rises(self):
        from blendini.sops import REGISTRY, default_parms, CookContext
        src = cook_sop('sphere', rad=(0.3, 0.3, 0.3), t=(0, 0.4, 0))
        p = default_parms('pyrosolver')
        p.update(voxelsize=0.12, pressureiters=10)
        st = {}
        heights = []
        for f in (1, 8, 16):
            out = REGISTRY['pyrosolver'].cook([src], p, CookContext(frame=f, time=(f - 1) / 24, state=st))
            if out.volumes:
                d = out.volumes[0]
                ys = d.origin[1] + (np.arange(d.res[1]) + 0.5) * d.voxelsize
                heights.append(float((d.data.sum(axis=(0, 2)) * ys).sum() / d.data.sum()))
                self.assertEqual([v.name for v in out.volumes], ['density', 'temperature', 'vel'])
        self.assertEqual(len(heights), 2)
        self.assertGreater(heights[1], heights[0] + 0.05)
        p.update(fire=True)
        st = {}
        for f in (1, 10):
            out = REGISTRY['pyrosolver'].cook([src], p, CookContext(frame=f, time=(f - 1) / 24, state=st))
        names = [v.name for v in out.volumes]
        self.assertIn('flame', names)
        self.assertGreater(out.volumes[names.index('flame')].data.max(), 0.5)

    def test_geometry_files_and_file_cache(self):
        import os
        import tempfile
        from blendini.core.geoio import save_geometry, load_geometry, expand_path
        from blendini.sops import REGISTRY, default_parms, CookContext
        w, _ = run_wrangle('s@name = "p" + itoa(@ptnum); v@v = {1, 2, 3};', [_grid()], run_over='point')
        w.groups['point']['top'] = w.P[:, 2] > 0
        vol = cook_sop('volumewrangle', [cook_sop('volume', voxelsize=0.25)], snippet='@density = @P.x;')
        full = merge([w, vol, cook_sop('pack', [merge([cook_sop('box'), cook_sop('sphere', t=(3, 0, 0))])])])
        path = os.path.join(tempfile.mkdtemp(), 'g.npz')
        save_geometry(full, path)
        back = load_geometry(path)
        self.assertEqual((back.num_points, back.num_prims, len(back.instances)), (full.num_points, full.num_prims, 2))
        self.assertEqual(list(back.attribs['point']['name'][:2]), ['p0', 'p1'])
        np.testing.assert_allclose(back.volumes[0].data, full.volumes[0].data)
        self.assertEqual(int(back.groups['point']['top'].sum()), int(full.groups['point']['top'].sum()))
        self.assertEqual(expand_path('$HIP/c/$OS.$F4.npz', 12, 'fc', '/p'), '/p/c/fc.0012.npz')
        p = default_parms('filecache')
        p.update(file=os.path.join(os.path.dirname(path), 'g.npz'), load=True)
        out = REGISTRY['filecache'].cook([cook_sop('box')], p, CookContext(frame=3))
        self.assertEqual(out.num_points, full.num_points)

    def test_material_sop(self):
        out = cook_sop('material', [cook_sop('box')], group='0-2', material='Red')
        self.assertEqual(list(out.attribs['prim']['shop_materialpath']), ['Red'] * 3 + [''] * 3)

    def test_heightfield_mask_scatter(self):
        hf = cook_sop('hfnoise', [cook_sop('heightfield', size=(8.0, 8.0), gridspacing=0.25)], amp=2.0, elementsize=4.0)
        m = cook_sop('hfmaskbyfeature', [hf], slopemax=20.0)
        mask = m.attribs['point']['mask']
        self.assertTrue(0 < mask.mean() < 1)
        s = cook_sop('hfscatter', [m], count=200)
        self.assertEqual(s.num_points, 200)
        self.assertGreater(s.attribs['point']['mask'].mean(), mask.mean())
        self.assertTrue({'N', 'up', 'pscale'} <= set(s.attribs['point']))
        b = cook_sop('hfblur', [hf], iterations=8)
        self.assertLess(b.P[:, 1].std(), hf.P[:, 1].std())
        t = cook_sop('hfterrace', [hf], spacing=0.5, sharpness=1.0)
        frac = np.mod(t.P[:, 1], 0.5)
        self.assertGreater(np.mean((frac < 0.05) | (frac > 0.45)), 0.5)   # mostly flat treads

    def test_vellum_hair_bend(self):
        from blendini.sops import REGISTRY, default_parms, CookContext
        pts = [[i * 0.02, i * 0.1, 0] for i in range(10)]
        g = Geometry.from_polys(pts, [list(range(10))], closed=False)
        tips = []
        for bend in (0.0, 1.0):
            p = default_parms('vellum')
            p.update(pingroup='@P.y < 0.15', bendstiffness=bend, ground=False)
            st = {}
            for f in (1, 30):
                out = REGISTRY['vellum'].cook([g], p, CookContext(frame=f, time=(f - 1) / 24, state=st))
            tips.append(out.P[9, 1])
        self.assertGreater(tips[1], tips[0] + 0.3)

    def test_skin(self):
        rows = Geometry.from_polys([[x, y, 0] for y in range(3) for x in range(4)],
                                   [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]], closed=False)
        out = cook_sop('skin', [rows])
        self.assertEqual(out.num_prims, 6)


if __name__ == '__main__':
    unittest.main()
