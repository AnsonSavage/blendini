# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
VEX fidelity audit: realistic Point Wrangle snippets checked against Houdini's behaviour.

Run from ``scripts/addons_core``:  python -m unittest blendini.tests.test_vex_audit
"""

import unittest

import numpy as np

from blendini.sops import cook_sop
from blendini.vex.parser import VexSyntaxError
from blendini.vex.values import VexError
from blendini.vex.wrangle import run_wrangle


def _grid():
    return cook_sop('grid', rows=4, cols=4, size=(2.0, 2.0))


def _run(code, geo=None, **kw):
    out, msgs = run_wrangle(code, [geo if geo is not None else _grid()], run_over='point', **kw)
    return out


def _pt(out, name, i=0):
    return out.attribs['point'][name][i]


# Snippets that must run without errors (typical tutorial wrangles).
SNIPPETS = [
    'float d = length(@P); @P.y = sin(d * 10 - @Time) * 0.1; @Cd = set(d, 0, 1 - d);',
    'if (@P.x > 0) i@group_right = 1; setpointgroup(0, "top", @ptnum, @P.z > 0);',
    'v@N = normalize(@P); f@pscale = fit01(rand(@ptnum), 0.1, 0.5); p@orient = quaternion(maketransform(@N, {0,1,0}));',
    'int pts[] = nearpoints(0, @P, 1.0, 5); vector a = 0; foreach (int p; pts) a += point(0, "P", p); v@a = a / len(pts);',
    'int h = pcopen(0, "P", @P, 1.0, 8); vector c = 0; while (pciterate(h)) { vector p; pcimport(h, "P", p); c += p; }',
    'int h = pcopen(0, "P", @P, 1.0, 8); v@sm = pcfilter(h, "P"); i@nf = pcnumfound(h);',
    'if (@ptnum == 0) { int a = addpoint(0, {0,5,0}); int b = addpoint(0, {1,5,0}); addprim(0, "polyline", a, b); }',
    'if (@P.x < 0) removepoint(0, @ptnum);',
    's@name = sprintf("piece%d", @ptnum); string parts[] = split("a,b,c", ","); s@j = join(parts, "-");',
    'v@v = curlnoise(@P); @P += v@v * 0.1;',
    'int fib(int n) { if (n < 2) return n; return fib(n - 1) + fib(n - 2); } i@f = fib(@ptnum % 8);',
    'void addone(int a[]) { for (int i = 0; i < len(a); i++) a[i] += 1; } int x[] = {1,2}; addone(x); i@x = x[1];',
    'struct Pair { float a; float b; } Pair p = Pair(1, 2); f@s = p.a + p.b;',
    'dict d; d["a"] = 1; i@x = d["a"];',
    'int pr; vector uv; f@d = xyzdist(0, @P + {0,1,0}, pr, uv);',
    '#include <math.h>\n#include "voplib.h"\n@P.y = M_PI;',
    'vector4 q = eulertoquaternion(radians({0,90,0}), 0); v@e = degrees(quaterniontoeuler(q, 0));',
    'int s = 0; for (int i = 0, j = 10; i < j; i++, j--) s += 1; i@s = s;',
]


class TestVexSnippets(unittest.TestCase):
    def test_snippets_run(self):
        for code in SNIPPETS:
            with self.subTest(code=code):
                _run(code)


class TestVexAudit(unittest.TestCase):
    def test_compound_assign_on_array_element(self):
        out = _run('int a[] = {1,2,3}; a[1] += 5; a[0]++; i@r = a[1]; i@r2 = a[0];')
        self.assertEqual((_pt(out, 'r'), _pt(out, 'r2')), (7, 2))

    def test_array_reductions(self):
        out = _run('float a[] = {3.0, 1.0, 2.0}; f@mn = min(a); f@mx = max(a); f@s = sum(a); f@av = avg(a);'
                   'f@p = product(a); int b[] = {4,2,9}; i@bm = max(b);')
        self.assertEqual([_pt(out, k) for k in ('mn', 'mx', 's', 'av', 'p', 'bm')], [1, 3, 6, 2, 6, 9])

    def test_sort_argsort_reorder(self):
        out = _run('int a[] = {30,10,20}; int o[] = argsort(a); int r[] = reorder(a, o); int s[] = sort(a);'
                   'i@o0 = o[0]; i@r2 = r[2]; i@s0 = s[0]; string t[] = sort(array("b","c","a")); s@t0 = t[0];')
        self.assertEqual((_pt(out, 'o0'), _pt(out, 'r2'), _pt(out, 's0'), _pt(out, 't0')), (1, 30, 10, 'a'))

    def test_find(self):
        out = _run('i@f = find("hello world", "wor"); i@g = find(array("a","b","c"), "c");'
                   'int a[] = {1,2,1,3,1}; int all[] = find(a, 1); i@n = len(all); i@fs = find("abcabc", "c", 3);'
                   'i@miss = find(a, 7);')
        self.assertEqual([_pt(out, k) for k in ('f', 'g', 'n', 'fs', 'miss')], [6, 2, 3, 5, -1])

    def test_slices(self):
        out = _run('int a[] = {1,2,3,4,5}; int b[] = a[1:3]; i@n = len(b); i@b0 = b[0];'
                   'int r[] = a[::-1]; i@r0 = r[0]; int t[] = a[-2:]; i@t0 = t[0]; s@s = "hello"[1:3];')
        self.assertEqual([_pt(out, k) for k in ('n', 'b0', 'r0', 't0', 's')], [2, 2, 5, 4, 'el'])

    def test_ternary_inside_index_is_not_a_slice(self):
        out = _run('int a[] = {10, 20}; i@v = a[@ptnum == 0 ? 1 : 0];')
        self.assertEqual(_pt(out, 'v', 0), 20)
        self.assertEqual(_pt(out, 'v', 1), 10)

    def test_foreach_string_array_and_do_while(self):
        out = _run('string s[] = {"x","yy"}; string o = ""; int t = 0;'
                   'foreach (int i; string v; s) { o += sprintf("%d%s", i, v); t += strlen(v); }'
                   's@o = o; i@t = t; int k = 0; do { k++; } while (k < 5); i@k = k;')
        self.assertEqual((_pt(out, 'o'), _pt(out, 't'), _pt(out, 'k')), ('0x1yy', 3, 5))

    def test_matrix_components(self):
        out = _run('matrix3 m = ident(); m.xy = 2; m.zz += 4; f@xy = m.xy; f@zz = m.zz; f@g = getcomp(m, 0, 1);'
                   'matrix m4 = ident(); m4.xw = 3; f@xw = getcomp(m4, 0, 3); f@ww = m4.ww;'
                   'setcomp(m4, 7, 2, 3); f@c = getcomp(m4, 2, 3); matrix2 m2 = ident(); f@m2 = m2.yx;')
        self.assertEqual([_pt(out, k) for k in ('xy', 'zz', 'g', 'xw', 'ww', 'c', 'm2')], [2, 5, 2, 3, 1, 7, 0])

    def test_matrix_component_errors(self):
        with self.assertRaises(VexError):
            _run('matrix3 m = ident(); f@a = m.xw;')

    def test_vector4_components(self):
        out = _run('vector4 q = {1,2,3,4}; f@w = q.w; f@x = getcomp(q, 0);')
        self.assertEqual((_pt(out, 'w'), _pt(out, 'x')), (4, 1))

    def test_integer_division_and_modulo(self):
        out = _run('i@a = -7 / 2; i@b = -7 % 2; f@c = -7.5 % 2; f@d = 1 / 2; i@e = 7 / 0;')
        self.assertEqual([_pt(out, k) for k in ('a', 'b', 'c', 'd', 'e')], [-3, -1, -1.5, 0, 0])

    def test_sprintf_formats(self):
        out = _run('s@a = sprintf("%g", 0.5); s@b = sprintf("%5.2f|", 3.14159); s@c = sprintf("%s", {1,2,3});'
                   's@d = sprintf("%03d", 7); s@e = sprintf("%-4d|", 7); s@f = sprintf("%d%%", 5);'
                   's@g = sprintf("[%-5s][%5s]", "ab", "cd"); s@h = sprintf("%.1f", {1,2,3}); s@i = sprintf("%x", 255);')
        self.assertEqual([_pt(out, k) for k in 'abcdefghi'],
                         ['0.5', ' 3.14|', '{1, 2, 3}', '007', '7   |', '5%', '[ab   ][   cd]', '{1.0, 2.0, 3.0}', 'ff'])

    def test_fit_edge_cases(self):
        out = _run('f@a = fit(5, 0, 10, 1, 0); f@b = fit(-1, 0, 10, 0, 1); f@c = fit(1, 1, 1, 0, 1);'
                   'f@d = fit(0.5, 1, 0, 0, 10); f@e = fit(2, 1, 0, 0, 10); v@v = fit(0.5, 0, 1, {0,0,0}, {2,4,6});')
        self.assertEqual([_pt(out, k) for k in 'abcde'], [0.5, 0, 0.5, 5, 0])
        np.testing.assert_allclose(_pt(out, 'v'), [1, 2, 3])

    def test_chramp_clamps(self):
        out = _run('f@r = chramp("ramp", 0.25); f@hi = chramp("ramp", 1.5);', ramps={'ramp': [(0, 0), (1, 1)]})
        self.assertAlmostEqual(_pt(out, 'r'), 0.25)
        self.assertAlmostEqual(_pt(out, 'hi'), 1.0)

    def test_worley_noise(self):
        out = _run('int seed; float f1, f2; wnoise(@P * 3, seed, f1, f2); f@f1 = f1; f@f2 = f2; i@seed = seed;'
                   'float g1, g2; vector p1, p2; int s2; vnoise(@P * 3, 1, s2, g1, g2, p1, p2);'
                   'f@g1 = g1; f@chk = distance(p1, @P * 3); float m1, m2; int s3; mwnoise(@P * 3, s3, m1, m2);'
                   'f@m1 = m1;')
        a = out.attribs['point']
        self.assertTrue(np.all(a['f1'] <= a['f2']))
        self.assertTrue(np.all(a['f1'] > 0) and np.all(a['f1'] < 1.8))
        np.testing.assert_allclose(a['f1'], a['g1'])
        np.testing.assert_allclose(a['chk'], a['g1'], atol=1e-9)
        self.assertTrue(np.all(a['m1'] >= a['f1'] - 1e-9))   # Manhattan distance >= Euclidean
        self.assertGreater(len(np.unique(a['seed'])), 1)
        # Not Perlin any more: wnoise is a distance, zero only at feature points.
        out2 = _run('int s; float f1, f2; wnoise(@P * 3, s, f1, f2, 2, 2, 2); f@f1 = f1;')
        self.assertTrue(np.all(out2.attribs['point']['f1'] >= 0))

    def test_array_attributes_round_trip(self):
        g = _run('i[]@ids = array(@ptnum, 5); s[]@tags = array("a", itoa(@ptnum)); v[]@vs = array(@P, @P * 2);'
                 'f[]@wt = {0.5};')
        out = _run('i@n = len(@ids); i@x = @ids[1]; s@t = @tags[1]; v@v1 = @vs[1]; @ids[0] += 10; f@w0 = @wt[0];'
                   'int q[] = i[]@ids; i@q = q[0]; i@at = attribtype(0, "point", "ids");'
                   'i@as = attribtype(0, "point", "tags");', g)
        self.assertEqual([_pt(out, k, 1) for k in ('n', 'x', 't', 'q', 'at', 'as')], [2, 5, '1', 11, 3, 5])
        np.testing.assert_allclose(_pt(out, 'v1', 1), out.P[1] * 2)
        self.assertEqual(list(_pt(out, 'ids', 1)), [11, 5])
        self.assertAlmostEqual(_pt(out, 'w0', 3), 0.5)

    def test_attribtypeinfo(self):
        out = _run('s@a = attribtypeinfo(0, "point", "P"); setattribtypeinfo(0, "point", "rest", "point");'
                   's@b = attribtypeinfo(0, "point", "rest"); s@c = attribtypeinfo(0, "point", "nothere");')
        self.assertEqual((_pt(out, 'a'), _pt(out, 'b'), _pt(out, 'c')), ('point', 'point', ''))

    def test_geometry_creation(self):
        out = _run('if (@ptnum == 0) { int pr = addprim(0, "poly"); addvertex(0, pr, 0); addvertex(0, pr, 1);'
                   'addvertex(0, pr, 5); setprimgroup(0, "tri", pr, 1); }')
        self.assertEqual(out.num_prims, 10)
        self.assertEqual(list(out.prim_points(9)), [0, 1, 5])
        self.assertTrue(out.groups['prim']['tri'][9])

    def test_topology_queries(self):
        out = _run('int pp[] = primpoints(0, 0); i@n = len(pp); i@nb = len(neighbours(0, @ptnum));'
                   'f@d = ptlined({0,0,0}, {1,0,0}, @P);')
        self.assertEqual(_pt(out, 'n'), 4)
        self.assertEqual(_pt(out, 'nb', 0), 2)    # corner of the grid
        self.assertEqual(_pt(out, 'nb', 5), 4)    # interior point
        self.assertAlmostEqual(_pt(out, 'd', 0), np.sqrt(2))

    def test_errors_have_line_numbers(self):
        cases = [
            ('@P.y = 1;\nint a = undefined_fn(3);', VexError, 2),
            ('@P.y = 1;\n\nfloat x = ;', VexSyntaxError, 3),
            ('int k = 1;\nswitch (k) { case 1: break; }', VexSyntaxError, 2),
            ('int a[] = {1};\n\na = a + 1;', VexError, 3),
            ('#define SQ(x) ((x)*(x))\nf@a = SQ(1, 2);', VexSyntaxError, 2),
            ('f@a = 1;\nmatrix3 m = ident(); f@b = m.q;', VexError, 2),
        ]
        for code, exc, line in cases:
            with self.subTest(code=code):
                with self.assertRaises(exc) as cm:
                    _run(code)
                self.assertTrue(str(cm.exception).startswith('Line %d:' % line), str(cm.exception))

    def test_garbage_never_raises_python_errors(self):
        import random
        rng = random.Random(7)
        toks = ['[', ']', '(', ')', '{', '}', ';', ',', '@P', '1', '"s"', ':', '.', 'int', 'm.xx', '[]', '?']
        for _ in range(150):
            code = rng.choice(SNIPPETS)
            k = rng.randint(0, len(code))
            code = code[:k] + rng.choice(toks) + code[k + rng.randint(0, 3):]
            try:
                _run(code)
            except (VexError, VexSyntaxError) as ex:
                self.assertRegex(str(ex), r'^Line [1-9]')


if __name__ == '__main__':
    unittest.main()
