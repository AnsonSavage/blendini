# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Ready made wrangle snippets (the wrangle's preset menu)."""

PRESETS = [
    ("Noise Displacement", '''// Push points along their normal with noise
float amp = ch("amp");
float freq = ch("freq");
@P += @N * (noise(@P * freq + @Time * 0.5) - 0.5) * amp;
'''),
    ("Color by Height", '''// Map height to a color ramp
vector lo = getbbox_min(0);
vector hi = getbbox_max(0);
float t = fit(@P.y, lo.y, hi.y, 0, 1);
@Cd = hsvtorgb(set(0.66 * (1 - t), 0.8, 1));
'''),
    ("Random Color per Primitive", '''// Run over Primitives
@Cd = rand(@primnum + ch("seed"));
'''),
    ("Random pscale and orient", '''// Variation for Copy to Points
f@pscale = fit01(rand(@ptnum + ch("seed")), ch("min_scale"), ch("max_scale"));
vector axis = @N;
p@orient = quaternion(rand(@ptnum + 3) * 6.283, axis);
'''),
    ("Delete Random Points", '''// Keep only a fraction of the points
if (rand(@ptnum + ch("seed")) > ch("keep")) removepoint(0, @ptnum);
'''),
    ("Connect Nearby Points (Plexus)", '''// Lines between neighbours
int pts[] = nearpoints(0, @P, ch("radius"), chi("maxpts"));
foreach (int pt; pts) {
    if (pt > @ptnum) addprim(0, "polyline", @ptnum, pt);
}
'''),
    ("Curl Noise Advect (use in a Solver)", '''// Move points through a curl noise field
v@v = curlnoise(@P * ch("freq") + @Time * 0.2) * ch("speed");
@P += v@v * @TimeInc;
'''),
    ("Twist Around Y", '''// Twist geometry around the Y axis
float angle = radians(ch("degrees_per_unit")) * @P.y;
matrix3 m = ident();
rotate(m, angle, {0, 1, 0});
@P *= m;
'''),
    ("Ripple from Origin", '''// Concentric animated ripples
float d = length(set(@P.x, 0, @P.z));
@P.y += sin(d * ch("freq") - @Time * ch("speed")) * ch("amp") * exp(-d * 0.3);
'''),
    ("Point Normals from Center", '''// Spherical normals, e.g. for copies pointing outwards
@N = normalize(@P - getbbox_center(0));
'''),
    ("Fit into Unit Box", '''// Normalize the geometry into a 1x1x1 box at the origin
vector size = getbbox_size(0);
float s = max(size.x, max(size.y, size.z));
@P = (@P - getbbox_center(0)) / max(s, 1e-6);
'''),
    ("Spiral Curve (Detail)", '''// Run over Detail: builds a polyline from scratch
int n = chi("count");
int prim = addprim(0, "polyline");
for (int i = 0; i < n; i++) {
    float t = i / float(max(n - 1, 1));
    float a = t * M_TWO_PI * ch("turns");
    int pt = addpoint(0, set(cos(a) * (1 - t), t * ch("height"), sin(a) * (1 - t)));
    addvertex(0, prim, pt);
}
'''),
    ("Shell (Inflate along Normal)", '''// Inflate the surface along its normals
@P += normalize(@N) * ch("distance");
'''),
    ("Print Info", '''// printf output appears in the Geometry Info panel
if (@ptnum == 0) printf("%d points, bbox size %g\\n", @numpt, getbbox_size(0));
'''),
]
