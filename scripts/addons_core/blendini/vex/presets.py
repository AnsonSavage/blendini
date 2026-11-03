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
    ("Snap to Surface (input 2)", '''// Project points onto the second input's surface
int prim; vector uv;
float d = xyzdist(1, @P, prim, uv);
if (d < ch("maxdist") || ch("maxdist") == 0)
    @P = primuv(1, "P", prim, uv);
'''),
    ("Struct Particles (struct + methods)", '''// VEX structs with methods, run over Detail
struct Particle {
    vector pos; vector vel;
    void step(float dt) { vel += {0, -9.8, 0} * dt; pos += vel * dt; }
}
Particle p = Particle({0, 0, 0}, {1, 6, 0});
int prev = -1;
for (int i = 0; i < 40; i++) {
    int pt = addpoint(0, p.pos);
    if (prev >= 0) addprim(0, "polyline", prev, pt);
    prev = pt;
    p->step(1.0 / 24);
}
'''),
    ("Count by Name (dict)", '''// Run over Detail: count primitives per name with a dict
dict counts;
for (int i = 0; i < nprims(0); i++) {
    string n = prim(0, "name", i);
    int c = isvalidindex(counts, n) ? counts[n] : 0;
    counts[n] = c + 1;
}
string names[] = keys(counts);
foreach (string n; names) printf("%s: %d\\n", n, counts[n]);
'''),
    ("Edge Lengths (half-edges)", '''// Run over Points: average length of the half-edges leaving each point
// (a half-edge id is the vertex it starts from)
int verts[] = pointvertices(0, @ptnum);
float total = 0;
foreach (int h; verts) total += distance(@P, point(0, "P", hedge_dstpoint(0, h)));
f@edgelen = len(verts) ? total / len(verts) : 0;
'''),
    ("Density from Distance (Volume Wrangle)", '''// For a Volume Wrangle: a soft sphere of fog
float r = 0.8;
@density = clamp((r - length(@P)) * 4, 0, 1) * (0.6 + 0.4 * noise(@P * 3));
'''),
]
