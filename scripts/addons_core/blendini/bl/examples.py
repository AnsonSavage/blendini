# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Example networks showing off the SOPs and the wrangle."""

import bpy

from .tree import TREE_ID, NODE_PREFIX, BlendiniNodeBase

RIPPLE = """\
// Concentric ripples driven by time
float d = length(@P);
float freq = ch("freq");
float amp = ch("amp");
@P.y = sin(d * freq - @Time * 4) * amp * exp(-d * 0.3);
@Cd = set(0.2, 0.5 + @P.y, 1.0 - @P.y);
"""

ORIENT = """\
// Random scale and orientation for the copies
f@pscale = fit01(rand(@ptnum), 0.05, 0.2);
vector up = {0, 1, 0};
matrix3 m = dihedral(up, @N);
rotate(m, rand(@ptnum + 7) * 6.283, @N);
p@orient = quaternion(m);
@Cd = set(rand(@ptnum), 0.6, 0.9);
"""

PLEXUS = """\
// Connect each point to its nearest neighbours with polylines
int pts[] = nearpoints(0, @P, ch("radius"), chi("maxpts"));
foreach (int pt; pts) {
    if (pt > @ptnum) {
        addprim(0, "polyline", @ptnum, pt);
    }
}
@Cd = set(1, 0.4 + 0.6 * rand(@ptnum), 0.2);
"""

CITY_GROUPS = """\
// Pick random primitives to extrude, with random heights
if (rand(@primnum * 3.17) < ch("density")) {
    i@group_tall = 1;
}
@Cd = set(0.35, 0.37, 0.4) + rand(@primnum) * 0.15;
"""

PARTICLES_INIT = """\
// Initial velocities
v@v = set(rand(@ptnum) - 0.5, 1.0 + rand(@ptnum + 1), rand(@ptnum + 2) - 0.5) * 4;
@Cd = rand(@ptnum);
"""

PARTICLES_SOLVER = """\
// Simple particle solver: gravity + bounce on the ground
v@v += {0, -9.8, 0} * @TimeInc;
@P += v@v * @TimeInc;
if (@P.y < 0) {
    @P.y = -@P.y;
    v@v.y *= -0.7;
    v@v *= 0.95;
}
"""

SPIRAL = """\
// Build a spiral polyline from scratch (Detail wrangle runs once)
int n = chi("count");
int prim = addprim(0, "polyline");
for (int i = 0; i < n; i++) {
    float t = i / float(n - 1);
    float a = t * 6.2831 * ch("turns");
    vector p = set(cos(a) * (1 - t), t * 3, sin(a) * (1 - t));
    int pt = addpoint(0, p);
    addvertex(0, prim, pt);
    setpointattrib(0, "Cd", pt, set(t, 1 - t, 0.5));
}
"""


PIECE = """\
// Runs once per piece inside the For-Each block
int it = detail(0, "iteration");
vector c = getbbox_center(0);
matrix3 m = ident();
rotate(m, rand(it) * 0.8, normalize(set(rand(it + 1) - 0.5, 1, rand(it + 2) - 0.5)));
@P = (@P - c) * m + c + set(0, rand(it + 3) * 0.6, 0);
@Cd = set(rand(it), 0.5, 1 - rand(it));
"""


EXPLODE = """\
// Push each piece away from the centre (runs per piece in the For-Each block)
vector c = getbbox_center(0);
@P += normalize(c) * ch("explode") * length(c);
"""

INSIDE = """\
// Colour the freshly cut inside faces
@Cd = @group_inside ? set(0.95, 0.45, 0.15) : set(0.75, 0.78, 0.82);
"""


TERRAIN = """\
// Colour the terrain by height, slope and water flow
float h = @P.y / ch("maxh");
float slope = 1 - abs(@N.y);
vector grass = {0.25, 0.42, 0.16};
vector rock = {0.42, 0.38, 0.34};
vector snow = {0.95, 0.95, 0.97};
vector c = lerp(grass, rock, smooth(0.15, 0.35, slope));
c = lerp(c, snow, smooth(0.75, 0.9, h) * (1 - smooth(0.4, 0.6, slope)));
c = lerp(c, {0.12, 0.2, 0.3}, smooth(5, 8, f@flow) * 0.8);
@Cd = c;
"""


def _node(tree, sop_id, loc, **parms):
    node = tree.nodes.new(NODE_PREFIX + sop_id)
    node.location = loc
    for k, v in parms.items():
        setattr(node, 'p_' + k, v)
    return node


def _spare(node, **values):
    for name, val in values.items():
        sp = node.spare.add()
        sp.name = name
        if isinstance(val, int):
            sp.kind = 'int'
            sp.i = val
        else:
            sp.kind = 'float'
            sp.f = val


def _chain(tree, nodes):
    for a, b in zip(nodes, nodes[1:]):
        tree.links.new(a.outputs[0], b.inputs[0])


def _finish(tree, display):
    for n in tree.nodes:
        if isinstance(n, BlendiniNodeBase):
            n.display_flag = (n == display)
    tree.nodes.active = display


def _new_tree(name, offset):
    from . import cook
    tree = bpy.data.node_groups.get(name)
    if tree is not None:
        bpy.data.node_groups.remove(tree)
    tree = bpy.data.node_groups.new(name, TREE_ID)
    tree.use_fake_user = True
    obj = bpy.data.objects.get(name)
    if obj is None:
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
    tree.target = obj
    obj["blendini_tree"] = tree.name
    from .display import setup_target
    setup_target(obj)
    obj.location = offset
    return tree


def build_examples():
    trees = []

    t = _new_tree("ripple", (0, 0, 0))
    grid = _node(t, 'grid', (0, 200), rows=80, cols=80, size=(8.0, 8.0))
    w = _node(t, 'pointwrangle', (0, 0), snippet=RIPPLE)
    _spare(w, freq=4.0, amp=0.4)
    _chain(t, [grid, w])
    _finish(t, w)
    trees.append(t)

    t = _new_tree("scatter_copy", (10, 0, 0))
    sph = _node(t, 'sphere', (0, 400), type='polymesh', freq=3, rad=(2.0, 2.0, 2.0))
    sc = _node(t, 'scatter', (0, 200), npts=400, seed=3)
    w = _node(t, 'pointwrangle', (0, 0), snippet=ORIENT)
    box = _node(t, 'box', (-250, 0))
    cp = _node(t, 'copytopoints', (0, -200))
    mrg = _node(t, 'merge', (0, -400))
    _chain(t, [sph, sc, w])
    t.links.new(box.outputs[0], cp.inputs[0])
    t.links.new(w.outputs[0], cp.inputs[1])
    t.links.new(cp.outputs[0], mrg.inputs[0])
    t.links.new(sph.outputs[0], mrg.inputs[0])
    _finish(t, mrg)
    trees.append(t)

    t = _new_tree("plexus", (20, 0, 0))
    tor = _node(t, 'torus', (0, 400), rad=(2.0, 0.7))
    sc = _node(t, 'scatter', (0, 200), npts=300, seed=1)
    w = _node(t, 'pointwrangle', (0, 0), snippet=PLEXUS)
    _spare(w, radius=0.6, maxpts=5)
    wire = _node(t, 'polywire', (0, -200), radius=0.015)
    _chain(t, [tor, sc, w, wire])
    _finish(t, wire)
    trees.append(t)

    t = _new_tree("city", (30, 0, 0))
    grid = _node(t, 'grid', (0, 400), rows=20, cols=20, size=(8.0, 8.0))
    w = _node(t, 'primwrangle', (0, 200), snippet=CITY_GROUPS)
    _spare(w, density=0.4)
    ex = _node(t, 'polyextrude', (0, 0), group='tall', dist=1.5, inset=0.1, splittype='individual')
    ex2 = _node(t, 'polyextrude', (0, -200), group='extrudeFront', dist=0.8, inset=0.2)
    _chain(t, [grid, w, ex, ex2])
    _finish(t, ex2)
    trees.append(t)

    t = _new_tree("particles", (40, 0, 0))
    grid = _node(t, 'grid', (0, 400), rows=2, cols=2, size=(1.0, 1.0))
    sc = _node(t, 'scatter', (0, 200), npts=500)
    w = _node(t, 'pointwrangle', (0, 0), snippet=PARTICLES_INIT)
    sol = _node(t, 'solverwrangle', (0, -200), snippet=PARTICLES_SOLVER)
    _chain(t, [grid, sc, w, sol])
    _finish(t, sol)
    trees.append(t)

    t = _new_tree("foreach", (60, 0, 0))
    box = _node(t, 'box', (0, 600), size=(0.9, 0.9, 0.9))
    cx = _node(t, 'copyxform', (0, 400), ncy=6, t=(1.0, 0.0, 0.0))
    cy = _node(t, 'copyxform', (0, 200), ncy=6, t=(0.0, 0.0, 1.0))
    begin = _node(t, 'foreach_begin', (0, 0))
    w = _node(t, 'attribwrangle', (0, -200), snippet=PIECE)
    end = _node(t, 'foreach_end', (0, -400))
    _chain(t, [box, cx, cy, begin, w, end])
    _finish(t, end)
    trees.append(t)

    t = _new_tree("lsystem_tree", (70, 0, 0))
    ls = _node(t, 'lsystem', (0, 400), generations=7, stepsize=0.35, angle=26.0, randangle=6.0)
    sw = _node(t, 'sweep', (0, 200), radius=0.06, cols=6)
    col = _node(t, 'pointwrangle', (0, 0), snippet='@Cd = lerp({0.35, 0.22, 0.1}, {0.3, 0.7, 0.2}, pow(f@curveu, 3));')
    _chain(t, [ls, sw, col])
    _finish(t, col)
    trees.append(t)

    t = _new_tree("vellum_cloth", (80, 0, 0))
    grid = _node(t, 'grid', (0, 400), rows=30, cols=30, size=(3.0, 3.0))
    lift = _node(t, 'xform', (0, 200), t=(0.0, 2.2, 0.0), r=(0.0, 20.0, 0.0))
    ball = _node(t, 'sphere', (300, 200), type='polymesh', freq=3, rad=(0.8, 0.8, 0.8), t=(0.0, 0.8, 0.0))
    vel = _node(t, 'vellum', (0, 0))
    col = _node(t, 'pointwrangle', (0, -200), snippet='@Cd = lerp({0.9, 0.3, 0.2}, {1, 0.85, 0.4}, @P.y / 2.2);')
    mrg = _node(t, 'merge', (0, -400))
    _chain(t, [grid, lift, vel, col])
    t.links.new(ball.outputs[0], vel.inputs[1])
    t.links.new(col.outputs[0], mrg.inputs[0])
    t.links.new(ball.outputs[0], mrg.inputs[0])
    _finish(t, mrg)
    trees.append(t)

    t = _new_tree("fracture", (90, 0, 0))
    box = _node(t, 'box', (0, 600), size=(2.0, 2.0, 2.0))
    fr = _node(t, 'voronoifracture', (0, 400), npts=30, seed=4)
    col = _node(t, 'primwrangle', (0, 200), snippet=INSIDE)
    begin = _node(t, 'foreach_begin', (0, 0))
    w = _node(t, 'attribwrangle', (0, -200), snippet=EXPLODE)
    _spare(w, explode=0.35)
    end = _node(t, 'foreach_end', (0, -400), attrib='name')
    _chain(t, [box, fr, col, begin, w, end])
    _finish(t, end)
    trees.append(t)

    t = _new_tree("rbd_fracture", (100, 0, 0))
    box = _node(t, 'box', (0, 600), size=(2.0, 2.0, 2.0))
    fr = _node(t, 'voronoifracture', (0, 400), npts=40, seed=7)
    col = _node(t, 'primwrangle', (0, 200), snippet=INSIDE)
    lift = _node(t, 'xform', (0, 0), t=(0.0, 3.0, 0.0), r=(25.0, 35.0, 10.0))
    rbd = _node(t, 'rbdsolver', (0, -200))
    _chain(t, [box, fr, col, lift, rbd])
    _finish(t, rbd)
    trees.append(t)

    t = _new_tree("terrain", (115, 0, 0))
    hf = _node(t, 'heightfield', (0, 600), size=(12.0, 12.0), gridspacing=0.06)
    nz = _node(t, 'hfnoise', (0, 400), amp=2.2, elementsize=6.0, octaves=7, roughness=0.55)
    er = _node(t, 'hferode', (0, 200), iterations=40, hydro=0.006)
    nm = _node(t, 'normal', (0, 0))
    col = _node(t, 'pointwrangle', (0, -200), snippet=TERRAIN)
    _spare(col, maxh=2.6)
    _chain(t, [hf, nz, er, nm, col])
    _finish(t, col)
    trees.append(t)

    t = _new_tree("spiral", (50, 0, 0))
    w = _node(t, 'detailwrangle', (0, 200), snippet=SPIRAL)
    _spare(w, count=200, turns=4.0)
    rs = _node(t, 'resample', (0, 0), length=0.05)
    sw = _node(t, 'sweep', (0, -200), radius=0.08, scaleramp=0.2, cols=10)
    _chain(t, [w, rs, sw])
    _finish(t, sw)
    trees.append(t)
    return trees


class BLENDINI_OT_load_examples(bpy.types.Operator):
    """Build a set of example networks (ripple, scatter/copy, plexus, city, particles, spiral)"""
    bl_idname = "blendini.load_examples"
    bl_label = "Load Example Networks"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from . import cook
        from .ui import show_tree_in_editor
        trees = build_examples()
        for t in trees:
            cook.cook_tree(t)
        show_tree_in_editor(context, trees[0])
        self.report({'INFO'}, "Created %d example networks" % len(trees))
        return {'FINISHED'}
