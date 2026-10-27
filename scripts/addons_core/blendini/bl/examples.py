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

    t = _new_tree("spiral", (50, 0, 0))
    w = _node(t, 'detailwrangle', (0, 200), snippet=SPIRAL)
    _spare(w, count=200, turns=4.0)
    rs = _node(t, 'resample', (0, 0), length=0.05)
    _chain(t, [w, rs])
    _finish(t, w)
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
