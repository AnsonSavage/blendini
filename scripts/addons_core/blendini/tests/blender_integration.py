# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Integration test inside Blender (needs bpy):

    blender --background --factory-startup --python scripts/addons_core/blendini/tests/blender_integration.py

or with the `bpy` module: python blender_integration.py

Builds every example network, plays frames, saves and reloads the file, runs the self test
and the operators with a mocked node editor context. Exits with a non-zero status on failure.
"""

import os
import sys
import tempfile
import traceback
from unittest import mock

import bpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

failures = []


def check(name, fn):
    try:
        fn()
        print("ok   ", name)
    except Exception:
        failures.append(name)
        print("FAIL ", name)
        traceback.print_exc()


def main():
    import blendini
    blendini.register()
    from blendini.bl import cook, examples, selftest, ui
    from blendini.bl.tree import NODE_PREFIX
    from blendini.sops import cook_sop

    def build_and_play():
        trees = examples.build_examples()
        for frame in (1, 2, 12, 30):
            bpy.context.scene.frame_set(frame)
            errors = [(t.name, n.name, cook.node_info(n).get('error')) for t in trees for n in t.nodes
                      if cook.node_info(n).get('error')]
            assert not errors, errors

    def save_and_reload():
        path = os.path.join(tempfile.mkdtemp(), "blendini_integration.blend")
        bpy.ops.wm.save_as_mainfile(filepath=path)
        bpy.ops.wm.open_mainfile(filepath=path)
        for t in cook.all_trees():
            cook.cook_tree(t)
            assert not any(cook.node_info(n).get('error') for n in t.nodes), t.name

    def self_test():
        ok, fails, _seconds = selftest.run_self_test()
        assert not fails, fails

    def operators():
        tree = next(t for t in cook.all_trees() if t.name == 'ripple')
        node = tree.nodes['pointwrangle1']
        tree.nodes.active = node
        ctx = mock.MagicMock()
        ctx.space_data.type = 'NODE_EDITOR'
        ctx.space_data.tree_type = 'BlendiniSOPTree'
        ctx.space_data.edit_tree = tree
        ctx.space_data.text = None
        ctx.scene, ctx.collection, ctx.view_layer = bpy.context.scene, bpy.context.collection, bpy.context.view_layer
        ctx.active_object, ctx.screen, ctx.area = tree.target, None, None
        skip = {'BLENDINI_OT_tab_menu', 'BLENDINI_OT_add_node', 'BLENDINI_OT_load_examples'}
        for cls in ui._all_classes():
            if not issubclass(cls, bpy.types.Operator) or cls.__name__ in skip:
                continue
            op = mock.MagicMock(node_name=node.name, index=0, kind='float', parm_name='p1', mode='single',
                                type=NODE_PREFIX + 'null', starter=False, primitive='', text='x',
                                filepath=os.path.join(tempfile.mkdtemp(), "x.obj"))
            if (not hasattr(cls, 'poll') or cls.poll(ctx)) and 'execute' in cls.__dict__:
                cls.execute(op, ctx)

    def rbd_scene_lifecycle():
        # Regression: removing one rigid body scene while another exists crashed Blender, and
        # stepping a simulation from the frame change handler corrupted memory.
        from blendini.sops import REGISTRY, default_parms, CookContext
        from blendini.sops.blender_ops import _rbd_cleanup
        # Glue constraints must be freed before their Bullet bodies (crashed in about 1 of 4 runs).
        import random
        rng = random.Random(1)
        fractured = cook_sop('voronoifracture', [cook_sop('box')])
        removed = []
        for _round in range(6):
            states = []
            for k in range(3):
                st = {}
                ctx = CookContext(frame=3, time=2 / 24, state=st)
                ctx.channels, ctx.ramps = {}, {}
                p = default_parms('rbdsolver')
                p['glue'] = k != 1
                REGISTRY['rbdsolver'].cook([fractured if k else cook_sop('box')], p, ctx)
                states.append(st)
            names = [st['scene'] for st in states]
            assert all(bpy.data.scenes.get(n) is not None for n in names)
            rng.shuffle(states)
            for st in states:
                _rbd_cleanup(st)
            assert not [n for n in names if bpy.data.scenes.get(n) is not None]
            removed += names
            bpy.context.scene.frame_set(rng.randint(1, 30))
        leaked = [ob.name for ob in bpy.data.objects if any(ob.name.startswith(n + '_') for n in removed)]
        assert not leaked, leaked
        # i@active = 0 pieces stay where they are.
        from blendini.core.geometry import merge
        from blendini.vex.wrangle import run_wrangle
        from blendini.sops.modify import split_pieces
        two = merge([cook_sop('box', size=(0.5, 0.5, 0.5), t=(0, 3, 0)), cook_sop('box', size=(0.5, 0.5, 0.5), t=(2, 3, 0))])
        two, _ = run_wrangle('i@active = @P.x < 1.0 ? 1 : 0;', [two], run_over='point')
        st = {}
        ctx = CookContext(frame=25, time=1.0, state=st)
        ctx.channels, ctx.ramps = {}, {}
        out = REGISTRY['rbdsolver'].cook([two], default_parms('rbdsolver'), ctx)
        heights = sorted((pc.P.mean(axis=0)[0], pc.P.mean(axis=0)[1]) for pc in split_pieces(out, 'name'))
        _rbd_cleanup(st)
        assert heights[0][1] < 1.0 and abs(heights[1][1] - 3.0) < 1e-4, heights

    def handle_bindings():
        from blendini.bl import handles
        tree = bpy.data.node_groups.new("handles_test", "BlendiniSOPTree")
        try:
            for sid in handles.HANDLES:
                node = tree.nodes.new(NODE_PREFIX + sid)
                for b in handles.bindings(node):
                    v = b['get']()
                    b['set'](v + 0.25)
                    assert abs(b['get']() - (v + 0.25)) < 1e-4, (sid, b['kind'])
        finally:
            bpy.data.node_groups.remove(tree)

    def file_cache():
        from blendini.bl.examples import _finish
        tree = bpy.data.node_groups.new("filecache_test", "BlendiniSOPTree")
        try:
            box = tree.nodes.new(NODE_PREFIX + 'box')
            w = tree.nodes.new(NODE_PREFIX + 'pointwrangle')
            w.p_snippet = '@P.y += @Frame * 0.1;'
            fc = tree.nodes.new(NODE_PREFIX + 'filecache')
            fc.p_file = os.path.join(tempfile.mkdtemp(), '$OS.$F4.npz')
            fc.p_endframe = 5
            tree.links.new(box.outputs[0], w.inputs[0])
            tree.links.new(w.outputs[0], fc.inputs[0])
            _finish(tree, fc)
            ctx = mock.MagicMock()
            ctx.space_data.edit_tree = tree
            ctx.space_data.tree_type = 'BlendiniSOPTree'
            ctx.scene = bpy.context.scene
            assert ui.BLENDINI_OT_filecache_save.execute(mock.MagicMock(), ctx) == {'FINISHED'}
            assert fc.p_load
            box.p_size = (5.0, 5.0, 5.0)    # ignored while loading from disk
            geo = cook.cook_node_at(fc, 4)
            assert abs(geo.P[:, 1].max() - (0.5 + 0.4)) < 1e-6, geo.P[:, 1].max()
        finally:
            bpy.data.node_groups.remove(tree)

    def material_slots():
        from blendini.bl.examples import _finish
        bpy.data.materials.new("IntegrationRed")
        tree = bpy.data.node_groups.new("material_test", "BlendiniSOPTree")
        try:
            box = tree.nodes.new(NODE_PREFIX + 'box')
            m = tree.nodes.new(NODE_PREFIX + 'material')
            m.p_group, m.p_material = '0-2', 'IntegrationRed'
            tree.links.new(box.outputs[0], m.inputs[0])
            _finish(tree, m)
            cook.cook_tree(tree)
            me = tree.target.data
            assert [x.name for x in me.materials][1:] == ['IntegrationRed']
            assert [p.material_index for p in me.polygons] == [1, 1, 1, 0, 0, 0]
        finally:
            target = tree.target
            bpy.data.node_groups.remove(tree)
            if target is not None:
                bpy.data.objects.remove(target)

    def digital_assets():
        from blendini.bl.examples import _finish
        tree = bpy.data.node_groups.new("hda_test", "BlendiniSOPTree")
        other = bpy.data.node_groups.new("hda_test2", "BlendiniSOPTree")
        box = tree.nodes.new(NODE_PREFIX + 'box')
        sub = tree.nodes.new(NODE_PREFIX + 'subnet')
        tree.links.new(box.outputs[0], sub.inputs[0])
        inner = sub.subtree
        w = inner.nodes.new(NODE_PREFIX + 'pointwrangle')
        w.p_snippet = '@P *= 2;'
        inp = next(n for n in inner.nodes if n.sop_id == 'subinput')
        out = next(n for n in inner.nodes if n.sop_id == 'output')
        inner.links.new(inp.outputs[0], w.inputs[0])
        inner.links.new(w.outputs[0], out.inputs[0])
        _finish(tree, sub)
        ctx = mock.MagicMock()
        ctx.space_data.edit_tree = tree
        ctx.space_data.tree_type = 'BlendiniSOPTree'
        tree.nodes.active = sub
        assert ui.BLENDINI_OT_create_hda.execute(mock.MagicMock(label='itest_doubler'), ctx) == {'FINISHED'}
        assert inner.is_hda and inner not in cook.all_trees()
        ctx.space_data.edit_tree = other
        ctx.space_data.cursor_location = (0.0, 0.0)
        assert ui.BLENDINI_OT_add_hda.execute(mock.MagicMock(definition=inner.name), ctx) == {'FINISHED'}
        inst = other.nodes.active
        assert inst.subtree == inner
        sph = other.nodes.new(NODE_PREFIX + 'sphere')
        other.links.new(sph.outputs[0], inst.inputs[0])
        _finish(other, inst)
        w.p_snippet = '@P *= 3;'
        assert abs(cook.cook_node_at(sub, 1).bbox()[1][0] - 1.5) < 1e-6
        assert abs(cook.cook_node_at(inst, 1).bbox()[1][0] - 3.0) < 1e-6
        assert {t.name for t in cook.trees_using(inner)} == {"hda_test", "hda_test2"}
        for t in (tree, other):
            target = t.target
            bpy.data.node_groups.remove(t)
            if target is not None:
                bpy.data.objects.remove(target)
        bpy.data.node_groups.remove(inner)

    def attribute_paint():
        import numpy as np
        from blendini.bl.examples import _finish
        tree = bpy.data.node_groups.new("paint_test", "BlendiniSOPTree")
        try:
            g = tree.nodes.new(NODE_PREFIX + 'grid')
            pn = tree.nodes.new(NODE_PREFIX + 'attribpaint')
            tree.links.new(g.outputs[0], pn.inputs[0])
            _finish(tree, pn)
            cook.cook_tree(tree)
            obj = tree.target
            vg = obj.vertex_groups.new(name=ui.PAINT_GROUP)
            vg.add([v.index for v in obj.data.vertices if v.co.x > 0], 1.0, 'REPLACE')
            ctx = mock.MagicMock()
            ctx.space_data.edit_tree = tree
            ctx.space_data.tree_type = 'BlendiniSOPTree'
            tree.nodes.active = pn
            assert ui.BLENDINI_OT_paint_commit.execute(mock.MagicMock(), ctx) == {'FINISHED'}
            m = cook.cook_node_at(pn, 1).attribs['point']['mask']
            assert int((m > 0.5).sum()) == 50, int((m > 0.5).sum())
        finally:
            target = tree.target
            bpy.data.node_groups.remove(tree)
            if target is not None:
                bpy.data.objects.remove(target)

    def instance_objects():
        from blendini.bl.examples import _finish
        mesh = bpy.data.meshes.new("itest_inst")
        cookbox = cook_sop('box')
        from blendini.bl.mesh_io import geo_to_mesh
        geo_to_mesh(cookbox, mesh)
        ob = bpy.data.objects.new("ItestInstance", mesh)
        tree = bpy.data.node_groups.new("instance_test", "BlendiniSOPTree")
        try:
            g = tree.nodes.new(NODE_PREFIX + 'grid')
            ins = tree.nodes.new(NODE_PREFIX + 'instance')
            ins.p_object = ob.name
            tree.links.new(g.outputs[0], ins.inputs[0])
            _finish(tree, ins)
            cook.cook_tree(tree)
            geo = cook.node_geometry(ins)
            assert len(geo.instances) == 1 and int((geo.attribs['point']['__inst'] == 0).sum()) == geo.num_points
        finally:
            target = tree.target
            bpy.data.node_groups.remove(tree)
            if target is not None:
                bpy.data.objects.remove(target)
            bpy.data.objects.remove(ob)

    def houdini_expression_variables():
        from blendini.bl.animation import eval_driver_expression
        assert eval_driver_expression('$F / $FPS', 12.0, 24.0) == 0.5
        assert abs(eval_driver_expression('sin($PI / 2) + $T', 25.0, 24.0) - 2.0) < 1e-9

    def undo_redo_with_simulations():
        import random
        rng = random.Random(1)
        names = [n for n in ('rbd_fracture', 'rbd_chain', 'pyro_fire', 'volume_cloud', 'fluid_dambreak')
                 if bpy.data.node_groups.get(n) is not None]
        for i in range(6):
            bpy.ops.ed.undo_push(message="blendini test %d" % i)
            bpy.context.scene.frame_set(rng.randint(1, 20))
            cook.cook_tree(bpy.data.node_groups[rng.choice(names)])
            if i % 2:
                bpy.ops.ed.undo()
            for t in cook.all_trees():
                cook.cook_tree(t)
        errors = [(t.name, n.name) for t in cook.all_trees() for n in t.nodes if cook.node_info(n).get('error')]
        assert not errors, errors

    def unregister():
        blendini.unregister()

    check("examples build and play", build_and_play)
    check("save and reload", save_and_reload)
    check("self test (every SOP)", self_test)
    check("operators with a node editor context", operators)
    check("rigid body scene lifecycle", rbd_scene_lifecycle)
    check("viewport handle bindings", handle_bindings)
    check("file cache save / load", file_cache)
    check("material slots from shop_materialpath", material_slots)
    check("digital assets (shared subnet definitions)", digital_assets)
    check("attribute paint commit", attribute_paint)
    check("instance Blender objects by name", instance_objects)
    check("Houdini expression variables", houdini_expression_variables)
    check("undo / redo with simulations", undo_redo_with_simulations)
    check("unregister", unregister)
    print("\n%d failure(s)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    if bpy.app.background:
        sys.exit(code)
