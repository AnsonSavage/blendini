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
        from blendini.sops import REGISTRY, default_parms, CookContext, cook_sop
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

    def unregister():
        blendini.unregister()

    check("examples build and play", build_and_play)
    check("save and reload", save_and_reload)
    check("self test (every SOP)", self_test)
    check("operators with a node editor context", operators)
    check("rigid body scene lifecycle", rbd_scene_lifecycle)
    check("viewport handle bindings", handle_bindings)
    check("unregister", unregister)
    print("\n%d failure(s)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    code = main()
    if bpy.app.background:
        sys.exit(code)
