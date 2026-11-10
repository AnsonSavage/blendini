# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Network copy / paste as text inside Blender (needs bpy):

    blender --background --factory-startup --python scripts/addons_core/blendini/tests/blender_clipboard.py

or with the `bpy` module: python blender_clipboard.py

For several example networks, copies all nodes to JSON, pastes them into a fresh network, cooks
both and compares point / primitive counts and bounding boxes. Also checks a subnet with a ramp
spare parameter and the two operators. Exits with a non-zero status on failure.
"""

import json
import os
import sys
import traceback
from unittest import mock

import bpy
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

failures = []

EXAMPLES = ('ripple', 'scatter_copy', 'plexus', 'city', 'foreach', 'lsystem_tree', 'fracture', 'terrain',
            'volume_cloud', 'streamlines', 'vop_displace', 'spiral')


def check(name, fn):
    try:
        fn()
        print("ok   ", name)
    except Exception:
        failures.append(name)
        print("FAIL ", name)
        traceback.print_exc()


def _summary(geo):
    lo, hi = geo.bbox()
    return geo.num_points, geo.num_prims, len(geo.volumes), np.round(lo, 5).tolist(), np.round(hi, 5).tolist()


def clipboard_roundtrip():
    import blendini
    blendini.register()
    from blendini.bl import cook, examples, clipboard
    from blendini.bl.tree import NODE_PREFIX, TREE_ID

    trees = {t.name: t for t in examples.build_examples()}

    def compare(src):
        data = json.loads(json.dumps(clipboard.nodes_to_dict(list(src.nodes))))
        dst = bpy.data.node_groups.new(src.name + "_pasted", TREE_ID)
        try:
            nodes = clipboard.dict_to_nodes(dst, data, (0.0, 0.0))
            assert len(nodes) == len(src.nodes), (len(nodes), len(src.nodes))
            assert len(dst.links) == len(src.links), (len(dst.links), len(src.links))
            a = cook.cook_node_at(cook.display_node(src), 1)
            b = cook.cook_node_at(cook.display_node(dst), 1)
            assert cook.display_node(dst).name == cook.display_node(src).name
            sa, sb = _summary(a), _summary(b)
            assert sa == sb, (sa, sb)
            assert a.num_points or a.volumes, "example cooked to nothing"
        finally:
            bpy.data.node_groups.remove(dst)

    for name in EXAMPLES:
        check("copy/paste example %s" % name, lambda name=name: compare(trees[name]))

    def subnet_ramp_and_props():
        tree = bpy.data.node_groups.new("clip_subnet", TREE_ID)
        try:
            grid = tree.nodes.new(NODE_PREFIX + 'grid')
            sub = tree.nodes.new(NODE_PREFIX + 'subnet')
            sub.label = "Lift"
            tree.links.new(grid.outputs[0], sub.inputs[0])
            sp = sub.spare.add()
            sp.name, sp.kind, sp.f = 'lift', 'float', 2.0
            inner = sub.subtree
            w = inner.nodes.new(NODE_PREFIX + 'pointwrangle')
            w.p_snippet = '@P.y += ch("lift") * chramp("r", @P.x * 0.5 + 0.5);'
            from blendini.bl.ramps import new_key
            from blendini.bl import clipboard as cb
            r = w.spare.add()
            r.name, r.kind = 'r', 'ramp'
            r.ramp_key = new_key()
            cb._set_ramp(r.ramp_key, [[0.0, 0.0, 'VECTOR'], [0.5, 1.0, 'VECTOR'], [1.0, 0.0, 'VECTOR']])
            inp = next(n for n in inner.nodes if n.sop_id == 'subinput')
            out = next(n for n in inner.nodes if n.sop_id == 'output')
            for link in list(inner.links):
                inner.links.remove(link)
            inner.links.new(inp.outputs[0], w.inputs[0])
            inner.links.new(w.outputs[0], out.inputs[0])
            grid["edit_idx"] = [1, 2]
            sub.template_flag = True
            for n in tree.nodes:
                n.display_flag = (n == sub)
            text = json.dumps(clipboard.nodes_to_dict(list(tree.nodes)))
            data = json.loads(text)
            assert data['nodes'][1]['subnet']['nodes'], "subnet content missing"
            dst = bpy.data.node_groups.new("clip_subnet_pasted", TREE_ID)
            try:
                clipboard.dict_to_nodes(dst, data)
                psub = next(n for n in dst.nodes if n.sop_id == 'subnet')
                assert psub.subtree is not None and psub.subtree != inner
                assert psub.label == "Lift" and psub.template_flag and psub.spare['lift'].f == 2.0
                pw = next(n for n in psub.subtree.nodes if n.sop_id == 'pointwrangle')
                assert pw.spare['r'].ramp_key != r.ramp_key
                assert list(dst.nodes['grid1']["edit_idx"]) == [1, 2]
                a = cook.cook_node_at(sub, 1)
                b = cook.cook_node_at(psub, 1)
                assert _summary(a) == _summary(b), (_summary(a), _summary(b))
                assert a.P[:, 1].max() > 0.5, a.P[:, 1].max()
                # Pasting again into the same network renames the nodes and keeps the display flag.
                again = clipboard.dict_to_nodes(dst, data, (300.0, 0.0))
                assert all(n.name not in ('grid1', 'subnet1') for n in again)
                assert sum(1 for n in dst.nodes if n.display_flag) == 1
            finally:
                for n in list(dst.nodes):
                    if n.sop_id == 'subnet' and n.subtree is not None:
                        bpy.data.node_groups.remove(n.subtree)
                bpy.data.node_groups.remove(dst)
        finally:
            bpy.data.node_groups.remove(inner)
            bpy.data.node_groups.remove(tree)

    check("copy/paste subnet, ramp, custom props", subnet_ramp_and_props)

    def operators():
        src = trees['scatter_copy']
        dst = bpy.data.node_groups.new("clip_ops", TREE_ID)
        try:
            ctx = mock.MagicMock()
            ctx.space_data.edit_tree = src
            for n in src.nodes:
                n.select = True
            ctx.window_manager.clipboard = ""
            assert clipboard.BLENDINI_OT_copy_nodes_text.execute(mock.MagicMock(), ctx) == {'FINISHED'}
            assert json.loads(ctx.window_manager.clipboard)['format'] == clipboard.FORMAT
            ctx.space_data.edit_tree = dst
            op = mock.MagicMock()
            op.offset = (0.0, 0.0)
            assert clipboard.BLENDINI_OT_paste_nodes_text.execute(op, ctx) == {'FINISHED'}
            assert len(dst.nodes) == len(src.nodes)
            ctx.window_manager.clipboard = "not json"
            assert clipboard.BLENDINI_OT_paste_nodes_text.execute(op, ctx) == {'CANCELLED'}
            assert hasattr(bpy.ops.blendini, 'copy_nodes_text') and hasattr(bpy.ops.blendini, 'paste_nodes_text')
        finally:
            bpy.data.node_groups.remove(dst)

    check("copy / paste operators", operators)


def main():
    clipboard_roundtrip()
    if failures:
        print("%d clipboard check(s) failed: %s" % (len(failures), ", ".join(failures)))
        sys.exit(1)
    print("all clipboard checks passed")


if __name__ == "__main__":
    main()
