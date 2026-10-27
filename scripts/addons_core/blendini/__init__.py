# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

bl_info = {
    "name": "Blendini",
    "author": "Blendini Authors",
    "version": (0, 1, 0),
    "blender": (5, 0, 0),
    "location": "3D Viewport > Add > Blendini Geometry, Node Editor > Blendini SOP Network",
    "description": "Houdini style procedural SOP networks with a VEX wrangle, using Blender as the backend",
    "warning": "Experimental",
    "doc_url": "",
    "support": 'COMMUNITY',
    "category": "Node",
}

if "_loaded" in locals():
    import importlib
    from . import core, vex, sops, bl  # noqa: F401
    for mod in list(__import__('sys').modules.values()):
        if getattr(mod, '__name__', '').startswith(__name__ + '.'):
            importlib.reload(mod)

_loaded = True

# Note: bpy is imported lazily so the core (geometry, VEX, SOPs) can be used and tested
# outside of Blender.


def register():
    import bpy
    from .bl import tree, ui
    for cls in tree.classes:
        bpy.utils.register_class(cls)
    for cls in tree.build_node_classes():
        bpy.utils.register_class(cls)
    import nodeitems_utils
    nodeitems_utils.register_node_categories("BLENDINI_NODES", _node_categories())
    from .bl import vop
    vop.register()
    ui.register()


def unregister():
    import bpy
    from .bl import tree, ui
    ui.unregister()
    from .bl import vop
    vop.unregister()
    import nodeitems_utils
    try:
        nodeitems_utils.unregister_node_categories("BLENDINI_NODES")
    except KeyError:
        pass
    for cls in reversed(tree.NODE_CLASSES):
        bpy.utils.unregister_class(cls)
    for cls in reversed(tree.classes):
        bpy.utils.unregister_class(cls)


def _node_categories():
    from nodeitems_utils import NodeCategory, NodeItem
    from .sops import REGISTRY, CATEGORIES
    from .bl.tree import TREE_ID, NODE_PREFIX

    class BlendiniCategory(NodeCategory):
        @classmethod
        def poll(cls, context):
            return context.space_data.tree_type == TREE_ID

    cats = []
    for cat in CATEGORIES:
        items = [NodeItem(NODE_PREFIX + sid) for sid, sop in sorted(REGISTRY.items(), key=lambda kv: kv[1].label)
                 if sop.category == cat]
        if items:
            cats.append(BlendiniCategory("BLENDINI_" + cat.upper(), cat, items=items))
    return cats
