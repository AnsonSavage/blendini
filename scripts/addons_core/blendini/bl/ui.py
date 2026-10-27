# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Operators, panels, menus, handlers and key-maps (the Houdini look and feel)."""

import bpy
from bpy.app.handlers import persistent
from bpy.props import StringProperty, BoolProperty

from ..sops import REGISTRY, CATEGORIES
from ..core.geometry import POINT, PRIM, VERTEX, DETAIL
from ..vex.values import type_of_array
from .tree import TREE_ID, NODE_PREFIX, BlendiniNodeBase, update_colors
from . import cook


def _active_node(context):
    space = context.space_data
    tree = getattr(space, 'edit_tree', None)
    if tree is None or tree.bl_idname != TREE_ID:
        return None, None
    node = tree.nodes.active
    if node is not None and not isinstance(node, BlendiniNodeBase):
        node = None
    return tree, node


def tree_for_object(obj):
    """The Blendini network whose output is obj."""
    if obj is None:
        return None
    for tree in cook.all_trees():
        if tree.target == obj:
            return tree
    return None


def _in_blendini_tree(context):
    space = context.space_data
    return space is not None and space.type == 'NODE_EDITOR' and getattr(space, 'tree_type', '') == TREE_ID


# ----------------------------------------------------------------------------
# Drawing


def draw_node_header(node, context, layout):
    row = layout.row(align=True)
    op = row.operator("blendini.set_display", text="", icon='HIDE_OFF' if node.display_flag else 'HIDE_ON',
                      depress=node.display_flag)
    op.node_name = node.name
    row.prop(node, "template_flag", text="", icon='MESH_DATA', toggle=True)
    op = row.operator("blendini.toggle_bypass", text="", icon='CANCEL', depress=node.mute)
    op.node_name = node.name
    info = cook.node_info(node)
    if info.get('error'):
        row.label(text="Error", icon='ERROR')
    elif node.mute:
        row.label(text="Bypassed")
    elif 'geo' in info:
        g = info['geo']
        row.label(text="%d pts  %d prims" % (g.num_points, g.num_prims))
    if info.get('error'):
        col = layout.column(align=True)
        for line in _wrap(info['error'], 40)[:4]:
            col.label(text=line)


def _wrap(text, width):
    import textwrap
    out = []
    for para in str(text).splitlines() or ['']:
        out.extend(textwrap.wrap(para, width) or [''])
    return out


def _hidden(node, parm):
    if not parm.hide_when:
        return False
    for other, vals in parm.hide_when.items():
        cur = getattr(node, 'p_' + other, None)
        if not isinstance(vals, tuple):
            vals = (vals,)
        if cur in vals:
            return True
    return False


def draw_parms(node, context, layout, compact=False):
    sop = node.sop
    col = layout.column()
    col.use_property_split = not compact
    col.use_property_decorate = not compact
    for parm in sop.parms:
        if _hidden(node, parm):
            continue
        pname = 'p_' + parm.name
        if parm.type == 'code':
            _draw_code(node, col, compact)
        elif parm.type == 'object':
            col.prop_search(node, pname, bpy.data, "objects", text=parm.label)
        elif parm.type == 'nodegroup':
            col.prop_search(node, pname, bpy.data, "node_groups", text=parm.label)
        elif parm.type == 'image':
            col.prop_search(node, pname, bpy.data, "images", text=parm.label)
        elif parm.type in ('vector', 'vector2') and not compact:
            row = col.row(align=True)
            row.prop(node, pname, text=parm.label)
        else:
            col.prop(node, pname, text=parm.label)
    if sop.category == 'VEX':
        _draw_spare(node, layout, compact)
    if sop.id == 'edit':
        idx, d = node.edit_deltas()
        box = layout.box()
        box.label(text="%d edited points" % len(idx), icon='EDITMODE_HLT')
        col2 = box.column(align=True)
        col2.operator("blendini.edit_begin", icon='EDITMODE_HLT')
        col2.operator("blendini.edit_commit", icon='CHECKMARK')
        col2.operator("blendini.edit_clear", icon='X')
    if sop.id == 'subnet':
        row = layout.row(align=True)
        row.prop(node, "subtree", text="")
        row.operator("blendini.dive_in", text="Dive In", icon='TRIA_DOWN_BAR')


def _draw_code(node, layout, compact):
    box = layout.box()
    row = box.row(align=True)
    row.label(text="VEXpression", icon='SCRIPT')
    op = row.operator("blendini.edit_code", text="Edit" if node.text is None else "Open", icon='TEXT')
    op.node_name = node.name
    op = row.operator("blendini.create_spare_parms", text="", icon='ADD')
    op.node_name = node.name
    row.operator("blendini.vex_functions", text="", icon='HELP')
    if node.text is not None:
        row = box.row(align=True)
        row.prop(node, "text", text="")
        lines = node.text.as_string().splitlines()
    else:
        box.prop(node, "p_snippet", text="")
        lines = node.p_snippet.splitlines()
    if not compact or len(lines) <= 12:
        code_col = box.column(align=True)
        code_col.scale_y = 0.75
        for i, line in enumerate(lines[:40]):
            code_col.label(text="%3d  %s" % (i + 1, line.replace('\t', '    ')))
        if len(lines) > 40:
            code_col.label(text="... %d more lines" % (len(lines) - 40))


def _draw_spare(node, layout, compact):
    if not len(node.spare):
        return
    box = layout.box()
    box.label(text="Spare Parameters", icon='PROPERTIES')
    col = box.column(align=True)
    for i, sp in enumerate(node.spare):
        row = col.row(align=True)
        if sp.kind == 'ramp':
            from .ramps import ramp_node
            row.label(text=sp.name, icon='FCURVE')
            op = row.operator("blendini.remove_spare_parm", text="", icon='X', emboss=False)
            op.node_name = node.name
            op.index = i
            rn = ramp_node(sp.ramp_key)
            if rn is not None and not compact:
                col.template_curve_mapping(rn, "mapping")
            continue
        attr = {'float': 'f', 'int': 'i', 'vector': 'v', 'color': 'c', 'string': 's'}[sp.kind]
        row.prop(sp, attr, text=sp.name)
        op = row.operator("blendini.remove_spare_parm", text="", icon='X', emboss=False)
        op.node_name = node.name
        op.index = i


# ----------------------------------------------------------------------------
# Operators


def _get_node(context, name):
    space = context.space_data
    tree = getattr(space, 'edit_tree', None)
    if tree is None:
        return None, None
    node = tree.nodes.get(name) if name else tree.nodes.active
    if node is not None and not isinstance(node, BlendiniNodeBase):
        node = None
    return tree, node


class BLENDINI_OT_set_display(bpy.types.Operator):
    """Set the display flag (show this node's result in the viewport)"""
    bl_idname = "blendini.set_display"
    bl_label = "Set Display Flag"
    bl_options = {'REGISTER', 'UNDO'}
    node_name: StringProperty()

    @classmethod
    def poll(cls, context):
        return _in_blendini_tree(context)

    def execute(self, context):
        tree, node = _get_node(context, self.node_name)
        if node is None:
            return {'CANCELLED'}
        for n in tree.nodes:
            if isinstance(n, BlendiniNodeBase):
                n.display_flag = (n == node)
        cook.request_cook(tree)
        update_colors(tree)
        return {'FINISHED'}


class BLENDINI_OT_toggle_bypass(bpy.types.Operator):
    """Toggle the bypass flag (pass the first input through unchanged)"""
    bl_idname = "blendini.toggle_bypass"
    bl_label = "Toggle Bypass"
    bl_options = {'REGISTER', 'UNDO'}
    node_name: StringProperty()

    @classmethod
    def poll(cls, context):
        return _in_blendini_tree(context)

    def execute(self, context):
        tree, node = _get_node(context, self.node_name)
        nodes = [node] if node is not None and self.node_name else [n for n in tree.nodes if n.select and isinstance(n, BlendiniNodeBase)]
        if not nodes:
            return {'CANCELLED'}
        state = not nodes[0].mute
        for n in nodes:
            n.mute = state
        cook.request_cook(tree)
        return {'FINISHED'}


class BLENDINI_OT_dive_in(bpy.types.Operator):
    """Dive into the selected subnetwork (Houdini: I)"""
    bl_idname = "blendini.dive_in"
    bl_label = "Dive Into Subnetwork"

    @classmethod
    def poll(cls, context):
        tree, node = _active_node(context)
        return node is not None and node.sop_id == 'subnet' and node.subtree is not None

    def execute(self, context):
        tree, node = _active_node(context)
        context.space_data.path.append(node.subtree, node=node)
        return {'FINISHED'}


class BLENDINI_OT_jump_up(bpy.types.Operator):
    """Go back up to the parent network (Houdini: U)"""
    bl_idname = "blendini.jump_up"
    bl_label = "Jump Up"

    @classmethod
    def poll(cls, context):
        return _in_blendini_tree(context) and len(context.space_data.path) > 1

    def execute(self, context):
        context.space_data.path.pop()
        return {'FINISHED'}


def _edit_node(context):
    tree, node = _active_node(context)
    if node is None or node.sop_id != 'edit':
        return None, None
    return tree, node


class BLENDINI_OT_edit_begin(bpy.types.Operator):
    """Show this Edit node's result and enter Edit Mode on the output object to move points by hand"""
    bl_idname = "blendini.edit_begin"
    bl_label = "Edit Points in Viewport"

    @classmethod
    def poll(cls, context):
        return _edit_node(context)[1] is not None

    def execute(self, context):
        tree, node = _edit_node(context)
        for n in tree.nodes:
            if isinstance(n, BlendiniNodeBase):
                n.display_flag = (n == node)
        cook.cook_tree(tree)
        obj = tree.target
        if obj is None:
            return {'CANCELLED'}
        if context.view_layer.objects.get(obj.name) is None:
            self.report({'ERROR'}, "The output object is not in the current view layer")
            return {'CANCELLED'}
        if context.object is not None and context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='EDIT')
        self.report({'INFO'}, "Move points, then press 'Commit Edits' on the Edit node")
        return {'FINISHED'}


class BLENDINI_OT_edit_commit(bpy.types.Operator):
    """Leave Edit Mode and store the point offsets on the Edit node"""
    bl_idname = "blendini.edit_commit"
    bl_label = "Commit Edits"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _edit_node(context)[1] is not None

    def execute(self, context):
        import numpy as np
        tree, node = _edit_node(context)
        obj = tree.target
        if obj is None:
            return {'CANCELLED'}
        if obj.mode == 'EDIT':
            with context.temp_override(active_object=obj, object=obj):
                bpy.ops.object.mode_set(mode='OBJECT')
        ups = cook._input_nodes(node.inputs[0])
        if not ups:
            return {'CANCELLED'}
        scene = context.scene
        base = cook.node_geometry(ups[0])
        mesh = obj.data
        if base is None or len(mesh.vertices) != base.num_points:
            self.report({'ERROR'}, "Point count changed: the Edit node only supports moving points")
            cook.request_cook(tree)
            return {'CANCELLED'}
        co = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
        mesh.vertices.foreach_get("co", co)
        delta = co.reshape(-1, 3).astype(np.float64) - base.P
        moved = np.nonzero(np.linalg.norm(delta, axis=1) > 1e-6)[0]
        node["edit_idx"] = moved.tolist()
        node["edit_delta"] = delta[moved].ravel().tolist()
        cook.request_cook(tree)
        cook.cook_tree(tree)
        self.report({'INFO'}, "Stored edits for %d points" % len(moved))
        return {'FINISHED'}


class BLENDINI_OT_edit_clear(bpy.types.Operator):
    """Remove all stored edits from the Edit node"""
    bl_idname = "blendini.edit_clear"
    bl_label = "Clear Edits"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return _edit_node(context)[1] is not None

    def execute(self, context):
        tree, node = _edit_node(context)
        node["edit_idx"] = []
        node["edit_delta"] = []
        cook.cook_tree(tree)
        return {'FINISHED'}


class BLENDINI_OT_copy_text(bpy.types.Operator):
    """Copy to the clipboard"""
    bl_idname = "blendini.copy_text"
    bl_label = "Copy"
    text: StringProperty()

    def execute(self, context):
        context.window_manager.clipboard = self.text
        self.report({'INFO'}, "Copied '%s'" % self.text)
        return {'FINISHED'}


class BLENDINI_OT_vex_functions(bpy.types.Operator):
    """List the VEX functions Blendini supports (click to copy)"""
    bl_idname = "blendini.vex_functions"
    bl_label = "VEX Functions"

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=720)

    def draw(self, context):
        from ..vex import builtins
        layout = self.layout
        layout.label(text="Supported VEX functions (%d). Click to copy." % len(builtins.FUNCTIONS), icon='SCRIPT')
        flow = layout.column_flow(columns=6, align=True)
        for name in sorted(builtins.FUNCTIONS):
            op = flow.operator("blendini.copy_text", text=name, emboss=False)
            op.text = name + "()"

    def execute(self, context):
        return {'FINISHED'}


class BLENDINI_OT_cook(bpy.types.Operator):
    """Force a full recook of the network"""
    bl_idname = "blendini.cook"
    bl_label = "Force Recook"

    @classmethod
    def poll(cls, context):
        return _in_blendini_tree(context)

    def execute(self, context):
        tree = context.space_data.edit_tree
        cook.clear_cache()
        cook.cook_tree(tree)
        return {'FINISHED'}


class BLENDINI_OT_edit_code(bpy.types.Operator):
    """Edit the VEX code in a Text Editor (it recooks live as you type)"""
    bl_idname = "blendini.edit_code"
    bl_label = "Edit VEX Code"
    node_name: StringProperty()

    @classmethod
    def poll(cls, context):
        return _in_blendini_tree(context)

    def execute(self, context):
        tree, node = _get_node(context, self.node_name)
        if node is None or not hasattr(node, 'p_snippet'):
            return {'CANCELLED'}
        if node.text is None:
            # The .osl extension gives C-like syntax highlighting that suits VEX.
            text = bpy.data.texts.new("%s_%s.vex.osl" % (tree.name, node.name))
            text.write(node.p_snippet)
            node.text = text
        text = node.text
        area = _find_or_make_text_editor(context)
        if area is not None:
            area.spaces.active.text = text
            area.spaces.active.show_line_numbers = True
            area.spaces.active.show_syntax_highlight = True
        else:
            self.report({'INFO'}, "Open a Text Editor and select '%s' to edit the code" % text.name)
        return {'FINISHED'}


def _find_or_make_text_editor(context):
    screen = context.screen
    if screen is None:
        return None
    for area in screen.areas:
        if area.type == 'TEXT_EDITOR':
            return area
    # Split the node editor and turn the new half into a text editor.
    area = context.area
    if area is None:
        return None
    before = set(screen.areas[:])
    try:
        with context.temp_override(area=area):
            bpy.ops.screen.area_split(direction='VERTICAL', factor=0.6)
    except Exception:
        return None
    new = [a for a in screen.areas if a not in before]
    if not new:
        return None
    new_area = new[0]
    new_area.type = 'TEXT_EDITOR'
    return new_area


class BLENDINI_OT_create_spare_parms(bpy.types.Operator):
    """Create spare parameters for each unique call of ch() in the code"""
    bl_idname = "blendini.create_spare_parms"
    bl_label = "Create Spare Parameters"
    bl_options = {'REGISTER', 'UNDO'}
    node_name: StringProperty()

    @classmethod
    def poll(cls, context):
        return _in_blendini_tree(context)

    def execute(self, context):
        from ..vex.wrangle import find_channels
        tree, node = _get_node(context, self.node_name)
        if node is None:
            return {'CANCELLED'}
        code = node.parm_values().get('snippet', '')
        existing = {sp.name for sp in node.spare}
        added = 0
        for name, kind in find_channels(code):
            if name in existing:
                continue
            sp = node.spare.add()
            sp.name = name
            sp.kind = {'vector2': 'vector', 'vector4': 'vector'}.get(kind, kind)
            if kind == 'ramp':
                from .ramps import new_key, ramp_node
                sp.ramp_key = new_key()
                ramp_node(sp.ramp_key, create=True)
            added += 1
        self.report({'INFO'}, "Added %d spare parameter(s)" % added)
        cook.request_cook(tree)
        return {'FINISHED'}


class BLENDINI_OT_remove_spare_parm(bpy.types.Operator):
    """Remove this spare parameter"""
    bl_idname = "blendini.remove_spare_parm"
    bl_label = "Remove Spare Parameter"
    bl_options = {'REGISTER', 'UNDO'}
    node_name: StringProperty()
    index: bpy.props.IntProperty()

    def execute(self, context):
        tree, node = _get_node(context, self.node_name)
        if node is None:
            return {'CANCELLED'}
        node.spare.remove(self.index)
        cook.request_cook(tree)
        return {'FINISHED'}


STARTER_GRAPH = [
    # (sop id, location, parm overrides)
    ('sphere', (0, 300), {'p_type': 'polymesh', 'p_freq': 4}),
    ('pointwrangle', (0, 100), {}),
    ('null', (0, -100), {}),
]


class BLENDINI_OT_new_network(bpy.types.Operator):
    """Create a Blendini geometry network (like /obj/geo1 in Houdini) and an object showing its output"""
    bl_idname = "blendini.new_network"
    bl_label = "Blendini Geometry"
    bl_options = {'REGISTER', 'UNDO'}

    starter: BoolProperty(name="Starter Network", default=True)
    primitive: StringProperty(name="Primitive", default="",
                              description="Create the network with just this SOP (Houdini shelf tool)")

    def execute(self, context):
        base = self.primitive or "geo"
        i = 1
        while bpy.data.node_groups.get("%s%d" % (base, i)) or bpy.data.objects.get("%s%d" % (base, i)):
            i += 1
        name = "%s%d" % (base, i)
        tree = bpy.data.node_groups.new(name, TREE_ID)
        tree.use_fake_user = True
        mesh = bpy.data.meshes.new(name)
        obj = bpy.data.objects.new(name, mesh)
        context.collection.objects.link(obj)
        obj["blendini_tree"] = tree.name
        tree.target = obj
        from .display import setup_target
        setup_target(obj)
        _show_attribute_colors(context)
        for o in context.view_layer.objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
        if self.primitive:
            node = tree.nodes.new(NODE_PREFIX + self.primitive)
            node.display_flag = True
            tree.nodes.active = node
        elif self.starter:
            build_starter(tree)
        show_tree_in_editor(context, tree)
        cook.cook_tree(tree)
        return {'FINISHED'}


def _show_attribute_colors(context):
    """Houdini displays Cd by default: switch solid shading to attribute colors."""
    screen = context.screen
    if screen is None:
        return
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            shading = area.spaces.active.shading
            if shading.type == 'SOLID':
                shading.color_type = 'VERTEX'


def build_starter(tree):
    prev = None
    for sop_id, loc, overrides in STARTER_GRAPH:
        node = tree.nodes.new(NODE_PREFIX + sop_id)
        node.location = loc
        for k, v in overrides.items():
            setattr(node, k, v)
        if prev is not None:
            tree.links.new(prev.outputs[0], node.inputs[0])
        prev = node
        if sop_id == 'pointwrangle':
            for name, val in (('freq', 2.0), ('amp', 0.3)):
                sp = node.spare.add()
                sp.name = name
                sp.kind = 'float'
                sp.f = val
    for n in tree.nodes:
        if isinstance(n, BlendiniNodeBase):
            n.display_flag = (n == prev)
    tree.nodes.active = prev


def show_tree_in_editor(context, tree):
    screen = context.screen
    if screen is None:
        return
    target = None
    for area in screen.areas:
        if area.type == 'NODE_EDITOR':
            target = area
            break
    if target is None:
        # Houdini keeps the network view below the viewport: reuse the timeline / dopesheet.
        for area in screen.areas:
            if area.type in ('DOPESHEET_EDITOR', 'TIMELINE', 'GRAPH_EDITOR', 'NLA_EDITOR'):
                target = area
                break
    if target is None:
        return
    target.type = 'NODE_EDITOR'
    space = target.spaces.active
    space.tree_type = TREE_ID
    space.node_tree = tree
    space.show_region_ui = True


class BLENDINI_OT_houdini_layout(bpy.types.Operator):
    """Arrange the current screen like Houdini: viewport left, network editor right, spreadsheet below"""
    bl_idname = "blendini.houdini_layout"
    bl_label = "Houdini-like Layout"

    def execute(self, context):
        screen = context.screen
        if screen is None:
            return {'CANCELLED'}
        tree = tree_for_object(context.active_object)
        if tree is None:
            trees = cook.all_trees()
            tree = trees[0] if trees else None
        if any(a.type == 'NODE_EDITOR' and getattr(a.spaces.active, 'tree_type', '') == TREE_ID for a in screen.areas):
            node_area = next(a for a in screen.areas if a.type == 'NODE_EDITOR')
        else:
            view3d = max((a for a in screen.areas if a.type == 'VIEW_3D'), key=lambda a: a.width * a.height, default=None)
            if view3d is None:
                self.report({'WARNING'}, "No 3D Viewport to split")
                return {'CANCELLED'}
            before = set(screen.areas[:])
            with context.temp_override(area=view3d):
                bpy.ops.screen.area_split(direction='VERTICAL', factor=0.55)
            new = [a for a in screen.areas if a not in before]
            if not new:
                return {'CANCELLED'}
            node_area = new[0]
            if node_area.x < view3d.x:
                node_area, view3d = view3d, node_area
                view3d.type = 'VIEW_3D'
            node_area.type = 'NODE_EDITOR'
        space = node_area.spaces.active
        space.tree_type = TREE_ID
        if tree is not None:
            space.node_tree = tree
        space.show_region_ui = True
        # The timeline becomes the geometry spreadsheet (Houdini's Geometry Spreadsheet pane).
        for area in screen.areas:
            if area.type in ('DOPESHEET_EDITOR', 'TIMELINE'):
                area.type = 'SPREADSHEET'
                break
        _show_attribute_colors(context)
        return {'FINISHED'}


class BLENDINI_OT_export_obj(bpy.types.Operator):
    """Write the active node's geometry to an .obj file (like a ROP Geometry Output)"""
    bl_idname = "blendini.export_obj"
    bl_label = "Save Geometry to Disk"
    filepath: StringProperty(subtype='FILE_PATH', default="//geo.obj")
    filter_glob: StringProperty(default="*.obj", options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        tree, node = _active_node(context)
        return node is not None and cook.node_geometry(node) is not None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        from ..sops.modify import write_obj
        from ..core.instances import unpack
        tree, node = _active_node(context)
        geo = unpack(cook.node_geometry(node))
        path = bpy.path.abspath(self.filepath)
        write_obj(geo, path)
        self.report({'INFO'}, "Wrote %d points, %d primitives to %s" % (geo.num_points, geo.num_prims, path))
        return {'FINISHED'}


class BLENDINI_OT_houdini_theme(bpy.types.Operator):
    """Apply a Houdini-like color theme (grey gradient viewport, dark network editor)"""
    bl_idname = "blendini.houdini_theme"
    bl_label = "Apply Houdini-like Theme"

    def execute(self, context):
        apply_houdini_theme(context)
        self.report({'INFO'}, "Houdini-like theme applied (Preferences > Themes > Reset to undo)")
        return {'FINISHED'}


def apply_houdini_theme(context):
    theme = context.preferences.themes[0]
    v3d = theme.view_3d
    try:
        sp = v3d.space
        grad = sp.gradients
        grad.background_type = 'LINEAR'
        grad.high_gradient = (0.44, 0.47, 0.52)
        grad.gradient = (0.20, 0.21, 0.23)
    except AttributeError:
        pass
    for attr, val in (('grid', (0.33, 0.35, 0.38, 1.0)),):
        try:
            setattr(v3d, attr, val)
        except (AttributeError, TypeError, ValueError):
            pass
    ne = theme.node_editor
    for attr, val in (
        ('grid', (0.22, 0.22, 0.22)), ('wire', (0.05, 0.05, 0.05)), ('wire_inner', (0.72, 0.72, 0.72)),
        ('node_selected', (1.0, 0.84, 0.0)), ('node_active', (1.0, 0.95, 0.4)),
        ('node_backdrop', (0.30, 0.30, 0.30, 1.0)), ('noodle_curving', 0),
    ):
        try:
            setattr(ne, attr, val)
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        ne.space.back = (0.16, 0.16, 0.16)
    except (AttributeError, TypeError, ValueError):
        pass


# ----------------------------------------------------------------------------
# Panels


class BLENDINI_PT_parameters(bpy.types.Panel):
    """Houdini style Parameter pane"""
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Parameters"
    bl_label = "Parameters"

    @classmethod
    def poll(cls, context):
        return _in_blendini_tree(context)

    def draw_header(self, context):
        tree, node = _active_node(context)
        if node is not None:
            self.layout.label(text="%s  %s" % (node.sop.label, node.name), icon=node.bl_icon)

    def draw(self, context):
        layout = self.layout
        tree, node = _active_node(context)
        if node is None:
            layout.label(text="Select a node")
            return
        row = layout.row(align=True)
        row.prop(node, "name", text="")
        draw_node_header(node, context, row.column())
        draw_parms(node, context, layout)


class BLENDINI_PT_info(bpy.types.Panel):
    """Node info: geometry statistics and attributes (Houdini's middle-click info)"""
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Parameters"
    bl_label = "Geometry Info"

    @classmethod
    def poll(cls, context):
        return _in_blendini_tree(context)

    def draw(self, context):
        layout = self.layout
        tree, node = _active_node(context)
        if node is None:
            return
        info = cook.node_info(node)
        if not info:
            layout.label(text="Not cooked (set the display flag on a downstream node)")
            return
        if info.get('error'):
            box = layout.box()
            box.alert = True
            for line in _wrap(info['error'], 50):
                box.label(text=line)
        geo = info.get('geo')
        if geo is None:
            return
        layout.operator("blendini.export_obj", icon='EXPORT')
        col = layout.column(align=True)
        col.label(text="Cook time: %.2f ms" % (info.get('time', 0.0) * 1000))
        col.label(text="%d points   %d primitives   %d vertices" % (geo.num_points, geo.num_prims, geo.num_vertices))
        lo, hi = geo.bbox()
        col.label(text="Bounds  min (%.2f, %.2f, %.2f)" % tuple(lo))
        col.label(text="             max (%.2f, %.2f, %.2f)" % tuple(hi))
        for cls, label in ((POINT, "Point"), (VERTEX, "Vertex"), (PRIM, "Primitive"), (DETAIL, "Detail")):
            attrs = geo.attribs[cls]
            if not attrs:
                continue
            box = layout.box()
            box.label(text="%s Attributes" % label)
            flow = box.column(align=True)
            for name, arr in attrs.items():
                if name.startswith('__'):
                    continue
                flow.label(text="  %s  (%s)" % (name, type_of_array(arr)))
        groups = [(c, n) for c in (POINT, PRIM) for n in geo.groups[c]]
        if groups:
            box = layout.box()
            box.label(text="Groups")
            for c, n in groups:
                box.label(text="  %s  (%s, %d)" % (n, c, int(geo.groups[c][n].sum())))
        msgs = info.get('messages') or []
        if msgs:
            box = layout.box()
            box.label(text="Console (printf)", icon='CONSOLE')
            col = box.column(align=True)
            col.scale_y = 0.75
            for m in msgs[-30:]:
                for line in str(m).rstrip('\n').splitlines():
                    col.label(text=line)


def _fmt_cell(v):
    import numpy as np
    if isinstance(v, np.ndarray):
        if v.ndim == 0:
            return _fmt_cell(v.item())
        return " ".join(_fmt_cell(x) for x in v.ravel()[:16])
    if isinstance(v, (float, np.floating)):
        return "%.3f" % v
    if isinstance(v, (bool, np.bool_)):
        return "1" if v else "0"
    return str(v)


class BLENDINI_PT_spreadsheet(bpy.types.Panel):
    """Geometry spreadsheet of the active node (Houdini's Geometry Spreadsheet)"""
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Spreadsheet"
    bl_label = "Geometry Spreadsheet"

    @classmethod
    def poll(cls, context):
        return _in_blendini_tree(context) and context.space_data.edit_tree is not None

    def draw(self, context):
        import numpy as np
        layout = self.layout
        tree, node = _active_node(context)
        tree = context.space_data.edit_tree
        if node is None:
            layout.label(text="Select a node")
            return
        geo = cook.node_geometry(node)
        if geo is None:
            layout.label(text="%s is not cooked" % node.name)
            return
        row = layout.row(align=True)
        row.prop(tree, "sheet_class", expand=True)
        row = layout.row(align=True)
        row.prop(tree, "sheet_offset")
        row.prop(tree, "sheet_rows")
        cls = tree.sheet_class
        n = geo.count(cls)
        layout.label(text="%s: %d %s" % (node.name, n, cls + ("s" if cls != 'detail' else "")))
        cols = [(name, arr) for name, arr in geo.attribs[cls].items() if not name.startswith('__')]
        groups = list(geo.groups.get(cls, {}).items()) if cls in ('point', 'prim') else []
        if cls == 'vertex':
            cols = [("point", geo.vtx_point)] + cols
        if cls == 'prim':
            cols = [("size", geo.prim_sizes())] + cols
        start = min(tree.sheet_offset, max(n - 1, 0))
        stop = min(start + tree.sheet_rows, n)
        box = layout.box()
        flow = box.grid_flow(row_major=True, columns=1 + len(cols) + len(groups), even_columns=False, align=True)
        flow.label(text="#")
        for name, arr in cols:
            flow.label(text=name)
        for name, g in groups:
            flow.label(text="grp:" + name)
        for i in range(start, stop):
            flow.label(text=str(i))
            for name, arr in cols:
                flow.label(text=_fmt_cell(arr[i]) if i < len(arr) else "")
            for name, g in groups:
                flow.label(text="1" if g[i] else "0")


class BLENDINI_PT_network(bpy.types.Panel):
    bl_space_type = 'NODE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Blendini"
    bl_label = "Network"

    @classmethod
    def poll(cls, context):
        return _in_blendini_tree(context) and context.space_data.edit_tree is not None

    def draw(self, context):
        layout = self.layout
        tree = context.space_data.edit_tree
        layout.prop(tree, "target")
        layout.prop(tree, "auto_cook")
        layout.prop(tree, "show_parms_in_nodes")
        box = layout.box()
        box.label(text="Viewport Display", icon='OVERLAY')
        row = box.row(align=True)
        row.prop(tree, "show_point_numbers", toggle=True, text="Point #")
        row.prop(tree, "show_prim_numbers", toggle=True, text="Prim #")
        row.prop(tree, "show_normals", toggle=True, text="Normals")
        if tree.show_normals:
            box.prop(tree, "normal_scale")
        layout.operator("blendini.cook", icon='FILE_REFRESH')
        layout.operator("blendini.houdini_theme", icon='COLOR')
        layout.operator("blendini.houdini_layout", icon='WORKSPACE')
        layout.operator("blendini.load_examples", icon='PRESET')
        box = layout.box()
        box.label(text="Hotkeys", icon='EVENT_TAB')
        col = box.column(align=True)
        col.scale_y = 0.8
        col.label(text="Tab: add node (type to search)")
        col.label(text="R: display flag   M: bypass")
        col.label(text="I: dive into subnet   U: jump up")
        col.label(text="Ctrl+Enter in text editor: recook")


SHELF_TOOLS = ('box', 'sphere', 'tube', 'torus', 'grid', 'circle', 'line', 'platonic', 'font', 'lsystem')


class BLENDINI_PT_shelf(bpy.types.Panel):
    """Houdini's Create shelf: each tool makes a new geometry object with that SOP inside"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Blendini"
    bl_label = "Create"

    def draw(self, context):
        layout = self.layout
        flow = layout.grid_flow(columns=2, even_columns=True, align=True)
        for sop_id in SHELF_TOOLS:
            sop = REGISTRY.get(sop_id)
            if sop is None:
                continue
            op = flow.operator("blendini.new_network", text=sop.label, icon=sop.icon if sop.icon != 'NONE' else 'NODE')
            op.primitive = sop_id
            op.starter = False
        col = layout.column(align=True)
        op = col.operator("blendini.new_network", text="Empty Geometry", icon='MESH_DATA')
        op.starter = False
        op.primitive = ""
        col.operator("blendini.load_examples", icon='PRESET')


class BLENDINI_PT_view3d(bpy.types.Panel):
    """Display options for the Blendini network of the active object (Houdini's display toolbar)"""
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Blendini"
    bl_label = "Blendini Display"

    @classmethod
    def poll(cls, context):
        return tree_for_object(context.active_object) is not None

    def draw(self, context):
        layout = self.layout
        tree = tree_for_object(context.active_object)
        if tree is None:
            layout.label(text="Network not found")
            return
        layout.label(text=tree.name, icon='NODETREE')
        col = layout.column(align=True)
        col.prop(tree, "show_point_numbers")
        col.prop(tree, "show_prim_numbers")
        col.prop(tree, "show_normals")
        if tree.show_normals:
            col.prop(tree, "normal_scale")
        node = cook.display_node(tree)
        geo = cook.node_geometry(node) if node is not None else None
        if geo is not None:
            box = layout.box()
            box.label(text="Display: %s" % node.name, icon='HIDE_OFF')
            box.label(text="%d points, %d prims" % (geo.num_points, geo.num_prims))
        layout.operator("blendini.houdini_layout", icon='WORKSPACE')


# ----------------------------------------------------------------------------
# Menus


class BLENDINI_OT_add_node(bpy.types.Operator):
    """Add a SOP. Like Houdini, it is wired to the selected node and takes over its display flag"""
    bl_idname = "blendini.add_node"
    bl_label = "Add SOP"
    bl_options = {'REGISTER', 'UNDO'}
    type: StringProperty()

    @classmethod
    def poll(cls, context):
        return _in_blendini_tree(context) and context.space_data.edit_tree is not None

    def invoke(self, context, event):
        tree = context.space_data.edit_tree
        prev = tree.nodes.active
        if prev is not None and (not prev.select or not isinstance(prev, BlendiniNodeBase) or not prev.outputs):
            prev = None
        result = bpy.ops.node.add_node('INVOKE_DEFAULT', type=self.type, use_transform=True)
        if 'CANCELLED' in result:
            return {'CANCELLED'}
        new = tree.nodes.active
        if prev is not None and new is not None and new != prev and isinstance(new, BlendiniNodeBase):
            if new.inputs:
                # Insert after prev: re-wire prev's downstream links through the new node.
                downstream = [(l.to_socket) for l in prev.outputs[0].links]
                tree.links.new(prev.outputs[0], new.inputs[0])
                for sock in downstream:
                    if sock.node != new:
                        tree.links.new(new.outputs[0], sock)
            if prev.display_flag and new.inputs:
                for n in tree.nodes:
                    if isinstance(n, BlendiniNodeBase):
                        n.display_flag = (n == new)
            cook.request_cook(tree)
        return {'FINISHED'}


def _category_menu_class(category):
    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_REGION_WIN'
        for sop_id, sop in sorted(REGISTRY.items(), key=lambda kv: kv[1].label):
            if sop.category == category:
                op = layout.operator("blendini.add_node", text=sop.label, icon=sop.icon if sop.icon != 'NONE' else 'NODE')
                op.type = NODE_PREFIX + sop_id
    return type("BLENDINI_MT_add_" + category.lower(), (bpy.types.Menu,), {
        'bl_idname': "BLENDINI_MT_add_" + category.lower(),
        'bl_label': category,
        'bl_space_type': 'NODE_EDITOR',
        'draw': draw,
    })


CATEGORY_MENUS = [_category_menu_class(c) for c in CATEGORIES]


class BLENDINI_MT_tab(bpy.types.Menu):
    """Houdini TAB menu"""
    bl_idname = "BLENDINI_MT_tab"
    bl_label = "Tab Menu"
    bl_space_type = 'NODE_EDITOR'
    bl_options = {'SEARCH_ON_KEY_PRESS'}

    def draw(self, context):
        layout = self.layout
        layout.operator_context = 'INVOKE_REGION_WIN'
        if layout.operator_context == 'EXEC_REGION_WIN':
            pass
        for menu in CATEGORY_MENUS:
            layout.menu(menu.bl_idname)


def draw_add_menu(self, context):
    if not _in_blendini_tree(context):
        return
    layout = self.layout
    for menu in CATEGORY_MENUS:
        layout.menu(menu.bl_idname)


class BLENDINI_OT_tab_menu(bpy.types.Operator):
    """Open the Houdini style TAB menu"""
    bl_idname = "blendini.tab_menu"
    bl_label = "Tab Menu"

    @classmethod
    def poll(cls, context):
        return _in_blendini_tree(context) and context.space_data.edit_tree is not None

    def invoke(self, context, event):
        bpy.ops.wm.call_menu(name=BLENDINI_MT_tab.bl_idname)
        return {'FINISHED'}


def draw_view3d_add(self, context):
    self.layout.separator()
    self.layout.operator(BLENDINI_OT_new_network.bl_idname, icon='MESH_ICOSPHERE')
    self.layout.operator("blendini.load_examples", text="Blendini Examples", icon='PRESET')
    self.layout.operator("blendini.houdini_layout", text="Blendini Houdini Layout", icon='WORKSPACE')


# ----------------------------------------------------------------------------
# Handlers


@persistent
def on_frame_change(scene, depsgraph=None):
    if cook.is_cooking():
        return
    for tree in cook.all_trees():
        if tree.auto_cook:
            cook.cook_tree(tree)


@persistent
def on_load_post(*_args):
    cook.clear_cache()
    try:
        from ..sops.blender_ops import remove_stale_rbd_scenes
        remove_stale_rbd_scenes()
    except Exception as ex:
        print("Blendini:", ex)
    for tree in cook.all_trees():
        if tree.auto_cook:
            cook.request_cook(tree)


@persistent
def on_depsgraph_update(scene, depsgraph):
    """Recook networks with Object Merge nodes when their source objects change."""
    if cook.is_cooking():
        return
    targets = {t.target for t in cook.all_trees() if t.target is not None}
    changed = set()
    for upd in depsgraph.updates:
        idb = upd.id
        if isinstance(idb, bpy.types.Object) and idb.original not in targets and not idb.name.startswith('_blendini'):
            if upd.is_updated_geometry or upd.is_updated_transform:
                changed.add(idb.original.name)
    if not changed:
        return
    for tree in cook.all_trees():
        for node in tree.nodes:
            if getattr(node, 'sop_id', '') == 'objectmerge' and node.p_objpath in changed:
                cook.bump_objmerge_epoch()
                cook.request_cook(tree)
                break


_text_hashes = {}


def _poll_texts():
    """Live coding: recook when VEX code in a linked Text datablock changes."""
    try:
        for tree in cook.all_trees():
            for node in tree.nodes:
                text = getattr(node, 'text', None)
                spare = getattr(node, 'spare', None)
                has_ramp = spare is not None and any(sp.kind == 'ramp' for sp in spare)
                if text is None and not has_ramp:
                    continue
                key = (tree.name, node.name)
                h = hash((text.as_string() if text is not None else '', tuple(sorted(node.ramps().items()))))
                if _text_hashes.get(key) != h:
                    if key in _text_hashes:
                        cook.request_cook(tree)
                    _text_hashes[key] = h
    except ReferenceError:
        pass
    return 0.3


# ----------------------------------------------------------------------------
# Registration

addon_keymaps = []

classes = (
    BLENDINI_OT_set_display,
    BLENDINI_OT_toggle_bypass,
    BLENDINI_OT_cook,
    BLENDINI_OT_copy_text,
    BLENDINI_OT_vex_functions,
    BLENDINI_OT_dive_in,
    BLENDINI_OT_jump_up,
    BLENDINI_OT_edit_begin,
    BLENDINI_OT_edit_commit,
    BLENDINI_OT_edit_clear,
    BLENDINI_OT_edit_code,
    BLENDINI_OT_create_spare_parms,
    BLENDINI_OT_remove_spare_parm,
    BLENDINI_OT_new_network,
    BLENDINI_OT_houdini_theme,
    BLENDINI_OT_houdini_layout,
    BLENDINI_OT_export_obj,
    BLENDINI_OT_tab_menu,
    BLENDINI_OT_add_node,
    BLENDINI_PT_parameters,
    BLENDINI_PT_info,
    BLENDINI_PT_network,
    BLENDINI_PT_spreadsheet,
    BLENDINI_PT_shelf,
    BLENDINI_PT_view3d,
    BLENDINI_MT_tab,
) + tuple(CATEGORY_MENUS)


def _all_classes():
    from .examples import BLENDINI_OT_load_examples
    return classes + (BLENDINI_OT_load_examples,)


def register():
    for cls in _all_classes():
        bpy.utils.register_class(cls)
    from . import overlay
    overlay.register()
    bpy.types.VIEW3D_MT_add.append(draw_view3d_add)
    bpy.app.handlers.frame_change_post.append(on_frame_change)
    bpy.app.handlers.load_post.append(on_load_post)
    bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)
    if not bpy.app.background:
        bpy.app.timers.register(_poll_texts, first_interval=1.0, persistent=True)
    kc = bpy.context.window_manager.keyconfigs.addon
    if kc is not None:
        km = kc.keymaps.new(name="Node Editor", space_type='NODE_EDITOR')
        addon_keymaps.append((km, km.keymap_items.new(BLENDINI_OT_tab_menu.bl_idname, 'TAB', 'PRESS')))
        addon_keymaps.append((km, km.keymap_items.new(BLENDINI_OT_set_display.bl_idname, 'R', 'PRESS')))
        addon_keymaps.append((km, km.keymap_items.new(BLENDINI_OT_dive_in.bl_idname, 'I', 'PRESS')))
        addon_keymaps.append((km, km.keymap_items.new(BLENDINI_OT_jump_up.bl_idname, 'U', 'PRESS')))


def unregister():
    from . import overlay
    overlay.unregister()
    for km, kmi in addon_keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    addon_keymaps.clear()
    if bpy.app.timers.is_registered(_poll_texts):
        bpy.app.timers.unregister(_poll_texts)
    for lst, fn in ((bpy.app.handlers.frame_change_post, on_frame_change),
                    (bpy.app.handlers.load_post, on_load_post),
                    (bpy.app.handlers.depsgraph_update_post, on_depsgraph_update)):
        if fn in lst:
            lst.remove(fn)
    bpy.types.VIEW3D_MT_add.remove(draw_view3d_add)
    for cls in reversed(_all_classes()):
        bpy.utils.unregister_class(cls)
