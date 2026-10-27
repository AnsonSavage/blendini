# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""The SOP network node tree, its sockets and the generated node classes."""

import re

import bpy
from bpy.props import (
    BoolProperty, CollectionProperty, EnumProperty, FloatProperty, FloatVectorProperty, IntProperty,
    PointerProperty, StringProperty,
)

from ..sops import REGISTRY, CATEGORY_COLORS

TREE_ID = 'BlendiniSOPTree'
SOCKET_ID = 'BlendiniGeometrySocket'
NODE_PREFIX = 'BlendiniSOP_'

DISPLAY_COLOR = (0.22, 0.42, 0.72)
ERROR_COLOR = (0.65, 0.15, 0.15)


def _request_cook(self, context=None):
    from . import cook
    tree = self.id_data
    if tree is not None and getattr(tree, 'bl_idname', '') == TREE_ID:
        cook.request_cook(tree)


def _redraw_views():
    wm = bpy.context.window_manager
    for win in (wm.windows if wm else ()):
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()


class BlendiniGeometrySocket(bpy.types.NodeSocket):
    bl_idname = SOCKET_ID
    bl_label = "Geometry"

    def draw(self, context, layout, node, text):
        layout.label(text=text)

    @classmethod
    def draw_color_simple(cls):
        return (0.55, 0.62, 0.70, 1.0)


class BlendiniSpareParm(bpy.types.PropertyGroup):
    kind: EnumProperty(items=[
        ('float', "Float", ""), ('int', "Integer", ""), ('vector', "Vector", ""), ('color', "Color", ""),
        ('string', "String", ""), ('ramp', "Ramp", ""),
    ], update=_request_cook)
    ramp_key: StringProperty()
    f: FloatProperty(name="Value", default=1.0, update=_request_cook)
    i: IntProperty(name="Value", default=1, update=_request_cook)
    v: FloatVectorProperty(name="Value", size=3, default=(1.0, 1.0, 1.0), update=_request_cook)
    c: FloatVectorProperty(name="Color", size=3, subtype='COLOR', default=(1.0, 1.0, 1.0), min=0.0,
                           update=_request_cook)
    s: StringProperty(name="Value", update=_request_cook)
    soft_max: FloatProperty(name="Range", default=1.0)

    def value(self):
        k = self.kind
        if k == 'float':
            return self.f
        if k == 'int':
            return self.i
        if k == 'vector':
            return tuple(self.v)
        if k == 'color':
            return tuple(self.c)
        return self.s


class BlendiniNodeTree(bpy.types.NodeTree):
    """Houdini style SOP network, cooked into a Blender object"""
    bl_idname = TREE_ID
    bl_label = "Blendini SOP Network"
    bl_icon = 'MESH_ICOSPHERE'

    target: PointerProperty(
        name="Output Object", type=bpy.types.Object,
        description="Object that receives the cooked geometry of the display node",
        update=_request_cook,
    )
    auto_cook: BoolProperty(name="Auto Update", default=True,
                            description="Cook automatically when parameters change (Houdini: Auto Update)")
    show_point_numbers: BoolProperty(name="Point Numbers", default=False, update=lambda s, c: _redraw_views(),
                                     description="Display point numbers in the viewport")
    show_prim_numbers: BoolProperty(name="Primitive Numbers", default=False, update=lambda s, c: _redraw_views(),
                                    description="Display primitive numbers in the viewport")
    show_normals: BoolProperty(name="Normals", default=False, update=lambda s, c: _redraw_views(),
                               description="Display normals in the viewport")
    normal_scale: FloatProperty(name="Normal Scale", default=1.0, min=0.0, soft_max=10.0,
                                update=lambda s, c: _redraw_views())
    sheet_class: EnumProperty(name="Class", items=[
        ('point', "Points", ""), ('vertex', "Vertices", ""), ('prim', "Primitives", ""), ('detail', "Detail", ""),
    ])
    sheet_offset: IntProperty(name="Start", default=0, min=0)
    sheet_rows: IntProperty(name="Rows", default=25, min=1, max=200)
    owner_tree: PointerProperty(name="Owner", type=bpy.types.NodeTree,
                                description="For subnetworks: the network containing the subnet node")
    show_parms_in_nodes: BoolProperty(name="Show Parameters on Nodes", default=False,
                                      description="Draw parameters inside the nodes as well as in the Parameters pane")

    def update(self):
        from . import cook
        cook.request_cook(self)


class BlendiniNodeBase:
    sop_id = ''

    display_flag: BoolProperty(name="Display", default=False,
                               description="Display flag: this node's output is shown in the viewport")
    template_flag: BoolProperty(name="Template", default=False, update=_request_cook,
                                description="Template flag: show this node's geometry as a wireframe guide")
    text: PointerProperty(name="Code Text", type=bpy.types.Text, update=_request_cook,
                          description="Text datablock holding the VEX code (edit in the Text Editor)")
    spare: CollectionProperty(type=BlendiniSpareParm)
    subtree: PointerProperty(name="Network", type=bpy.types.NodeTree, update=_request_cook,
                             poll=lambda self, t: t.bl_idname == TREE_ID and t != self.id_data,
                             description="The network inside this subnet")

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == TREE_ID

    @property
    def sop(self):
        return REGISTRY[self.sop_id]

    def init(self, context):
        sop = self.sop
        for i in range(sop.inputs):
            label = sop.input_labels[i] if i < len(sop.input_labels) else "Input %d" % (i + 1)
            self.inputs.new(SOCKET_ID, label, use_multi_input=(sop.multi_input and i == 0))
        self.outputs.new(SOCKET_ID, "Output")
        self.name = unique_node_name(self.id_data, sop.id)
        self.width = 160
        self.use_custom_color = True
        self.color = CATEGORY_COLORS.get(sop.category, (0.3, 0.3, 0.3))
        tree = self.id_data
        if not any(getattr(n, 'display_flag', False) for n in tree.nodes if n != self):
            self.display_flag = True
        if sop.id == 'subnet':
            self.subtree = make_subnet_tree(tree, self.name)
        update_colors(tree)

    def copy(self, node):
        self.name = unique_node_name(self.id_data, self.sop_id)
        self.display_flag = False
        self.template_flag = False
        if self.text is not None:
            # The copy gets its own code: take the current text, then unlink the shared Text.
            if hasattr(self, 'p_snippet'):
                self.p_snippet = self.text.as_string()
            self.text = None

    def free(self):
        from . import cook
        cook.forget_node(self)

    def draw_label(self):
        return self.name

    def draw_buttons(self, context, layout):
        from . import ui
        ui.draw_node_header(self, context, layout)
        if self.id_data.show_parms_in_nodes:
            ui.draw_parms(self, context, layout, compact=True)

    def draw_buttons_ext(self, context, layout):
        from . import ui
        ui.draw_parms(self, context, layout)

    def parm_values(self):
        """Current parameter values as a ParmValues dict."""
        from ..sops import ParmValues
        values = ParmValues()
        for parm in self.sop.parms:
            v = getattr(self, 'p_' + parm.name)
            if parm.type in ('vector', 'vector2', 'color'):
                v = tuple(v)
            values[parm.name] = v
        if self.text is not None and 'snippet' in values:
            values['snippet'] = self.text.as_string()
        return values

    def channels(self):
        return {sp.name: sp.value() for sp in self.spare if sp.kind != 'ramp'}

    def edit_deltas(self):
        """Stored point edits of an Edit SOP: (indices, deltas)."""
        import numpy as np
        idx = np.asarray(self.get("edit_idx", []), dtype=np.int64)
        d = np.asarray(self.get("edit_delta", []), dtype=np.float64).reshape(-1, 3)
        return idx, d

    def ramps(self):
        from .ramps import sample
        return {sp.name: sample(sp.ramp_key) for sp in self.spare if sp.kind == 'ramp'}


def make_subnet_tree(owner, name):
    inner = bpy.data.node_groups.new("%s/%s" % (owner.name, name), TREE_ID)
    inner.owner_tree = owner
    inner.use_fake_user = True
    inp = inner.nodes.new(NODE_PREFIX + 'subinput')
    inp.location = (0, 200)
    out = inner.nodes.new(NODE_PREFIX + 'output')
    out.location = (0, 0)
    inner.links.new(inp.outputs[0], out.inputs[0])
    for n in inner.nodes:
        n.display_flag = (n == out)
    return inner


def unique_node_name(tree, base):
    existing = {n.name for n in tree.nodes}
    i = 1
    while "%s%d" % (base, i) in existing:
        i += 1
    return "%s%d" % (base, i)


def update_colors(tree):
    from . import cook
    for node in tree.nodes:
        if not isinstance(node, BlendiniNodeBase):
            continue
        info = cook.node_info(node)
        if info.get('error'):
            col = ERROR_COLOR
        elif node.display_flag:
            col = DISPLAY_COLOR
        else:
            col = CATEGORY_COLORS.get(node.sop.category, (0.3, 0.3, 0.3))
        node.use_custom_color = True
        if tuple(round(c, 3) for c in node.color) != tuple(round(c, 3) for c in col):
            node.color = col


def _make_prop(parm):
    kw = {'name': parm.label, 'update': _request_cook, 'description': parm.description}
    t = parm.type
    if t == 'float':
        for k in ('min', 'max', 'soft_min', 'soft_max'):
            if getattr(parm, k) is not None:
                kw[k] = getattr(parm, k)
        return FloatProperty(default=parm.default, **kw)
    if t == 'int':
        for k in ('min', 'max', 'soft_min', 'soft_max'):
            if getattr(parm, k) is not None:
                kw[k] = int(getattr(parm, k))
        return IntProperty(default=int(parm.default), **kw)
    if t == 'bool':
        return BoolProperty(default=bool(parm.default), **kw)
    if t in ('vector', 'vector2'):
        size = 3 if t == 'vector' else 2
        return FloatVectorProperty(default=tuple(parm.default), size=size, subtype='XYZ' if size == 3 else 'NONE',
                                   **kw)
    if t == 'color':
        return FloatVectorProperty(default=tuple(parm.default), size=3, subtype='COLOR', min=0.0, soft_max=1.0, **kw)
    if t == 'enum':
        return EnumProperty(items=parm.items, default=parm.default, **kw)
    # string, code, object
    return StringProperty(default=parm.default or '', **kw)


NODE_CLASSES = []


def build_node_classes():
    NODE_CLASSES.clear()
    for sop_id, sop in REGISTRY.items():
        annotations = {'p_' + parm.name: _make_prop(parm) for parm in sop.parms}
        cls = type(
            'BLENDINI_NT_' + re.sub(r'\W', '_', sop_id),
            (BlendiniNodeBase, bpy.types.Node),
            {
                'bl_idname': NODE_PREFIX + sop_id,
                'bl_label': sop.label,
                'bl_icon': sop.icon if sop.icon != 'NONE' else 'NODE',
                'bl_description': sop.description or sop.label,
                'sop_id': sop_id,
                '__annotations__': annotations,
            },
        )
        NODE_CLASSES.append(cls)
    return NODE_CLASSES


classes = (
    BlendiniGeometrySocket,
    BlendiniSpareParm,
    BlendiniNodeTree,
)
