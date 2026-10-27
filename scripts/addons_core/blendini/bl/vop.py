# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
VOPs: visual VEX. An Attribute VOP SOP owns a VOP network; the network is turned into VEX
code (you can read it in the parameter pane) and runs through the same SIMD interpreter as
the wrangles.
"""

import bpy
from bpy.props import EnumProperty, FloatProperty, FloatVectorProperty, StringProperty

VOP_TREE_ID = 'BlendiniVOPTree'
VOP_PREFIX = 'BlendiniVOP_'
FLOAT_SOCKET = 'BlendiniVOPFloatSocket'
VECTOR_SOCKET = 'BlendiniVOPVectorSocket'


def _request_owner_cook(self, context=None):
    tree = self.id_data if not isinstance(self, bpy.types.NodeTree) else self
    request_owner_cook(tree)


def request_owner_cook(vop_tree):
    from . import cook
    from .tree import TREE_ID
    for tree in bpy.data.node_groups:
        if tree.bl_idname != TREE_ID:
            continue
        for node in tree.nodes:
            if getattr(node, 'sop_id', '') == 'attribvop' and node.p_voptree == vop_tree.name:
                cook.request_cook(tree)
                break


class BlendiniVOPFloatSocket(bpy.types.NodeSocket):
    bl_idname = FLOAT_SOCKET
    bl_label = "Float"
    default_value: FloatProperty(update=_request_owner_cook)

    def draw(self, context, layout, node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            layout.prop(self, "default_value", text=text)

    @classmethod
    def draw_color_simple(cls):
        return (0.55, 0.85, 0.55, 1.0)


class BlendiniVOPVectorSocket(bpy.types.NodeSocket):
    bl_idname = VECTOR_SOCKET
    bl_label = "Vector"
    default_value: FloatVectorProperty(size=3, update=_request_owner_cook)

    def draw(self, context, layout, node, text):
        if self.is_output or self.is_linked:
            layout.label(text=text)
        else:
            col = layout.column(align=True)
            col.label(text=text)
            col.prop(self, "default_value", text="")

    @classmethod
    def draw_color_simple(cls):
        return (0.95, 0.85, 0.35, 1.0)


class BlendiniVOPTree(bpy.types.NodeTree):
    """VOP network (visual VEX) used by Attribute VOP SOPs"""
    bl_idname = VOP_TREE_ID
    bl_label = "Blendini VOP Network"
    bl_icon = 'NODE_MATERIAL'

    def update(self):
        request_owner_cook(self)


F, V = 'float', 'vector'

# id: (label, category, inputs [(name, type, default)], outputs [(name, type)], expression template)
VOPS = {
    'add': ("Add", 'Math', [('A', V, 0.0), ('B', V, 0.0)], [('Sum', V)], "{A} + {B}"),
    'subtract': ("Subtract", 'Math', [('A', V, 0.0), ('B', V, 0.0)], [('Difference', V)], "{A} - {B}"),
    'multiply': ("Multiply", 'Math', [('A', V, 1.0), ('B', V, 1.0)], [('Product', V)], "{A} * {B}"),
    'divide': ("Divide", 'Math', [('A', V, 1.0), ('B', V, 1.0)], [('Quotient', V)], "{A} / {B}"),
    'scale': ("Multiply by Float", 'Math', [('Vector', V, 1.0), ('Scale', F, 1.0)], [('Result', V)], "{Vector} * {Scale}"),
    'addfloat': ("Add Float", 'Math', [('A', F, 0.0), ('B', F, 0.0)], [('Sum', F)], "{A} + {B}"),
    'mulfloat': ("Multiply Float", 'Math', [('A', F, 1.0), ('B', F, 1.0)], [('Product', F)], "{A} * {B}"),
    'sin': ("Sine", 'Math', [('Value', F, 0.0)], [('Result', F)], "sin({Value})"),
    'cos': ("Cosine", 'Math', [('Value', F, 0.0)], [('Result', F)], "cos({Value})"),
    'abs': ("Absolute", 'Math', [('Value', F, 0.0)], [('Result', F)], "abs({Value})"),
    'pow': ("Power", 'Math', [('Base', F, 1.0), ('Exponent', F, 2.0)], [('Result', F)], "pow({Base}, {Exponent})"),
    'fit': ("Fit Range", 'Math', [('Value', F, 0.5), ('Src Min', F, 0.0), ('Src Max', F, 1.0), ('Dest Min', F, 0.0),
                                    ('Dest Max', F, 1.0)], [('Shifted', F)],
            "fit({Value}, {Src_Min}, {Src_Max}, {Dest_Min}, {Dest_Max})"),
    'clamp': ("Clamp", 'Math', [('Value', F, 0.0), ('Min', F, 0.0), ('Max', F, 1.0)], [('Clamped', F)],
              "clamp({Value}, {Min}, {Max})"),
    'mix': ("Mix", 'Math', [('A', V, 0.0), ('B', V, 1.0), ('Bias', F, 0.5)], [('Blend', V)], "lerp({A}, {B}, {Bias})"),
    'length': ("Length", 'Vector', [('Vector', V, 0.0)], [('Length', F)], "length({Vector})"),
    'normalize': ("Normalize", 'Vector', [('Vector', V, 0.0)], [('Normalized', V)], "normalize({Vector})"),
    'dot': ("Dot Product", 'Vector', [('A', V, 0.0), ('B', V, 0.0)], [('Dot', F)], "dot({A}, {B})"),
    'cross': ("Cross Product", 'Vector', [('A', V, 0.0), ('B', V, 0.0)], [('Cross', V)], "cross({A}, {B})"),
    'compose': ("Float to Vector", 'Vector', [('X', F, 0.0), ('Y', F, 0.0), ('Z', F, 0.0)], [('Vector', V)],
                "set({X}, {Y}, {Z})"),
    'getx': ("Vector to Float X", 'Vector', [('Vector', V, 0.0)], [('X', F)], "({Vector}).x"),
    'gety': ("Vector to Float Y", 'Vector', [('Vector', V, 0.0)], [('Y', F)], "({Vector}).y"),
    'getz': ("Vector to Float Z", 'Vector', [('Vector', V, 0.0)], [('Z', F)], "({Vector}).z"),
    'noise': ("Noise", 'Noise', [('Position', V, 0.0), ('Frequency', F, 1.0), ('Offset', V, 0.0), ('Amplitude', F, 1.0)],
              [('Noise', F)], "(noise({Position} * {Frequency} + {Offset}) - 0.5) * {Amplitude}"),
    'vnoise': ("Vector Noise", 'Noise', [('Position', V, 0.0), ('Frequency', F, 1.0), ('Offset', V, 0.0),
                                         ('Amplitude', F, 1.0)], [('Noise', V)],
               "(vector(noise({Position} * {Frequency} + {Offset})) - 0.5) * {Amplitude}"),
    'curlnoise': ("Curl Noise", 'Noise', [('Position', V, 0.0), ('Frequency', F, 1.0), ('Amplitude', F, 1.0)],
                  [('Noise', V)], "curlnoise({Position} * {Frequency}) * {Amplitude}"),
    'random': ("Random", 'Noise', [('Seed', F, 0.0)], [('Random', F)], "rand({Seed})"),
    'hsv': ("HSV to RGB", 'Color', [('HSV', V, 0.0)], [('RGB', V)], "hsvtorgb({HSV})"),
    'constant': ("Constant", 'Input', [('Value', F, 0.0)], [('Value', F)], "{Value}"),
    'vconstant': ("Vector Constant", 'Input', [('Value', V, 0.0)], [('Value', V)], "{Value}"),
}

CATEGORIES = ['Input', 'Math', 'Vector', 'Noise', 'Color', 'Output']


def _vex_literal(sock):
    if sock.bl_idname == VECTOR_SOCKET:
        v = sock.default_value
        return "set(%r, %r, %r)" % (float(v[0]), float(v[1]), float(v[2]))
    return repr(float(sock.default_value))


class BlendiniVOPNodeBase:
    vop_id = ''

    @classmethod
    def poll(cls, ntree):
        return ntree.bl_idname == VOP_TREE_ID


def _make_simple_vop(vop_id, spec):
    label, category, inputs, outputs, template = spec

    def init(self, context):
        for name, typ, default in inputs:
            s = self.inputs.new(VECTOR_SOCKET if typ == V else FLOAT_SOCKET, name)
            if typ == V:
                s.default_value = (default, default, default)
            else:
                s.default_value = default
        for name, typ in outputs:
            self.outputs.new(VECTOR_SOCKET if typ == V else FLOAT_SOCKET, name)

    return type('BLENDINI_VOP_' + vop_id, (BlendiniVOPNodeBase, bpy.types.Node), {
        'bl_idname': VOP_PREFIX + vop_id, 'bl_label': label, 'vop_id': vop_id, 'vop_category': category,
        'template': template, 'init': init,
    })


class BlendiniVOPGlobals(BlendiniVOPNodeBase, bpy.types.Node):
    """Geometry VOP Global Parameters"""
    bl_idname = VOP_PREFIX + 'globals'
    bl_label = "Geometry VOP Global Parameters"
    vop_id = 'globals'
    vop_category = 'Input'
    OUTPUTS = [('P', V, '@P'), ('N', V, '@N'), ('Cd', V, '@Cd'), ('v', V, 'v@v'), ('ptnum', F, 'float(@ptnum)'),
               ('numpt', F, 'float(@numpt)'), ('Time', F, '@Time'), ('Frame', F, '@Frame')]

    def init(self, context):
        for name, typ, _expr in self.OUTPUTS:
            self.outputs.new(VECTOR_SOCKET if typ == V else FLOAT_SOCKET, name)


class BlendiniVOPBind(BlendiniVOPNodeBase, bpy.types.Node):
    """Read an attribute"""
    bl_idname = VOP_PREFIX + 'bind'
    bl_label = "Bind"
    vop_id = 'bind'
    vop_category = 'Input'
    attrib: StringProperty(name="Name", default="foo", update=_request_owner_cook)
    attrtype: EnumProperty(name="Type", items=[('float', "Float", ""), ('vector', "Vector", "")], update=_request_owner_cook)

    def init(self, context):
        self.outputs.new(FLOAT_SOCKET, "Value")
        self.outputs.new(VECTOR_SOCKET, "Vector")

    def draw_buttons(self, context, layout):
        layout.prop(self, "attrib")
        layout.prop(self, "attrtype")


class BlendiniVOPParm(BlendiniVOPNodeBase, bpy.types.Node):
    """A channel (spare parameter on the Attribute VOP)"""
    bl_idname = VOP_PREFIX + 'parm'
    bl_label = "Parameter"
    vop_id = 'parm'
    vop_category = 'Input'
    parmname: StringProperty(name="Name", default="amp", update=_request_owner_cook)

    def init(self, context):
        self.outputs.new(FLOAT_SOCKET, "Value")

    def draw_buttons(self, context, layout):
        layout.prop(self, "parmname")


class BlendiniVOPOutput(BlendiniVOPNodeBase, bpy.types.Node):
    """Geometry VOP Output: writes P, N, Cd and v"""
    bl_idname = VOP_PREFIX + 'output'
    bl_label = "Geometry VOP Output"
    vop_id = 'output'
    vop_category = 'Output'
    TARGETS = [('P', V, '@P'), ('N', V, '@N'), ('Cd', V, '@Cd'), ('v', V, 'v@v')]

    def init(self, context):
        for name, typ, _target in self.TARGETS:
            self.inputs.new(VECTOR_SOCKET, name)


class BlendiniVOPBindExport(BlendiniVOPNodeBase, bpy.types.Node):
    """Write an attribute"""
    bl_idname = VOP_PREFIX + 'bindexport'
    bl_label = "Bind Export"
    vop_id = 'bindexport'
    vop_category = 'Output'
    attrib: StringProperty(name="Name", default="foo", update=_request_owner_cook)
    attrtype: EnumProperty(name="Type", items=[('float', "Float", ""), ('vector', "Vector", "")], update=_request_owner_cook)

    def init(self, context):
        self.inputs.new(FLOAT_SOCKET, "Value")
        self.inputs.new(VECTOR_SOCKET, "Vector")

    def draw_buttons(self, context, layout):
        layout.prop(self, "attrib")
        layout.prop(self, "attrtype")


# ----------------------------------------------------------------------------
# Code generation


def generate_vex(tree):
    """Turn a VOP network into VEX code."""
    lines = ["// Generated from VOP network '%s'" % tree.name]
    names = {}
    done = set()
    visiting = set()

    def var_name(node, out_index):
        base = ''.join(ch if ch.isalnum() else '_' for ch in node.name)
        return "%s_%s" % (base, ''.join(ch if ch.isalnum() else '_' for ch in node.outputs[out_index].name))

    def input_expr(sock):
        if not sock.is_linked:
            return _vex_literal(sock)
        link = sock.links[0]
        while link.from_node.bl_idname == 'NodeReroute':
            if not link.from_node.inputs[0].is_linked:
                return _vex_literal(sock)
            link = link.from_node.inputs[0].links[0]
        src = link.from_node
        emit(src)
        idx = list(src.outputs).index(link.from_socket)
        return names[(src.name, idx)]

    def emit(node):
        if node.name in done:
            return
        if node.name in visiting:
            raise ValueError("Cycle in VOP network at %s" % node.name)
        visiting.add(node.name)
        vid = getattr(node, 'vop_id', '')
        if vid == 'globals':
            for i, (_n, _t, expr) in enumerate(BlendiniVOPGlobals.OUTPUTS):
                names[(node.name, i)] = expr
        elif vid == 'bind':
            prefix = 'f' if node.attrtype == 'float' else 'v'
            names[(node.name, 0)] = "float(%s@%s)" % (prefix, node.attrib)
            names[(node.name, 1)] = "vector(%s@%s)" % (prefix, node.attrib)
        elif vid == 'parm':
            names[(node.name, 0)] = 'ch("%s")' % node.parmname
        elif vid in VOPS:
            args = {}
            for sock in node.inputs:
                args[sock.name.replace(' ', '_')] = input_expr(sock)
            expr = node.template.format(**args)
            for i, (oname, otype) in enumerate(VOPS[vid][3]):
                var = var_name(node, i)
                lines.append("%s %s = %s;" % (otype, var, expr))
                names[(node.name, i)] = var
        visiting.discard(node.name)
        done.add(node.name)

    outputs = [n for n in tree.nodes if getattr(n, 'vop_id', '') in ('output', 'bindexport')]
    for node in outputs:
        if node.vop_id == 'output':
            for sock, (_n, _t, target) in zip(node.inputs, BlendiniVOPOutput.TARGETS):
                if sock.is_linked:
                    lines.append("%s = %s;" % (target, input_expr(sock)))
        else:
            if node.attrtype == 'float' and node.inputs[0].is_linked:
                lines.append("f@%s = %s;" % (node.attrib, input_expr(node.inputs[0])))
            elif node.attrtype == 'vector' and node.inputs[1].is_linked:
                lines.append("v@%s = %s;" % (node.attrib, input_expr(node.inputs[1])))
    return "\n".join(lines) + "\n"


def make_default_vop_tree(name):
    vt = bpy.data.node_groups.new(name, VOP_TREE_ID)
    vt.use_fake_user = True
    g = vt.nodes.new(VOP_PREFIX + 'globals')
    g.location = (-300, 0)
    o = vt.nodes.new(VOP_PREFIX + 'output')
    o.location = (300, 0)
    vt.links.new(g.outputs['P'], o.inputs['P'])
    return vt


def vop_node_categories():
    from nodeitems_utils import NodeCategory, NodeItem

    class VOPCategory(NodeCategory):
        @classmethod
        def poll(cls, context):
            return context.space_data.tree_type == VOP_TREE_ID

    items = {c: [] for c in CATEGORIES}
    for cls in all_vop_node_classes():
        items.setdefault(cls.vop_category, []).append(NodeItem(cls.bl_idname))
    return [VOPCategory("BLENDINI_VOP_" + c.upper(), c, items=v) for c, v in items.items() if v]


_SIMPLE = []


def all_vop_node_classes():
    if not _SIMPLE:
        _SIMPLE.extend(_make_simple_vop(k, v) for k, v in VOPS.items())
    return [BlendiniVOPGlobals, BlendiniVOPBind, BlendiniVOPParm, BlendiniVOPOutput, BlendiniVOPBindExport] + _SIMPLE


classes = (BlendiniVOPFloatSocket, BlendiniVOPVectorSocket, BlendiniVOPTree)


def register():
    import nodeitems_utils
    for cls in classes:
        bpy.utils.register_class(cls)
    for cls in all_vop_node_classes():
        bpy.utils.register_class(cls)
    nodeitems_utils.register_node_categories("BLENDINI_VOP_NODES", vop_node_categories())


def unregister():
    import nodeitems_utils
    try:
        nodeitems_utils.unregister_node_categories("BLENDINI_VOP_NODES")
    except KeyError:
        pass
    for cls in reversed(all_vop_node_classes()):
        bpy.utils.unregister_class(cls)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
