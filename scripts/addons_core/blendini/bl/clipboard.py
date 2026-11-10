# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Copy / paste SOP nodes as JSON text (via the system clipboard).

Schema (``format`` "blendini.nodes", ``version`` 1)::

    {"format": "blendini.nodes", "version": 1,
     "nodes": [{
        "type": "pointwrangle",              # SOP id (bl_idname is NODE_PREFIX + type)
        "name": "pointwrangle1", "label": "", "location": [x, y], "width": 160,
        "flags": {"display": true, "template": false, "mute": false},
        "parms": {"snippet": "...", "class": "point", ...},   # every p_* parameter (vectors as lists);
                                                              # the snippet holds the linked Text's code
        "spare": [{"name": "amp", "kind": "float", "value": 0.4, "soft_max": 1.0,
                   "ramp": [[x, y, "AUTO"], ...]}],          # "ramp" only for kind "ramp"
        "props": {"edit_idx": [...], "edit_delta": [...], "paint_values": [...]},  # custom props
        "subnet": {"nodes": [...], "links": [...]},          # subnet: its inner network, recursively
        "hda": "TreeName",                                   # subnet that is a digital asset instance
        "vop_tree": "TreeName"}],                            # Attribute / Volume VOP network name
     "links": [[from_node, from_socket, to_node, to_socket], ...]}   # only links between copied nodes,
                                                                     # in multi-input order

VOP networks are not serialized: pasting into the same .blend duplicates the named VOP network,
otherwise the pasted VOP node gets a default network. Digital asset instances reference their
definition by name (and fall back to the stored copy of the network).
"""

import json
import re

import bpy
from bpy.props import FloatVectorProperty

from .tree import TREE_ID, NODE_PREFIX, BlendiniNodeBase, unique_node_name
from ..sops import REGISTRY
from ..sops.base import VOP_SOPS

FORMAT = "blendini.nodes"
VERSION = 1


# ----------------------------------------------------------------------------
# Serialization

def _jsonable(value):
    if hasattr(value, 'to_list'):
        value = value.to_list()
    elif hasattr(value, 'to_dict'):
        value = value.to_dict()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    try:
        return [_jsonable(v) for v in value]
    except TypeError:
        return str(value)


def _ramp_points(key):
    from .ramps import ramp_node
    node = ramp_node(key)
    if node is None:
        return []
    curve = node.mapping.curves[0]
    return [[p.location[0], p.location[1], p.handle_type] for p in curve.points]


def _node_to_dict(node):
    sop = REGISTRY[node.sop_id]
    parms = {}
    for parm in sop.parms:
        v = getattr(node, 'p_' + parm.name, None)
        parms[parm.name] = _jsonable(v)
    if node.text is not None and 'snippet' in parms:
        parms['snippet'] = node.text.as_string()
    spare = []
    for sp in node.spare:
        entry = {'name': sp.name, 'kind': sp.kind, 'value': _jsonable(sp.value()), 'soft_max': sp.soft_max}
        if sp.kind == 'ramp':
            entry['ramp'] = _ramp_points(sp.ramp_key)
        spare.append(entry)
    d = {
        'type': node.sop_id,
        'name': node.name,
        'label': node.label,
        'location': [node.location[0], node.location[1]],
        'width': node.width,
        'flags': {'display': bool(node.display_flag), 'template': bool(node.template_flag),
                  'mute': bool(node.mute)},
        'parms': parms,
        'spare': spare,
    }
    props = {k: _jsonable(node[k]) for k in node.keys() if not k.startswith('_')}
    if props:
        d['props'] = props
    if node.sop_id == 'subnet' and node.subtree is not None:
        sub = node.subtree
        if getattr(sub, 'is_hda', False):
            d['hda'] = sub.name
        d['subnet'] = nodes_to_dict([n for n in sub.nodes if isinstance(n, BlendiniNodeBase)])
    if node.sop_id in VOP_SOPS:
        from .tree import node_vop_tree
        vt = node_vop_tree(node)
        if vt is not None:
            d['vop_tree'] = vt.name
    return d


def nodes_to_dict(nodes):
    """Serialize SOP nodes (and the links between them) to a JSON-ready dict."""
    nodes = [n for n in nodes if isinstance(n, BlendiniNodeBase) and n.sop_id in REGISTRY]
    names = {n.name for n in nodes}
    links = []
    if nodes:
        tree = nodes[0].id_data
        ordered = sorted(tree.links, key=lambda l: (l.to_node.name, _socket_index(l.to_node.inputs, l.to_socket),
                                                    getattr(l, 'multi_input_sort_id', 0)))
        for link in ordered:
            if link.from_node.name in names and link.to_node.name in names and link.is_valid:
                links.append([link.from_node.name, _socket_index(link.from_node.outputs, link.from_socket),
                              link.to_node.name, _socket_index(link.to_node.inputs, link.to_socket)])
    return {'format': FORMAT, 'version': VERSION, 'nodes': [_node_to_dict(n) for n in nodes], 'links': links}


def _socket_index(sockets, socket):
    for i, s in enumerate(sockets):
        if s == socket:
            return i
    return 0


# ----------------------------------------------------------------------------
# Deserialization

def _set_ramp(key, points):
    from .ramps import ramp_node
    node = ramp_node(key, create=True)
    mapping = node.mapping
    curve = mapping.curves[0]
    pts = [p for p in points if len(p) >= 2]
    if len(pts) < 2:
        return
    while len(curve.points) > 2:
        curve.points.remove(curve.points[-1])
    for i, p in enumerate(pts):
        cp = curve.points[i] if i < 2 else curve.points.new(float(p[0]), float(p[1]))
        cp.location = (float(p[0]), float(p[1]))
        if len(p) > 2:
            try:
                cp.handle_type = p[2]
            except (TypeError, ValueError):
                pass
    mapping.update()


def _set_spare(node, entry):
    sp = node.spare.add()
    sp.name = str(entry.get('name', 'parm'))
    kind = entry.get('kind', 'float')
    try:
        sp.kind = kind
    except TypeError:
        sp.kind = 'float'
        kind = 'float'
    if 'soft_max' in entry:
        sp.soft_max = float(entry['soft_max'])
    v = entry.get('value')
    try:
        if kind == 'float':
            sp.f = float(v)
        elif kind == 'int':
            sp.i = int(v)
        elif kind == 'vector':
            sp.v = tuple(v)
        elif kind == 'color':
            sp.c = tuple(v)
        elif kind == 'string':
            sp.s = str(v)
    except (TypeError, ValueError):
        pass
    if kind == 'ramp':
        from .ramps import new_key
        sp.ramp_key = new_key()
        _set_ramp(sp.ramp_key, entry.get('ramp') or [])


def _free_name(tree, name, node):
    other = tree.nodes.get(name) if name else node
    if other is None or other == node:
        return name
    base = re.sub(r'\d+$', '', name or '') or 'node'
    return unique_node_name(tree, base)


def _make_node(tree, d, offset):
    sop_id = d.get('type')
    if sop_id not in REGISTRY:
        return None
    node = tree.nodes.new(NODE_PREFIX + sop_id)
    node.name = _free_name(tree, d.get('name', sop_id), node)
    node.label = d.get('label', '') or ''
    loc = d.get('location', (0, 0))
    node.location = (loc[0] + offset[0], loc[1] + offset[1])
    if d.get('width'):
        node.width = d['width']
    sop = REGISTRY[sop_id]
    parms = d.get('parms', {})
    for parm in sop.parms:
        if parm.name not in parms or parm.name == 'voptree':
            continue
        value = parms[parm.name]
        try:
            setattr(node, 'p_' + parm.name, tuple(value) if isinstance(value, list) else value)
        except (TypeError, ValueError, AttributeError) as ex:
            print("Blendini paste: %s.%s not set: %s" % (node.name, parm.name, ex))
    for entry in d.get('spare', ()):
        _set_spare(node, entry)
    for k, v in (d.get('props') or {}).items():
        try:
            node[k] = v
        except (TypeError, ValueError):
            pass
    if sop_id == 'subnet' and node.subtree is not None:
        auto = node.subtree
        hda = bpy.data.node_groups.get(d.get('hda') or '')
        if hda is not None and hda.bl_idname == TREE_ID and hda != tree:
            node.subtree = hda
            bpy.data.node_groups.remove(auto)
        elif d.get('subnet') is not None:
            for n in list(auto.nodes):
                auto.nodes.remove(n)
            dict_to_nodes(auto, d['subnet'], (0.0, 0.0))
    if sop_id in VOP_SOPS:
        src = bpy.data.node_groups.get(d.get('vop_tree') or '')
        if src is not None and src.bl_idname == 'BlendiniVOPTree':
            fresh = node.vop_tree
            vt = src.copy()
            node.vop_tree = vt
            node.p_voptree = vt.name
            if fresh is not None and fresh != src:
                bpy.data.node_groups.remove(fresh)
    return node


def dict_to_nodes(tree, data, offset=(0.0, 0.0)):
    """Create the nodes of a nodes_to_dict() dict in tree. Returns the new nodes."""
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict) or data.get('format') != FORMAT:
        raise ValueError("not Blendini node data")
    if int(data.get('version', 1)) > VERSION:
        raise ValueError("node data was written by a newer Blendini")
    had_display = any(getattr(n, 'display_flag', False) for n in tree.nodes)
    created = {}
    for d in data.get('nodes', ()):
        node = _make_node(tree, d, offset)
        if node is not None:
            created[d.get('name')] = (node, d)
    for a, ai, b, bi in data.get('links', ()):
        if a in created and b in created:
            na, nb = created[a][0], created[b][0]
            if ai < len(na.outputs) and bi < len(nb.inputs):
                tree.links.new(na.outputs[ai], nb.inputs[bi])
    for node, d in created.values():
        flags = d.get('flags', {})
        node.display_flag = bool(flags.get('display')) and not had_display
        node.template_flag = bool(flags.get('template'))
        node.mute = bool(flags.get('mute'))
    nodes = [v[0] for v in created.values()]
    if nodes and not any(getattr(n, 'display_flag', False) for n in tree.nodes):
        nodes[-1].display_flag = True
    from .tree import update_colors
    update_colors(tree)
    return nodes


# ----------------------------------------------------------------------------
# Operators

def _edit_tree(context):
    space = getattr(context, 'space_data', None)
    tree = getattr(space, 'edit_tree', None)
    if tree is None or getattr(tree, 'bl_idname', '') != TREE_ID:
        return None
    return tree


class BLENDINI_OT_copy_nodes_text(bpy.types.Operator):
    """Copy the selected nodes (parameters, spare parameters, code and wiring) to the clipboard as text"""
    bl_idname = "blendini.copy_nodes_text"
    bl_label = "Copy Nodes as Text"

    @classmethod
    def poll(cls, context):
        return _edit_tree(context) is not None

    def execute(self, context):
        tree = _edit_tree(context)
        nodes = [n for n in tree.nodes if n.select and isinstance(n, BlendiniNodeBase)]
        if not nodes:
            self.report({'WARNING'}, "No nodes selected")
            return {'CANCELLED'}
        context.window_manager.clipboard = json.dumps(nodes_to_dict(nodes), indent=1)
        self.report({'INFO'}, "Copied %d node(s) as text" % len(nodes))
        return {'FINISHED'}


class BLENDINI_OT_paste_nodes_text(bpy.types.Operator):
    """Paste nodes copied as text (from Copy Nodes as Text) into this network"""
    bl_idname = "blendini.paste_nodes_text"
    bl_label = "Paste Nodes from Text"
    bl_options = {'REGISTER', 'UNDO'}

    offset: FloatVectorProperty(name="Offset", size=2, default=(40.0, -40.0))

    @classmethod
    def poll(cls, context):
        return _edit_tree(context) is not None

    def execute(self, context):
        tree = _edit_tree(context)
        try:
            data = json.loads(context.window_manager.clipboard)
            nodes = dict_to_nodes(tree, data, tuple(self.offset))
        except (ValueError, TypeError, KeyError) as ex:
            self.report({'ERROR'}, "Clipboard has no Blendini nodes: %s" % ex)
            return {'CANCELLED'}
        for n in tree.nodes:
            n.select = False
        for n in nodes:
            n.select = True
        from . import cook
        cook.request_cook(tree)
        self.report({'INFO'}, "Pasted %d node(s)" % len(nodes))
        return {'FINISHED'}


classes = (BLENDINI_OT_copy_nodes_text, BLENDINI_OT_paste_nodes_text)


def draw_menu(self, context):
    if _edit_tree(context) is None:
        return
    layout = self.layout
    layout.separator()
    layout.operator(BLENDINI_OT_copy_nodes_text.bl_idname, icon='COPYDOWN')
    layout.operator(BLENDINI_OT_paste_nodes_text.bl_idname, icon='PASTEDOWN')


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    menu = getattr(bpy.types, 'NODE_MT_node', None)
    if menu is not None:
        menu.append(draw_menu)


def unregister():
    menu = getattr(bpy.types, 'NODE_MT_node', None)
    if menu is not None:
        menu.remove(draw_menu)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
