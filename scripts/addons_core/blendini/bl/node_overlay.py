# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Houdini style node flags drawn on the node edges in the network editor.

Right edge: display flag (blue) and template flag (pink). Left edge: bypass (yellow).
The node type is written under each node in grey, like Houdini's network view.
"""

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader

from .tree import TREE_ID, BlendiniNodeBase

DISPLAY_COLOR = (0.15, 0.55, 1.0, 0.95)
TEMPLATE_COLOR = (0.95, 0.45, 0.75, 0.95)
BYPASS_COLOR = (1.0, 0.85, 0.1, 0.95)
ERROR_COLOR = (1.0, 0.2, 0.15, 0.95)

_handle = None


def _rect(shader, x0, y0, x1, y1, color):
    batch = batch_for_shader(shader, 'TRIS', {"pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)]})
    shader.uniform_float("color", color)
    batch.draw(shader)


def draw():
    context = bpy.context
    space = context.space_data
    if space is None or getattr(space, 'tree_type', '') != TREE_ID:
        return
    tree = space.edit_tree
    region = context.region
    if tree is None or region is None:
        return
    from . import cook
    v2d = region.view2d
    dpi = context.preferences.system.dpi / 72.0
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    font = 0
    for node in tree.nodes:
        if not isinstance(node, BlendiniNodeBase):
            continue
        loc = getattr(node, 'location_absolute', node.location)
        w, h = node.dimensions
        x0, y0 = v2d.view_to_region(loc.x * dpi, loc.y * dpi, clip=False)
        x1, y1 = v2d.view_to_region(loc.x * dpi + w, loc.y * dpi - h, clip=False)
        if x1 < 0 or x0 > region.width or y0 < 0 or y1 > region.height:
            continue
        bw = max(3.0, min(12.0, (x1 - x0) * 0.06))
        gap = 2.0
        if node.display_flag:
            _rect(shader, x1 + gap, y1, x1 + gap + bw, y0, DISPLAY_COLOR)
        if node.template_flag:
            off = gap + (bw + gap if node.display_flag else 0)
            _rect(shader, x1 + off, y1, x1 + off + bw, y0, TEMPLATE_COLOR)
        if node.mute:
            _rect(shader, x0 - gap - bw, y1, x0 - gap, y0, BYPASS_COLOR)
        elif cook.node_info(node).get('error'):
            _rect(shader, x0 - gap - bw, y1, x0 - gap, y0, ERROR_COLOR)
        # Node type under the node.
        size = max(8, min(14, int((x1 - x0) * 0.07)))
        blf.size(font, size)
        blf.color(font, 0.62, 0.62, 0.62, 1.0)
        blf.position(font, x0 + 2, y1 - size - 3, 0)
        blf.draw(font, node.sop.label)
    gpu.state.blend_set('NONE')


def register():
    global _handle
    register_text()
    if bpy.app.background or _handle is not None:
        return
    _handle = bpy.types.SpaceNodeEditor.draw_handler_add(draw, (), 'WINDOW', 'POST_PIXEL')


def unregister():
    global _handle
    unregister_text()
    if _handle is not None:
        try:
            bpy.types.SpaceNodeEditor.draw_handler_remove(_handle, 'WINDOW')
        except ValueError:
            pass
        _handle = None


# ----------------------------------------------------------------------------
# Text editor status line for wrangle code (errors appear while you type)

_text_handle = None


def _node_for_text(text):
    from .cook import all_trees
    for tree in bpy.data.node_groups:
        if tree.bl_idname != TREE_ID:
            continue
        for node in tree.nodes:
            if getattr(node, 'text', None) == text:
                return node
    return None


def draw_text_status():
    context = bpy.context
    space = context.space_data
    text = getattr(space, 'text', None)
    region = context.region
    if text is None or region is None:
        return
    node = _node_for_text(text)
    if node is None:
        return
    from . import cook
    info = cook.node_info(node)
    err = info.get('error')
    msgs = info.get('messages') or []
    if err:
        line = "%s: %s" % (node.name, err.splitlines()[0])
        bg, fg = (0.45, 0.08, 0.08, 0.9), (1.0, 0.9, 0.9, 1.0)
    elif 'geo' in info:
        g = info['geo']
        line = "%s cooked in %.1f ms: %d points, %d primitives%s" % (
            node.name, info.get('time', 0.0) * 1000, g.num_points, g.num_prims,
            ("   |   " + str(msgs[-1]).strip()) if msgs else "")
        bg, fg = (0.1, 0.25, 0.12, 0.85), (0.85, 1.0, 0.85, 1.0)
    else:
        return
    h = 22
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    _rect(shader, 0, 0, region.width, h, bg)
    gpu.state.blend_set('NONE')
    blf.size(0, 12)
    blf.color(0, *fg)
    blf.position(0, 8, 6, 0)
    blf.draw(0, line[:300])


def register_text():
    global _text_handle
    if bpy.app.background or _text_handle is not None:
        return
    _text_handle = bpy.types.SpaceTextEditor.draw_handler_add(draw_text_status, (), 'WINDOW', 'POST_PIXEL')


def unregister_text():
    global _text_handle
    if _text_handle is not None:
        try:
            bpy.types.SpaceTextEditor.draw_handler_remove(_text_handle, 'WINDOW')
        except ValueError:
            pass
        _text_handle = None
