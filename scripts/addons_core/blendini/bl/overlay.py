# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Houdini style viewport overlays: point / primitive numbers and normals."""

import bpy
import blf
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from . import cook

MAX_LABELS = 3000
MAX_NORMALS = 50000

_handles = []


def _world_points(obj, P):
    m = np.array(obj.matrix_world)
    return P @ m[:3, :3].T + m[:3, 3]


def _project(region, rv3d, co):
    """Project world points (n, 3) to region pixels. Returns (xy (n, 2), visible (n,))."""
    pm = np.array(rv3d.perspective_matrix)
    h = np.concatenate([co, np.ones((len(co), 1))], axis=1) @ pm.T
    w = h[:, 3]
    vis = w > 1e-6
    w = np.where(vis, w, 1)
    x = (h[:, 0] / w * 0.5 + 0.5) * region.width
    y = (h[:, 1] / w * 0.5 + 0.5) * region.height
    vis &= (x >= 0) & (x <= region.width) & (y >= 0) & (y <= region.height)
    return np.stack([x, y], axis=1), vis


def _displayed(context):
    """(tree, geometry, object) pairs to draw overlays for."""
    out = []
    for tree in cook.all_trees():
        if not (tree.show_point_numbers or tree.show_prim_numbers or tree.show_normals):
            continue
        obj = tree.target
        if obj is None or not obj.visible_get():
            continue
        node = cook.display_node(tree)
        if node is None:
            continue
        geo = cook.node_geometry(node)
        if geo is None:
            continue
        out.append((tree, geo, obj))
    return out


def draw_text():
    context = bpy.context
    region = context.region
    rv3d = context.region_data
    if region is None or rv3d is None:
        return
    font = 0
    blf.size(font, 11)
    for tree, geo, obj in _displayed(context):
        if tree.show_point_numbers and geo.num_points:
            co = _world_points(obj, geo.P)
            xy, vis = _project(region, rv3d, co)
            idx = np.nonzero(vis)[0][:MAX_LABELS]
            blf.color(font, 0.95, 0.95, 0.55, 1.0)
            for i in idx.tolist():
                blf.position(font, xy[i, 0] + 3, xy[i, 1] + 3, 0)
                blf.draw(font, str(i))
        if tree.show_prim_numbers and geo.num_prims:
            co = _world_points(obj, geo.prim_centers())
            xy, vis = _project(region, rv3d, co)
            idx = np.nonzero(vis)[0][:MAX_LABELS]
            blf.color(font, 0.55, 0.85, 1.0, 1.0)
            for i in idx.tolist():
                blf.position(font, xy[i, 0], xy[i, 1], 0)
                blf.draw(font, str(i))


def _draw_templates():
    shader = None
    for tree in cook.all_trees():
        obj = tree.target
        if obj is None or not obj.visible_get():
            continue
        for node in tree.nodes:
            if not getattr(node, 'template_flag', False):
                continue
            geo = cook.node_geometry(node)
            if geo is None or geo.num_points == 0:
                continue
            E = geo.edges()[:MAX_NORMALS * 2]
            if len(E):
                co = _world_points(obj, geo.P)
                coords = co[E.ravel()].astype(np.float32)
                prim = 'LINES'
            else:
                coords = _world_points(obj, geo.P[:MAX_NORMALS]).astype(np.float32)
                prim = 'POINTS'
            if shader is None:
                shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            batch = batch_for_shader(shader, prim, {"pos": coords})
            shader.uniform_float("color", (0.85, 0.55, 0.75, 0.6))
            gpu.state.blend_set('ALPHA')
            gpu.state.point_size_set(3.0)
            batch.draw(shader)
            gpu.state.blend_set('NONE')


def draw_lines():
    context = bpy.context
    _draw_templates()
    for tree, geo, obj in _displayed(context):
        if not tree.show_normals:
            continue
        pa = geo.attribs['point']
        if 'N' in pa:
            base, N = geo.P, pa['N']
        elif 'N' in geo.attribs['vertex']:
            base, N = geo.P[geo.vtx_point], geo.attribs['vertex']['N']
        elif geo.num_prims:
            base, N = geo.prim_centers(), geo.prim_normals()
        else:
            continue
        base = base[:MAX_NORMALS]
        N = N[:MAX_NORMALS]
        lo, hi = geo.bbox()
        scale = tree.normal_scale * max(float(np.linalg.norm(hi - lo)), 1e-6) * 0.05
        a = _world_points(obj, base)
        b = _world_points(obj, base + N * scale)
        coords = np.empty((len(a) * 2, 3), dtype=np.float32)
        coords[0::2] = a
        coords[1::2] = b
        shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        batch = batch_for_shader(shader, 'LINES', {"pos": coords})
        shader.uniform_float("color", (0.3, 0.75, 1.0, 1.0))
        gpu.state.depth_test_set('LESS_EQUAL')
        batch.draw(shader)
        gpu.state.depth_test_set('NONE')


def register():
    if bpy.app.background:
        return
    _handles.append((bpy.types.SpaceView3D.draw_handler_add(draw_lines, (), 'WINDOW', 'POST_VIEW'), 'WINDOW'))
    _handles.append((bpy.types.SpaceView3D.draw_handler_add(draw_text, (), 'WINDOW', 'POST_PIXEL'), 'WINDOW'))


def unregister():
    for h, region in _handles:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(h, region)
        except ValueError:
            pass
    _handles.clear()
