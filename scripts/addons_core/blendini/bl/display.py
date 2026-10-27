# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Viewport display helpers: show loose points like Houdini does (Blender hides them in object mode)."""

import bpy

GROUP_NAME = "Blendini Point Display"
MOD_NAME = "Blendini Points"


def _point_display_group():
    ng = bpy.data.node_groups.get(GROUP_NAME)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(GROUP_NAME, 'GeometryNodeTree')
    ng.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    nodes, links = ng.nodes, ng.links
    gin = nodes.new('NodeGroupInput')
    gin.location = (-600, 0)
    gout = nodes.new('NodeGroupOutput')
    gout.location = (400, 0)
    neighbours = nodes.new('GeometryNodeInputMeshVertexNeighbors')
    neighbours.location = (-600, -200)
    is_loose = nodes.new('FunctionNodeBooleanMath')
    is_loose.operation = 'NOT'
    is_loose.location = (-400, -200)
    to_points = nodes.new('GeometryNodeMeshToPoints')
    to_points.mode = 'VERTICES'
    to_points.location = (-200, -100)
    to_points.inputs['Radius'].default_value = 0.015
    join = nodes.new('GeometryNodeJoinGeometry')
    join.location = (100, 0)
    links.new(neighbours.outputs['Vertex Count'], is_loose.inputs[0])
    links.new(gin.outputs[0], to_points.inputs['Mesh'])
    links.new(is_loose.outputs[0], to_points.inputs['Selection'])
    links.new(gin.outputs[0], join.inputs[0])
    links.new(to_points.outputs[0], join.inputs[0])
    links.new(join.outputs[0], gout.inputs[0])
    return ng


def ensure_point_display(obj):
    if obj is None or obj.type != 'MESH':
        return
    if obj.modifiers.get(MOD_NAME) is not None:
        return
    try:
        mod = obj.modifiers.new(MOD_NAME, 'NODES')
        mod.node_group = _point_display_group()
        mod.show_in_editmode = False
    except Exception as ex:
        print("Blendini: could not add point display modifier:", ex)


def setup_target(obj):
    """Prepare an output object: Houdini is Y-up, Blender is Z-up, so the object is rotated
    90 degrees around X. VEX code can keep using @P.y as 'up'."""
    import math
    obj.rotation_euler = (math.pi / 2, 0.0, 0.0)
    ensure_point_display(obj)
