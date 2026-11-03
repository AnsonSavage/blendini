# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Houdini style viewport handle: translate arrows for the active Transform SOP.

Select the network's output object; when the active node of its network is a Transform SOP,
three arrows appear at the pivot. Dragging them edits the node's Translate parameter.
All callbacks are defensive: if anything is missing the handle simply isn't shown.
"""

import bpy
from mathutils import Matrix, Vector

AXIS_COLORS = ((0.9, 0.25, 0.25), (0.35, 0.85, 0.3), (0.3, 0.45, 1.0))


def active_transform_node(context):
    """(tree, node) when the active object's network has an active Transform SOP."""
    from .ui import tree_for_object
    try:
        tree = tree_for_object(context.active_object)
        if tree is None:
            return None, None
        node = tree.nodes.active
        if node is None or getattr(node, 'sop_id', '') != 'xform' or node.mute:
            return None, None
        return tree, node
    except (ReferenceError, AttributeError):
        return None, None


def axis_matrix(obj_matrix, pivot, axis):
    """Gizmo basis: origin at the (world space) pivot, Z pointing along the network's local axis."""
    m3 = obj_matrix.to_3x3()
    direction = m3 @ Vector([1.0 if i == axis else 0.0 for i in range(3)])
    if direction.length < 1e-9:
        direction = Vector((0.0, 0.0, 1.0))
    rot = direction.normalized().to_track_quat('Z', 'Y').to_matrix().to_4x4()
    return Matrix.Translation(obj_matrix @ Vector(pivot)) @ rot


class BLENDINI_GGT_transform(bpy.types.GizmoGroup):
    bl_idname = "BLENDINI_GGT_transform"
    bl_label = "Blendini Transform Handle"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    @classmethod
    def poll(cls, context):
        return active_transform_node(context)[1] is not None

    def setup(self, context):
        self.arrows = []
        for axis in range(3):
            gz = self.gizmos.new("GIZMO_GT_arrow_3d")
            gz.color = AXIS_COLORS[axis]
            gz.color_highlight = (1.0, 1.0, 0.6)
            gz.alpha = 0.8
            gz.scale_basis = 1.2
            gz.target_set_handler("offset", get=self._getter(axis), set=self._setter(axis))
            self.arrows.append(gz)

    @staticmethod
    def _getter(axis):
        def get():
            _tree, node = active_transform_node(bpy.context)
            try:
                return float(node.p_t[axis]) if node is not None else 0.0
            except (ReferenceError, AttributeError):
                return 0.0
        return get

    @staticmethod
    def _setter(axis):
        def set_value(value):
            _tree, node = active_transform_node(bpy.context)
            if node is None:
                return
            try:
                t = list(node.p_t)
                t[axis] = value
                node.p_t = t
            except (ReferenceError, AttributeError, TypeError):
                pass
        return set_value

    def draw_prepare(self, context):
        tree, node = active_transform_node(context)
        if node is None or tree.target is None:
            for gz in getattr(self, 'arrows', []):
                gz.hide = True
            return
        for axis, gz in enumerate(self.arrows):
            gz.hide = False
            gz.matrix_basis = axis_matrix(tree.target.matrix_world, node.p_p, axis)


classes = (BLENDINI_GGT_transform,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
