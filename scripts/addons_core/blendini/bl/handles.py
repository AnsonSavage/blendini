# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Houdini style viewport handles for the active SOP.

Select the network's output object. When the active node of its network has a handle (see
``HANDLES``), gizmos appear in the 3D view and edit the node's parameters:

- arrows move a position parameter (Transform translate / pivot, primitive centers, origins)
- dials rotate (Transform rotate)
- boxes resize (Transform scale, Box size, Sphere radius, bounding boxes, lengths)

Handles are described as data, so the binding logic is testable without a viewport. Gizmos
are created once per group (a fixed pool) and re-bound to the active node on every redraw.
All callbacks are defensive: if anything is missing the handle simply isn't shown.
"""

import math

import bpy
from mathutils import Matrix, Vector

AXIS_COLORS = ((0.9, 0.25, 0.25), (0.35, 0.85, 0.3), (0.3, 0.45, 1.0))
AXES = ('X', 'Y', 'Z')

# Handle specs. Each entry: (kind, parm, options)
#   move:   3 arrows; offset = parm[axis]; drawn from options['base'] (a vector parm or origin)
#   rotate: 3 dials (degrees) at options['at']
#   size:   3 boxes; offset = parm[axis] * factor (+ 'extra'), centred at options['at']
#   length: 1 box along the axis in options['axis'] (axis index, or an enum parm name holding X/Y/Z)
#   radius: 1 box along X for a float parm
# 'when': {parm: value(s)} shows the handle only for those parameter values.
HANDLES = {
    'xform': [('move', 't', {'base': 'p'}), ('move', 'p', {'pivot': True}), ('rotate', 'r', {'at': ('p', 't')}),
              ('size', 's', {'at': ('p', 't'), 'factor': 1.0, 'extra': 0.6})],
    'box': [('move', 't', {}), ('size', 'size', {'at': ('t',), 'factor': 0.5})],
    'sphere': [('move', 't', {}), ('size', 'rad', {'at': ('t',), 'factor': 1.0})],
    'tube': [('move', 't', {}), ('radius', 'rad1', {'at': ('t',)}), ('length', 'height', {'at': ('t',), 'axis': 1, 'factor': 0.5})],
    'platonic': [('move', 't', {}), ('radius', 'rad', {'at': ('t',)})],
    'torus': [('move', 't', {})],
    'circle': [('move', 't', {})],
    'grid': [('move', 't', {})],
    'groupcreate': [('move', 't', {'when': {'mode': 'bbox'}}),
                    ('size', 'size', {'at': ('t',), 'factor': 0.5, 'when': {'mode': 'bbox'}})],
    'delete': [('move', 't', {'when': {'mode': 'bbox'}}),
               ('size', 'size', {'at': ('t',), 'factor': 0.5, 'when': {'mode': 'bbox'}})],
    'mirror': [('move', 'origin', {})],
    'revolve': [('move', 'origin', {})],
    'line': [('move', 'origin', {}), ('length', 'dist', {'at': ('origin',), 'dir': 'dir'})],
    'bend': [('move', 'origin', {}), ('length', 'length', {'at': ('origin',), 'axis': 'axis'})],
    'softxform': [('move', 't', {})],
    'magnet': [('move', 't', {'when': {'mode': 'translate'}})],
    'volume': [('move', 't', {'when': {'usebounds': False}}),
               ('size', 'size', {'at': ('t',), 'factor': 0.5, 'when': {'usebounds': False}})],
    'pyrosolver': [('move', 't', {}), ('size', 'size', {'at': ('t',), 'factor': 0.5})],
}


def _vec(node, name):
    try:
        return [float(x) for x in getattr(node, 'p_' + name)]
    except (AttributeError, TypeError, ReferenceError):
        return [0.0, 0.0, 0.0]


def _visible(node, opts):
    for parm, values in opts.get('when', {}).items():
        values = values if isinstance(values, (tuple, list)) else (values,)
        if getattr(node, 'p_' + parm, None) not in values:
            return False
    return True


def _at(node, names):
    c = [0.0, 0.0, 0.0]
    for name in names:
        v = _vec(node, name)
        c = [c[i] + v[i] for i in range(3)]
    return c


def _set_component(node, name, axis, value):
    v = list(getattr(node, 'p_' + name))
    v[axis] = value
    setattr(node, 'p_' + name, v)


def bindings(node):
    """Handle bindings for a node: a list of dicts with kind ('arrow' / 'dial' / 'box'), axis,
    origin (network space), direction (network space unit vector), get() and set(value)."""
    out = []
    for kind, parm, opts in HANDLES.get(getattr(node, 'sop_id', ''), ()):
        if not _visible(node, opts) or not hasattr(node, 'p_' + parm):
            continue
        if kind == 'move':
            value = _vec(node, parm)
            base = _vec(node, opts['base']) if 'base' in opts else [0.0, 0.0, 0.0]
            for axis in range(3):
                # The arrow slides along its axis only: its base carries the other two components,
                # so the arrow sits exactly at base + value.
                origin = [base[i] + (value[i] if i != axis else 0.0) for i in range(3)]
                out.append(dict(kind='arrow', axis=axis, origin=origin, direction=_unit(axis),
                                small=bool(opts.get('pivot')),
                                get=lambda n=node, p=parm, a=axis: _vec(n, p)[a],
                                set=lambda v, n=node, p=parm, a=axis: _set_component(n, p, a, v)))
        elif kind == 'rotate':
            for axis in range(3):
                out.append(dict(kind='dial', axis=axis, origin=_at(node, opts['at']), direction=_unit(axis),
                                get=lambda n=node, p=parm, a=axis: math.radians(_vec(n, p)[a]),
                                set=lambda v, n=node, p=parm, a=axis: _set_component(n, p, a, math.degrees(v))))
        elif kind == 'size':
            f = opts.get('factor', 1.0)
            extra = opts.get('extra', 0.0)
            for axis in range(3):
                out.append(dict(kind='box', axis=axis, origin=_at(node, opts['at']), direction=_unit(axis),
                                get=lambda n=node, p=parm, a=axis, f=f, e=extra: _vec(n, p)[a] * f + e,
                                set=lambda v, n=node, p=parm, a=axis, f=f, e=extra:
                                    _set_component(n, p, a, max((v - e) / f, 0.0) if f else 0.0)))
        elif kind in ('length', 'radius'):
            f = opts.get('factor', 1.0)
            if 'dir' in opts:
                d = Vector(_vec(node, opts['dir']))
                direction = list(d.normalized()) if d.length > 1e-9 else [0.0, 1.0, 0.0]
                axis = max(range(3), key=lambda i: abs(direction[i]))
            else:
                axis = opts.get('axis', 0)
                if isinstance(axis, str):
                    axis = AXES.index(str(getattr(node, 'p_' + axis, 'Y')).upper()[:1] or 'Y')
                direction = _unit(axis)
            out.append(dict(kind='box', axis=axis, origin=_at(node, opts['at']), direction=direction,
                            get=lambda n=node, p=parm, f=f: float(getattr(n, 'p_' + p)) * f,
                            set=lambda v, n=node, p=parm, f=f: setattr(n, 'p_' + p, max(v / f, 0.0))))
    return out


def _unit(axis):
    return [1.0 if i == axis else 0.0 for i in range(3)]


def active_handle_node(context):
    """(tree, node) when the active object's network has an active node with handles."""
    from .ui import tree_for_object
    try:
        tree = tree_for_object(context.active_object)
        if tree is None:
            return None, None
        node = tree.nodes.active
        if node is None or getattr(node, 'sop_id', '') not in HANDLES or node.mute:
            return None, None
        return tree, node
    except (ReferenceError, AttributeError):
        return None, None


def active_transform_node(context):
    tree, node = active_handle_node(context)
    if node is None or node.sop_id != 'xform':
        return None, None
    return tree, node


def direction_matrix(obj_matrix, origin, direction):
    """Gizmo basis: origin at the (world space) point, Z pointing along the network space direction."""
    m3 = obj_matrix.to_3x3()
    d = m3 @ Vector(direction)
    if d.length < 1e-9:
        d = Vector((0.0, 0.0, 1.0))
    rot = d.normalized().to_track_quat('Z', 'Y').to_matrix().to_4x4()
    return Matrix.Translation(obj_matrix @ Vector(origin)) @ rot


def axis_matrix(obj_matrix, pivot, axis):
    return direction_matrix(obj_matrix, pivot, _unit(axis))


POOL = {'arrow': 6, 'dial': 3, 'box': 3}


class BLENDINI_GGT_handles(bpy.types.GizmoGroup):
    bl_idname = "BLENDINI_GGT_handles"
    bl_label = "Blendini Handles"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    @classmethod
    def poll(cls, context):
        return active_handle_node(context)[1] is not None

    def setup(self, context):
        self.pool = {}
        self.slots = {}
        for kind, count in POOL.items():
            gizmos = []
            for i in range(count):
                gtype = "GIZMO_GT_dial_3d" if kind == 'dial' else "GIZMO_GT_arrow_3d"
                gz = self.gizmos.new(gtype)
                gz.color_highlight = (1.0, 1.0, 0.6)
                if kind == 'dial':
                    gz.alpha = 0.6
                    gz.scale_basis = 0.8
                    gz.line_width = 2.0
                    try:
                        gz.draw_options = {'ANGLE_VALUE'}
                    except (TypeError, ValueError):
                        pass
                else:
                    gz.alpha = 0.8
                    gz.scale_basis = 1.2 if kind == 'arrow' else 0.9
                    if kind == 'box':
                        try:
                            gz.draw_style = 'BOX'
                            gz.length = 0.0
                        except (TypeError, ValueError, AttributeError):
                            pass
                slot = {'binding': None}
                gz.target_set_handler("offset", get=self._getter(slot), set=self._setter(slot))
                gz.hide = True
                self.slots[gz.as_pointer()] = slot
                gizmos.append((gz, slot))
            self.pool[kind] = gizmos

    @staticmethod
    def _getter(slot):
        def get():
            b = slot['binding']
            try:
                return float(b['get']()) if b is not None else 0.0
            except (ReferenceError, AttributeError, TypeError, ValueError):
                return 0.0
        return get

    @staticmethod
    def _setter(slot):
        def set_value(value):
            b = slot['binding']
            if b is None:
                return
            try:
                b['set'](value)
            except (ReferenceError, AttributeError, TypeError, ValueError):
                pass
        return set_value

    def draw_prepare(self, context):
        tree, node = active_handle_node(context)
        used = {kind: 0 for kind in POOL}
        pool = getattr(self, 'pool', {})
        if node is not None and tree.target is not None:
            mw = tree.target.matrix_world
            for b in bindings(node):
                kind = b['kind']
                gizmos = pool.get(kind, [])
                if used[kind] >= len(gizmos):
                    continue
                gz, slot = gizmos[used[kind]]
                used[kind] += 1
                slot['binding'] = b
                gz.hide = False
                gz.color = AXIS_COLORS[b['axis']]
                if kind == 'arrow':
                    gz.scale_basis = 0.6 if b.get('small') else 1.2
                gz.matrix_basis = direction_matrix(mw, b['origin'], b['direction'])
        for kind, gizmos in pool.items():
            for gz, slot in gizmos[used.get(kind, 0):]:
                gz.hide = True
                slot['binding'] = None


# Kept for scripts written against the first version (only the Transform SOP had handles).
BLENDINI_GGT_transform = BLENDINI_GGT_handles

classes = (BLENDINI_GGT_handles,)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
