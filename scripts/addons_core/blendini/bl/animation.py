# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Evaluate keyframes and drivers on Blendini networks.

Custom node trees are not evaluated by Blender's depsgraph, so their animation is applied
here before cooking. Driver expressions are evaluated with Blendini's VEX interpreter (no
Python from the file is executed): `frame`, `time` and `fps` are available, so expressions
such as `frame * 0.5` or `sin(frame / 10)` work like Houdini's `$F` expressions.
"""

import bpy

_expr_cache = {}


def _fcurves(anim_data):
    action = anim_data.action
    if action is None:
        return []
    try:
        from bpy_extras import anim_utils
        slot = getattr(anim_data, 'action_slot', None)
        bag = anim_utils.action_get_channelbag_for_slot(action, slot)
        if bag is not None:
            return list(bag.fcurves)
    except (ImportError, AttributeError, TypeError):
        pass
    return list(getattr(action, 'fcurves', []))


def _assign(tree, data_path, index, value):
    try:
        prop_owner_path, _, prop = data_path.rpartition('.')
        owner = tree.path_resolve(prop_owner_path) if prop_owner_path else tree
        current = getattr(owner, prop)
        if hasattr(current, '__len__') and not isinstance(current, str):
            if abs(current[index] - value) > 1e-9:
                current[index] = value
        else:
            if isinstance(current, bool):
                value = value >= 0.5
            elif isinstance(current, int):
                value = int(round(value))
            if current != value:
                setattr(owner, prop, value)
    except (ValueError, AttributeError, TypeError, IndexError):
        pass


def eval_driver_expression(expr, frame, fps):
    """Evaluate a driver expression with the VEX interpreter. Returns a float or None."""
    import numpy as np
    from ..vex.wrangle import run_wrangle
    from ..core.geometry import Geometry
    expr = expr.strip().lstrip('#')
    if not expr:
        return None
    code = ("float frame = @Frame; float time = @Time; float fps = %r;\n"
            "f@__value = (%s);" % (float(fps), expr))
    try:
        geo, _msgs = run_wrangle(code, [Geometry()], run_over='detail', frame=frame, time=(frame - 1) / fps, fps=fps)
    except Exception as ex:
        if _expr_cache.get(expr) != str(ex):
            print("Blendini: can't evaluate driver '%s': %s" % (expr, ex))
            _expr_cache[expr] = str(ex)
        return None
    v = geo.attribs['detail'].get('__value')
    return float(np.asarray(v).ravel()[0]) if v is not None else None


def apply_animation(tree, frame, fps):
    ad = tree.animation_data
    if ad is None:
        return False
    changed = False
    for fc in _fcurves(ad):
        if fc.mute:
            continue
        _assign(tree, fc.data_path, fc.array_index, fc.evaluate(frame))
        changed = True
    for fc in ad.drivers:
        if fc.mute or fc.driver is None:
            continue
        drv = fc.driver
        if drv.type == 'SCRIPTED':
            value = eval_driver_expression(drv.expression, frame, fps)
            if value is None:
                continue
        else:
            continue
        _assign(tree, fc.data_path, fc.array_index, value)
        changed = True
    return changed


def apply_all(frame, fps):
    from .tree import TREE_ID
    for tree in bpy.data.node_groups:
        if tree.bl_idname == TREE_ID:
            apply_animation(tree, frame, fps)
