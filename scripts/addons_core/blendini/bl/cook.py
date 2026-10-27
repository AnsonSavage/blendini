# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Cooking: evaluate the network upstream of the display node and write the result into the
network's output object.

Every node result is cached together with a signature made of its parameter values, its
inputs' signatures and (for time dependent nodes) the frame, so only nodes whose
upstream changed are recooked, like Houdini.
"""

import time

import bpy
import numpy as np

from ..core.geometry import Geometry
from ..sops import REGISTRY, CookContext, CookError
from .tree import TREE_ID, BlendiniNodeBase, update_colors

# (tree session uid, node name) -> dict(sig=..., geo=..., info...)
_cache = {}
# Persistent solver states per node.
_states = {}
_pending = set()
_timer_registered = False
_cooking = False


def _tree_key(tree):
    return tree.session_uid


def _node_key(node):
    return (_tree_key(node.id_data), node.name)


def node_info(node):
    entry = _cache.get(_node_key(node))
    return entry if entry is not None else {}


def node_geometry(node):
    entry = _cache.get(_node_key(node))
    return entry.get('geo') if entry else None


def forget_node(node):
    try:
        key = _node_key(node)
    except ReferenceError:
        return
    _cache.pop(key, None)
    _states.pop(key, None)


def is_cooking():
    return _cooking


# ----------------------------------------------------------------------------
# Scheduling


def request_cook(tree):
    if tree is None or not getattr(tree, 'auto_cook', True):
        return
    if bpy.app.background:
        # No timers in background mode, and cooking inside update callbacks is unsafe:
        # scripts call cook_tree() explicitly.
        return
    global _timer_registered
    _pending.add(tree.name_full if hasattr(tree, 'name_full') else tree.name)
    if not _timer_registered:
        _timer_registered = True
        bpy.app.timers.register(_flush, first_interval=0.0)


def _flush():
    global _timer_registered
    _timer_registered = False
    names = list(_pending)
    _pending.clear()
    for name in names:
        tree = bpy.data.node_groups.get(name)
        if tree is not None and tree.bl_idname == TREE_ID:
            try:
                cook_tree(tree)
            except Exception as ex:  # Never let a cook error kill the timer system.
                print("Blendini cook error:", ex)
    return None


# ----------------------------------------------------------------------------
# Evaluation


def _freeze(v):
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if hasattr(v, '__len__') and not isinstance(v, str):
        return tuple(v)
    return v


def _input_nodes(socket):
    """Upstream Blendini nodes connected to a socket (following reroutes), in order."""
    out = []
    links = sorted(socket.links, key=lambda l: getattr(l, 'multi_input_sort_id', 0)) if socket.is_multi_input else socket.links
    for link in links:
        if not link.is_valid or link.is_muted:
            continue
        node = link.from_node
        while node is not None and node.bl_idname == 'NodeReroute':
            if not node.inputs[0].links:
                node = None
                break
            node = node.inputs[0].links[0].from_node
        if node is not None and isinstance(node, BlendiniNodeBase):
            out.append(node)
    return out


def _is_time_dependent(node, parms):
    sop = node.sop
    if sop.id == 'solverwrangle':
        return True
    if 'snippet' in parms:
        code = parms['snippet']
        return any(k in code for k in ('@Time', '@Frame', '@TimeInc', '@SimTime', '@SimFrame'))
    return sop.time_dependent


def _tree_has_animation(tree):
    ad = tree.animation_data
    return ad is not None and (ad.action is not None or len(ad.drivers) > 0)


def cook_node(node, ctx_base, visiting=None):
    """Cook a node (and its inputs). Returns (geometry, signature)."""
    visiting = visiting if visiting is not None else set()
    key = _node_key(node)
    if key in visiting:
        raise CookError("Cycle detected at %s" % node.name)
    visiting.add(key)
    try:
        sop = node.sop
        inputs = []
        input_sigs = []
        upstream_ctx = ctx_base
        if sop.id == 'timeshift' and not node.mute:
            f = node.p_frame if node.p_mode == 'frame' else ctx_base['frame'] + node.p_frame
            upstream_ctx = dict(ctx_base, frame=float(f), time=float(f - 1) / ctx_base['fps'])
        for sock in node.inputs:
            ups = _input_nodes(sock)
            if sock.is_multi_input:
                for up in ups:
                    g, s = cook_node(up, upstream_ctx, visiting)
                    inputs.append(g)
                    input_sigs.append(s)
            else:
                if ups:
                    g, s = cook_node(ups[0], upstream_ctx, visiting)
                else:
                    g, s = None, None
                inputs.append(g)
                input_sigs.append(s)
        parms = node.parm_values()
        channels = node.channels()
        ramps = node.ramps() if len(node.spare) else {}
        time_dep = _is_time_dependent(node, parms) or ctx_base['animated']
        sig = (
            sop.id, node.mute,
            tuple(sorted((k, _freeze(v)) for k, v in parms.items())),
            tuple(sorted((k, _freeze(v)) for k, v in channels.items())),
            tuple(sorted(ramps.items())),
            tuple(input_sigs),
            ctx_base['frame'] if time_dep else None,
            ctx_base['objmerge_epoch'] if sop.id == 'objectmerge' else None,
        )
        entry = _cache.get(key)
        if entry is not None and entry.get('sig') == sig and 'geo' in entry:
            return entry['geo'], sig

        if node.mute:
            geo = inputs[0] if inputs and inputs[0] is not None else Geometry()
            _cache[key] = {'sig': sig, 'geo': geo, 'time': 0.0, 'messages': [], 'error': None, 'bypassed': True}
            return geo, sig

        ctx = CookContext(time=ctx_base['time'], frame=ctx_base['frame'], fps=ctx_base['fps'],
                          node_path=node.name, state=_states.setdefault(key, {}))
        ctx.channels = channels
        ctx.ramps = ramps
        t0 = time.perf_counter()
        error = None
        try:
            geo = sop.cook(inputs, parms, ctx)
            if geo is None:
                geo = Geometry()
        except CookError as ex:
            error = str(ex)
            geo = Geometry()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            error = "%s: %s" % (type(ex).__name__, ex)
            geo = Geometry()
        dt = time.perf_counter() - t0
        _cache[key] = {
            'sig': sig, 'geo': geo, 'time': dt, 'messages': ctx.messages[-200:], 'warnings': ctx.warnings,
            'error': error,
        }
        if error:
            # Don't keep a signature for failed cooks so fixing upstream retries.
            _cache[key]['sig'] = ('error',) + sig
        return geo, sig
    finally:
        visiting.discard(key)


def display_node(tree):
    nodes = [n for n in tree.nodes if isinstance(n, BlendiniNodeBase)]
    for n in nodes:
        if n.display_flag:
            return n
    for n in nodes:
        if n.sop_id == 'output':
            return n
    return nodes[-1] if nodes else None


_objmerge_epoch = 0


def bump_objmerge_epoch():
    global _objmerge_epoch
    _objmerge_epoch += 1


def ensure_target(tree):
    obj = tree.target
    if obj is not None and obj.type == 'MESH':
        return obj
    name = tree.name
    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    scene = bpy.context.scene
    if scene is not None:
        scene.collection.objects.link(obj)
    obj["blendini_tree"] = tree.name
    tree.target = obj
    from .display import setup_target
    setup_target(obj)
    return obj


def cook_tree(tree, frame=None):
    """Cook the display node and write it into the output object."""
    global _cooking
    if _cooking:
        return None
    _cooking = True
    try:
        node = display_node(tree)
        if node is None:
            return None
        scene = bpy.context.scene
        fps = scene.render.fps / scene.render.fps_base if scene else 24.0
        if frame is None:
            frame = scene.frame_current + scene.frame_subframe if scene else 1.0
        ctx_base = {
            'frame': float(frame), 'time': float(frame - 1) / fps, 'fps': fps,
            'animated': _tree_has_animation(tree), 'objmerge_epoch': _objmerge_epoch,
        }
        try:
            geo, sig = cook_node(node, ctx_base)
        except CookError as ex:
            _cache.setdefault(_node_key(node), {})['error'] = str(ex)
            geo, sig = Geometry(), None
        obj = ensure_target(tree)
        tkey = ('__written__', _tree_key(tree))
        if _cache.get(tkey) != (sig, obj.data.name) or sig is None:
            from .mesh_io import geo_to_mesh
            geo_to_mesh(geo, obj.data)
            obj.data.update()
            _cache[tkey] = (sig, obj.data.name)
        try:
            update_colors(tree)
        except AttributeError:
            pass
        _tag_redraw()
        return geo
    finally:
        _cooking = False


def _tag_redraw():
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type in ('NODE_EDITOR', 'VIEW_3D', 'SPREADSHEET'):
                area.tag_redraw()


def all_trees():
    return [t for t in bpy.data.node_groups if t.bl_idname == TREE_ID]


def clear_cache():
    _cache.clear()
    _states.clear()
