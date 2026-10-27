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
    # Use pscale as the point radius when present (Houdini convention).
    pscale = nodes.new('GeometryNodeInputNamedAttribute')
    pscale.data_type = 'FLOAT'
    pscale.inputs['Name'].default_value = "pscale"
    pscale.location = (-600, -400)
    switch = nodes.new('GeometryNodeSwitch')
    switch.input_type = 'FLOAT'
    switch.location = (-400, -400)
    switch.inputs['False'].default_value = 0.015
    links.new(pscale.outputs['Exists'], switch.inputs['Switch'])
    links.new(pscale.outputs['Attribute'], switch.inputs['True'])
    links.new(switch.outputs[0], to_points.inputs['Radius'])
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
    ensure_material(obj)


# ----------------------------------------------------------------------------
# Packed geometry -> Blender instances

INST_GROUP = "Blendini Instancer"
INST_MOD = "Blendini Instances"


def _instancer_group():
    ng = bpy.data.node_groups.get(INST_GROUP)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(INST_GROUP, 'GeometryNodeTree')
    ng.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Collection", in_out='INPUT', socket_type='NodeSocketCollection')
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N, L = ng.nodes, ng.links
    gin = N.new('NodeGroupInput')
    gout = N.new('NodeGroupOutput')
    sel = N.new('GeometryNodeInputNamedAttribute')
    sel.data_type = 'BOOLEAN'
    sel.inputs['Name'].default_value = "__instpt"
    idx = N.new('GeometryNodeInputNamedAttribute')
    idx.data_type = 'INT'
    idx.inputs['Name'].default_value = "__inst"
    rot = N.new('GeometryNodeInputNamedAttribute')
    rot.data_type = 'QUATERNION'
    rot.inputs['Name'].default_value = "__inst_rot"
    scl = N.new('GeometryNodeInputNamedAttribute')
    scl.data_type = 'FLOAT_VECTOR'
    scl.inputs['Name'].default_value = "__inst_scale"
    coll = N.new('GeometryNodeCollectionInfo')
    coll.inputs['Separate Children'].default_value = True
    coll.inputs['Reset Children'].default_value = True
    iop = N.new('GeometryNodeInstanceOnPoints')
    iop.inputs['Pick Instance'].default_value = True
    delete = N.new('GeometryNodeDeleteGeometry')
    join = N.new('GeometryNodeJoinGeometry')
    L.new(gin.outputs[1], coll.inputs['Collection'])
    L.new(gin.outputs[0], iop.inputs['Points'])
    L.new(sel.outputs['Attribute'], iop.inputs['Selection'])
    L.new(coll.outputs[0], iop.inputs['Instance'])
    L.new(idx.outputs['Attribute'], iop.inputs['Instance Index'])
    L.new(rot.outputs['Attribute'], iop.inputs['Rotation'])
    L.new(scl.outputs['Attribute'], iop.inputs['Scale'])
    L.new(gin.outputs[0], delete.inputs['Geometry'])
    L.new(sel.outputs['Attribute'], delete.inputs['Selection'])
    L.new(delete.outputs[0], join.inputs[0])
    L.new(iop.outputs[0], join.inputs[0])
    L.new(join.outputs[0], gout.inputs[0])
    return ng


def _collection_input_id(ng):
    for item in ng.interface.items_tree:
        if getattr(item, 'in_out', None) == 'INPUT' and item.socket_type == 'NodeSocketCollection':
            return item.identifier
    return None


def sync_instances(tree, obj, geo):
    """Mirror packed geometry sources as hidden objects and drive the instancer modifier."""
    from .mesh_io import geo_to_mesh
    mod = obj.modifiers.get(INST_MOD)
    if not geo.instances:
        if mod is not None:
            mod.show_viewport = False
            mod.show_render = False
        return
    cname = ".blendini_instances_" + tree.name
    coll = bpy.data.collections.get(cname)
    if coll is None:
        coll = bpy.data.collections.new(cname)
    prefix = ".blendini_inst_%s_" % tree.name
    for i, src in enumerate(geo.instances):
        name = "%s%03d" % (prefix, i)
        ob = bpy.data.objects.get(name)
        if ob is None:
            ob = bpy.data.objects.new(name, bpy.data.meshes.new(name))
        if ob.name not in coll.objects:
            coll.objects.link(ob)
        geo_to_mesh(src, ob.data)
        for m in list(ob.data.materials):
            pass
        if obj.data.materials and not ob.data.materials:
            ob.data.materials.append(obj.data.materials[0])
    # Remove sources that are no longer used (keep order: objects are sorted by name).
    for ob in list(coll.objects):
        if ob.name.startswith(prefix):
            try:
                k = int(ob.name[len(prefix):])
            except ValueError:
                continue
            if k >= len(geo.instances):
                mesh = ob.data
                bpy.data.objects.remove(ob)
                if mesh is not None and mesh.users == 0:
                    bpy.data.meshes.remove(mesh)
    if mod is None:
        mod = obj.modifiers.new(INST_MOD, 'NODES')
        mod.node_group = _instancer_group()
        # The instancer has to run before the point display modifier.
        try:
            obj.modifiers.move(len(obj.modifiers) - 1, 0)
        except (AttributeError, RuntimeError, TypeError):
            pass
    mod.show_viewport = True
    mod.show_render = True
    ident = _collection_input_id(mod.node_group)
    if ident is not None:
        _set_modifier_input(mod, ident, coll)
        obj.update_tag()


def _set_modifier_input(mod, ident, value):
    props = getattr(mod, 'properties', None)
    if props is not None:
        # Blender 5.x: typed modifier inputs.
        prop = getattr(props.inputs, ident, None)
        if prop is not None:
            if prop.value != value:
                prop.value = value
            return
    try:
        if mod.get(ident) != value:
            mod[ident] = value
    except TypeError:
        pass


MATERIAL_NAME = "Blendini Cd"


def cd_material():
    """A material that renders the Cd attribute (mesh or instance), grey when there is none."""
    mat = bpy.data.materials.get(MATERIAL_NAME)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(MATERIAL_NAME)
    try:
        mat.use_nodes = True
    except AttributeError:
        pass
    nt = mat.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        return mat
    geo_cd = nt.nodes.new('ShaderNodeAttribute')
    geo_cd.attribute_name = 'Cd'
    geo_cd.attribute_type = 'GEOMETRY'
    geo_cd.location = (-700, 200)
    inst_cd = nt.nodes.new('ShaderNodeAttribute')
    inst_cd.attribute_name = 'Cd'
    inst_cd.attribute_type = 'INSTANCER'
    inst_cd.location = (-700, -100)
    mix1 = nt.nodes.new('ShaderNodeMix')
    mix1.data_type = 'RGBA'
    mix1.location = (-450, 200)
    mix1.inputs['A'].default_value = (0.8, 0.8, 0.8, 1.0)
    mix2 = nt.nodes.new('ShaderNodeMix')
    mix2.data_type = 'RGBA'
    mix2.location = (-250, 100)
    links = nt.links
    # Attribute 'Alpha' is 1 where the attribute exists, 0 otherwise.
    links.new(geo_cd.outputs['Alpha'], mix1.inputs['Factor'])
    links.new(geo_cd.outputs['Color'], mix1.inputs['B'])
    links.new(mix1.outputs['Result'], mix2.inputs['A'])
    links.new(inst_cd.outputs['Alpha'], mix2.inputs['Factor'])
    links.new(inst_cd.outputs['Color'], mix2.inputs['B'])
    links.new(mix2.outputs['Result'], bsdf.inputs['Base Color'])
    return mat


def ensure_material(obj):
    mesh = obj.data
    if mesh is not None and not mesh.materials:
        try:
            mesh.materials.append(cd_material())
        except Exception as ex:
            print("Blendini: could not create the Cd material:", ex)


VIS_NAME = "visualize"


def visualize_attribute(obj, geo, name):
    """Write a false-color 'visualize' color attribute and make it the displayed color."""
    import numpy as np
    from ..sops.attribs import _infra_red
    mesh = obj.data
    if not name:
        old = mesh.color_attributes.get(VIS_NAME)
        if old is not None:
            mesh.color_attributes.remove(old)
        cd = mesh.color_attributes.get('Cd')
        if cd is not None:
            mesh.color_attributes.active_color = cd
        return
    domain = None
    arr = geo.attribs['point'].get(name)
    if arr is not None:
        domain = 'POINT'
    else:
        arr = geo.attribs['prim'].get(name)
        if arr is not None:
            # Only polygons become Blender faces.
            sizes = geo.prim_sizes()
            arr = arr[geo.prim_closed & (sizes >= 3)]
            domain = 'FACE'
    if arr is None or arr.dtype == object:
        return
    v = arr.astype(np.float64)
    if v.ndim > 1:
        v = np.linalg.norm(v.reshape(len(v), -1), axis=1) if v.shape[1] != 1 else v[:, 0]
    lo, hi = (float(v.min()), float(v.max())) if len(v) else (0.0, 1.0)
    u = (v - lo) / (hi - lo if hi > lo else 1.0)
    rgba = np.ones((len(u), 4), dtype=np.float32)
    rgba[:, :3] = _infra_red(u)
    old = mesh.color_attributes.get(VIS_NAME)
    if old is not None:
        mesh.color_attributes.remove(old)
    attr = mesh.color_attributes.new(VIS_NAME, 'FLOAT_COLOR', domain)
    if len(attr.data) != len(rgba):
        mesh.color_attributes.remove(attr)
        return
    attr.data.foreach_set("color", rgba.ravel())
    mesh.color_attributes.active_color = attr
