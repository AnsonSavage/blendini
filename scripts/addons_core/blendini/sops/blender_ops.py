# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
SOPs implemented with Blender as the backend.

Geometry is converted to a temporary mesh, processed by a Blender modifier (evaluated in a
private scene so the user's scene is never touched) or by bmesh, and converted back.
"""

import bpy
import bmesh
import numpy as np

from .base import SOP, Parm, register_sop, CookError
from ..core.geometry import Geometry

_TEMP_COLLECTION = "_blendini_eval"


def _temp_collection():
    """A collection in the active scene used to evaluate temporary objects."""
    scene = bpy.context.scene
    coll = bpy.data.collections.get(_TEMP_COLLECTION)
    if coll is None:
        coll = bpy.data.collections.new(_TEMP_COLLECTION)
    if scene is not None and coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    coll.hide_render = True
    return coll


def _in(inputs, i=0):
    if len(inputs) > i and inputs[i] is not None:
        return inputs[i]
    return Geometry()


def apply_modifier(geo, mod_type, settings=None, extra_objects=()):
    """Run a Blender modifier on geometry and return the result."""
    from ..bl.mesh_io import geo_to_mesh, mesh_to_geo
    coll = _temp_collection()
    mesh = bpy.data.meshes.new("_blendini_tmp")
    geo_to_mesh(geo, mesh)
    obj = bpy.data.objects.new("_blendini_tmp", mesh)
    coll.objects.link(obj)
    temp_objs = []
    try:
        mod = obj.modifiers.new("blendini", mod_type)
        for other in extra_objects:
            om = bpy.data.meshes.new("_blendini_tmp_other")
            geo_to_mesh(other, om)
            oo = bpy.data.objects.new("_blendini_tmp_other", om)
            coll.objects.link(oo)
            oo.hide_render = True
            temp_objs.append(oo)
        if settings:
            settings(mod, temp_objs)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        result = bpy.data.meshes.new_from_object(obj_eval, preserve_all_data_layers=True, depsgraph=depsgraph)
        try:
            out = mesh_to_geo(result)
        finally:
            bpy.data.meshes.remove(result)
    finally:
        bpy.data.objects.remove(obj)
        bpy.data.meshes.remove(mesh)
        for oo in temp_objs:
            m = oo.data
            bpy.data.objects.remove(oo)
            bpy.data.meshes.remove(m)
    return out


def _with_bmesh(geo, fn):
    from ..bl.mesh_io import geo_to_mesh, mesh_to_geo
    mesh = bpy.data.meshes.new("_blendini_tmp")
    try:
        geo_to_mesh(geo, mesh)
        bm = bmesh.new()
        bm.from_mesh(mesh)
        fn(bm)
        bm.to_mesh(mesh)
        bm.free()
        return mesh_to_geo(mesh)
    finally:
        bpy.data.meshes.remove(mesh)


@register_sop
class SubdivideSOP(SOP):
    id = 'subdivide'
    label = "Subdivide"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_SUBSURF'
    parms = (
        Parm('iterations', "Depth", 'int', default=1, min=0, max=6),
        Parm('algorithm', "Algorithm", 'enum', items=[
            ('CATMULL_CLARK', "OpenSubdiv Catmull-Clark", ""), ('SIMPLE', "Simple (Linear)", ""),
        ]),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        if p.iterations <= 0 or geo.num_prims == 0:
            return geo

        def settings(mod, _):
            mod.levels = p.iterations
            mod.render_levels = p.iterations
            mod.subdivision_type = p.algorithm
        return apply_modifier(geo, 'SUBSURF', settings)


@register_sop
class PolyBevelSOP(SOP):
    id = 'polybevel'
    label = "PolyBevel"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_BEVEL'
    parms = (
        Parm('offset', "Distance", 'float', default=0.05, min=0, soft_max=1),
        Parm('divisions', "Divisions", 'int', default=1, min=1, max=20),
        Parm('affect', "Affect", 'enum', items=[('EDGES', "Edges", ""), ('VERTICES', "Points", "")]),
        Parm('angle', "Ignore Flat Edges Below (degrees)", 'float', default=30.0, min=0, max=180),
    )

    def cook(self, inputs, p, ctx):
        import math

        def settings(mod, _):
            mod.width = p.offset
            mod.segments = p.divisions
            mod.affect = p.affect
            mod.limit_method = 'ANGLE'
            mod.angle_limit = math.radians(p.angle)
        return apply_modifier(_in(inputs), 'BEVEL', settings)


@register_sop
class BooleanSOP(SOP):
    id = 'boolean'
    label = "Boolean"
    category = 'Polygon'
    inputs = 2
    input_labels = ("A", "B")
    icon = 'MOD_BOOLEAN'
    parms = (
        Parm('booleanop', "Operation", 'enum', items=[
            ('UNION', "Union", ""), ('INTERSECT', "Intersect", ""), ('DIFFERENCE', "Subtract", ""),
        ]),
        Parm('solver', "Solver", 'enum', items=[('EXACT', "Exact", ""), ('FLOAT', "Fast", "")]),
    )

    def cook(self, inputs, p, ctx):
        a, b = _in(inputs, 0), _in(inputs, 1)
        if b.num_prims == 0:
            return a
        if a.num_prims == 0:
            return b if p.booleanop == 'UNION' else a

        def settings(mod, others):
            mod.operation = p.booleanop
            mod.object = others[0]
            try:
                mod.solver = p.solver
            except TypeError:
                pass
        return apply_modifier(a, 'BOOLEAN', settings, extra_objects=[b])


@register_sop
class RemeshSOP(SOP):
    id = 'remesh'
    label = "Remesh"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_REMESH'
    parms = (
        Parm('voxelsize', "Voxel Size", 'float', default=0.1, min=0.001, soft_max=1),
        Parm('adaptivity', "Adaptivity", 'float', default=0.0, min=0, max=1),
    )

    def cook(self, inputs, p, ctx):
        def settings(mod, _):
            mod.mode = 'VOXEL'
            mod.voxel_size = max(p.voxelsize, 0.001)
            mod.adaptivity = p.adaptivity
        return apply_modifier(_in(inputs), 'REMESH', settings)


@register_sop
class PolyReduceSOP(SOP):
    id = 'polyreduce'
    label = "PolyReduce"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_DECIM'
    parms = (Parm('percentage', "Keep Percentage", 'float', default=50.0, min=0.1, max=100),)

    def cook(self, inputs, p, ctx):
        def settings(mod, _):
            mod.ratio = p.percentage / 100.0
        return apply_modifier(_in(inputs), 'DECIMATE', settings)


@register_sop
class ClipSOP(SOP):
    id = 'clip'
    label = "Clip"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_BOOLEAN'
    parms = (
        Parm('clipop', "Keep", 'enum', items=[('above', "Primitives Above the Plane", ""),
                                              ('below', "Primitives Below the Plane", ""),
                                              ('both', "All Primitives (split)", "")]),
        Parm('origin', "Origin", 'vector'),
        Parm('dir', "Direction", 'vector', default=(0.0, 1.0, 0.0)),
        Parm('dist', "Distance", 'float', default=0.0),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        n = np.asarray(p.dir, dtype=np.float64)
        ln = np.linalg.norm(n)
        if ln == 0:
            return geo
        n = n / ln
        co = np.asarray(p.origin) + n * p.dist

        def fn(bm):
            geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
            bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6, plane_co=co, plane_no=n,
                                   clear_outer=(p.clipop == 'below'), clear_inner=(p.clipop == 'above'))
        return _with_bmesh(geo, fn)


@register_sop
class SolidifySOP(SOP):
    id = 'solidify'
    label = "Solidify (Thickness)"
    category = 'Blender'
    inputs = 1
    icon = 'MOD_SOLIDIFY'
    parms = (
        Parm('thickness', "Thickness", 'float', default=0.1, soft_min=-1, soft_max=1),
        Parm('offset', "Offset", 'float', default=-1.0, min=-1, max=1),
    )

    def cook(self, inputs, p, ctx):
        def settings(mod, _):
            mod.thickness = p.thickness
            mod.offset = p.offset
        return apply_modifier(_in(inputs), 'SOLIDIFY', settings)


@register_sop
class ObjectMergeSOP(SOP):
    id = 'objectmerge'
    label = "Object Merge"
    category = 'Blender'
    inputs = 0
    icon = 'OBJECT_DATA'
    time_dependent = True
    parms = (
        Parm('objpath', "Object", 'object'),
        Parm('xformtype', "Transform", 'enum', items=[('none', "None", ""), ('world', "Into World Space", "")]),
        Parm('evaluated', "Use Evaluated (Modified) Mesh", 'bool', default=True),
    )

    def cook(self, inputs, p, ctx):
        from ..bl.mesh_io import mesh_to_geo
        obj = bpy.data.objects.get(p.objpath) if p.objpath else None
        if obj is None:
            if p.objpath:
                raise CookError("Object '%s' not found" % p.objpath)
            return Geometry()
        # Blender is Z-up, Blendini (like Houdini) is Y-up.
        z_to_y = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
        matrix = z_to_y @ np.array(obj.matrix_world) if p.xformtype == 'world' else z_to_y
        if p.evaluated:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            obj_eval = obj.evaluated_get(depsgraph)
            try:
                mesh = obj_eval.to_mesh()
            except RuntimeError:
                return Geometry()
            if mesh is None:
                return Geometry()
            try:
                return mesh_to_geo(mesh, matrix)
            finally:
                obj_eval.to_mesh_clear()
        if obj.type != 'MESH':
            raise CookError("Object '%s' is not a mesh" % obj.name)
        return mesh_to_geo(obj.data, matrix)


_MODIFIER_ITEMS = [
    ('NODES', "Geometry Nodes", ""), ('SUBSURF', "Subdivision Surface", ""), ('BEVEL', "Bevel", ""),
    ('REMESH', "Remesh", ""), ('DECIMATE', "Decimate", ""), ('SOLIDIFY', "Solidify", ""),
    ('SMOOTH', "Smooth", ""), ('TRIANGULATE', "Triangulate", ""), ('WIREFRAME', "Wireframe", ""),
    ('SKIN', "Skin", ""), ('SCREW', "Screw", ""), ('DISPLACE', "Displace", ""), ('WELD', "Weld", ""),
    ('MIRROR', "Mirror", ""), ('ARRAY', "Array", ""), ('TRIANGULATE', "Triangulate", ""),
    ('CAST', "Cast", ""), ('SIMPLE_DEFORM', "Simple Deform", ""), ('LAPLACIANSMOOTH', "Laplacian Smooth", ""),
]
_seen = set()
_MODIFIER_ITEMS = [i for i in _MODIFIER_ITEMS if not (i[0] in _seen or _seen.add(i[0]))]


@register_sop
class BlenderModifierSOP(SOP):
    id = 'blendermodifier'
    label = "Blender Modifier"
    category = 'Blender'
    inputs = 1
    icon = 'MODIFIER'
    description = "Run any Blender modifier, including Geometry Nodes groups"
    parms = (
        Parm('modtype', "Modifier", 'enum', items=_MODIFIER_ITEMS),
        Parm('nodegroup', "Node Group", 'nodegroup', hide_when={'modtype': tuple(i[0] for i in _MODIFIER_ITEMS if i[0] != 'NODES')},
             description="Name of a Geometry Nodes group"),
        Parm('settings', "Settings", 'string',
             description="Assignments separated by ';', e.g. levels=2; quality=3 (Geometry Nodes: input names, e.g. Count=10)"),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)

        def settings(mod, _):
            if p.modtype == 'NODES':
                ng = bpy.data.node_groups.get(p.nodegroup)
                if ng is None:
                    raise CookError("Geometry Nodes group '%s' not found" % p.nodegroup)
                mod.node_group = ng
            for part in p.settings.split(';'):
                if '=' not in part:
                    continue
                key, val = (s.strip() for s in part.split('=', 1))
                try:
                    import ast
                    value = ast.literal_eval(val)
                except (ValueError, SyntaxError):
                    value = val
                try:
                    if p.modtype == 'NODES' and not hasattr(mod, key):
                        _set_group_input(mod, key, value)
                    else:
                        setattr(mod, key, value)
                except (AttributeError, TypeError, ValueError) as ex:
                    raise CookError("Can't set %s: %s" % (key, ex))
        return apply_modifier(geo, p.modtype, settings)


@register_sop
class FontSOP(SOP):
    id = 'font'
    label = "Font"
    category = 'Primitive'
    inputs = 0
    icon = 'FONT_DATA'
    parms = (
        Parm('text', "Text", 'string', default="Blendini"),
        Parm('fontsize', "Font Size", 'float', default=1.0, min=0.01, soft_max=10),
        Parm('t', "Translate", 'vector'),
        Parm('halign', "Horizontal Alignment", 'enum', items=[('CENTER', "Center", ""), ('LEFT', "Left", ""),
                                                               ('RIGHT', "Right", "")]),
        Parm('extrude', "Extrude", 'float', default=0.0, min=0, soft_max=1),
        Parm('resolution', "Curve Resolution", 'int', default=4, min=1, max=32),
    )

    def cook(self, inputs, p, ctx):
        from ..bl.mesh_io import mesh_to_geo
        coll = _temp_collection()
        curve = bpy.data.curves.new("_blendini_font", 'FONT')
        curve.body = p.text or " "
        curve.size = p.fontsize
        curve.align_x = p.halign
        curve.extrude = p.extrude
        curve.resolution_u = p.resolution
        curve.fill_mode = 'BOTH' if p.extrude else 'FRONT'
        obj = bpy.data.objects.new("_blendini_font", curve)
        coll.objects.link(obj)
        try:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            mesh = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph), depsgraph=depsgraph)
            try:
                geo = mesh_to_geo(mesh)
            finally:
                bpy.data.meshes.remove(mesh)
        finally:
            bpy.data.objects.remove(obj)
            bpy.data.curves.remove(curve)
        geo.P = geo.P + np.asarray(p.t)
        return geo


def _set_group_input(mod, key, value):
    """Set a Geometry Nodes modifier input by socket name or identifier."""
    from ..bl.display import _set_modifier_input
    ident = None
    for item in mod.node_group.interface.items_tree:
        if getattr(item, 'in_out', None) == 'INPUT' and key in (item.name, item.identifier):
            ident = item.identifier
            break
    if ident is None:
        raise CookError("Geometry Nodes group has no input '%s'" % key)
    if isinstance(value, (list, tuple)):
        value = tuple(value)
    _set_modifier_input(mod, ident, value)


@register_sop
class AttribFromMapSOP(SOP):
    id = 'attribfrommap'
    label = "Attribute from Map"
    category = 'Attribute'
    inputs = 1
    icon = 'IMAGE_DATA'
    parms = (
        Parm('image', "Texture Map", 'image', description="Blender image name or file path"),
        Parm('uvattrib', "UV Attribute", 'string', default='uv'),
        Parm('exportattrib', "Export Attribute", 'string', default='Cd'),
    )

    def cook(self, inputs, p, ctx):
        from ..core.images import image_pixels, sample
        from ..core.geometry import POINT, VERTEX
        from .attribs import promote
        geo = _in(inputs).copy()
        if not p.image:
            return geo
        px = image_pixels(p.image)
        if px is None:
            raise CookError("Image '%s' not found" % p.image)
        cls = POINT
        uv = geo.attribs[POINT].get(p.uvattrib)
        if uv is None and p.uvattrib in geo.attribs[VERTEX]:
            uv = geo.attribs[VERTEX][p.uvattrib]
            cls = VERTEX
        if uv is None:
            raise CookError("No '%s' attribute (use a UV Texture SOP first)" % p.uvattrib)
        rgba = sample(px, uv[:, 0], uv[:, 1]).astype(np.float64)
        geo.attribs[cls][p.exportattrib or 'Cd'] = rgba[:, :3]
        if cls == VERTEX and (p.exportattrib or 'Cd') == 'Cd':
            # Also provide point colors so wrangles can read them.
            geo.attribs[POINT]['Cd'] = promote(geo, rgba[:, :3], VERTEX, POINT, 'mean')
        return geo
