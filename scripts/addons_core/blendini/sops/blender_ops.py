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


def _release_temp_collection():
    coll = bpy.data.collections.get(_TEMP_COLLECTION)
    if coll is not None and not coll.objects:
        bpy.data.collections.remove(coll)


def remove_stale_rbd_scenes():
    """Private RBD scenes don't survive a reload of the solver state: remove leftovers."""
    for scene in list(bpy.data.scenes):
        if scene.name.startswith("_blendini_rbd_"):
            _rbd_cleanup({'scene': scene.name})


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
        _release_temp_collection()
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
            _release_temp_collection()
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


@register_sop
class VoronoiFractureSOP(SOP):
    id = 'voronoifracture'
    label = "Voronoi Fracture"
    category = 'Polygon'
    inputs = 2
    input_labels = ("Geometry to Fracture", "Cell Points")
    icon = 'MOD_EXPLODE'
    description = "Cut closed geometry into Voronoi cells (uses bmesh bisect + hole filling)"
    parms = (
        Parm('npts', "Cell Count (no 2nd input)", 'int', default=20, min=1, soft_max=200),
        Parm('seed', "Seed", 'int', default=0),
        Parm('neighbours', "Max Neighbour Cells (0 = all)", 'int', default=0, min=0, max=512,
             description="Cut each cell only against its nearest sites (faster, but too few cause overlapping cells)"),
        Parm('cutgroup', "Inside Group", 'string', default='inside'),
    )

    def cook(self, inputs, p, ctx):
        from ..bl.mesh_io import geo_to_mesh, mesh_to_geo
        from ..core.geometry import merge, PRIM
        geo = _in(inputs, 0)
        if geo.num_prims == 0:
            return geo
        sites_geo = _in(inputs, 1)
        if sites_geo.num_points:
            sites = sites_geo.P.copy()
        else:
            lo, hi = geo.bbox()
            rng = np.random.RandomState(p.seed & 0x7FFFFFFF)
            sites = lo + rng.random_sample((max(p.npts, 1), 3)) * (hi - lo)
        k = len(sites) - 1 if p.neighbours <= 0 else min(max(p.neighbours, 3), len(sites) - 1)
        src_mesh = bpy.data.meshes.new("_blendini_vfrac")
        try:
            geo_to_mesh(geo, src_mesh)
            pieces = []
            for i, site in enumerate(sites):
                d = np.linalg.norm(sites - site, axis=1)
                order = np.argsort(d)[1:k + 1]
                bm = bmesh.new()
                bm.from_mesh(src_mesh)
                for j in order:
                    other = sites[j]
                    normal = other - site
                    ln = np.linalg.norm(normal)
                    if ln == 0:
                        continue
                    normal /= ln
                    mid = (site + other) / 2
                    geom = bm.verts[:] + bm.edges[:] + bm.faces[:]
                    res = bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-6, plane_co=mid, plane_no=normal,
                                                 clear_outer=True)
                    cut_edges = [e for e in res['geom_cut'] if isinstance(e, bmesh.types.BMEdge)]
                    if cut_edges:
                        bmesh.ops.holes_fill(bm, edges=cut_edges, sides=0)
                    if not bm.faces:
                        break
                if not bm.faces:
                    bm.free()
                    continue
                tmp = bpy.data.meshes.new("_blendini_vcell")
                bm.to_mesh(tmp)
                bm.free()
                try:
                    piece = mesh_to_geo(tmp)
                finally:
                    bpy.data.meshes.remove(tmp)
                piece.attribs[PRIM]['name'] = np.array(["piece%d" % i] * piece.num_prims, dtype=object)
                piece.attribs[PRIM]['class'] = np.full(piece.num_prims, len(pieces), dtype=np.int64)
                pieces.append(piece)
        finally:
            bpy.data.meshes.remove(src_mesh)
        out = merge(pieces)
        if p.cutgroup and out.num_prims:
            # Faces whose normal doesn't match any original face are the new inside faces.
            out.groups[PRIM][p.cutgroup] = _inside_faces(out, geo)
        return out


def _inside_faces(out, original):
    from ..core.spatial import closest
    centers = out.prim_centers()
    loc, nrm, prim, dist = closest(original, centers)
    return dist > 1e-4


# ----------------------------------------------------------------------------
# Rigid bodies (Blender's Bullet integration as the backend)


def _rbd_cleanup(state):
    scene = bpy.data.scenes.get(state.get('scene', ''))
    if scene is not None:
        objs = list(scene.collection.objects)
        if scene.rigidbody_world is not None and scene.rigidbody_world.collection is not None:
            coll = scene.rigidbody_world.collection
        else:
            coll = None
        for ob in objs:
            mesh = ob.data
            bpy.data.objects.remove(ob)
            if mesh is not None and mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        bpy.data.scenes.remove(scene)
        if coll is not None and coll.users == 0:
            bpy.data.collections.remove(coll)
    state.clear()


def _add_rigid_body(scene, ob, kind):
    with bpy.context.temp_override(scene=scene, object=ob, active_object=ob, selected_objects=[ob],
                                   selected_editable_objects=[ob]):
        bpy.ops.rigidbody.object_add(type=kind)


@register_sop
class RBDSolverSOP(SOP):
    id = 'rbdsolver'
    label = "RBD Bullet Solver"
    category = 'Simulation'
    inputs = 2
    input_labels = ("Pieces", "Static Colliders")
    icon = 'RIGID_BODY'
    time_dependent = True
    description = "Rigid body simulation of pieces (by 'name'/'class' attribute or connectivity) using Bullet"
    parms = (
        Parm('pieceattrib', "Piece Attribute", 'string', default='name'),
        Parm('startframe', "Start Frame", 'int', default=1),
        Parm('ground', "Ground Plane", 'bool', default=True),
        Parm('groundy', "Ground Height", 'float', default=0.0),
        Parm('gravity', "Gravity", 'float', default=9.81),
        Parm('friction', "Friction", 'float', default=0.5, min=0, max=1),
        Parm('bounce', "Bounce", 'float', default=0.1, min=0, max=1),
        Parm('density', "Density", 'float', default=1.0, min=0.001),
        Parm('substeps', "Substeps per Frame", 'int', default=10, min=1, max=100),
        Parm('shrink', "Collision Shrink", 'float', default=0.97, min=0.5, max=1.0,
             description="Shrink collision shapes so pieces that touch at the start don't push each other apart"),
        Parm('margin', "Collision Margin", 'float', default=0.001, min=0, soft_max=0.1,
             description="Small margins stop touching pieces (e.g. from a fracture) from pushing apart"),
    )

    def _build(self, pieces, colliders, p, state, start, fps):
        from ..bl.mesh_io import geo_to_mesh
        name = "_blendini_rbd_%x" % id(state)
        scene = bpy.data.scenes.new(name)
        state['scene'] = scene.name
        with bpy.context.temp_override(scene=scene):
            bpy.ops.rigidbody.world_add()
        world = scene.rigidbody_world
        if world.collection is None:
            world.collection = bpy.data.collections.new(name + "_objs")
        # Blendini is Y-up: point gravity along -Y.
        scene.use_gravity = True
        scene.gravity = (0.0, -p.gravity, 0.0)
        scene.render.fps = int(round(fps))
        scene.frame_start = start
        scene.frame_end = start + 100000
        world.point_cache.frame_start = start
        world.point_cache.frame_end = start + 100000
        try:
            world.substeps_per_frame = p.substeps
        except AttributeError:
            pass
        objs = []
        centers = []
        for i, piece in enumerate(pieces):
            c = piece.P.mean(axis=0) if piece.num_points else np.zeros(3)
            local = piece.copy()
            local.P = (local.P - c) * p.shrink
            mesh = bpy.data.meshes.new("%s_%d" % (name, i))
            geo_to_mesh(local, mesh)
            ob = bpy.data.objects.new(mesh.name, mesh)
            ob.location = c
            scene.collection.objects.link(ob)
            _add_rigid_body(scene, ob, 'ACTIVE')
            rb = ob.rigid_body
            rb.collision_shape = 'CONVEX_HULL'
            rb.friction = p.friction
            rb.restitution = p.bounce
            rb.use_margin = True
            rb.collision_margin = p.margin
            vol = _hull_volume(local)
            rb.mass = max(vol * p.density, 1e-4)
            objs.append(ob.name)
            centers.append(c)
        statics = []
        if p.ground:
            from ..core.geometry import Geometry
            gsize = 1000.0
            ground = Geometry.from_polys(
                np.array([[-gsize, p.groundy, -gsize], [gsize, p.groundy, -gsize], [gsize, p.groundy, gsize],
                          [-gsize, p.groundy, gsize]]), [[0, 3, 2, 1]])
            statics.append(ground)
        if colliders is not None and colliders.num_prims:
            statics.append(colliders)
        for j, sgeo in enumerate(statics):
            mesh = bpy.data.meshes.new("%s_static%d" % (name, j))
            geo_to_mesh(sgeo, mesh)
            ob = bpy.data.objects.new(mesh.name, mesh)
            scene.collection.objects.link(ob)
            _add_rigid_body(scene, ob, 'PASSIVE')
            ob.rigid_body.collision_shape = 'MESH' if j > 0 or not p.ground else 'BOX'
            ob.rigid_body.friction = p.friction
            ob.rigid_body.restitution = p.bounce
        state['objects'] = objs
        state['centers'] = np.array(centers)
        state['frames'] = {}
        state['last'] = start - 1
        scene.frame_set(start)

    def cook(self, inputs, p, ctx):
        from ..core.geometry import merge
        from ..core.instances import unpack
        from .modify import split_pieces
        geo = _in(inputs, 0)
        if geo.instances:
            # Packed pieces (Pack / Assemble) are simulated as their real geometry.
            geo = unpack(geo)
        colliders = _in(inputs, 1) if len(inputs) > 1 and inputs[1] is not None else None
        state = ctx.state
        frame = int(round(ctx.frame))
        start = int(p.startframe)
        key = (tuple(sorted(p.items())), geo.num_points, geo.num_prims, float(geo.P.sum()) if geo.num_points else 0.0,
               colliders.num_points if colliders is not None else 0)
        if state.get('key') != key or bpy.data.scenes.get(state.get('scene', '')) is None:
            _rbd_cleanup(state)
            state['key'] = key
            pieces = split_pieces(geo, p.pieceattrib.strip())
            state['pieces'] = pieces
            if not pieces:
                return geo
            self._build(pieces, colliders, p, state, start, ctx.fps)
        pieces = state['pieces']
        if frame <= start:
            return geo
        frames = state['frames']
        if frame not in frames:
            scene = bpy.data.scenes[state['scene']]
            if frame < state['last']:
                # Scrubbing backwards past the cache: rebuild.
                _rbd_cleanup(state)
                return self.cook(inputs, p, ctx)
            for f in range(max(state['last'] + 1, start), frame + 1):
                scene.frame_set(f)
                mats = []
                for name in state['objects']:
                    ob = bpy.data.objects[name]
                    mats.append(np.array(ob.matrix_world))
                frames[f] = np.array(mats)
                state['last'] = f
        mats = frames[frame]
        out_pieces = []
        for piece, c, m in zip(pieces, state['centers'], mats):
            g = piece.copy()
            local = g.P - c
            g.P = local @ m[:3, :3].T + m[:3, 3]
            out_pieces.append(g)
        return merge(out_pieces)


def _hull_volume(geo):
    tris, _ = geo.triangulate_flat()
    if len(tris) == 0:
        return 1.0
    A, B, C = geo.P[tris[:, 0]], geo.P[tris[:, 1]], geo.P[tris[:, 2]]
    vol = abs(np.sum(np.einsum('ij,ij->i', A, np.cross(B, C))) / 6.0)
    return vol if vol > 1e-9 else 1.0


def _surface_group():
    name = "Blendini Particle Surface"
    ng = bpy.data.node_groups.get(name)
    if ng is not None:
        return ng
    ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    ng.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    for sock_name, default in (("Voxel Size", 0.05), ("Radius", 0.1), ("Threshold", 0.1)):
        s = ng.interface.new_socket(name=sock_name, in_out='INPUT', socket_type='NodeSocketFloat')
        s.default_value = default
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N, L = ng.nodes, ng.links
    gin = N.new('NodeGroupInput')
    gout = N.new('NodeGroupOutput')
    to_pts = N.new('GeometryNodeMeshToPoints')
    to_vol = N.new('GeometryNodePointsToVolume')
    try:
        to_vol.resolution_mode = 'VOXEL_SIZE'
    except (AttributeError, TypeError):
        pass
    to_mesh = N.new('GeometryNodeVolumeToMesh')
    L.new(gin.outputs[0], to_pts.inputs['Mesh'])
    L.new(to_pts.outputs[0], to_vol.inputs['Points'])
    L.new(gin.outputs['Voxel Size'], to_vol.inputs['Voxel Size'])
    L.new(gin.outputs['Radius'], to_vol.inputs['Radius'])
    L.new(to_vol.outputs[0], to_mesh.inputs['Volume'])
    L.new(gin.outputs['Threshold'], to_mesh.inputs['Threshold'])
    L.new(to_mesh.outputs[0], gout.inputs[0])
    return ng


@register_sop
class ParticleFluidSurfaceSOP(SOP):
    id = 'particlefluidsurface'
    label = "Particle Fluid Surface"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_OCEAN'
    description = "Mesh a point cloud (e.g. fluid particles) through an OpenVDB volume"
    parms = (
        Parm('particlesep', "Particle Separation", 'float', default=0.08, min=0.001, soft_max=1),
        Parm('radiusscale', "Influence Scale", 'float', default=1.6, min=0.1, soft_max=4),
        Parm('voxelscale', "Voxel Scale", 'float', default=0.5, min=0.05, soft_max=2),
        Parm('threshold', "Iso Threshold", 'float', default=0.3, min=0.0, soft_max=2),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        if geo.num_points == 0:
            return Geometry()
        pts = Geometry()
        pts.P = geo.P
        ng = _surface_group()

        def settings(mod, _):
            mod.node_group = ng
            vals = {"Voxel Size": p.particlesep * p.voxelscale, "Radius": p.particlesep * p.radiusscale,
                    "Threshold": p.threshold}
            for name, value in vals.items():
                _set_group_input(mod, name, float(value))
        return apply_modifier(pts, 'NODES', settings)
