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
        from ..core.geometry import merge
        import fnmatch
        patterns = p.objpath.replace(',', ' ').split()
        if not patterns:
            return Geometry()
        if len(patterns) == 1 and bpy.data.objects.get(patterns[0]) is not None:
            return self._read(bpy.data.objects[patterns[0]], p)
        objs = [o for o in bpy.data.objects if any(fnmatch.fnmatchcase(o.name, pat) for pat in patterns)
                and not o.name.startswith(('.', '_blendini'))]
        if not objs:
            raise CookError("No object matches '%s'" % p.objpath)
        pieces = []
        for o in objs:
            g = self._read(o, p)
            g.attribs[PRIM_CLASS]['name'] = np.array([o.name] * g.num_prims, dtype=object)
            pieces.append(g)
        return merge(pieces)

    def _read(self, obj, p):
        from ..bl.mesh_io import mesh_to_geo
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


PRIM_CLASS = 'prim'


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
    """Remove a private simulation scene and everything it owns.

    Order matters to avoid crashes inside Bullet: glue constraints are freed while their bodies
    still exist, then the rigid body world, then the scene, objects, meshes and collections in
    a single batch removal.
    """
    scene = bpy.data.scenes.get(state.get('scene', ''))
    if scene is not None:
        objs = list(scene.collection.all_objects)
        rbw = scene.rigidbody_world
        colls = [c for c in ((rbw.collection, rbw.constraints) if rbw is not None else ()) if c is not None]
        if rbw is not None:
            # Constraints first: freeing one dereferences the Bullet bodies of its two objects,
            # which removing the world (or an object) frees.
            for ob in objs:
                if ob.rigid_body_constraint is not None:
                    with bpy.context.temp_override(scene=scene, object=ob, active_object=ob):
                        bpy.ops.rigidbody.constraint_remove()
            with bpy.context.temp_override(scene=scene):
                bpy.ops.rigidbody.world_remove()
        meshes = {ob.data for ob in objs if ob.data is not None and ob.data.users <= 1}
        ids = [scene] + objs + list(meshes) + [c for c in colls if c.users <= 1]
        bpy.data.batch_remove(ids)
    state.clear()


def _add_rigid_body(scene, ob, kind):
    with bpy.context.temp_override(scene=scene, object=ob, active_object=ob, selected_objects=[ob],
                                   selected_editable_objects=[ob]):
        bpy.ops.rigidbody.object_add(type=kind)


CONSTRAINT_TYPES = [
    ('glue', "Glue (breakable fixed)", ""), ('fixed', "Hard / Fixed", ""), ('point', "Point (ball joint)", ""),
    ('hinge', "Hinge", "Rotates around the primitive's 'axis' attribute (default Z)"),
    ('slider', "Slider", ""), ('spring', "Spring", "'stiffness' and 'damping' primitive attributes"),
]
_BLENDER_CONSTRAINT = {'glue': 'FIXED', 'fixed': 'FIXED', 'hard': 'FIXED', 'point': 'POINT', 'ball': 'POINT',
                       'hinge': 'HINGE', 'slider': 'SLIDER', 'spring': 'GENERIC_SPRING', 'soft': 'GENERIC_SPRING'}


def _add_constraint(scene, name, ob1, ob2, kind, location, strength=-1.0, axis=None, stiffness=50.0, damping=0.5):
    """Rigid body constraint (an empty in the simulation scene) between two objects."""
    from mathutils import Vector
    empty = bpy.data.objects.new(name, None)
    empty.location = location
    if axis is not None and Vector(axis).length > 1e-9:
        empty.rotation_mode = 'QUATERNION'
        empty.rotation_quaternion = Vector(axis).normalized().to_track_quat('Z', 'Y')
    scene.collection.objects.link(empty)
    btype = _BLENDER_CONSTRAINT.get(kind, 'FIXED')
    with bpy.context.temp_override(scene=scene, object=empty, active_object=empty, selected_objects=[empty],
                                   selected_editable_objects=[empty]):
        bpy.ops.rigidbody.constraint_add(type=btype)
    c = empty.rigid_body_constraint
    c.object1 = ob1
    c.object2 = ob2
    if kind == 'glue' and strength <= 0:
        strength = 20.0
    if strength > 0:
        c.use_breaking = True
        c.breaking_threshold = strength
    if btype == 'GENERIC_SPRING':
        try:
            c.spring_type = 'SPRING2'
        except (TypeError, AttributeError):
            pass
        for ax in 'xyz':
            setattr(c, 'use_spring_' + ax, True)
            setattr(c, 'spring_stiffness_' + ax, stiffness)
            setattr(c, 'spring_damping_' + ax, damping)
            setattr(c, 'use_spring_ang_' + ax, True)
            setattr(c, 'spring_stiffness_ang_' + ax, stiffness)
            setattr(c, 'spring_damping_ang_' + ax, damping)
    return empty


def constraint_pairs(network, names, centers, default_type='glue'):
    """Read a constraint network: 2-point polylines whose end points name pieces (point 'name'
    attribute, else the nearest piece center). Returns dicts with a, b (piece index or -1 for
    the world), type, strength, axis, stiffness, damping, location."""
    from ..core.geometry import POINT, PRIM
    out = []
    if network is None or network.num_prims == 0 or not len(centers):
        return out
    pnames = network.attribs[POINT].get('name')
    index = {str(n): i for i, n in enumerate(names)}
    prima = network.attribs[PRIM]

    def piece_of(pt):
        if pnames is not None:
            nm = str(pnames[pt])
            if nm in index:
                return index[nm]
            if nm in ('', 'world', '__world__'):
                return -1
        d = np.linalg.norm(np.asarray(centers) - network.P[pt], axis=1)
        return int(np.argmin(d))

    def prim_value(name, i, default):
        arr = prima.get(name)
        if arr is None:
            return default
        v = arr[i]
        return v if arr.dtype == object else (v.tolist() if np.ndim(v) else float(v))
    for i in range(network.num_prims):
        pts = network.prim_points(i)
        for k in range(len(pts) - 1):
            a, b = piece_of(pts[k]), piece_of(pts[k + 1])
            if a == b:
                continue
            if a < 0 or b < 0:
                # Pinned to the world: the joint sits at the world point.
                location = network.P[pts[k] if a < 0 else pts[k + 1]].copy()
            else:
                location = (network.P[pts[k]] + network.P[pts[k + 1]]) / 2
            out.append(dict(a=a, b=b, type=str(prim_value('constraint_type', i, default_type)).lower(),
                            strength=float(prim_value('strength', i, -1.0)),
                            axis=prim_value('axis', i, None), stiffness=float(prim_value('stiffness', i, 50.0)),
                            damping=float(prim_value('damping', i, 0.5)), location=location))
    return out


@register_sop
class RBDSolverSOP(SOP):
    id = 'rbdsolver'
    label = "RBD Bullet Solver"
    category = 'Simulation'
    inputs = 3
    input_labels = ("Pieces", "Static Colliders", "Constraint Network")
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
        Parm('glue', "Glue Neighbouring Pieces", 'bool', default=False,
             description="Connect touching pieces with breakable constraints (Houdini glue network)"),
        Parm('gluestrength', "Glue Strength", 'float', default=20.0, min=0.0, soft_max=500.0,
             hide_when={'glue': False}, description="Impulse needed to break a glue bond"),
        Parm('gluepartners', "Max Partners", 'int', default=6, min=1, max=20, hide_when={'glue': False}),
        Parm('shrink', "Collision Shrink", 'float', default=0.97, min=0.5, max=1.0,
             description="Shrink collision shapes so pieces that touch at the start don't push each other apart"),
        Parm('margin', "Collision Margin", 'float', default=0.001, min=0, soft_max=0.1,
             description="Small margins stop touching pieces (e.g. from a fracture) from pushing apart"),
        Parm('contype', "Default Constraint Type", 'enum', items=CONSTRAINT_TYPES,
             description="Used for constraint network polylines without a 'constraint_type' attribute"),
    )

    @staticmethod
    def _bake_end(start):
        scene = bpy.context.scene
        return min(scene.frame_end, start + 500) if scene is not None else start

    @staticmethod
    def _bake(state, start, target):
        """Step the private simulation scene up to target, caching every frame's transforms."""
        scene = bpy.data.scenes.get(state.get('scene', ''))
        if scene is None:
            return
        frames = state['frames']
        for f in range(max(state['last'] + 1, start), target + 1):
            scene.frame_set(f)
            frames[f] = np.array([np.array(bpy.data.objects[name].matrix_world) for name in state['objects']])
            state['last'] = f

    def _build(self, pieces, colliders, p, state, start, fps, network=None, names=()):
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
        if p.glue and len(objs) > 1:
            self._glue(scene, objs, np.array(centers), p)
        if network is not None and network.num_prims:
            self._network(scene, objs, np.array(centers), list(names), network, p)
        state['objects'] = objs
        state['centers'] = np.array(centers)
        state['frames'] = {}
        state['last'] = start - 1
        scene.frame_set(start)

    def _glue(self, scene, objs, centers, p):
        """Breakable FIXED constraints between neighbouring pieces."""
        n = len(objs)
        d = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        nearest = d.min(axis=1)
        radius = 1.6 * float(np.median(nearest[np.isfinite(nearest)]))
        k = min(max(int(p.gluepartners), 1), n - 1)
        pairs = set()
        for i in range(n):
            for j in np.argsort(d[i])[:k]:
                if d[i, j] <= radius:
                    pairs.add((min(i, int(j)), max(i, int(j))))
        for a, b in sorted(pairs):
            _add_constraint(scene, "%s_glue_%d_%d" % (scene.name, a, b), bpy.data.objects[objs[a]],
                            bpy.data.objects[objs[b]], 'glue', (centers[a] + centers[b]) / 2,
                            strength=max(p.gluestrength, 1e-6))
        return len(pairs)

    def _network(self, scene, objs, centers, names, network, p):
        """Constraints from a constraint network geometry (third input)."""
        anchors = {}
        count = 0
        for k, c in enumerate(constraint_pairs(network, names, centers, p.contype)):
            ends = []
            for idx in (c['a'], c['b']):
                if idx >= 0:
                    ends.append(bpy.data.objects[objs[idx]])
                    continue
                # World anchor: a tiny passive body that doesn't collide with the pieces.
                key = tuple(np.round(c['location'], 6))
                if key not in anchors:
                    mesh = bpy.data.meshes.new("%s_anchor%d" % (scene.name, len(anchors)))
                    ob = bpy.data.objects.new(mesh.name, mesh)
                    ob.location = c['location']
                    scene.collection.objects.link(ob)
                    _add_rigid_body(scene, ob, 'PASSIVE')
                    ob.rigid_body.collision_shape = 'SPHERE'
                    groups = [False] * 20
                    groups[19] = True
                    ob.rigid_body.collision_collections = groups
                    anchors[key] = ob
                ends.append(anchors[key])
            _add_constraint(scene, "%s_con_%d" % (scene.name, k), ends[0], ends[1], c['type'], c['location'],
                            strength=c['strength'], axis=c['axis'], stiffness=c['stiffness'], damping=c['damping'])
            count += 1
        return count

    def cook(self, inputs, p, ctx):
        from ..core.geometry import merge
        from ..core.instances import unpack
        from .modify import split_pieces
        geo = _in(inputs, 0)
        if geo.instances:
            # Packed pieces (Pack / Assemble) are simulated as their real geometry.
            geo = unpack(geo)
        colliders = _in(inputs, 1) if len(inputs) > 1 and inputs[1] is not None else None
        network = _in(inputs, 2) if len(inputs) > 2 and inputs[2] is not None else None
        state = ctx.state
        frame = int(round(ctx.frame))
        start = int(p.startframe)
        key = (tuple(sorted(p.items())), geo.num_points, geo.num_prims, float(geo.P.sum()) if geo.num_points else 0.0,
               colliders.num_points if colliders is not None else 0,
               (network.num_prims, float(network.P.sum())) if network is not None and network.num_points else None)
        from ..bl.cook import in_frame_handler
        deferred = in_frame_handler()
        if state.get('key') != key or bpy.data.scenes.get(state.get('scene', '')) is None:
            if deferred:
                # Building the simulation scene inside a frame change is unsafe: do it right after.
                ctx.uncacheable = True
                return geo
            _rbd_cleanup(state)
            state['key'] = key
            pieces = split_pieces(geo, p.pieceattrib.strip())
            state['pieces'] = pieces
            if not pieces:
                return geo
            from .modify import piece_names
            self._build(pieces, colliders, p, state, start, ctx.fps, network, piece_names(pieces, p.pieceattrib.strip()))
            # Bake ahead right away, so frames reached later from the frame change handler
            # (playback, renders) are already in the cache.
            self._bake(state, start, max(frame, self._bake_end(start)))
        pieces = state['pieces']
        if frame <= start:
            return geo
        frames = state['frames']
        if frame not in frames and deferred:
            # Stepping the simulation scene inside a frame change is unsafe: show the closest
            # cached frame now and simulate right after the frame change.
            ctx.uncacheable = True
            cached = [f for f in frames if f <= frame]
            if not cached:
                return geo
            frame = max(cached)
        if frame not in frames:
            if frame < state['last']:
                # Scrubbing backwards past the cache: rebuild.
                _rbd_cleanup(state)
                return self.cook(inputs, p, ctx)
            self._bake(state, start, max(frame, self._bake_end(start)))
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


@register_sop
class UVUnwrapSOP(SOP):
    id = 'uvunwrap'
    label = "UV Unwrap"
    category = 'Attribute'
    inputs = 1
    icon = 'UV'
    description = "UV unwrap with Blender's Smart UV Project / Unwrap (writes vertex uv)"
    parms = (
        Parm('method', "Method", 'enum', items=[('smart', "Smart UV Project", ""), ('unwrap', "Angle Based Unwrap", ""),
                                                ('cube', "Cube Projection", "")]),
        Parm('angle', "Angle Limit", 'float', default=66.0, min=1, max=89, hide_when={'method': ('unwrap', 'cube')}),
        Parm('margin', "Island Margin", 'float', default=0.02, min=0, max=1),
    )

    def cook(self, inputs, p, ctx):
        import math
        from ..bl.mesh_io import geo_to_mesh, mesh_to_geo
        from ..core.geometry import VERTEX
        geo = _in(inputs)
        if geo.num_prims == 0:
            return geo
        coll = _temp_collection()
        mesh = bpy.data.meshes.new("_blendini_uv")
        geo_to_mesh(geo, mesh)
        if mesh.uv_layers.get('uv') is None:
            mesh.uv_layers.new(name='uv')
        mesh.uv_layers.active = mesh.uv_layers['uv']
        obj = bpy.data.objects.new("_blendini_uv", mesh)
        coll.objects.link(obj)
        view_layer = bpy.context.view_layer
        prev_active = view_layer.objects.active
        prev_selected = [o for o in view_layer.objects if o.select_get()]
        try:
            for o in prev_selected:
                o.select_set(False)
            view_layer.objects.active = obj
            obj.select_set(True)
            with bpy.context.temp_override(active_object=obj, object=obj, selected_objects=[obj],
                                           selected_editable_objects=[obj]):
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                if p.method == 'smart':
                    bpy.ops.uv.smart_project(angle_limit=math.radians(p.angle), island_margin=p.margin)
                elif p.method == 'unwrap':
                    bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=p.margin)
                else:
                    bpy.ops.uv.cube_project()
                bpy.ops.object.mode_set(mode='OBJECT')
            out = geo.copy()
            n = len(mesh.loops)
            uv = np.empty(n * 2, dtype=np.float32)
            mesh.uv_layers['uv'].data.foreach_get("uv", uv)
            # Loops follow the face order of the polygons that were written.
            sizes = geo.prim_sizes()
            face_mask = geo.prim_closed & (sizes >= 3)
            vmask = np.repeat(face_mask, sizes)
            full = np.zeros((geo.num_vertices, 3))
            if vmask.sum() == n:
                full[vmask, :2] = uv.reshape(-1, 2)
            out.attribs[VERTEX]['uv'] = full
            return out
        except RuntimeError as ex:
            raise CookError("UV unwrap failed: %s" % ex)
        finally:
            if obj.mode != 'OBJECT':
                try:
                    with bpy.context.temp_override(active_object=obj, object=obj):
                        bpy.ops.object.mode_set(mode='OBJECT')
                except RuntimeError:
                    pass
            bpy.data.objects.remove(obj)
            bpy.data.meshes.remove(mesh)
            _release_temp_collection()
            for o in prev_selected:
                if view_layer.objects.get(o.name) is not None:
                    o.select_set(True)
            if prev_active is not None and view_layer.objects.get(prev_active.name) is not None:
                view_layer.objects.active = prev_active


# ----------------------------------------------------------------------------
# OpenVDB signed distance fields (Blender 5 Geometry Nodes grid nodes)


def _new_sdf_group(name):
    ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    ng.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    return ng


def _sdf_node(ng, kind):
    try:
        return ng.nodes.new(kind)
    except RuntimeError:
        bpy.data.node_groups.remove(ng)
        raise CookError("This Blender has no '%s' node (OpenVDB SDF nodes need Blender 5.0+)" % kind)


def _run_group(geo, ng, extra=()):
    def settings(mod, others):
        mod.node_group = ng
        for node in ng.nodes:
            if node.bl_idname == 'GeometryNodeObjectInfo' and others:
                node.inputs['Object'].default_value = others[0]
    try:
        return apply_modifier(geo, 'NODES', settings, extra_objects=extra)
    finally:
        bpy.data.node_groups.remove(ng)


@register_sop
class VDBReshapeSOP(SOP):
    id = 'vdbreshape'
    label = "VDB Reshape / Smooth SDF"
    category = 'Polygon'
    inputs = 1
    icon = 'VOLUME_DATA'
    description = "Polygons -> signed distance field -> dilate/erode, smooth, fillet -> polygons (OpenVDB)"
    parms = (
        Parm('voxelsize', "Voxel Size", 'float', default=0.05, min=0.001, soft_max=1),
        Parm('offset', "Offset (+dilate / -erode)", 'float', default=0.0, soft_min=-1, soft_max=1),
        Parm('filter', "Smooth Filter", 'enum', items=[('mean', "Mean", ""), ('laplacian', "Laplacian Flow", ""),
                                                      ('median', "Median", ""), ('none', "None", "")]),
        Parm('iterations', "Smooth Iterations", 'int', default=2, min=0, max=50, hide_when={'filter': 'none'}),
        Parm('width', "Filter Width (voxels)", 'int', default=1, min=1, max=10, hide_when={'filter': ('laplacian', 'none')}),
        Parm('fillet', "Fillet Iterations", 'int', default=0, min=0, max=50,
             description="Round off concave corners"),
        Parm('adaptivity', "Adaptivity", 'float', default=0.0, min=0, max=1),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        if geo.num_prims == 0:
            return Geometry()
        lo, hi = geo.bbox()
        cells = np.prod(np.maximum((hi - lo) / max(p.voxelsize, 1e-4), 1))
        if cells > 64e6:
            raise CookError("Voxel size too small for this geometry (%.0fM voxels)" % (cells / 1e6))
        ng = _new_sdf_group("_blendini_vdb")
        gin = next(n for n in ng.nodes if n.bl_idname == 'NodeGroupInput') if any(
            n.bl_idname == 'NodeGroupInput' for n in ng.nodes) else ng.nodes.new('NodeGroupInput')
        gout = ng.nodes.new('NodeGroupOutput')
        to_sdf = _sdf_node(ng, 'GeometryNodeMeshToSDFGrid')
        to_sdf.inputs['Voxel Size'].default_value = max(p.voxelsize, 1e-4)
        to_sdf.inputs['Band Width'].default_value = int(max(3, p.iterations * p.width + 3 + abs(p.offset) / max(p.voxelsize, 1e-4)))
        ng.links.new(gin.outputs[0], to_sdf.inputs['Mesh'])
        last = to_sdf.outputs[0]
        if p.offset:
            off = _sdf_node(ng, 'GeometryNodeSDFGridOffset')
            off.inputs['Distance'].default_value = p.offset
            ng.links.new(last, off.inputs['Grid'])
            last = off.outputs[0]
        if p.filter != 'none' and p.iterations > 0:
            kind = {'mean': 'GeometryNodeSDFGridMean', 'laplacian': 'GeometryNodeSDFGridLaplacian',
                    'median': 'GeometryNodeSDFGridMedian'}[p.filter]
            f = _sdf_node(ng, kind)
            f.inputs['Iterations'].default_value = p.iterations
            if 'Width' in f.inputs:
                f.inputs['Width'].default_value = p.width
            ng.links.new(last, f.inputs['Grid'])
            last = f.outputs[0]
        if p.fillet > 0:
            fil = _sdf_node(ng, 'GeometryNodeSDFGridFillet')
            fil.inputs['Iterations'].default_value = p.fillet
            ng.links.new(last, fil.inputs['Grid'])
            last = fil.outputs[0]
        to_mesh = _sdf_node(ng, 'GeometryNodeGridToMesh')
        to_mesh.inputs['Threshold'].default_value = 0.0
        to_mesh.inputs['Adaptivity'].default_value = p.adaptivity
        ng.links.new(last, to_mesh.inputs['Grid'])
        ng.links.new(to_mesh.outputs[0], gout.inputs[0])
        return _run_group(geo, ng)


@register_sop
class VDBCombineSOP(SOP):
    id = 'vdbcombine'
    label = "VDB Combine (SDF Boolean)"
    category = 'Polygon'
    inputs = 2
    input_labels = ("A", "B")
    icon = 'MOD_BOOLEAN'
    description = "Voxel based boolean of two closed surfaces through OpenVDB signed distance fields"
    parms = (
        Parm('operation', "Operation", 'enum', items=[('UNION', "Union", ""), ('INTERSECT', "Intersect", ""),
                                                     ('DIFFERENCE', "Subtract (A - B)", "")]),
        Parm('voxelsize', "Voxel Size", 'float', default=0.05, min=0.001, soft_max=1),
        Parm('smooth', "Smooth Iterations", 'int', default=0, min=0, max=20),
    )

    def cook(self, inputs, p, ctx):
        a, b = _in(inputs, 0), _in(inputs, 1)
        if a.num_prims == 0 or b.num_prims == 0:
            return a.copy() if b.num_prims == 0 else (b.copy() if p.operation == 'UNION' else Geometry())
        ng = _new_sdf_group("_blendini_vdbcombine")
        gin = ng.nodes.new('NodeGroupInput')
        gout = ng.nodes.new('NodeGroupOutput')
        info = ng.nodes.new('GeometryNodeObjectInfo')
        info.transform_space = 'RELATIVE'
        sdf_a = _sdf_node(ng, 'GeometryNodeMeshToSDFGrid')
        sdf_b = _sdf_node(ng, 'GeometryNodeMeshToSDFGrid')
        for s in (sdf_a, sdf_b):
            s.inputs['Voxel Size'].default_value = max(p.voxelsize, 1e-4)
            s.inputs['Band Width'].default_value = 3 + p.smooth
        boolean = _sdf_node(ng, 'GeometryNodeSDFGridBoolean')
        try:
            boolean.operation = p.operation
        except (AttributeError, TypeError):
            pass
        ng.links.new(gin.outputs[0], sdf_a.inputs['Mesh'])
        ng.links.new(info.outputs['Geometry'], sdf_b.inputs['Mesh'])
        enabled = [s for s in boolean.inputs if s.enabled]
        # Difference: 'Grid 1' minus the multi-input; union / intersect: one multi-input for both.
        ng.links.new(sdf_a.outputs[0], enabled[0])
        ng.links.new(sdf_b.outputs[0], enabled[-1])
        last = boolean.outputs[0]
        if p.smooth > 0:
            f = _sdf_node(ng, 'GeometryNodeSDFGridMean')
            f.inputs['Iterations'].default_value = p.smooth
            ng.links.new(last, f.inputs['Grid'])
            last = f.outputs[0]
        to_mesh = _sdf_node(ng, 'GeometryNodeGridToMesh')
        to_mesh.inputs['Threshold'].default_value = 0.0  # the zero level set is the surface
        ng.links.new(last, to_mesh.inputs['Grid'])
        ng.links.new(to_mesh.outputs[0], gout.inputs[0])
        return _run_group(a, ng, extra=[b])


@register_sop
class PolyFillSOP(SOP):
    id = 'polyfill'
    label = "PolyFill"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_TRIANGULATE'
    description = "Fill holes (open boundary loops) with polygons"
    parms = (
        Parm('maxsides', "Max Hole Sides (0 = any)", 'int', default=0, min=0),
        Parm('triangulate', "Triangulate Fill", 'bool', default=False),
        Parm('fillgroup', "Fill Group", 'string', default='fill'),
    )

    def cook(self, inputs, p, ctx):
        from ..core.geometry import PRIM
        geo = _in(inputs)
        if geo.num_prims == 0:
            return geo
        n_before = geo.num_prims

        def fn(bm):
            res = bmesh.ops.holes_fill(bm, edges=bm.edges[:], sides=p.maxsides)
            new_faces = [f for f in res.get('faces', []) if f.is_valid]
            if p.triangulate and new_faces:
                bmesh.ops.triangulate(bm, faces=new_faces)
            # New faces are appended after the original ones: flag them for the fill group.
            bm.faces.index_update()
            layer = bm.faces.layers.int.new('__fill')
            for f in bm.faces:
                f[layer] = 1 if f.index >= n_before else 0
        out = _with_bmesh(geo, fn)
        flag = out.attribs[PRIM].pop('__fill', None)
        if p.fillgroup and flag is not None:
            out.groups[PRIM][p.fillgroup] = flag.astype(bool)
        return out


@register_sop
class DissolveSOP(SOP):
    id = 'dissolve'
    label = "Dissolve (Limited)"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_DECIM'
    description = "Merge coplanar faces and remove inline points (Blender limited dissolve)"
    parms = (
        Parm('angle', "Max Angle (degrees)", 'float', default=5.0, min=0, max=180),
        Parm('boundaries', "Delimit by Boundaries", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        import math
        geo = _in(inputs)
        if geo.num_prims == 0:
            return geo

        def fn(bm):
            bmesh.ops.dissolve_limit(bm, angle_limit=math.radians(p.angle), use_dissolve_boundaries=p.boundaries,
                                     verts=bm.verts[:], edges=bm.edges[:])
        return _with_bmesh(geo, fn)


@register_sop
class PolyBridgeSOP(SOP):
    id = 'polybridge'
    label = "PolyBridge"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_LATTICE'
    description = "Bridge open edge loops (e.g. two circles or two holes) with faces"
    parms = (
        Parm('divisions', "Divisions", 'int', default=0, min=0, max=64),
        Parm('twist', "Twist Offset", 'int', default=0),
        Parm('mergecaps', "Merge Bridged Polygons", 'bool', default=True,
             description="Remove closed polygons whose edges were bridged (open the tube ends)"),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        if geo.num_points == 0:
            return geo

        def fn(bm):
            if p.mergecaps:
                faces = [f for f in bm.faces]
                if faces and len(faces) <= 2:
                    # Two caps (e.g. two closed circles): bridge their outlines instead.
                    bmesh.ops.delete(bm, geom=faces, context='FACES_ONLY')
            edges = [e for e in bm.edges if e.is_boundary or e.is_wire]
            if not edges:
                raise CookError("PolyBridge needs open boundaries or curves to bridge")
            try:
                res = bmesh.ops.bridge_loops(bm, edges=edges, use_pairs=False, use_cyclic=False,
                                             twist_offset=p.twist)
            except RuntimeError as ex:
                raise CookError("Can't bridge: %s" % ex)
            if p.divisions > 0 and res.get('edges'):
                bmesh.ops.subdivide_edgering(bm, edges=res['edges'], cuts=p.divisions)
        return _with_bmesh(geo, fn)


@register_sop
class ShrinkwrapSOP(SOP):
    id = 'shrinkwrap'
    label = "Shrinkwrap (Convex Hull)"
    category = 'Polygon'
    inputs = 1
    icon = 'MESH_ICOSPHERE'
    description = "Convex hull of the input points (Houdini's Shrinkwrap SOP)"

    def cook(self, inputs, p, ctx):
        geo = _in(inputs)
        if geo.num_points < 4:
            raise CookError("Shrinkwrap needs at least 4 points")
        pts = Geometry()
        pts.P = geo.P

        def fn(bm):
            res = bmesh.ops.convex_hull(bm, input=bm.verts[:], use_existing_faces=False)
            # Drop interior points that the hull didn't use.
            unused = [v for v in res.get('geom_interior', []) if isinstance(v, bmesh.types.BMVert)]
            unused += [v for v in bm.verts if not v.link_faces]
            if unused:
                bmesh.ops.delete(bm, geom=list(set(unused)), context='VERTS')
        return _with_bmesh(pts, fn)


@register_sop
class Triangulate2DSOP(SOP):
    id = 'triangulate2d'
    label = "Triangulate 2D"
    category = 'Polygon'
    inputs = 1
    icon = 'MOD_TRIANGULATE'
    description = "Delaunay triangulation of the points, projected onto a plane"
    parms = (
        Parm('plane', "Projection Plane", 'enum', items=[('ZX', "ZX (ground)", ""), ('XY', "XY", ""), ('YZ', "YZ", "")]),
        Parm('keepprims', "Use Existing Polylines as Constraints", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        from mathutils import Vector
        from mathutils.geometry import delaunay_2d_cdt
        geo = _in(inputs)
        if geo.num_points < 3:
            raise CookError("Triangulate 2D needs at least 3 points")
        axes = {'ZX': (0, 2), 'XY': (0, 1), 'YZ': (1, 2)}[p.plane]
        pts2 = [Vector((float(x[axes[0]]), float(x[axes[1]]))) for x in geo.P]
        edges = []
        if p.keepprims:
            for i in range(geo.num_prims):
                pp = geo.prim_points(i).tolist()
                edges += list(zip(pp[:-1], pp[1:]))
                if geo.prim_closed[i]:
                    edges.append((pp[-1], pp[0]))
        verts, out_edges, faces, orig_verts, _oe, _of = delaunay_2d_cdt(pts2, edges, [], 0, 1e-9)
        # Map output vertices back to input points (CDT may merge duplicates).
        index = [ov[0] if ov else -1 for ov in orig_verts]
        out = geo.copy()
        out._delete_prims_only(np.ones(out.num_prims, dtype=bool))
        tris = [[index[i] for i in f] for f in faces if all(index[i] >= 0 for i in f)]
        if tris:
            out.add_prims(tris)
            # Face winding: make normals point along the plane's normal axis.
            n = out.prim_normals()
            normal_axis = 3 - sum(axes)
            if np.mean(n[:, normal_axis]) < 0:
                from .modify import ReverseSOP
                from . import default_parms
                out = ReverseSOP().cook([out], default_parms('reverse'), ctx)
        return out
