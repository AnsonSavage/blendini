# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Show native volumes as a real Blender Volume object (renders as smoke in Cycles / EEVEE).

Fog and vector volumes are written to an OpenVDB file in Blender's temporary directory (with
the openvdb module Blender ships) and loaded by a Volume object parented to the network's
output object. Signed distance volumes are shown as their surface in the mesh instead.
"""

import itertools
import os
import sys

import bpy
import numpy as np

_counter = itertools.count()
MATERIAL = "Blendini Volume"


def openvdb():
    try:
        import openvdb as vdb
        return vdb
    except ImportError:
        pass
    try:
        loc = bpy.utils.resource_path('LOCAL')
        path = os.path.join(loc, 'python', 'lib', 'python%d.%d' % sys.version_info[:2], 'site-packages')
        if os.path.isdir(path) and path not in sys.path:
            sys.path.append(path)
        import openvdb as vdb
        return vdb
    except ImportError:
        return None


def renderable(geo):
    return [v for v in getattr(geo, 'volumes', ()) if v.kind != 'sdf']


def write_vdb(volumes, path):
    vdb = openvdb()
    if vdb is None:
        return False
    grids = []
    for v in volumes:
        data = np.ascontiguousarray(v.data, dtype=np.float32)
        g = vdb.Vec3SGrid() if v.is_vector else vdb.FloatGrid()
        g.copyFromArray(data)
        g.name = v.name
        vs = v.voxelsize
        t = v.origin + 0.5 * vs
        g.transform = vdb.createLinearTransform([[vs, 0, 0, 0], [0, vs, 0, 0], [0, 0, vs, 0],
                                                 [float(t[0]), float(t[1]), float(t[2]), 1]])
        grids.append(g)
    vdb.write(path, grids=grids)
    return True


def _material():
    mat = bpy.data.materials.get(MATERIAL)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(MATERIAL)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    pv = nt.nodes.new('ShaderNodeVolumePrincipled')
    pv.location = (-300, 0)
    try:
        pv.inputs['Density'].default_value = 1.0
        pv.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)
    except (KeyError, AttributeError):
        pass
    nt.links.new(pv.outputs[0], out.inputs['Volume'])
    return mat


def sync_volumes(tree, obj, geo):
    """Create / update / remove the volume object. Returns True when fog volumes are shown by it."""
    name = obj.name + "_volume"
    vols = renderable(geo)
    vob = bpy.data.objects.get(name)
    if not vols:
        if vob is not None:
            data = vob.data
            bpy.data.objects.remove(vob)
            if data is not None and data.users == 0:
                bpy.data.volumes.remove(data)
        return False
    if openvdb() is None:
        return False
    folder = bpy.app.tempdir or os.path.join(os.path.expanduser('~'), '.blendini_cache')
    os.makedirs(folder, exist_ok=True)
    safe = "".join(c if c.isalnum() else '_' for c in tree.name)
    path = os.path.join(folder, "blendini_%s_%d.vdb" % (safe, next(_counter)))
    try:
        if not write_vdb(vols, path):
            return False
    except (RuntimeError, TypeError, ValueError, OSError) as ex:
        print("Blendini: couldn't write volumes:", ex)
        return False
    if vob is None:
        data = bpy.data.volumes.new(name)
        vob = bpy.data.objects.new(name, data)
        for coll in obj.users_collection:
            coll.objects.link(vob)
        vob.parent = obj
        vob.hide_select = True
        data.materials.append(_material())
    old = vob.data.filepath
    vob.data.filepath = path
    try:
        vob.data.grids.load()
    except RuntimeError:
        pass
    if old and old != path and os.path.basename(old).startswith("blendini_") and os.path.isfile(old):
        try:
            os.remove(old)
        except OSError:
            pass
    return True
