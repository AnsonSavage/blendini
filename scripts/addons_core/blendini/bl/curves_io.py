# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Render polylines like Houdini does: as curves with a width.

Open polylines of the displayed geometry are written to a Blender Curves (hair) object parented
to the network's output object. The radius comes from the 'width' point attribute (diameter,
Houdini's convention) and defaults to 0.01. Cd is copied so the Blendini material colors them.
"""

import bpy
import numpy as np

from ..core.geometry import POINT

DEFAULT_WIDTH = 0.02


def polyline_prims(geo):
    sizes = geo.prim_sizes()
    return np.nonzero(~geo.prim_closed & (sizes >= 2))[0]


def sync_curves(tree, obj, geo):
    name = obj.name + "_curves"
    cob = bpy.data.objects.get(name)
    prims = polyline_prims(geo) if geo.num_prims else np.zeros(0, dtype=np.int64)
    if not len(prims):
        if cob is not None:
            data = cob.data
            bpy.data.objects.remove(cob)
            if data is not None and data.users == 0:
                bpy.data.hair_curves.remove(data)
        return False
    sizes = geo.prim_sizes()[prims]
    pts = np.concatenate([geo.prim_points(int(i)) for i in prims])
    if cob is None:
        data = bpy.data.hair_curves.new(name)
        cob = bpy.data.objects.new(name, data)
        for coll in obj.users_collection:
            coll.objects.link(cob)
        cob.parent = obj
        cob.hide_select = True
        from .display import cd_material
        data.materials.append(cd_material())
    curves = cob.data
    curves.remove_curves()
    curves.add_curves(sizes.tolist())
    curves.position_data.foreach_set('vector', geo.P[pts].astype(np.float32).ravel())
    width = geo.attribs[POINT].get('width')
    if width is not None and width.dtype != object and width.ndim == 1:
        radius = width[pts] * 0.5
    else:
        radius = np.full(len(pts), DEFAULT_WIDTH * 0.5)
    attr = curves.attributes.get('radius') or curves.attributes.new('radius', 'FLOAT', 'POINT')
    attr.data.foreach_set('value', radius.astype(np.float32))
    cd = geo.attribs[POINT].get('Cd')
    old = curves.attributes.get('Cd')
    if cd is not None and cd.dtype != object and cd.ndim == 2 and cd.shape[1] >= 3:
        if old is None:
            old = curves.attributes.new('Cd', 'FLOAT_COLOR', 'POINT')
        rgba = np.concatenate([cd[pts, :3], np.ones((len(pts), 1))], axis=1).astype(np.float32)
        old.data.foreach_set('color', rgba.ravel())
    elif old is not None:
        curves.attributes.remove(old)
    curves.update_tag()
    return True
