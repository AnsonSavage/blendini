# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Conversion between Blendini geometry and Blender meshes."""

import bpy
import numpy as np

from ..core.geometry import Geometry, POINT, PRIM, VERTEX, DETAIL

_DOMAIN_OUT = {POINT: 'POINT', PRIM: 'FACE', VERTEX: 'CORNER'}
_DOMAIN_IN = {'POINT': POINT, 'FACE': PRIM, 'CORNER': VERTEX}

# Blender attributes that are internal or represent topology.
_SKIP_IN = {'position', 'sharp_face', 'sharp_edge', 'material_index', '.edge_verts', '.corner_vert', '.corner_edge',
            'UVMap', 'normal'}


def geo_to_mesh(geo, mesh):
    """Write geometry into an existing mesh datablock (replacing its contents)."""
    mesh.clear_geometry()
    npts = geo.num_points
    sizes = geo.prim_sizes()
    face_mask = geo.prim_closed & (sizes >= 3)
    face_prims = np.nonzero(face_mask)[0]
    line_prims = np.nonzero(~face_mask & (sizes >= 2))[0]

    mesh.vertices.add(npts)
    if npts:
        mesh.vertices.foreach_set("co", geo.P.astype(np.float32).ravel())

    # Faces
    vmask = np.repeat(face_mask, sizes)
    loop_verts = geo.vtx_point[vmask]
    fsizes = sizes[face_prims]
    if len(face_prims):
        mesh.loops.add(len(loop_verts))
        mesh.loops.foreach_set("vertex_index", loop_verts.astype(np.int32))
        mesh.polygons.add(len(face_prims))
        starts = np.concatenate([[0], np.cumsum(fsizes)[:-1]]).astype(np.int32)
        mesh.polygons.foreach_set("loop_start", starts)

    # Polylines -> loose edges
    if len(line_prims):
        edges = []
        for i in line_prims:
            pts = geo.prim_points(i)
            seg = np.stack([pts[:-1], pts[1:]], axis=1)
            if geo.prim_closed[i] and len(pts) > 2:
                seg = np.concatenate([seg, [[pts[-1], pts[0]]]])
            edges.append(seg)
        edges = np.concatenate(edges).astype(np.int32)
        edges = edges[edges[:, 0] != edges[:, 1]]
        mesh.edges.add(len(edges))
        mesh.edges.foreach_set("vertices", edges.ravel())

    mesh.update(calc_edges=True, calc_edges_loose=True)

    # Attributes
    loop_src = np.nonzero(vmask)[0]
    for cls in (POINT, PRIM, VERTEX):
        for name, arr in geo.attribs[cls].items():
            if cls == POINT and name == 'P':
                continue
            if name.startswith('__') or name.startswith('.'):
                continue
            if cls == PRIM:
                data = arr[face_prims]
            elif cls == VERTEX:
                data = arr[loop_src]
            else:
                data = arr
            if cls == VERTEX and name == 'uv' and data.dtype != object and data.ndim == 2:
                uv = mesh.uv_layers.new(name='uv')
                if uv is not None and len(data):
                    uv.data.foreach_set("uv", data[:, :2].astype(np.float32).ravel())
                continue
            if cls == POINT and name == 'uv' and data.dtype != object and data.ndim == 2:
                uv = mesh.uv_layers.new(name='uv')
                if uv is not None and len(loop_verts):
                    uv.data.foreach_set("uv", data[loop_verts, :2].astype(np.float32).ravel())
                continue
            _write_attr(mesh, name, _DOMAIN_OUT[cls], data)
    for name, arr in geo.attribs[DETAIL].items():
        if name.startswith('__'):
            continue
        # Detail attributes are stored as custom properties on the mesh.
        try:
            val = arr[0]
            if isinstance(val, np.ndarray):
                mesh["detail_" + name] = val.tolist()
            elif isinstance(val, (np.integer, int)):
                mesh["detail_" + name] = int(val)
            elif isinstance(val, (np.floating, float)):
                mesh["detail_" + name] = float(val)
            elif isinstance(val, str):
                mesh["detail_" + name] = val
        except Exception:
            pass
    # Groups -> boolean attributes.
    for name, g in geo.groups[POINT].items():
        _write_attr(mesh, "group_" + name if name in mesh.attributes else name, 'POINT', g)
    for name, g in geo.groups[PRIM].items():
        if name in mesh.attributes:
            name = "group_" + name
        _write_attr(mesh, name, 'FACE', g[face_prims])

    # Display color: make Cd the active color attribute.
    cd = mesh.color_attributes.get('Cd')
    if cd is not None:
        try:
            mesh.color_attributes.active_color = cd
            mesh.color_attributes.render_color_index = mesh.color_attributes.active_color_index
        except Exception:
            pass

    # Normals
    has_n = 'N' in geo.attribs[VERTEX] or 'N' in geo.attribs[POINT]
    if not has_n and len(face_prims):
        # Like Houdini's viewport: smooth below a 60 degree cusp angle.
        try:
            import math
            mesh.set_sharp_from_angle(angle=math.radians(60.0))
        except (AttributeError, TypeError):
            pass
    if 'N' in geo.attribs[VERTEX] and len(loop_src):
        N = geo.attribs[VERTEX]['N'][loop_src]
        _set_custom_normals(mesh, N, per_vertex=False)
    elif 'N' in geo.attribs[POINT] and npts and len(face_prims):
        _set_custom_normals(mesh, geo.attribs[POINT]['N'], per_vertex=True)
    mesh.update()
    return mesh


def _set_custom_normals(mesh, N, per_vertex):
    N = np.asarray(N, dtype=np.float64)
    if N.ndim != 2 or N.shape[1] != 3:
        return
    ln = np.linalg.norm(N, axis=1, keepdims=True)
    N = N / np.where(ln == 0, 1, ln)
    try:
        if per_vertex:
            mesh.normals_split_custom_set_from_vertices(N.tolist())
        else:
            mesh.normals_split_custom_set(N.tolist())
    except Exception:
        pass


def _write_attr(mesh, name, domain, data):
    if name in mesh.attributes:
        try:
            mesh.attributes.remove(mesh.attributes[name])
        except Exception:
            return
    data = np.asarray(data)
    if data.dtype == object:
        if len(data) and not isinstance(data[0], str):
            return  # array attributes are not representable
        attr = _new_attr(mesh, name, 'STRING', domain)
        if attr is None:
            return
        for i, s in enumerate(data):
            attr.data[i].value = str(s)
        return
    if data.dtype == bool:
        attr = _new_attr(mesh, name, 'BOOLEAN', domain)
        if attr is not None:
            attr.data.foreach_set("value", data.astype(bool))
        return
    shape = data.shape[1:]
    if name == 'Cd' and shape == (3,):
        attr = mesh.color_attributes.new(name, 'FLOAT_COLOR', domain)
        rgba = np.ones((len(data), 4), dtype=np.float32)
        rgba[:, :3] = data
        attr.data.foreach_set("color", rgba.ravel())
        return
    if shape == ():
        if data.dtype.kind in 'iu':
            attr = _new_attr(mesh, name, 'INT', domain)
            if attr is not None:
                attr.data.foreach_set("value", data.astype(np.int32))
        else:
            attr = _new_attr(mesh, name, 'FLOAT', domain)
            if attr is not None:
                attr.data.foreach_set("value", data.astype(np.float32))
        return
    if shape == (2,):
        attr = _new_attr(mesh, name, 'FLOAT2', domain)
        if attr is not None:
            attr.data.foreach_set("vector", data.astype(np.float32).ravel())
        return
    if shape == (3,):
        attr = _new_attr(mesh, name, 'FLOAT_VECTOR', domain)
        if attr is not None:
            attr.data.foreach_set("vector", data.astype(np.float32).ravel())
        return
    if shape == (4,):
        if name in ('orient', 'rot'):
            # Houdini quaternions are (x, y, z, w), Blender's are (w, x, y, z).
            attr = _new_attr(mesh, name, 'QUATERNION', domain)
            if attr is not None:
                attr.data.foreach_set("value", data[:, [3, 0, 1, 2]].astype(np.float32).ravel())
            return
        attr = _new_attr(mesh, name, 'FLOAT_COLOR', domain)
        if attr is not None:
            attr.data.foreach_set("color", data.astype(np.float32).ravel())
        return
    if shape == (4, 4):
        attr = _new_attr(mesh, name, 'FLOAT4X4', domain)
        if attr is not None:
            attr.data.foreach_set("value", np.swapaxes(data, 1, 2).astype(np.float32).ravel())
        return


def _new_attr(mesh, name, type, domain):
    try:
        return mesh.attributes.new(name, type, domain)
    except Exception:
        return None


def mesh_to_geo(mesh, matrix=None):
    """Read a Blender mesh into Blendini geometry."""
    geo = Geometry()
    nv = len(mesh.vertices)
    co = np.empty(nv * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", co)
    P = co.reshape(-1, 3).astype(np.float64)
    if matrix is not None:
        m = np.array(matrix, dtype=np.float64)
        P = P @ m[:3, :3].T + m[:3, 3]
    geo.P = P

    npoly = len(mesh.polygons)
    nloop = len(mesh.loops)
    if npoly:
        starts = np.empty(npoly, dtype=np.int32)
        totals = np.empty(npoly, dtype=np.int32)
        mesh.polygons.foreach_get("loop_start", starts)
        mesh.polygons.foreach_get("loop_total", totals)
        lv = np.empty(nloop, dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", lv)
        order = np.argsort(starts, kind='stable')
        if not np.array_equal(order, np.arange(npoly)):
            loops = np.concatenate([np.arange(starts[i], starts[i] + totals[i]) for i in range(npoly)])
        else:
            loops = None
        flat = lv if loops is None else lv[loops]
        geo.add_prims_flat(totals, flat, True)
    # Loose edges become 2 point polylines.
    ne = len(mesh.edges)
    loose_edges = np.zeros((0, 2), dtype=np.int64)
    if ne:
        ev = np.empty(ne * 2, dtype=np.int32)
        mesh.edges.foreach_get("vertices", ev)
        ev = ev.reshape(-1, 2)
        used = np.zeros(ne, dtype=bool)
        if npoly:
            le = np.empty(nloop, dtype=np.int32)
            mesh.loops.foreach_get("edge_index", le)
            used[le] = True
        loose_edges = ev[~used].astype(np.int64)
        if len(loose_edges):
            geo.add_prims_flat(np.full(len(loose_edges), 2), loose_edges.ravel(), False)

    # Attributes
    for attr in mesh.attributes:
        name = attr.name
        if name.startswith('.') or name in _SKIP_IN:
            continue
        cls = _DOMAIN_IN.get(attr.domain)
        if cls is None:
            continue
        data = _read_attr(attr)
        if data is None:
            continue
        if cls == PRIM and len(loose_edges):
            pad = np.zeros((len(loose_edges),) + data.shape[1:], dtype=data.dtype)
            data = np.concatenate([data, pad])
        if attr.data_type == 'BOOLEAN' and cls in (POINT, PRIM):
            geo.groups[cls][name] = data.astype(bool)
            continue
        if attr.data_type in ('FLOAT_COLOR', 'BYTE_COLOR') and name not in ('Cd',):
            if name == getattr(mesh.color_attributes.active_color, 'name', None) and 'Cd' not in geo.attribs[cls]:
                geo.attribs[cls]['Cd'] = data[:, :3].astype(np.float64)
        if name == 'Cd' and data.ndim == 2 and data.shape[1] == 4:
            data = data[:, :3]
        geo.attribs[cls][name] = data
    for uv in mesh.uv_layers:
        if not len(mesh.loops):
            break
        d = np.empty(len(mesh.loops) * 2, dtype=np.float32)
        uv.data.foreach_get("uv", d)
        uv3 = np.zeros((len(mesh.loops), 3))
        uv3[:, :2] = d.reshape(-1, 2)
        geo.attribs[VERTEX]['uv' if uv.active_render else uv.name] = uv3
    return geo


def _read_attr(attr):
    n = len(attr.data)
    t = attr.data_type
    if t == 'FLOAT':
        d = np.empty(n, dtype=np.float32)
        attr.data.foreach_get("value", d)
        return d.astype(np.float64)
    if t in ('INT', 'INT8'):
        d = np.empty(n, dtype=np.int32)
        attr.data.foreach_get("value", d)
        return d.astype(np.int64)
    if t == 'BOOLEAN':
        d = np.empty(n, dtype=bool)
        attr.data.foreach_get("value", d)
        return d
    if t == 'FLOAT_VECTOR':
        d = np.empty(n * 3, dtype=np.float32)
        attr.data.foreach_get("vector", d)
        return d.reshape(-1, 3).astype(np.float64)
    if t == 'FLOAT2':
        d = np.empty(n * 2, dtype=np.float32)
        attr.data.foreach_get("vector", d)
        return d.reshape(-1, 2).astype(np.float64)
    if t in ('FLOAT_COLOR', 'BYTE_COLOR'):
        d = np.empty(n * 4, dtype=np.float32)
        attr.data.foreach_get("color", d)
        return d.reshape(-1, 4).astype(np.float64)
    if t == 'QUATERNION':
        d = np.empty(n * 4, dtype=np.float32)
        attr.data.foreach_get("value", d)
        return d.reshape(-1, 4)[:, [1, 2, 3, 0]].astype(np.float64)
    if t == 'STRING':
        out = np.empty(n, dtype=object)
        for i, item in enumerate(attr.data):
            out[i] = item.value
        return out
    return None
