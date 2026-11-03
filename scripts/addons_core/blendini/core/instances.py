# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Packed (instanced) geometry helpers."""

import numpy as np

from .geometry import Geometry, POINT, PRIM, VERTEX, DETAIL, merge


def quat_to_matrix(q):
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
    ], axis=1)


def matrix_to_quat(R):
    """Column vector rotation matrices (n, 3, 3) to quaternions (x, y, z, w)."""
    n = len(R)
    q = np.zeros((n, 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    for i in range(n):
        m = R[i]
        if tr[i] > 0:
            s = np.sqrt(tr[i] + 1.0) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)


def instance_matrices(geo):
    """(n, 3, 3) rotation*scale matrices of packed points."""
    pa = geo.attribs[POINT]
    R = quat_to_matrix(pa['__inst_rot'])
    return R * pa['__inst_scale'][:, None, :]


def set_instance_matrices(geo, M):
    scale = np.linalg.norm(M, axis=1)
    R = M / np.where(scale == 0, 1, scale)[:, None, :]
    geo.attribs[POINT]['__inst_rot'] = matrix_to_quat(R)
    geo.attribs[POINT]['__inst_scale'] = scale


def transform_instances(geo, m3):
    if geo.num_points == 0:
        return
    M = np.einsum('ij,njk->nik', m3, instance_matrices(geo))
    set_instance_matrices(geo, M)


def pack(src, positions, M, extra_attribs=None):
    """Points instancing src at positions with 3x3 transforms M."""
    out = Geometry()
    out.P = positions
    out.instances = [src]
    out.attribs[POINT]['__inst'] = np.zeros(len(positions), dtype=np.int64)
    set_instance_matrices(out, M)
    for name, arr in (extra_attribs or {}).items():
        out.attribs[POINT][name] = arr
    return out


def unpack(geo):
    """Replace packed points with real copies of their geometry."""
    pa = geo.attribs[POINT]
    inst = pa.get('__inst')
    if inst is None or not geo.instances:
        return geo
    packed = inst >= 0
    if not packed.any():
        return geo
    M = instance_matrices(geo)
    pieces = []
    skip = {'P', '__inst', '__inst_rot', '__inst_scale', 'N', 'orient', 'pscale', 'scale', 'up'}
    for k, src in enumerate(geo.instances):
        sel = np.nonzero(inst == k)[0]
        if len(sel) == 0 or src.num_points == 0:
            continue
        S = src.num_points
        K = len(sel)
        out = Geometry()
        out.P = (np.einsum('kij,sj->ksi', M[sel], src.P) + geo.P[sel][:, None, :]).reshape(-1, 3)
        for name, arr in src.attribs[POINT].items():
            if name != 'P':
                out.attribs[POINT][name] = np.tile(arr, (K,) + (1,) * (arr.ndim - 1))
        for name, arr in pa.items():
            if name not in skip and name not in out.attribs[POINT]:
                out.attribs[POINT][name] = np.repeat(arr[sel], S, axis=0)
        sizes = src.prim_sizes()
        flat = (src.vtx_point[None, :] + (np.arange(K) * S)[:, None]).ravel()
        out.add_prims_flat(np.tile(sizes, K), flat, np.tile(src.prim_closed, K))
        for name, arr in src.attribs[PRIM].items():
            out.attribs[PRIM][name] = np.tile(arr, (K,) + (1,) * (arr.ndim - 1))
        for name, arr in src.attribs[VERTEX].items():
            out.attribs[VERTEX][name] = np.tile(arr, (K,) + (1,) * (arr.ndim - 1))
        pieces.append(out)
    rest = geo.copy()
    rest.delete_points(packed)
    rest.instances = []
    for name in ('__inst', '__inst_rot', '__inst_scale'):
        rest.attribs[POINT].pop(name, None)
    return merge([rest] + pieces)


def instance_points(geo, sources, attrib='instance'):
    """Turn points whose string attribute names a source geometry into packed instances.

    sources: {name: Geometry}. Orientation comes from orient / N+up, size from pscale and scale
    (Copy to Points conventions). Points naming no known source stay plain points.
    """
    from ..sops.base import orient_matrices
    names = geo.attribs[POINT].get(attrib)
    if names is None or names.dtype != object or not sources:
        return geo
    out = geo.copy()
    n = out.num_points
    keys = [k for k in sorted(sources) if sources[k] is not None]
    index = {k: i for i, k in enumerate(keys)}
    base = len(out.instances)
    inst = out.attribs[POINT].get('__inst')
    inst = np.full(n, -1, dtype=np.int64) if inst is None else np.array(inst, dtype=np.int64)
    hit = np.array([str(v) in index for v in names.tolist()], dtype=bool)
    if not hit.any():
        return geo
    inst[hit] = [base + index[str(v)] for v in names[hit].tolist()]
    out.instances = list(out.instances) + [sources[k] for k in keys]
    R = orient_matrices(out)
    pa = out.attribs[POINT]
    s = np.ones((n, 3))
    if 'pscale' in pa and pa['pscale'].dtype != object:
        s *= np.asarray(pa['pscale'], dtype=np.float64).reshape(n, -1)[:, :1]
    if 'scale' in pa and pa['scale'].dtype != object and np.asarray(pa['scale']).ndim == 2:
        s *= pa['scale']
    M = R * s[:, None, :]
    old_rot = pa.get('__inst_rot')
    old_scale = pa.get('__inst_scale')
    pa['__inst'] = inst
    set_instance_matrices(out, M)
    if old_rot is not None and old_scale is not None:
        # Keep the transforms of points that were already packed.
        keep = ~hit & (inst >= 0)
        pa['__inst_rot'][keep] = old_rot[keep]
        pa['__inst_scale'][keep] = old_scale[keep]
    return out
