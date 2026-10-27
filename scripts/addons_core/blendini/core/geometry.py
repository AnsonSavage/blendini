# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Houdini-style geometry container.

Geometry is made of:

- points:     positions + point attributes.
- primitives: polygons (closed) or polylines (open), each an ordered list of vertices.
- vertices:   a reference from a primitive corner to a point, can carry vertex attributes.
- detail:     global attributes (one value for the whole geometry).

Everything is stored in flat numpy arrays so it can be processed quickly and converted
to / from Blender meshes with ``foreach_set``.
"""

import numpy as np

POINT = 'point'
PRIM = 'prim'
VERTEX = 'vertex'
DETAIL = 'detail'
CLASSES = (POINT, PRIM, VERTEX, DETAIL)

# Attribute names Houdini treats as vectors (3 floats) when created implicitly.
VECTOR_ATTRIBS = {'P', 'N', 'Cd', 'v', 'up', 'scale', 'rest', 'uv', 'force', 'accel', 'center', 'w', 'trans', 'pivot'}
VECTOR4_ATTRIBS = {'orient', 'rot'}
INT_ATTRIBS = {'id', 'class', 'ptnum', 'primnum'}
STRING_ATTRIBS = {'name', 'path', 'instance', 'material'}


def _default_shape(name):
    if name in VECTOR_ATTRIBS:
        return (3,)
    if name in VECTOR4_ATTRIBS:
        return (4,)
    return ()


class Geometry:
    __slots__ = ('attribs', 'prim_start', 'vtx_point', 'prim_closed', 'groups', 'instances')

    def __init__(self):
        self.attribs = {POINT: {'P': np.zeros((0, 3))}, PRIM: {}, VERTEX: {}, DETAIL: {}}
        # prim_start[i]:prim_start[i+1] slices vtx_point for primitive i.
        self.prim_start = np.zeros(1, dtype=np.int64)
        self.vtx_point = np.zeros(0, dtype=np.int64)
        self.prim_closed = np.zeros(0, dtype=bool)
        # Groups: {POINT: {name: bool array}, PRIM: {...}}
        self.groups = {POINT: {}, PRIM: {}}
        # Packed geometry: points with an '__inst' attribute >= 0 instance instances[__inst]
        # with the '__inst_rot' (quaternion xyzw) and '__inst_scale' point attributes.
        self.instances = []

    # ------------------------------------------------------------------
    # Construction helpers

    @classmethod
    def from_polys(cls, points, polys=(), closed=True):
        """Create geometry from a (N, 3) array of positions and a list of index lists."""
        geo = cls()
        geo.attribs[POINT]['P'] = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        geo.add_prims(polys, closed=closed)
        return geo

    def copy(self):
        geo = Geometry()
        geo.attribs = {c: {k: v.copy() for k, v in d.items()} for c, d in self.attribs.items()}
        geo.prim_start = self.prim_start.copy()
        geo.vtx_point = self.vtx_point.copy()
        geo.prim_closed = self.prim_closed.copy()
        geo.groups = {c: {k: v.copy() for k, v in d.items()} for c, d in self.groups.items()}
        geo.instances = list(self.instances)
        return geo

    # ------------------------------------------------------------------
    # Counts / accessors

    @property
    def P(self):
        return self.attribs[POINT]['P']

    @P.setter
    def P(self, value):
        self.attribs[POINT]['P'] = np.asarray(value, dtype=np.float64).reshape(-1, 3)

    @property
    def num_points(self):
        return len(self.attribs[POINT]['P'])

    @property
    def num_prims(self):
        return len(self.prim_start) - 1

    @property
    def num_vertices(self):
        return len(self.vtx_point)

    def count(self, cls):
        if cls == POINT:
            return self.num_points
        if cls == PRIM:
            return self.num_prims
        if cls == VERTEX:
            return self.num_vertices
        return 1

    def prim_sizes(self):
        return np.diff(self.prim_start)

    def vertex_prim(self):
        """Primitive index for each vertex."""
        return np.repeat(np.arange(self.num_prims), self.prim_sizes())

    def prim_points(self, prim):
        return self.vtx_point[self.prim_start[prim]:self.prim_start[prim + 1]]

    def prims_as_lists(self):
        s = self.prim_start
        v = self.vtx_point
        return [v[s[i]:s[i + 1]].tolist() for i in range(self.num_prims)]

    # ------------------------------------------------------------------
    # Attributes

    def has_attrib(self, cls, name):
        return name in self.attribs[cls]

    def find_attrib_class(self, name, order=(POINT, VERTEX, PRIM, DETAIL)):
        for cls in order:
            if name in self.attribs[cls]:
                return cls
        return None

    def attrib(self, cls, name):
        return self.attribs[cls].get(name)

    def add_attrib(self, cls, name, default=None, shape=None, dtype=None):
        """Create attribute if needed and return its array."""
        arr = self.attribs[cls].get(name)
        if arr is not None:
            return arr
        n = self.count(cls)
        if default is not None:
            default = np.asarray(default)
            if shape is None:
                shape = default.shape
            if dtype is None:
                dtype = default.dtype if default.dtype.kind in 'iufbO' else object
        if shape is None:
            shape = _default_shape(name)
        if dtype is None:
            if name in INT_ATTRIBS:
                dtype = np.int64
            elif name in STRING_ATTRIBS:
                dtype = object
            else:
                dtype = np.float64
        if np.dtype(dtype).kind == 'U':
            dtype = object
        arr = np.zeros((n,) + tuple(shape), dtype=dtype)
        if dtype is object:
            arr[...] = ''
        if default is not None:
            arr[...] = default
        self.attribs[cls][name] = arr
        return arr

    def set_attrib(self, cls, name, values):
        n = self.count(cls)
        values = np.asarray(values)
        if values.dtype.kind == 'U':
            values = values.astype(object)
        if values.ndim == 0 or values.shape[0] != n:
            values = np.broadcast_to(values, (n,) + values.shape[values.ndim > 0 and values.shape[0] == 1:]).copy()
        self.attribs[cls][name] = values
        return values

    def remove_attrib(self, cls, name):
        self.attribs[cls].pop(name, None)

    # ------------------------------------------------------------------
    # Modification

    def add_points(self, positions):
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        n_new = len(positions)
        start = self.num_points
        for name, arr in list(self.attribs[POINT].items()):
            if name == 'P':
                self.attribs[POINT]['P'] = np.concatenate([arr, positions])
            elif name == '__inst':
                self.attribs[POINT][name] = np.concatenate([arr, np.full(n_new, -1, dtype=arr.dtype)])
            else:
                self.attribs[POINT][name] = _grow(arr, n_new)
        for name, g in self.groups[POINT].items():
            self.groups[POINT][name] = np.concatenate([g, np.zeros(n_new, dtype=bool)])
        return np.arange(start, start + n_new)

    def add_prims(self, polys, closed=True):
        polys = list(polys)
        if not polys:
            return np.zeros(0, dtype=np.int64)
        sizes = np.array([len(p) for p in polys], dtype=np.int64)
        flat = np.concatenate([np.asarray(p, dtype=np.int64) for p in polys]) if len(polys) else np.zeros(0, np.int64)
        return self.add_prims_flat(sizes, flat, closed)

    def add_prims_flat(self, sizes, flat, closed=True):
        sizes = np.asarray(sizes, dtype=np.int64)
        flat = np.asarray(flat, dtype=np.int64)
        n_new = len(sizes)
        start = self.num_prims
        offsets = self.prim_start[-1] + np.concatenate([[0], np.cumsum(sizes)])
        self.prim_start = np.concatenate([self.prim_start, offsets[1:]])
        nv_new = len(flat)
        self.vtx_point = np.concatenate([self.vtx_point, flat])
        closed_arr = np.broadcast_to(np.asarray(closed, dtype=bool), (n_new,))
        self.prim_closed = np.concatenate([self.prim_closed, closed_arr])
        for name, arr in list(self.attribs[PRIM].items()):
            self.attribs[PRIM][name] = _grow(arr, n_new)
        for name, arr in list(self.attribs[VERTEX].items()):
            self.attribs[VERTEX][name] = _grow(arr, nv_new)
        for name, g in self.groups[PRIM].items():
            self.groups[PRIM][name] = np.concatenate([g, np.zeros(n_new, dtype=bool)])
        return np.arange(start, start + n_new)

    def transform(self, matrix, vectors=True):
        """Apply a 4x4 matrix (column-vector convention, like Blender/mathutils)."""
        m = np.asarray(matrix, dtype=np.float64)
        P = self.P
        self.P = P @ m[:3, :3].T + m[:3, 3]
        if '__inst_rot' in self.attribs[POINT]:
            from .instances import transform_instances
            transform_instances(self, m[:3, :3])
        if vectors:
            rot = m[:3, :3]
            try:
                normal_mat = np.linalg.inv(rot).T
            except np.linalg.LinAlgError:
                normal_mat = rot
            for cls in (POINT, VERTEX, PRIM):
                for name in ('N',):
                    arr = self.attribs[cls].get(name)
                    if arr is not None and arr.ndim == 2 and arr.shape[1] == 3:
                        n = arr @ normal_mat.T
                        ln = np.linalg.norm(n, axis=1, keepdims=True)
                        self.attribs[cls][name] = n / np.where(ln == 0, 1, ln)
                for name in ('v', 'up'):
                    arr = self.attribs[cls].get(name)
                    if arr is not None and arr.ndim == 2 and arr.shape[1] == 3:
                        self.attribs[cls][name] = arr @ rot.T
        return self

    def delete_prims(self, mask, keep_points=False):
        """Remove primitives where mask is True (and points only they used)."""
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return self
        used_before = np.zeros(self.num_points, dtype=bool)
        used_before[self.vtx_point] = True
        self._delete_prims_only(mask)
        if not keep_points:
            used_after = np.zeros(self.num_points, dtype=bool)
            used_after[self.vtx_point] = True
            self.delete_points(used_before & ~used_after)
        return self

    def delete_points(self, mask):
        """Remove points where mask is True, along with primitives referencing them."""
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return self
        if self.num_prims:
            prim_bad = np.zeros(self.num_prims, dtype=bool)
            vp = self.vertex_prim()
            prim_bad[vp[mask[self.vtx_point]]] = True
            if prim_bad.any():
                self._delete_prims_only(prim_bad)
        keep = ~mask
        remap = np.cumsum(keep) - 1
        self.vtx_point = remap[self.vtx_point]
        for name, arr in self.attribs[POINT].items():
            self.attribs[POINT][name] = arr[keep]
        for name, g in self.groups[POINT].items():
            self.groups[POINT][name] = g[keep]
        return self

    def _delete_prims_only(self, mask):
        keep = ~mask
        sizes = self.prim_sizes()
        vkeep = np.repeat(keep, sizes)
        self.vtx_point = self.vtx_point[vkeep]
        self.prim_start = np.concatenate([[0], np.cumsum(sizes[keep])]).astype(np.int64)
        self.prim_closed = self.prim_closed[keep]
        for name, arr in self.attribs[PRIM].items():
            self.attribs[PRIM][name] = arr[keep]
        for name, arr in self.attribs[VERTEX].items():
            self.attribs[VERTEX][name] = arr[vkeep]
        for name, g in self.groups[PRIM].items():
            self.groups[PRIM][name] = g[keep]

    def delete_unused_points(self):
        used = np.zeros(self.num_points, dtype=bool)
        used[self.vtx_point] = True
        return self.delete_points(~used)

    def select_points(self, mask):
        """Keep only the points in mask (removing prims that reference removed points)."""
        return self.delete_points(~np.asarray(mask, dtype=bool))

    def unique_points(self):
        """Give every vertex its own point (Houdini 'Facet > Unique Points')."""
        nv = self.num_vertices
        if nv == 0:
            return self
        src = self.vtx_point
        for name, arr in list(self.attribs[POINT].items()):
            self.attribs[POINT][name] = arr[src]
        for name, g in list(self.groups[POINT].items()):
            self.groups[POINT][name] = g[src]
        self.vtx_point = np.arange(nv, dtype=np.int64)
        return self

    # ------------------------------------------------------------------
    # Derived data

    def prim_centers(self):
        if self.num_prims == 0:
            return np.zeros((0, 3))
        P = self.P[self.vtx_point]
        sums = np.add.reduceat(P, self.prim_start[:-1], axis=0) if len(P) else np.zeros((0, 3))
        sizes = np.maximum(self.prim_sizes(), 1)[:, None]
        # reduceat misbehaves for empty prims, guard them.
        sums[self.prim_sizes() == 0] = 0
        return sums / sizes

    def prim_normals(self):
        """Newell's method normals for each primitive (unit length)."""
        n = self.num_prims
        if n == 0:
            return np.zeros((0, 3))
        P = self.P[self.vtx_point]
        idx = np.arange(len(P))
        # Next vertex index within each primitive (wrapping).
        vp = self.vertex_prim()
        nxt = idx + 1
        last = self.prim_start[1:] - 1
        nxt[last[self.prim_sizes() > 0]] = self.prim_start[:-1][self.prim_sizes() > 0]
        a = P
        b = P[nxt]
        cross = np.stack([
            (a[:, 1] - b[:, 1]) * (a[:, 2] + b[:, 2]),
            (a[:, 2] - b[:, 2]) * (a[:, 0] + b[:, 0]),
            (a[:, 0] - b[:, 0]) * (a[:, 1] + b[:, 1]),
        ], axis=1)
        N = np.zeros((n, 3))
        np.add.at(N, vp, cross)
        ln = np.linalg.norm(N, axis=1, keepdims=True)
        return N / np.where(ln == 0, 1, ln)

    def prim_areas(self):
        n = self.num_prims
        if n == 0:
            return np.zeros(0)
        P = self.P[self.vtx_point]
        vp = self.vertex_prim()
        idx = np.arange(len(P))
        nxt = idx + 1
        last = self.prim_start[1:] - 1
        nz = self.prim_sizes() > 0
        nxt[last[nz]] = self.prim_start[:-1][nz]
        cr = np.cross(P, P[nxt])
        tot = np.zeros((n, 3))
        np.add.at(tot, vp, cr)
        area = 0.5 * np.linalg.norm(tot, axis=1)
        area[~self.prim_closed] = 0.0
        return area

    def point_normals(self):
        """Area weighted average of adjacent face normals."""
        N = np.zeros((self.num_points, 3))
        if self.num_prims:
            fn = self.prim_normals() * np.maximum(self.prim_areas(), 1e-12)[:, None]
            np.add.at(N, self.vtx_point, fn[self.vertex_prim()])
        ln = np.linalg.norm(N, axis=1, keepdims=True)
        N = N / np.where(ln == 0, 1, ln)
        N[ln[:, 0] == 0] = (0, 1, 0)
        return N

    def bbox(self):
        P = self.P
        if len(P) == 0:
            return np.zeros(3), np.zeros(3)
        return P.min(axis=0), P.max(axis=0)

    def triangulate_flat(self):
        """Fan-triangulate closed polygons. Returns (tri_points (T,3), tri_prim (T,))."""
        sizes = self.prim_sizes()
        ok = self.prim_closed & (sizes >= 3)
        ntri = np.where(ok, sizes - 2, 0)
        total = int(ntri.sum())
        if total == 0:
            return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64)
        tri_prim = np.repeat(np.arange(self.num_prims), ntri)
        # Index of triangle within its prim.
        first = np.concatenate([[0], np.cumsum(ntri)[:-1]])
        k = np.arange(total) - np.repeat(first, ntri)
        base = self.prim_start[:-1][tri_prim]
        tris = np.stack([
            self.vtx_point[base],
            self.vtx_point[base + k + 1],
            self.vtx_point[base + k + 2],
        ], axis=1)
        return tris, tri_prim

    def edges(self):
        """Unique undirected edges (E, 2) from all primitives."""
        if self.num_vertices == 0:
            return np.zeros((0, 2), dtype=np.int64)
        idx = np.arange(self.num_vertices)
        nxt = idx + 1
        last = self.prim_start[1:] - 1
        sizes = self.prim_sizes()
        nz = sizes > 0
        nxt[last[nz]] = self.prim_start[:-1][nz]
        valid = np.ones(self.num_vertices, dtype=bool)
        # Open polylines don't have a closing edge.
        open_last = last[nz & ~self.prim_closed]
        valid[open_last] = False
        a = self.vtx_point[idx[valid]]
        b = self.vtx_point[nxt[valid]]
        e = np.sort(np.stack([a, b], axis=1), axis=1)
        e = e[e[:, 0] != e[:, 1]]
        if len(e) == 0:
            return e
        return np.unique(e, axis=0)

    def __repr__(self):
        return "<Geometry points=%d prims=%d vertices=%d>" % (self.num_points, self.num_prims, self.num_vertices)


def _grow(arr, n_new):
    pad = np.zeros((n_new,) + arr.shape[1:], dtype=arr.dtype)
    if arr.dtype == object:
        pad[...] = ''
    return np.concatenate([arr, pad])


def merge(geos):
    """Merge several geometries, unioning attributes (Houdini 'Merge' SOP)."""
    geos = [g for g in geos if g is not None]
    out = Geometry()
    if not geos:
        return out
    if len(geos) == 1:
        return geos[0].copy()
    # Collect attribute signatures.
    for cls in (POINT, PRIM, VERTEX):
        names = {}
        for g in geos:
            for name, arr in g.attribs[cls].items():
                if name not in names:
                    names[name] = (arr.shape[1:], arr.dtype)
        for name, (shape, dtype) in names.items():
            parts = []
            inst_offset = 0
            for g in geos:
                n = g.count(cls)
                arr = g.attribs[cls].get(name)
                if arr is None or arr.shape[1:] != shape:
                    arr = np.zeros((n,) + shape, dtype=dtype)
                    if dtype == object:
                        arr[...] = ''
                    if name == '__inst':
                        arr[...] = -1
                    if name == '__inst_scale':
                        arr[...] = 1
                elif name == '__inst':
                    arr = np.where(arr >= 0, arr + inst_offset, -1)
                if name == '__inst':
                    inst_offset += len(g.instances)
                parts.append(arr.astype(dtype, copy=False))
            out.attribs[cls][name] = np.concatenate(parts) if parts else np.zeros((0,) + shape, dtype)
    for g in geos:
        for name, arr in g.attribs[DETAIL].items():
            if name not in out.attribs[DETAIL]:
                out.attribs[DETAIL][name] = arr.copy()
    # Topology.
    point_offset = 0
    vtx = []
    starts = [np.zeros(1, dtype=np.int64)]
    vstart = 0
    for g in geos:
        vtx.append(g.vtx_point + point_offset)
        starts.append(g.prim_start[1:] + vstart)
        vstart += g.num_vertices
        point_offset += g.num_points
    out.vtx_point = np.concatenate(vtx).astype(np.int64)
    out.prim_start = np.concatenate(starts).astype(np.int64)
    out.prim_closed = np.concatenate([g.prim_closed for g in geos])
    out.instances = [inst for g in geos for inst in g.instances]
    for cls in (POINT, PRIM):
        names = set()
        for g in geos:
            names.update(g.groups[cls].keys())
        for name in names:
            out.groups[cls][name] = np.concatenate([
                g.groups[cls].get(name, np.zeros(g.count(cls), dtype=bool)) for g in geos
            ])
    return out
