# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Native volume primitives: dense voxel grids carried by a Geometry (``geo.volumes``).

A volume samples its value at voxel centers: voxel (i, j, k) is centered at
``origin + (i + 0.5, j + 0.5, k + 0.5) * voxelsize``. Values are float (scalar fields such as
density or a signed distance) or vector (shape ``(nx, ny, nz, 3)``, e.g. vel).
"""

import numpy as np


class Volume:
    __slots__ = ('name', 'data', 'origin', 'voxelsize', 'kind')

    def __init__(self, name, data, origin, voxelsize, kind='fog'):
        self.name = name
        self.data = np.asarray(data, dtype=np.float64)
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.voxelsize = float(voxelsize)
        self.kind = kind   # 'fog' (density like) or 'sdf' (signed distance, negative inside)

    @classmethod
    def from_bounds(cls, name, lo, hi, voxelsize, value=0.0, kind='fog', vector=False):
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        vs = max(float(voxelsize), 1e-6)
        res = np.maximum(np.ceil((hi - lo) / vs - 1e-9).astype(np.int64), 1)
        if np.prod(res) > 16_000_000:
            raise ValueError("volume too large (%s voxels); use a bigger voxel size" % 'x'.join(map(str, res)))
        center = (lo + hi) / 2
        origin = center - res * vs / 2
        shape = tuple(res) + ((3,) if vector else ())
        return cls(name, np.full(shape, value, dtype=np.float64), origin, vs, kind)

    def copy(self, data=None, name=None):
        return Volume(self.name if name is None else name, self.data.copy() if data is None else data,
                      self.origin.copy(), self.voxelsize, self.kind)

    @property
    def res(self):
        return np.array(self.data.shape[:3], dtype=np.int64)

    @property
    def is_vector(self):
        return self.data.ndim == 4

    def bounds(self):
        return self.origin, self.origin + self.res * self.voxelsize

    def centers(self):
        """World positions of all voxel centers, (nx*ny*nz, 3) in C order."""
        idx = np.indices(tuple(self.res)).reshape(3, -1).T
        return self.origin + (idx + 0.5) * self.voxelsize

    def index_to_pos(self, idx):
        return self.origin + (np.asarray(idx, dtype=np.float64) + 0.5) * self.voxelsize

    def pos_to_index(self, pos):
        return np.floor((np.asarray(pos, dtype=np.float64) - self.origin) / self.voxelsize).astype(np.int64)

    def sample(self, pos):
        """Trilinear interpolation at world positions (n, 3); clamps to the border voxels."""
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        res = self.res
        f = (pos - self.origin) / self.voxelsize - 0.5
        i0 = np.floor(f).astype(np.int64)
        t = f - i0
        out = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ix = np.clip(i0[:, 0] + dx, 0, res[0] - 1)
                    iy = np.clip(i0[:, 1] + dy, 0, res[1] - 1)
                    iz = np.clip(i0[:, 2] + dz, 0, res[2] - 1)
                    w = ((t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1])
                         * (t[:, 2] if dz else 1 - t[:, 2]))
                    v = self.data[ix, iy, iz]
                    out = out + (w[:, None] * v if v.ndim == 2 else w * v)
        return out

    def gradient(self, pos):
        h = self.voxelsize * 0.5
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        g = np.zeros((len(pos), 3))
        for k in range(3):
            e = np.zeros(3)
            e[k] = h
            g[:, k] = (self.sample(pos + e) - self.sample(pos - e)) / (2 * h)
        return g


def volume_by_name(geo, name_or_index):
    vols = getattr(geo, 'volumes', None) or []
    if isinstance(name_or_index, (int, np.integer)):
        return vols[name_or_index] if 0 <= name_or_index < len(vols) else None
    for v in vols:
        if v.name == name_or_index:
            return v
    return None


# ----------------------------------------------------------------------------
# Iso surfaces (surface nets: one vertex per surface cell, quads across sign changes)


def surface_nets(vol, iso=0.0, inside_below=True):
    """Polygonize a scalar volume at an iso value. Returns (points (n, 3), quads (m, 4))."""
    f = vol.data
    if f.ndim != 3 or min(f.shape) < 2:
        return np.zeros((0, 3)), np.zeros((0, 4), dtype=np.int64)
    s = f - iso
    inside = s < 0 if inside_below else s > 0
    nx, ny, nz = f.shape
    # Cells span 2x2x2 samples.
    corners = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
    cin = np.stack([inside[dx:nx - 1 + dx, dy:ny - 1 + dy, dz:nz - 1 + dz] for dx, dy, dz in corners], axis=-1)
    active = cin.any(axis=-1) & ~cin.all(axis=-1)
    cell_idx = np.full(active.shape, -1, dtype=np.int64)
    ncell = int(active.sum())
    cell_idx[active] = np.arange(ncell)
    # Vertex: average of the edge crossings of the cell.
    acc = np.zeros(active.shape + (3,))
    cnt = np.zeros(active.shape)
    cvals = [s[dx:nx - 1 + dx, dy:ny - 1 + dy, dz:nz - 1 + dz] for dx, dy, dz in corners]
    for a in range(8):
        for b in range(a + 1, 8):
            da, db = np.array(corners[a]), np.array(corners[b])
            if np.abs(da - db).sum() != 1:
                continue
            va, vb = cvals[a], cvals[b]
            cross = (va < 0) != (vb < 0)
            t = np.where(cross, va / np.where(va - vb == 0, 1, va - vb), 0.0)
            pt = da[None, None, None, :] + (db - da)[None, None, None, :] * t[..., None]
            acc += np.where(cross[..., None], pt, 0.0)
            cnt += cross
    ii = np.indices(active.shape).transpose(1, 2, 3, 0)
    local = acc / np.maximum(cnt, 1)[..., None]
    pts = vol.origin + (ii + local + 0.5) * vol.voxelsize
    points = pts[active]
    # Quads across sample edges with a sign change.
    quads = []
    for axis in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[axis] = slice(0, -1)
        sl1[axis] = slice(1, None)
        a = inside[tuple(sl0)]
        b = inside[tuple(sl1)]
        change = a != b
        o1, o2 = [k for k in range(3) if k != axis]
        # The four cells around an edge at sample (i, j, k): offsets 0/-1 on the two other axes.
        idx = np.argwhere(change)
        if not len(idx):
            continue
        ok = (idx[:, o1] >= 1) & (idx[:, o2] >= 1) & (idx[:, o1] <= f.shape[o1] - 2) & (idx[:, o2] <= f.shape[o2] - 2)
        idx = idx[ok]
        flip = a[tuple(idx.T)]
        quad = []
        for d1, d2 in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            c = idx.copy()
            c[:, o1] += d1
            c[:, o2] += d2
            quad.append(cell_idx[c[:, 0], c[:, 1], c[:, 2]])
        quad = np.stack(quad, axis=1)
        # Consistent winding: outward facing from the inside region.
        reverse = flip if axis == 1 else ~flip
        quad[reverse] = quad[reverse][:, ::-1]
        quads.append(quad)
    quads = np.concatenate(quads) if quads else np.zeros((0, 4), dtype=np.int64)
    quads = quads[(quads >= 0).all(axis=1)]
    return points, quads


# ----------------------------------------------------------------------------
# Signed distance from polygons


def winding_numbers(P, tris, queries, chunk=200_000):
    """Generalized winding number of a triangle mesh at query points (~1 inside, ~0 outside)."""
    A, B, C = P[tris[:, 0]], P[tris[:, 1]], P[tris[:, 2]]
    out = np.zeros(len(queries))
    m = max(len(tris), 1)
    step = max(1, chunk // m)
    for s in range(0, len(queries), step):
        q = queries[s:s + step][:, None, :]
        a, b, c = A[None] - q, B[None] - q, C[None] - q
        la, lb, lc = (np.linalg.norm(x, axis=2) for x in (a, b, c))
        det = np.einsum('ijk,ijk->ij', a, np.cross(b, c))
        div = (la * lb * lc + np.einsum('ijk,ijk->ij', a, b) * lc + np.einsum('ijk,ijk->ij', b, c) * la
               + np.einsum('ijk,ijk->ij', c, a) * lb)
        out[s:s + step] = np.arctan2(det, div).sum(axis=1) / (2 * np.pi)
    return out


def sdf_from_geometry(geo, voxelsize, band=3, kind='sdf'):
    """Signed distance (negative inside) or fog (1 inside) volume around a closed polygon mesh."""
    from .spatial import closest
    lo, hi = geo.bbox()
    pad = band * voxelsize
    vol = Volume.from_bounds('surface' if kind == 'sdf' else 'density', lo - pad, hi + pad, voxelsize, kind=kind)
    tris, _ = geo.triangulate_flat()
    pts = vol.centers()
    inside = winding_numbers(geo.P, tris, pts) > 0.5 if len(tris) else np.zeros(len(pts), dtype=bool)
    if kind == 'fog':
        vol.data = inside.reshape(tuple(vol.res)).astype(np.float64)
        return vol
    _loc, _n, _prim, dist = closest(geo, pts)
    vol.data = np.where(inside, -dist, dist).reshape(tuple(vol.res))
    return vol


def transform_volumes(volumes, m):
    """Apply a 4x4 (column vector) matrix. Translation and uniform scale are exact; rotations and
    non-uniform scales resample into a new axis-aligned grid."""
    m = np.asarray(m, dtype=np.float64)
    R = m[:3, :3]
    t = m[:3, 3]
    out = []
    scales = np.linalg.norm(R, axis=0)
    uniform = np.allclose(R, np.diag(np.diag(R))) and np.allclose(np.diag(R), scales[0]) and scales[0] > 0
    for v in volumes:
        if uniform:
            nv = v.copy()
            nv.origin = v.origin * scales[0] + t
            nv.voxelsize = v.voxelsize * scales[0]
            if v.kind == 'sdf':
                nv.data = v.data * scales[0]
            out.append(nv)
            continue
        lo, hi = v.bounds()
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        tc = corners @ R.T + t
        vs = v.voxelsize * float(np.cbrt(abs(np.linalg.det(R)))) if abs(np.linalg.det(R)) > 1e-12 else v.voxelsize
        nv = Volume.from_bounds(v.name, tc.min(axis=0), tc.max(axis=0), vs, kind=v.kind, vector=v.is_vector)
        try:
            Rinv = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            out.append(v.copy())
            continue
        src = (nv.centers() - t) @ Rinv.T
        vals = v.sample(src)
        if v.is_vector:
            vals = vals @ R.T
        # Outside the source grid: background (0 for fog, far for sdf).
        outside = np.any((src < lo) | (src > hi), axis=1)
        if v.kind == 'sdf':
            vals = np.where(outside, np.abs(vals) + v.voxelsize, vals)
        elif not v.is_vector:
            vals = np.where(outside, 0.0, vals)
        else:
            vals = np.where(outside[:, None], 0.0, vals)
        nv.data = vals.reshape(nv.data.shape)
        out.append(nv)
    return out


def volume_display(geo, fog_surfaces=True):
    """Display geometry for volumes: SDFs as their zero surface, fog as an iso surface at half
    the max density, plus a bounding box outline."""
    from .geometry import Geometry, merge
    parts = []
    for v in geo.volumes:
        lo, hi = v.bounds()
        c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
        parts.append(Geometry.from_polys(c, edges, closed=False))
        if v.is_vector:
            continue
        if v.kind == 'sdf':
            pts, quads = surface_nets(v, 0.0)
        else:
            if not fog_surfaces:
                continue
            mx = float(v.data.max()) if v.data.size else 0.0
            if mx <= 0:
                continue
            pts, quads = surface_nets(v, mx * 0.5, inside_below=False)
        if len(quads):
            g = Geometry.from_polys(pts)
            g.add_prims_flat(np.full(len(quads), 4), quads.ravel())
            parts.append(g)
    return merge(parts) if parts else Geometry()
