# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Primitive generator SOPs: Box, Grid, Sphere, Tube, Torus, Circle, Line, Add, Platonic."""

import numpy as np

from .base import SOP, Parm, register_sop
from ..core.geometry import Geometry, merge

ORIENT_ITEMS = [('ZX', "ZX Plane", ""), ('XY', "XY Plane", ""), ('YZ', "YZ Plane", "")]
AXIS_ITEMS = [('Y', "Y Axis", ""), ('X', "X Axis", ""), ('Z', "Z Axis", "")]


def grid_points(nu, nv, su, sv):
    """Points of a nu x nv grid in the XZ plane (Y up), Houdini row ordering."""
    u = np.linspace(-su / 2, su / 2, nu) if nu > 1 else np.zeros(1)
    v = np.linspace(-sv / 2, sv / 2, nv) if nv > 1 else np.zeros(1)
    U, Vv = np.meshgrid(u, v)
    P = np.zeros((nu * nv, 3))
    P[:, 0] = U.ravel()
    P[:, 2] = Vv.ravel()
    return P


def grid_quads(nu, nv, wrap_u=False, wrap_v=False, flip=False):
    """Quad index array (F, 4) for a nu x nv point lattice."""
    cu = nu if wrap_u else nu - 1
    cv = nv if wrap_v else nv - 1
    if cu <= 0 or cv <= 0:
        return np.zeros((0, 4), dtype=np.int64)
    i = np.arange(cu)
    j = np.arange(cv)
    I, J = np.meshgrid(i, j)
    I = I.ravel()
    J = J.ravel()
    a = J * nu + I
    b = J * nu + (I + 1) % nu
    c = ((J + 1) % nv) * nu + (I + 1) % nu
    d = ((J + 1) % nv) * nu + I
    quads = np.stack([a, d, c, b], axis=1) if not flip else np.stack([a, b, c, d], axis=1)
    return quads


def orient_plane(P, orient):
    """Grid points are built in ZX (Y up). Rotate into the requested plane."""
    if orient == 'XY':
        return np.stack([P[:, 0], -P[:, 2], P[:, 1]], axis=1)
    if orient == 'YZ':
        return np.stack([P[:, 1], -P[:, 0], P[:, 2]], axis=1)
    return P


def orient_axis(P, axis):
    """Shapes are built around Y. Rotate so the main axis matches."""
    if axis == 'X':
        return np.stack([P[:, 1], -P[:, 0], P[:, 2]], axis=1)
    if axis == 'Z':
        return np.stack([P[:, 0], -P[:, 2], P[:, 1]], axis=1)
    return P


@register_sop
class GridSOP(SOP):
    id = 'grid'
    label = "Grid"
    category = 'Primitive'
    icon = 'MESH_GRID'
    parms = (
        Parm('orient', "Orientation", 'enum', items=ORIENT_ITEMS),
        Parm('size', "Size", 'vector2', default=(10.0, 10.0)),
        Parm('t', "Center", 'vector'),
        Parm('rows', "Rows", 'int', default=10, min=1, soft_max=200),
        Parm('cols', "Columns", 'int', default=10, min=1, soft_max=200),
    )

    def cook(self, inputs, p, ctx):
        nu, nv = max(p.cols, 1), max(p.rows, 1)
        P = grid_points(nu, nv, p.size[0], p.size[1])
        quads = grid_quads(nu, nv)
        P = orient_plane(P, p.orient) + np.asarray(p.t)
        geo = Geometry.from_polys(P)
        geo.add_prims_flat(np.full(len(quads), 4), quads.ravel())
        return geo


def make_box(size, divs=(1, 1, 1)):
    sx, sy, sz = size
    dx, dy, dz = (max(int(d), 1) for d in divs)
    if (dx, dy, dz) == (1, 1, 1):
        x, y, z = sx / 2, sy / 2, sz / 2
        P = np.array([[-x, -y, -z], [x, -y, -z], [x, -y, z], [-x, -y, z],
                      [-x, y, -z], [x, y, -z], [x, y, z], [-x, y, z]])
        faces = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]
        return Geometry.from_polys(P, faces)
    # Divided box: six grids fused together.
    sides = []

    def side(nu, nv, su, sv, fn):
        P = grid_points(nu + 1, nv + 1, su, sv)
        q = grid_quads(nu + 1, nv + 1)
        g = Geometry.from_polys(fn(P))
        g.add_prims_flat(np.full(len(q), 4), q.ravel())
        return g
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    sides.append(side(dx, dz, sx, sz, lambda P: P + [0, hy, 0]))                                     # top
    sides.append(side(dx, dz, sx, sz, lambda P: np.stack([P[:, 0], -P[:, 1] - hy, -P[:, 2]], 1)))    # bottom
    sides.append(side(dx, dy, sx, sy, lambda P: np.stack([P[:, 0], -P[:, 2], P[:, 1] + hz], 1)))     # front
    sides.append(side(dx, dy, sx, sy, lambda P: np.stack([-P[:, 0], -P[:, 2], -P[:, 1] - hz], 1)))   # back
    sides.append(side(dz, dy, sz, sy, lambda P: np.stack([P[:, 1] + hx, -P[:, 2], -P[:, 0]], 1)))    # right
    sides.append(side(dz, dy, sz, sy, lambda P: np.stack([-P[:, 1] - hx, -P[:, 2], P[:, 0]], 1)))    # left
    geo = merge(sides)
    from .modify import fuse_points
    fuse_points(geo, min(sx / dx, sy / dy, sz / dz) * 1e-3)
    return geo


@register_sop
class BoxSOP(SOP):
    id = 'box'
    label = "Box"
    category = 'Primitive'
    icon = 'MESH_CUBE'
    parms = (
        Parm('size', "Size", 'vector', default=(1.0, 1.0, 1.0)),
        Parm('t', "Center", 'vector'),
        Parm('r', "Rotate", 'vector'),
        Parm('scale', "Uniform Scale", 'float', default=1.0, soft_min=0, soft_max=10),
        Parm('divs', "Axis Divisions", 'vector', default=(1, 1, 1)),
    )

    def cook(self, inputs, p, ctx):
        from .base import transform_matrix
        geo = make_box(np.asarray(p.size) * p.scale, [round(d) for d in p.divs])
        geo.transform(transform_matrix(t=p.t, r=p.r))
        return geo


def uv_sphere(rows, cols, radius=(1, 1, 1)):
    rows = max(rows, 3)
    cols = max(cols, 3)
    pts = [[0, 1, 0]]
    for i in range(1, rows - 1):
        th = np.pi * i / (rows - 1)
        for j in range(cols):
            ph = 2 * np.pi * j / cols
            pts.append([np.sin(th) * np.cos(ph), np.cos(th), -np.sin(th) * np.sin(ph)])
    pts.append([0, -1, 0])
    P = np.array(pts) * np.asarray(radius)
    polys = []
    bottom = len(pts) - 1
    for j in range(cols):
        polys.append([0, 1 + (j + 1) % cols, 1 + j][::-1])
    for i in range(rows - 3):
        r0 = 1 + i * cols
        r1 = r0 + cols
        for j in range(cols):
            j1 = (j + 1) % cols
            polys.append([r0 + j, r1 + j, r1 + j1, r0 + j1])
    last = 1 + (rows - 3) * cols
    for j in range(cols):
        polys.append([last + j, bottom, last + (j + 1) % cols])
    return Geometry.from_polys(P, polys)


def icosahedron():
    t = (1 + 5 ** 0.5) / 2
    P = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=np.float64)
    P /= np.linalg.norm(P, axis=1, keepdims=True)
    F = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
         [8, 6, 7], [9, 8, 1]]
    return P, F


def ico_sphere(freq):
    P, F = icosahedron()
    P = list(map(tuple, P))
    for _ in range(max(0, int(freq))):
        cache = {}
        newF = []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = (np.asarray(P[a]) + np.asarray(P[b])) / 2
                m /= np.linalg.norm(m)
                P.append(tuple(m))
                cache[key] = len(P) - 1
            return cache[key]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            newF += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        F = newF
    return Geometry.from_polys(np.array(P), F)


@register_sop
class SphereSOP(SOP):
    id = 'sphere'
    label = "Sphere"
    category = 'Primitive'
    icon = 'MESH_UVSPHERE'
    parms = (
        Parm('type', "Primitive Type", 'enum', items=[
            ('polygon', "Polygon", "UV sphere"), ('polymesh', "Polygon Mesh (Icosahedron)", "Subdivided icosahedron"),
        ]),
        Parm('rad', "Radius", 'vector', default=(1.0, 1.0, 1.0)),
        Parm('t', "Center", 'vector'),
        Parm('scale', "Uniform Scale", 'float', default=1.0, soft_min=0, soft_max=10),
        Parm('rows', "Rows", 'int', default=13, min=3, soft_max=100, hide_when={'type': 'polymesh'}),
        Parm('cols', "Columns", 'int', default=24, min=3, soft_max=100, hide_when={'type': 'polymesh'}),
        Parm('freq', "Frequency", 'int', default=2, min=0, max=7, hide_when={'type': 'polygon'}),
    )

    def cook(self, inputs, p, ctx):
        if p.type == 'polymesh':
            geo = ico_sphere(p.freq)
            geo.P = geo.P * np.asarray(p.rad)
        else:
            geo = uv_sphere(p.rows, p.cols, p.rad)
        geo.P = geo.P * p.scale + np.asarray(p.t)
        return geo


@register_sop
class TubeSOP(SOP):
    id = 'tube'
    label = "Tube"
    category = 'Primitive'
    icon = 'MESH_CYLINDER'
    parms = (
        Parm('orient', "Orientation", 'enum', items=AXIS_ITEMS),
        Parm('cap', "End Caps", 'bool', default=True),
        Parm('t', "Center", 'vector'),
        Parm('rad1', "Radius Top", 'float', default=1.0, soft_min=0, soft_max=10),
        Parm('rad2', "Radius Bottom", 'float', default=1.0, soft_min=0, soft_max=10),
        Parm('height', "Height", 'float', default=1.0, soft_min=0, soft_max=10),
        Parm('rows', "Rows", 'int', default=2, min=2, soft_max=100),
        Parm('cols', "Columns", 'int', default=12, min=3, soft_max=100),
    )

    def cook(self, inputs, p, ctx):
        rows, cols = max(p.rows, 2), max(p.cols, 3)
        ang = 2 * np.pi * np.arange(cols) / cols
        pts = []
        for i in range(rows):
            f = i / (rows - 1)
            y = p.height / 2 - f * p.height
            r = p.rad1 + (p.rad2 - p.rad1) * f
            pts.append(np.stack([r * np.cos(ang), np.full(cols, y), -r * np.sin(ang)], axis=1))
        P = np.concatenate(pts)
        quads = grid_quads(cols, rows, wrap_u=True)
        geo = Geometry.from_polys(P)
        geo.add_prims_flat(np.full(len(quads), 4), quads.ravel())
        if p.cap:
            geo.add_prims([list(range(cols)), list(range((rows - 1) * cols, rows * cols))[::-1]])
        geo.P = orient_axis(geo.P, p.orient) + np.asarray(p.t)
        return geo


@register_sop
class TorusSOP(SOP):
    id = 'torus'
    label = "Torus"
    category = 'Primitive'
    icon = 'MESH_TORUS'
    parms = (
        Parm('orient', "Orientation", 'enum', items=AXIS_ITEMS),
        Parm('rad', "Radius", 'vector2', default=(1.0, 0.5)),
        Parm('t', "Center", 'vector'),
        Parm('rows', "Rows", 'int', default=12, min=3, soft_max=100),
        Parm('cols', "Columns", 'int', default=24, min=3, soft_max=100),
    )

    def cook(self, inputs, p, ctx):
        rows, cols = max(p.rows, 3), max(p.cols, 3)
        R, r = p.rad
        u = 2 * np.pi * np.arange(cols) / cols
        v = 2 * np.pi * np.arange(rows) / rows
        U, Vv = np.meshgrid(u, v)
        x = (R + r * np.cos(Vv)) * np.cos(U)
        z = -(R + r * np.cos(Vv)) * np.sin(U)
        y = r * np.sin(Vv)
        P = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        quads = grid_quads(cols, rows, wrap_u=True, wrap_v=True)
        geo = Geometry.from_polys(P)
        geo.add_prims_flat(np.full(len(quads), 4), quads.ravel())
        geo.P = orient_axis(geo.P, p.orient) + np.asarray(p.t)
        return geo


@register_sop
class CircleSOP(SOP):
    id = 'circle'
    label = "Circle"
    category = 'Primitive'
    icon = 'MESH_CIRCLE'
    parms = (
        Parm('type', "Primitive Type", 'enum', items=[
            ('poly', "Polygon", "Closed polygon"), ('open', "Open Arc", "Open polyline"), ('points', "Points Only", ""),
        ]),
        Parm('orient', "Orientation", 'enum', items=ORIENT_ITEMS),
        Parm('rad', "Radius", 'vector2', default=(1.0, 1.0)),
        Parm('t', "Center", 'vector'),
        Parm('divs', "Divisions", 'int', default=12, min=3, soft_max=200),
        Parm('angle', "Arc Angles", 'vector2', default=(0.0, 360.0)),
    )

    def cook(self, inputs, p, ctx):
        a0, a1 = np.radians(p.angle[0]), np.radians(p.angle[1])
        full_circle = abs((p.angle[1] - p.angle[0]) % 360.0) < 1e-6 and p.type != 'open'
        n = max(p.divs, 3)
        if full_circle:
            ang = a0 + (a1 - a0) * np.arange(n) / n
        else:
            ang = np.linspace(a0, a1, n)
        P = np.stack([p.rad[0] * np.cos(ang), np.zeros(n), -p.rad[1] * np.sin(ang)], axis=1)
        P = orient_plane(P, p.orient) + np.asarray(p.t)
        geo = Geometry.from_polys(P)
        if p.type == 'poly':
            geo.add_prims([list(range(n))], closed=True)
        elif p.type == 'open':
            geo.add_prims([list(range(n))], closed=False)
        return geo


@register_sop
class LineSOP(SOP):
    id = 'line'
    label = "Line"
    category = 'Primitive'
    icon = 'IPO_LINEAR'
    parms = (
        Parm('origin', "Origin", 'vector'),
        Parm('dir', "Direction", 'vector', default=(0.0, 1.0, 0.0)),
        Parm('dist', "Length", 'float', default=1.0, soft_min=0, soft_max=10),
        Parm('points', "Points", 'int', default=2, min=1, soft_max=200),
    )

    def cook(self, inputs, p, ctx):
        d = np.asarray(p.dir, dtype=np.float64)
        ln = np.linalg.norm(d)
        d = d / ln if ln else d
        n = max(p.points, 1)
        t = np.linspace(0, p.dist, n) if n > 1 else np.zeros(1)
        P = np.asarray(p.origin) + t[:, None] * d
        geo = Geometry.from_polys(P)
        if n > 1:
            geo.add_prims([list(range(n))], closed=False)
        return geo


@register_sop
class PlatonicSOP(SOP):
    id = 'platonic'
    label = "Platonic Solids"
    category = 'Primitive'
    icon = 'MESH_ICOSPHERE'
    parms = (
        Parm('type', "Solid Type", 'enum', items=[
            ('tetra', "Tetrahedron", ""), ('cube', "Cube", ""), ('octa', "Octahedron", ""), ('icosa', "Icosahedron", ""),
        ]),
        Parm('t', "Center", 'vector'),
        Parm('rad', "Radius", 'float', default=1.0, soft_min=0, soft_max=10),
    )

    def cook(self, inputs, p, ctx):
        if p.type == 'tetra':
            P = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64)
            F = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]
        elif p.type == 'cube':
            geo = make_box((2, 2, 2))
            P = geo.P
            F = geo.prims_as_lists()
        elif p.type == 'octa':
            P = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float64)
            F = [[0, 2, 4], [4, 2, 1], [1, 2, 5], [5, 2, 0], [4, 3, 0], [1, 3, 4], [5, 3, 1], [0, 3, 5]]
        else:
            P, F = icosahedron()
        P = P / np.linalg.norm(P, axis=1, keepdims=True) * p.rad + np.asarray(p.t)
        return Geometry.from_polys(P, F)


@register_sop
class AddSOP(SOP):
    id = 'add'
    label = "Add"
    category = 'Polygon'
    icon = 'ADD'
    inputs = 1
    input_labels = ("Geometry",)
    parms = (
        Parm('addpoint', "Add Point", 'bool', default=True),
        Parm('pt', "Position", 'vector', hide_when={'addpoint': False}),
        Parm('connect', "Polygons", 'enum', items=[
            ('none', "None", ""), ('open', "Connect All Points (Open)", ""), ('closed', "Connect All Points (Closed)", ""),
        ]),
        Parm('remove', "Delete Geometry But Keep Points", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        geo = inputs[0].copy() if inputs and inputs[0] is not None else Geometry()
        if p.remove:
            geo._delete_prims_only(np.ones(geo.num_prims, dtype=bool))
        if p.addpoint:
            geo.add_points([p.pt])
        if p.connect != 'none' and geo.num_points > 1:
            geo.add_prims([list(range(geo.num_points))], closed=(p.connect == 'closed'))
        return geo
