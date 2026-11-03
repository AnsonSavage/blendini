# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Volume SOPs on Blendini's native voxel grids (core/volume.py)."""

import numpy as np

from .base import SOP, Parm, register_sop, CookError
from ..core.geometry import Geometry, POINT
from ..core.volume import Volume, sdf_from_geometry, surface_nets
from .modify import _in


def _keep_geometry(src, volumes, keep):
    out = src.copy() if keep else Geometry()
    if not keep:
        out.attribs['detail'] = {k: v.copy() for k, v in src.attribs['detail'].items()}
    out.volumes = list(volumes)
    return out


@register_sop
class VolumeSOP(SOP):
    id = 'volume'
    label = "Volume"
    category = 'Volume'
    inputs = 1
    input_labels = ("Bounds (optional)",)
    icon = 'VOLUME_DATA'
    description = "Create a voxel grid (fog density, signed distance or vector field)"
    parms = (
        Parm('name', "Name", 'string', default='density'),
        Parm('kind', "Type", 'enum', items=[('fog', "Fog (density)", ""), ('sdf', "Signed Distance", ""),
                                            ('vector', "Vector", "")]),
        Parm('initval', "Initial Value", 'float', default=0.0),
        Parm('size', "Size", 'vector', default=(2.0, 2.0, 2.0)),
        Parm('t', "Center", 'vector'),
        Parm('voxelsize', "Voxel Size", 'float', default=0.05, min=0.001, soft_max=1.0),
        Parm('usebounds', "Size from Input Bounds", 'bool', default=True,
             description="When something is connected, fit the grid around it"),
        Parm('padding', "Bounds Padding", 'float', default=0.1, min=0, hide_when={'usebounds': False}),
    )

    def cook(self, inputs, p, ctx):
        src = _in(inputs)
        if p.usebounds and src.num_points:
            lo, hi = src.bbox()
            lo, hi = lo - p.padding, hi + p.padding
        else:
            half = np.abs(np.asarray(p.size, dtype=np.float64)) / 2
            lo, hi = np.asarray(p.t) - half, np.asarray(p.t) + half
        try:
            vol = Volume.from_bounds(p.name.strip() or 'density', lo, hi, p.voxelsize, value=p.initval,
                                     kind='sdf' if p.kind == 'sdf' else 'fog', vector=p.kind == 'vector')
        except ValueError as ex:
            raise CookError(str(ex))
        out = Geometry()
        out.volumes = [vol]
        return out


@register_sop
class VolumeFromPolygonsSOP(SOP):
    id = 'volumefrompolygons'
    label = "Volume from Polygons"
    category = 'Volume'
    inputs = 1
    icon = 'MOD_REMESH'
    description = "Signed distance field ('surface') or fog ('density') from a closed polygon mesh (VDB from Polygons)"
    parms = (
        Parm('kind', "Output", 'enum', items=[('sdf', "Distance VDB (surface)", ""), ('fog', "Fog VDB (density)", "")]),
        Parm('voxelsize', "Voxel Size", 'float', default=0.1, min=0.005, soft_max=1.0),
        Parm('band', "Exterior Band Voxels", 'int', default=3, min=1, max=20),
        Parm('name', "Name", 'string', default='', description="Default: surface (SDF) or density (fog)"),
        Parm('keep', "Keep Polygons", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        src = _in(inputs)
        if src.num_prims == 0:
            raise CookError("Volume from Polygons needs a closed polygon mesh")
        try:
            vol = sdf_from_geometry(src, p.voxelsize, p.band, p.kind)
        except ValueError as ex:
            raise CookError(str(ex))
        if p.name.strip():
            vol.name = p.name.strip()
        return _keep_geometry(src, src.volumes + [vol], p.keep)


@register_sop
class ConvertVolumeSOP(SOP):
    id = 'convertvolume'
    label = "Convert Volume"
    category = 'Volume'
    inputs = 1
    icon = 'MESH_ICOSPHERE'
    description = "Polygonize volumes at an iso value (surface nets); SDFs default to their zero surface"
    parms = (
        Parm('group', "Volumes (names)", 'string', default='*'),
        Parm('iso', "Iso Value", 'float', default=0.0,
             description="For fog volumes 0 means half of the maximum density"),
        Parm('invert', "Invert (inside is above the iso value)", 'bool', default=False),
        Parm('keepvolumes', "Keep Volumes", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        import fnmatch
        src = _in(inputs)
        out = src.copy()
        if not p.keepvolumes:
            out.volumes = []
        parts = []
        pattern = p.group.strip() or '*'
        for v in src.volumes:
            if v.is_vector or not any(fnmatch.fnmatchcase(v.name, pat) for pat in pattern.split()):
                continue
            if v.kind == 'sdf':
                pts, quads = surface_nets(v, p.iso, inside_below=not p.invert)
            else:
                iso = p.iso if p.iso != 0 else 0.5 * float(v.data.max())
                pts, quads = surface_nets(v, iso, inside_below=p.invert)
            if len(quads):
                g = Geometry.from_polys(pts)
                g.add_prims_flat(np.full(len(quads), 4), quads.ravel())
                g.attribs['prim']['name'] = np.array([v.name] * len(quads), dtype=object)
                parts.append(g)
        from ..core.geometry import merge
        result = merge([out] + parts)
        result.volumes = out.volumes
        return result


@register_sop
class VolumeWrangleSOP(SOP):
    id = 'volumewrangle'
    label = "Volume Wrangle"
    category = 'Volume'
    inputs = 4
    input_labels = ("Volumes", "Input 2", "Input 3", "Input 4")
    icon = 'SCRIPT'
    time_dependent = True
    description = ("VEX over every voxel. Volumes are bound by name (@density, v@vel ...), @P is the voxel center, "
                   "@ix @iy @iz the voxel index. Writing a new @name creates a volume.")
    parms = (
        Parm('snippet', "VEXpression", 'code',
             default='// Voxels: @P, @density, @ix @iy @iz\n@density = 1.0 - length(@P);\n'),
        Parm('volume', "Reference Volume", 'string', default='',
             description="Volume whose grid is used (default: the first one)"),
    )

    def cook(self, inputs, p, ctx):
        from ..vex.wrangle import run_wrangle
        from ..vex.parser import VexSyntaxError
        from ..vex.values import VexError
        src = _in(inputs)
        if not src.volumes:
            raise CookError("Volume Wrangle needs a volume on the first input (use the Volume SOP)")
        ref = next((v for v in src.volumes if v.name == p.volume.strip()), src.volumes[0]) if p.volume.strip() \
            else src.volumes[0]
        res = tuple(ref.res)
        same = [v for v in src.volumes if tuple(v.res) == res and np.allclose(v.origin, ref.origin)
                and abs(v.voxelsize - ref.voxelsize) < 1e-12]
        pts = Geometry()
        pts.P = ref.centers()
        idx = np.indices(res).reshape(3, -1)
        for k, nm in enumerate(('ix', 'iy', 'iz')):
            pts.attribs[POINT][nm] = idx[k].astype(np.int64)
        for v in same:
            pts.attribs[POINT][v.name] = v.data.reshape(len(pts.P), -1)[:, 0] if not v.is_vector \
                else v.data.reshape(-1, 3).copy()
        pts.volumes = list(src.volumes)
        ins = [pts] + list(inputs[1:]) + [None] * (3 - len(inputs[1:]))
        try:
            geo, messages = run_wrangle(p.snippet, ins, run_over='point', channels=getattr(ctx, 'channels', None),
                                        ramps=getattr(ctx, 'ramps', None), time=ctx.time, frame=ctx.frame, fps=ctx.fps)
        except (VexSyntaxError, VexError) as ex:
            raise CookError(str(ex))
        for m in messages:
            ctx.log(m)
        by_name = {v.name: v for v in same}
        new_vols = []
        written = set()
        for v in src.volumes:
            if v.name in by_name and v.name in geo.attribs[POINT]:
                arr = geo.attribs[POINT][v.name]
                new_vols.append(v.copy(data=np.asarray(arr, dtype=np.float64).reshape(v.data.shape)))
                written.add(v.name)
            else:
                new_vols.append(v)
        for name, arr in geo.attribs[POINT].items():
            if name in ('P', 'ix', 'iy', 'iz') or name in written or name in by_name or arr.dtype == object:
                continue
            if arr.ndim == 1 or (arr.ndim == 2 and arr.shape[1] == 3):
                shape = res + (() if arr.ndim == 1 else (3,))
                new_vols.append(Volume(name, np.asarray(arr, dtype=np.float64).reshape(shape), ref.origin,
                                       ref.voxelsize, 'fog'))
        out = src.copy()
        out.volumes = new_vols
        return out


@register_sop
class VolumeBlurSOP(SOP):
    id = 'volumeblur'
    label = "Volume Blur"
    category = 'Volume'
    inputs = 1
    icon = 'MOD_SMOOTH'
    description = "Box blur volumes (repeated 3x3x3 averaging)"
    parms = (
        Parm('group', "Volumes (names)", 'string', default='*'),
        Parm('iterations', "Iterations", 'int', default=2, min=1, max=50),
    )

    def cook(self, inputs, p, ctx):
        import fnmatch
        src = _in(inputs)
        out = src.copy()
        pats = (p.group.strip() or '*').split()
        vols = []
        for v in src.volumes:
            if not any(fnmatch.fnmatchcase(v.name, pat) for pat in pats):
                vols.append(v)
                continue
            d = v.data.copy()
            for _ in range(p.iterations):
                for axis in range(3):
                    pad = [(0, 0)] * d.ndim
                    pad[axis] = (1, 1)
                    e = np.pad(d, pad, mode='edge')
                    sl = [slice(None)] * d.ndim
                    parts = []
                    for off in range(3):
                        sl[axis] = slice(off, off + d.shape[axis])
                        parts.append(e[tuple(sl)])
                    d = (parts[0] + parts[1] + parts[2]) / 3.0
            vols.append(v.copy(data=d))
        out.volumes = vols
        return out


@register_sop
class VolumeRasterizePointsSOP(SOP):
    id = 'volumerasterizepoints'
    label = "Volume Rasterize Points"
    category = 'Volume'
    inputs = 1
    icon = 'PARTICLES'
    description = "Splat points into a fog volume (smooth falloff over pscale), optionally also v as a vector field"
    parms = (
        Parm('name', "Name", 'string', default='density'),
        Parm('voxelsize', "Voxel Size", 'float', default=0.05, min=0.002, soft_max=1.0),
        Parm('radius', "Particle Radius Scale", 'float', default=0.1, min=0.0001, soft_max=2.0,
             description="Multiplied by pscale"),
        Parm('density', "Density Scale", 'float', default=1.0, min=0.0),
        Parm('velocity', "Also Rasterize v", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        src = _in(inputs)
        out = src.copy()
        n = src.num_points
        if n == 0:
            return out
        pscale = src.attribs[POINT].get('pscale')
        rad = (pscale if pscale is not None and pscale.dtype != object else np.ones(n)) * p.radius
        rad = np.maximum(rad, 1e-6)
        lo = (src.P - rad[:, None]).min(axis=0)
        hi = (src.P + rad[:, None]).max(axis=0)
        try:
            vol = Volume.from_bounds(p.name.strip() or 'density', lo, hi, p.voxelsize)
        except ValueError as ex:
            raise CookError(str(ex))
        res = vol.res
        dens = np.zeros(tuple(res))
        vel_on = p.velocity and 'v' in src.attribs[POINT]
        vacc = np.zeros(tuple(res) + (3,)) if vel_on else None
        vsrc = src.attribs[POINT]['v'] if vel_on else None
        vs = vol.voxelsize
        for i in range(n):
            r = rad[i]
            c = src.P[i]
            i0 = np.maximum(np.floor((c - r - vol.origin) / vs).astype(np.int64), 0)
            i1 = np.minimum(np.ceil((c + r - vol.origin) / vs).astype(np.int64), res)
            if np.any(i1 <= i0):
                continue
            grids = np.meshgrid(*[np.arange(i0[k], i1[k]) for k in range(3)], indexing='ij')
            pos = vol.origin + (np.stack(grids, axis=-1) + 0.5) * vs
            d = np.linalg.norm(pos - c, axis=-1) / r
            w = np.clip(1 - d * d, 0, None) ** 2
            sl = tuple(slice(i0[k], i1[k]) for k in range(3))
            dens[sl] += w * p.density
            if vel_on:
                vacc[sl] += w[..., None] * vsrc[i]
        vol.data = dens
        out.volumes = list(src.volumes) + [vol]
        if vel_on:
            vel = vacc / np.maximum(dens / max(p.density, 1e-9), 1e-9)[..., None]
            out.volumes.append(Volume('vel', vel, vol.origin, vs, 'fog'))
        return out
