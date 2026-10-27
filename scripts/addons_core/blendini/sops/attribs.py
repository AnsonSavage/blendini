# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Attribute SOPs and the VEX wrangles."""

import fnmatch

import numpy as np

from .base import SOP, Parm, register_sop, parse_group_pattern, CookError
from ..core.geometry import Geometry, POINT, PRIM, VERTEX, DETAIL
from ..core import noise as _noise

CLASS_ITEMS = [('point', "Points", ""), ('prim', "Primitives", ""), ('vertex', "Vertices", ""), ('detail', "Detail", "")]
RUN_OVER_ITEMS = [('point', "Points", ""), ('prim', "Primitives", ""), ('vertex', "Vertices", ""),
                  ('detail', "Detail (only once)", ""), ('numbers', "Numbers", "Run a fixed number of times")]


def _in(inputs, i=0):
    if len(inputs) > i and inputs[i] is not None:
        return inputs[i]
    return Geometry()


def _class_mask(geo, cls, pattern):
    if cls in (POINT, PRIM):
        return parse_group_pattern(geo, cls, pattern)
    return np.ones(geo.count(cls), dtype=bool)


@register_sop
class AttribCreateSOP(SOP):
    id = 'attribcreate'
    label = "Attribute Create"
    category = 'Attribute'
    inputs = 1
    icon = 'ADD'
    parms = (
        Parm('group', "Group", 'string'),
        Parm('name', "Name", 'string', default='attrib'),
        Parm('class', "Class", 'enum', items=CLASS_ITEMS),
        Parm('type', "Type", 'enum', items=[
            ('float', "Float", ""), ('int', "Integer", ""), ('vector', "Vector", ""), ('string', "String", ""),
        ]),
        Parm('value', "Value", 'vector', hide_when={'type': 'string'}),
        Parm('string', "String", 'string', hide_when={'type': ('float', 'int', 'vector')}),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        cls = p['class']
        name = p.name.strip() or 'attrib'
        mask = _class_mask(geo, cls, p.group)
        if p.type == 'float':
            arr = geo.add_attrib(cls, name, shape=(), dtype=np.float64)
            val = p.value[0]
        elif p.type == 'int':
            arr = geo.add_attrib(cls, name, shape=(), dtype=np.int64)
            val = int(p.value[0])
        elif p.type == 'vector':
            arr = geo.add_attrib(cls, name, shape=(3,), dtype=np.float64)
            val = np.asarray(p.value)
        else:
            arr = geo.add_attrib(cls, name, shape=(), dtype=object)
            val = p.string
        if cls == DETAIL:
            arr[0] = val
        else:
            arr[mask] = val
        return geo


@register_sop
class AttribDeleteSOP(SOP):
    id = 'attribdelete'
    label = "Attribute Delete"
    category = 'Attribute'
    inputs = 1
    icon = 'X'
    parms = (
        Parm('ptdel', "Point Attributes", 'string'),
        Parm('primdel', "Primitive Attributes", 'string'),
        Parm('vtxdel', "Vertex Attributes", 'string'),
        Parm('dtldel', "Detail Attributes", 'string'),
        Parm('negate', "Delete Non-Selected", 'bool', default=False),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        for cls, pattern in ((POINT, p.ptdel), (PRIM, p.primdel), (VERTEX, p.vtxdel), (DETAIL, p.dtldel)):
            pats = pattern.split()
            if not pats and not p.negate:
                continue
            for name in list(geo.attribs[cls]):
                if name == 'P':
                    continue
                hit = any(fnmatch.fnmatchcase(name, pat) for pat in pats)
                if hit != p.negate:
                    del geo.attribs[cls][name]
        return geo


@register_sop
class AttribRenameSOP(SOP):
    id = 'attribrename'
    label = "Attribute Rename"
    category = 'Attribute'
    inputs = 1
    icon = 'SORTALPHA'
    parms = (
        Parm('class', "Class", 'enum', items=CLASS_ITEMS),
        Parm('from', "From", 'string'),
        Parm('to', "To", 'string'),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        cls = p['class']
        src, dst = p['from'].strip(), p['to'].strip()
        if src and dst and src in geo.attribs[cls] and src != 'P':
            geo.attribs[cls][dst] = geo.attribs[cls].pop(src)
        return geo


@register_sop
class AttribPromoteSOP(SOP):
    id = 'attribpromote'
    label = "Attribute Promote"
    category = 'Attribute'
    inputs = 1
    icon = 'SORT_DESC'
    parms = (
        Parm('names', "Original Name", 'string', default='Cd'),
        Parm('inclass', "Original Class", 'enum', items=CLASS_ITEMS),
        Parm('outclass', "New Class", 'enum', items=[CLASS_ITEMS[1], CLASS_ITEMS[0], CLASS_ITEMS[2], CLASS_ITEMS[3]]),
        Parm('method', "Promotion Method", 'enum', items=[
            ('mean', "Average", ""), ('min', "Minimum", ""), ('max', "Maximum", ""), ('sum', "Sum", ""),
            ('first', "First Match", ""),
        ]),
        Parm('deleteorig', "Delete Original", 'bool', default=True),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        for name in p.names.split():
            arr = geo.attribs[p.inclass].get(name)
            if arr is None or arr.dtype == object:
                continue
            out = promote(geo, arr, p.inclass, p.outclass, p.method)
            if p.deleteorig and name != 'P':
                del geo.attribs[p.inclass][name]
            if name == 'P' and p.outclass == POINT:
                continue
            geo.attribs[p.outclass][name] = out
        return geo


def promote(geo, arr, src, dst, method='mean'):
    """Move attribute values between classes."""
    vp = geo.vertex_prim()
    vpt = geo.vtx_point
    # Express source as per-vertex values when needed.
    if src == dst:
        return arr.copy()
    if dst == DETAIL:
        if len(arr) == 0:
            return np.zeros((1,) + arr.shape[1:], dtype=arr.dtype)
        fn = {'mean': np.mean, 'min': np.min, 'max': np.max, 'sum': np.sum, 'first': lambda a, axis: a[0]}[method]
        return np.asarray(fn(arr, axis=0), dtype=np.float64 if method == 'mean' else arr.dtype)[None]
    if src == DETAIL:
        return np.repeat(arr[:1], geo.count(dst), axis=0)
    if src == POINT:
        per_vtx = arr[vpt]
    elif src == PRIM:
        per_vtx = arr[vp]
    else:
        per_vtx = arr
    if dst == VERTEX:
        return per_vtx.copy()
    target = vpt if dst == POINT else vp
    n = geo.count(dst)
    shape = (n,) + arr.shape[1:]
    if method in ('mean', 'sum'):
        acc = np.zeros(shape)
        np.add.at(acc, target, per_vtx)
        if method == 'mean':
            cnt = np.bincount(target, minlength=n).reshape((-1,) + (1,) * (arr.ndim - 1))
            acc = acc / np.maximum(cnt, 1)
        return acc.astype(np.float64 if method == 'mean' else arr.dtype)
    if method == 'min':
        acc = np.full(shape, np.inf)
        np.minimum.at(acc, target, per_vtx)
    elif method == 'max':
        acc = np.full(shape, -np.inf)
        np.maximum.at(acc, target, per_vtx)
    else:
        acc = np.zeros(shape)
        acc[target[::-1]] = per_vtx[::-1]
    acc[~np.isfinite(acc)] = 0
    return acc.astype(arr.dtype)


@register_sop
class AttribRandomizeSOP(SOP):
    id = 'attribrandomize'
    label = "Attribute Randomize"
    category = 'Attribute'
    inputs = 1
    icon = 'MOD_NOISE'
    parms = (
        Parm('group', "Group", 'string'),
        Parm('name', "Attribute Name", 'string', default='Cd'),
        Parm('class', "Class", 'enum', items=CLASS_ITEMS[:2]),
        Parm('dimension', "Dimensions", 'int', default=3, min=1, max=4),
        Parm('min', "Min Value", 'vector'),
        Parm('max', "Max Value", 'vector', default=(1.0, 1.0, 1.0)),
        Parm('seed', "Global Seed", 'int', default=0),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        cls = p['class']
        n = geo.count(cls)
        mask = _class_mask(geo, cls, p.group)
        dims = int(np.clip(p.dimension, 1, 3))
        ids = geo.attribs[cls].get('id', np.arange(n))
        seed = np.stack([ids.astype(np.float64), np.full(n, float(p.seed))], axis=1)
        r = _noise.rand(seed, dims)
        lo = np.asarray(p.min)[:dims]
        hi = np.asarray(p.max)[:dims]
        vals = lo + (hi - lo) * (r if dims > 1 else r[:, None])
        if dims == 1:
            vals = vals[:, 0]
        arr = geo.attribs[cls].get(p.name)
        if arr is None or arr.shape[1:] != vals.shape[1:]:
            arr = np.zeros_like(vals)
        arr = arr.astype(np.float64)
        arr[mask] = vals[mask]
        geo.attribs[cls][p.name] = arr
        return geo


@register_sop
class ColorSOP(SOP):
    id = 'color'
    label = "Color"
    category = 'Attribute'
    inputs = 1
    icon = 'COLOR'
    parms = (
        Parm('group', "Group", 'string'),
        Parm('class', "Class", 'enum', items=CLASS_ITEMS),
        Parm('colortype', "Color Type", 'enum', items=[
            ('constant', "Constant", ""), ('random', "Random", ""), ('ramp', "Ramp from Attribute", ""),
        ]),
        Parm('color', "Color", 'color', default=(1.0, 1.0, 1.0), hide_when={'colortype': ('random', 'ramp')}),
        Parm('seed', "Seed", 'int', default=0, hide_when={'colortype': ('constant', 'ramp')}),
        Parm('rampattribute', "Attribute", 'string', default='P', hide_when={'colortype': ('constant', 'random')}),
        Parm('rampcomp', "Component", 'int', default=1, min=0, max=2, hide_when={'colortype': ('constant', 'random')}),
    )

    def cook(self, inputs, p, ctx):
        geo = _in(inputs).copy()
        cls = p['class']
        n = geo.count(cls)
        mask = _class_mask(geo, cls, p.group)
        arr = geo.attribs[cls].get('Cd')
        if arr is None or arr.shape[1:] != (3,):
            arr = np.ones((n, 3))
        arr = arr.astype(np.float64).copy()
        if p.colortype == 'constant':
            arr[mask] = p.color
        elif p.colortype == 'random':
            seed = np.stack([np.arange(n, dtype=np.float64), np.full(n, float(p.seed))], axis=1)
            arr[mask] = _noise.rand(seed, 3)[mask]
        else:
            src = geo.attribs[cls].get(p.rampattribute)
            if src is None and cls == PRIM and p.rampattribute == 'P':
                src = geo.prim_centers()
            if src is not None and src.dtype != object:
                v = src.astype(np.float64)
                if v.ndim > 1:
                    v = v[:, min(p.rampcomp, v.shape[1] - 1)]
                lo, hi = (v.min(), v.max()) if len(v) else (0, 1)
                u = (v - lo) / (hi - lo if hi > lo else 1)
                arr[mask] = _infra_red(u)[mask]
        geo.attribs[cls]['Cd'] = arr
        return geo


def _infra_red(u):
    """Houdini's default 'infra-red' false color ramp."""
    keys = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    cols = np.array([[0.2, 0.0, 1.0], [0.0, 0.85, 1.0], [0.0, 1.0, 0.1], [0.95, 1.0, 0.0], [1.0, 0.0, 0.0]])
    return np.stack([np.interp(u, keys, cols[:, k]) for k in range(3)], axis=1)


# ----------------------------------------------------------------------------
# Wrangles


WRANGLE_HELP = "VEX snippet. Use @attrib bindings, ch(\"name\") for spare parameters"


class _WrangleBase(SOP):
    category = 'VEX'
    inputs = 4
    input_labels = ("Input 1", "Input 2", "Input 3", "Input 4")
    icon = 'SCRIPT'
    default_class = 'point'
    default_code = ''
    time_dependent = True

    @classmethod
    def make_parms(cls):
        return (
            Parm('group', "Group", 'string'),
            Parm('class', "Run Over", 'enum', items=RUN_OVER_ITEMS, default=cls.default_class),
            Parm('snippet', "VEXpression", 'code', default=cls.default_code, description=WRANGLE_HELP),
            Parm('numcount', "Number Count", 'int', default=10, min=0, hide_when={'class': ('point', 'prim', 'vertex', 'detail')}),
        )

    def cook(self, inputs, p, ctx):
        from ..vex.wrangle import run_wrangle
        from ..vex.parser import VexSyntaxError
        from ..vex.values import VexError
        inputs = list(inputs) + [None] * (4 - len(inputs))
        if inputs[0] is None:
            inputs[0] = Geometry()
        try:
            geo, messages = run_wrangle(
                p.snippet, inputs, run_over=p['class'], group=p.group.strip() or None, count=p.get('numcount', 10),
                channels=getattr(ctx, 'channels', None), ramps=getattr(ctx, 'ramps', None),
                time=ctx.time, frame=ctx.frame, fps=ctx.fps,
            )
        except (VexSyntaxError, VexError) as ex:
            raise CookError(str(ex))
        for m in messages:
            ctx.log(m)
        return geo


@register_sop
class AttribWrangleSOP(_WrangleBase):
    id = 'attribwrangle'
    label = "Attribute Wrangle"
    default_code = '@P.y += sin(@P.x * 2.0 + @Time) * 0.5;\n'


AttribWrangleSOP.parms = AttribWrangleSOP.make_parms()


@register_sop
class PointWrangleSOP(_WrangleBase):
    id = 'pointwrangle'
    label = "Point Wrangle"
    default_code = (
        '// Point Wrangle: runs once per point (in parallel)\n'
        'float freq = ch("freq");\n'
        'float amp = ch("amp");\n'
        '@P += @N * noise(@P * freq + @Time) * amp;\n'
        '@Cd = set(fit01(@P.y, 0, 1), 0.3, 1 - @P.y);\n'
    )


PointWrangleSOP.parms = PointWrangleSOP.make_parms()


@register_sop
class PrimWrangleSOP(_WrangleBase):
    id = 'primwrangle'
    label = "Primitive Wrangle"
    default_class = 'prim'
    default_code = '@Cd = rand(@primnum);\n'


PrimWrangleSOP.parms = PrimWrangleSOP.make_parms()


@register_sop
class DetailWrangleSOP(_WrangleBase):
    id = 'detailwrangle'
    label = "Detail Wrangle"
    default_class = 'detail'
    default_code = (
        '// Runs once. Build geometry from scratch:\n'
        'for (int i = 0; i < 100; i++) {\n'
        '    float t = i / 99.0 * 6.2831 * 3;\n'
        '    addpoint(0, set(cos(t), i * 0.02, sin(t)));\n'
        '}\n'
    )


DetailWrangleSOP.parms = DetailWrangleSOP.make_parms()


@register_sop
class SolverWrangleSOP(_WrangleBase):
    """A tiny SOP Solver: the snippet runs every frame on the previous frame's result."""
    id = 'solverwrangle'
    label = "Solver (VEX)"
    category = 'Simulation'
    inputs = 2
    input_labels = ("Initial State", "Collision / Extra")
    icon = 'PHYSICS'
    default_code = (
        '// Runs every frame on the previous frame\'s geometry.\n'
        'v@v += {0, -9.8, 0} * @TimeInc;\n'
        '@P += v@v * @TimeInc;\n'
        'if (@P.y < 0) { @P.y = 0; v@v.y *= -0.6; }\n'
    )

    @classmethod
    def make_parms(cls):
        return _WrangleBase.make_parms.__func__(cls) + (
            Parm('startframe', "Start Frame", 'int', default=1),
            Parm('substeps', "Sub Steps", 'int', default=1, min=1, max=20),
        )

    def cook(self, inputs, p, ctx):
        state = ctx.state
        frame = int(round(ctx.frame))
        start = int(p.startframe)
        init = _in(inputs, 0)
        key = (p.snippet, p['class'], p.group, p.substeps)
        if state.get('key') != key or state.get('init_id') != id(init) and frame <= start:
            state.clear()
            state['key'] = key
        state['init_id'] = id(init)
        cache = state.setdefault('frames', {})
        if frame <= start:
            cache.clear()
            cache[start] = init.copy()
            return init.copy()
        if frame in cache:
            return cache[frame]
        prev_frames = [f for f in cache if f < frame]
        f0 = max(prev_frames) if prev_frames else start
        geo = cache.get(f0, init.copy())
        if f0 not in cache:
            cache[f0] = geo
        sub = max(int(p.substeps), 1)

        class _Sub:
            pass
        for f in range(f0 + 1, frame + 1):
            for s in range(sub):
                sctx = _Sub()
                sctx.time = (f - 1 + (s + 1) / sub) / ctx.fps
                sctx.frame = f - 1 + (s + 1) / sub
                sctx.fps = ctx.fps * sub
                sctx.channels = getattr(ctx, 'channels', None)
                sctx.ramps = getattr(ctx, 'ramps', None)
                sctx.log = ctx.log
                geo = _WrangleBase.cook(self, [geo] + list(inputs[1:]), p, sctx)
            cache[f] = geo
        # Keep memory bounded.
        if len(cache) > 500:
            for k in sorted(cache)[:-250]:
                if k != start:
                    del cache[k]
        return geo


SolverWrangleSOP.parms = SolverWrangleSOP.make_parms()


@register_sop
class AttribVOPSOP(_WrangleBase):
    """Visual VEX: the VOP network inside is turned into VEX code"""
    id = 'attribvop'
    label = "Attribute VOP"
    icon = 'NODE_MATERIAL'
    parms = (
        Parm('group', "Group", 'string'),
        Parm('class', "Run Over", 'enum', items=RUN_OVER_ITEMS[:4]),
        Parm('voptree', "VOP Network", 'vopnetwork', description="The VOP network (dive in with I)"),
    )

    def cook(self, inputs, p, ctx):
        if not p.get('snippet'):
            return _in(inputs)
        p = dict(p)
        p['numcount'] = 0
        from . import ParmValues
        return _WrangleBase.cook(self, inputs, ParmValues(p), ctx)
