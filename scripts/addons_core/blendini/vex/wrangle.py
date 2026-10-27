# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Run a VEX snippet over geometry (the Attribute Wrangle SOP)."""

import numpy as np

from ..core.geometry import Geometry, POINT, PRIM, VERTEX, DETAIL, VECTOR_ATTRIBS, VECTOR4_ATTRIBS, INT_ATTRIBS, STRING_ATTRIBS
from .parser import parse
from .interp import Interpreter
from .values import V, VexError, SHAPES, is_array, elem_type, convert, full, type_of_array, dtype_for, zeros, const

_cache = {}


def compile_cached(code):
    prog = _cache.get(code)
    if prog is None:
        prog = parse(code)
        if len(_cache) > 64:
            _cache.clear()
        _cache[code] = prog
    return prog


class WrangleContext:
    def __init__(self, inputs, run_over, elems, channels=None, ramps=None, time=0.0, frame=1.0, fps=24.0):
        self.inputs = inputs
        self.geo_in = inputs[0] if inputs and inputs[0] is not None else Geometry()
        self.run_over = run_over
        self.elems = elems
        self.channels = channels or {}
        self.ramps = ramps or {}
        self.time = time
        self.frame = frame
        self.fps = fps
        self.messages = []
        self.cache = {}
        # Deferred geometry edits.
        self.new_points = []      # list of (positions (k,3), copy_from (k,) or None)
        self.num_new_points = 0
        self.new_prims = []       # list of (pts list, closed)
        self.new_vertices = []    # (prim, pt)
        self.removed = {POINT: [], PRIM: []}
        self.remove_prim_points = False
        self.sets = []            # (cls, name, idx, type, values, mode)
        self.groups = []          # (cls, name, idx, values)

    # -- reading

    def input(self, idx):
        if 0 <= idx < len(self.inputs):
            return self.inputs[idx]
        return None

    def read_attrib(self, geo, cls, name):
        if geo is None:
            return None
        if cls == POINT and name == 'ptnum':
            return np.arange(geo.num_points)
        if cls == PRIM and name == 'primnum':
            return np.arange(geo.num_prims)
        return geo.attribs[cls].get(name)

    def channel(self, name):
        return self.channels.get(name)

    def ramp(self, name, x):
        keys = self.ramps.get(name)
        if not keys:
            return np.clip(x, 0, 1)
        keys = sorted(keys)
        xs = np.array([k[0] for k in keys])
        ys = np.array([k[1] for k in keys])
        return np.interp(x, xs, ys)

    def log(self, msg):
        if len(self.messages) < 1000:
            self.messages.append(str(msg))

    def implicit_type(self, name):
        if name in ('ptnum', 'primnum', 'vtxnum', 'elemnum', 'numpt', 'numprim', 'numvtx', 'numelem', 'id'):
            return 'int'
        if name.startswith('group_'):
            return 'int'
        if name in ('Time', 'Frame', 'TimeInc', 'SimTime', 'SimFrame'):
            return 'float'
        if name in VECTOR_ATTRIBS:
            return 'vector'
        if name in VECTOR4_ATTRIBS:
            return 'vector4'
        if name in STRING_ATTRIBS:
            return 'string'
        geo = self.geo_in
        for cls in (self.run_over, DETAIL):
            arr = geo.attribs[cls].get(name)
            if arr is not None:
                return type_of_array(arr)
        return 'float'

    def bind_attr(self, name, t, n):
        geo = self.geo_in
        elems = self.elems
        cls = self.run_over
        if name in ('ptnum', 'primnum', 'vtxnum', 'elemnum'):
            return V('int', elems.astype(np.int64))
        if name in ('numpt',):
            return const('int', geo.num_points)
        if name in ('numprim',):
            return const('int', geo.num_prims)
        if name in ('numvtx',):
            return const('int', geo.num_vertices)
        if name == 'numelem':
            return const('int', geo.count(cls))
        if name == 'Time':
            return const('float', self.time)
        if name == 'Frame':
            return const('float', self.frame)
        if name == 'TimeInc':
            return const('float', 1.0 / self.fps)
        if name.startswith('group_') and cls in (POINT, PRIM):
            g = geo.groups[cls].get(name[6:])
            if g is None:
                return V('int', np.zeros(n, dtype=np.int64))
            return V('int', g[elems].astype(np.int64))
        if name.startswith('opinput') and '_' in name:
            # @opinput1_P: same element of another input.
            idx_s, _, aname = name[7:].partition('_')
            other = self.input(int(idx_s)) if idx_s.isdigit() else None
            if other is None:
                return zeros(t, n)
            arr = other.attribs[cls].get(aname)
            if arr is None or len(arr) == 0:
                return zeros(t, n)
            return V(type_of_array(arr), arr[np.clip(elems, 0, len(arr) - 1)])
        arr = geo.attribs[cls].get(name)
        if arr is not None:
            return V(type_of_array(arr) if not is_array(t) else t, arr[elems])
        if cls == PRIM and name == 'P':
            return V('vector', geo.prim_centers()[elems])
        if cls == PRIM and name == 'N':
            return V('vector', geo.prim_normals()[elems])
        if cls == POINT and name == 'N':
            # Like Houdini, reading @N without a normal attribute gives computed normals.
            return V('vector', geo.point_normals()[elems])
        if cls == VERTEX and name == 'P':
            return V('vector', geo.P[geo.vtx_point[elems]])
        darr = geo.attribs[DETAIL].get(name)
        if darr is not None and len(darr):
            return V(type_of_array(darr) if not is_array(t) else t, np.repeat(darr[:1], n, axis=0))
        if name == 'Cd':
            return V('vector', np.ones((n, 3)))
        if name in ('pscale',):
            return V('float', np.ones(n))
        if name == 'scale':
            return V('vector', np.ones((n, 3)))
        if name == 'orient':
            d = np.zeros((n, 4))
            d[:, 3] = 1
            return V('vector4', d)
        return zeros(t, n)

    # -- deferred writing

    def add_points(self, positions, copy_from=None):
        if positions is None:
            src = self.geo_in.P
            safe = np.clip(copy_from, 0, max(len(src) - 1, 0))
            positions = src[safe] if len(src) else np.zeros((len(copy_from), 3))
        k = len(positions)
        start = self.geo_in.num_points + self.num_new_points
        self.new_points.append((np.asarray(positions, dtype=np.float64), copy_from))
        self.num_new_points += k
        return np.arange(start, start + k)

    def add_prims(self, pt_lists, closed):
        start = self.geo_in.num_prims + len(self.new_prims)
        for pts in pt_lists:
            self.new_prims.append((list(pts), closed))
        return np.arange(start, start + len(pt_lists))

    def add_vertex(self, prim, pt):
        base = self.geo_in.num_prims
        if prim >= base and prim - base < len(self.new_prims):
            self.new_prims[prim - base][0].append(pt)
        else:
            self.new_vertices.append((prim, pt))

    def vertex_index(self, prim, vtx):
        geo = self.geo_in
        p = np.clip(prim, 0, max(geo.num_prims - 1, 0))
        return geo.prim_start[p] + vtx

    def remove(self, cls, idx, andpoints=False):
        self.removed[cls].append(np.asarray(idx, dtype=np.int64))
        if andpoints:
            self.remove_prim_points = True

    def queue_set(self, cls, name, idx, t, values, mode):
        self.sets.append((cls, name, np.asarray(idx, dtype=np.int64), t, values, mode))

    def queue_group(self, cls, name, idx, values):
        self.groups.append((cls, name, np.asarray(idx, dtype=np.int64), values))

    def finalize(self, interp):
        geo = self.geo_in.copy()
        cls = self.run_over
        elems = self.elems
        # 1. Write @ bindings.
        for name in interp.written:
            v = interp.attrs[name]
            if name.startswith('opinput'):
                continue
            if name in ('ptnum', 'primnum', 'vtxnum', 'elemnum', 'numpt', 'numprim', 'numvtx', 'numelem',
                        'Time', 'Frame', 'TimeInc'):
                continue
            data = full(v, len(elems))
            if name.startswith('group_') and cls in (POINT, PRIM):
                gname = name[6:]
                g = geo.groups[cls].setdefault(gname, np.zeros(geo.count(cls), dtype=bool))
                g[elems] = data != 0
                continue
            if cls == PRIM and name == 'P' and not geo.has_attrib(PRIM, 'P'):
                # Moving a primitive's P translates its points (centroid offset).
                continue
            _write_attrib(geo, cls, name, v.t, elems, data)
        # 2. New geometry.
        if self.new_points:
            positions = np.concatenate([p for p, _ in self.new_points])
            start = geo.num_points
            geo.add_points(positions)
            # Copy attributes when created from an existing point.
            offset = start
            for pos, copy_from in self.new_points:
                if copy_from is not None:
                    src = np.clip(copy_from, 0, max(start - 1, 0))
                    for aname, arr in geo.attribs[POINT].items():
                        if aname != 'P' and start > 0:
                            arr[offset:offset + len(pos)] = arr[src]
                offset += len(pos)
        if self.new_prims:
            closed_flags = np.array([c for _, c in self.new_prims], dtype=bool)
            sizes = [len(p) for p, _ in self.new_prims]
            flat = np.array([x for p, _ in self.new_prims for x in p], dtype=np.int64)
            if len(flat) and (flat.max() >= geo.num_points or flat.min() < 0):
                bad = (flat >= geo.num_points) | (flat < 0)
                self.log("Warning: addprim() used %d invalid point numbers" % int(bad.sum()))
                flat = np.clip(flat, 0, max(geo.num_points - 1, 0))
            geo.add_prims_flat(sizes, flat, closed_flags)
        if self.new_vertices:
            # Appending vertices to existing prims: rebuild topology.
            polys = geo.prims_as_lists()
            for prim, pt in self.new_vertices:
                if 0 <= prim < len(polys):
                    polys[prim].append(pt)
            _rebuild(geo, polys)
        # 3. setattrib / setgroup calls.
        for scls, name, idx, t, values, mode in self.sets:
            n = geo.count(scls)
            ok = (idx >= 0) & (idx < n)
            if not ok.all():
                idx = idx[ok]
                values = values[ok]
            if len(idx) == 0 and scls != DETAIL:
                _ensure_attrib(geo, scls, name, t)
                continue
            if mode == 'add':
                arr = _ensure_attrib(geo, scls, name, t)
                vals = convert(V(t, values), type_of_array(arr)).d
                np.add.at(arr, idx, vals)
            elif mode in ('max', 'min', 'mult', 'multiply'):
                arr = _ensure_attrib(geo, scls, name, t)
                vals = convert(V(t, values), type_of_array(arr)).d
                fn = {'max': np.maximum, 'min': np.minimum, 'mult': np.multiply, 'multiply': np.multiply}[mode]
                fn.at(arr, idx, vals)
            else:
                _write_attrib(geo, scls, name, t, idx, values)
        for gcls, name, idx, values in self.groups:
            g = geo.groups[gcls].setdefault(name, np.zeros(geo.count(gcls), dtype=bool))
            ok = (idx >= 0) & (idx < len(g))
            g[idx[ok]] = values[ok]
        # 4. Removals.
        if self.removed[PRIM]:
            idx = np.unique(np.concatenate(self.removed[PRIM]))
            idx = idx[(idx >= 0) & (idx < geo.num_prims)]
            m = np.zeros(geo.num_prims, dtype=bool)
            m[idx] = True
            geo.delete_prims(m, keep_points=not self.remove_prim_points)
        if self.removed[POINT]:
            idx = np.unique(np.concatenate(self.removed[POINT]))
            idx = idx[(idx >= 0) & (idx < geo.num_points)]
            m = np.zeros(geo.num_points, dtype=bool)
            m[idx] = True
            geo.delete_points(m)
        return geo


def _ensure_attrib(geo, cls, name, t):
    arr = geo.attribs[cls].get(name)
    if arr is None:
        if is_array(t):
            arr = np.empty(geo.count(cls), dtype=object)
            for i in range(len(arr)):
                arr[i] = np.zeros(0)
            geo.attribs[cls][name] = arr
        else:
            arr = geo.add_attrib(cls, name, shape=SHAPES.get(t, ()), dtype=dtype_for(t))
    return arr


def _write_attrib(geo, cls, name, t, idx, data):
    arr = geo.attribs[cls].get(name)
    if arr is not None and not is_array(t):
        at = type_of_array(arr)
        if at != t:
            if name == 'P' or at in ('vector', 'vector4', 'vector2') and t in ('float', 'int'):
                data = convert(V(t, data), at).d
            else:
                # Type changed: recreate attribute with the new type.
                arr = None
    if arr is None:
        if name == 'P' and cls == POINT:
            arr = geo.attribs[POINT]['P']
        else:
            geo.attribs[cls].pop(name, None)
            arr = _ensure_attrib(geo, cls, name, t)
    if cls == DETAIL:
        if len(data):
            arr[0] = data[-1]
        return
    arr[idx] = data


def _rebuild(geo, polys):
    vtx_attribs = geo.attribs[VERTEX]
    geo.attribs[VERTEX] = {}
    closed = geo.prim_closed.copy()
    sizes = np.array([len(p) for p in polys], dtype=np.int64)
    geo.prim_start = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    geo.vtx_point = np.array([x for p in polys for x in p], dtype=np.int64)
    geo.prim_closed = closed
    del vtx_attribs


def run_wrangle(code, inputs, run_over=POINT, group=None, channels=None, ramps=None, time=0.0, frame=1.0, fps=24.0):
    """Execute VEX code. Returns (geometry, messages)."""
    prog = compile_cached(code)
    geo = inputs[0] if inputs and inputs[0] is not None else Geometry()
    inputs = list(inputs) + [None] * (4 - len(inputs))
    inputs[0] = geo
    if run_over == DETAIL:
        elems = np.zeros(1, dtype=np.int64)
    else:
        count = geo.count(run_over)
        elems = np.arange(count)
        if group:
            g = geo.groups.get(run_over, {}).get(group)
            if g is not None:
                elems = np.nonzero(g)[0]
    ctx = WrangleContext(inputs, run_over, elems, channels, ramps, time, frame, fps)
    n = len(elems)
    if n == 0:
        return geo.copy(), ctx.messages
    interp = Interpreter(prog, ctx, n)
    interp.run()
    return ctx.finalize(interp), ctx.messages


def find_channels(code):
    """Return [(name, kind)] for ch*() calls with literal names (for spare parameters)."""
    import re
    found = []
    seen = set()
    for m in re.finditer(r'\b(ch|chf|chi|chv|chs|chu|chp|chramp)\s*\(\s*["\']([^"\']+)["\']', code):
        fn, name = m.groups()
        if name in seen:
            continue
        seen.add(name)
        kind = {'ch': 'float', 'chf': 'float', 'chi': 'int', 'chv': 'vector', 'chs': 'string', 'chu': 'vector2',
                'chp': 'vector4', 'chramp': 'ramp'}[fn]
        found.append((name, kind))
    return found
