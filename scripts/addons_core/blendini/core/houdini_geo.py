# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Houdini ASCII ``.geo`` (JSON geometry) reader and writer. Pure Python + numpy.

The file is a JSON list of alternating keys and values::

    ["fileversion", "19.5.303", "pointcount", 8, "vertexcount", 24, "primitivecount", 6,
     "topology", ["pointref", ["indices", [...vertex -> point...]]],
     "attributes", ["pointattributes", [...], "vertexattributes", [...],
                    "primitiveattributes", [...], "globalattributes", [...]],
     "primitives", [[header, body], ...],
     "pointgroups", [...], "primitivegroups", [...]]

Houdini writes many "maps" either as JSON objects or as flat [key, value, ...] lists; both are
accepted everywhere. Supported:

- primitives: ``Poly`` (closed or open), ``PolygonCurve`` (open), ``run`` with
  ``runtype`` Poly / PolygonCurve (``varyingfields`` / ``uniformfields``), and the compact
  ``Polygon_run`` / ``PolygonCurve_run`` (``startvertex``, ``nprimitives``, ``nvertices`` or
  ``nvertices_rle``). Other primitive types are skipped (their primitive attributes and group
  memberships are dropped, their vertices are not used).
- numeric attributes (``tuples``, ``arrays``, ``rawpagedata`` with ``pagesize``, ``packing`` and
  ``constantpageflags``), sizes 1-4, 9 and 16 (9 / 16 become 3x3 / 4x4 matrices), on points,
  vertices, primitives and detail; string attributes (``strings`` + ``indices``).
  P of size 4 (homogeneous) drops w.
- point and primitive groups (``unordered`` / ``ordered`` selections stored as ``i8``,
  ``bools`` or ``boolRLE``).

The writer emits the classic layout (Poly runs, tuples, i8 group selections) that Houdini reads.
Packed instances and volumes are not written.
"""

import json

import numpy as np

from .geometry import Geometry, POINT, PRIM, VERTEX, DETAIL

FILE_VERSION = "18.5.0"

_ATTR_KEYS = (('pointattributes', POINT), ('vertexattributes', VERTEX),
              ('primitiveattributes', PRIM), ('globalattributes', DETAIL))
_GROUP_KEYS = (('pointgroups', POINT), ('primitivegroups', PRIM))
_POLY_TYPES = {'Poly': None, 'PolygonCurve': False}
_POLY_RUN_TYPES = {'Polygon_run': True, 'PolygonCurve_run': False}
_INT_STORAGE = {'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool'}
_TYPEINFO = {'P': 'point', 'N': 'normal', 'Cd': 'color', 'v': 'vector', 'up': 'vector',
             'uv': 'texturecoord', 'rest': 'point', 'orient': 'quaternion', 'rot': 'quaternion'}


# ----------------------------------------------------------------------------
# Reading

def _kv(obj):
    """A Houdini map: a dict, or a flat [key, value, key, value ...] list."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (list, tuple)):
        if len(obj) % 2 == 0 and all(isinstance(k, str) for k in obj[0::2]):
            return dict(zip(obj[0::2], obj[1::2]))
    return {}


def _is_int_storage(storage):
    return str(storage) in _INT_STORAGE


def _decode_values(vals, count, default_size=1):
    """Decode a "values" (or "indices") map to a (count, size) numpy array."""
    vals = _kv(vals)
    size = int(vals.get('size', default_size) or default_size)
    is_int = _is_int_storage(vals.get('storage', 'fpreal32'))
    dtype = np.int64 if is_int else np.float64
    if 'tuples' in vals:
        arr = np.asarray(vals['tuples'], dtype=dtype).reshape(-1, size)
    elif 'arrays' in vals:
        comps = [np.asarray(a, dtype=dtype).ravel() for a in vals['arrays']]
        if len(comps) == 1 and size > 1:
            arr = comps[0].reshape(-1, size)
        else:
            arr = np.stack(comps, axis=1) if comps else np.zeros((0, size), dtype)
    elif 'rawpagedata' in vals:
        arr = _decode_pages(vals, count, size, dtype)
    else:
        arr = np.zeros((0, size), dtype)
    if len(arr) < count:
        arr = np.concatenate([arr, np.zeros((count - len(arr), arr.shape[1]), dtype)])
    return arr[:count]


def _decode_pages(vals, count, size, dtype):
    raw = np.asarray(vals['rawpagedata'], dtype=dtype).ravel()
    pagesize = int(vals.get('pagesize', 1024) or 1024)
    packing = [int(x) for x in (vals.get('packing') or [size])]
    flags = vals.get('constantpageflags') or []
    npages = (count + pagesize - 1) // pagesize
    out = np.zeros((count, size), dtype)
    pos = 0
    for page in range(npages):
        lo = page * pagesize
        hi = min(count, lo + pagesize)
        comp = 0
        for si, sub in enumerate(packing):
            const = False
            if si < len(flags) and page < len(flags[si]):
                const = bool(flags[si][page])
            if const:
                out[lo:hi, comp:comp + sub] = raw[pos:pos + sub]
                pos += sub
            else:
                n = (hi - lo) * sub
                out[lo:hi, comp:comp + sub] = raw[pos:pos + n].reshape(hi - lo, sub)
                pos += n
            comp += sub
    if pos > len(raw):
        raise ValueError("rawpagedata is shorter than the attribute needs")
    return out


def _read_attrib(entry, count):
    """(name, array) for one [definition, value] attribute entry, or None if unsupported."""
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return None
    defn, value = _kv(entry[0]), _kv(entry[1])
    name = defn.get('name')
    atype = defn.get('type', 'numeric')
    if not name:
        return None
    if atype == 'numeric':
        size = int(value.get('size', 1) or 1)
        is_int = _is_int_storage(value.get('storage', 'fpreal32'))
        if 'values' in value:
            arr = _decode_values(value['values'], count, size)
        else:
            arr = np.zeros((count, size), np.int64 if is_int else np.float64)
            dflt = _kv(value.get('defaults')).get('values')
            if dflt:
                d = np.asarray(dflt, dtype=arr.dtype).ravel()
                arr[:] = d[:size] if len(d) >= size else d[0]
        if is_int:
            arr = arr.astype(np.int64)
        else:
            arr = arr.astype(np.float64)
        if name == 'P':
            if arr.shape[1] >= 3:
                arr = arr[:, :3]
            else:
                arr = np.concatenate([arr, np.zeros((count, 3 - arr.shape[1]))], axis=1)
            return name, arr
        if arr.shape[1] == 1:
            arr = arr[:, 0]
        elif arr.shape[1] == 9:
            arr = arr.reshape(count, 3, 3)
        elif arr.shape[1] == 16:
            arr = arr.reshape(count, 4, 4)
        return name, arr
    if atype == 'string':
        strings = [str(s) for s in (value.get('strings') or [])]
        idx = _decode_values(value.get('indices', {}), count, 1)[:, 0].astype(np.int64)
        table = np.array(strings + [''], dtype=object)
        idx = np.where((idx >= 0) & (idx < len(strings)), idx, len(strings))
        arr = np.empty(count, dtype=object)
        arr[:] = table[idx]
        return name, arr
    return None


def _decode_selection(sel, count):
    """Boolean membership array from a group "selection" map."""
    sel = _kv(sel)
    mask = np.zeros(count, dtype=bool)
    body = None
    for key in ('unordered', 'ordered'):
        if key in sel:
            body = sel[key]
            break
    if body is None:
        return mask
    if isinstance(body, (list, tuple)) and body and not isinstance(body[0], str):
        # Ordered selection as a plain index list.
        idx = np.asarray(body, dtype=np.int64)
        mask[idx[(idx >= 0) & (idx < count)]] = True
        return mask
    body = _kv(body)
    if 'i8' in body or 'bools' in body:
        bits = body.get('i8', body.get('bools'))
        if isinstance(bits, str):
            vals = np.array([c == '1' for c in bits if c in '01'], dtype=bool)
        else:
            vals = np.asarray(bits).astype(bool).ravel()
        n = min(count, len(vals))
        mask[:n] = vals[:n]
    elif 'boolRLE' in body:
        rle = body['boolRLE']
        pos = 0
        for run, val in zip(rle[0::2], rle[1::2]):
            run = int(run)
            if val:
                mask[pos:pos + run] = True
            pos += run
    elif 'indices' in body or 'i32' in body:
        idx = np.asarray(body.get('indices', body.get('i32')), dtype=np.int64).ravel()
        mask[idx[(idx >= 0) & (idx < count)]] = True
    return mask


def _read_primitives(prims, nverts):
    """Walk the primitive list.

    Returns (kept_mask, vertex_lists, closed_flags) where kept_mask has one entry per primitive
    in the file (including skipped ones), and vertex_lists / closed_flags one per kept polygon.
    """
    kept, vlists, closed = [], [], []

    def add(verts, cl):
        kept.append(True)
        vlists.append(np.asarray(verts, dtype=np.int64).ravel())
        closed.append(bool(cl))

    for entry in prims or ():
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        header, body = _kv(entry[0]), entry[1]
        ptype = header.get('type')
        if ptype == 'run':
            runtype = header.get('runtype')
            varying = list(header.get('varyingfields') or [])
            uniform = _kv(header.get('uniformfields'))
            body = body or []
            if runtype not in _POLY_TYPES:
                kept.extend([False] * len(body))
                continue
            for item in body:
                fields = dict(uniform)
                if isinstance(item, (list, tuple)):
                    fields.update(zip(varying, item))
                cl = fields.get('closed', _POLY_TYPES[runtype] is None)
                if _POLY_TYPES[runtype] is False:
                    cl = False
                add(fields.get('vertex', []), cl)
        elif ptype in _POLY_RUN_TYPES:
            b = _kv(body)
            n = int(b.get('nprimitives', 0) or 0)
            if 'nvertices' in b:
                sizes = [int(x) for x in b['nvertices']]
            elif 'nvertices_rle' in b:
                rle = b['nvertices_rle']
                sizes = []
                for size, rep in zip(rle[0::2], rle[1::2]):
                    sizes.extend([int(size)] * int(rep))
            else:
                sizes = []
            n = n or len(sizes)
            start = int(b.get('startvertex', 0) or 0)
            cl = _POLY_RUN_TYPES[ptype]
            if 'closed' in b and not isinstance(b['closed'], (list, tuple)):
                cl = bool(b['closed']) and cl
            for i in range(n):
                s = sizes[i] if i < len(sizes) else 0
                add(np.arange(start, start + s), cl)
                start += s
        elif ptype in _POLY_TYPES:
            b = _kv(body)
            cl = b.get('closed', True)
            if _POLY_TYPES[ptype] is False:
                cl = False
            add(b.get('vertex', []), cl)
        else:
            kept.append(False)
    for v in vlists:
        if len(v) and (v.min() < 0 or v.max() >= nverts):
            raise ValueError("primitive references vertex out of range")
    return np.asarray(kept, dtype=bool), vlists, closed


def parse_geo(data):
    """Build a Geometry from already parsed Houdini JSON geometry (the top-level list)."""
    if not isinstance(data, list) or not data or not isinstance(data[0], str):
        raise ValueError("not a Houdini JSON geometry file (expected a top-level key/value list)")
    top = _kv(data)
    if not top or not ({'fileversion', 'pointcount', 'topology'} & set(top)):
        raise ValueError("not a Houdini JSON geometry file (no fileversion / pointcount / topology)")
    npts = int(top.get('pointcount', 0) or 0)
    topo = _kv(_kv(top.get('topology')).get('pointref'))
    pointref = np.asarray(topo.get('indices', []), dtype=np.int64).ravel()
    nverts = int(top.get('vertexcount', len(pointref)) or len(pointref))
    if len(pointref) < nverts:
        raise ValueError("topology has %d vertices, expected %d" % (len(pointref), nverts))
    pointref = pointref[:nverts]
    if len(pointref) and (pointref.min() < 0 or pointref.max() >= npts):
        raise ValueError("topology references a point out of range")
    nprims_file = int(top.get('primitivecount', 0) or 0)

    kept, vlists, closed = _read_primitives(top.get('primitives'), nverts)
    nprims_file = max(nprims_file, len(kept))
    if len(kept) < nprims_file:
        kept = np.concatenate([kept, np.zeros(nprims_file - len(kept), bool)])

    geo = Geometry()
    geo.attribs[POINT]['P'] = np.zeros((npts, 3))
    order = np.concatenate(vlists).astype(np.int64) if vlists else np.zeros(0, np.int64)
    sizes = np.array([len(v) for v in vlists], dtype=np.int64)
    geo.prim_start = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    geo.vtx_point = pointref[order]
    geo.prim_closed = np.asarray(closed, dtype=bool)

    attrs = _kv(top.get('attributes'))
    counts = {POINT: npts, VERTEX: nverts, PRIM: nprims_file, DETAIL: 1}
    for key, cls in _ATTR_KEYS:
        for entry in attrs.get(key) or ():
            res = _read_attrib(entry, counts[cls])
            if res is None:
                continue
            name, arr = res
            if cls == VERTEX:
                arr = arr[order]
            elif cls == PRIM:
                arr = arr[kept]
            geo.attribs[cls][name] = arr

    for key, cls in _GROUP_KEYS:
        for entry in top.get(key) or ():
            if not isinstance(entry, (list, tuple)) or len(entry) < 2:
                continue
            name = _kv(entry[0]).get('name')
            if not name:
                continue
            mask = _decode_selection(_kv(entry[1]).get('selection'), counts[cls])
            geo.groups[cls][name] = mask[kept] if cls == PRIM else mask
    return geo


def read_geo(path):
    """Read a Houdini ASCII .geo file into a Geometry."""
    with open(path, 'rb') as fh:
        raw = fh.read()
    if raw[:1] in (b'\x7f', b'\x00') or raw[:4] == b'BloS':
        raise ValueError("%s is a binary Houdini file (.bgeo); only ASCII .geo is supported" % path)
    try:
        data = json.loads(raw.decode('utf-8', errors='replace'))
    except ValueError as ex:
        raise ValueError("%s is not a Houdini JSON geometry file: %s" % (path, ex))
    try:
        return parse_geo(data)
    except ValueError as ex:
        raise ValueError("%s: %s" % (path, ex))
    except (TypeError, KeyError, IndexError) as ex:
        raise ValueError("%s: malformed Houdini geometry (%s)" % (path, ex))


# ----------------------------------------------------------------------------
# Writing

def _attrib_entry(name, arr, cls):
    n = len(arr)
    if arr.dtype == object:
        if n and not all(isinstance(x, str) for x in arr.tolist()):
            return None
        strings, idx = np.unique(np.asarray([str(x) for x in arr.tolist()] or [''], dtype=object),
                                 return_inverse=True)
        if not n:
            strings, idx = [], np.zeros(0, np.int64)
        return [
            ["scope", "public", "type", "string", "name", name, "options", {}],
            ["size", 1, "storage", "int32", "strings", [str(s) for s in strings],
             "indices", ["size", 1, "storage", "int32", "arrays", [np.asarray(idx).astype(int).ravel().tolist()]]],
        ]
    if arr.dtype.kind not in 'biuf':
        return None
    flat = arr.reshape(n, -1) if n else arr.reshape(0, int(np.prod(arr.shape[1:])) if arr.ndim > 1 else 1)
    size = flat.shape[1]
    is_int = arr.dtype.kind in 'biu'
    storage = 'int32' if is_int else ('fpreal64' if name == 'P' or arr.dtype == np.float64 else 'fpreal32')
    if is_int:
        values = flat.astype(np.int64)
    else:
        values = flat.astype(np.float64)
    options = {}
    tinfo = _TYPEINFO.get(name)
    if size in (9, 16):
        tinfo = 'matrix'
    if tinfo and not is_int:
        options = {"type": {"type": "string", "value": tinfo}}
    if size == 1:
        vals = ["size", 1, "storage", storage, "arrays", [values[:, 0].tolist()]]
    else:
        vals = ["size", size, "storage", storage, "tuples", values.tolist()]
    return [
        ["scope", "public", "type", "numeric", "name", name, "options", options],
        ["size", size, "storage", storage,
         "defaults", ["size", 1, "storage", "fpreal64", "values", [0]],
         "values", vals],
    ]


def _group_entry(name, mask):
    return [["name", name], ["selection", ["unordered", ["i8", np.asarray(mask, dtype=np.int64).tolist()]]]]


def build_geo(geo):
    """The JSON-ready top-level list for a Geometry."""
    nv = geo.num_vertices
    sizes = geo.prim_sizes()
    prims = []
    i = 0
    n = geo.num_prims
    while i < n:
        cl = bool(geo.prim_closed[i])
        j = i
        while j < n and bool(geo.prim_closed[j]) == cl:
            j += 1
        body = [[list(range(int(geo.prim_start[k]), int(geo.prim_start[k + 1])))] for k in range(i, j)]
        prims.append([["type", "run", "runtype", "Poly", "varyingfields", ["vertex"],
                       "uniformfields", {"closed": cl}], body])
        i = j
    attributes = []
    for key, cls in _ATTR_KEYS:
        entries = []
        for name, arr in geo.attribs[cls].items():
            e = _attrib_entry(name, np.asarray(arr), cls)
            if e is not None:
                entries.append(e)
        if entries:
            attributes += [key, entries]
    data = [
        "fileversion", FILE_VERSION,
        "hasindex", False,
        "pointcount", int(geo.num_points),
        "vertexcount", int(nv),
        "primitivecount", int(len(sizes)),
        "info", {"software": "Blendini"},
        "topology", ["pointref", ["indices", geo.vtx_point.astype(np.int64).tolist()]],
        "attributes", attributes,
        "primitives", prims,
    ]
    for key, cls in _GROUP_KEYS:
        if geo.groups[cls]:
            data += [key, [_group_entry(name, m) for name, m in geo.groups[cls].items()]]
    return data


def write_geo(geo, path):
    """Write a Geometry as a Houdini ASCII .geo file (polygons, polylines, attributes, groups)."""
    data = build_geo(geo)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, separators=(',', ':'))
