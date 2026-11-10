# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Lossless geometry files (.npz): attributes, topology, groups, packed instances and volumes.

Blendini's equivalent of .bgeo for the File Cache SOP. Only numpy arrays and a JSON header are
stored (np.load with allow_pickle=False), so cache files can't execute code.
"""

import json

import numpy as np

from .geometry import Geometry, POINT, PRIM, VERTEX, DETAIL

VERSION = 1


def _flatten(geo, prefix, arrays, meta):
    meta['attribs'] = {}
    for cls in (POINT, PRIM, VERTEX, DETAIL):
        meta['attribs'][cls] = []
        for name, arr in geo.attribs[cls].items():
            key = "%sa/%s/%s" % (prefix, cls, name)
            if arr.dtype == object:
                arrays[key] = np.array([str(x) for x in arr.tolist()], dtype=str) if len(arr) else np.zeros(0, dtype=str)
                meta['attribs'][cls].append([name, 'string'])
            else:
                arrays[key] = arr
                meta['attribs'][cls].append([name, 'num'])
    arrays[prefix + 'prim_start'] = geo.prim_start
    arrays[prefix + 'vtx_point'] = geo.vtx_point
    arrays[prefix + 'prim_closed'] = geo.prim_closed
    meta['groups'] = {}
    for cls in (POINT, PRIM):
        meta['groups'][cls] = list(geo.groups[cls].keys())
        for name, g in geo.groups[cls].items():
            arrays["%sg/%s/%s" % (prefix, cls, name)] = g
    meta['volumes'] = []
    for i, v in enumerate(getattr(geo, 'volumes', ())):
        arrays["%sv/%d" % (prefix, i)] = v.data
        meta['volumes'].append({'name': v.name, 'origin': v.origin.tolist(), 'voxelsize': v.voxelsize,
                                'kind': v.kind})
    if geo.typeinfo:
        meta['typeinfo'] = [[c, n, t] for (c, n), t in geo.typeinfo.items()]
    meta['instances'] = []
    for i, inst in enumerate(geo.instances):
        sub = {}
        _flatten(inst, "%si%d/" % (prefix, i), arrays, sub)
        meta['instances'].append(sub)


def save_geometry(geo, path):
    arrays = {}
    meta = {'version': VERSION}
    _flatten(geo, '', arrays, meta)
    arrays['__meta__'] = np.array(json.dumps(meta))
    with open(path, 'wb') as fh:
        np.savez_compressed(fh, **arrays)


def _unflatten(data, prefix, meta):
    from .volume import Volume
    geo = Geometry()
    for cls in (POINT, PRIM, VERTEX, DETAIL):
        attribs = {}
        for name, kind in meta['attribs'].get(cls, []):
            arr = data["%sa/%s/%s" % (prefix, cls, name)]
            if kind == 'string':
                out = np.empty(len(arr), dtype=object)
                out[:] = [str(x) for x in arr.tolist()]
                arr = out
            attribs[name] = arr
        if cls == POINT and 'P' not in attribs:
            attribs['P'] = np.zeros((0, 3))
        geo.attribs[cls] = attribs
    geo.prim_start = data[prefix + 'prim_start'].astype(np.int64)
    geo.vtx_point = data[prefix + 'vtx_point'].astype(np.int64)
    geo.prim_closed = data[prefix + 'prim_closed'].astype(bool)
    for cls in (POINT, PRIM):
        for name in meta['groups'].get(cls, []):
            geo.groups[cls][name] = data["%sg/%s/%s" % (prefix, cls, name)].astype(bool)
    geo.volumes = [Volume(v['name'], data["%sv/%d" % (prefix, i)], v['origin'], v['voxelsize'], v['kind'])
                   for i, v in enumerate(meta.get('volumes', []))]
    geo.typeinfo = {(c, n): t for c, n, t in meta.get('typeinfo', [])}
    geo.instances = [_unflatten(data, "%si%d/" % (prefix, i), sub) for i, sub in enumerate(meta.get('instances', []))]
    return geo


def load_geometry(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data['__meta__']))
        if meta.get('version', 0) > VERSION:
            raise ValueError("%s was written by a newer Blendini" % path)
        return _unflatten(data, '', meta)


def expand_path(pattern, frame, node_name='', blend_dir=''):
    """Houdini style variables: $F, $F2..$F6 (padded frame), $FF (float frame), $OS (node name), $HIP."""
    import re
    f = int(round(frame))
    out = pattern.replace('$HIP', blend_dir or '.').replace('$OS', node_name)
    out = re.sub(r'\$F([2-9])', lambda m: str(f).zfill(int(m.group(1))), out)
    out = out.replace('$FF', ('%g' % frame)).replace('$F', str(f))
    return out
