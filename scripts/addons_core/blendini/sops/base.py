# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Declarative SOP (Surface OPerator) definitions.

A SOP declares its parameters and implements ``cook(inputs, parms, ctx)``. The Blender
node classes are generated from these declarations, so the operators themselves stay
independent from Blender and can be unit tested with plain Python.
"""

import numpy as np

REGISTRY = {}

# Categories in the TAB menu (Houdini style).
CATEGORIES = ['Primitive', 'Manipulate', 'Polygon', 'Attribute', 'Group', 'Copy', 'Utility', 'VEX', 'Blender']

# Node header colors per category.
CATEGORY_COLORS = {
    'Primitive': (0.30, 0.38, 0.47),
    'Manipulate': (0.36, 0.33, 0.45),
    'Polygon': (0.33, 0.42, 0.33),
    'Attribute': (0.45, 0.40, 0.28),
    'Group': (0.30, 0.42, 0.42),
    'Copy': (0.45, 0.33, 0.30),
    'Utility': (0.32, 0.32, 0.32),
    'VEX': (0.50, 0.30, 0.38),
    'Blender': (0.50, 0.38, 0.22),
}


class Parm:
    """Parameter declaration.

    type: float, int, bool, vector, vector2, color, string, enum, code, object, ramp
    """
    __slots__ = ('name', 'label', 'type', 'default', 'min', 'max', 'soft_min', 'soft_max', 'items', 'hide_when',
                 'description')

    def __init__(self, name, label, type='float', default=None, min=None, max=None, soft_min=None, soft_max=None,
                 items=None, hide_when=None, description=""):
        self.name = name
        self.label = label
        self.type = type
        if default is None:
            default = {
                'float': 0.0, 'int': 0, 'bool': False, 'vector': (0.0, 0.0, 0.0), 'vector2': (0.0, 0.0),
                'color': (1.0, 1.0, 1.0), 'string': '', 'code': '', 'object': '', 'nodegroup': '',
            }.get(type, items[0][0] if items else None)
        self.default = default
        self.min = min
        self.max = max
        self.soft_min = soft_min
        self.soft_max = soft_max
        self.items = items
        # hide_when: dict {parm_name: value or tuple of values} -> hide when any matches
        self.hide_when = hide_when
        self.description = description


class SOP:
    id = ''
    label = ''
    category = 'Utility'
    inputs = 0
    input_labels = ()
    multi_input = False   # first input accepts many links (Merge)
    parms = ()
    icon = 'NONE'
    time_dependent = False
    description = ""

    def cook(self, inputs, p, ctx):
        raise NotImplementedError


def register_sop(cls):
    REGISTRY[cls.id] = cls()
    return cls


class CookContext:
    """Information available while cooking a node."""

    def __init__(self, time=0.0, frame=1.0, fps=24.0, node_path='', state=None):
        self.time = time
        self.frame = frame
        self.fps = fps
        self.node_path = node_path
        self.messages = []
        self.warnings = []
        # Persistent per-node storage (for solvers).
        self.state = state if state is not None else {}

    def warn(self, msg):
        self.warnings.append(msg)

    def log(self, msg):
        self.messages.append(msg)


class CookError(Exception):
    pass


# ----------------------------------------------------------------------------
# Shared helpers


def euler_matrix(r_deg, order='xyz'):
    rx, ry, rz = np.radians(r_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    X = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    mats = {'x': X, 'y': Y, 'z': Z}
    m = np.eye(3)
    # Rotation applied in order: first axis first (column vectors => left multiply).
    for axis in order:
        m = mats[axis] @ m
    return m


def transform_matrix(t=(0, 0, 0), r=(0, 0, 0), s=(1, 1, 1), uniform=1.0, pivot=(0, 0, 0), rord='xyz', xord='srt'):
    """4x4 column-vector matrix built like Houdini's Transform SOP."""
    T = np.eye(4)
    T[:3, 3] = t
    R = np.eye(4)
    R[:3, :3] = euler_matrix(r, rord)
    S = np.eye(4)
    S[:3, :3] = np.diag(np.asarray(s, dtype=np.float64) * uniform)
    P = np.eye(4)
    P[:3, 3] = pivot
    Pi = np.eye(4)
    Pi[:3, 3] = -np.asarray(pivot, dtype=np.float64)
    parts = {'s': S, 'r': R, 't': T}
    m = np.eye(4)
    for c in xord:
        if c == 't':
            m = T @ m
        else:
            m = P @ parts[c] @ Pi @ m
    return m


def parse_group_pattern(geo, cls, pattern):
    """Return a bool mask for a Houdini style group pattern.

    Supports: '' / '*' (everything), group names, numbers and ranges ('0-10 15'), '!' negation
    prefix, and '@attr' VEX expressions like '@P.y>0' or '@ptnum%2==0'.
    """
    n = geo.count(cls)
    pattern = (pattern or '').strip()
    if pattern in ('', '*'):
        return np.ones(n, dtype=bool)
    if '@' in pattern:
        return vex_expression_mask(geo, cls, pattern)
    mask = np.zeros(n, dtype=bool)
    for tok in pattern.replace(',', ' ').split():
        neg = tok.startswith('!') or tok.startswith('^')
        if neg:
            tok = tok[1:]
        sel = np.zeros(n, dtype=bool)
        if tok == '*':
            sel[:] = True
        elif tok in geo.groups.get(cls, {}):
            sel = geo.groups[cls][tok].copy()
        else:
            try:
                if '-' in tok[1:]:
                    a, b = tok.split('-', 1) if not tok.startswith('-') else ('-' + tok[1:].split('-', 1)[0], tok[1:].split('-', 1)[1])
                    a, b = int(a), int(b)
                    lo, hi = min(a, b), max(a, b)
                    sel[max(lo, 0):min(hi + 1, n)] = True
                else:
                    i = int(tok)
                    if 0 <= i < n:
                        sel[i] = True
            except ValueError:
                import fnmatch
                for gname, g in geo.groups.get(cls, {}).items():
                    if fnmatch.fnmatchcase(gname, tok):
                        sel |= g
        if neg:
            mask &= ~sel
        else:
            mask |= sel
    return mask


def vex_expression_mask(geo, cls, expr):
    from ..vex.wrangle import run_wrangle
    code = "i@__sel = (%s) ? 1 : 0;" % expr.strip().rstrip(';')
    out, _ = run_wrangle(code, [geo], run_over=cls)
    sel = out.attribs[cls].get('__sel')
    if sel is None:
        return np.zeros(geo.count(cls), dtype=bool)
    return sel.astype(bool)


def orient_matrices(geo):
    """Per-point 3x3 rotation matrices (column vector) from orient / N+up."""
    n = geo.num_points
    pa = geo.attribs['point']
    if 'orient' in pa:
        q = pa['orient']
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        return np.stack([
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=-1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], axis=-1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1),
        ], axis=1)
    if 'N' in pa:
        zaxis = pa['N']
        up = pa['up'] if 'up' in pa else np.broadcast_to([0.0, 1.0, 0.0], (n, 3))
        zl = np.linalg.norm(zaxis, axis=1, keepdims=True)
        z = zaxis / np.where(zl == 0, 1, zl)
        x = np.cross(up, z)
        xl = np.linalg.norm(x, axis=1, keepdims=True)
        # Up parallel to N: pick another up vector.
        alt = np.cross(np.broadcast_to([1.0, 0.0, 0.0], (n, 3)), z)
        x = np.where(xl < 1e-8, alt, x)
        xl = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.where(xl == 0, 1, xl)
        y = np.cross(z, x)
        m = np.stack([x, y, z], axis=2)
        m[zl[:, 0] == 0] = np.eye(3)
        return m
    return np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
