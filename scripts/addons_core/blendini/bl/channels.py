# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Houdini style channel references: ch("../box1/sizex") reads another node's parameter.

Paths are relative to the network: "../node/parm" and "node/parm" both name a sibling node.
A trailing x / y / z / w (or r / g / b) picks a component of a vector parameter, so
"../xform1/tx" is the X translation, like Houdini's parameter names.
"""

import re

_COMPONENTS = {'x': 0, 'y': 1, 'z': 2, 'w': 3, 'r': 0, 'g': 1, 'b': 2}
_ref_re = re.compile(r'\bch[a-z0-9]*\s*\(\s*["\']([^"\']*/[^"\']*)["\']')


def referenced_paths(code):
    """Channel paths (containing '/') used by ch*() calls in code or an expression."""
    return list(dict.fromkeys(_ref_re.findall(code or '')))


def resolve(tree, path):
    """Value of a parameter path, or None when it doesn't resolve."""
    parts = [p for p in path.split('/') if p and p != '.']
    while parts and parts[0] == '..':
        parts.pop(0)
    if len(parts) != 2 or tree is None:
        return None
    node_name, parm = parts
    node = tree.nodes.get(node_name)
    if node is None:
        return None
    for sp in getattr(node, 'spare', ()):
        if sp.name == parm:
            return sp.value()
    value = getattr(node, 'p_' + parm, None)
    if value is not None:
        return _plain(value)
    if len(parm) > 1 and parm[-1] in _COMPONENTS:
        vec = getattr(node, 'p_' + parm[:-1], None)
        try:
            return float(vec[_COMPONENTS[parm[-1]]])
        except (TypeError, IndexError, ValueError):
            return None
    return None


def _plain(value):
    if isinstance(value, (bool, int, float, str)):
        return float(value) if isinstance(value, bool) else value
    try:
        return tuple(float(v) for v in value)
    except TypeError:
        return None


def resolved_channels(tree, code):
    """{path: value} for every resolvable channel reference in code."""
    out = {}
    for path in referenced_paths(code):
        v = resolve(tree, path)
        if v is not None:
            out[path] = v
    return out
