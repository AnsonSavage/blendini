# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

from .base import REGISTRY, CATEGORIES, CATEGORY_COLORS, CookContext, CookError, Parm, SOP
from . import generators, modify, attribs, vellum, heightfield  # noqa: F401  (registers SOPs)

try:
    import bpy  # noqa: F401
    from . import blender_ops  # noqa: F401
except ImportError:
    pass


class ParmValues(dict):
    """Parameter values accessible as attributes (p.size) or items (p['class'])."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def default_parms(sop_id):
    sop = REGISTRY[sop_id]
    return ParmValues({p.name: p.default for p in sop.parms})


def cook_sop(sop_id, inputs=(), ctx=None, **overrides):
    """Cook a SOP with default parameters (handy for scripting and tests)."""
    sop = REGISTRY[sop_id]
    p = default_parms(sop_id)
    for k, v in overrides.items():
        p[k.rstrip('_')] = v
    return sop.cook(list(inputs), p, ctx or CookContext())
