# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Ramp spare parameters (chramp) backed by hidden Float Curve nodes."""

import uuid

import bpy

GROUP_NAME = ".blendini_ramps"


def _group(create=True):
    ng = bpy.data.node_groups.get(GROUP_NAME)
    if ng is None and create:
        ng = bpy.data.node_groups.new(GROUP_NAME, 'ShaderNodeTree')
        ng.use_fake_user = True
    return ng


def new_key():
    return "ramp_" + uuid.uuid4().hex[:12]


def ramp_node(key, create=False):
    ng = _group(create)
    if ng is None or not key:
        return None
    node = ng.nodes.get(key)
    if node is None and create:
        node = ng.nodes.new('ShaderNodeFloatCurve')
        node.name = key
        node.label = key
    return node


def sample(key, n=64):
    """Sample the ramp into (x, y) keys for np.interp."""
    node = ramp_node(key)
    if node is None:
        return ()
    mapping = node.mapping
    try:
        mapping.initialize()
    except Exception:
        pass
    curve = mapping.curves[0]
    return tuple((i / (n - 1), mapping.evaluate(curve, i / (n - 1))) for i in range(n))
