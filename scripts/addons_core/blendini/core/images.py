# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Image sampling (Blender images are the backend for colormap() and Attribute from Map)."""

import numpy as np

_cache = {}


def image_pixels(name):
    """Pixels (h, w, 4) float32 of a Blender image (by datablock name or file path)."""
    import bpy
    img = bpy.data.images.get(name)
    if img is None and name:
        path = bpy.path.abspath(name)
        for im in bpy.data.images:
            if bpy.path.abspath(im.filepath) == path:
                img = im
                break
        if img is None:
            try:
                img = bpy.data.images.load(path, check_existing=True)
            except RuntimeError:
                img = None
    if img is None:
        return None
    key = (img.name, img.size[0], img.size[1], img.is_dirty)
    cached = _cache.get(img.name)
    if cached is not None and cached[0] == key:
        return cached[1]
    w, h = img.size
    if w == 0 or h == 0:
        return None
    px = np.empty(w * h * img.channels, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, img.channels)
    if img.channels < 4:
        pad = np.ones((h, w, 4), dtype=np.float32)
        pad[..., :img.channels] = px
        px = pad
    _cache[img.name] = (key, px)
    return px


def sample(px, u, v):
    """Bilinear sample: clamped to the edges inside [0, 1], repeating outside. Returns (n, 4)."""
    h, w = px.shape[:2]
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u = np.where((u >= 0) & (u <= 1), u, u % 1.0)
    v = np.where((v >= 0) & (v <= 1), v, v % 1.0)
    x = np.clip(u * w - 0.5, 0, w - 1)
    y = np.clip(v * h - 0.5, 0, h - 1)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    top = px[y0, x0] * (1 - fx) + px[y0, x1] * fx
    bot = px[y1, x0] * (1 - fx) + px[y1, x1] * fx
    return top * (1 - fy) + bot * fy
