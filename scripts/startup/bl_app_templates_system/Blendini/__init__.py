# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Blendini application template: a Houdini-like startup.

File > New > Blendini starts with an empty scene, a `geo1` SOP network
and the viewport / network editor / spreadsheet layout.
"""

import bpy
from bpy.app.handlers import persistent


def _enable_addon():
    import addon_utils
    loaded_default, loaded_state = addon_utils.check("blendini")
    if not loaded_state:
        addon_utils.enable("blendini", default_set=True, persistent=True)


def _setup_ui():
    wm = bpy.context.window_manager
    if wm is None or not wm.windows:
        return 0.1  # Try again once the window exists.
    win = wm.windows[0]
    screen = win.screen
    view3d = max((a for a in screen.areas if a.type == 'VIEW_3D'), key=lambda a: a.width * a.height, default=None)
    if view3d is None:
        return None
    region = next((r for r in view3d.regions if r.type == 'WINDOW'), None)
    shading = view3d.spaces.active.shading
    # A neutral grey viewport (per viewport, the user's theme is left alone).
    shading.background_type = 'VIEWPORT'
    shading.background_color = (0.27, 0.29, 0.32)
    try:
        with bpy.context.temp_override(window=win, screen=screen, area=view3d, region=region):
            if not any(t.bl_idname == 'BlendiniSOPTree' for t in bpy.data.node_groups):
                bpy.ops.blendini.new_network()
            bpy.ops.blendini.houdini_layout()
    except Exception as ex:
        print("Blendini template:", ex)
    return None


@persistent
def load_handler(_):
    _enable_addon()
    # Houdini starts with an empty scene.
    cube = bpy.data.objects.get("Cube")
    if cube is not None and cube.type == 'MESH':
        mesh = cube.data
        bpy.data.objects.remove(cube)
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    if not bpy.app.background:
        bpy.app.timers.register(_setup_ui, first_interval=0.05)


def register():
    bpy.app.handlers.load_factory_startup_post.append(load_handler)


def unregister():
    bpy.app.handlers.load_factory_startup_post.remove(load_handler)
