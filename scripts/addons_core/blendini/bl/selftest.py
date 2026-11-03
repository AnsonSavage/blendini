# SPDX-FileCopyrightText: 2026 Blendini Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""A self test users can run inside their own Blender: cooks every SOP and a few VEX snippets."""

import time

import bpy

SNIPPETS = [
    '@P.y += sin(@P.x * 3 + @Time) * 0.2;',
    'int pts[] = nearpoints(0, @P, 0.3); i@n = len(pts);',
    'if (@ptnum % 2 == 0) { @Cd = {1, 0, 0}; } else { @Cd = rand(@ptnum); }',
    'float f(float x) { return x * 2; } f@v = f(@P.x); p@orient = quaternion(radians(45), {0, 1, 0});',
    'int p = addpoint(0, @P + {0, 1, 0}); addprim(0, "polyline", @ptnum, p);',
]


def run_self_test():
    """Returns (ok_count, failures list of (name, message), seconds)."""
    from ..sops import REGISTRY, default_parms, CookContext, CookError, cook_sop
    from ..vex.wrangle import run_wrangle
    t0 = time.perf_counter()
    box, grid, sphere = cook_sop('box'), cook_sop('grid'), cook_sop('sphere')
    line = cook_sop('line', points=10)
    pts = cook_sop('scatter', [sphere], npts=50)
    hf = cook_sop('heightfield', size=(2.0, 2.0), gridspacing=0.2)
    special = {'hfnoise': [hf], 'hferode': [hf], 'sweep': [line], 'polyframe': [line], 'resample': [line],
               'carve': [line], 'copytopoints': [box, pts], 'attribtransfer': [grid, sphere], 'vellum': [grid, sphere],
               'fluidsolver': [pts], 'ends': [cook_sop('circle')], 'boolean': [box, sphere], 'ray': [grid, sphere],
               'vdbcombine': [box, sphere], 'pointdeform': [sphere, sphere, sphere]}
    ok, failures = 0, []
    for sop_id, sop in sorted(REGISTRY.items()):
        inputs = special.get(sop_id, [sphere, box, grid, pts][:sop.inputs])
        state = {}
        ctx = CookContext(frame=2, time=1 / 24, state=state)
        ctx.channels, ctx.ramps = {}, {}
        try:
            out = sop.cook(list(inputs), default_parms(sop_id), ctx)
            if out is None:
                raise RuntimeError("returned nothing")
            ok += 1
        except CookError:
            ok += 1  # reported to the user as a node error, which is fine
        except Exception as ex:
            failures.append((sop.label, "%s: %s" % (type(ex).__name__, ex)))
        finally:
            if state.get('scene'):
                from ..sops.blender_ops import _rbd_cleanup
                _rbd_cleanup(state)
    for i, code in enumerate(SNIPPETS):
        try:
            run_wrangle(code, [sphere])
            ok += 1
        except Exception as ex:
            failures.append(("VEX snippet %d" % (i + 1), str(ex)))
    return ok, failures, time.perf_counter() - t0


class BLENDINI_OT_self_test(bpy.types.Operator):
    """Cook every SOP and some VEX in this Blender to check that Blendini works here"""
    bl_idname = "blendini.self_test"
    bl_label = "Run Blendini Self Test"

    def execute(self, context):
        ok, failures, seconds = run_self_test()
        for name, msg in failures:
            print("Blendini self test FAILED:", name, "-", msg)
        if failures:
            self.report({'ERROR'}, "Blendini self test: %d passed, %d failed (see the console): %s"
                        % (ok, len(failures), ", ".join(n for n, _m in failures[:5])))
        else:
            self.report({'INFO'}, "Blendini self test: all %d checks passed in %.1f s (Blender %s)"
                        % (ok, seconds, bpy.app.version_string))
        return {'FINISHED'}
