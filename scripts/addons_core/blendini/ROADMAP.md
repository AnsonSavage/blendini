# Blendini roadmap

Status of the Houdini-in-Blender clone. "Verified" means covered by the unit tests
(`tests/test_blendini.py`), the every-SOP smoke test, or a headless Blender run.
The interactive UI (panels, hotkeys, draw overlays) could only be exercised with mocks
and needs a real Blender session to confirm.

## Done

### Core
- [x] Houdini geometry model: points / primitives (polygons + polylines) / vertices / detail
      attributes, point and primitive groups, Houdini attribute defaults (`Cd` white, `pscale` 1)
- [x] Packed geometry (instances) with transforms, merge, unpack
- [x] Signature based cook cache (Houdini style dirty propagation), time dependence, errors per node
- [x] Y-up networks, Z-up Blender objects; mesh <-> geometry conversion with attributes, groups,
      uv, normals, polylines (chained loose edges), strings (Blender 5 bytes)

### VEX
- [x] Lexer / parser / SIMD masked interpreter (if/else, loops, break/continue, return, user
      functions with pass by reference, recursion, `#define` macros, casts, swizzles, arrays)
- [x] 189 builtins: math, vectors, matrices, quaternions, noise, geometry read/write, groups,
      nearpoints, pcopen/pcfilter, xyzdist/primuv/minpos, strings, arrays, colormap, ch*/chramp
- [x] Wrangles: point / prim / vertex / detail / numbers, spare parameters, ramps, presets menu
- [x] Attribute VOP (visual VEX network that generates code)
- [x] Driver expressions on parameters evaluated through VEX (`#frame * 0.1`), keyframes

### SOPs (104)
- [x] Primitives, manipulation, polygon, attribute, group, copy/instancing, terrain, utility,
      Blender-backed (modifiers, bmesh, BVH, Bullet, OpenVDB) — see README for the full list
- [x] Simulations: Solver (VEX), Vellum-style cloth (PBD), RBD (Bullet), PBF fluid + meshing
- [x] For-Each (pieces, feedback, count), Subnetworks, Time Shift, Edit (hand edits), Bake

### Look and feel
- [x] TAB menu with auto-wiring, display/template/bypass flags (buttons, hotkeys, edge bars)
- [x] Parameter pane, geometry info, spreadsheet, VEX status bar in the text editor
- [x] Viewport overlays (point/prim numbers, normals, template wireframe), attribute visualizer
- [x] Houdini-like theme, layout, Create shelf, application template, default `Cd` material

## Next (in priority order)

- [x] For-Each "by count" (run N iterations and merge, without feedback)
- [x] More VOP nodes: compare + two-way switch, ramp (chramp), trig/exp/min/max, random vector
- [x] VDB nodes through Geometry Nodes: VDB Reshape / Smooth SDF, VDB Combine (Blender 5 OpenVDB grid nodes)
- [x] Delete SOP (by pattern / expression, bounding box, normal, random) as in Houdini
- [ ] Attribute Interpolate (sourceprim + uv from Scatter)
- [x] Self-test operator that runs the smoke test inside the user's Blender

## Later / ideas

- [ ] Multi-output nodes (Split, Two Way Switch outputs)
- [ ] VEX `struct`s, function overloading by signature, function-like macros
- [ ] NURBS / Bézier primitives, volumes as a native primitive type
- [ ] Constraint networks for RBD, FLIP-style surface tension / viscosity
- [ ] Viewport handles and tool states (Transform handle, Edit brush)
- [ ] Multi-line code editor in the parameter pane (Blender UI has no widget for it today)

## Deliberately not planned

- Python SOP: networks must stay safe to open (no Python from .blend files is executed).
