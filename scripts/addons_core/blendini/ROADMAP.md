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
- [x] 280 builtins: math, vectors, matrices, quaternions, noise, geometry read/write, groups, half-edges,
      nearpoints, pcopen/pcfilter/pciterate/pcimport, xyzdist/primuv/minpos/intersect, strings and
      regular expressions, arrays, unique values, spline, sampling, colormap, ch*/chramp
- [x] Wrangles: point / prim / vertex / detail / numbers, spare parameters, ramps, presets menu
- [x] Attribute VOP (visual VEX network that generates code)
- [x] Driver expressions on parameters evaluated through VEX (`#frame * 0.1`), keyframes

### SOPs (140)
- [x] Primitives, manipulation, polygon, attribute, group, copy/instancing, terrain, utility,
      Blender-backed (modifiers, bmesh, BVH, Bullet, OpenVDB) — see README for the full list
- [x] Simulations: Solver (VEX), Vellum-style cloth (PBD), RBD (Bullet), PBF fluid + meshing, Pyro smoke
- [x] For-Each (pieces, feedback, count), Subnetworks, Time Shift, Edit (hand edits), Bake,
      File Cache (lossless .npz per frame, Save to Disk / Load from Disk)
- [x] Everyday additions: Enumerate, Group by Range, Switch-If, Revolve, Skin, Path Deform,
      Distance from Geometry, Lattice, UV Transform, UV Layout, Attribute Adjust Float / Vector,
      Extract Centroid, Group from Attribute Boundary, Calculate Occlusion

### Look and feel
- [x] TAB menu with auto-wiring, display/template/bypass flags (buttons, hotkeys, edge bars)
- [x] Parameter pane, geometry info, spreadsheet, VEX status bar in the text editor
- [x] Viewport overlays (point/prim numbers, normals, template wireframe), attribute visualizer
- [x] Houdini-like theme, layout, Create shelf, application template, default `Cd` material

### Stability
- [x] Integration test inside Blender (`tests/blender_integration.py`)
- [x] Fixed: crash when an RBD simulation stepped from the frame change handler, and when
      removing one rigid body scene while another existed; RBD now pre-bakes the scene range
      (also makes command line renders correct)
- [x] Fixed: intermittent crash when a glued RBD simulation was removed (constraints are now
      freed before their Bullet bodies); stress-tested with random build / remove orders

## Next (in priority order)

- [x] For-Each "by count" (run N iterations and merge, without feedback)
- [x] More VOP nodes: compare + two-way switch, ramp (chramp), trig/exp/min/max, random vector
- [x] VDB nodes through Geometry Nodes: VDB Reshape / Smooth SDF, VDB Combine (Blender 5 OpenVDB grid nodes)
- [x] Delete SOP (by pattern / expression, bounding box, normal, random) as in Houdini
- [x] Attribute Interpolate (sourceprim + sourceprimuv from Scatter, exact on triangles and quads)
- [x] Self-test operator that runs the smoke test inside the user's Blender

## Later / ideas

- [x] Subnet promoted parameters (HDA-style interface: `ch()` in wrangles and drivers inside)
- [x] Channel references between nodes: `ch("../node/parm")` in VEX and driver expressions

- [x] Multi-output nodes (cook engine reads the linked output) with the Split SOP
- [x] VEX function overloading by signature, function-like `#define` macros
- [x] VEX `struct`s (defaults, constructors, members, function parameters / returns)
- [x] Arrays of structs, struct methods (`obj->method()`, members visible inside methods)
- [x] Curve SOP (polygon, Catmull-Rom, Bézier — evaluated to polylines)
- [x] Volumes as a native primitive type: dense grids on the geometry (fog, SDF, vector), Volume,
      Volume from Polygons, Volume Wrangle, Volume Blur, Rasterize Points, Convert Volume (surface
      nets), VEX volumesample / volumegradient / ..., smoke display and rendering via OpenVDB files
- [ ] Native NURBS / Bézier primitives, sparse (VDB-style) volumes
- [x] RBD glue constraints (breakable FIXED constraints between neighbouring pieces)
- [x] Constraint networks from geometry (Connect Adjacent Pieces → RBD third input: glue, fixed,
      point, hinge, slider, spring; per-primitive strength / stiffness / axis; world pins)
- [x] Pyro-style smoke solver (stable fluids on native volumes: buoyancy, vorticity, turbulence)
- [x] Fire: combustion in the Pyro Solver (fuel, burn, heat, flame field + fire material)
- [ ] FLIP-style fluid (surface tension, narrow band)
- [x] Viewport handle for the Transform SOP (translate arrows + rotate dials)
- [x] More handles: Transform scale and pivot, primitives (center / size / radius), origins,
      lengths, bounding box groups (table driven, `bl/handles.py`)
- [ ] Tool states (Edit brush / soft selection)
- [ ] Multi-line code editor in the parameter pane (Blender UI has no widget for it today)

## Deliberately not planned

- Python SOP: networks must stay safe to open (no Python from .blend files is executed).
