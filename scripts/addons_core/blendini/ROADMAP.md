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
      functions with pass by reference, recursion, `#define` macros, casts, swizzles, arrays,
      structs with methods, dicts, hex literals, return type overloading through expressions)
- [x] ~300 builtins: math, vectors, matrices, quaternions, noise, geometry read/write, groups, half-edges,
      nearpoints, pcopen/pcfilter/pciterate/pcimport, xyzdist/primuv/minpos/intersect, strings and
      regular expressions, arrays, unique values, spline, sampling, colormap, ch*/chramp
- [x] Wrangles: point / prim / vertex / detail / numbers, spare parameters, ramps, presets menu
- [x] Attribute VOP and Volume VOP (visual VEX networks that generate code)
- [x] Driver expressions on parameters evaluated through VEX (`#frame * 0.1`, `$F`, `$T`, `$FPS`), keyframes
- [x] Unconnected inputs read as empty geometry; wrong-typed calls report the function and argument types

### SOPs (161)
- [x] Primitives, manipulation, polygon, attribute, group, copy/instancing, terrain, utility,
      Blender-backed (modifiers, bmesh, BVH, Bullet, OpenVDB) — see README for the full list
- [x] Simulations: Solver (VEX), Vellum-style cloth (PBD), RBD (Bullet), FLIP and PBF liquids + meshing, Pyro smoke and fire,
      Vellum hair and pressure (balloons); animated colliders (Vellum, FLIP, Pyro) and animated
      Vellum pins don't restart sims, upstream edits do
- [x] For-Each (pieces, feedback, count), Subnetworks, Time Shift, Edit (hand edits), Bake,
      File Cache (lossless .npz per frame, Save to Disk / Load from Disk), Material (material slots
      from shop_materialpath)
- [x] Terrain: HeightField Mask by Feature, HeightField Scatter, HeightField Terrace, HeightField Blur
- [x] Everyday additions: Enumerate, Group by Range, Switch-If, Revolve, Skin, Path Deform,
      Distance from Geometry, Lattice, UV Transform, UV Layout, Attribute Adjust Float / Vector,
      Extract Centroid, Group from Attribute Boundary, Calculate Occlusion, Blend Shapes, Exploded View,
      PolyCap, Point Replicate, Group Transfer, Soft Transform, Magnet, Instance (s@instance),
      Extract Transform, Transform Pieces

### Look and feel
- [x] TAB menu with auto-wiring, display/template/bypass flags (buttons, hotkeys, edge bars)
- [x] Parameter pane, geometry info, spreadsheet, VEX status bar in the text editor, cook time badges
      (performance monitor), network boxes and dots
- [x] Viewport overlays (point/prim numbers, normals, vectors such as v, template wireframe), attribute visualizer
- [x] Houdini-like theme, layout, Create shelf, application template, default `Cd` material
- [x] Polylines render as curves (width attribute), volumes as smoke / fire, material slots from
      shop_materialpath

### Stability
- [x] Integration test inside Blender (`tests/blender_integration.py`)
- [x] Fixed: crash when an RBD simulation stepped from the frame change handler, and when
      removing one rigid body scene while another existed; RBD now pre-bakes the scene range
      (also makes command line renders correct)
- [x] Fixed: intermittent crash when a glued RBD simulation was removed (constraints are now
      freed before their Bullet bodies); stress-tested with random build / remove orders
- [x] Random network fuzzer (`tests/fuzz_networks.py`) and a builtin fuzzer: fixed empty-input,
      negative seed, divide-by-zero and runaway voxel / grid cases (VDB, Remesh, FLIP, Volume guards),
      vectorized slow SOPs (Revolve, Skin, Relax)

## Also done (previously listed under Next / Later)

For-Each by count, more VOPs, VDB nodes via Geometry Nodes, Delete SOP, Attribute Interpolate,
self-test operator, subnet promoted parameters, digital assets, channel references, multi-output
nodes, VEX overloading / macros / structs / struct arrays and methods, Curve SOP, native dense
volumes, RBD glue and constraint networks, RBD piece attributes and output modes, Pyro smoke and
fire, FLIP liquid, viewport handles, Edit and Attribute Paint tool states.

---

# Future development

## How to work on this roadmap

Every item below is written so that one person (or one coding agent) can pick it up without
other context. Each has an id, a size (S: under an hour, M: a few hours, L: a day or more), the
files it touches, and an acceptance check.

**Rules that keep the product working**

1. Every commit must pass, from `scripts/addons_core`:
   - `python3 -m unittest discover -s blendini/tests -p "test_*.py" -t .` (pure Python, ~30 s)
   - `blender --background --factory-startup --python blendini/tests/blender_integration.py`
     (or the `bpy` wheel: `python blendini/tests/blender_integration.py`)
   - after touching SOPs: `python3 blendini/tests/fuzz_networks.py 1 300` reports 0 failures.
2. Gate commits on the test result (`tests && git commit`), never commit after a failed run.
3. New SOPs go in their own module under `sops/` and are registered in `sops/__init__.py`.
   Use icon names that already appear in `sops/*.py`: an invalid icon breaks registration of the
   whole add-on. A SOP may only raise `CookError`, and must survive empty and arbitrary inputs
   (the every-SOP test cooks each one with every enum / bool variant).
4. Keep `core/`, `vex/` and `sops/` (except `blender_ops.py`) free of `bpy` so they stay unit
   testable. Blender-only code lives in `bl/` and `sops/blender_ops.py`.
5. Vectorize with numpy. A per-point Python loop is a bug at 100k points. Guard anything whose
   cost is set by a parameter (voxel size, grid size, counts) with a `CookError` above a limit.
6. Simulations: restart on parameter change or an upstream edit at an already-simulated frame
   (`sops.base.upstream_changed`), never on animation. Never build or step Blender scenes inside
   the frame change handler (`bl.cook.in_frame_handler`).
7. No Python execution from .blend files, ever (see "Deliberately not planned").
8. Update this file and README.md in the same commit as the feature.

**Suggested order.** P0 first (it is what makes the project trustworthy), then P1 in the listed
order. P2 and P3 items are independent of each other and can be done in parallel by separate
agents as long as they own separate files.

## P0: Verify in a real Blender session (needs a human or a GUI runner)

Everything interactive has only been tested headless and with mocks. This is the largest risk.

- [ ] **UI-1 (M) Manual UI pass.** Follow README "Quick start" in the Blender 5.2 GUI and file
      every break as an item here. Checklist: TAB menu and auto-wiring, D/T/B flag hotkeys, I/U
      dive in and out, parameter panel, spreadsheet, point / prim number overlays, normals and
      vector overlays, template wireframe, cook-time badges, text editor status bar and
      Ctrl+Enter, Create shelf, theme and layout operators, application template, Edit SOP round
      trip, Attribute Paint round trip, File Cache Save to Disk, Create Digital Asset dialog.
      *Accept:* checklist table in `docs/ui-checklist.md` with pass / fail per row.
- [ ] **UI-2 (M) Gizmo pass.** `bl/handles.py` binds a fixed gizmo pool on redraw. Verify drag
      direction, scale and undo for each of the 18 SOPs in `HANDLES`. Fix axis mapping (networks
      are Y-up, objects are rotated 90 degrees on X). *Accept:* dragging an arrow moves geometry
      under the cursor for box, xform, tube, pyrosolver.
- [ ] **UI-3 (S) GUI smoke script.** A script that runs under `blender --python` *with* a window
      (xvfb is enough): opens the node editor, calls each operator through `bpy.ops` with a real
      context override, takes a screenshot per editor. Add to `tests/`. *Accept:* runs unattended
      and saves screenshots; wired into the checklist above.
- [ ] **UI-4 (S) Playback under the GUI.** Play all 20 examples for 100 frames with the timeline
      (not `frame_set`), then save, reload and render one frame of each. This is the path that
      crashed twice before (RBD in the frame handler, constraint teardown).
- [ ] **UI-5 (S) Older / newer Blender.** Record which Blender versions load the extension
      (4.5 LTS, 5.0, 5.1, 5.2) and gate the unsupported API uses (slotted actions, SDF grid nodes,
      `hair_curves.add_curves`) behind version checks with a clear message.

## P1: Core gaps (highest user value)

- [ ] **GEO-1 (L) Vertex-level and primitive-type model.** Today a primitive is a closed
      polygon or an open polyline. Add a per-primitive `type` (poly, polyline, nurbs, bezier,
      packed, volume) with `primintrinsic` exposure, so curves keep their basis through the
      network and are evaluated only at display / conversion. Unblocks native NURBS and Bézier.
      *Files:* `core/geometry.py`, `bl/mesh_io.py`, `bl/curves_io.py`, Convert SOP.
      *Accept:* a NURBS curve survives Transform + Merge + File Cache and renders smooth.
- [x] **GEO-2 (M) Attribute type info.** (done: `geo.typeinfo`, used by Transform; not yet persisted by VEX `setattribtypeinfo`) Store type qualifiers (point, vector, normal, color,
      quaternion, matrix) per attribute so Transform handles every vector attribute correctly,
      not just the hard-coded `N`, `v`, `up`. Expose `setattribtypeinfo` / `attribtypeinfo`.
      *Accept:* a custom `v@dir` typed "vector" rotates with Transform; typed "point" translates.
- [ ] **GEO-3 (M) Array and dict attributes on geometry.** VEX has arrays and dicts, but only
      scalars, vectors and strings round-trip through `Geometry`, File Cache and the spreadsheet.
      *Accept:* `i[]@nbrs = nearpoints(...)` in one wrangle is readable in the next and in a
      cache file.
- [ ] **COOK-1 (M) Partial recook and cook cancellation.** A long cook blocks Blender. Run cooks
      in slices from a timer with an ESC check between nodes, and show progress in the status
      bar. *Accept:* ESC during a 30 s simulation bake returns control within 1 s and leaves the
      previous result displayed.
- [ ] **COOK-2 (M) Memory budget for caches.** Each solver bounds its own frame cache. Add one
      global budget (preference, default 2 GB) with LRU eviction across node caches and solver
      frames, and show usage in the Network panel. *Accept:* playing all examples stays under
      the budget.
- [ ] **COOK-3 (S) Background / command line rendering test.** `blender -b file.blend -a` must
      produce correct frames for every solver (they bake ahead on load). Add an integration
      check that renders frames 1 and 20 of each simulation example in a fresh process and
      compares point positions with an in-session cook.
- [ ] **VEX-1 (M) VEX conformance suite.** Collect 200 snippets with expected results taken
      from Houdini documentation examples into `tests/vex_conformance/*.vfl` + expected JSON.
      Report a pass percentage in the README. *Accept:* suite runs in under 20 s; failures are
      listed as roadmap items.
- [ ] **VEX-2 (M) Missing language features.** `switch`-less is fine, but still missing:
      function prototypes / forward declarations, `export` parameters, `foreach` with `break`
      inside nested loops under divergent masks (verify), integer overflow semantics (32-bit
      wrap), `float` precision (32-bit) switch for exact Houdini matches, string indexing and
      slicing with negative steps, `%` formatting of arrays in `printf`.
- [ ] **VEX-3 (L) Compile to numpy kernels.** The interpreter walks the AST per statement. Cache
      a compiled closure tree per snippet (no dict lookups or `kind` dispatch at run time), and
      fuse pure arithmetic expression chains with `numexpr`-style evaluation when available.
      *Accept:* the 100k point noise and `for` loop benchmarks in README get 2x faster with
      identical results.
- [ ] **VEX-4 (M) Multi-line code editor.** Blender has no multi-line text widget in panels. Do
      it with a synced Text datablock: an "Edit Code" button splits the node editor and opens
      the Text Editor on the node's text with live recook (exists), plus a read-only syntax
      colored preview in the parameter panel (exists, 40 lines). Missing: error line highlight
      in the Text Editor, autocomplete of builtins and attribute names, and a function tooltip.
      *Files:* `bl/ui.py`, `bl/node_overlay.py`.

## P2: Simulation

- [ ] **SIM-1 (L) DOP-like solver framework.** Each solver is a monolithic SOP. Extract the
      shared pattern (state key, per-frame cache, upstream-edit detection, substeps, cache
      eviction, bake-ahead) into `sops/solver_base.py` and port Vellum, FLIP, PBF, Pyro, POP and
      grains onto it. Then add **forces as nodes** (gravity, wind, drag, noise, attract) feeding
      a forces input. *Accept:* each solver file loses its copy of the cache logic; behaviour
      tests unchanged.
- [ ] **SIM-2 (M) Solver checkpoints on disk.** Solver frames live in RAM and are lost on file
      reload. Reuse `core/geoio.py` to persist frames next to the .blend (opt-in per node), so
      reopening a scene does not resimulate. *Accept:* reload + jump to frame 60 costs one file
      read.
- [ ] **FLIP-1 (L) MAC grid FLIP.** The collocated grid dissipates and can checkerboard. Move
      velocities to a staggered grid, use a conjugate-gradient pressure solve (matrix-free,
      numpy), APIC transfer. *Accept:* a still tank stays still (kinetic energy decays to 0),
      dam break volume loss under 2 percent after 100 frames.
- [ ] **FLIP-2 (M) Narrow band and reseeding**, **FLIP-3 (M) surface tension and viscosity**,
      **FLIP-4 (M) sources and sinks** (emit from geometry over time, kill box). After FLIP-1.
- [ ] **FLIP-5 (M) Better liquid surface.** Particle Fluid Surface uses Blender's points to
      volume. Add anisotropic kernels or a native `core/volume.py` splat + smooth + surface nets
      path so it works without `bpy` and keeps thin sheets.
- [ ] **PYRO-1 (M) Pyro quality.** MacCormack advection, multigrid or CG pressure solve,
      disturbance and shredding, sourcing from multiple named fields, sparse / resizing
      container that tracks the smoke bounds. *Accept:* same look at half the voxel count.
- [ ] **PYRO-2 (S) Fire shading.** Blackbody from a `temperature` grid instead of the flame
      ramp, plus a `heat` field for emission falloff.
- [ ] **RBD-1 (M) Initial velocity and forces.** Blender's rigid body API has no velocity input.
      Options: (a) keyframed kinematic lead-in over two frames using `use_deactivation` off and
      animated `kinematic` (tried once, did not transfer velocity; re-test with substeps and
      `rigidbody_world.time_scale`), (b) apply an impulse through a force field object for one
      frame, (c) native Python rigid body solver for small piece counts. *Accept:* `v@v` on a
      piece produces the expected ballistic arc.
- [ ] **RBD-2 (M) Collision shape and per-piece attributes.** `i@collisionshape` (hull, box,
      sphere, mesh), `f@friction`, `f@bounce`, `i@animated`, deforming colliders, and reading
      per-piece transforms from an animated input (kinematic pieces).
- [ ] **RBD-3 (M) Fracture tools.** RBD Material Fracture style presets (concrete, glass, wood
      via anisotropic Voronoi scaling), edge detail noise on interior faces, clustering of
      pieces (`name` merge by noise), interior UVs.
- [ ] **VEL-1 (M) Vellum constraints as geometry.** Like RBD constraint networks: a Vellum
      Constraints SOP that writes stretch / bend / pin / attach / stitch constraints as
      polylines with `stiffness`, `restlength`, and a solver input for them. Enables attach to
      animated geometry and stitched panels.
- [ ] **VEL-2 (M) Cloth self-collision** (point vs triangle with the spatial hash), **VEL-3 (S)
      wind / drag from normals**, **VEL-4 (M) soft bodies** (tetrahedral or shape-matching),
      **VEL-5 (S) XPBD compliance** so stiffness is independent of iterations and substeps.
- [ ] **POP-1 (M) Particle follow-ups** (after the POP Solver lands): POP nodes as inputs
      (force, wind, drag, attract, curve force, kill, color, replicate), collision with packed
      RBD pieces, instancing onto particles with `orient` from velocity, sprite / trail output.
- [ ] **CROWD-1 (L, optional) Agents.** Packed agents with clips from Blender armature actions
      and a simple steering solver. Only after SIM-1.

## P2: Modeling SOPs

Listed by how often they appear in Houdini scenes and are still missing (check `REGISTRY`
first, some may have landed):

- [ ] **MOD-1 (M) Edge-based modeling:** PolySplit / Edge Loop (insert loops on quad strips),
      Edge Cusp, Edge Collapse, Edge Flip, Edge Divide. Needs an edge group type: add
      `geo.groups['edge']` storing point pairs, with pattern syntax `p0-1 p3-4`.
- [ ] **MOD-2 (M) Vertex groups and vertex wrangle parity** (`vtxnum` patterns, `Group Create`
      for vertices, promote to and from vertex groups).
- [ ] **MOD-3 (M) PolyExtrude parity:** extrude along point normals with `zscale` ramp, twist,
      divisions with a profile ramp, output front / side / back groups (front exists), extrude
      edges and curves.
- [ ] **MOD-4 (M) Boolean parity:** output groups for seams and inside / outside faces, shatter
      mode (cut A by surface B into pieces), detriangulate. Wraps Blender's exact solver.
- [ ] **MOD-5 (S each)** Remesh to grid / quad remesh (Blender QuadriFlow), Subdivide with
      creases (`creaseweight` edge attribute), PolyReduce with keep-borders and attribute
      preservation, Thicken (extrude both sides), Clip with fill, Knife, Hole, Dissolve by
      angle, Reverse per group, Peak with mask, Smooth with attribute and cotangent weights,
      Twist / Taper / Linear Taper / Squash as separate nodes, Wave and Ripple deformers.
- [ ] **MOD-6 (M) Curve tools** (after the current curve batch): Sweep parity (roll and scale
      ramps, UVs, end cap types, apply `orient`), Rails, Loft / Skin with different point
      counts (resample first), Curve Boolean in 2D, Find Shortest Path (Dijkstra with cost
      attribute, start and end groups), Trace (image to curves), Carve animated by `u` ramps.
- [ ] **MOD-7 (M) UV tools:** UV Flatten with seam edge groups, UV Quick Shade (checker
      material), UV Project with a handle, UV Edit round trip in Blender's UV editor (same
      pattern as the Edit SOP), UDIM layout, texel density measure.
- [ ] **MOD-8 (M) Scatter parity:** relax against a density attribute, scale by area with
      per-primitive count, output `sourceprim` (exists) and `sourceprimuv` (exists), Scatter
      and Align (pscale / orient / N with ramps), blue noise by Poisson disk in volumes,
      Points from Volume jitter and SDF source from native volumes.
- [ ] **MOD-9 (S each)** Labs-style helpers: Edge Damage, Lot Subdivision, Building Generator
      (floors / windows from modules via Copy to Points), Quick Material, Color Gradient, Axis
      Align, Auto Pivot, Delete Small Parts, Measure Curvature visualizer, Gaea-style Terrace.
- [ ] **HF-1 (M) Terrain:** HeightField layers as real 2D volumes (height, mask, any named
      layer) on top of `core/volume.py` instead of a point grid, so resolution is independent of
      mesh density. Then HeightField Distort, Remap (ramp), Clip, Layer (combine), Project
      (geometry to height), Draw Mask (Attribute Paint pattern), Mask by Occlusion / Flow,
      Slump, Tile Split, export to image, and an Erode with water / sediment / debris layers.

## P2: Volumes

- [ ] **VOL-1 (L) Sparse volumes.** Dense numpy grids cap out around 256^3. Wrap OpenVDB grids
      through the `openvdb` module Blender ships when it is importable (keep dense as the
      fallback so the unit tests still run without it): `Volume` gets a backend interface
      (`sample`, `gradient`, `to_dense(bbox)`, `from_dense`), SOPs call the interface.
      *Accept:* VDB from Polygons at voxel size 0.01 on a unit sphere cooks in under 5 s.
- [ ] **VOL-2 (M) VDB tools on native volumes:** Combine (union / intersect / difference /
      add / max / blend, with resampling between different grids), Resample, Reshape SDF
      (dilate / erode / open / close) and Smooth SDF natively (today they go through Geometry
      Nodes on meshes), Analysis (gradient, curvature, divergence, curl), Advect, Renormalize,
      Fracture (cut an SDF by another), Segment by connectivity, Activate / Clip.
- [ ] **VOL-3 (S) File formats:** read and write `.vdb` in the File and File Cache SOPs (the
      writer exists in `bl/volume_io.py`), and volume export for the Bake operator.
- [ ] **VOL-4 (S) Volume display:** slice plane overlay in the viewport, density / emission
      controls on the display node, velocity field vector overlay (reuse the vector overlay).

## P2: Pipeline and interchange

- [ ] **IO-1 (M) Alembic and USD in / out.** Use Blender's importers and exporters through a
      temporary object (pattern: `sops/blender_ops.py` `ObjectMergeSOP._read`). File SOP reads
      .abc / .usd / .usdc / .ply / .stl / .fbx / .gltf with attributes where Blender exposes
      them; a ROP-style Export node writes sequences with `$F4`.
- [ ] **IO-2 (M) bgeo.sc / .geo (JSON) reader.** (ASCII reader / writer done from the format spec; still needs a real Houdini file) Houdini's ASCII `.geo` is documented JSON. A
      reader makes real Houdini caches and test assets loadable. Binary `.bgeo.sc` needs blosc;
      do ASCII first. *Accept:* loads a `.geo` saved from Houdini Apprentice with point / prim /
      vertex / detail attributes and groups.
- [ ] **IO-3 (S) VEX library files.** `#include "file.h"` from a search path next to the .blend
      and from a preference directory (text only, still no Python).
- [ ] **IO-4 (M) Network copy / paste as text.** Serialize selected nodes with parameters,
      spare parameters, snippets and wiring to JSON on the clipboard, and paste it back. Also
      the basis for sharing snippets online and for diff-friendly asset files (`.blendini`).
- [ ] **IO-5 (M) Digital asset library.** Assets live in a .blend today. Add save / load of a
      definition to a standalone file (IO-4 format), versioning (`name::2.0`), an asset
      parameter interface editor (order, labels, ranges, folders, hide-when), and help text.
- [ ] **IO-6 (S) Render integration.** Per-point `Cd` works. Add automatic motion blur from `v`
      for meshes and points (velocity attribute name Blender expects), `shop_materialpath` per
      instance, and a "Render Flag" separate from the display flag.

## P3: Look and feel

- [ ] **UX-1 (M) Network editor parity:** node ring / radial flag menu, Y to cut wires, Shift+Z
      / L layout, colored wires for errors, node badges for time dependence and caching, node
      comments, sticky notes, quick-insert on wire drop, drag a node onto a wire, Alt+drag to
      copy, Ctrl+click for a secondary display, select-and-press-Shift+C to collapse into a
      subnet (Extract Contents / Collapse into Subnet operators).
- [ ] **UX-2 (M) Parameter interface:** folders / tabs, ramp parameters on standard SOPs (not
      only spare parameters), multiparms (lists of rows), menu-driven group fields listing
      existing groups, attribute name dropdowns populated from the input, expression fields
      with a green background when driven, RMB "Copy Parameter" / "Paste Relative Reference",
      revert to default, presets per node type saved to the user's config.
- [ ] **UX-3 (M) Geometry Spreadsheet as a real editor:** a dedicated area with sorting,
      filtering by group / expression, per-class tabs, selection sync with the viewport
      (requires a custom draw panel or Blender's spreadsheet through a mirrored mesh).
- [ ] **UX-4 (S) Viewport:** display options per node (points on / off, point size, wireframe
      over shaded, backface tint, group visualization colors, UV view), visualizers as nodes
      (marker, color, vector trail), ghost other objects when diving in, guide geometry from
      SOPs (bounding regions, container boxes) drawn as overlay not as mesh.
- [ ] **UX-5 (S) Onboarding:** a first-run panel with the examples, a searchable in-app help
      page per node generated from `Parm.description` and `SOP.description` (also exported to
      `docs/nodes.md` so the README table is generated, not hand-edited), and tooltips that
      name the Houdini equivalent.
- [ ] **UX-6 (S) Localization-ready strings and keymap preferences** (the keymap is fixed).

## P3: Quality and infrastructure

- [ ] **QA-1 (S) Continuous integration.** GitHub Action: unit tests on Python 3.11 to 3.13,
      the `bpy` wheel for the integration test, the fuzzers with fixed seeds, pyflakes. Cache
      the wheel. *Accept:* green badge in the README.
- [ ] **QA-2 (M) Golden results.** Hash geometry (point count, bbox, attribute checksums) of
      every example at frames 1, 12 and 24 into `tests/golden.json` and compare in CI, so
      behaviour changes are deliberate. Solvers need fixed seeds (they have them).
- [ ] **QA-3 (S) Performance benchmarks** in `tests/bench.py` with a JSON history and a 20
      percent regression alarm: wrangles at 100k and 1M points, Copy to Points 10k x 100,
      Scatter 1M, Voronoi fracture 200 pieces, each solver for 24 frames.
- [ ] **QA-4 (M) Split the large files.** `sops/modify.py` (~3000 lines), `bl/ui.py` (~1900),
      `vex/builtins.py` (~2000) and `sops/extra.py` are hard to navigate. Split by topic
      (deform, polygon, group, copy, utility; operators, panels, menus, handlers) with no
      behaviour change, one commit per file, tests green after each.
- [ ] **QA-5 (S) Type hints and docstrings on the public core API** (`Geometry`, `Volume`,
      `cook_sop`, `run_wrangle`) and a short `docs/architecture.md` with the cook and VEX data
      flow, so new contributors do not need to read the whole tree.
- [ ] **QA-6 (S) Extension release.** Version bump policy, changelog, license check of the
      bundled examples, publish to extensions.blender.org (needs the P0 pass first).
- [x] **QA-7 (S) Property-based tests** for the geometry core (random delete / merge / sort /
      fuse sequences keep topology arrays consistent: `prim_start` monotone, vertex indices in
      range, attribute lengths equal counts, groups sized right). A `Geometry.validate()`
      method used by the fuzzer after every cook.

## Known limitations to keep in mind

- RBD uses private Blender scenes. Creating or stepping them inside a frame change handler
  crashes Blender, and constraints must be removed before bodies. See `_rbd_cleanup`.
- Face and corner string attributes crash Geometry Nodes in Blender 5.2, so only point strings
  are written to meshes.
- Fog volumes are written as temporary `.vdb` files each cook (one file per network).
- Networks are Y-up, Blender objects are Z-up (output objects are rotated 90 degrees on X).
- The numpy fallbacks for BVH and KD-tree queries are exact but slow. Inside Blender,
  `mathutils` is used for closest-point and ray queries.

## Deliberately not planned

- Python SOP: networks must stay safe to open (no Python from .blend files is executed).
