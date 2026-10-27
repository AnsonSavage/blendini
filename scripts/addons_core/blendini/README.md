# Blendini

**Houdini-style procedural SOP networks for Blender, with a VEX wrangle.**

Blendini is an open-source take on Houdini's geometry context. It uses Blender as the
host and backend. You build a network of SOP nodes in a node editor. The node with the
**display flag** cooks into a Blender object. The **Point Wrangle** runs real (subset)
VEX code over every point in parallel.

It ships as a core add-on in this Blender fork (`scripts/addons_core/blendini`). It's
also plain Python + numpy, so it can be zipped and installed into a stock Blender 5.x.

## Installing into a regular Blender

This fork ships Blendini as a core add-on, but it is pure Python + numpy and also works in a
stock Blender (4.2+, developed against 5.2):

```sh
cd scripts/addons_core/blendini
blender --command extension build      # creates blendini-0.1.0.zip
```

Then use *Preferences → Get Extensions → Install from Disk* with the zip.

## Quick start

**Fastest:** *File → New → Blendini* (or pick Blendini on the splash screen). This
application template enables the add-on, starts with an empty scene and a `geo1` network,
and arranges a Houdini-style layout.

Or step by step:

1. Enable **Blendini** in *Preferences → Add-ons*.
2. In the 3D viewport choose *Add → Blendini Geometry*. This creates `geo1` (an object
   plus its SOP network, like `/obj/geo1`) and opens the network in a node editor.
3. Or use *Add → Blendini Examples* to build eight example networks: ripple, scatter and
   copy, plexus, city, particles, for-each, L-system tree and spiral.
4. In the node editor:
   - **Tab**: Houdini-style TAB menu (start typing to search).
   - **R**: set the display flag on the active node.
   - **M**: bypass (Blender's mute, shown as a pass-through).
   - Each node has **display**, **template** (ghosted wireframe guide) and **bypass** flag
     buttons.
   - **N** sidebar → **Parameters** tab: the parameter pane and the geometry info
     (point and primitive counts, attributes, groups, cook time, `printf` output), plus
     *Save Geometry to Disk* (.obj).
   - *Blendini* tab → *Apply Houdini-like Theme* for the grey gradient viewport and a dark
     network editor. *Houdini-like Layout* splits the screen into viewport + network +
     spreadsheet.
   - **N** sidebar → **Spreadsheet** tab: Houdini's Geometry Spreadsheet for the *active*
     node (points / vertices / primitives / detail, attributes and groups).
   - Viewport display of point numbers, primitive numbers and normals: the *Blendini* tab of
     the node editor, or the 3D viewport's sidebar.
5. On a wrangle, click **Edit**. The VEX code opens in a Text Editor, with C-like syntax
   highlighting, and the network **recooks live as you type**. The **+** button creates
   spare parameters for every `ch("name")` call, like Houdini's "create spare parameters"
   button.

Blender is Z-up and Houdini is Y-up. Output objects are rotated 90° around X so
`@P.y` stays "up" in your VEX. Object Merge converts the other way.

## Nodes

| Category  | SOPs |
|-----------|------|
| Primitive | Box, Grid, Sphere (UV / icosahedron), Tube, Torus, Circle, Line, Platonic Solids, Bound, Font\*, L-System |
| Manipulate| Transform, Mountain, Point Jitter, Peak, Smooth, Ray\* |
| Polygon   | Add, Fuse, Facet, Divide, Reverse, PolyExtrude, PolyWire, Sweep, Resample, Subdivide\*, PolyBevel\*, Boolean\*, Remesh\*, PolyReduce\*, Clip\* |
| Attribute | Attribute Create / Delete / Rename / Promote / Randomize / Noise / Transfer, Color, Normal, Connectivity, Measure, UV Texture, PolyFrame |
| Group     | Group Create (pattern / bounding box / normal / random / `@expression`), Blast |
| Copy      | Copy to Points (orient, N/up, pscale, scale, **Pack and Instance**), Unpack, Copy and Transform, Scatter |
| VEX       | Attribute Wrangle, **Point Wrangle**, Primitive Wrangle, Detail Wrangle, **Solver (VEX)** |
| Utility   | File (.obj), Merge (multi-input), Switch, Null, Output, Sort, Time Shift, **Block Begin / Block End (For-Each pieces or feedback)** |
| Blender   | Object Merge, Solidify\*, **Blender Modifier\*** (any modifier, including Geometry Nodes groups) |

\* These use Blender as the backend: the geometry passes through a real Blender modifier
or `bmesh` operator and comes back with its attributes.

**Packed geometry**: when *Pack and Instance* is on, Copy to Points outputs one point per copy
that references the source geometry, the same as Houdini's packed primitives. Transform,
Merge and so on work on the packed points. On output, the copies become real Blender
instances through a generated Geometry Nodes modifier, so millions of copies stay cheap.
Per-copy attributes such as `Cd` are available to materials through the
*Attribute (Instancer)* node. *Unpack* converts packed points back into real geometry.

Group fields accept Houdini patterns: group names, `0-10 15`, `!name`, and VEX
expressions such as `@P.y > 0` or `@ptnum % 2 == 0`.

## VEX

The wrangle doesn't loop over points in Python. The snippet is parsed into an AST, then
run **SIMD style**: every value is a numpy array with one lane per element. `if`/`else`,
loops, `break`/`continue` and `return` work by masking lanes, much like a GPU, so
typical wrangles run at numpy speed. For example, 10k points with `nearpoints` and
`noise` cook in about 0.2 s.

Supported language:

- Types: `int float vector2 vector vector4 matrix2 matrix3 matrix string` and arrays (`int[]` …).
- `@` bindings with prefixes (`f@ i@ v@ p@ u@ s@ 2@ 3@ 4@ i[]@` …) and known names
  (`@P @N @Cd @v @orient @pscale @id @name` …).
- `@ptnum @primnum @numpt @numprim @elemnum @numelem @Time @Frame @TimeInc`,
  `@group_name` and `@opinput1_P`.
- Operators with VEX semantics: integer division, `vector * matrix`, `.xyz`/`.zyx`
  swizzles, component and array indexing, `+= -= *= /= %=`, `++`/`--`, ternary,
  `&& || !`, and C-style or function-style casts.
- `if/else`, `for`, `while`, `do/while`, `foreach (int i; int x; arr)`, `break`,
  `continue`, `return`.
- User functions with VEX's **pass-by-reference** parameters, recursion and early `return`.
- Return-type overloading from context: `vector r = rand(@ptnum);` and
  `vector n = noise(@P);` return vectors.

Builtins (partial list):

- Math: `sin cos tan asin acos atan atan2 sqrt pow exp log abs sign floor ceil frac round
  min max clamp fit fit01 fit10 fit11 efit lerp smooth avg sum`.
- Vectors: `length length2 distance distance2 normalize dot cross reflect set getcomp`.
- Matrices and quaternions: `ident rotate scale translate transpose invert determinant
  maketransform dihedral quaternion qconvert qrotate qmultiply slerp eulertoquaternion`.
- Random: `rand random noise snoise onoise curlnoise sample_direction_uniform`.
- Geometry reads: `point prim vertex detail npoints nprims nvertices primpoints pointprims
  primpoint primvertexcount neighbours neighbourcount nearpoint nearpoints
  getbbox_min/max/center/size relbbox inpointgroup inprimgroup haspointattrib
  vertexpoint vertexprim idtopoint findattribval primarea`.
- Geometry writes: `addpoint addprim addvertex removepoint removeprim setpointattrib
  setprimattrib setvertexattrib setdetailattrib setpointgroup setprimgroup`. The `"add"`,
  `"min"`, `"max"` and `"mult"` modes are supported.
- Arrays: `len append push pop removeindex resize find sort argsort reverse array
  isvalidindex`.
- Strings: `sprintf printf itoa atof atoi strlen concat startswith endswith toupper tolower`.
- Channels: `ch chf chi chv chs chramp`, backed by spare parameters. Ramps are curve widgets.
- Point clouds: `pcopen pcfilter pcnumfound pcclose pcfind`.
- Closest point queries: `xyzdist` (with `prim`/`uv` output arguments), `primuv`, `minpos`, using Blender's BVH tree.
- Constants from `math.h`: `M_PI`, `M_TWO_PI`, `M_E`, `FLT_MAX` and others.

As in Houdini, reads with `point()` etc. see the **input** geometry. `@` writes and
geometry edits are applied after the snippet finishes.

## Example: ripple Point Wrangle

```c
float d = length(@P);
float freq = ch("freq");
float amp = ch("amp");
@P.y = sin(d * freq - @Time * 4) * amp * exp(-d * 0.3);
@Cd = set(0.2, 0.5 + @P.y, 1.0 - @P.y);
```

## Architecture

```
blendini/
  core/geometry.py   points / prims / vertices / detail attributes + groups (numpy)
  core/noise.py      vectorized Perlin noise, curl noise, hash random
  vex/parser.py      lexer + recursive descent parser for VEX
  vex/interp.py      SIMD masked interpreter
  vex/builtins.py    VEX function library
  vex/wrangle.py     binds attributes, runs the snippet, applies deferred geometry edits
  sops/              declarative SOP definitions (no Blender dependency, except blender_ops.py)
  bl/                Blender integration: node tree, cooking/cache, mesh conversion, UI
  tests/             unit tests runnable without Blender
```

Cooking is signature-based. Each node result is cached with a hash of its parameters,
its inputs' signatures and, when time dependent, the frame. Only nodes whose upstream
changed recook, like Houdini's dirty propagation. Parameters can be animated with
Blender keyframes or drivers (`#frame`), which play the role of Houdini's `$F`
expressions.

Run the tests (no Blender required, only numpy):

```sh
cd scripts/addons_core
python -m unittest blendini.tests.test_blendini
```

## Security

Opening a .blend file with Blendini networks never runs Python from the file. VEX is
interpreted by Blendini itself and can only read and write geometry. That's also why
there is no Python SOP.

## Not done yet (ideas)

NURBS/curves, volumes/VDB,
a DOP context, and a multi-line code editor inside the parameter pane.
