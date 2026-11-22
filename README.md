# 3d-world

A procedural 3D planet simulator tech demo in Rust: GPU-simulated terrain, climate and
ecosystems, rendered as photorealistically as we can manage.

Built on [Bevy](https://bevy.org) 0.20 (wgpu, Vulkan backend). The heavy lifting is GPU
compute: each simulation subsystem is a set of compute passes over planet-wide fields.

![planet](docs/planet.png)

## Status

Step 1 of the plan: the app skeleton and a reusable GPU simulation framework, exercised by a
procedural heightfield and a thermal-erosion pass. Next: hydraulic erosion, climate,
vegetation, animals, terrain LOD, atmosphere.

## Running

```sh
cargo run --release                  # window: drag to orbit, scroll to zoom
cargo run --release -- --help        # options
cargo run --release -- --headless --steps 400 --out out   # no window: writes out/planet.png
```

Keys: **Space** pause, **R** new planet (next seed), **Up/Down** simulation steps per frame.

Headless mode renders to an offscreen image, saves `planet.png` and `height_stats.txt`, and
exits. It works on a software Vulkan driver (Mesa lavapipe), which is what CI uses.

## Layout

```
src/sim/          GPU simulation framework (reusable, knows nothing about planets)
  mod.rs          fields, passes, controls, the main-world API
  render.rs       render-world side: pipelines, bind groups, dispatch, double buffering
src/planet/       the planet: declares fields + passes, renders from them
  mod.rs          TerrainParams, PlanetMaterial, the simulation declaration
  mesh.rs         cube-sphere mesh (same face convention as the shaders)
assets/shaders/
  sim/common.wesl         globals, cube-sphere addressing, cross-face neighbours, noise
  planet/terrain_*.wesl   terrain passes
  planet/planet_material.wesl   displaces + shades the planet from the height field
```

## The simulation framework

Planet state lives in **fields**: cube-sphere texture arrays with 6 layers (one per cube face)
of `resolution × resolution` texels. A **pass** is a compute entry point that reads some
fields and writes others. A subsystem is declared in a few lines:

```rust
graph
    .add_field(HEIGHT, FieldFormat::R32Float)
    .add_pass(
        SimPass::new("terrain_init", "shaders/planet/terrain_init.wesl", "terrain_init")
            .write(HEIGHT)
            .trigger(PassTrigger::OnReset)
            .with_params(params),
    )
    .add_pass(
        SimPass::new("terrain_relax", "shaders/planet/terrain_relax.wesl", "terrain_relax")
            .read(HEIGHT)
            .write(HEIGHT)          // read + write => framework double-buffers it
            .with_params(params),
    );
```

The framework allocates the textures, builds bind group layouts and bind groups from the
declaration, compiles pipelines (and waits for all of them before the first step), runs passes
in order each step, double-buffers fields a pass both reads and writes, uploads per-step
globals and per-pass params, and handles reset and pause.

Shader contract, all in `@group(0)`:

| binding | contents |
|---|---|
| 0 | `globals: SimGlobals` (resolution, step, seed, dt, time) |
| 1 | `params`: the pass's own uniform struct (`#[repr(C)]` + `bytemuck::Pod` on the Rust side) |
| 2.. | the pass's field bindings in declaration order: reads are `texture_2d_array<f32>` (use `textureLoad`), writes are `texture_storage_2d_array<FORMAT, write>` |

Dispatch is `@workgroup_size(8, 8, 1)` over `res × res × 6`, with `global_invocation_id.z` as
the cube face. `neighbor_texel` in `sim/common.wesl` steps across cube edges, so stencils
work seamlessly over the whole sphere.

### Single queue

wgpu exposes one queue, so simulation and rendering share every frame's GPU time.
`SimControl::steps_per_frame` is the time-slicing knob; for long runs, run headless with many
steps per frame (a "bake"), or pause the simulation while exploring.

### Cube-sphere convention

Face `f` has axes `(N, U, V)` (see `FACE_AXES` in `src/planet/mesh.rs` and
`face_normal/face_u/face_v` in `common.wesl`; they must agree). Texel `(x, y)` maps to
`s = (x + 0.5) / res * 2 - 1`, then an equal-angle warp `a = tan(s · π/4)`, then
`dir = normalize(N + a·U + b·V)`. Texel areas vary by about 1.4× across a face.

## Licence

MIT
