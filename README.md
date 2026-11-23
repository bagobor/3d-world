# 3d-world

A procedural 3D planet simulator tech demo in Rust: GPU-generated planets and living
ecosystems, rendered as photorealistically as we can manage.

Built on [Bevy](https://bevy.org) 0.20 (wgpu, Vulkan backend). The heavy lifting is GPU
compute: each simulation subsystem is a set of compute passes over planet-wide fields.

![planet](docs/planet.png)

| low orbit | eye height |
|---|---|
| ![orbit](docs/orbit.png) | ![surface](docs/surface.png) |

| over a forest | in a meadow |
|---|---|
| ![forest](docs/forest.png) | ![meadow](docs/meadow.png) |

![herd](docs/herd.png)

| snow-capped peaks | canyon country |
|---|---|
| ![peaks](docs/peaks.png) | ![canyon](docs/canyon.png) |

| open sea | surf on the coast |
|---|---|
| ![sea](docs/sea.png) | ![coast](docs/coast.png) |

## Status

- Step 1: the app skeleton and a reusable GPU simulation framework, with a procedural
  heightfield and thermal erosion.
- Step 2: hydraulic erosion. Rain, pipe-model water flow, sediment transport, erosion and
  deposition carve drainage networks, lakes and deltas.
- Step 3: planet generation bake. Terrain and erosion run once, offline, as a generate
  phase; then static moisture, temperature and biome maps are derived. The result is saved
  to a disk cache and later starts load it and run no geology (screenshot: seed 7, 3000
  generation steps at 512²).

- Step 4: live plant ecosystem. Grass, shrubs and trees grow, shade each other, spread and
  burn in wildfires, on top of the baked planet (screenshot: 600 live steps).

- Step 5: level-of-detail terrain. A quadtree of chunks per cube face streams in detail
  around the camera, from the whole planet down to eye height, with a floating origin.

- Step 6: atmosphere, ocean and clouds. Physically based sky and aerial perspective from the
  ground to orbit, a shaded ocean with waves and surf, and a drifting cloud layer that casts
  shadows.

- Step 7: vegetation. Grass, shrubs and trees as real geometry near the camera, placed per
  biome and grown in by the live ecosystem, so forests spread and burn on screen.

- Step 8: animal populations. Grazers and predators live on the plant ecosystem: they feed,
  breed, die, migrate towards food and are killed by wildfire; grazing thins the plant
  cover.

- Step 9: animals near the camera. Herds of grazers and packs of predators appear in
  proportion to the live population densities; herds graze and keep together, flee from
  predators, and predators hunt.

- Step 10: lighting polish. Sun shadows (cascaded shadow maps out to 6 km) cast by the
  terrain, trees, shrubs and animals, received by everything; plants cast shadows grown in
  exactly as drawn.

There is no climate simulation and no seasons: the environment maps are static.

The plan's steps are all in; what's next is open.

## Running

```sh
cargo run --release                  # window (controls below)
cargo run --release -- --help        # options
cargo run --release -- --headless --out out  # no window: writes out/planet.png
cargo run --release -- --seed 3 --regenerate  # ignore the cached planet, generate again
```

Camera modes, switched with **1 / 2 / 3**:

- **1 Orbit** (default): **left-drag** moves over the planet, **right-drag** looks around,
  **scroll** changes altitude (from 20 planet radii down to eye height), **WASD** moves
  (**Shift** faster).
- **2 Fly**: free flight. **Left-click** captures the mouse for looking around (**Esc**
  releases it; **right-drag** also looks). **WASD** flies along the view, **E/Q** up and
  down, **Shift** faster, **scroll** sets the speed. Speed grows with height above the
  ground, from 15 m/s near it to hundreds of km/s in space.
- **3 Walk**: on foot at 1.7 m eye height, following the terrain (on the ocean, on its
  surface). Mouse look as in fly mode, **WASD** walks, **Shift** runs, **Space** jumps.

Other keys: **Space** (or **P**; only **P** while walking) pauses the simulation, **R**
generates a new planet (next seed), **Up/Down** live simulation steps per frame.

The first start with a given seed and resolution generates the planet
(`--generation-steps`, default 3000) and writes it to `cache/planet-v1-seed<seed>-res<res>.bin`
(about 36 MiB at 512²). Later starts load that file in well under a second. `--no-cache`
turns the cache off, `--cache DIR` moves it.

Headless mode renders offscreen and saves twelve views, `planet.png` (whole planet),
`orbit.png` (low orbit), `surface.png` (40 m above hilly land near the middle of the first
view), `forest.png` (45 m over the nearest forest), `meadow.png` (eye height in the
nearest grassland), `fly.png` (fly mode, 600 m up), `walk.png` (walk mode, eye height) and
`herd.png` (next to the nearest herd), `peaks.png` (the highest summit outside the polar
caps, from 25 km) `canyon.png` (down in the deepest canyon in view), `sea.png` (7 m above open sea) and
`coast.png` (in the shallows, looking at a beach), each once its terrain and vegetation have streamed in
(`--views forest,meadow` picks some), plus `stats.txt` (min/max/mean of every
field and plant layer, and the share of each biome), and exits. `--steps N` runs N
live steps first. It works on a software Vulkan
driver (Mesa lavapipe), which is what CI uses; generating at 512² takes about 8 minutes there.

## Layout

```
src/sim/          GPU simulation framework (reusable, knows nothing about planets)
  mod.rs          fields, passes, controls, the main-world API
  render.rs       render-world side: pipelines, bind groups, dispatch, double buffering
  cache.rs        baked fields on the CPU, disk cache
src/planet/       the planet: declares fields + passes, renders from them
  mod.rs          TerrainParams, PlanetMaterial, the simulation declaration
  mesh.rs         cube-sphere mesh (same face convention as the shaders)
  surface.rs      CPU surface: bicubic baked height + procedural detail, in f64
  lod.rs          quadtree LOD terrain chunks
  camera.rs       planet camera (f64) and floating origin
  sky.rs          atmosphere
  vegetation.rs   grass, shrub and tree tiles near the camera
  fauna.rs        individual animals near the camera, from the population fields
assets/shaders/
  sim/common.wesl         globals, cube-sphere addressing, cross-face neighbours, noise
  planet/terrain_*.wesl   terrain passes
  planet/hydro_*.wesl     hydraulic erosion passes
  planet/moisture_spread.wesl, generation_finalize.wesl   environment maps + biomes
  planet/plants_*.wesl, fire_spread.wesl                  live plant ecosystem
  planet/animal_params.wesl, animals_*.wesl               live animal populations
  planet/planet_material.wesl   displaces + shades the planet from the height field
  planet/clouds.wesl, cloud_field.wesl                    cloud shell and cloud shadows
  planet/vegetation*.wesl       grows, sways and shades vegetation (+ its shadow pass)
  planet/terrain_prepass.wesl   terrain shadow pass
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

### Generate, then live

Every pass belongs to a phase. **Generate** runs `SimPhase::Generate` passes for
`SimControl::generation_steps` steps, many steps per frame, then the
`PassTrigger::AfterGeneration` passes once. **Live** then runs `SimPhase::Live` passes for
the rest of the session. Fields marked with `SimGraph::bake` are read back and written to
the disk cache when generation finishes; a start that finds a valid file uploads it into the
field textures and goes straight to Live. Bump `GENERATOR_VERSION` in `src/planet/mod.rs`
whenever generation changes, so old planets are not loaded.

The planet's generate phase is terrain + hydraulic erosion + moisture spreading. Its live
phase is the plant ecosystem. `PassTrigger::OnReset` passes run at step 0 of their own phase,
so live state is initialised whether the planet was generated or loaded.

### Environment maps

- `moisture` (0..1): spreads inland from oceans, lakes and rivers, decaying with distance
  and with every climb over high ground, so ranges cast rain shadows. It also scales
  rainfall during erosion, so wet coasts carve more than dry interiors.
- `temperature` (°C, `generation_finalize`): warm equator, cold poles, 6.5 °C/km lapse rate.
- `biome`: from temperature and precipitation (moisture, dried in the subtropical belt):
  ocean, lake, ice, tundra, desert, boreal and temperate forest, grassland, shrubland,
  savanna, tropical forest. The planet material colours land by biome, with beaches, bare
  rock on steep slopes and snow from temperature.

### Plant ecosystem

Live fields: `plants` (rgba: grass, shrub and tree cover 0..1, plus fuel, the dead litter
fires burn) and `fire` (intensity burning this step). One live step is roughly a month.

1. `plants_init` (step 0): a sparse, patchy first cover wherever each type can grow.
2. `plants_grow`: each type grows logistically toward a carrying capacity set by temperature
   and precipitation. Trees shade shrubs and grass, shrubs shade grass, so forests take over
   where it is wet enough. Bare land is colonised from neighbouring cover. Living cover drops
   litter, which decomposes faster when wet. Fires remove cover (grass most, trees least)
   and all fuel.
3. `fire_spread`: lightning ignites texels at random, and burning texels ignite neighbours,
   both in proportion to flammability (fuel, dryness, warmth). A burnt texel has no fuel, so
   fires leave firebreaks and burn out; in dry grassland and savanna they recur and keep
   the land open, in wet forest they hardly start.

Constants are in `PlantParams`. The planet material draws bare ground by biome, then grass,
shrubs and trees by cover, and fires as an emissive glow.

### Animal populations

Field `animals`: r = grazers, g = predators (population density; about 1 is what the best
pasture feeds), b and a = grass and shrub cover eaten this step. Populations live on
habitable land (not water or ice), which is also what bounds their migration. Per step
(`animals_step`, before `plants_grow`):

1. **Feeding**: grazers eat grass and some shrub cover with a saturating (Holling type II)
   response; predators kill grazers the same way.
2. **Breeding and death**: grazers grow with what they ate and die off at a base rate plus a
   crowding term; predators grow with their kills. Together this is Lotka-Volterra with
   logistic prey, so the two populations cycle and settle.
3. **Migration**: random wandering (diffusion) plus movement up the gradient of food:
   grazers towards plant cover and away from crowds, predators towards grazers.
4. **Fire** kills half the animals in a fully burning texel; a trickle of grazers moves
   into good empty pasture.

`plants_grow` then takes what was eaten off the plant cover, so heavy grazing keeps
grasslands short and herds move on. These are populations, not individuals; `planet/fauna.rs` draws individuals near the camera:

- Every 5 s the `animals` field is read back to the CPU. Each ~300 m cell within 1.2 km
  of the camera gets a herd (3 to 13 grazers) with a chance that grows with the local grazer
  density, and a pack (2 to 4 predators) likewise; deterministic per cell. Cells beyond
  1.7 km are dropped with their animals.
- Behaviour on the CPU: grazers stand and graze, walk off now and then, keep within ~18 m of
  their herd and bolt from predators within 70 m; predators roam as a pack, chase the
  nearest grazer within 160 m (a little slower than a fleeing grazer), and stop for a while
  to eat one they catch. Nobody walks into water.
- Low-poly bodies from Bevy primitives, sized with some variety, with legs that swing in
  diagonal pairs at a rate set by their speed.

### Level-of-detail terrain

While the planet generates, a GPU-displaced sphere previews the live height field. Once it
is baked, the baked fields are copied to the CPU and the planet is drawn as LOD terrain:

- **Surface** (`surface.rs`): the baked height (~20 km texels, sampled bicubically across
  cube edges) plus fractal detail from 40 km down to the mesh spacing: gentle in lowlands,
  rougher in uplands, and in high mountains nearly three times rougher with sharp ridged
  crests, so the broad baked highlands break up into peaks and valleys. Everything is f64.
- **Canyons** (`surface.rs`, `CanyonParams`): branching canyon networks cut into dry
  plateaus (250 m to 4 km up, low baked precipitation, in regions picked by large-scale
  noise). Main canyons are the warped zero lines of a 90 km noise, about 5 km wide and
  1.3 km deep; side canyons (24 km noise, 550 m deep) branch off near them. Walls are
  stepped into five cliff-and-bench terraces, and canyons never cut below sea level. They are
  part of the deterministic surface, from the baked maps, and only appear on mesh levels
  fine enough to resolve their walls.
- **Snow and rock**: snow follows the baked temperature corrected by the lapse rate to the
  actual height, so peaks above the snow line are white even inside a warmer 20 km texel;
  the snow line is ragged, and cliffs too steep to hold snow stay bare except on ice caps.
  Steep rock in warm, dry country is layered sandstone (red, ochre and cream bands a few tens
  of metres thick), grey rock elsewhere.
- **Quadtree** (`lod.rs`): each cube face is a quadtree of 32×32-quad chunks, up to level 19
  (~0.3 m between vertices). A chunk splits when the camera is nearer than 2.4 edge lengths;
  chunks wholly behind the horizon are skipped. Chunks are built on the async compute pool,
  nearest first, with positions relative to the chunk centre. A chunk is replaced by its
  children only once all four are built, so there are never holes; skirts cover cracks
  between levels.
- **Floating origin** (`camera.rs`): the camera's position is kept in f64, the render camera
  sits at the origin, and every chunk is placed at `centre − camera` each frame, so f32
  precision is never a problem, from 20 radii out to 1.8 m above the ground.
- **Shading**: the same field-driven colours as before; up close, noise breaks the 20 km
  plant cover into groves and clearings, and octaves smaller than a few pixels fade out.

### Atmosphere, ocean and clouds

- **Sky** (`planet/sky.rs`): Bevy's atmosphere (Hillaire 2020), an Earth-like scattering
  medium 100 km deep, centred on the planet and moved by the floating origin like everything
  else. Below 30 km it uses lookup textures; above, raymarching, which holds up from orbit.
  The aerial-perspective range grows with the distance to the horizon. The camera renders in
  HDR with physical units (100 klx sun, `Exposure::SUNLIGHT`), and the atmosphere also
  provides the environment light, so shadows are sky-blue and the ocean reflects the sky.
- **Ocean**: colour by depth (green shallows, deep blue offshore). Near the camera the sea
  surface moves with eight Gerstner waves (`planet/ocean.rs`): a 120 m swell along the
  prevailing wind (trade winds, westerlies, polar easterlies by latitude) and shorter waves
  spread around it, with deep-water dispersion. Their phases at the camera are computed in
  f64 each frame and the shader adds only camera-relative offsets, so 6 m waves stay exact
  6 400 km from the planet centre. Vertices move only where the mesh can carry a wave;
  further out the same waves tilt the normals, then noise ripples, then roughness. Seas get
  up to 3.7× higher under heavy cloud, waves flatten in the last few metres to the shore,
  crests pinch into streaky whitecaps in rough seas, and rolling bands of surf break in the
  shallows. Wave normals that would mirror the sky below the horizon are tilted back, so
  grazing views don't reflect the atmosphere's ground colour. Offshore, terrain detail is
  kept below the sea surface (no stray sand flats in open water).
- **Clouds**: a shell 6 km up with procedural cover (`cloud_field.wesl`): warped fractal noise
  shaped by latitude bands (wet equator, clear subtropics, stormy mid-latitudes) and the
  baked moisture map, drifting slowly. It is not a weather simulation. The terrain shader
  evaluates the same function where the ray to the sun crosses the cloud layer, which gives
  cloud shadows.

### Vegetation

`planet/vegetation.rs` draws plants as meshes in tiles of the terrain quadtree, one fixed
level per layer:

| layer | tile | spacing | drawn to |
|---|---|---|---|
| grass (clumps of 5 blades) | ~38 m | 0.45 m | 45 m |
| shrubs | ~150 m | 4 m | 600 m |
| trees, detailed (trunk + crown) | ~300 m | 11 m | 700 m |
| trees, far (one cone or blob) | ~1.2 km | 11 m | 3 km |

- **Placement (CPU)**: a jittered grid that is global to each cube face, so near and far tree
  tiles put every tree in the same place. Each plant gets a random cover threshold; plants
  whose biome can never reach that cover, those under water and trees on cliffs are skipped.
  Conifers win where it is cold. Tiles build on the async task pool like terrain chunks.
- **Growth (GPU)**: the vertex shader reads the live `plants` field and scales each plant in
  where its type's cover exceeds its threshold. The drawn density follows the simulation:
  burnt forest disappears and regrows as the ecosystem recovers. Trees switch from detailed
  to far meshes at 700 m, and every layer shrinks away towards the end of its range,
  where the terrain colour (which already shows plant cover) takes over.
- **Shading**: colour by type, temperature and dryness, darker inside canopies, wind sway that
  bends tall parts most, PBR lighting and cloud shadows. No shadow maps yet.

### Lighting

- **Sun**: 100 klx directional light with four shadow cascades, from 40 cm to 6 km (first
  cascade ends at 40 m, so animals and shrubs get crisp shadows; the last reaches hills).
  Sky light comes from the atmosphere's environment map, so shadows are sky-blue.
  `--no-shadows` turns shadow maps off for weak GPUs.
- **Casters**: terrain chunks and vegetation have their own shadow-pass vertex shaders
  (`terrain_prepass.wesl`, `vegetation_prepass.wesl`): Bevy's default prepass would not know
  the chunk layout, and would draw every plant at full size. Vegetation grows in by the
  same `plant_scale()` (`vegetation_common.wesl`) as in the main pass. Grass does not cast.
- **Receivers**: the custom terrain and vegetation shaders build their own `PbrInput`; they
  set the shadow-receiver flag that `pbr_input_new()` leaves clear.

### Hydraulic erosion

Fields: `water` (depth, m), `sediment` (suspended, m of rock), `flux` (outflow to the 4
neighbours, rgba), `discharge` (smoothed inflow, high along rivers). Per step:

1. `hydro_flux`: pipe model (Mei et al. 2007). Outflow grows with the water-surface
   difference to each neighbour and is scaled so a cell never sends more than it holds.
2. `hydro_water`: apply fluxes, move sediment with the same fractions as water, add rain,
   evaporate; oceans hold their surface at sea level and act as the sink.
3. `hydro_erode`: sediment capacity = `capacity × discharge × slope`. Below it the water
   digs (never below the lowest neighbour), above it drops sediment.

Material only moves between `height` and `sediment`, so their sum is conserved (drift
~0.001 m over 3000 steps from float rounding); `generation_finalize` settles the remaining
sediment back onto the terrain. Fluxes are stored in
each cell's own face frame, and `reverse_direction` in `common.wesl` finds the matching slot
across rotated cube faces, so rivers cross face edges seamlessly. Constants live in
`HydroParams` and are tuned for geological time at planet scale, not real-world units.

### Single queue

wgpu exposes one queue, so simulation and rendering share every frame's GPU time.
`SimControl::steps_per_frame` is the live time-slicing knob; generation instead uses
`generation_steps_per_frame` and gives up frame rate to finish quickly.

### Cube-sphere convention

Face `f` has axes `(N, U, V)` (see `FACE_AXES` in `src/planet/mesh.rs` and
`face_normal/face_u/face_v` in `common.wesl`; they must agree). Texel `(x, y)` maps to
`s = (x + 0.5) / res * 2 - 1`, then an equal-angle warp `a = tan(s · π/4)`, then
`dir = normalize(N + a·U + b·V)`. Texel areas vary by about 1.4× across a face.

## Licence

MIT
