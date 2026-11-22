// Params for the static environment maps derived during generation. Mirrors `EnvParams` in
// `src/planet/mod.rs`. There is no climate simulation: these maps are computed once and
// never change at runtime.

struct EnvParams {
    planet_radius_m: f32,
    moisture_length_m: f32,   // e-folding distance of moisture inland from its sources
    barrier_height_m: f32,    // climbing this much divides moisture by e (rain shadows)
    lake_depth_m: f32,        // standing water deeper than this is a lake
    river_discharge: f32,     // discharge at which a river becomes a moisture source
    equator_temp_c: f32,
    pole_temp_c: f32,
    lapse_rate_c_per_m: f32,
    subtropic_dryness: f32,   // 0..1: how much drier the ~27° latitude belt is
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// Biome classes written to the `biome` field (as f32). Keep in sync with `Biome` in Rust.
const BIOME_OCEAN: f32 = 0.0;
const BIOME_LAKE: f32 = 1.0;
const BIOME_ICE: f32 = 2.0;
const BIOME_TUNDRA: f32 = 3.0;
const BIOME_DESERT: f32 = 4.0;
const BIOME_BOREAL_FOREST: f32 = 5.0;
const BIOME_GRASSLAND: f32 = 6.0;
const BIOME_TEMPERATE_FOREST: f32 = 7.0;
const BIOME_SHRUBLAND: f32 = 8.0;
const BIOME_TROPICAL_FOREST: f32 = 9.0;
const BIOME_SAVANNA: f32 = 10.0;
const BIOME_COUNT: u32 = 11u;
