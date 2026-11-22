// Params for the live plant ecosystem. Mirrors `PlantParams` in `src/planet/mod.rs`.
import super::super::sim::common::pcg;
//
// The `plants` field holds, per texel: r = grass cover, g = shrub cover, b = tree cover
// (each 0..1 of the ground), a = fuel (dead litter, 0..2). `fire` holds fire intensity
// (0..1) burning this step. One live step is about a month.

struct PlantParams {
    growth_grass: f32,     // logistic growth rates per step
    growth_shrub: f32,
    growth_tree: f32,
    spread: f32,           // colonisation from neighbouring cover
    seed_rain: f32,        // background establishment, keeps empty land from staying bare
    litter_rate: f32,      // fuel produced per unit of cover per step
    decay_rate: f32,       // fuel decomposed per step (faster when wet)
    ignition_rate: f32,    // lightning starts per texel per step at full flammability
    spread_chance: f32,    // chance one burning neighbour ignites a fully flammable texel
    burn_grass: f32,       // fraction of each cover a full-intensity fire removes
    burn_shrub: f32,
    burn_tree: f32,
    subtropic_dryness: f32, // same as EnvParams, for precipitation()
    lake_depth_m: f32,
    _pad0: f32,
    _pad1: f32,
}

const BIOME_OCEAN: f32 = 0.0;
const BIOME_LAKE: f32 = 1.0;
const BIOME_ICE: f32 = 2.0;

// Land that can carry plants: not ocean, lake or ice sheet.
fn plantable(biome: f32) -> bool {
    return biome > BIOME_ICE + 0.5;
}

// Carrying capacity of each plant type (grass, shrub, tree) for a climate, before shading.
fn capacity(temp_c: f32, precip: f32) -> vec3<f32> {
    let grass = smoothstep(-8.0, 2.0, temp_c) * smoothstep(0.03, 0.25, precip);
    let shrub = 0.8 * smoothstep(-4.0, 6.0, temp_c) * smoothstep(0.08, 0.3, precip);
    let tree = smoothstep(-2.0, 8.0, temp_c) * smoothstep(0.25, 0.55, precip);
    return vec3(grass, shrub, tree);
}

// Uniform random number in [0, 1) for a texel, step and stream.
fn random(texel: vec3<u32>, step: u32, seed: u32, stream: u32) -> f32 {
    let h = pcg(texel.x ^ pcg(texel.y ^ pcg((texel.z * 977u) ^ pcg(step ^ pcg((seed * 31u) + stream)))));
    return f32(h >> 8u) * (1.0 / 16777216.0);
}

