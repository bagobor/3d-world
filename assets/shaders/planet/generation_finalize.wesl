// Runs once after the last generation step: settles leftover sediment into the terrain and
// derives the static temperature and biome maps the live simulation and renderer use.
import super::super::sim::common::{SimGlobals, texel_to_dir};
import super::env_params::{
    EnvParams, precipitation, BIOME_OCEAN, BIOME_LAKE, BIOME_ICE, BIOME_TUNDRA, BIOME_DESERT,
    BIOME_BOREAL_FOREST, BIOME_GRASSLAND, BIOME_TEMPERATE_FOREST, BIOME_SHRUBLAND,
    BIOME_TROPICAL_FOREST, BIOME_SAVANNA,
};

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: EnvParams;
@group(0) @binding(2) var height: texture_2d_array<f32>;
@group(0) @binding(3) var sediment: texture_2d_array<f32>;
@group(0) @binding(4) var water: texture_2d_array<f32>;
@group(0) @binding(5) var moisture: texture_2d_array<f32>;
@group(0) @binding(6) var height_out: texture_storage_2d_array<r32float, write>;
@group(0) @binding(7) var temperature_out: texture_storage_2d_array<r32float, write>;
@group(0) @binding(8) var biome_out: texture_storage_2d_array<r32float, write>;

fn classify(h: f32, w: f32, t: f32, p: f32) -> f32 {
    if h < 0.0 { return BIOME_OCEAN; }
    if w > params.lake_depth_m { return BIOME_LAKE; }
    if t < -8.0 { return BIOME_ICE; }
    if t < 2.0 { return BIOME_TUNDRA; }
    if p < 0.12 { return BIOME_DESERT; }
    if t < 10.0 {
        return select(BIOME_GRASSLAND, BIOME_BOREAL_FOREST, p >= 0.35);
    }
    if t < 20.0 {
        if p >= 0.45 { return BIOME_TEMPERATE_FOREST; }
        return select(BIOME_SHRUBLAND, BIOME_GRASSLAND, p >= 0.22);
    }
    if p >= 0.6 { return BIOME_TROPICAL_FOREST; }
    return select(BIOME_DESERT, BIOME_SAVANNA, p >= 0.25);
}

@compute @workgroup_size(8, 8, 1)
fn generation_finalize(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= globals.resolution || id.y >= globals.resolution {
        return;
    }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    // Whatever the water still carries settles where it is.
    let h = textureLoad(height, p, layer, 0).r + textureLoad(sediment, p, layer, 0).r;
    let w = textureLoad(water, p, layer, 0).r;

    let dir = texel_to_dir(id.z, vec2<f32>(id.xy) + 0.5, globals.inv_resolution);
    let lat = asin(clamp(dir.y, -1.0, 1.0));
    let sea_temp = mix(params.pole_temp_c, params.equator_temp_c, cos(lat));
    let temp = sea_temp - params.lapse_rate_c_per_m * max(h, 0.0);

    let precip = precipitation(textureLoad(moisture, p, layer, 0).r, dir, params.subtropic_dryness);

    textureStore(height_out, p, layer, vec4(h, 0.0, 0.0, 0.0));
    textureStore(temperature_out, p, layer, vec4(temp, 0.0, 0.0, 0.0));
    textureStore(biome_out, p, layer, vec4(classify(h, w, temp, precip), 0.0, 0.0, 0.0));
}
