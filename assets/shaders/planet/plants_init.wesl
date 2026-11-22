// Live phase, once: a sparse first cover of every plant type wherever it can grow, no fuel,
// no fires. The ecosystem then grows into its own equilibrium.
import super::super::sim::common::{SimGlobals, texel_to_dir, gradient_noise};
import super::env_params::precipitation;
import super::plant_params::{PlantParams, plantable, capacity};

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: PlantParams;
@group(0) @binding(2) var temperature: texture_2d_array<f32>;
@group(0) @binding(3) var moisture: texture_2d_array<f32>;
@group(0) @binding(4) var biome: texture_2d_array<f32>;
@group(0) @binding(5) var plants_out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(6) var fire_out: texture_storage_2d_array<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn plants_init(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= globals.resolution || id.y >= globals.resolution {
        return;
    }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    var cover = vec3(0.0);
    if plantable(textureLoad(biome, p, layer, 0).r) {
        let dir = texel_to_dir(id.z, vec2<f32>(id.xy) + 0.5, globals.inv_resolution);
        let precip = precipitation(textureLoad(moisture, p, layer, 0).r, dir, params.subtropic_dryness);
        let k = capacity(textureLoad(temperature, p, layer, 0).r, precip);
        // Patchy: some regions start greener than others.
        let patchy = 0.5 + 0.5 * gradient_noise(dir * 12.0, globals.seed + 101u);
        cover = k * vec3(0.3, 0.1, 0.05) * patchy;
    }
    textureStore(plants_out, p, layer, vec4(cover, 0.0));
    textureStore(fire_out, p, layer, vec4(0.0));
}
