// Live phase, once: a few grazers and fewer predators wherever there is land to live on.
import super::super::sim::common::{SimGlobals, texel_to_dir, gradient_noise};
import super::animal_params::{AnimalParams, habitable};

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: AnimalParams;
@group(0) @binding(2) var biome: texture_2d_array<f32>;
@group(0) @binding(3) var animals_out: texture_storage_2d_array<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn animals_init(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= globals.resolution || id.y >= globals.resolution {
        return;
    }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    var out = vec4(0.0);
    if habitable(textureLoad(biome, p, layer, 0).r) {
        let dir = texel_to_dir(id.z, vec2<f32>(id.xy) + 0.5, globals.inv_resolution);
        let patchy = max(gradient_noise(dir * 9.0, globals.seed + 211u), 0.0);
        out = vec4(0.1 * patchy, 0.01 * patchy, 0.0, 0.0);
    }
    textureStore(animals_out, p, layer, out);
}
