// Live phase, once: bake the clouds' warped noise into the `cloud_noise` field (see
// cloud_field.wesl). It doesn't depend on the seed or on time, so it never needs redoing.
import super::super::sim::common::{SimGlobals, texel_to_dir};
import super::cloud_field::cloud_noise;

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(2) var noise_out: texture_storage_2d_array<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cloud_bake(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= globals.resolution || id.y >= globals.resolution {
        return;
    }
    let dir = texel_to_dir(id.z, vec2<f32>(id.xy) + 0.5, globals.inv_resolution);
    textureStore(noise_out, vec2<i32>(id.xy), i32(id.z), cloud_noise(dir));
}
