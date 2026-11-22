// Resets the water system: oceans filled to sea level, land dry, no flux or sediment.
import super::super::sim::common::SimGlobals;
import super::hydro_params::HydroParams;

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: HydroParams;
@group(0) @binding(2) var height: texture_2d_array<f32>;
@group(0) @binding(3) var water_out: texture_storage_2d_array<r32float, write>;
@group(0) @binding(4) var sediment_out: texture_storage_2d_array<r32float, write>;
@group(0) @binding(5) var flux_out: texture_storage_2d_array<rgba32float, write>;
@group(0) @binding(6) var discharge_out: texture_storage_2d_array<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn hydro_init(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= globals.resolution || id.y >= globals.resolution {
        return;
    }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    let h = textureLoad(height, p, layer, 0).r;
    textureStore(water_out, p, layer, vec4(max(-h, 0.0), 0.0, 0.0, 0.0));
    textureStore(sediment_out, p, layer, vec4(0.0));
    textureStore(flux_out, p, layer, vec4(0.0));
    textureStore(discharge_out, p, layer, vec4(0.0));
}
