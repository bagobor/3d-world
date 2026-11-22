// Moisture map, relaxed a little every generation step: oceans, lakes and rivers are sources,
// and moisture carries inland along the wettest path, decaying with distance and when it has
// to climb (so mountain ranges cast rain shadows). Converges like a distance transform: the
// value at a cell is the best exp(-path cost) to any source.
import super::super::sim::common::{SimGlobals, neighbor4, texel_distance};
import super::env_params::EnvParams;

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: EnvParams;
@group(0) @binding(2) var height: texture_2d_array<f32>;
@group(0) @binding(3) var water: texture_2d_array<f32>;
@group(0) @binding(4) var discharge: texture_2d_array<f32>;
@group(0) @binding(5) var moisture: texture_2d_array<f32>;
@group(0) @binding(6) var moisture_out: texture_storage_2d_array<r32float, write>;

fn load(tex: texture_2d_array<f32>, t: vec3<u32>) -> f32 {
    return textureLoad(tex, vec2<i32>(t.xy), i32(t.z), 0).r;
}

@compute @workgroup_size(8, 8, 1)
fn moisture_spread(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = globals.resolution;
    if id.x >= res || id.y >= res {
        return;
    }
    let h = load(height, id);
    var source = 0.0;
    if h < 0.0 {
        source = 1.0;
    } else if load(water, id) > params.lake_depth_m {
        source = 0.9;
    } else {
        source = 0.7 * smoothstep(params.river_discharge, params.river_discharge * 4.0, load(discharge, id));
    }

    var m = source;
    for (var k = 0u; k < 4u; k++) {
        let n = neighbor4(id, k, res);
        let climb = max(max(h, 0.0) - max(load(height, n), 0.0), 0.0);
        let dist = texel_distance(id, n, globals.inv_resolution, params.planet_radius_m);
        let carried = load(moisture, n)
            * exp(-dist / params.moisture_length_m - climb / params.barrier_height_m);
        m = max(m, carried);
    }
    textureStore(moisture_out, vec2<i32>(id.xy), i32(id.z), vec4(m, 0.0, 0.0, 0.0));
}
