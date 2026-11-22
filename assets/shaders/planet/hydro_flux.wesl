// Pipe model (Mei et al. 2007), step 1: outflow from each cell to its 4 neighbours, driven by
// the water-surface height difference and scaled so a cell never sends more water than it has.
import super::super::sim::common::{SimGlobals, neighbor4};
import super::hydro_params::HydroParams;

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: HydroParams;
@group(0) @binding(2) var height: texture_2d_array<f32>;
@group(0) @binding(3) var water: texture_2d_array<f32>;
@group(0) @binding(4) var flux: texture_2d_array<f32>;
@group(0) @binding(5) var flux_out: texture_storage_2d_array<rgba32float, write>;

fn surface(t: vec3<u32>) -> f32 {
    let p = vec2<i32>(t.xy);
    return textureLoad(height, p, i32(t.z), 0).r + textureLoad(water, p, i32(t.z), 0).r;
}

@compute @workgroup_size(8, 8, 1)
fn hydro_flux(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = globals.resolution;
    if id.x >= res || id.y >= res {
        return;
    }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    let w = textureLoad(water, p, layer, 0).r;
    let s = surface(id);
    let old = textureLoad(flux, p, layer, 0);

    var f = vec4(0.0);
    for (var k = 0u; k < 4u; k++) {
        let dh = s - surface(neighbor4(id, k, res));
        f[k] = max(0.0, params.flux_damping * old[k] + params.pipe_gain * dh);
    }
    let total = f.x + f.y + f.z + f.w;
    if total > w {
        f *= w / max(total, 1e-12);
    }
    textureStore(flux_out, p, layer, f);
}
