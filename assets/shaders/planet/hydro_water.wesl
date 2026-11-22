// Pipe model step 2: move water and suspended sediment along the fluxes, add rain, evaporate,
// and track a smoothed discharge (the flow rate rivers are made of).
//
// Sediment moves with the water: a cell sends the same fraction of its sediment as of its water
// along each pipe, so sediment is conserved exactly (oceans included).
import super::super::sim::common::{SimGlobals, neighbor4, reverse_direction, vec4_at};
import super::hydro_params::HydroParams;

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: HydroParams;
@group(0) @binding(2) var height: texture_2d_array<f32>;
@group(0) @binding(3) var water: texture_2d_array<f32>;
@group(0) @binding(4) var flux: texture_2d_array<f32>;
@group(0) @binding(5) var sediment: texture_2d_array<f32>;
@group(0) @binding(6) var discharge: texture_2d_array<f32>;
@group(0) @binding(7) var water_out: texture_storage_2d_array<r32float, write>;
@group(0) @binding(8) var sediment_out: texture_storage_2d_array<r32float, write>;
@group(0) @binding(9) var discharge_out: texture_storage_2d_array<r32float, write>;
@group(0) @binding(10) var moisture: texture_2d_array<f32>;

fn load(tex: texture_2d_array<f32>, t: vec3<u32>) -> f32 {
    return textureLoad(tex, vec2<i32>(t.xy), i32(t.z), 0).r;
}

// Fraction of a cell's contents that leaves through each pipe.
fn out_fraction(f: f32, w: f32) -> f32 {
    return select(0.0, f / w, w > 1e-9);
}

@compute @workgroup_size(8, 8, 1)
fn hydro_water(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = globals.resolution;
    if id.x >= res || id.y >= res {
        return;
    }
    let p = vec2<i32>(id.xy);
    let layer = i32(id.z);
    let h = load(height, id);
    let w = load(water, id);
    let s = load(sediment, id);
    let f = textureLoad(flux, p, layer, 0);
    let outflow = f.x + f.y + f.z + f.w;

    var inflow = 0.0;
    var s_in = 0.0;
    for (var k = 0u; k < 4u; k++) {
        let n = neighbor4(id, k, res);
        let f_in = vec4_at(textureLoad(flux, vec2<i32>(n.xy), i32(n.z), 0), reverse_direction(id, k, res));
        inflow += f_in;
        s_in += load(sediment, n) * out_fraction(f_in, load(water, n));
    }

    let s_out = s * min(out_fraction(outflow, w), 1.0);
    let s_new = s - s_out + s_in;

    var w_new = w - outflow + inflow;
    if h < 0.0 {
        // Ocean: the sea surface stays at the datum (an infinite reservoir and sink).
        w_new = -h;
    } else {
        // Wet regions get more rain, so deserts erode less than rainforests.
        let rain = params.rain_m * (0.2 + 1.6 * load(moisture, id));
        w_new = (w_new + rain) * (1.0 - params.evaporation);
    }

    let q = mix(load(discharge, id), inflow, params.discharge_smoothing);

    textureStore(water_out, p, layer, vec4(max(w_new, 0.0), 0.0, 0.0, 0.0));
    textureStore(sediment_out, p, layer, vec4(max(s_new, 0.0), 0.0, 0.0, 0.0));
    textureStore(discharge_out, p, layer, vec4(q, 0.0, 0.0, 0.0));
}
