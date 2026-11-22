// Erosion and deposition: flowing water carries sediment up to a capacity that grows with
// discharge and slope. Below capacity it digs into the terrain, above it drops sediment.
// Material moves only between `height` and `sediment`, so height + sediment is conserved.
import super::super::sim::common::{SimGlobals, neighbor4, texel_distance};
import super::hydro_params::HydroParams;

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: HydroParams;
@group(0) @binding(2) var height: texture_2d_array<f32>;
@group(0) @binding(3) var sediment: texture_2d_array<f32>;
@group(0) @binding(4) var discharge: texture_2d_array<f32>;
@group(0) @binding(5) var height_out: texture_storage_2d_array<r32float, write>;
@group(0) @binding(6) var sediment_out: texture_storage_2d_array<r32float, write>;

fn load(tex: texture_2d_array<f32>, t: vec3<u32>) -> f32 {
    return textureLoad(tex, vec2<i32>(t.xy), i32(t.z), 0).r;
}

@compute @workgroup_size(8, 8, 1)
fn hydro_erode(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = globals.resolution;
    if id.x >= res || id.y >= res {
        return;
    }
    let h = load(height, id);
    let s = load(sediment, id);
    let q = load(discharge, id);

    // Steepest downhill slope (rise over run).
    var slope = 0.0;
    var lowest = h;
    for (var k = 0u; k < 4u; k++) {
        let n = neighbor4(id, k, res);
        let hn = load(height, n);
        lowest = min(lowest, hn);
        slope = max(slope, (h - hn) / texel_distance(id, n, globals.inv_resolution, params.planet_radius_m));
    }

    // Under the sea, water doesn't flow: everything settles.
    let capacity = select(params.capacity * q * max(slope, params.min_slope), 0.0, h < 0.0);

    var dh = 0.0;
    if s < capacity {
        // Never dig below the lowest neighbour: that would only make pits.
        let room = max(h - lowest, 0.0);
        dh = -min(min(params.erosion_rate * (capacity - s), params.max_erosion_m), room);
    } else {
        dh = params.deposition_rate * (s - capacity);
    }

    let p = vec2<i32>(id.xy);
    textureStore(height_out, p, i32(id.z), vec4(h + dh, 0.0, 0.0, 0.0));
    textureStore(sediment_out, p, i32(id.z), vec4(s - dh, 0.0, 0.0, 0.0));
}
