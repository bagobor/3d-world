// Thermal-erosion relaxation of the heightfield.
import super::super::sim::common::{SimGlobals, texel_to_dir, neighbor_texel};
import super::terrain_params::TerrainParams;

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: TerrainParams;

// ---------------------------------------------------------------------------------------
// terrain_relax: reads `height`, writes `height` (double-buffered by the framework).
// Thermal erosion: wherever the slope to a neighbour exceeds the talus slope, material moves
// downhill. The exchange is symmetric between neighbour pairs, so mass is conserved up to
// the small spacing differences of the cube-sphere grid.

@group(0) @binding(2) var height_in: texture_2d_array<f32>;
@group(0) @binding(3) var height_next: texture_storage_2d_array<r32float, write>;

fn load_height(t: vec3<u32>) -> f32 {
    return textureLoad(height_in, vec2<i32>(t.xy), i32(t.z), 0).r;
}

@compute @workgroup_size(8, 8, 1)
fn terrain_relax(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = globals.resolution;
    if id.x >= res || id.y >= res {
        return;
    }
    let texel = id;
    let h = load_height(texel);
    let dir = texel_to_dir(texel.z, vec2<f32>(texel.xy) + 0.5, globals.inv_resolution);

    let offsets = array(vec2(1, 0), vec2(-1, 0), vec2(0, 1), vec2(0, -1));
    var delta = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let n = neighbor_texel(texel, offsets[i], res);
        let hn = load_height(n);
        let ndir = texel_to_dir(n.z, vec2<f32>(n.xy) + 0.5, globals.inv_resolution);
        let dist = params.planet_radius_m * length(dir - ndir);
        let excess = abs(hn - h) - params.talus_slope * dist;
        if excess > 0.0 {
            delta += sign(hn - h) * excess;
        }
    }
    // erosion_rate <= 0.25 keeps the explicit update stable with 4 neighbours.
    let h_new = h + params.erosion_rate * 0.5 * delta;
    textureStore(height_next, vec2<i32>(texel.xy), i32(texel.z), vec4(h_new, 0.0, 0.0, 0.0));
}
