// Initial procedural heightfield, in metres above the sea-level datum.
import super::super::sim::common::{SimGlobals, texel_to_dir, gradient_noise};
import super::terrain_params::TerrainParams;

@group(0) @binding(0) var<uniform> globals: SimGlobals;
@group(0) @binding(1) var<uniform> params: TerrainParams;

// ---------------------------------------------------------------------------------------
// terrain_init: write-only `height`.

@group(0) @binding(2) var height_out: texture_storage_2d_array<r32float, write>;

fn fbm(p: vec3<f32>, octaves: u32, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    for (var i = 0u; i < octaves; i++) {
        sum += amp * gradient_noise(p * freq, seed + i * 101u);
        freq *= 2.03;
        amp *= 0.5;
    }
    return sum;
}

fn ridged(p: vec3<f32>, octaves: u32, seed: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = 1.0;
    var weight = 1.0;
    for (var i = 0u; i < octaves; i++) {
        var n = 1.0 - abs(gradient_noise(p * freq, seed + i * 211u));
        n = n * n * weight;
        weight = clamp(n * 2.0, 0.0, 1.0);
        sum += amp * n;
        freq *= 2.1;
        amp *= 0.5;
    }
    return sum;
}

@compute @workgroup_size(8, 8, 1)
fn terrain_init(@builtin(global_invocation_id) id: vec3<u32>) {
    let res = globals.resolution;
    if id.x >= res || id.y >= res {
        return;
    }
    let dir = texel_to_dir(id.z, vec2<f32>(id.xy) + 0.5, globals.inv_resolution);
    let seed = globals.seed;

    // Domain warp gives continents less blobby coastlines.
    let warp = vec3(
        gradient_noise(dir * 1.5, seed + 11u),
        gradient_noise(dir * 1.5, seed + 23u),
        gradient_noise(dir * 1.5, seed + 37u),
    ) * 0.35;
    let p = (dir + warp) * params.continent_frequency;

    let continents = fbm(p, 6u, seed);                       // ~[-0.6, 0.6]
    let land = smoothstep(-0.05, 0.25, continents);           // 0 ocean .. 1 inland
    let mountains = ridged(p * 3.0, 6u, seed + 1000u);        // ~[0, 1]
    let detail = fbm(p * 12.0, 4u, seed + 2000u);

    var h = mix(-params.ocean_depth_m, 0.0, smoothstep(-0.45, 0.0, continents));
    h += land * (200.0 + mountains * mountains * params.mountain_height_m);
    h += detail * mix(150.0, 600.0, land);

    textureStore(height_out, vec2<i32>(id.xy), i32(id.z), vec4(h, 0.0, 0.0, 0.0));
}

