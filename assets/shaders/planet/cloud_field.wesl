// Procedural cloud cover (0..1) for a direction on the planet. There is no weather
// simulation: coverage follows latitude bands (wet equator, clear subtropics, stormy
// mid-latitudes) and the baked moisture map, broken up by warped fractal noise, and drifts
// slowly eastwards. Shared by the cloud shell and by cloud shadows on the ground.
//
// The warped noise is the expensive part (dozens of gradient-noise lookups), and it never
// changes in the clouds' own drifting frame. So the `cloud_bake` pass evaluates it once into
// the `cloud_noise` field, indexed by drifting-frame direction: the domain warp in xyz and
// the coarse octaves in w. Renderers rotate their direction into that frame, read the field,
// and add the cheap parts (bands, moisture). The cloud shell also adds the fine octaves,
// which the field's resolution can't hold; cloud shadows on the ground skip them.
import super::super::sim::common::{gradient_noise, dir_to_face, dir_to_face_st, neighbor_texel};

// Octaves of the main noise, and how many of them the field holds.
const CLOUD_OCTAVES: u32 = 7u;
const CLOUD_BAKED_OCTAVES: u32 = 3u;

// Octaves [first, last) of fractal noise at p.
fn cloud_fbm(p: vec3<f32>, first: u32, last: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0u; i < first; i++) {
        q *= 2.1;
        amp *= 0.5;
    }
    for (var i = first; i < last; i++) {
        sum += amp * gradient_noise(q, 1700u + i * 13u);
        q *= 2.1;
        amp *= 0.5;
    }
    return sum;
}

// A planet direction in the clouds' frame, which rotates slowly about the planet's axis.
fn cloud_drift(dir_in: vec3<f32>, time: f32) -> vec3<f32> {
    let a = time * 0.0004;
    return vec3(dir_in.x * cos(a) - dir_in.z * sin(a), dir_in.y, dir_in.x * sin(a) + dir_in.z * cos(a));
}

// What `cloud_bake` stores for a drifting-frame direction: domain warp (xyz) so clouds form
// swirls and streaks rather than blobs, and the coarse octaves of the warped noise (w).
fn cloud_noise(dir: vec3<f32>) -> vec4<f32> {
    let p = dir * 5.0;
    let warp = vec3(cloud_fbm(p + 11.0, 0u, 3u), cloud_fbm(p + 23.0, 0u, 3u), cloud_fbm(p + 37.0, 0u, 3u));
    return vec4(warp, cloud_fbm(p * 1.6 + warp * 1.4, 0u, CLOUD_BAKED_OCTAVES));
}

// Bilinear `cloud_noise` field lookup that stays continuous across cube edges.
fn sample_cloud_noise(tex: texture_2d_array<f32>, dir: vec3<f32>, resolution: f32) -> vec4<f32> {
    let res = u32(resolution);
    let face = dir_to_face(dir);
    let st = dir_to_face_st(face, dir);
    let c = (st * 0.5 + 0.5) * resolution - 0.5;
    let base = floor(c);
    let f = c - base;
    let b = vec2<i32>(base);
    let origin = vec3<u32>(vec2<u32>(clamp(b, vec2(0), vec2(i32(res) - 1))), face);
    let d = b - vec2<i32>(origin.xy);
    let t00 = neighbor_texel(origin, d, res);
    let t10 = neighbor_texel(origin, d + vec2(1, 0), res);
    let t01 = neighbor_texel(origin, d + vec2(0, 1), res);
    let t11 = neighbor_texel(origin, d + vec2(1, 1), res);
    let n00 = textureLoad(tex, vec2<i32>(t00.xy), i32(t00.z), 0);
    let n10 = textureLoad(tex, vec2<i32>(t10.xy), i32(t10.z), 0);
    let n01 = textureLoad(tex, vec2<i32>(t01.xy), i32(t01.z), 0);
    let n11 = textureLoad(tex, vec2<i32>(t11.xy), i32(t11.z), 0);
    return mix(mix(n00, n10, f.x), mix(n01, n11, f.x), f.y);
}

// Cloud cover at planet direction `dir_in`, with `moisture` sampled there. `detail` adds the
// fine octaves (for the cloud shell itself; shadows don't need them).
fn cloud_coverage(
    noise_tex: texture_2d_array<f32>,
    resolution: f32,
    dir_in: vec3<f32>,
    moisture: f32,
    time: f32,
    detail: bool,
) -> f32 {
    let dir = cloud_drift(dir_in, time);
    let noise = sample_cloud_noise(noise_tex, dir, resolution);
    var n = noise.w;
    if detail {
        n += cloud_fbm(dir * 5.0 * 1.6 + noise.xyz * 1.4, CLOUD_BAKED_OCTAVES, CLOUD_OCTAVES);
    }
    let lat = abs(asin(clamp(dir.y, -1.0, 1.0))) * 57.29578;
    let band = 0.25 * exp(-pow(lat / 9.0, 2.0))
        - 0.22 * exp(-pow((lat - 25.0) / 9.0, 2.0))
        + 0.12 * exp(-pow((lat - 55.0) / 12.0, 2.0));
    let v = n + band + (moisture - 0.6) * 0.35;
    return smoothstep(0.02, 0.32, v);
}
