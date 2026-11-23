// Procedural cloud cover (0..1) for a direction on the planet. There is no weather
// simulation: coverage follows latitude bands (wet equator, clear subtropics, stormy
// mid-latitudes) and the baked moisture map, broken up by warped fractal noise, and drifts
// slowly eastwards. Shared by the cloud shell and by cloud shadows on the ground.
import super::super::sim::common::gradient_noise;

fn cloud_fbm(p: vec3<f32>, octaves: u32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0u; i < octaves; i++) {
        sum += amp * gradient_noise(q, 1700u + i * 13u);
        q *= 2.1;
        amp *= 0.5;
    }
    return sum;
}

// Raw cloud density before it is mapped to coverage: above ~0.32 the sky is overcast, and
// the higher it goes the thicker the cloud (rain, then storms). Mirrored on the CPU by
// `cloud_density` in src/planet/weather.rs.
fn cloud_density(dir_in: vec3<f32>, moisture: f32, time: f32) -> f32 {
    // Drift: rotate the pattern slowly about the planet's axis.
    let a = time * 0.0004;
    let dir = vec3(dir_in.x * cos(a) - dir_in.z * sin(a), dir_in.y, dir_in.x * sin(a) + dir_in.z * cos(a));
    let lat = abs(asin(clamp(dir.y, -1.0, 1.0))) * 57.29578;
    let band = 0.25 * exp(-pow(lat / 9.0, 2.0))
        - 0.22 * exp(-pow((lat - 25.0) / 9.0, 2.0))
        + 0.12 * exp(-pow((lat - 55.0) / 12.0, 2.0));
    // Warp the domain so clouds form swirls and streaks rather than blobs.
    let p = dir * 5.0;
    let warp = vec3(cloud_fbm(p + 11.0, 3), cloud_fbm(p + 23.0, 3), cloud_fbm(p + 37.0, 3));
    let n = cloud_fbm(p * 1.6 + warp * 1.4, 7);
    return n + band + (moisture - 0.6) * 0.35;
}

fn coverage_from_density(v: f32) -> f32 {
    return smoothstep(0.02, 0.32, v);
}

// Rain falls from the thickest cloud, storms from the thickest of that.
fn rain_from_density(v: f32) -> f32 {
    return smoothstep(0.22, 0.38, v);
}

fn storm_from_density(v: f32) -> f32 {
    return smoothstep(0.38, 0.55, v);
}

fn cloud_coverage(dir_in: vec3<f32>, moisture: f32, time: f32) -> f32 {
    return coverage_from_density(cloud_density(dir_in, moisture, time));
}
