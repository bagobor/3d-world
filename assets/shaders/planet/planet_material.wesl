// Planet shading, shared by two materials:
// - `vertex`/`fragment` (PlanetMaterial): the generation preview. The vertex stage displaces
//   a unit cube-sphere by the live `height` field; normals come from the same field.
// - `terrain_vertex`/`terrain_fragment` (TerrainMaterial): LOD terrain chunks built on the
//   CPU (see src/planet/lod.rs), with chunk-local positions, normals including the
//   procedural detail, and per-vertex direction + height.
// Both colour the surface from the simulation's fields and run Bevy's PBR lighting.

import bevy_pbr::render::{
    mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_normal_local_to_world},
    view_transformations::position_world_to_clip,
    mesh_view_bindings::{view, globals, lights},
    pbr_types::{PbrInput, pbr_input_new},
    pbr_functions as fns,
};
import bevy_core_pipeline::tonemapping::tone_mapping;
import bevy_pbr::render::mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT;
import super::super::sim::common::{dir_to_face, dir_to_face_st, neighbor_texel, gradient_noise};
import super::cloud_field::{cloud_coverage, cloud_density, storm_from_density};
import super::haze::with_haze;

struct PlanetMaterialParams {
    planet_radius_m: f32,
    height_exaggeration: f32,
    resolution: f32,
    river_discharge: f32,
    // 1 once the static maps exist (after generation or a cache load).
    show_biomes: f32,
    // Gerstner waves near the camera (src/planet/ocean.rs): wave vector (rad/m) and
    // amplitude (m), and each wave's phase at the camera.
    waves: array<vec4<f32>, 8>,
    wave_phases: array<vec4<f32>, 2>,
}

const WAVE_COUNT: u32 = 8u;
// Horizontal sharpness of the wave crests (0 = sine waves, 1 = the steepest Gerstner).
const WAVE_SHARPNESS: f32 = 0.8;

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var height_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var<uniform> material: PlanetMaterialParams;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var water_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(3) var discharge_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(4) var temperature_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(5) var biome_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(6) var plants_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(7) var fire_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(8) var moisture_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(9) var cloud_noise_tex: texture_2d_array<f32>;

fn load_field(field: u32, t: vec3<u32>) -> f32 {
    let p = vec2<i32>(t.xy);
    switch field {
        case 1u: { return textureLoad(water_tex, p, i32(t.z), 0).r; }
        case 2u: { return textureLoad(discharge_tex, p, i32(t.z), 0).r; }
        case 3u: { return textureLoad(temperature_tex, p, i32(t.z), 0).r; }
        case 4u: { return textureLoad(biome_tex, p, i32(t.z), 0).r; }
        case 5u: { return textureLoad(fire_tex, p, i32(t.z), 0).r; }
        case 6u: { return textureLoad(moisture_tex, p, i32(t.z), 0).r; }
        default: { return textureLoad(height_tex, p, i32(t.z), 0).r; }
    }
}

const HEIGHT: u32 = 0u;
const WATER: u32 = 1u;
const DISCHARGE: u32 = 2u;
const TEMPERATURE: u32 = 3u;
const BIOME: u32 = 4u;
const FIRE: u32 = 5u;
const MOISTURE: u32 = 6u;
const CLOUD_HEIGHT_M: f32 = 6000.0;

fn load_plants(t: vec3<u32>) -> vec4<f32> {
    return textureLoad(plants_tex, vec2<i32>(t.xy), i32(t.z), 0);
}

// Bilinear plant cover (grass, shrub, tree, fuel), continuous across cube edges.
fn sample_plants(dir: vec3<f32>) -> vec4<f32> {
    let res = u32(material.resolution);
    let face = dir_to_face(dir);
    let st = dir_to_face_st(face, dir);
    let c = (st * 0.5 + 0.5) * material.resolution - 0.5;
    let base = floor(c);
    let f = c - base;
    let b = vec2<i32>(base);
    let origin = vec3<u32>(vec2<u32>(clamp(b, vec2(0), vec2(i32(res) - 1))), face);
    let d = b - vec2<i32>(origin.xy);
    let p00 = load_plants(neighbor_texel(origin, d, res));
    let p10 = load_plants(neighbor_texel(origin, d + vec2(1, 0), res));
    let p01 = load_plants(neighbor_texel(origin, d + vec2(0, 1), res));
    let p11 = load_plants(neighbor_texel(origin, d + vec2(1, 1), res));
    return mix(mix(p00, p10, f.x), mix(p01, p11, f.x), f.y);
}

// Bare ground albedo per biome class (see `env_params.wesl`); plants are drawn over it.
fn biome_albedo(biome: f32) -> vec3<f32> {
    switch u32(biome + 0.5) {
        case 0u: { return vec3(0.02, 0.08, 0.15); }  // ocean (not used on land)
        case 1u: { return vec3(0.04, 0.16, 0.24); }  // lake
        case 2u: { return vec3(0.90, 0.92, 0.96); }  // ice
        case 3u: { return vec3(0.40, 0.38, 0.32); }  // tundra: stony
        case 4u: { return vec3(0.80, 0.66, 0.44); }  // desert: sand
        case 9u, 10u: { return vec3(0.55, 0.34, 0.22); }  // tropical: red soil
        default: { return vec3(0.42, 0.34, 0.24); }  // brown soil
    }
}

// Biome colour, blended bilinearly between the 4 nearest texels' classes.
fn sample_biome_albedo(dir: vec3<f32>) -> vec3<f32> {
    let res = u32(material.resolution);
    let face = dir_to_face(dir);
    let st = dir_to_face_st(face, dir);
    let c = (st * 0.5 + 0.5) * material.resolution - 0.5;
    let base = floor(c);
    let f = c - base;
    let b = vec2<i32>(base);
    let origin = vec3<u32>(vec2<u32>(clamp(b, vec2(0), vec2(i32(res) - 1))), face);
    let d = b - vec2<i32>(origin.xy);
    let c00 = biome_albedo(load_field(BIOME, neighbor_texel(origin, d, res)));
    let c10 = biome_albedo(load_field(BIOME, neighbor_texel(origin, d + vec2(1, 0), res)));
    let c01 = biome_albedo(load_field(BIOME, neighbor_texel(origin, d + vec2(0, 1), res)));
    let c11 = biome_albedo(load_field(BIOME, neighbor_texel(origin, d + vec2(1, 1), res)));
    return mix(mix(c00, c10, f.x), mix(c01, c11, f.x), f.y);
}

fn sample_height(dir: vec3<f32>) -> f32 {
    return sample_field(HEIGHT, dir);
}

// Bilinear field lookup that stays continuous across cube edges.
fn sample_field(field: u32, dir: vec3<f32>) -> f32 {
    let res = u32(material.resolution);
    let face = dir_to_face(dir);
    let st = dir_to_face_st(face, dir);
    let c = (st * 0.5 + 0.5) * material.resolution - 0.5;
    let base = floor(c);
    let f = c - base;
    let b = vec2<i32>(base);
    // `neighbor_texel` wraps any out-of-face coordinate onto the right face.
    let origin = vec3<u32>(vec2<u32>(clamp(b, vec2(0), vec2(i32(res) - 1))), face);
    let d = b - vec2<i32>(origin.xy);
    let h00 = load_field(field, neighbor_texel(origin, d, res));
    let h10 = load_field(field, neighbor_texel(origin, d + vec2(1, 0), res));
    let h01 = load_field(field, neighbor_texel(origin, d + vec2(0, 1), res));
    let h11 = load_field(field, neighbor_texel(origin, d + vec2(1, 1), res));
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// Surface point (unit-radius model space) for a direction; oceans are flat at sea level.
fn surface(dir: vec3<f32>) -> vec3<f32> {
    let h = max(sample_height(dir), 0.0);
    return dir * (1.0 + h / material.planet_radius_m * material.height_exaggeration);
}

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) local_dir: vec3<f32>,
    @location(2) @interpolate(flat) instance_index: u32,
};

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    let dir = normalize(in.position);
    let world_from_local = get_world_from_local(in.instance_index);
    let world = mesh_position_local_to_world(world_from_local, vec4(surface(dir), 1.0));
    var out: VertexOutput;
    out.clip_position = position_world_to_clip(world.xyz);
    out.world_position = world;
    out.local_dir = dir;
    out.instance_index = in.instance_index;
    return out;
}

// Fractal noise in metres for surface detail below the fields' resolution. Octaves smaller
// than a few pixels fade out, so distant views don't shimmer.
fn detail_noise(dir: vec3<f32>, pixel_m: f32, seed: u32) -> f32 {
    let p = dir * material.planet_radius_m;
    var sum = 0.0;
    // Nearly equal amplitude per octave, so there is variation at every distance.
    var amp = 0.3;
    var wavelength = 6000.0;
    for (var i = 0u; i < 9u; i++) {
        let fade = smoothstep(2.0 * pixel_m, 8.0 * pixel_m, wavelength);
        sum += amp * fade * gradient_noise(p / wavelength, seed + i * 31u);
        wavelength *= 0.4;
        amp *= 0.8;
    }
    return sum;
}

// Land colour: bare ground by biome, then grass, shrubs and trees by their live cover.
// Up close, noise breaks the smooth cover fields into groves and clearings.
fn biome_color(dir: vec3<f32>, h: f32, slope: f32, pixel_m: f32) -> vec3<f32> {
    var c = sample_biome_albedo(dir);
    var plants = sample_plants(dir);
    let n_tree = detail_noise(dir, pixel_m, 7u);
    let n_ground = detail_noise(dir, pixel_m, 91u);
    plants.r = clamp(plants.r + n_ground * 1.2, 0.0, 1.0);
    plants.g = clamp(plants.g + (n_ground - n_tree) * 0.8, 0.0, 1.0);
    plants.b = clamp(plants.b + n_tree * 1.4, 0.0, 1.0);

    let temp = sample_field(TEMPERATURE, dir);
    // Cold forests are dark conifers; warm ones a brighter, more saturated green.
    let warm = smoothstep(5.0, 22.0, temp);
    // Grass varies between lush and dry patches.
    let lush = mix(vec3(0.33, 0.40, 0.17), vec3(0.27, 0.44, 0.11), warm);
    let grass = mix(lush, vec3(0.45, 0.42, 0.24), clamp(0.5 + n_tree * 2.5, 0.0, 1.0) * 0.8);
    let shrub = vec3(0.26, 0.31, 0.14);
    let tree = mix(vec3(0.05, 0.13, 0.08), vec3(0.06, 0.22, 0.05), warm);
    c = mix(c, grass, clamp(plants.r, 0.0, 1.0));
    c = mix(c, shrub, clamp(plants.g, 0.0, 1.0));
    c = mix(c, tree, clamp(plants.b, 0.0, 1.0));
    c *= 0.9 + 1.0 * n_ground;
    // Beaches, then bare rock on steep slopes: layered red sandstone in dry country (canyon
    // walls), grey rock elsewhere.
    c = mix(vec3(0.76, 0.70, 0.50), c, smoothstep(2.0, 30.0, h));
    // Sandstone only in warm dry country, not on cold dry mountains.
    let warm_rock = smoothstep(-2.0, 6.0, local_temperature(dir, h));
    let dry = warm_rock * smoothstep(0.32, 0.16, precipitation(sample_field(MOISTURE, dir), dir));
    let rock = mix(vec3(0.35, 0.30, 0.27), strata_color(dir, h, pixel_m), dry);
    c = mix(c, rock, smoothstep(0.35, 0.6, slope));
    return mix(c, vec3(0.92, 0.94, 0.97), snow_cover(dir, h, slope, pixel_m));
}

// Mirrors `EnvParams::lapse_rate_c_per_m` and `subtropic_dryness`.
const LAPSE_RATE_C_PER_M: f32 = 0.0065;
const SUBTROPIC_DRYNESS: f32 = 0.6;

// Same as `precipitation` in `env_params.wesl`: moisture, dried out in the subtropics.
fn precipitation(moisture: f32, dir: vec3<f32>) -> f32 {
    let lat_deg = abs(asin(clamp(dir.y, -1.0, 1.0))) * 57.29578;
    let belt = exp(-pow((lat_deg - 27.0) / 9.0, 2.0));
    return moisture * (1.0 - SUBTROPIC_DRYNESS * belt);
}

// 0..1 snow on the ground. The baked temperature is for the ~20 km texel's mean height, so
// it is corrected by the lapse rate to the actual height: peaks above the snow line are
// white even where the texel around them is not. Noise makes the snow line ragged, and
// cliffs too steep to hold snow stay bare unless it is very cold (ice caps, glaciers).
fn local_temperature(dir: vec3<f32>, h: f32) -> f32 {
    let field_h = max(sample_field(HEIGHT, dir), 0.0);
    return sample_field(TEMPERATURE, dir) - LAPSE_RATE_C_PER_M * (h - field_h);
}

fn snow_cover(dir: vec3<f32>, h: f32, slope: f32, pixel_m: f32) -> f32 {
    let t = local_temperature(dir, h);
    let ragged = t + 4.0 * detail_noise(dir, pixel_m, 700u);
    let cold = smoothstep(-1.0, -5.0, ragged);
    let holds = 1.0 - smoothstep(0.4, 0.75, slope);
    let glacier = smoothstep(-10.0, -16.0, ragged);
    return clamp(max(cold * holds, glacier * (1.0 - smoothstep(0.9, 1.4, slope))), 0.0, 1.0);
}

// Sedimentary layers: bands a few tens of metres thick in reds, ochres and creams,
// slightly wavy so they don't look ruled.
fn strata_color(dir: vec3<f32>, h: f32, pixel_m: f32) -> vec3<f32> {
    let y = h / 35.0 + 3.0 * detail_noise(dir, pixel_m, 811u);
    let band = floor(y);
    let k = fract(sin(band * 12.9898) * 43758.547);
    let red = vec3(0.55, 0.24, 0.13);
    let ochre = vec3(0.70, 0.45, 0.24);
    let cream = vec3(0.78, 0.68, 0.52);
    var c = mix(red, ochre, smoothstep(0.2, 0.6, k));
    c = mix(c, cream, smoothstep(0.75, 0.95, k));
    // Soft edges between bands, darker toward each band's base; the average from afar.
    c *= 0.8 + 0.2 * smoothstep(0.0, 0.4, fract(y));
    return mix(vec3(0.6, 0.38, 0.22), c, smoothstep(40.0, 10.0, pixel_m));
}

// Placeholder colours while the planet is still generating (no biome maps yet).
fn terrain_color(h: f32, slope: f32, lat: f32) -> vec3<f32> {
    if h < 0.0 {
        let depth = clamp(-h / 5000.0, 0.0, 1.0);
        return mix(vec3(0.05, 0.25, 0.35), vec3(0.005, 0.02, 0.08), sqrt(depth));
    }
    var c = mix(vec3(0.76, 0.70, 0.50), vec3(0.18, 0.35, 0.10), smoothstep(5.0, 80.0, h));
    c = mix(c, vec3(0.30, 0.32, 0.18), smoothstep(800.0, 2000.0, h));
    c = mix(c, vec3(0.35, 0.30, 0.27), smoothstep(0.35, 0.6, slope));
    let snow_line = mix(4200.0, 200.0, smoothstep(0.55, 0.95, lat));
    c = mix(c, vec3(0.92, 0.94, 0.97), smoothstep(snow_line, snow_line + 600.0, h) * (1.0 - smoothstep(0.5, 0.8, slope)));
    return c;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.local_dir);
    let h = sample_height(dir);

    // Normal by central differences, one texel apart.
    let eps = 1.0 / material.resolution;
    var t = cross(dir, vec3(0.0, 1.0, 0.0));
    if dot(t, t) < 1e-6 {
        t = cross(dir, vec3(1.0, 0.0, 0.0));
    }
    t = normalize(t);
    let b = cross(dir, t);
    let px = surface(normalize(dir + t * eps)) - surface(normalize(dir - t * eps));
    let py = surface(normalize(dir + b * eps)) - surface(normalize(dir - b * eps));
    var local_n = normalize(cross(px, py));
    if dot(local_n, dir) < 0.0 {
        local_n = -local_n;
    }
    let world_n = normalize(mesh_normal_local_to_world(local_n, in.instance_index));
    let pixel_m = length(fwidth(in.world_position.xyz));
    return shade(dir, h, 1.0 - dot(local_n, dir), world_n, in.world_position, in.world_position.xyz, in.clip_position, pixel_m);
}

struct TerrainVertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Unit direction from the planet centre, and height above sea level in metres.
    @location(2) surface: vec4<f32>,
};

struct TerrainOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) surface: vec4<f32>,
    // Camera-relative position before the waves move it (the waves' own coordinates).
    @location(3) rest: vec3<f32>,
};

@vertex
fn terrain_vertex(in: TerrainVertex) -> TerrainOutput {
    let world_from_local = get_world_from_local(in.instance_index);
    var world = mesh_position_local_to_world(world_from_local, vec4(in.position, 1.0));
    var out: TerrainOutput;
    out.rest = world.xyz;
    out.clip_position = position_world_to_clip(world.xyz);
    out.world_position = world;
    out.world_normal = mesh_normal_local_to_world(in.normal, in.instance_index);
    out.surface = in.surface;
    return out;
}

@fragment
fn terrain_fragment(in: TerrainOutput) -> @location(0) vec4<f32> {
    let dir = normalize(in.surface.xyz);
    // Chunks are only translated, so world and planet-local normals agree.
    let n = normalize(in.world_normal);
    let pixel_m = length(fwidth(in.world_position.xyz));
    if in.surface.w < 0.0 && material.show_biomes > 0.5 {
        return seabed(dir, in.surface.w, 1.0 - dot(n, dir), n, in.world_position, in.clip_position, pixel_m);
    }
    return shade(dir, in.surface.w, 1.0 - dot(n, dir), n, in.world_position, in.rest, in.clip_position, pixel_m);
}

// The sea surface: the chunk mesh again, each vertex dropped (or raised) back to sea level,
// then moved by the waves. Skirt vertices (normals stretched to 1 + skirt depth) are folded
// back up so they don't hang down into the water.
@vertex
fn sea_vertex(in: TerrainVertex) -> TerrainOutput {
    let world_from_local = get_world_from_local(in.instance_index);
    var world = mesh_position_local_to_world(world_from_local, vec4(in.position, 1.0));
    let up = normalize(in.surface.xyz);
    let skirt = max(length(in.normal) - 1.0, 0.0);
    world = vec4(world.xyz + up * (skirt - in.surface.w * material.height_exaggeration), 1.0);
    var out: TerrainOutput;
    out.rest = world.xyz;
    // Waves, faded out where the mesh is too coarse to carry them (chunks near the camera
    // have roughly distance / 40 between vertices) and toward the shore.
    let scale = sea_state(up) * smoothstep(0.0, 4.0, -in.surface.w);
    let sea = gerstner(world.xyz, up, scale, length(world.xyz) / 40.0);
    world = vec4(world.xyz + sea.offset, 1.0);
    out.clip_position = position_world_to_clip(world.xyz);
    out.world_position = world;
    out.world_normal = up;
    out.surface = in.surface;
    return out;
}

@fragment
fn sea_fragment(in: TerrainOutput) -> @location(0) vec4<f32> {
    // Over land the sea surface runs under the ground; cut it at the coastline.
    if in.surface.w > 0.0 {
        discard;
    }
    let dir = normalize(in.surface.xyz);
    let pixel_m = length(fwidth(in.world_position.xyz));
    // The camera is at the origin: below the surface when the surface is above it.
    if dot(in.world_position.xyz, dir) > 0.0 {
        return sea_from_below(dir, in.surface.w, in.world_position, in.rest, in.clip_position, pixel_m);
    }
    return shade(dir, in.surface.w, 0.0, dir, in.world_position, in.rest, in.clip_position, pixel_m);
}

// Light reaching depth `depth_m`, per channel: red goes first, blue-green last.
fn water_transmittance(depth_m: f32) -> vec3<f32> {
    return exp(-max(depth_m, 0.0) * vec3(0.35, 0.07, 0.05));
}

// Dancing caustics: sunlight focused by the waves, a few metres across, fading with depth.
fn caustics(dir: vec3<f32>, depth_m: f32) -> f32 {
    let p = dir * material.planet_radius_m;
    let t = globals.time * 0.6;
    let a = gradient_noise(p / 3.0 + vec3(t, 0.0, t * 0.7), 901u);
    let b = gradient_noise(p / 2.2 - vec3(t * 0.8, t * 0.5, 0.0), 902u);
    let lines = pow(clamp(1.0 - abs(a + b) * 0.8, 0.0, 1.0), 6.0);
    return lines * exp(-depth_m / 18.0);
}

// The seabed: sand in the shallows, mud deeper, rock on slopes; tinted and darkened by the
// water above it, with caustics. Only seen when diving, since the sea surface is opaque.
fn seabed(
    dir: vec3<f32>,
    h: f32,
    slope: f32,
    n: vec3<f32>,
    world_position: vec4<f32>,
    clip_position: vec4<f32>,
    pixel_m: f32,
) -> vec4<f32> {
    let depth = -h;
    let ripples = detail_noise(dir, pixel_m, 950u);
    var c = mix(vec3(0.76, 0.70, 0.52), vec3(0.32, 0.30, 0.25), smoothstep(20.0, 120.0, depth));
    c *= 0.85 + 0.3 * ripples;
    c = mix(c, vec3(0.30, 0.30, 0.28), smoothstep(0.35, 0.6, slope * 3.0));
    c = c * water_transmittance(depth) * (1.0 + 2.5 * caustics(dir, depth));

    var pbr: PbrInput = pbr_input_new();
    pbr.material.base_color = vec4(c, 1.0);
    pbr.material.perceptual_roughness = 0.9;
    pbr.material.metallic = 0.0;
    pbr.flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;
    pbr.frag_coord = clip_position;
    pbr.world_position = world_position;
    pbr.world_normal = n;
    pbr.N = n;
    pbr.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr.V = fns::calculate_view(world_position, pbr.is_orthographic);
    let lit = with_haze(fns::apply_pbr_lighting(pbr), world_position.xyz, clip_position.xy);
    return tone_mapping(lit, view.color_grading);
}

// The underside of the sea surface. Looking up steeply you see the bright sky through
// Snell's window (~49° half-angle); outside it, total internal reflection mirrors the dim
// water below.
fn sea_from_below(
    dir: vec3<f32>,
    h: f32,
    world_position: vec4<f32>,
    rest: vec3<f32>,
    clip_position: vec4<f32>,
    pixel_m: f32,
) -> vec4<f32> {
    let sea = gerstner(rest, dir, sea_state(dir), pixel_m);
    let n = ocean_normal(dir, sea.normal, pixel_m);
    let v = normalize(world_position.xyz);
    let cos_up = dot(v, n);
    let window = smoothstep(0.62, 0.70, cos_up);
    let sun = max(dot(lights.directional_lights[0].direction_to_light, dir), 0.0);
    // Luminance in the same physical units as the sky (Exposure::SUNLIGHT).
    let sky = vec3(0.55, 0.80, 0.90) * 9000.0 * (0.3 + 0.7 * sun);
    let reflected = vec3(0.02, 0.10, 0.12) * 3000.0 * (0.3 + 0.7 * sun);
    var color = mix(reflected, sky, window) * view.exposure;
    let lit = with_haze(vec4(color, 1.0), world_position.xyz, clip_position.xy);
    return tone_mapping(lit, view.color_grading);
}

struct Sea {
    // Displacement of the surface point, camera-relative world space (m).
    offset: vec3<f32>,
    normal: vec3<f32>,
    // 0..1: how compressed the surface is at a crest, where waves break into whitecaps.
    crest: f32,
}

// Wave height relative to a calm sea: a little rougher under cloud, much rougher in storms.
fn sea_state(dir: vec3<f32>) -> f32 {
    let density = cloud_density(cloud_noise_tex, material.resolution, dir, sample_field(MOISTURE, dir), globals.time, false);
    let cover = smoothstep(0.02, 0.32, density);
    return 0.7 + 0.6 * cover + 1.8 * storm_from_density(density);
}

// Sum of Gerstner waves at camera-relative rest position `rest` with local vertical `up`.
// Waves shorter than a few `resolution_m` fade out (their effect goes into roughness).
fn gerstner(rest: vec3<f32>, up: vec3<f32>, scale: f32, resolution_m: f32) -> Sea {
    var offset = vec3(0.0);
    var tilt = vec3(0.0);
    var pinch = 0.0;
    for (var i = 0u; i < WAVE_COUNT; i++) {
        let w = material.waves[i];
        // The wave vectors are tangent at the camera; re-project them for this point.
        let kv = w.xyz - up * dot(w.xyz, up);
        let k = length(kv);
        if k < 1e-6 {
            continue;
        }
        let d = kv / k;
        let fade = smoothstep(2.0 * resolution_m, 6.0 * resolution_m, 6.2831853 / k);
        let a = w.w * scale * fade;
        let theta = dot(w.xyz, rest) + material.wave_phases[i / 4u][i % 4u];
        let c = cos(theta);
        let s = sin(theta);
        offset += d * (WAVE_SHARPNESS * a * c) + up * (a * s);
        tilt += d * (k * a * c);
        pinch += WAVE_SHARPNESS * k * a * s;
    }
    var sea: Sea;
    sea.offset = offset;
    sea.normal = normalize(up * (1.0 - pinch) - tilt);
    sea.crest = clamp(pinch, 0.0, 1.0);
    return sea;
}

// Wave normal for the ocean: a few octaves of moving noise, tilting the flat sea surface.
// Octaves below a few pixels fade out (their effect goes into roughness instead).
fn ocean_normal(dir: vec3<f32>, flat_n: vec3<f32>, pixel_m: f32) -> vec3<f32> {
    var t = cross(dir, vec3(0.0, 1.0, 0.0));
    if dot(t, t) < 1e-6 {
        t = cross(dir, vec3(1.0, 0.0, 0.0));
    }
    t = normalize(t);
    let b = cross(dir, t);
    let p = dir * material.planet_radius_m;
    var slope = vec2(0.0);
    var wavelength = 120.0;
    var amp = 0.12;
    for (var i = 0u; i < 5u; i++) {
        let fade = smoothstep(2.0 * pixel_m, 8.0 * pixel_m, wavelength);
        let q = p / wavelength + vec3(globals.time * 0.6, 0.0, globals.time * 0.4) / (1.0 + f32(i));
        let e = 0.15;
        let n0 = gradient_noise(q, 500u + i);
        let nx = gradient_noise(q + t * e, 500u + i);
        let ny = gradient_noise(q + b * e, 500u + i);
        slope += amp * fade * vec2(nx - n0, ny - n0) / e;
        wavelength *= 0.45;
        amp *= 0.8;
    }
    return normalize(flat_n - t * slope.x - b * slope.y);
}

// Surface colour and lighting for a point at `dir` with height `h` (m above sea level).
// `slope` is 1 - cos(angle from vertical); `pixel_m` is the footprint of a pixel in metres.
fn shade(
    dir: vec3<f32>,
    h: f32,
    slope: f32,
    world_n: vec3<f32>,
    world_position: vec4<f32>,
    rest: vec3<f32>,
    clip_position: vec4<f32>,
    pixel_m: f32,
) -> vec4<f32> {
    var pbr: PbrInput = pbr_input_new();
    var n = world_n;
    var color = terrain_color(h, slope * 8.0, abs(dir.y));
    var roughness = select(0.85, 0.12, h < 0.0);
    var reflectance = 0.5;
    let live = material.show_biomes > 0.5;
    if h >= 0.0 && live {
        color = biome_color(dir, h, slope * 8.0, pixel_m);
    }
    if h >= 0.0 {
        // Rivers and lakes: strong flow or standing water on land. The fields are ~20 km
        // texels, so rivers fade out up close instead of drawing as wide blue bands.
        let river_scale = smoothstep(800.0, 4000.0, pixel_m);
        let river = river_scale * smoothstep(material.river_discharge * 2.0, material.river_discharge * 6.0, sample_field(DISCHARGE, dir));
        let lake = smoothstep(20.0, 60.0, sample_field(WATER, dir));
        let wet = max(river, lake);
        color = mix(color, vec3(0.02, 0.09, 0.12), wet);
        roughness = mix(roughness, 0.08, wet);
    } else if live {
        // Ocean: clearer and greener in the shallows, deep blue offshore, surf at the coast.
        let depth = -h;
        let deep = 1.0 - exp(-depth / 80.0);
        color = mix(vec3(0.03, 0.22, 0.24), vec3(0.003, 0.025, 0.07), deep);
        // Waves: the same Gerstner set as the vertex stage (which only moves vertices where
        // the mesh is fine enough), with small noise ripples on top.
        let scale = sea_state(dir) * smoothstep(0.0, 4.0, depth);
        let sea = gerstner(rest, dir, scale, pixel_m);
        n = ocean_normal(dir, sea.normal, pixel_m);
        // At grazing angles a tilted wave would mirror the sky below the horizon (the
        // atmosphere's brown "ground"), where really it would see the next wave. Tilt such
        // normals back toward the vertical so reflections stay above the horizon.
        let v = normalize(-world_position.xyz);
        let below = smoothstep(0.08, -0.15, dot(reflect(-v, n), dir));
        n = normalize(mix(n, dir, below));
        // Whitecaps where crests pinch, streaky, more of them in rough seas.
        let streaks = smoothstep(0.0, 0.6, detail_noise(dir, pixel_m, 330u));
        let whitecap = smoothstep(0.45, 0.75, sea.crest) * streaks;
        // Surf: bands of breaking foam rolling in over the last few metres of depth.
        let shallow = 1.0 - smoothstep(0.5, 6.0, depth);
        let rollers = smoothstep(0.55, 0.9, fract(depth * 0.4 - globals.time * 0.15 + detail_noise(dir, pixel_m, 300u)));
        let surf = shallow * max(rollers, 1.0 - smoothstep(0.3, 1.5, depth));
        let foam = clamp(max(whitecap, surf) * 0.85, 0.0, 1.0);
        color = mix(color, vec3(0.92, 0.95, 0.97), foam);
        // Unresolved waves look rougher from afar.
        roughness = mix(mix(0.04, 0.3, smoothstep(5.0, 3000.0, pixel_m)), 0.7, foam);
        reflectance = 0.35;
    }
    if live {
        // Cloud shadows: the cloud cover where the ray towards the sun crosses the cloud layer.
        let sun = lights.directional_lights[0].direction_to_light;
        let up = max(dot(dir, sun), 0.05);
        let p = dir * (material.planet_radius_m + max(h, 0.0)) + sun * (CLOUD_HEIGHT_M - max(h, 0.0)) / up;
        let shadow_dir = normalize(p);
        let cover = cloud_coverage(
            cloud_noise_tex, material.resolution, shadow_dir,
            sample_field(MOISTURE, shadow_dir), globals.time, false,
        );
        color *= 1.0 - 0.6 * cover;
    }
    pbr.material.base_color = vec4(color, 1.0);
    if h >= 0.0 && live {
        // Wildfires glow; bright enough to see on the night side.
        let fire = sample_field(FIRE, dir);
        pbr.material.emissive = vec4(vec3(8000.0, 2200.0, 300.0) * fire, 1.0);
    }
    pbr.material.reflectance = vec3(reflectance);
    pbr.material.perceptual_roughness = roughness;
    pbr.material.metallic = 0.0;
    // Receive sun shadows (pbr_input_new clears the mesh flags).
    pbr.flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;
    pbr.frag_coord = clip_position;
    pbr.world_position = world_position;
    pbr.world_normal = n;
    pbr.N = n;
    pbr.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr.V = fns::calculate_view(world_position, pbr.is_orthographic);

    let lit = with_haze(fns::apply_pbr_lighting(pbr), world_position.xyz, clip_position.xy);
    return tone_mapping(lit, view.color_grading);
}
