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
import super::cloud_field::cloud_coverage;

struct PlanetMaterialParams {
    planet_radius_m: f32,
    height_exaggeration: f32,
    resolution: f32,
    river_discharge: f32,
    // 1 once the static maps exist (after generation or a cache load).
    show_biomes: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var height_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var<uniform> material: PlanetMaterialParams;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var water_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(3) var discharge_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(4) var temperature_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(5) var biome_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(6) var plants_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(7) var fire_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(8) var moisture_tex: texture_2d_array<f32>;

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
    // Beaches, bare rock on steep slopes, snow where it is cold enough.
    c = mix(vec3(0.76, 0.70, 0.50), c, smoothstep(2.0, 30.0, h));
    c = mix(c, vec3(0.35, 0.30, 0.27), smoothstep(0.35, 0.6, slope));
    let snow = smoothstep(-3.0, -9.0, sample_field(TEMPERATURE, dir)) * (1.0 - smoothstep(0.5, 0.8, slope));
    return mix(c, vec3(0.92, 0.94, 0.97), snow);
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
    return shade(dir, h, 1.0 - dot(local_n, dir), world_n, in.world_position, in.clip_position, pixel_m);
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
};

@vertex
fn terrain_vertex(in: TerrainVertex) -> TerrainOutput {
    let world_from_local = get_world_from_local(in.instance_index);
    let world = mesh_position_local_to_world(world_from_local, vec4(in.position, 1.0));
    var out: TerrainOutput;
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
    return shade(dir, in.surface.w, 1.0 - dot(n, dir), n, in.world_position, in.clip_position, pixel_m);
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
        let surf = (1.0 - smoothstep(0.5, 4.0, depth)) * smoothstep(-0.2, 0.4, detail_noise(dir, pixel_m, 300u) + 0.2);
        color = mix(color, vec3(0.8, 0.85, 0.85), surf * 0.7);
        n = ocean_normal(dir, n, pixel_m);
        // Unresolved waves look rougher from afar.
        roughness = mix(0.04, 0.3, smoothstep(5.0, 3000.0, pixel_m));
        reflectance = 0.35;
    }
    if live {
        // Cloud shadows: the cloud cover where the ray towards the sun crosses the cloud layer.
        let sun = lights.directional_lights[0].direction_to_light;
        let up = max(dot(dir, sun), 0.05);
        let p = dir * (material.planet_radius_m + max(h, 0.0)) + sun * (CLOUD_HEIGHT_M - max(h, 0.0)) / up;
        let shadow_dir = normalize(p);
        let cover = cloud_coverage(shadow_dir, sample_field(MOISTURE, shadow_dir), globals.time);
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

    return tone_mapping(fns::apply_pbr_lighting(pbr), view.color_grading);
}
