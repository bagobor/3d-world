// Renders the planet straight from the simulation's `height` field: the vertex stage
// displaces a unit cube-sphere, the fragment stage derives normals and colours from the same
// field and runs Bevy's PBR lighting.

import bevy_pbr::render::{
    mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_normal_local_to_world},
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
    pbr_types::{PbrInput, pbr_input_new},
    pbr_functions as fns,
};
import bevy_core_pipeline::tonemapping::tone_mapping;
import super::super::sim::common::{dir_to_face, dir_to_face_st, neighbor_texel};

struct PlanetMaterialParams {
    planet_radius_m: f32,
    height_exaggeration: f32,
    resolution: f32,
    river_discharge: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var height_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var<uniform> material: PlanetMaterialParams;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var water_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(3) var discharge_tex: texture_2d_array<f32>;

fn load_field(field: u32, t: vec3<u32>) -> f32 {
    let p = vec2<i32>(t.xy);
    switch field {
        case 1u: { return textureLoad(water_tex, p, i32(t.z), 0).r; }
        case 2u: { return textureLoad(discharge_tex, p, i32(t.z), 0).r; }
        default: { return textureLoad(height_tex, p, i32(t.z), 0).r; }
    }
}

const HEIGHT: u32 = 0u;
const WATER: u32 = 1u;
const DISCHARGE: u32 = 2u;

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
    let slope = 1.0 - dot(local_n, dir);
    let world_n = normalize(mesh_normal_local_to_world(local_n, in.instance_index));

    var pbr: PbrInput = pbr_input_new();
    var color = terrain_color(h, slope * 8.0, abs(dir.y));
    var roughness = select(0.85, 0.12, h < 0.0);
    if h >= 0.0 {
        // Rivers and lakes: strong flow or standing water on land.
        let river = smoothstep(material.river_discharge, material.river_discharge * 4.0, sample_field(DISCHARGE, dir));
        let lake = smoothstep(20.0, 60.0, sample_field(WATER, dir));
        let wet = max(river, lake);
        color = mix(color, vec3(0.04, 0.16, 0.24), wet);
        roughness = mix(roughness, 0.1, wet);
    }
    pbr.material.base_color = vec4(color, 1.0);
    pbr.material.perceptual_roughness = roughness;
    pbr.material.metallic = 0.0;
    pbr.frag_coord = in.clip_position;
    pbr.world_position = in.world_position;
    pbr.world_normal = world_n;
    pbr.N = world_n;
    pbr.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr.V = fns::calculate_view(in.world_position, pbr.is_orthographic);

    return tone_mapping(fns::apply_pbr_lighting(pbr), view.color_grading);
}
