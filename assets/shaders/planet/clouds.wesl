// The cloud shell: a sphere at cloud height, alpha-blended, lit by the sun through Bevy's
// PBR path. Seen from below, cloud bases are darker.
import bevy_pbr::render::{
    mesh_functions::{get_world_from_local, mesh_position_local_to_world},
    view_transformations::position_world_to_clip,
    mesh_view_bindings::{view, globals},
    pbr_types::{PbrInput, pbr_input_new},
    pbr_functions as fns,
};
import bevy_core_pipeline::tonemapping::tone_mapping;
import super::super::sim::common::{dir_to_face, dir_to_face_st, neighbor_texel, pcg3};
import super::cloud_field::{cloud_density, coverage_from_density, storm_from_density};

struct CloudParams {
    planet_radius_m: f32,
    cloud_height_m: f32,
    resolution: f32,
    visible: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> params: CloudParams;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var moisture_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var cloud_noise_tex: texture_2d_array<f32>;

fn load_moisture(t: vec3<u32>) -> f32 {
    return textureLoad(moisture_tex, vec2<i32>(t.xy), i32(t.z), 0).r;
}

fn sample_moisture(dir: vec3<f32>) -> f32 {
    let res = u32(params.resolution);
    let face = dir_to_face(dir);
    let st = dir_to_face_st(face, dir);
    let c = (st * 0.5 + 0.5) * params.resolution - 0.5;
    let base = floor(c);
    let f = c - base;
    let b = vec2<i32>(base);
    let origin = vec3<u32>(vec2<u32>(clamp(b, vec2(0), vec2(i32(res) - 1))), face);
    let d = b - vec2<i32>(origin.xy);
    let m00 = load_moisture(neighbor_texel(origin, d, res));
    let m10 = load_moisture(neighbor_texel(origin, d + vec2(1, 0), res));
    let m01 = load_moisture(neighbor_texel(origin, d + vec2(0, 1), res));
    let m11 = load_moisture(neighbor_texel(origin, d + vec2(1, 1), res));
    return mix(mix(m00, m10, f.x), mix(m01, m11, f.x), f.y);
}

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) dir: vec3<f32>,
    @location(2) @interpolate(flat) instance_index: u32,
};

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    let dir = normalize(in.position);
    let world = mesh_position_local_to_world(get_world_from_local(in.instance_index), vec4(dir, 1.0));
    var out: VertexOutput;
    out.clip_position = position_world_to_clip(world.xyz);
    out.world_position = world;
    out.dir = dir;
    out.instance_index = in.instance_index;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if params.visible < 0.5 {
        discard;
    }
    let dir = normalize(in.dir);
    let density = cloud_density(cloud_noise_tex, params.resolution, dir, sample_moisture(dir), globals.time, true);
    let cover = coverage_from_density(density);
    let storm = storm_from_density(density);
    if cover < 0.01 {
        discard;
    }
    // Is the camera below the cloud layer?
    let centre = (get_world_from_local(in.instance_index) * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    let below = length(view.world_position - centre) < params.planet_radius_m + params.cloud_height_m;
    let shade = select(1.0, 1.0 - 0.55 * cover, below);

    var pbr: PbrInput = pbr_input_new();
    // Storm clouds are thick and dark.
    pbr.material.base_color = vec4(vec3(0.95 * shade * (1.0 - 0.55 * storm)), 1.0);
    pbr.material.emissive = vec4(lightning(dir, storm), 1.0);
    pbr.material.perceptual_roughness = 1.0;
    pbr.material.metallic = 0.0;
    pbr.frag_coord = in.clip_position;
    pbr.world_position = in.world_position;
    pbr.world_normal = dir;
    pbr.N = dir;
    pbr.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr.V = fns::calculate_view(in.world_position, pbr.is_orthographic);
    var color = fns::apply_pbr_lighting(pbr);
    color.a = cover * 0.95;
    return tone_mapping(color, view.color_grading);
}

// Lightning inside storm clouds: ~30 km cells that light up at random, a few times a
// second somewhere in every storm, each flash a soft glow fading over a third of a second.
// Bright enough to see from orbit on the night side.
fn lightning(dir: vec3<f32>, storm: f32) -> vec3<f32> {
    if storm < 0.05 {
        return vec3(0.0);
    }
    let q = dir * 200.0;
    let cell = floor(q);
    let t = globals.time * 3.0;
    let slot = floor(t);
    let h = pcg3(bitcast<vec3<u32>>(vec3<i32>(cell)) ^ vec3(u32(slot), 7u, 13u));
    let r = f32(h.x & 0xffffu) / 65535.0;
    let strike = select(0.0, 1.0, r > 1.0 - 0.03 * storm);
    // A glow around a random point well inside the cell, so it never shows the cell's edges.
    let centre = 0.35 + 0.3 * vec3(f32(h.y & 0xffu), f32(h.z & 0xffu), f32((h.y >> 8u) & 0xffu)) / 255.0;
    let d = length(fract(q) - centre) / 0.35;
    let glow = exp(-4.0 * d * d) * (1.0 - smoothstep(0.7, 1.0, d));
    let fade = 1.0 - fract(t);
    return vec3(0.8, 0.85, 1.0) * 30000.0 * strike * glow * fade * fade;
}
