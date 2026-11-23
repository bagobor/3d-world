// Vegetation tiles (see src/planet/vegetation.rs): grass, shrubs and trees built on the CPU
// at full size. Here each plant is grown in by the live ecosystem: it shows where the local
// cover of its type exceeds the plant's random threshold, so the drawn density follows the
// simulation. Plants sway in the wind, fade out at their layer's draw distance, and are
// lit with Bevy's PBR lighting plus the same cloud shadows as the terrain.

import bevy_pbr::render::{
    mesh_functions::{get_world_from_local, mesh_position_local_to_world, mesh_normal_local_to_world},
    view_transformations::position_world_to_clip,
    mesh_view_bindings::{view, globals, lights},
    pbr_types::{PbrInput, pbr_input_new},
    pbr_functions as fns,
};
import bevy_core_pipeline::tonemapping::tone_mapping;
import bevy_pbr::render::mesh_types::MESH_FLAGS_SHADOW_RECEIVER_BIT;
import super::vegetation_common::{
    params, sample_field, plant_scale, KIND_GRASS, PART_STEM,
    TEMPERATURE, MOISTURE,
};
import super::cloud_field::cloud_coverage;

@group(constants::MATERIAL_BIND_GROUP) @binding(4) var cloud_noise_tex: texture_2d_array<f32>;

fn hash01(x: f32) -> f32 {
    return fract(sin(x * 91.3458) * 47453.5453);
}

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Plant root (tile-relative) and cover threshold.
    @location(2) root: vec4<f32>,
    // Up at the root; w = code (kind + 4 part + 8 far) + 0.99 ao.
    @location(3) plant: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) up: vec3<f32>,
    // Albedo (rgb) and ambient occlusion (a).
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) kind: u32,
};

// Albedo of a plant part for its climate; `v` is a per-plant random number.
fn plant_color(kind: u32, part: u32, temp_c: f32, moisture: f32, v: f32) -> vec3<f32> {
    if part == PART_STEM {
        return mix(vec3(0.16, 0.11, 0.07), vec3(0.28, 0.24, 0.2), v * v);
    }
    // Dry places have yellower, greyer foliage; the tropics a deeper green.
    let dry = 1.0 - smoothstep(0.2, 0.6, moisture);
    let warm = smoothstep(15.0, 27.0, temp_c);
    switch kind {
        case 0u: {
            let lush = mix(vec3(0.09, 0.17, 0.035), vec3(0.07, 0.16, 0.03), warm);
            let c = mix(lush, vec3(0.36, 0.3, 0.13), dry * (0.6 + 0.4 * v));
            return c * (0.8 + 0.4 * v);
        }
        case 1u: {
            let c = mix(vec3(0.1, 0.17, 0.05), vec3(0.24, 0.23, 0.13), dry);
            return c * (0.75 + 0.5 * v);
        }
        case 2u: {
            let c = mix(vec3(0.075, 0.15, 0.035), vec3(0.04, 0.12, 0.03), warm);
            let tint = mix(vec3(1.0), vec3(1.3, 1.15, 0.7), dry * 0.6);
            // Species variety: some crowns yellower, some bluer.
            let hue = mix(vec3(0.85, 1.0, 1.2), vec3(1.35, 1.15, 0.75), fract(v * 3.7));
            return c * tint * hue * (0.7 + 0.6 * v);
        }
        default: {
            return vec3(0.035, 0.075, 0.04) * (0.75 + 0.5 * v);
        }
    }
}

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    let code = u32(floor(in.plant.w));
    let ao = fract(in.plant.w);
    let kind = code & 3u;
    let part = (code >> 2u) & 1u;
    let up = normalize(in.plant.xyz);
    let threshold = in.root.w;

    let world_from_local = get_world_from_local(in.instance_index);
    let root = mesh_position_local_to_world(world_from_local, vec4(in.root.xyz, 1.0)).xyz;
    // The camera is at the origin (floating origin).
    let distance = length(root);

    let s = plant_scale(code, threshold, up, distance);

    // Wind: a gust travelling across the land, bending tall parts most.
    var local = in.position - in.root.xyz;
    let h = max(dot(local, up), 0.0);
    var side = cross(up, vec3(0.0, 1.0, 0.0));
    if dot(side, side) < 1e-6 {
        side = cross(up, vec3(1.0, 0.0, 0.0));
    }
    side = normalize(side);
    let phase = threshold * 37.0 + dot(root, side) * 0.15;
    let gust = sin(globals.time * 1.7 + phase) * 0.6 + sin(globals.time * 3.1 + phase * 1.7) * 0.4;
    var bend = 0.0;
    switch kind {
        case 0u: { bend = 0.35 * h * h; }
        case 1u: { bend = 0.02 * h * h; }
        default: { bend = 0.0006 * h * h; }
    }
    local += side * gust * bend * 0.5;

    let position = in.root.xyz + local * s;
    let world = mesh_position_local_to_world(world_from_local, vec4(position, 1.0));
    var out: VertexOutput;
    out.clip_position = position_world_to_clip(world.xyz);
    out.world_position = world;
    out.world_normal = mesh_normal_local_to_world(in.normal, in.instance_index);
    out.up = up;
    let v = hash01(threshold * 113.0 + f32(kind));
    let color = plant_color(kind, part, sample_field(TEMPERATURE, up).r, sample_field(MOISTURE, up).r, v);
    out.color = vec4(color, ao);
    out.kind = kind;
    return out;
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    var n = normalize(in.world_normal);
    if !front && in.kind != KIND_GRASS {
        n = -n;
    }
    // Self-shadowing inside canopies and at the foot of grass.
    var color = in.color.rgb * mix(0.4, 1.0, in.color.a);

    // Cloud shadows, as on the terrain.
    let sun = lights.directional_lights[0].direction_to_light;
    let dir = normalize(in.up);
    let lift = max(dot(dir, sun), 0.05);
    let shadow_dir = normalize(dir * params.planet_radius_m + sun * 6000.0 / lift);
    let cover = cloud_coverage(
        cloud_noise_tex, params.resolution, shadow_dir,
        sample_field(MOISTURE, shadow_dir).r, globals.time, false,
    );
    color *= 1.0 - 0.6 * cover;

    var pbr: PbrInput = pbr_input_new();
    pbr.material.base_color = vec4(color, 1.0);
    pbr.material.perceptual_roughness = 0.75;
    pbr.material.reflectance = vec3(0.3);
    pbr.material.metallic = 0.0;
    // Receive sun shadows (pbr_input_new clears the mesh flags).
    pbr.flags = MESH_FLAGS_SHADOW_RECEIVER_BIT;
    pbr.frag_coord = in.clip_position;
    pbr.world_position = in.world_position;
    pbr.world_normal = n;
    pbr.N = n;
    pbr.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr.V = fns::calculate_view(in.world_position, pbr.is_orthographic);
    return tone_mapping(fns::apply_pbr_lighting(pbr), view.color_grading);
}
