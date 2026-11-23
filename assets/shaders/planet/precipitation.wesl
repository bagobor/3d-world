// Rain and snow around the camera (src/planet/weather.rs). Each particle is a quad with a
// random position in a box centred on the camera; the box's contents drift with the air
// (and against the camera's motion) and wrap around, so the box never seems to move.
// Rain draws as thin streaks along the fall direction (motion blur), snow as soft flakes.
import bevy_pbr::render::{
    view_transformations::position_world_to_clip,
    mesh_view_bindings::view,
    pbr_types::{PbrInput, pbr_input_new},
    pbr_functions as fns,
};
import bevy_core_pipeline::tonemapping::tone_mapping;

struct PrecipitationParams {
    east: vec3<f32>,
    box_m: f32,
    north: vec3<f32>,
    intensity: f32,
    up: vec3<f32>,
    snow: f32,
    velocity: vec3<f32>,
    time: f32,
    drift: vec3<f32>,
    _pad: f32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> params: PrecipitationParams;

struct Vertex {
    @location(0) corner: vec3<f32>,
    @location(5) seed: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) fade: f32,
};

@vertex
fn vertex(in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    // Only a share of the particles, by rank, so intensity changes smoothly.
    if in.seed.w >= params.intensity {
        out.clip_position = vec4(0.0, 0.0, 2.0, 1.0);
        return out;
    }
    let b = params.box_m;
    var local = fract(in.seed.xyz + params.drift / b) * b - 0.5 * b;
    let snow = params.snow > 0.5;
    if snow {
        // Flakes flutter.
        let phase = in.seed.x * 40.0 + params.time * (1.0 + in.seed.y);
        local += vec3(sin(phase), cos(phase * 0.8), 0.0) * 0.25;
    }
    let to_world = mat3x3(params.east, params.north, params.up);
    let centre = to_world * local;
    let dist = length(centre);
    // Particles closer than ~1 m are skipped (they'd fill the screen), and far ones fade
    // so the box's edge doesn't show.
    out.fade = smoothstep(0.8, 2.0, dist) * (1.0 - smoothstep(0.3 * b, 0.5 * b, dist));
    let view_dir = centre / max(dist, 1e-3);
    // At least about a pixel wide.
    let min_width = dist * 0.002;
    var p: vec3<f32>;
    if snow {
        let right = normalize(cross(view_dir, params.up));
        let top = cross(right, view_dir);
        let size = max(0.012 + 0.01 * in.seed.z, min_width);
        p = centre + (right * in.corner.x + top * (in.corner.y * 2.0 - 1.0)) * size;
    } else {
        // A streak as long as the distance a drop falls in 1/30 s (motion blur).
        let v = to_world * params.velocity;
        let axis = normalize(v);
        let across = normalize(cross(axis, view_dir));
        let width = max(0.005, min_width);
        p = centre + across * (in.corner.x * width) + v * (in.corner.y * 0.033);
    }
    out.world_position = vec4(p, 1.0);
    out.clip_position = position_world_to_clip(p);
    out.uv = in.corner.xy;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    var alpha: f32;
    if params.snow > 0.5 {
        let r = length(vec2(in.uv.x, in.uv.y * 2.0 - 1.0));
        alpha = (1.0 - smoothstep(0.4, 1.0, r)) * 0.9;
    } else {
        alpha = (1.0 - abs(in.uv.x)) * 0.5;
    }
    alpha *= in.fade;
    if alpha < 0.01 {
        discard;
    }
    // Lit like a small diffuse object facing the camera, so drops are dim at night and
    // flash white in lightning.
    var pbr: PbrInput = pbr_input_new();
    pbr.material.base_color = vec4(select(vec3(0.75, 0.78, 0.82), vec3(0.95), params.snow > 0.5), 1.0);
    pbr.material.perceptual_roughness = 1.0;
    pbr.material.metallic = 0.0;
    pbr.frag_coord = in.clip_position;
    pbr.world_position = in.world_position;
    pbr.is_orthographic = view.clip_from_view[3].w == 1.0;
    pbr.V = fns::calculate_view(in.world_position, pbr.is_orthographic);
    pbr.world_normal = normalize(mix(pbr.V, params.up, 0.5));
    pbr.N = pbr.world_normal;
    var color = fns::apply_pbr_lighting(pbr);
    color.a = alpha;
    return tone_mapping(color, view.color_grading);
}
