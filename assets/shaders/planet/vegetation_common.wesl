// Shared by the vegetation material (`vegetation.wesl`) and its shadow pass
// (`vegetation_prepass.wesl`): material bindings, field lookups, and how far each plant is
// grown in.
import super::super::sim::common::{dir_to_face, dir_to_face_st, neighbor_texel};

struct VegetationParams {
    planet_radius_m: f32,
    resolution: f32,
    tree_switch_m: f32,
    _pad: f32,
    // Draw distance of grass, shrubs and trees.
    fade_end_m: vec4<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> params: VegetationParams;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var plants_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var temperature_tex: texture_2d_array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(3) var moisture_tex: texture_2d_array<f32>;

const KIND_GRASS: u32 = 0u;
const KIND_SHRUB: u32 = 1u;
const KIND_BROADLEAF: u32 = 2u;
const KIND_CONIFER: u32 = 3u;
const PART_STEM: u32 = 0u;

const PLANTS: u32 = 0u;
const TEMPERATURE: u32 = 1u;
const MOISTURE: u32 = 2u;

fn load(field: u32, t: vec3<u32>) -> vec4<f32> {
    let p = vec2<i32>(t.xy);
    switch field {
        case 1u: { return textureLoad(temperature_tex, p, i32(t.z), 0); }
        case 2u: { return textureLoad(moisture_tex, p, i32(t.z), 0); }
        default: { return textureLoad(plants_tex, p, i32(t.z), 0); }
    }
}

// Bilinear lookup, continuous across cube edges.
fn sample_field(field: u32, dir: vec3<f32>) -> vec4<f32> {
    let res = u32(params.resolution);
    let face = dir_to_face(dir);
    let st = dir_to_face_st(face, dir);
    let c = (st * 0.5 + 0.5) * params.resolution - 0.5;
    let base = floor(c);
    let f = c - base;
    let b = vec2<i32>(base);
    let origin = vec3<u32>(vec2<u32>(clamp(b, vec2(0), vec2(i32(res) - 1))), face);
    let d = b - vec2<i32>(origin.xy);
    let v00 = load(field, neighbor_texel(origin, d, res));
    let v10 = load(field, neighbor_texel(origin, d + vec2(1, 0), res));
    let v01 = load(field, neighbor_texel(origin, d + vec2(0, 1), res));
    let v11 = load(field, neighbor_texel(origin, d + vec2(1, 1), res));
    return mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);
}

// How far a plant is grown in (0..1): by the shown cover of its type (`plants_shown`, see
// plants_settle.wesl) against the plant's threshold, faded out towards its layer's draw
// distance, and switched between the near and far tree meshes. `distance` is from the
// camera (the world origin).
fn plant_scale(code: u32, threshold: f32, up: vec3<f32>, distance: f32) -> f32 {
    let kind = code & 3u;
    let far = (code >> 3u) & 1u;
    let cover = sample_field(PLANTS, up);
    var c = cover.z;
    if kind == KIND_GRASS {
        c = cover.x;
    } else if kind == KIND_SHRUB {
        c = cover.y;
    }
    var s = clamp((c - threshold) * 8.0, 0.0, 1.0);
    let end = params.fade_end_m[min(kind, 2u)];
    // Grass hands over to the terrain colour quickly; shrubs and trees thin out gradually.
    let start = select(0.45, 0.75, kind == KIND_GRASS);
    s *= 1.0 - smoothstep(end * start, end, distance);
    if kind >= KIND_BROADLEAF {
        // Detailed trees near, cheap ones far.
        let near = distance < params.tree_switch_m;
        s *= select(0.0, 1.0, near == (far == 0u));
    }
    return s;
}
