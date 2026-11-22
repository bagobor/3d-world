// Shadow pass of the vegetation: the same plants, grown in the same way as in
// `vegetation.wesl` (without the wind sway), so shadows match what is drawn.
import bevy_pbr::render::{
    mesh_functions::{get_world_from_local, mesh_position_local_to_world},
    view_transformations::position_world_to_clip,
};
import super::vegetation_common::plant_scale;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(2) root: vec4<f32>,
    @location(3) plant: vec4<f32>,
};

@vertex
fn vertex(in: Vertex) -> @builtin(position) vec4<f32> {
    let world_from_local = get_world_from_local(in.instance_index);
    let root = mesh_position_local_to_world(world_from_local, vec4(in.root.xyz, 1.0)).xyz;
    let s = plant_scale(u32(floor(in.plant.w)), in.root.w, normalize(in.plant.xyz), length(root));
    let position = in.root.xyz + (in.position - in.root.xyz) * s;
    let world = mesh_position_local_to_world(world_from_local, vec4(position, 1.0));
    return position_world_to_clip(world.xyz);
}
