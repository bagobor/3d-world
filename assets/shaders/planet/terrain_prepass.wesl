// Shadow pass of the LOD terrain: chunk-local positions, nothing else. The entry point has
// the main pass's name because `TerrainMaterial::specialize` sets it for both.
import bevy_pbr::render::{
    mesh_functions::{get_world_from_local, mesh_position_local_to_world},
    view_transformations::position_world_to_clip,
};

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
};

@vertex
fn terrain_vertex(in: Vertex) -> @builtin(position) vec4<f32> {
    let world_from_local = get_world_from_local(in.instance_index);
    let world = mesh_position_local_to_world(world_from_local, vec4(in.position, 1.0));
    return position_world_to_clip(world.xyz);
}
