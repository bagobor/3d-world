// Bevy's distance fog for custom materials: the camera's `DistanceFog` is the rain and
// snow haze (src/planet/weather.rs), transparent in clear weather.
import bevy_pbr::render::{
    mesh_view_bindings::{view, fog},
    pbr_functions as fns,
};

fn with_haze(color: vec4<f32>, world_position: vec3<f32>, frag_coord: vec2<f32>) -> vec4<f32> {
    var c = color;
    @if(DISTANCE_FOG) {
        c = fns::apply_fog(fog, color, world_position, view.world_position.xyz, frag_coord);
    }
    return c;
}
