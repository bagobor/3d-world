//! The atmosphere: Bevy's physically based sky (Hillaire 2020) wrapped around the planet.
//!
//! The atmosphere entity sits at the planet centre, so with the floating origin it is placed
//! at `-camera` every frame like everything else. Near the ground the cheap lookup-texture
//! mode is used; from high altitude and space, the raymarched mode, which is accurate for
//! planets seen from orbit.

use bevy::{
    light::{Atmosphere, atmosphere::ScatteringMedium},
    math::DVec3,
    pbr::{AtmosphereMode, AtmosphereSettings},
    prelude::*,
};

use super::camera::{PlanetAnchor, PlanetCamera};

/// Height of the top of the atmosphere above sea level, in metres.
pub const ATMOSPHERE_HEIGHT_M: f32 = 100_000.0;

/// Above this altitude the sky is raymarched instead of read from lookup textures.
const RAYMARCH_ABOVE_M: f64 = 30_000.0;

pub fn spawn_atmosphere(
    mut commands: Commands,
    camera: Res<PlanetCamera>,
    mut media: ResMut<Assets<ScatteringMedium>>,
) {
    let radius = camera.radius_m as f32;
    commands.spawn((
        Name::new("Atmosphere"),
        Atmosphere {
            inner_radius: radius,
            outer_radius: radius + ATMOSPHERE_HEIGHT_M,
            ground_albedo: Vec3::splat(0.25),
            medium: media.add(ScatteringMedium::earth(256, 256)),
        },
        Transform::default(),
        PlanetAnchor(DVec3::ZERO),
    ));
}

/// Picks the rendering mode and aerial-perspective range for the camera's altitude.
pub fn adapt_atmosphere(camera: Res<PlanetCamera>, mut settings: Query<&mut AtmosphereSettings>) {
    let altitude = camera.altitude_m;
    for mut s in &mut settings {
        let mode = if altitude > RAYMARCH_ABOVE_M {
            AtmosphereMode::Raymarched
        } else {
            AtmosphereMode::LookupTexture
        };
        // Haze out to the horizon, which is farther away the higher the camera is.
        let horizon = (2.0 * camera.radius_m * altitude.max(100.0)).sqrt();
        let max_distance = horizon.clamp(30_000.0, 800_000.0) as f32;
        if s.rendering_method as u32 != mode as u32
            || s.aerial_view_lut_max_distance != max_distance
        {
            s.rendering_method = mode;
            s.aerial_view_lut_max_distance = max_distance;
        }
    }
}
