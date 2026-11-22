//! The planet camera and the floating origin.
//!
//! The camera's real position is kept in double precision ([`PlanetCamera`]), planet-centred,
//! in metres. The render camera itself always sits at the world origin; every planet entity
//! is placed relative to it each frame. That keeps f32 world coordinates small, so the
//! surface renders without jitter from 20 planet radii out down to a few metres above
//! the ground.
//!
//! The view is described like a pilot's: a point on the planet (`dir`), an altitude above sea
//! level, a heading (0 = north) and a pitch (0 = horizon, -90° = straight down).

use bevy::{
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    math::DVec3,
    prelude::*,
};

use super::lod::{TerrainChunk, TerrainLod};

/// Lowest eye height above the ground, in metres.
pub const MIN_EYE_HEIGHT_M: f64 = 1.8;

#[derive(Resource, Clone, Copy, Debug)]
pub struct PlanetCamera {
    pub radius_m: f64,
    /// Unit direction of the point under the camera.
    pub dir: DVec3,
    /// Metres above sea level.
    pub altitude_m: f64,
    /// Radians clockwise from north.
    pub heading: f64,
    /// Radians above the horizon (negative looks down).
    pub pitch: f64,
    /// Ground height under the camera last time it was known (metres above sea level).
    pub ground_m: f64,
}

impl PlanetCamera {
    pub fn new(radius_m: f64) -> Self {
        Self {
            radius_m,
            dir: DVec3::Z,
            altitude_m: radius_m * 2.2,
            heading: 0.0,
            pitch: -std::f64::consts::FRAC_PI_2,
            ground_m: 0.0,
        }
    }

    /// Camera position, planet-centred, in metres.
    pub fn position(&self) -> DVec3 {
        self.dir * (self.radius_m + self.altitude_m)
    }

    pub fn height_above_ground(&self) -> f64 {
        (self.altitude_m - self.ground_m.max(0.0)).max(0.0)
    }

    /// Local (north, east, up) at the camera.
    pub fn frame(&self) -> (DVec3, DVec3, DVec3) {
        let up = self.dir;
        let pole = if up.y.abs() > 0.999 {
            DVec3::X
        } else {
            DVec3::Y
        };
        let north = (pole - up * pole.dot(up)).normalize();
        let east = north.cross(up);
        (north, east, up)
    }

    /// Unit vectors the camera looks along, and its up.
    pub fn look(&self) -> (DVec3, DVec3) {
        let (north, east, up) = self.frame();
        let horizontal = north * self.heading.cos() + east * self.heading.sin();
        let forward = horizontal * self.pitch.cos() + up * self.pitch.sin();
        let cam_up = up * self.pitch.cos() - horizontal * self.pitch.sin();
        (forward, cam_up)
    }

    /// Moves over the surface by metres at the ground, along and across the heading.
    pub fn move_by(&mut self, forward_m: f64, right_m: f64) {
        let (north, east, up) = self.frame();
        let horizontal = north * self.heading.cos() + east * self.heading.sin();
        let right = horizontal.cross(up);
        let step = (horizontal * forward_m + right * right_m) / self.radius_m;
        self.dir = (self.dir + step).normalize();
    }
}

/// Marks entities placed at a fixed planet-centred position (the floating origin moves them).
#[derive(Component)]
pub struct PlanetAnchor(pub DVec3);

/// Interactive controls: left-drag moves over the planet, right-drag looks around, scroll
/// changes altitude, WASD flies (Shift for faster).
pub fn camera_controls(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    time: Res<Time>,
    mut camera: ResMut<PlanetCamera>,
) {
    let above = camera.height_above_ground().max(MIN_EYE_HEIGHT_M);
    if buttons.pressed(MouseButton::Left) && motion.delta != Vec2::ZERO {
        let scale = 0.0015 * above.min(camera.radius_m * 2.0);
        camera.move_by(
            motion.delta.y as f64 * scale,
            -motion.delta.x as f64 * scale,
        );
    }
    if buttons.pressed(MouseButton::Right) {
        camera.heading += motion.delta.x as f64 * 0.003;
        camera.pitch = (camera.pitch - motion.delta.y as f64 * 0.003).clamp(-1.57, 1.4);
    }
    if scroll.delta.y != 0.0 {
        let factor = (1.0 - scroll.delta.y as f64 * 0.12).clamp(0.5, 2.0);
        let ground = camera.ground_m.max(0.0);
        camera.altitude_m = ground + above * factor;
    }
    let mut forward = 0.0;
    let mut right = 0.0;
    for (key, f, r) in [
        (KeyCode::KeyW, 1.0, 0.0),
        (KeyCode::KeyS, -1.0, 0.0),
        (KeyCode::KeyD, 0.0, 1.0),
        (KeyCode::KeyA, 0.0, -1.0),
    ] {
        if keys.pressed(key) {
            forward += f;
            right += r;
        }
    }
    if forward != 0.0 || right != 0.0 {
        let boost = if keys.pressed(KeyCode::ShiftLeft) {
            5.0
        } else {
            1.0
        };
        let speed = (above * 0.6).max(4.0) * boost * time.delta_secs_f64();
        camera.move_by(forward * speed, right * speed);
    }
}

/// Keeps the camera above the ground and within range.
pub fn clamp_camera(mut camera: ResMut<PlanetCamera>, lod: Res<TerrainLod>) {
    if let Some(surface) = lod.surface() {
        camera.ground_m = surface.height(camera.dir, 1.0);
    }
    let min = camera.ground_m.max(0.0) + MIN_EYE_HEIGHT_M;
    let max = camera.radius_m * 20.0;
    camera.altitude_m = camera.altitude_m.clamp(min, max);
}

/// Floating origin: the render camera stays at the origin, looking where [`PlanetCamera`]
/// looks; terrain chunks and anchored entities move instead.
pub fn apply_floating_origin(
    camera: Res<PlanetCamera>,
    mut cameras: Query<
        &mut Transform,
        (With<Camera3d>, Without<TerrainChunk>, Without<PlanetAnchor>),
    >,
    mut chunks: Query<(&TerrainChunk, &mut Transform), Without<PlanetAnchor>>,
    mut anchored: Query<(&PlanetAnchor, &mut Transform), Without<TerrainChunk>>,
) {
    let eye = camera.position();
    let (forward, up) = camera.look();
    for mut transform in &mut cameras {
        *transform =
            Transform::from_translation(Vec3::ZERO).looking_to(forward.as_vec3(), up.as_vec3());
    }
    for (chunk, mut transform) in &mut chunks {
        transform.translation = (chunk.center - eye).as_vec3();
    }
    for (anchor, mut transform) in &mut anchored {
        transform.translation = (anchor.0 - eye).as_vec3();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looking_down_points_at_the_centre() {
        let cam = PlanetCamera::new(1000.0);
        let (forward, _) = cam.look();
        assert!(forward.distance(-cam.dir) < 1e-9);
    }

    #[test]
    fn heading_zero_looks_north() {
        let mut cam = PlanetCamera::new(1000.0);
        cam.dir = DVec3::X;
        cam.pitch = 0.0;
        let (forward, up) = cam.look();
        assert!(forward.distance(DVec3::Y) < 1e-9);
        assert!(up.distance(DVec3::X) < 1e-9);
    }
}
