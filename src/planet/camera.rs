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
//!
//! Three control modes ([`CameraMode`], keys 1/2/3): orbit (drag the planet around), fly
//! (free flight with mouse look) and walk (on foot, with gravity).

use bevy::{
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
    math::DVec3,
    prelude::*,
    window::{CursorGrabMode, CursorOptions},
};

use super::lod::{TerrainChunk, TerrainLod};

/// Lowest eye height above the ground, in metres.
pub const MIN_EYE_HEIGHT_M: f64 = 1.8;
/// Eye height when walking, in metres.
pub const WALK_EYE_HEIGHT_M: f64 = 1.7;
const WALK_SPEED_M_S: f64 = 1.6;
const RUN_SPEED_M_S: f64 = 7.0;
const JUMP_SPEED_M_S: f64 = 4.0;
const GRAVITY_M_S2: f64 = 9.81;

/// How the camera is controlled.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CameraMode {
    /// Drag the planet around, scroll to change altitude (key 1).
    #[default]
    Orbit,
    /// Free flight: mouse look, WASD along the view, E/Q up and down, scroll sets speed (2).
    Fly,
    /// On foot at eye height, following the ground: WASD walk, Shift runs, Space jumps (3).
    Walk,
}

impl CameraMode {
    pub fn name(self) -> &'static str {
        match self {
            CameraMode::Orbit => "orbit",
            CameraMode::Fly => "fly",
            CameraMode::Walk => "walk",
        }
    }
}

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
    pub mode: CameraMode,
    /// Fly mode: speed multiplier set with the scroll wheel.
    pub fly_speed: f64,
    /// Walk mode: vertical speed (jumping and falling), m/s.
    pub vertical_speed: f64,
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
            mode: CameraMode::Orbit,
            fly_speed: 1.0,
            vertical_speed: 0.0,
        }
    }

    /// Switches control mode. Walking starts on the ground, looking at the horizon.
    pub fn set_mode(&mut self, mode: CameraMode) {
        if mode == self.mode {
            return;
        }
        self.mode = mode;
        self.vertical_speed = 0.0;
        match mode {
            CameraMode::Walk => {
                self.altitude_m = self.ground_m.max(0.0) + WALK_EYE_HEIGHT_M;
                self.pitch = self.pitch.clamp(-0.3, 0.3);
            }
            CameraMode::Fly => self.fly_speed = 1.0,
            CameraMode::Orbit => {}
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

/// Interactive controls, per [`CameraMode`]. In every mode right-drag looks around. In fly
/// and walk mode a left click captures the mouse for looking around, Esc releases it.
pub fn camera_controls(
    buttons: Res<ButtonInput<MouseButton>>,
    keys: Res<ButtonInput<KeyCode>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    time: Res<Time>,
    mut camera: ResMut<PlanetCamera>,
    mut cursors: Query<&mut CursorOptions>,
) {
    for (key, mode) in [
        (KeyCode::Digit1, CameraMode::Orbit),
        (KeyCode::Digit2, CameraMode::Fly),
        (KeyCode::Digit3, CameraMode::Walk),
    ] {
        if keys.just_pressed(key) {
            camera.set_mode(mode);
        }
    }

    // Mouse capture for the first-person modes.
    let mut captured = false;
    for mut cursor in &mut cursors {
        let want = if camera.mode == CameraMode::Orbit || keys.just_pressed(KeyCode::Escape) {
            Some(false)
        } else if buttons.just_pressed(MouseButton::Left) {
            Some(true)
        } else {
            None
        };
        if let Some(grab) = want {
            cursor.grab_mode = if grab {
                CursorGrabMode::Locked
            } else {
                CursorGrabMode::None
            };
            cursor.visible = !grab;
        }
        captured |= cursor.grab_mode != CursorGrabMode::None;
    }
    if captured || buttons.pressed(MouseButton::Right) {
        camera.heading += motion.delta.x as f64 * 0.003;
        camera.pitch = (camera.pitch - motion.delta.y as f64 * 0.003).clamp(-1.57, 1.57);
    }

    let dt = time.delta_secs_f64();
    let axis =
        |pos: KeyCode, neg: KeyCode| (keys.pressed(pos) as i32 - keys.pressed(neg) as i32) as f64;
    let forward = axis(KeyCode::KeyW, KeyCode::KeyS);
    let right = axis(KeyCode::KeyD, KeyCode::KeyA);
    let shift = keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight);
    let above = camera.height_above_ground().max(MIN_EYE_HEIGHT_M);

    match camera.mode {
        CameraMode::Orbit => {
            if buttons.pressed(MouseButton::Left) && motion.delta != Vec2::ZERO {
                let scale = 0.0015 * above.min(camera.radius_m * 2.0);
                camera.move_by(
                    motion.delta.y as f64 * scale,
                    -motion.delta.x as f64 * scale,
                );
            }
            if scroll.delta.y != 0.0 {
                let factor = (1.0 - scroll.delta.y as f64 * 0.12).clamp(0.5, 2.0);
                let ground = camera.ground_m.max(0.0);
                camera.altitude_m = ground + above * factor;
            }
            if forward != 0.0 || right != 0.0 {
                let boost = if shift { 5.0 } else { 1.0 };
                let speed = (above * 0.6).max(4.0) * boost * dt;
                camera.move_by(forward * speed, right * speed);
            }
        }
        CameraMode::Fly => {
            if scroll.delta.y != 0.0 {
                camera.fly_speed =
                    (camera.fly_speed * (1.0 + scroll.delta.y as f64 * 0.15)).clamp(0.05, 50.0);
            }
            // Speed grows with height, so low passes are controllable and climbs are quick.
            let boost = if shift { 5.0 } else { 1.0 };
            let speed = (above * 0.5).clamp(15.0, 200_000.0) * camera.fly_speed * boost * dt;
            let up = axis(KeyCode::KeyE, KeyCode::KeyQ);
            let (sin, cos) = camera.pitch.sin_cos();
            // Along the view: horizontal part over the ground, vertical part as altitude.
            camera.move_by(forward * cos * speed, right * speed);
            camera.altitude_m += (forward * sin + up) * speed;
        }
        CameraMode::Walk => {
            let speed = if shift { RUN_SPEED_M_S } else { WALK_SPEED_M_S };
            let len = (forward * forward + right * right).sqrt().max(1.0);
            camera.move_by(forward / len * speed * dt, right / len * speed * dt);
            let floor = camera.ground_m.max(0.0) + WALK_EYE_HEIGHT_M;
            let on_ground = camera.altitude_m <= floor + 0.05;
            if on_ground && keys.just_pressed(KeyCode::Space) {
                camera.vertical_speed = JUMP_SPEED_M_S;
            }
            camera.vertical_speed -= GRAVITY_M_S2 * dt;
            camera.altitude_m += camera.vertical_speed * dt;
        }
    }
}

/// Keeps the camera above the ground and within range.
pub fn clamp_camera(mut camera: ResMut<PlanetCamera>, lod: Res<TerrainLod>) {
    if let Some(surface) = lod.surface() {
        camera.ground_m = surface.height(camera.dir, 1.0);
    }
    let eye = if camera.mode == CameraMode::Walk {
        WALK_EYE_HEIGHT_M
    } else {
        MIN_EYE_HEIGHT_M
    };
    let min = camera.ground_m.max(0.0) + eye;
    let max = camera.radius_m * 20.0;
    if camera.altitude_m <= min {
        camera.vertical_speed = 0.0;
    }
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
    fn walking_starts_on_the_ground_looking_ahead() {
        let mut cam = PlanetCamera::new(6_371_000.0);
        cam.altitude_m = 5_000.0;
        cam.ground_m = 320.0;
        cam.set_mode(CameraMode::Walk);
        assert_eq!(cam.altitude_m, 320.0 + WALK_EYE_HEIGHT_M);
        assert!(cam.pitch.abs() <= 0.3);
        // On the sea, walk on its surface.
        cam.ground_m = -40.0;
        cam.set_mode(CameraMode::Fly);
        cam.set_mode(CameraMode::Walk);
        assert_eq!(cam.altitude_m, WALK_EYE_HEIGHT_M);
    }

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
