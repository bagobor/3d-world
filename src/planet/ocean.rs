//! Ocean waves: a small set of Gerstner waves that move the sea surface near the camera.
//!
//! Wave phases need more precision than f32 has at planet scale (a 7 m wave 6 400 km from
//! the planet centre), so the shader only sees camera-relative positions: each wave's phase
//! at the camera is computed here in f64 every frame, and the shader adds `K · x` for the
//! small world-space offset `x` from the camera. The wave vectors lie in the tangent plane at
//! the camera, which is accurate for the few kilometres around it where waves are visible.

use bevy::{math::DVec3, prelude::*};
use std::f64::consts::TAU;

use super::{PlanetMaterialParams, SeaMaterial, SeaMaterialHandle, camera::PlanetCamera};

/// Number of Gerstner waves (must match `WAVE_COUNT` in `planet_material.wesl`).
pub const WAVE_COUNT: usize = 8;

const GRAVITY_M_S2: f64 = 9.81;

/// One wave of the set: wavelength, amplitude relative to the wavelength (calm sea), and
/// its direction relative to the wind, in radians.
struct Wave {
    wavelength_m: f64,
    steepness: f64,
    angle: f64,
}

/// A long swell along the wind, and shorter waves spread around it.
const WAVES: [Wave; WAVE_COUNT] = [
    Wave {
        wavelength_m: 120.0,
        steepness: 0.0055,
        angle: 0.0,
    },
    Wave {
        wavelength_m: 77.0,
        steepness: 0.0060,
        angle: 0.45,
    },
    Wave {
        wavelength_m: 53.0,
        steepness: 0.0065,
        angle: -0.35,
    },
    Wave {
        wavelength_m: 34.0,
        steepness: 0.0070,
        angle: 0.8,
    },
    Wave {
        wavelength_m: 23.0,
        steepness: 0.0075,
        angle: -0.7,
    },
    Wave {
        wavelength_m: 15.0,
        steepness: 0.0080,
        angle: 0.25,
    },
    Wave {
        wavelength_m: 10.0,
        steepness: 0.0085,
        angle: -1.0,
    },
    Wave {
        wavelength_m: 6.5,
        steepness: 0.0090,
        angle: 1.1,
    },
];

/// Prevailing wind heading (radians from north, towards which it blows) at a latitude:
/// trade winds blow west near the equator, westerlies east in mid-latitudes, polar
/// easterlies west again.
pub fn wind_heading(dir: DVec3) -> f64 {
    let lat = dir.y.clamp(-1.0, 1.0).asin().abs().to_degrees();
    let east = std::f64::consts::FRAC_PI_2;
    let west = -east;
    // Lean a little poleward or equatorward, like real surface winds.
    if lat < 30.0 {
        west - 0.3 * dir.y.signum()
    } else if lat < 60.0 {
        east + 0.3 * dir.y.signum()
    } else {
        west
    }
}

/// Each wave's vector (rad/m, planet frame, tangent at the camera), angular frequency
/// (rad/s) and amplitude (m).
fn wave_vectors(camera: &PlanetCamera) -> [(DVec3, f64, f64); WAVE_COUNT] {
    let (north, east, _) = camera.frame();
    let wind = wind_heading(camera.dir);
    WAVES.map(|w| {
        let k = TAU / w.wavelength_m;
        let heading = wind + w.angle;
        let kv = (north * heading.cos() + east * heading.sin()) * k;
        // Deep-water dispersion.
        let omega = (GRAVITY_M_S2 * k).sqrt();
        (kv, omega, w.steepness * w.wavelength_m)
    })
}

/// Wave vectors (xyz, rad/m, in the planet frame) with amplitudes (w, m), and each wave's
/// phase at the camera, for time `t` seconds.
pub fn wave_uniforms(camera: &PlanetCamera, t: f64) -> ([Vec4; WAVE_COUNT], [Vec4; 2]) {
    let position = camera.position();
    let mut waves = [Vec4::ZERO; WAVE_COUNT];
    let mut phases = [Vec4::ZERO; 2];
    for (i, (kv, omega, amplitude)) in wave_vectors(camera).into_iter().enumerate() {
        let phase = (kv.dot(position) - omega * t).rem_euclid(TAU);
        waves[i] = kv.as_vec3().extend(amplitude as f32);
        phases[i / 4][i % 4] = phase as f32;
    }
    (waves, phases)
}

/// Moves the waves with time and the camera.
pub(crate) fn update_waves(
    camera: Res<PlanetCamera>,
    time: Res<Time>,
    handle: Option<Res<SeaMaterialHandle>>,
    mut materials: ResMut<Assets<SeaMaterial>>,
) {
    let Some(handle) = handle else {
        return;
    };
    let Some(mut material) = materials.get_mut(&handle.0) else {
        return;
    };
    let (waves, phases) = wave_uniforms(&camera, time.elapsed_secs_f64());
    let params: &mut PlanetMaterialParams = &mut material.params;
    params.waves = waves;
    params.wave_phases = phases;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The phase the shader reconstructs (camera phase + K · offset) matches the exact
    /// planet-frame phase near the camera, even though f32 can't hold the latter directly.
    #[test]
    fn camera_relative_phase_is_exact() {
        let mut camera = PlanetCamera::new(6_371_000.0);
        camera.dir = DVec3::new(0.3, 0.5, 0.8).normalize();
        camera.altitude_m = 12.0;
        let t = 1234.5;
        let (waves, phases) = wave_uniforms(&camera, t);
        let (north, east, up) = camera.frame();
        let offset = north * 37.0 + east * -51.0 - up * 12.0;
        for (i, (kv, omega, _)) in wave_vectors(&camera).into_iter().enumerate() {
            let exact = (kv.dot(camera.position() + offset) - omega * t).rem_euclid(TAU);
            // What the shader does: f32 wave vector, f32 phase, f32 offset.
            let k = waves[i].truncate();
            let shader = (phases[i / 4][i % 4] + k.dot(offset.as_vec3())) as f64;
            let shader = shader.rem_euclid(TAU);
            let diff = (exact - shader).abs();
            assert!(diff.min(TAU - diff) < 1e-3, "wave {i}: {exact} vs {shader}");
        }
    }

    #[test]
    fn waves_lie_in_the_tangent_plane() {
        let camera = PlanetCamera::new(6_371_000.0);
        let (waves, _) = wave_uniforms(&camera, 0.0);
        for w in waves {
            let k = w.truncate().as_dvec3();
            assert!(k.normalize().dot(camera.dir).abs() < 1e-6);
            assert!(w.w > 0.0 && w.w < 1.0);
        }
    }
}
