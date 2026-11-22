//! The planet surface on the CPU, in double precision: the baked height field (bicubic)
//! plus procedural detail below the field's resolution. The LOD terrain builds its meshes
//! from it, and the camera uses it to stay above the ground.

use bevy::math::{DVec2, DVec3, IVec3};
use std::f64::consts::FRAC_PI_4;

use super::mesh::FACE_AXES;

fn face_axes(face: usize) -> (DVec3, DVec3, DVec3) {
    let (n, u, v) = FACE_AXES[face];
    (n.as_dvec3(), u.as_dvec3(), v.as_dvec3())
}

/// Face-local `st` (pre-warp, [-1, 1]^2 on the face, a little beyond it for neighbours) to a
/// unit direction. Same convention as `face_st_to_dir` in `common.wesl`.
pub fn face_st_to_dir(face: usize, st: DVec2) -> DVec3 {
    let (n, u, v) = face_axes(face);
    let ab = DVec2::new((st.x * FRAC_PI_4).tan(), (st.y * FRAC_PI_4).tan());
    (n + ab.x * u + ab.y * v).normalize()
}

pub fn dir_to_face(dir: DVec3) -> usize {
    let a = dir.abs();
    if a.x >= a.y && a.x >= a.z {
        if dir.x >= 0.0 { 0 } else { 1 }
    } else if a.y >= a.z {
        if dir.y >= 0.0 { 2 } else { 3 }
    } else if dir.z >= 0.0 {
        4
    } else {
        5
    }
}

pub fn dir_to_face_st(face: usize, dir: DVec3) -> DVec2 {
    let (n, u, v) = face_axes(face);
    let d = dir.dot(n);
    DVec2::new((dir.dot(u) / d).atan(), (dir.dot(v) / d).atan()) / FRAC_PI_4
}

/// A baked R32Float cube-sphere field, sampled bicubically across cube edges.
pub struct CubeField {
    res: usize,
    data: Vec<f32>,
}

impl CubeField {
    pub fn new(res: u32, data: Vec<f32>) -> Self {
        let res = res as usize;
        assert_eq!(data.len(), res * res * 6, "field size");
        Self { res, data }
    }

    pub fn resolution(&self) -> u32 {
        self.res as u32
    }

    /// Texel value; coordinates a little outside the face wrap onto the neighbouring face.
    fn texel(&self, face: usize, x: i64, y: i64) -> f32 {
        let res = self.res as i64;
        let (face, x, y) = if (0..res).contains(&x) && (0..res).contains(&y) {
            (face, x as usize, y as usize)
        } else {
            let st = (DVec2::new(x as f64, y as f64) + 0.5) / res as f64 * 2.0 - 1.0;
            let dir = face_st_to_dir(face, st);
            let f = dir_to_face(dir);
            let p = ((dir_to_face_st(f, dir) * 0.5 + 0.5) * res as f64).floor();
            let c = |v: f64| (v as i64).clamp(0, res - 1) as usize;
            (f, c(p.x), c(p.y))
        };
        self.data[(face * self.res + y) * self.res + x]
    }

    /// Bicubic (Catmull-Rom) sample at a unit direction.
    pub fn sample(&self, dir: DVec3) -> f64 {
        let face = dir_to_face(dir);
        let c = (dir_to_face_st(face, dir) * 0.5 + 0.5) * self.res as f64 - 0.5;
        let base = c.floor();
        let f = c - base;
        let (bx, by) = (base.x as i64, base.y as i64);
        let wx = catmull_rom(f.x);
        let wy = catmull_rom(f.y);
        let mut sum = 0.0;
        for (j, wy) in wy.iter().enumerate() {
            let mut row = 0.0;
            for (i, wx) in wx.iter().enumerate() {
                row += wx * self.texel(face, bx + i as i64 - 1, by + j as i64 - 1) as f64;
            }
            sum += wy * row;
        }
        sum
    }
}

fn catmull_rom(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    ]
}

/// Procedural detail added to the baked height below its ~20 km texels.
#[derive(Clone, Copy, Debug)]
pub struct DetailParams {
    /// Wavelength of the coarsest detail octave, in metres.
    pub base_wavelength_m: f64,
    /// Amplitude of the coarsest octave in rough (mountain) terrain, in metres.
    pub amplitude_m: f64,
    /// Amplitude ratio between successive octaves (each half the wavelength).
    pub persistence: f64,
    /// Detail amplitude in flat lowlands, relative to mountains.
    pub lowland_roughness: f64,
    pub seed: u32,
}

impl Default for DetailParams {
    fn default() -> Self {
        Self {
            base_wavelength_m: 40_000.0,
            amplitude_m: 700.0,
            persistence: 0.56,
            lowland_roughness: 0.35,
            seed: 0,
        }
    }
}

/// The planet surface: baked height plus detail, in metres above sea level.
pub struct PlanetSurface {
    pub radius_m: f64,
    pub height: CubeField,
    pub detail: DetailParams,
}

impl PlanetSurface {
    /// Height above sea level at a unit direction, with detail octaves down to
    /// `min_wavelength_m` (pass about twice the mesh spacing, so LOD levels don't alias).
    pub fn height(&self, dir: DVec3, min_wavelength_m: f64) -> f64 {
        let base = self.height.sample(dir);
        base + self.detail(dir, base, min_wavelength_m)
    }

    fn detail(&self, dir: DVec3, base: f64, min_wavelength_m: f64) -> f64 {
        let d = &self.detail;
        let rough =
            d.lowland_roughness + (1.0 - d.lowland_roughness) * smoothstep(150.0, 2000.0, base);
        let p = dir * self.radius_m;
        let mut wavelength = d.base_wavelength_m;
        let mut amp = d.amplitude_m * rough;
        let mut sum = 0.0;
        let mut octave = 0;
        while wavelength >= min_wavelength_m && octave < 24 {
            // Fade the finest octave in, so a LOD switch doesn't pop.
            let fade = ((wavelength / min_wavelength_m - 1.0) * 2.0).clamp(0.0, 1.0);
            let n = gradient_noise(p / wavelength, d.seed.wrapping_add(octave * 131));
            // Mountains get sharper, ridged detail; lowlands stay rolling.
            let ridged = (1.0 - 2.0 * n.abs()) * 0.8;
            let n = n + (ridged - n) * smoothstep(800.0, 3000.0, base);
            sum += amp * fade * n;
            wavelength *= 0.5;
            amp *= d.persistence;
            octave += 1;
        }
        sum
    }
}

pub fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn hash(v: u32) -> u32 {
    let state = v.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    (word >> 22) ^ word
}

fn lattice_gradient(cell: IVec3, seed: u32) -> DVec3 {
    let h = hash(cell.x as u32 ^ hash(cell.y as u32 ^ hash(cell.z as u32 ^ hash(seed))));
    let g = |shift: u32| ((h >> shift) & 0x3ff) as f64 / 511.5 - 1.0;
    DVec3::new(g(0), g(10), g(20))
}

/// 3D gradient noise, roughly in [-1, 1].
pub fn gradient_noise(p: DVec3, seed: u32) -> f64 {
    let i = p.floor();
    let f = p - i;
    let i = IVec3::new(i.x as i32, i.y as i32, i.z as i32);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let mut n = [0.0; 8];
    for (k, n) in n.iter_mut().enumerate() {
        let o = IVec3::new((k & 1) as i32, ((k >> 1) & 1) as i32, ((k >> 2) & 1) as i32);
        *n = lattice_gradient(i + o, seed).dot(f - o.as_dvec3());
    }
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let x0 = lerp(lerp(n[0], n[1], u.x), lerp(n[2], n[3], u.x), u.y);
    let x1 = lerp(lerp(n[4], n[5], u.x), lerp(n[6], n[7], u.x), u.y);
    lerp(x0, x1, u.z) * 1.6
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn st_round_trip() {
        for face in 0..6 {
            for st in [DVec2::ZERO, DVec2::new(0.7, -0.3), DVec2::new(-0.99, 0.99)] {
                let dir = face_st_to_dir(face, st);
                assert_eq!(dir_to_face(dir), face);
                assert!(dir_to_face_st(face, dir).distance(st) < 1e-9);
            }
        }
    }

    /// A smooth function of direction, baked into a field, must sample back smoothly and
    /// continuously across cube edges and corners.
    #[test]
    fn bicubic_is_continuous_across_faces() {
        let res = 32u32;
        let f = |d: DVec3| 100.0 * d.x + 50.0 * d.y * d.z;
        let mut data = vec![0.0f32; (res * res * 6) as usize];
        for face in 0..6 {
            for y in 0..res {
                for x in 0..res {
                    let st = (DVec2::new(x as f64, y as f64) + 0.5) / res as f64 * 2.0 - 1.0;
                    data[((face * res + y) * res + x) as usize] =
                        f(face_st_to_dir(face as usize, st)) as f32;
                }
            }
        }
        let field = CubeField::new(res, data);
        let mut worst: f64 = 0.0;
        for i in 0..2000 {
            let t = i as f64 * 0.731;
            let dir = DVec3::new(t.sin() * 1.3, (t * 0.37).cos(), (t * 1.7).sin()).normalize();
            worst = worst.max((field.sample(dir) - f(dir)).abs());
        }
        assert!(worst < 2.0, "max error {worst}");
    }

    #[test]
    fn noise_is_bounded_and_deterministic() {
        for i in 0..1000 {
            let p = DVec3::new(i as f64 * 0.37, i as f64 * -1.1, 3.0e6 + i as f64 * 0.01);
            let n = gradient_noise(p, 7);
            assert!(n.abs() < 1.6 && n == gradient_noise(p, 7));
        }
    }
}
