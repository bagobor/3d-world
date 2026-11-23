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

/// Precipitation from the baked moisture map: the CPU twin of `precipitation` in
/// `env_params.wesl` (moisture, dried out in the subtropical belts).
pub fn precipitation_field(res: u32, moisture: &[f32], subtropic_dryness: f64) -> CubeField {
    let mut data = moisture.to_vec();
    for face in 0..6 {
        for y in 0..res {
            for x in 0..res {
                let st = (DVec2::new(x as f64, y as f64) + 0.5) / res as f64 * 2.0 - 1.0;
                let dir = face_st_to_dir(face, st);
                let lat_deg = dir.y.clamp(-1.0, 1.0).asin().abs().to_degrees();
                let belt = (-((lat_deg - 27.0) / 9.0).powi(2)).exp();
                let i = ((face as u32 * res + y) * res + x) as usize;
                data[i] = (moisture[i] as f64 * (1.0 - subtropic_dryness * belt)) as f32;
            }
        }
    }
    CubeField::new(res, data)
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
    /// Detail amplitude in flat lowlands, relative to uplands.
    pub lowland_roughness: f64,
    /// Extra detail amplitude in high mountains, relative to uplands.
    pub mountain_relief: f64,
    pub seed: u32,
    /// Canyons cut into dry plateaus (needs [`PlanetSurface::precipitation`]).
    pub canyons: CanyonParams,
}

/// Canyon networks: deep channels with stepped walls, carved into dry plateaus. They are
/// part of the deterministic terrain surface (like the detail octaves), driven by the baked
/// precipitation and height maps; nothing about them is simulated at runtime.
#[derive(Clone, Copy, Debug)]
pub struct CanyonParams {
    /// Spacing of the main canyons, in metres.
    pub trunk_wavelength_m: f64,
    /// Half-width of a main canyon at the rim, in metres.
    pub trunk_half_width_m: f64,
    /// Depth of a main canyon, in metres.
    pub trunk_depth_m: f64,
    /// Spacing, half-width and depth of the side canyons.
    pub branch_wavelength_m: f64,
    pub branch_half_width_m: f64,
    pub branch_depth_m: f64,
    /// Number of cliff-and-bench steps in the walls.
    pub terraces: f64,
    /// Precipitation (see `precipitation` in `env_params.wesl`) below which plateaus are dry
    /// enough for canyons; they fade out between this and twice this.
    pub dry_precipitation: f64,
}

impl Default for CanyonParams {
    fn default() -> Self {
        Self {
            trunk_wavelength_m: 90_000.0,
            trunk_half_width_m: 2_600.0,
            trunk_depth_m: 1_300.0,
            branch_wavelength_m: 24_000.0,
            branch_half_width_m: 900.0,
            branch_depth_m: 550.0,
            terraces: 5.0,
            dry_precipitation: 0.16,
        }
    }
}

impl Default for DetailParams {
    fn default() -> Self {
        Self {
            base_wavelength_m: 40_000.0,
            amplitude_m: 700.0,
            persistence: 0.56,
            lowland_roughness: 0.35,
            mountain_relief: 1.8,
            seed: 0,
            canyons: CanyonParams::default(),
        }
    }
}

/// The planet surface: baked height plus detail, in metres above sea level.
pub struct PlanetSurface {
    pub radius_m: f64,
    pub height: CubeField,
    pub detail: DetailParams,
    /// Baked precipitation (moisture with the subtropical dry belt applied); canyons need it.
    pub precipitation: Option<CubeField>,
}

impl PlanetSurface {
    /// Height above sea level at a unit direction, with detail octaves down to
    /// `min_wavelength_m` (pass about twice the mesh spacing, so LOD levels don't alias).
    pub fn height(&self, dir: DVec3, min_wavelength_m: f64) -> f64 {
        let base = self.height.sample(dir);
        let mut detail = self.detail(dir, base, min_wavelength_m);
        // Offshore, detail must not raise the seabed through the sea surface (it would show
        // as scattered sand flats); near the coast islands and shoals are fine.
        let offshore = smoothstep(-20.0, -60.0, base);
        detail -= offshore * (detail - detail.min(-0.8 * base)).max(0.0);
        base + detail - self.canyon_depth_at(dir, base, min_wavelength_m)
    }

    /// How deep canyons cut into the surface at `dir`, in metres (0 outside canyons).
    pub fn canyon_depth(&self, dir: DVec3, min_wavelength_m: f64) -> f64 {
        self.canyon_depth_at(dir, self.height.sample(dir), min_wavelength_m)
    }

    fn canyon_depth_at(&self, dir: DVec3, base: f64, min_wavelength_m: f64) -> f64 {
        let Some(precipitation) = &self.precipitation else {
            return 0.0;
        };
        let c = &self.detail.canyons;
        // Only where a mesh can resolve the walls; distant LOD levels stay smooth.
        let resolve = smoothstep(
            4.0 * c.branch_half_width_m,
            1.5 * c.branch_half_width_m,
            min_wavelength_m,
        )
        .max(smoothstep(
            4.0 * c.trunk_half_width_m,
            1.5 * c.trunk_half_width_m,
            min_wavelength_m,
        ));
        // Dry plateaus well above the sea, below the high mountains.
        let plateau = smoothstep(250.0, 700.0, base) * (1.0 - smoothstep(3000.0, 4200.0, base));
        if resolve <= 0.0 || plateau <= 0.0 {
            return 0.0;
        }
        let dry = smoothstep(
            2.0 * c.dry_precipitation,
            c.dry_precipitation,
            precipitation.sample(dir),
        );
        if dry <= 0.0 {
            return 0.0;
        }
        let seed = self.detail.seed.wrapping_add(7_000);
        let p = dir * self.radius_m;
        // Canyon country comes in regions, not everywhere that is dry.
        let region = smoothstep(-0.15, 0.2, gradient_noise(p / 700_000.0, seed));
        let mask = dry * plateau * region * resolve;
        if mask <= 0.0 {
            return 0.0;
        }
        // Meandering channels: the zero set of warped noise. |noise| / gradient approximates
        // the distance to the channel's centre line.
        let warp_wl = 0.35 * c.trunk_wavelength_m;
        let warp = DVec3::new(
            gradient_noise(p / warp_wl, seed + 1),
            gradient_noise(p / warp_wl, seed + 2),
            gradient_noise(p / warp_wl, seed + 3),
        ) * (0.12 * c.trunk_wavelength_m);
        let q = p + warp;
        // Distance from a channel's centre line, in metres.
        let distance = |wavelength: f64, seed: u32| {
            gradient_noise(q / wavelength, seed).abs() * wavelength / 2.5
        };
        // Cross-section: 0 at the rim .. 1 on the floor, which is about a third of the width.
        let profile = |distance: f64, half_width: f64| {
            terrace(
                smoothstep(half_width, 0.3 * half_width, distance),
                c.terraces,
            )
        };
        let trunk_distance = distance(c.trunk_wavelength_m, seed + 4);
        let trunk = c.trunk_depth_m * profile(trunk_distance, c.trunk_half_width_m);
        // Side canyons are shallower, and only branch off near a main canyon.
        let near_trunk = smoothstep(
            10.0 * c.trunk_half_width_m,
            2.0 * c.trunk_half_width_m,
            trunk_distance,
        );
        let branch = near_trunk
            * c.branch_depth_m
            * profile(
                distance(c.branch_wavelength_m, seed + 5),
                c.branch_half_width_m,
            );
        // Never cut below the sea.
        (mask * trunk.max(branch)).min((base - 60.0).max(0.0))
    }

    fn detail(&self, dir: DVec3, base: f64, min_wavelength_m: f64) -> f64 {
        let d = &self.detail;
        // Rolling lowlands, rough uplands, and high mountains with much more relief, so the
        // broad baked highlands (one texel is ~20 km) break up into peaks and valleys.
        let rough = d.lowland_roughness
            + (1.0 - d.lowland_roughness) * smoothstep(150.0, 2000.0, base)
            + d.mountain_relief * smoothstep(1500.0, 4000.0, base);
        let p = dir * self.radius_m;
        let mut wavelength = d.base_wavelength_m;
        let mut amp = d.amplitude_m * rough;
        let mut sum = 0.0;
        let mut octave = 0;
        while wavelength >= min_wavelength_m && octave < 24 {
            // Fade the finest octave in, so a LOD switch doesn't pop.
            let fade = ((wavelength / min_wavelength_m - 1.0) * 2.0).clamp(0.0, 1.0);
            let n = gradient_noise(p / wavelength, d.seed.wrapping_add(octave * 131));
            // Mountains get sharp ridged crests (squared, so crests are narrow and valleys
            // broad); lowlands stay rolling.
            let r = 1.0 - n.abs();
            let ridged = 1.8 * r * r - 0.9;
            let n = n + (ridged - n) * smoothstep(800.0, 3000.0, base);
            sum += amp * fade * n;
            wavelength *= 0.5;
            amp *= d.persistence;
            octave += 1;
        }
        sum
    }
}

/// Steps `s` (0..1) into `steps` flat benches joined by steep cliffs.
fn terrace(s: f64, steps: f64) -> f64 {
    let x = s * steps;
    let i = x.floor();
    ((i + smoothstep(0.65, 1.0, x - i)) / steps).min(1.0)
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

    /// Dry plateaus get canyons that are deep but never reach the sea; wet ones get none.
    #[test]
    fn canyons_cut_dry_plateaus_only() {
        let res = 8u32;
        let n = (res * res * 6) as usize;
        let surface = |precipitation: f32| PlanetSurface {
            radius_m: 6_371_000.0,
            height: CubeField::new(res, vec![1_500.0; n]),
            detail: DetailParams::default(),
            precipitation: Some(CubeField::new(res, vec![precipitation; n])),
        };
        let (dry, wet) = (surface(0.02), surface(0.8));
        let mut deepest: f64 = 0.0;
        for i in 0..20_000 {
            let t = i as f64 * 0.0137;
            let dir = DVec3::new(t.sin(), 0.3 + 0.1 * (t * 0.7).cos(), t.cos()).normalize();
            let d = dry.canyon_depth(dir, 50.0);
            assert!((0.0..=1_440.0).contains(&d), "depth {d}");
            assert_eq!(wet.canyon_depth(dir, 50.0), 0.0);
            // Too coarse a mesh to resolve the walls: no canyon.
            assert_eq!(dry.canyon_depth(dir, 50_000.0), 0.0);
            deepest = deepest.max(d);
        }
        assert!(deepest > 1_000.0, "deepest canyon {deepest}");
    }

    /// Offshore, the detail never lifts the seabed above the sea.
    #[test]
    fn open_sea_has_no_islands() {
        let res = 8u32;
        let n = (res * res * 6) as usize;
        let surface = PlanetSurface {
            radius_m: 6_371_000.0,
            height: CubeField::new(res, vec![-150.0; n]),
            detail: DetailParams::default(),
            precipitation: None,
        };
        for i in 0..5_000 {
            let t = i as f64 * 0.0137;
            let dir = DVec3::new(t.sin(), 0.3 * (t * 0.7).cos(), t.cos()).normalize();
            assert!(surface.height(dir, 5.0) < 0.0);
        }
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
