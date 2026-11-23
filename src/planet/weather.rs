//! Weather near the camera: rain, snow, storms and lightning.
//!
//! There is no weather simulation. The procedural cloud field (`cloud_field.wesl`, mirrored
//! here on the CPU) already drifts over the planet; where it is thickest it rains, where it
//! is thicker still there are storms. Whether it rains or snows follows the baked
//! temperature, corrected to the camera's height. Everything is evaluated at the camera:
//! the precipitation particles, the rain haze, the darker light under storm clouds and the
//! lightning flashes. The cloud shell and the sea read the same field on the GPU, so storm
//! clouds, lightning seen from orbit and rough seas all line up with it.

use bevy::{
    asset::RenderAssetUsages,
    camera::visibility::NoFrustumCulling,
    light::{NotShadowCaster, NotShadowReceiver},
    math::{DVec3, IVec3},
    mesh::{Indices, PrimitiveTopology},
    pbr::{DistanceFog, FogFalloff},
    prelude::*,
    render::render_resource::{AsBindGroup, ShaderType},
    shader::ShaderRef,
};
use std::sync::Arc;

use super::{
    CLOUD_HEIGHT_M, EnvParams,
    camera::PlanetCamera,
    surface::{CubeField, smoothstep},
};

/// The scene's ambient light outside lightning flashes (keeps the night side from going
/// black).
pub const AMBIENT_BRIGHTNESS: f32 = 300.0;

/// Precipitation particles drawn around the camera (rain uses all, snow fewer).
const PARTICLES: u32 = 24_000;

/// The sun (the scene's main directional light).
#[derive(Component)]
pub struct Sun;

/// What the weather is doing at the camera.
#[derive(Resource, Default)]
pub struct Weather {
    moisture: Option<Arc<CubeField>>,
    temperature: Option<Arc<CubeField>>,
    height: Option<Arc<CubeField>>,
    /// 0..1 cloud cover overhead.
    pub cover: f32,
    /// 0..1 precipitation intensity.
    pub precipitation: f32,
    /// 0..1 storminess.
    pub storm: f32,
    /// Whether precipitation falls as snow.
    pub snow: bool,
    /// 0..1 brightness of the current lightning flash.
    pub flash: f32,
    /// Seconds until the next lightning strike.
    next_strike: f32,
    /// Time since the last strike started, in seconds.
    since_strike: f32,
    rng: u32,
}

impl Weather {
    /// Gives the weather the baked maps it needs.
    pub fn set_planet(
        &mut self,
        res: u32,
        moisture: Vec<f32>,
        temperature: Vec<f32>,
        height: Vec<f32>,
    ) {
        self.moisture = Some(Arc::new(CubeField::new(res, moisture)));
        self.temperature = Some(Arc::new(CubeField::new(res, temperature)));
        self.height = Some(Arc::new(CubeField::new(res, height)));
    }

    pub fn ready(&self) -> bool {
        self.moisture.is_some()
    }

    /// Cloud density at a direction and time (see [`cloud_density`]).
    pub fn density(&self, dir: DVec3, time: f64) -> f64 {
        let moisture = self.moisture.as_ref().map_or(0.6, |m| m.sample(dir));
        cloud_density(dir, moisture, time)
    }

    /// Air temperature at a direction and height above sea level, in °C.
    pub fn temperature(&self, dir: DVec3, height_m: f64) -> f64 {
        let (Some(t), Some(h)) = (&self.temperature, &self.height) else {
            return 15.0;
        };
        let lapse = EnvParams::default().lapse_rate_c_per_m as f64;
        t.sample(dir) - lapse * (height_m - h.sample(dir).max(0.0))
    }

    fn random(&mut self) -> f32 {
        self.rng = hash(self.rng.wrapping_add(0x9e37_79b9));
        self.rng as f32 / u32::MAX as f32
    }
}

pub fn coverage_from_density(v: f64) -> f64 {
    smoothstep(0.02, 0.32, v)
}

pub fn rain_from_density(v: f64) -> f64 {
    smoothstep(0.22, 0.38, v)
}

pub fn storm_from_density(v: f64) -> f64 {
    smoothstep(0.38, 0.55, v)
}

/// Raw cloud density: the CPU twin of `cloud_density` in `cloud_field.wesl`.
pub fn cloud_density(dir_in: DVec3, moisture: f64, time: f64) -> f64 {
    let a = time * 0.0004;
    let dir = DVec3::new(
        dir_in.x * a.cos() - dir_in.z * a.sin(),
        dir_in.y,
        dir_in.x * a.sin() + dir_in.z * a.cos(),
    );
    let lat = dir.y.clamp(-1.0, 1.0).asin().abs().to_degrees();
    let band = 0.25 * (-(lat / 9.0).powi(2)).exp() - 0.22 * (-((lat - 25.0) / 9.0).powi(2)).exp()
        + 0.12 * (-((lat - 55.0) / 12.0).powi(2)).exp();
    let p = dir * 5.0;
    let warp = DVec3::new(
        cloud_fbm(p + 11.0, 3),
        cloud_fbm(p + 23.0, 3),
        cloud_fbm(p + 37.0, 3),
    );
    let n = cloud_fbm(p * 1.6 + warp * 1.4, 7);
    n + band + (moisture - 0.6) * 0.35
}

fn cloud_fbm(p: DVec3, octaves: u32) -> f64 {
    let mut sum = 0.0;
    let mut amp = 0.5;
    let mut q = p;
    for i in 0..octaves {
        sum += amp * gpu_gradient_noise(q, 1700 + i * 13);
        q *= 2.1;
        amp *= 0.5;
    }
    sum
}

/// `pcg3` from `common.wesl`.
fn pcg3(v: [u32; 3]) -> [u32; 3] {
    let mut w = v.map(|x| x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223));
    w[0] = w[0].wrapping_add(w[1].wrapping_mul(w[2]));
    w[1] = w[1].wrapping_add(w[2].wrapping_mul(w[0]));
    w[2] = w[2].wrapping_add(w[0].wrapping_mul(w[1]));
    w = w.map(|x| x ^ (x >> 16));
    w[0] = w[0].wrapping_add(w[1].wrapping_mul(w[2]));
    w[1] = w[1].wrapping_add(w[2].wrapping_mul(w[0]));
    w[2] = w[2].wrapping_add(w[0].wrapping_mul(w[1]));
    w
}

/// `gradient_noise` from `common.wesl` (the GPU's noise, which differs from
/// [`super::surface::gradient_noise`]).
fn gpu_gradient_noise(p: DVec3, seed: u32) -> f64 {
    let i = p.floor();
    let f = p - i;
    let i = IVec3::new(i.x as i32, i.y as i32, i.z as i32);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let gradient = |c: IVec3| {
        let h = pcg3([
            c.x as u32 ^ seed,
            c.y as u32 ^ seed.wrapping_mul(3).wrapping_add(1),
            c.z as u32 ^ seed.wrapping_mul(7).wrapping_add(2),
        ]);
        DVec3::new(
            (h[0] & 0xffff) as f64 / 32767.5 - 1.0,
            (h[1] & 0xffff) as f64 / 32767.5 - 1.0,
            (h[2] & 0xffff) as f64 / 32767.5 - 1.0,
        )
    };
    let mut n = [0.0; 8];
    for (k, n) in n.iter_mut().enumerate() {
        let o = IVec3::new((k & 1) as i32, ((k >> 1) & 1) as i32, ((k >> 2) & 1) as i32);
        *n = gradient(i + o).dot(f - o.as_dvec3());
    }
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let x0 = lerp(lerp(n[0], n[1], u.x), lerp(n[2], n[3], u.x), u.y);
    let x1 = lerp(lerp(n[4], n[5], u.x), lerp(n[6], n[7], u.x), u.y);
    lerp(x0, x1, u.z) * 1.6
}

fn hash(v: u32) -> u32 {
    let state = v.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// Precipitation particles: a box of streaks (rain) or flakes (snow) that moves with the
/// camera, see `precipitation.wesl`.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct PrecipitationMaterial {
    #[uniform(0)]
    pub params: PrecipitationParams,
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct PrecipitationParams {
    /// Local frame at the camera.
    pub east: Vec3,
    pub box_m: f32,
    pub north: Vec3,
    /// Fraction of the particles shown.
    pub intensity: f32,
    pub up: Vec3,
    /// 1 for snow, 0 for rain.
    pub snow: f32,
    /// Particle velocity in the local frame (east, north, up), m/s.
    pub velocity: Vec3,
    pub time: f32,
    /// How far all particles have moved relative to the camera since the start, in the
    /// local frame, wrapped to the box (the camera's own motion included).
    pub drift: Vec3,
    pub _pad: f32,
}

impl Material for PrecipitationMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/planet/precipitation.wesl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/planet/precipitation.wesl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn specialize(
        _pipeline: &bevy::pbr::MaterialPipeline,
        descriptor: &mut bevy::render::render_resource::RenderPipelineDescriptor,
        _layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: bevy::pbr::MaterialPipelineKey<Self>,
    ) -> Result<(), bevy::render::render_resource::SpecializedMeshPipelineError> {
        // Quads face whichever way their streak happens to be built.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// A unit quad per particle: corners in `POSITION` (x across, y along), a random seed in
/// `COLOR` (xyz: position in the box, w: rank, shown when below the intensity).
fn particle_mesh(count: u32) -> Mesh {
    let mut positions = Vec::with_capacity(count as usize * 4);
    let mut seeds = Vec::with_capacity(count as usize * 4);
    let mut indices = Vec::with_capacity(count as usize * 6);
    let mut rng = 12_345u32;
    let mut random = || {
        rng = hash(rng);
        rng as f32 / u32::MAX as f32
    };
    for i in 0..count {
        let seed = [random(), random(), random(), random()];
        for corner in [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ] {
            positions.push(corner);
            seeds.push(seed);
        }
        let b = i * 4;
        indices.extend_from_slice(&[b, b + 1, b + 2, b, b + 2, b + 3]);
    }
    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, seeds)
    .with_inserted_indices(Indices::U32(indices))
}

#[derive(Resource)]
pub struct PrecipitationHandle(Handle<PrecipitationMaterial>);

pub fn spawn_weather(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<PrecipitationMaterial>>,
) {
    let material = materials.add(PrecipitationMaterial {
        params: PrecipitationParams::default(),
    });
    commands.insert_resource(PrecipitationHandle(material.clone()));
    commands.spawn((
        Name::new("Precipitation"),
        Mesh3d(meshes.add(particle_mesh(PARTICLES))),
        MeshMaterial3d(material),
        Transform::default(),
        // The particles are placed around the camera in the vertex shader.
        NoFrustumCulling,
        NotShadowCaster,
        NotShadowReceiver,
    ));
}

/// Samples the weather at the camera and steps the lightning.
pub fn update_weather(mut weather: ResMut<Weather>, camera: Res<PlanetCamera>, time: Res<Time>) {
    if !weather.ready() {
        return;
    }
    let t = time.elapsed_secs_wrapped() as f64;
    // Weather is what the clouds overhead do, so only below them.
    let below = smoothstep(
        CLOUD_HEIGHT_M as f64 + 500.0,
        CLOUD_HEIGHT_M as f64 - 1_000.0,
        camera.altitude_m,
    ) as f32;
    let v = weather.density(camera.dir, t);
    weather.cover = coverage_from_density(v) as f32;
    weather.precipitation = rain_from_density(v) as f32 * below;
    weather.storm = storm_from_density(v) as f32 * below;
    weather.snow = weather.temperature(camera.dir, camera.altitude_m) < 0.5;

    // Lightning: random strikes, more often in stronger storms; each is a double flicker.
    let dt = time.delta_secs();
    weather.since_strike += dt;
    weather.next_strike -= dt;
    if weather.storm > 0.05 && weather.next_strike <= 0.0 {
        weather.since_strike = 0.0;
        let r = weather.random();
        weather.next_strike = (2.0 + 10.0 * r) / weather.storm.max(0.1);
    }
    let s = weather.since_strike;
    let pulse = |start: f32, length: f32| {
        if s >= start && s < start + length {
            1.0 - (s - start) / length
        } else {
            0.0
        }
    };
    weather.flash = if weather.storm > 0.05 && !weather.snow {
        pulse(0.0, 0.12).max(0.7 * pulse(0.2, 0.25))
    } else {
        0.0
    };
}

/// Moves the precipitation with the camera and wind, dims the sun under storms, flashes
/// lightning and thickens the haze in rain.
pub fn apply_weather(
    weather: Res<Weather>,
    camera: Res<PlanetCamera>,
    time: Res<Time>,
    handle: Option<Res<PrecipitationHandle>>,
    mut materials: ResMut<Assets<PrecipitationMaterial>>,
    mut suns: Query<&mut DirectionalLight, With<Sun>>,
    mut ambient: ResMut<GlobalAmbientLight>,
    mut fogs: Query<&mut DistanceFog>,
    mut last_position: Local<Option<DVec3>>,
    mut drift: Local<DVec3>,
) {
    let (north, east, up) = camera.frame();
    let snow = weather.snow;
    // Wind from the same prevailing winds as the waves, stronger in storms.
    let wind_heading = super::ocean::wind_heading(camera.dir);
    let wind = (2.0 + 10.0 * weather.storm as f64) * if snow { 0.4 } else { 1.0 };
    let fall = if snow { 1.1 } else { 9.0 };
    let velocity = DVec3::new(wind * wind_heading.sin(), wind * wind_heading.cos(), -fall);
    let box_m = if snow { 24.0 } else { 30.0 };
    // Particles move with the air; the camera's own motion shifts them the other way.
    let position = camera.position();
    let moved = last_position.map_or(DVec3::ZERO, |last| position - last);
    *last_position = Some(position);
    let moved_local = DVec3::new(moved.dot(east), moved.dot(north), moved.dot(up));
    let dt = time.delta_secs_f64();
    *drift = ((*drift + velocity * dt - moved_local) / box_m).fract_gl() * box_m;

    if let Some(handle) = handle
        && let Some(mut material) = materials.get_mut(&handle.0)
    {
        material.params = PrecipitationParams {
            east: east.as_vec3(),
            box_m: box_m as f32,
            north: north.as_vec3(),
            intensity: weather.precipitation * if snow { 0.6 } else { 1.0 },
            up: up.as_vec3(),
            snow: if snow { 1.0 } else { 0.0 },
            velocity: velocity.as_vec3(),
            time: time.elapsed_secs_wrapped(),
            drift: drift.as_vec3(),
            _pad: 0.0,
        };
    }
    for mut sun in &mut suns {
        let dim = 1.0 - 0.65 * weather.storm - 0.2 * weather.precipitation;
        sun.illuminance = bevy::light::light_consts::lux::DIRECT_SUNLIGHT * dim;
    }
    // A lightning flash lights everything at once, from the whole sky.
    ambient.brightness = AMBIENT_BRIGHTNESS + 25_000.0 * weather.flash;
    ambient.color = Color::srgb(0.85, 0.9, 1.0).mix(&Color::WHITE, 1.0 - weather.flash);
    for mut fog in &mut fogs {
        // Rain and snow cut visibility from ~60 km down to a few km (snow even less).
        let heavy = weather.precipitation.max(weather.storm);
        let visibility = if snow {
            60_000.0 * (1.0 - heavy) + 1_200.0 * heavy
        } else {
            60_000.0 * (1.0 - heavy) + 3_000.0 * heavy
        };
        let grey = 0.55 - 0.3 * weather.storm;
        fog.color = Color::srgba(grey, grey + 0.02, grey + 0.05, heavy.min(1.0));
        fog.falloff = FogFalloff::from_visibility_squared(visibility);
    }
}

/// The camera's haze, thickened by [`apply_weather`] in rain and snow.
pub fn rain_and_haze_fog() -> DistanceFog {
    DistanceFog {
        color: Color::srgba(0.5, 0.52, 0.55, 0.0),
        directional_light_color: Color::NONE,
        directional_light_exponent: 8.0,
        falloff: FogFalloff::from_visibility_squared(60_000.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The CPU noise is the GPU's: spot values computed by hand from `common.wesl`.
    #[test]
    fn gpu_noise_is_deterministic_and_bounded() {
        for i in 0..2000 {
            let p = DVec3::new(
                i as f64 * 0.173,
                -(i as f64) * 0.071,
                3.0 + i as f64 * 0.013,
            );
            let n = gpu_gradient_noise(p, 1700);
            assert!(n.abs() < 1.7);
            assert_eq!(n, gpu_gradient_noise(p, 1700));
        }
        // Zero on lattice points.
        assert_eq!(gpu_gradient_noise(DVec3::new(3.0, -2.0, 5.0), 9), 0.0);
    }

    /// Rain covers a minority of the planet, storms a smaller part of that.
    #[test]
    fn rain_and_storms_are_regional() {
        let (mut rain, mut storm, n) = (0, 0, 20_000);
        for i in 0..n {
            // Fibonacci sphere.
            let y = 1.0 - 2.0 * (i as f64 + 0.5) / n as f64;
            let r = (1.0 - y * y).sqrt();
            let a = i as f64 * 2.399_963;
            let dir = DVec3::new(r * a.cos(), y, r * a.sin());
            let v = cloud_density(dir, 0.75, 0.0);
            rain += (rain_from_density(v) > 0.5) as u32;
            storm += (storm_from_density(v) > 0.5) as u32;
        }
        let (rain, storm) = (rain as f64 / n as f64, storm as f64 / n as f64);
        println!("rain {rain:.3}, storm {storm:.3}");
        assert!((0.04..0.35).contains(&rain), "rain {rain}");
        assert!((0.005..0.12).contains(&storm), "storm {storm}");
        assert!(storm < rain);
    }
}
