//! Procedural planet simulator tech demo.
//!
//! ```text
//! cargo run --release                         # interactive window
//! cargo run --release -- --headless --out out  # generate (or load) a planet, save a picture
//! ```
//!
//! The first start with a seed and resolution generates the planet (terrain, erosion,
//! moisture, temperature, biomes) and saves it under `cache/`; later starts load it.
//!
//! Once a planet exists it is drawn as level-of-detail terrain, from space down to eye height.
//!
//! Interactive controls: left-drag moves over the planet, right-drag looks around, scroll
//! changes altitude, WASD flies (Shift faster), Space pauses, R generates a new planet with
//! the next seed, Up/Down change simulation steps per frame.

use std::{path::PathBuf, time::Duration};

use bevy::{
    app::ScheduleRunnerPlugin,
    camera::RenderTarget,
    math::{DQuat, DVec3},
    prelude::*,
    render::{
        gpu_readback::{ReadbackComplete, ReadbackOnce},
        render_resource::TextureFormat,
        view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk},
    },
    window::ExitCondition,
    winit::WinitPlugin,
};

use planet_sim::{
    planet::{
        self, Biome, CameraControl, PlanetPlugin, camera,
        camera::PlanetCamera,
        lod::{LodSettings, TerrainLod},
        vegetation::Vegetation,
    },
    sim::{
        self, GpuSimPlugin, SimControl, SimFields, SimStats,
        cache::{CacheConfig, SimCache},
    },
};

#[derive(Resource, Clone, Debug)]
struct Args {
    headless: bool,
    /// Headless: live simulation steps to run after generation, before capturing.
    steps: u64,
    steps_per_frame: u32,
    generation_steps: u64,
    generation_steps_per_frame: u32,
    /// Planet cache directory; `None` disables the cache.
    cache: Option<PathBuf>,
    /// Ignore a cached planet and generate it again.
    regenerate: bool,
    resolution: u32,
    seed: u32,
    out: PathBuf,
    /// Initial camera orbit: yaw, pitch (radians) and distance (planet radii).
    view: (f32, f32, f32),
    /// Vertical exaggeration of the LOD terrain.
    exaggeration: f64,
    /// Headless: capture only these views (all when empty).
    views: Vec<String>,
}

impl Args {
    fn parse() -> Self {
        let mut args = Args {
            headless: false,
            steps: 0,
            steps_per_frame: 4,
            generation_steps: 3000,
            generation_steps_per_frame: 64,
            cache: Some(PathBuf::from("cache")),
            regenerate: false,
            resolution: 512,
            seed: 7,
            out: PathBuf::from("out"),
            view: (0.6, 0.35, 3.2),
            exaggeration: 1.0,
            views: Vec::new(),
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut value = || it.next().unwrap_or_else(|| panic!("{flag} needs a value"));
            match flag.as_str() {
                "--headless" => args.headless = true,
                "--steps" => args.steps = value().parse().expect("--steps"),
                "--steps-per-frame" => {
                    args.steps_per_frame = value().parse().expect("--steps-per-frame")
                }
                "--generation-steps" => {
                    args.generation_steps = value().parse().expect("--generation-steps")
                }
                "--generation-steps-per-frame" => {
                    args.generation_steps_per_frame =
                        value().parse().expect("--generation-steps-per-frame")
                }
                "--cache" => args.cache = Some(value().into()),
                "--no-cache" => args.cache = None,
                "--regenerate" => args.regenerate = true,
                "--resolution" => args.resolution = value().parse().expect("--resolution"),
                "--seed" => args.seed = value().parse().expect("--seed"),
                "--out" => args.out = value().into(),
                "--yaw" => args.view.0 = value().parse().expect("--yaw"),
                "--pitch" => args.view.1 = value().parse().expect("--pitch"),
                "--distance" => args.view.2 = value().parse().expect("--distance"),
                "--exaggeration" => args.exaggeration = value().parse().expect("--exaggeration"),
                "--views" => args.views = value().split(',').map(str::to_owned).collect(),
                "-h" | "--help" => {
                    println!(
                        "planet-sim [--headless] [--steps N] [--steps-per-frame N] \
                         [--generation-steps N] [--generation-steps-per-frame N] \
                         [--cache DIR | --no-cache] [--regenerate] \
                         [--resolution N] [--seed N] [--out DIR] \
                         [--yaw RAD] [--pitch RAD] [--distance RADII] [--exaggeration X] \\
                         [--views planet,orbit,surface,forest,meadow]"
                    );
                    std::process::exit(0);
                }
                other => panic!("unknown argument {other}; try --help"),
            }
        }
        args
    }
}

fn main() -> AppExit {
    let args = Args::parse();
    let mut app = App::new();

    if args.headless {
        app.add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::ZERO))
        .add_systems(Update, headless_capture)
        .add_systems(Last, exit_when_captured);
    } else {
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "planet-sim".into(),
                resolution: (1600, 900).into(),
                ..default()
            }),
            ..default()
        }))
        .add_systems(
            Update,
            (
                camera::camera_controls.in_set(CameraControl),
                keyboard_controls,
                window_title,
            ),
        );
    }

    app.add_plugins((
        GpuSimPlugin {
            resolution: args.resolution,
            cache: args.cache.clone().map(|dir| CacheConfig {
                dir,
                tag: planet::GENERATOR_VERSION.into(),
                reuse: !args.regenerate,
            }),
        },
        PlanetPlugin {
            terrain: LodSettings {
                exaggeration: args.exaggeration,
                ..default()
            },
            ..default()
        },
    ))
    .insert_resource({
        let mut control = SimControl::default();
        control.steps_per_frame = args.steps_per_frame;
        control.generation_steps = args.generation_steps;
        control.generation_steps_per_frame = args.generation_steps_per_frame;
        control.seed = args.seed;
        control
    })
    .insert_resource(ClearColor(Color::BLACK))
    .insert_resource(args)
    .add_systems(Startup, (setup_scene, set_initial_view))
    .run()
}

/// Headless render target, captured by `headless_capture`.
#[derive(Resource)]
struct CaptureTarget(Handle<Image>);

fn setup_scene(mut commands: Commands, args: Res<Args>, mut images: ResMut<Assets<Image>>) {
    commands.spawn((
        DirectionalLight {
            illuminance: bevy::light::light_consts::lux::DIRECT_SUNLIGHT,
            ..default()
        },
        Transform::from_xyz(4.0, 1.5, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    // Physical units: full sunlight and a daylight exposure. Skylight comes from the
    // atmosphere's environment map; a little ambient keeps the night side from going black.
    commands.insert_resource(GlobalAmbientLight {
        brightness: 300.0,
        ..default()
    });

    // The render camera stays at the origin (floating origin, see `planet::camera`).
    let mut camera = commands.spawn((
        Camera3d::default(),
        bevy::camera::Hdr,
        bevy::camera::Exposure::SUNLIGHT,
        bevy::pbr::AtmosphereSettings::default(),
        bevy::light::AtmosphereEnvironmentMapLight::default(),
        Projection::Perspective(PerspectiveProjection {
            near: 0.3,
            far: 1.0e9,
            ..default()
        }),
    ));
    if args.headless {
        let mut image = Image::new_target_texture(1280, 720, TextureFormat::Rgba8UnormSrgb, None);
        image.texture_descriptor.usage |= bevy::render::render_resource::TextureUsages::COPY_SRC;
        let handle = images.add(image);
        camera.insert(RenderTarget::Image(handle.clone().into()));
        commands.insert_resource(CaptureTarget(handle));
    }
}

/// The starting view: looking straight down at the planet from `--distance` radii, from the
/// direction given by `--yaw` and `--pitch`.
fn initial_view(args: &Args, radius_m: f64) -> PlanetCamera {
    let (yaw, pitch, distance) = args.view;
    let rot = DQuat::from_euler(EulerRot::YXZ, yaw as f64, -pitch as f64, 0.0);
    let mut cam = PlanetCamera::new(radius_m);
    cam.dir = rot * DVec3::Z;
    cam.altitude_m = (distance as f64 - 1.0) * radius_m;
    cam
}

fn set_initial_view(args: Res<Args>, mut camera: ResMut<PlanetCamera>) {
    *camera = initial_view(&args, camera.radius_m);
}

fn keyboard_controls(
    keys: Res<ButtonInput<KeyCode>>,
    camera: Res<PlanetCamera>,
    mut control: ResMut<SimControl>,
) {
    // Space jumps when walking; P pauses in every mode.
    let pause = keys.just_pressed(KeyCode::KeyP)
        || (keys.just_pressed(KeyCode::Space) && camera.mode != camera::CameraMode::Walk);
    if pause {
        control.paused = !control.paused;
    }
    if keys.just_pressed(KeyCode::KeyR) {
        control.seed = control.seed.wrapping_add(1);
        control.request_reset();
    }
    if keys.just_pressed(KeyCode::ArrowUp) {
        control.steps_per_frame = (control.steps_per_frame * 2).clamp(1, sim::MAX_STEPS_PER_FRAME);
    }
    if keys.just_pressed(KeyCode::ArrowDown) {
        control.steps_per_frame = (control.steps_per_frame / 2).max(1);
    }
}

fn window_title(
    stats: Res<SimStats>,
    control: Res<SimControl>,
    camera: Res<PlanetCamera>,
    lod: Res<TerrainLod>,
    time: Res<Time>,
    mut window: Single<&mut Window>,
) {
    let fps = 1.0 / time.delta_secs().max(1e-6);
    let above = camera.height_above_ground();
    let altitude = if above >= 10_000.0 {
        format!("{:.0} km", above / 1000.0)
    } else {
        format!("{above:.0} m")
    };
    window.title = if stats.generating() {
        format!(
            "planet-sim | seed {} | generating {}/{} | {fps:.0} fps",
            control.seed,
            stats.step(),
            control.generation_steps,
        )
    } else {
        format!(
            "planet-sim | seed {} | step {} | {} steps/frame{} | {} | {altitude} up | {} chunks | {fps:.0} fps",
            control.seed,
            stats.step(),
            control.steps_per_frame,
            if control.paused { " (paused)" } else { "" },
            camera.mode.name(),
            lod.drawn(),
        )
    };
}

/// Headless run: wait for generation (or the cache load), the cache write and `--steps` live
/// steps; then save one screenshot per view in [`shots`] (each once its terrain detail has
/// streamed in) and the field statistics (`stats.txt`), then exit.
fn headless_capture(
    mut commands: Commands,
    args: Res<Args>,
    stats: Res<SimStats>,
    fields: Option<Res<SimFields>>,
    cache: Res<SimCache>,
    target: Res<CaptureTarget>,
    lod: Res<TerrainLod>,
    vegetation: Res<Vegetation>,
    mut camera: ResMut<PlanetCamera>,
    mut control: ResMut<SimControl>,
    mut frames: Local<u64>,
    mut wait: Local<u32>,
    done: Option<ResMut<CaptureDone>>,
    mut exit: MessageWriter<AppExit>,
) {
    *frames += 1;
    let Some(mut done) = done else {
        if frames.is_multiple_of(60) {
            let phase = if stats.generating() {
                "generating"
            } else {
                "live"
            };
            info!("headless: frame {}, {phase} step {}", *frames, stats.step());
        }
        if stats.failed() {
            error!("headless: a simulation pass failed to compile");
            exit.write(AppExit::error());
            return;
        }
        let saved = cache.saved_generations() >= stats.generations_completed();
        if !stats.live() || !saved || stats.step() < args.steps || lod.surface().is_none() {
            if *frames > 200_000 {
                error!("headless: simulation never became ready");
                exit.write(AppExit::error());
            }
            return;
        }
        let Some(fields) = fields else { return };
        // Freeze the simulation so the screenshots and readback show the same state.
        control.paused = true;
        std::fs::create_dir_all(&args.out).expect("create output dir");
        info!("headless: capturing at live step {}", stats.step());

        let resolution = args.resolution;
        for (field, channels) in REPORTED_FIELDS {
            commands
                .spawn(ReadbackOnce::texture(fields.front(field)))
                .observe(
                    move |event: On<ReadbackComplete>, mut done: ResMut<CaptureDone>| {
                        for (c, &name) in channels.iter().enumerate() {
                            let s =
                                FieldStats::from_texels(&event.data, resolution, channels.len(), c);
                            done.fields.push((name, s));
                        }
                    },
                );
        }
        let surface = lod.surface().unwrap();
        commands.insert_resource(CaptureDone {
            shots: shots(&initial_view(&args, camera.radius_m), surface, &vegetation)
                .into_iter()
                .filter(|(name, _)| args.views.is_empty() || args.views.iter().any(|v| v == name))
                .collect(),
            ..default()
        });
        return;
    };

    let Some((name, view)) = done.shots.get(done.taken).cloned() else {
        return;
    };
    *camera = view;
    *wait += 1;
    if done.retake {
        done.retake = false;
        take_screenshot(&mut commands, &target, &args, name);
    } else if !done.shooting && *wait > 5 && lod.settled() && vegetation.settled() {
        info!(
            "headless: {name} view, {} chunks, {} vegetation tiles",
            lod.drawn(),
            vegetation.drawn()
        );
        done.shooting = true;
        take_screenshot(&mut commands, &target, &args, name);
    }
    // Detail streaming, async copies and pipeline compilation take a while on a CPU driver.
    if *wait > 20_000 {
        error!("headless: capture of the {name} view timed out");
        exit.write(AppExit::error());
    }
    if done.shooting && done.screenshot_saved {
        done.shooting = false;
        done.screenshot_saved = false;
        done.taken += 1;
        *wait = 0;
    }
}

/// Views captured by a headless run: the whole planet from `start`, then low orbit and eye
/// height over the first patch of hilly land near the middle of that view.
fn shots(
    start: &PlanetCamera,
    surface: &planet::surface::PlanetSurface,
    vegetation: &Vegetation,
) -> Vec<(&'static str, PlanetCamera)> {
    let (north, east, up) = start.frame();
    // The nearest point to the middle of the view that passes `test`.
    let find = |test: &dyn Fn(DVec3) -> bool| -> DVec3 {
        for ring in 0..120 {
            let radius = ring as f64 * 0.01;
            for k in 0..(ring * 6).max(1) {
                let a = k as f64 / (ring * 6).max(1) as f64 * std::f64::consts::TAU;
                let dir = (up + (north * a.cos() + east * a.sin()) * radius).normalize();
                if test(dir) {
                    return dir;
                }
            }
        }
        up
    };
    let target = find(&|d| (700.0..2500.0).contains(&surface.height.sample(d)));
    let mut orbit = *start;
    orbit.dir = target;
    orbit.altitude_m = 600_000.0;
    orbit.pitch = -0.6;
    orbit.heading = 0.5;
    // Stand back from the target so it is in view.
    orbit.move_by(-700_000.0, 0.0);
    let mut ground = *start;
    ground.dir = target;
    ground.heading = 0.5;
    ground.pitch = -0.12;
    ground.altitude_m = surface.height(target, 1.0).max(0.0) + 40.0;

    // Low over a forest, and standing in a meadow.
    // Inland lowland where the test holds for a few km all around.
    let around = |d: DVec3, test: &dyn Fn(DVec3) -> bool| {
        let (n, e) = (north - d * north.dot(d), east - d * east.dot(d));
        (0..8).all(|k| {
            let a = k as f64 * std::f64::consts::FRAC_PI_4;
            let p = (d + (n * a.cos() + e * a.sin()) * 0.0008).normalize();
            (150.0..1200.0).contains(&surface.height(p, 50.0)) && test(p)
        })
    };
    let forest_dir = find(&|d| {
        around(d, &|p| {
            vegetation.allowed_cover(p).is_some_and(|c| c[2] > 0.95)
        })
    });
    let mut forest = *start;
    forest.dir = forest_dir;
    forest.heading = 1.2;
    forest.pitch = -0.18;
    forest.altitude_m = surface.height(forest_dir, 1.0).max(0.0) + 45.0;
    let meadow_dir = find(&|d| {
        around(d, &|p| {
            vegetation
                .allowed_cover(p)
                .is_some_and(|c| c[0] > 0.95 && c[2] < 0.6)
        })
    });
    let mut meadow = *start;
    meadow.dir = meadow_dir;
    meadow.heading = 2.0;
    meadow.pitch = -0.1;
    meadow.altitude_m = surface.height(meadow_dir, 1.0).max(0.0) + 1.7;
    // The other camera modes: flying low over the land, and walking on it.
    let mut fly = *start;
    fly.dir = target;
    fly.heading = 2.2;
    fly.pitch = -0.25;
    fly.altitude_m = surface.height(target, 1.0).max(0.0) + 600.0;
    fly.mode = camera::CameraMode::Fly;
    let mut walk = fly;
    walk.pitch = 0.02;
    walk.altitude_m = surface.height(target, 1.0).max(0.0) + camera::WALK_EYE_HEIGHT_M;
    walk.mode = camera::CameraMode::Walk;
    vec![
        ("planet", *start),
        ("orbit", orbit),
        ("surface", ground),
        ("forest", forest),
        ("meadow", meadow),
        ("fly", fly),
        ("walk", walk),
    ]
}

/// Saves `<name>.png` once the planet is on screen. A loaded planet is ready before its
/// render pipelines finish compiling, so blank frames are retaken.
fn take_screenshot(commands: &mut Commands, target: &CaptureTarget, args: &Args, name: &str) {
    let mut save = save_to_disk(args.out.join(format!("{name}.png")));
    commands.spawn(Screenshot::image(target.0.clone())).observe(
        move |shot: On<ScreenshotCaptured>, mut done: ResMut<CaptureDone>| {
            let drawn = shot
                .image
                .data
                .as_ref()
                .is_some_and(|px| px.chunks(4).any(|p| p[..3].iter().any(|&c| c > 8)));
            if drawn {
                done.screenshot_saved = true;
                save(shot);
            } else {
                done.retake = true;
            }
        },
    );
}

/// Fields summarised at the end of a headless run, with a report name per channel.
const REPORTED_FIELDS: [(&str, &[&str]); 8] = [
    (planet::HEIGHT, &[planet::HEIGHT]),
    (planet::WATER, &[planet::WATER]),
    (planet::DISCHARGE, &[planet::DISCHARGE]),
    (planet::MOISTURE, &[planet::MOISTURE]),
    (planet::TEMPERATURE, &[planet::TEMPERATURE]),
    (planet::BIOME, &[planet::BIOME]),
    (planet::PLANTS, &["grass", "shrubs", "trees", "fuel"]),
    (planet::FIRE, &[planet::FIRE]),
];

fn reported_channels() -> usize {
    REPORTED_FIELDS.iter().map(|(_, c)| c.len()).sum()
}

#[derive(Resource, Default)]
struct CaptureDone {
    shots: Vec<(&'static str, PlanetCamera)>,
    /// Screenshots saved so far.
    taken: usize,
    /// A screenshot of the current view is in flight.
    shooting: bool,
    screenshot_saved: bool,
    /// The last screenshot was blank (planet pipelines still compiling): take another.
    retake: bool,
    fields: Vec<(&'static str, FieldStats)>,
}

fn exit_when_captured(
    done: Option<Res<CaptureDone>>,
    args: Res<Args>,
    stats: Res<SimStats>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some(done) = done else { return };
    if done.shots.is_empty()
        || done.taken < done.shots.len()
        || done.fields.len() < reported_channels()
    {
        return;
    }
    let get = |name: &str| done.fields.iter().find(|(n, _)| *n == name).map(|(_, s)| s);
    let origin = if stats.generations_completed() == 0 {
        "loaded from cache".to_string()
    } else {
        format!("generated in {} steps", args.generation_steps)
    };
    let mut report = format!(
        "planet seed {}, {r}x{r}x6 texels, {origin}; live step {}\n",
        args.seed,
        stats.step(),
        r = args.resolution
    );
    for name in REPORTED_FIELDS.iter().flat_map(|(_, c)| c.iter()) {
        let s = get(name).unwrap();
        report += &format!(
            "{name:>10}: min {:>9.2}  max {:>9.2}  mean {:>9.3}  >0 {:>6.2}%  non-finite {}\n",
            s.min,
            s.max,
            s.mean(),
            100.0 * s.positive as f64 / s.count.max(1) as f64,
            s.non_finite
        );
    }
    let biomes = &get(planet::BIOME).unwrap().histogram;
    let total: u64 = biomes.iter().sum();
    report += "biomes:";
    for biome in Biome::ALL {
        let n = biomes[biome as usize];
        report += &format!(
            " {} {:.1}%,",
            biome.name(),
            100.0 * n as f64 / total.max(1) as f64
        );
    }
    report.pop();
    report.push('\n');
    for line in report.lines() {
        info!("headless: {line}");
    }
    std::fs::write(args.out.join("stats.txt"), report).ok();
    let bad: u64 = done.fields.iter().map(|(_, s)| s.non_finite).sum();
    exit.write(if bad == 0 {
        AppExit::Success
    } else {
        error!("headless: {bad} non-finite texels");
        AppExit::error()
    });
}

#[derive(Debug, Default)]
struct FieldStats {
    min: f32,
    max: f32,
    sum: f64,
    count: u64,
    positive: u64,
    non_finite: u64,
    /// Texels per rounded value 0..16 (biome classes).
    histogram: [u64; 16],
}

impl FieldStats {
    /// Summarises one channel of a 32-bit float cube-sphere readback with `channels`
    /// channels per texel (rows padded to 256 bytes).
    fn from_texels(data: &[u8], resolution: u32, channels: usize, channel: usize) -> Self {
        let res = resolution as usize;
        let texel = 4 * channels;
        let row = (res * texel).next_multiple_of(256);
        let mut s = FieldStats {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            ..default()
        };
        for layer in 0..6 {
            for y in 0..res {
                let start = (layer * res + y) * row;
                let bytes = data
                    .get(start..start + res * texel)
                    .expect("readback shorter than the field");
                for t in bytes.chunks_exact(texel) {
                    let v = &t[channel * 4..channel * 4 + 4];
                    let x = f32::from_le_bytes(v.try_into().unwrap());
                    if !x.is_finite() {
                        s.non_finite += 1;
                        continue;
                    }
                    s.min = s.min.min(x);
                    s.max = s.max.max(x);
                    s.sum += x as f64;
                    s.positive += (x > 0.0) as u64;
                    if (0.0..16.0).contains(&x.round()) {
                        s.histogram[x.round() as usize] += 1;
                    }
                    s.count += 1;
                }
            }
        }
        s
    }

    fn mean(&self) -> f64 {
        self.sum / self.count.max(1) as f64
    }
}
