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
//! Interactive controls: drag to orbit, scroll to zoom, Space pause, R generate a new planet
//! with the next seed, Up/Down change simulation steps per frame.

use std::{path::PathBuf, time::Duration};

use bevy::{
    app::ScheduleRunnerPlugin,
    camera::RenderTarget,
    input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll},
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
    planet::Biome,
    planet::{self, PlanetPlugin},
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
                "-h" | "--help" => {
                    println!(
                        "planet-sim [--headless] [--steps N] [--steps-per-frame N] \
                         [--generation-steps N] [--generation-steps-per-frame N] \
                         [--cache DIR | --no-cache] [--regenerate] \
                         [--resolution N] [--seed N] [--out DIR] \
                         [--yaw RAD] [--pitch RAD] [--distance RADII]"
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
        .add_systems(Update, (orbit_camera, keyboard_controls, window_title));
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
        PlanetPlugin::default(),
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
    .add_systems(Startup, setup_scene)
    .run()
}

#[derive(Component)]
struct Orbit {
    yaw: f32,
    pitch: f32,
    distance: f32,
}

/// Headless render target, captured by `headless_capture`.
#[derive(Resource)]
struct CaptureTarget(Handle<Image>);

fn setup_scene(mut commands: Commands, args: Res<Args>, mut images: ResMut<Assets<Image>>) {
    commands.spawn((
        DirectionalLight {
            illuminance: 6_000.0,
            ..default()
        },
        Transform::from_xyz(4.0, 1.5, 2.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.insert_resource(GlobalAmbientLight {
        brightness: 25.0,
        ..default()
    });

    let (yaw, pitch, distance) = args.view;
    let orbit = Orbit {
        yaw,
        pitch,
        distance,
    };
    let mut camera = commands.spawn((Camera3d::default(), orbit_transform(&orbit), orbit));
    if args.headless {
        let mut image = Image::new_target_texture(1280, 720, TextureFormat::Rgba8UnormSrgb, None);
        image.texture_descriptor.usage |= bevy::render::render_resource::TextureUsages::COPY_SRC;
        let handle = images.add(image);
        camera.insert(RenderTarget::Image(handle.clone().into()));
        commands.insert_resource(CaptureTarget(handle));
    }
}

fn orbit_transform(orbit: &Orbit) -> Transform {
    let rot = Quat::from_euler(EulerRot::YXZ, orbit.yaw, -orbit.pitch, 0.0);
    Transform::from_translation(rot * Vec3::Z * orbit.distance).looking_at(Vec3::ZERO, Vec3::Y)
}

fn orbit_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut cameras: Query<(&mut Orbit, &mut Transform)>,
) {
    for (mut orbit, mut transform) in &mut cameras {
        if buttons.pressed(MouseButton::Left) {
            let speed = 0.005 * (orbit.distance - 1.0).clamp(0.05, 3.0);
            orbit.yaw -= motion.delta.x * speed;
            orbit.pitch = (orbit.pitch + motion.delta.y * speed).clamp(-1.5, 1.5);
        }
        if scroll.delta.y != 0.0 {
            let altitude = (orbit.distance - 1.0) * (1.0 - scroll.delta.y * 0.1);
            orbit.distance = 1.0 + altitude.clamp(0.02, 20.0);
        }
        *transform = orbit_transform(&orbit);
    }
}

fn keyboard_controls(keys: Res<ButtonInput<KeyCode>>, mut control: ResMut<SimControl>) {
    if keys.just_pressed(KeyCode::Space) {
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
    time: Res<Time>,
    mut window: Single<&mut Window>,
) {
    let fps = 1.0 / time.delta_secs().max(1e-6);
    window.title = if stats.generating() {
        format!(
            "planet-sim | seed {} | generating {}/{} | {fps:.0} fps",
            control.seed,
            stats.step(),
            control.generation_steps,
        )
    } else {
        format!(
            "planet-sim | seed {} | step {} | {} steps/frame{} | {fps:.0} fps",
            control.seed,
            stats.step(),
            control.steps_per_frame,
            if control.paused { " (paused)" } else { "" },
        )
    };
}

/// Headless run: wait for generation (or the cache load), the cache write and `--steps` live
/// steps, then save a screenshot and the field statistics (`stats.txt`), then exit.
fn headless_capture(
    mut commands: Commands,
    args: Res<Args>,
    stats: Res<SimStats>,
    fields: Option<Res<SimFields>>,
    cache: Res<SimCache>,
    target: Res<CaptureTarget>,
    mut control: ResMut<SimControl>,
    mut state: Local<(bool, u32)>,
    mut frames: Local<u64>,
    done: Option<ResMut<CaptureDone>>,
    mut exit: MessageWriter<AppExit>,
) {
    *frames += 1;
    let (requested, _) = *state;
    if !requested {
        if frames.is_multiple_of(60) {
            let phase = if stats.generating() {
                "generating"
            } else {
                "live"
            };
            info!("headless: frame {}, {phase} step {}", *frames, stats.step());
        }
        let saved = !cache.enabled() || cache.saved_generations() >= stats.generations_completed();
        if !stats.live() || !saved || stats.step() < args.steps {
            if *frames > 200_000 {
                error!("headless: simulation never became ready");
                exit.write(AppExit::error());
            }
            return;
        }
        let Some(fields) = fields else { return };
        // Freeze the simulation so the screenshot and readback show the same state.
        control.paused = true;
        std::fs::create_dir_all(&args.out).expect("create output dir");
        info!("headless: capturing at live step {}", stats.step());

        take_screenshot(&mut commands, &target, &args);

        let resolution = args.resolution;
        for name in REPORTED_FIELDS {
            commands
                .spawn(ReadbackOnce::texture(fields.front(name)))
                .observe(
                    move |event: On<ReadbackComplete>, mut done: ResMut<CaptureDone>| {
                        done.fields
                            .push((name, FieldStats::from_r32(&event.data, resolution)));
                    },
                );
        }
        commands.insert_resource(CaptureDone::default());
        state.0 = true;
        return;
    }
    state.1 += 1;
    if let Some(mut done) = done
        && done.retake
    {
        done.retake = false;
        take_screenshot(&mut commands, &target, &args);
    }
    // Give the async copies and pipeline compilation time; then give up.
    if state.1 > 3000 {
        error!("headless: capture timed out");
        exit.write(AppExit::error());
    }
}

/// Saves `planet.png` once the planet is on screen. A loaded planet is ready before its
/// render pipelines finish compiling, so blank frames are retaken.
fn take_screenshot(commands: &mut Commands, target: &CaptureTarget, args: &Args) {
    let mut save = save_to_disk(args.out.join("planet.png"));
    commands.spawn(Screenshot::image(target.0.clone())).observe(
        move |shot: On<ScreenshotCaptured>, mut done: ResMut<CaptureDone>| {
            let drawn = shot
                .image
                .data
                .as_ref()
                .is_some_and(|px| px.chunks(4).any(|p| p[..3].iter().any(|&c| c > 8)));
            if drawn {
                done.screenshot = true;
                save(shot);
            } else {
                done.retake = true;
            }
        },
    );
}

/// R32Float fields summarised at the end of a headless run.
const REPORTED_FIELDS: [&str; 6] = [
    planet::HEIGHT,
    planet::WATER,
    planet::DISCHARGE,
    planet::MOISTURE,
    planet::TEMPERATURE,
    planet::BIOME,
];

#[derive(Resource, Default)]
struct CaptureDone {
    screenshot: bool,
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
    if !done.screenshot || done.fields.len() < REPORTED_FIELDS.len() {
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
    for name in REPORTED_FIELDS {
        let s = get(name).unwrap();
        report += &format!(
            "{name:>10}: min {:>9.2}  max {:>9.2}  mean {:>9.3}  >0 {:>5.1}%  non-finite {}\n",
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
    /// Summarises an R32Float cube-sphere readback (rows padded to 256 bytes).
    fn from_r32(data: &[u8], resolution: u32) -> Self {
        let res = resolution as usize;
        let row = (res * 4).next_multiple_of(256);
        let mut s = FieldStats {
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            ..default()
        };
        for layer in 0..6 {
            for y in 0..res {
                let start = (layer * res + y) * row;
                let bytes = data
                    .get(start..start + res * 4)
                    .expect("readback shorter than the field");
                for v in bytes.as_chunks::<4>().0 {
                    let x = f32::from_le_bytes(*v);
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
