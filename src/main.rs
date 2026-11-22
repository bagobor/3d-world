//! Procedural planet simulator tech demo.
//!
//! ```text
//! cargo run --release                         # interactive window
//! cargo run --release -- --headless --steps 300 --out out
//! ```
//!
//! Interactive controls: drag to orbit, scroll to zoom, Space pause, R regenerate with a new
//! seed, Up/Down change simulation steps per frame.

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
    planet::{self, PlanetPlugin},
    sim::{self, GpuSimPlugin, SimControl, SimFields, SimStats},
};

#[derive(Resource, Clone, Debug)]
struct Args {
    headless: bool,
    /// Headless: simulation steps to run before capturing.
    steps: u64,
    steps_per_frame: u32,
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
            steps: 200,
            steps_per_frame: 4,
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
                "--resolution" => args.resolution = value().parse().expect("--resolution"),
                "--seed" => args.seed = value().parse().expect("--seed"),
                "--out" => args.out = value().into(),
                "--yaw" => args.view.0 = value().parse().expect("--yaw"),
                "--pitch" => args.view.1 = value().parse().expect("--pitch"),
                "--distance" => args.view.2 = value().parse().expect("--distance"),
                "-h" | "--help" => {
                    println!(
                        "planet-sim [--headless] [--steps N] [--steps-per-frame N] \
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
        },
        PlanetPlugin::default(),
    ))
    .insert_resource({
        let mut control = SimControl::default();
        control.steps_per_frame = args.steps_per_frame;
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
    window.title = format!(
        "planet-sim | seed {} | step {} | {} steps/frame{} | {:.0} fps",
        control.seed,
        stats.step(),
        control.steps_per_frame,
        if control.paused { " (paused)" } else { "" },
        1.0 / time.delta_secs().max(1e-6),
    );
}

/// Headless run: wait for `--steps` simulation steps, then save a screenshot and the
/// height field statistics, then exit.
fn headless_capture(
    mut commands: Commands,
    args: Res<Args>,
    stats: Res<SimStats>,
    fields: Option<Res<SimFields>>,
    target: Res<CaptureTarget>,
    mut control: ResMut<SimControl>,
    mut state: Local<(bool, u32)>,
    mut frames: Local<u64>,
    mut exit: MessageWriter<AppExit>,
) {
    *frames += 1;
    let (requested, _) = *state;
    if !requested {
        if frames.is_multiple_of(60) {
            info!("headless: frame {}, step {}", *frames, stats.step());
        }
        if !stats.ready() || stats.step() < args.steps {
            if *frames > 20_000 {
                error!("headless: simulation never became ready");
                exit.write(AppExit::error());
            }
            return;
        }
        let Some(fields) = fields else { return };
        // Freeze the simulation so the screenshot and readback show the same state.
        control.paused = true;
        std::fs::create_dir_all(&args.out).expect("create output dir");
        info!("headless: capturing at step {}", stats.step());

        let shot = args.out.join("planet.png");
        commands
            .spawn(Screenshot::image(target.0.clone()))
            .observe(save_to_disk(shot))
            .observe(|_: On<ScreenshotCaptured>, mut done: ResMut<CaptureDone>| {
                done.screenshot = true;
            });

        let resolution = args.resolution;
        let out = args.out.clone();
        commands
            .spawn(ReadbackOnce::texture(fields.front(planet::HEIGHT)))
            .observe(
                move |event: On<ReadbackComplete>, mut done: ResMut<CaptureDone>| {
                    let report = height_report(&event.data, resolution);
                    info!("headless: {report}");
                    std::fs::write(out.join("height_stats.txt"), report + "\n").ok();
                    done.readback = true;
                },
            );
        commands.insert_resource(CaptureDone::default());
        state.0 = true;
        return;
    }
    state.1 += 1;
    // Give the async copies a few frames; then exit.
    if state.1 > 600 {
        error!("headless: capture timed out");
        exit.write(AppExit::error());
    }
}

#[derive(Resource, Default)]
struct CaptureDone {
    screenshot: bool,
    readback: bool,
}

fn exit_when_captured(done: Option<Res<CaptureDone>>, mut exit: MessageWriter<AppExit>) {
    if let Some(done) = done
        && done.screenshot
        && done.readback
    {
        exit.write(AppExit::Success);
    }
}

/// Summarises an R32Float cube-sphere readback (rows padded to 256 bytes).
fn height_report(data: &[u8], resolution: u32) -> String {
    let row = (resolution as usize * 4).next_multiple_of(256);
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let (mut sum, mut land, mut count, mut bad) = (0.0f64, 0u64, 0u64, 0u64);
    for layer in 0..6 {
        for y in 0..resolution as usize {
            let start = (layer * resolution as usize + y) * row;
            let Some(bytes) = data.get(start..start + resolution as usize * 4) else {
                return format!("readback too short: {} bytes", data.len());
            };
            for v in bytes.as_chunks::<4>().0 {
                let h = f32::from_le_bytes(*v);
                if !h.is_finite() {
                    bad += 1;
                    continue;
                }
                min = min.min(h);
                max = max.max(h);
                sum += h as f64;
                land += (h > 0.0) as u64;
                count += 1;
            }
        }
    }
    format!(
        "height field {resolution}x{resolution}x6: min {min:.0} m, max {max:.0} m, mean {:.0} m, \
         land {:.1}%, non-finite texels {bad}",
        sum / count.max(1) as f64,
        100.0 * land as f64 / count.max(1) as f64
    )
}
