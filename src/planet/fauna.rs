//! Individual animals near the camera, drawn from the live population fields.
//!
//! The simulation tracks grazers and predators as planet-wide densities (the `animals`
//! field). Around the camera, this module turns them into individuals: each cell of a fixed
//! quadtree level near the camera gets herds of grazers and packs of predators in proportion
//! to the local densities (read back from the GPU every few seconds), deterministically per
//! cell. Animals then behave on the CPU: herds graze, wander and keep together, and flee
//! from predators; predators roam in packs, chase the nearest grazer in sight and stop to
//! eat when they catch one. Cells far from the camera are dropped with their animals.
//!
//! Animals are low-poly, built from Bevy primitives, with legs that swing with their speed.

use bevy::{
    math::DVec3,
    platform::collections::{HashMap, HashSet},
    prelude::*,
    render::gpu_readback::{ReadbackComplete, ReadbackOnce},
};

use super::{
    ANIMALS,
    camera::PlanetCamera,
    lod::{NodeId, TerrainLod},
    surface::{dir_to_face, dir_to_face_st},
};
use crate::sim::{SimFields, SimStats};

/// Quadtree level of spawn cells (~300 m).
const CELL_LEVEL: u8 = 15;
/// Cells within this distance of the camera are populated...
const SPAWN_RADIUS_M: f64 = 1_200.0;
/// ...and dropped beyond this one.
const DESPAWN_RADIUS_M: f64 = 1_700.0;
/// Seconds between density readbacks.
const READBACK_PERIOD_S: f32 = 5.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Species {
    Grazer,
    Predator,
}

impl Species {
    fn walk_speed(self) -> f64 {
        match self {
            Species::Grazer => 1.3,
            Species::Predator => 1.8,
        }
    }

    fn run_speed(self) -> f64 {
        match self {
            Species::Grazer => 13.0,
            Species::Predator => 11.5,
        }
    }

    /// Distance covered by one full stride, for the leg animation.
    fn stride_m(self) -> f64 {
        match self {
            Species::Grazer => 1.6,
            Species::Predator => 1.3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Activity {
    /// Standing (grazing or resting) until the timer runs out.
    Idle(f64),
    /// Walking somewhere for a while.
    Walk(f64),
    /// Running from or after something.
    Run,
    /// A predator eating its catch.
    Eat(f64),
}

struct Animal {
    species: Species,
    /// Unit direction from the planet centre.
    dir: DVec3,
    /// Unit tangent the animal faces.
    facing: DVec3,
    speed: f64,
    activity: Activity,
    /// Herd or pack id; members keep together.
    group: u32,
    home: NodeId,
    /// Leg cycle phase, radians.
    stride: f64,
    entity: Entity,
    legs: [Entity; 4],
    rng: u64,
}

/// Grazer and predator densities read back from the GPU, RGBA per texel.
#[derive(Default)]
struct Densities {
    resolution: usize,
    texels: Vec<[f32; 4]>,
}

impl Densities {
    fn at(&self, dir: DVec3) -> (f64, f64) {
        if self.texels.is_empty() {
            return (0.0, 0.0);
        }
        let face = dir_to_face(dir);
        let st = dir_to_face_st(face, dir);
        let res = self.resolution;
        let c = |v: f64| (((v * 0.5 + 0.5) * res as f64) as usize).min(res - 1);
        let t = self.texels[(face * res + c(st.y)) * res + c(st.x)];
        (t[0] as f64, t[1] as f64)
    }
}

/// Meshes and materials shared by all animals.
struct Assets3d {
    grazer: Vec<(Handle<Mesh>, Handle<StandardMaterial>, Transform)>,
    grazer_leg: (Handle<Mesh>, Handle<StandardMaterial>),
    predator: Vec<(Handle<Mesh>, Handle<StandardMaterial>, Transform)>,
    predator_leg: (Handle<Mesh>, Handle<StandardMaterial>),
}

#[derive(Resource, Default)]
pub struct Fauna {
    densities: Densities,
    /// Readbacks received so far.
    readbacks: u32,
    readback_in_flight: bool,
    since_readback: f32,
    cells: HashSet<NodeId>,
    animals: Vec<Animal>,
    next_group: u32,
    assets: Option<Assets3d>,
    /// Every cell the camera wants is populated.
    settled: bool,
}

impl Fauna {
    /// Densities have arrived and the cells around the camera are populated.
    pub fn ready(&self) -> bool {
        self.readbacks > 0 && self.settled
    }

    pub fn count(&self, species: Species) -> usize {
        self.animals.iter().filter(|a| a.species == species).count()
    }

    /// Centre of the grazer herd nearest to `position` (planet-centred metres), as a unit
    /// direction.
    pub fn nearest_herd(&self, position: DVec3) -> Option<DVec3> {
        let mut groups: HashMap<u32, (DVec3, usize)> = HashMap::new();
        for a in self.animals.iter().filter(|a| a.species == Species::Grazer) {
            let e = groups.entry(a.group).or_insert((DVec3::ZERO, 0));
            e.0 += a.dir;
            e.1 += 1;
        }
        let up = position.normalize();
        groups
            .values()
            .filter(|(_, n)| *n >= 3)
            .map(|(sum, _)| sum.normalize())
            .min_by(|a, b| a.angle_between(up).total_cmp(&b.angle_between(up)))
    }
}

fn rand(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

/// A random unit tangent at `dir`.
fn random_tangent(dir: DVec3, rng: &mut u64) -> DVec3 {
    let a = rand(rng) * std::f64::consts::TAU;
    let (n, e) = frame(dir);
    n * a.cos() + e * a.sin()
}

/// North and east at `dir`.
fn frame(dir: DVec3) -> (DVec3, DVec3) {
    let pole = if dir.y.abs() > 0.999 {
        DVec3::X
    } else {
        DVec3::Y
    };
    let north = (pole - dir * pole.dot(dir)).normalize();
    (north, north.cross(dir))
}

fn build_assets(meshes: &mut Assets<Mesh>, materials: &mut Assets<StandardMaterial>) -> Assets3d {
    let fur = |c: Color, materials: &mut Assets<StandardMaterial>| {
        materials.add(StandardMaterial {
            base_color: c,
            perceptual_roughness: 0.9,
            reflectance: 0.2,
            ..default()
        })
    };
    // Grazers: a deer-like body, tan with a pale belly, dark legs.
    let tan = fur(Color::srgb(0.42, 0.28, 0.15), materials);
    let pale = fur(Color::srgb(0.62, 0.52, 0.4), materials);
    let dark = fur(Color::srgb(0.18, 0.12, 0.08), materials);
    // Bevy's forward is -Z; capsules are along Y, so lay them down with a rotation.
    let lay = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2);
    let grazer = vec![
        (
            meshes.add(Capsule3d::new(0.27, 0.75)),
            tan.clone(),
            Transform::from_xyz(0.0, 0.95, 0.0).with_rotation(lay),
        ),
        (
            meshes.add(Capsule3d::new(0.2, 0.5)),
            pale,
            Transform::from_xyz(0.0, 0.86, 0.0).with_rotation(lay),
        ),
        (
            meshes.add(Capsule3d::new(0.09, 0.42)),
            tan.clone(),
            Transform::from_xyz(0.0, 1.28, -0.55).with_rotation(Quat::from_rotation_x(-0.6)),
        ),
        (
            meshes.add(Cuboid::new(0.16, 0.18, 0.36)),
            tan.clone(),
            Transform::from_xyz(0.0, 1.5, -0.78).with_rotation(Quat::from_rotation_x(0.35)),
        ),
        (
            meshes.add(Cuboid::new(0.05, 0.16, 0.04)),
            dark.clone(),
            Transform::from_xyz(0.08, 1.66, -0.68),
        ),
        (
            meshes.add(Cuboid::new(0.05, 0.16, 0.04)),
            dark.clone(),
            Transform::from_xyz(-0.08, 1.66, -0.68),
        ),
        (
            meshes.add(Cuboid::new(0.08, 0.08, 0.18)),
            pale_tail(materials),
            Transform::from_xyz(0.0, 1.05, 0.62).with_rotation(Quat::from_rotation_x(0.5)),
        ),
    ];
    let grazer_leg = (meshes.add(Cuboid::new(0.07, 0.82, 0.07)), dark);

    // Predators: a wolf-like body, grey.
    let grey = fur(Color::srgb(0.3, 0.29, 0.27), materials);
    let light = fur(Color::srgb(0.55, 0.53, 0.5), materials);
    let predator = vec![
        (
            meshes.add(Capsule3d::new(0.2, 0.6)),
            grey.clone(),
            Transform::from_xyz(0.0, 0.62, 0.0).with_rotation(lay),
        ),
        (
            meshes.add(Capsule3d::new(0.15, 0.2)),
            light,
            Transform::from_xyz(0.0, 0.72, -0.42).with_rotation(lay),
        ),
        (
            meshes.add(Cuboid::new(0.2, 0.2, 0.3)),
            grey.clone(),
            Transform::from_xyz(0.0, 0.8, -0.62),
        ),
        (
            meshes.add(Cuboid::new(0.1, 0.1, 0.18)),
            grey.clone(),
            Transform::from_xyz(0.0, 0.75, -0.82),
        ),
        (
            meshes.add(Cuboid::new(0.05, 0.12, 0.04)),
            grey.clone(),
            Transform::from_xyz(0.07, 0.95, -0.58),
        ),
        (
            meshes.add(Cuboid::new(0.05, 0.12, 0.04)),
            grey.clone(),
            Transform::from_xyz(-0.07, 0.95, -0.58),
        ),
        (
            meshes.add(Capsule3d::new(0.06, 0.4)),
            grey.clone(),
            Transform::from_xyz(0.0, 0.55, 0.62).with_rotation(Quat::from_rotation_x(1.0)),
        ),
    ];
    let predator_leg = (meshes.add(Cuboid::new(0.07, 0.52, 0.07)), grey);
    Assets3d {
        grazer,
        grazer_leg,
        predator,
        predator_leg,
    }
}

fn pale_tail(materials: &mut Assets<StandardMaterial>) -> Handle<StandardMaterial> {
    materials.add(StandardMaterial {
        base_color: Color::srgb(0.85, 0.82, 0.76),
        perceptual_roughness: 0.9,
        reflectance: 0.2,
        ..default()
    })
}

fn spawn_animal(
    commands: &mut Commands,
    assets: &Assets3d,
    species: Species,
    dir: DVec3,
    group: u32,
    home: NodeId,
    mut rng: u64,
) -> Animal {
    let (parts, leg, hip_y, hips) = match species {
        Species::Grazer => (
            &assets.grazer,
            &assets.grazer_leg,
            0.82,
            [(0.15, -0.42), (-0.15, -0.42), (0.15, 0.4), (-0.15, 0.4)],
        ),
        Species::Predator => (
            &assets.predator,
            &assets.predator_leg,
            0.55,
            [(0.12, -0.36), (-0.12, -0.36), (0.12, 0.34), (-0.12, 0.34)],
        ),
    };
    let size = 0.85 + 0.3 * rand(&mut rng) as f32;
    let entity = commands
        .spawn((
            Name::new(format!("{species:?}")),
            Transform::from_scale(Vec3::splat(size)),
            Visibility::Inherited,
        ))
        .id();
    for (mesh, material, transform) in parts {
        commands.spawn((
            Mesh3d(mesh.clone()),
            MeshMaterial3d(material.clone()),
            *transform,
            ChildOf(entity),
        ));
    }
    let leg_len = match species {
        Species::Grazer => 0.82,
        Species::Predator => 0.52,
    };
    let legs = hips.map(|(x, z)| {
        // A pivot at the hip, with the leg hanging below it.
        let pivot = commands
            .spawn((
                Transform::from_xyz(x, hip_y, z),
                Visibility::Inherited,
                ChildOf(entity),
            ))
            .id();
        commands.spawn((
            Mesh3d(leg.0.clone()),
            MeshMaterial3d(leg.1.clone()),
            Transform::from_xyz(0.0, -leg_len * 0.5, 0.0),
            ChildOf(pivot),
        ));
        pivot
    });
    let facing = random_tangent(dir, &mut rng);
    let idle = 2.0 + 8.0 * rand(&mut rng);
    Animal {
        species,
        dir,
        facing,
        speed: 0.0,
        activity: Activity::Idle(idle),
        group,
        home,
        stride: rand(&mut rng) * std::f64::consts::TAU,
        entity,
        legs,
        rng,
    }
}

/// Cells of [`CELL_LEVEL`] within `radius` of `cam`.
fn cells_near(cam: DVec3, radius_m: f64, planet_radius: f64, out: &mut Vec<NodeId>) {
    fn visit(node: NodeId, cam: DVec3, r: f64, reach: f64, out: &mut Vec<NodeId>) {
        let d = (node.center_dir() * cam.length()).distance(cam) - node.edge_m(r) * 0.75;
        if d > reach {
            return;
        }
        if node.level == CELL_LEVEL {
            out.push(node);
            return;
        }
        for child in node.children() {
            visit(child, cam, r, reach, out);
        }
    }
    for face in 0..6 {
        visit(NodeId::root(face), cam, planet_radius, radius_m, out);
    }
}

/// Every few seconds, read the population densities back from the GPU.
pub fn read_densities(
    mut commands: Commands,
    mut fauna: ResMut<Fauna>,
    fields: Option<Res<SimFields>>,
    stats: Res<SimStats>,
    time: Res<Time>,
) {
    let Some(fields) = fields else { return };
    fauna.since_readback += time.delta_secs();
    if !stats.live() || fauna.readback_in_flight {
        return;
    }
    if fauna.readbacks > 0 && fauna.since_readback < READBACK_PERIOD_S {
        return;
    }
    fauna.readback_in_flight = true;
    fauna.since_readback = 0.0;
    commands
        .spawn(ReadbackOnce::texture(fields.front(ANIMALS)))
        .observe(
            |event: On<ReadbackComplete>, mut fauna: ResMut<Fauna>, mut commands: Commands| {
                let texels: Vec<[f32; 4]> = event
                    .data
                    .as_chunks::<16>()
                    .0
                    .iter()
                    .map(|t| {
                        let f = |i: usize| f32::from_le_bytes([t[i], t[i + 1], t[i + 2], t[i + 3]]);
                        [f(0), f(4), f(8), f(12)]
                    })
                    .collect();
                let resolution = ((texels.len() / 6) as f64).sqrt().round() as usize;
                if resolution * resolution * 6 == texels.len() {
                    fauna.densities = Densities { resolution, texels };
                    fauna.readbacks += 1;
                }
                fauna.readback_in_flight = false;
                commands.entity(event.entity).despawn();
            },
        );
}

/// Spawns and drops animals as the camera moves, steers them, and places them.
#[allow(clippy::too_many_arguments)]
pub fn update_fauna(
    mut commands: Commands,
    mut fauna: ResMut<Fauna>,
    camera: Res<PlanetCamera>,
    lod: Res<TerrainLod>,
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut transforms: Query<&mut Transform>,
) {
    let Some(surface) = lod.surface().cloned() else {
        return;
    };
    let fauna = &mut *fauna;
    if fauna.assets.is_none() {
        fauna.assets = Some(build_assets(&mut meshes, &mut materials));
    }
    let radius = surface.radius_m;
    let exaggeration = lod.settings.exaggeration;
    let eye = camera.position();
    let ground = |dir: DVec3| surface.height(dir, 1.0);

    // Populate cells near the camera; drop far ones.
    if fauna.readbacks > 0 {
        let mut near = Vec::new();
        if camera.height_above_ground() < 3_000.0 {
            cells_near(eye, SPAWN_RADIUS_M, radius, &mut near);
        }
        let assets = fauna.assets.as_ref().unwrap();
        for cell in near {
            if !fauna.cells.insert(cell) {
                continue;
            }
            let mut rng = (cell.face as u64) << 56 ^ (cell.x as u64) << 28 ^ cell.y as u64;
            rand(&mut rng);
            let centre = cell.center_dir();
            let (grazers, predators) = fauna.densities.at(centre);
            let edge = cell.edge_m(radius);
            let (n, e) = frame(centre);
            let place = |rng: &mut u64, around: DVec3, spread: f64| {
                let p = around + (n * (rand(rng) - 0.5) + e * (rand(rng) - 0.5)) * spread / radius;
                p.normalize()
            };
            // A herd where grazers are common, a pack where predators are.
            if rand(&mut rng) < (grazers * 1.2).min(0.8) {
                let at = place(&mut rng, centre, edge * 0.8);
                if ground(at) > 0.5 {
                    let size = 3 + (rand(&mut rng) * 10.0 * grazers.min(1.0)) as usize;
                    let group = fauna.next_group;
                    fauna.next_group += 1;
                    for _ in 0..size {
                        let dir = place(&mut rng, at, 25.0);
                        let seed = (rand(&mut rng) * u64::MAX as f64) as u64;
                        let a = spawn_animal(
                            &mut commands,
                            assets,
                            Species::Grazer,
                            dir,
                            group,
                            cell,
                            seed,
                        );
                        fauna.animals.push(a);
                    }
                }
            }
            if rand(&mut rng) < (predators * 4.0).min(0.35) {
                let at = place(&mut rng, centre, edge * 0.8);
                if ground(at) > 0.5 {
                    let size = 2 + (rand(&mut rng) * 3.0) as usize;
                    let group = fauna.next_group;
                    fauna.next_group += 1;
                    for _ in 0..size {
                        let dir = place(&mut rng, at, 12.0);
                        let seed = (rand(&mut rng) * u64::MAX as f64) as u64;
                        let a = spawn_animal(
                            &mut commands,
                            assets,
                            Species::Predator,
                            dir,
                            group,
                            cell,
                            seed,
                        );
                        fauna.animals.push(a);
                    }
                }
            }
        }
        fauna.settled = true;
        let far: Vec<NodeId> = fauna
            .cells
            .iter()
            .filter(|c| {
                (c.center_dir() * eye.length()).distance(eye) - c.edge_m(radius) > DESPAWN_RADIUS_M
            })
            .copied()
            .collect();
        for cell in far {
            fauna.cells.remove(&cell);
            fauna.animals.retain(|a| {
                if a.home == cell {
                    commands.entity(a.entity).despawn();
                    false
                } else {
                    true
                }
            });
        }
    }

    // Behaviour.
    let dt = time.delta_secs_f64().min(0.1);
    let snapshot: Vec<(Species, DVec3, u32, bool)> = fauna
        .animals
        .iter()
        .map(|a| {
            (
                a.species,
                a.dir,
                a.group,
                matches!(a.activity, Activity::Eat(_)),
            )
        })
        .collect();
    let mut group_centre: HashMap<u32, (DVec3, usize)> = HashMap::new();
    for (_, dir, group, _) in &snapshot {
        let g = group_centre.entry(*group).or_insert((DVec3::ZERO, 0));
        g.0 += *dir;
        g.1 += 1;
    }
    let metres = |a: DVec3, b: DVec3| a.angle_between(b) * radius;
    let mut caught = Vec::new();
    for (i, a) in fauna.animals.iter_mut().enumerate() {
        let tangent = |v: DVec3| v - a.dir * v.dot(a.dir);
        let towards = |b: DVec3| tangent(b - a.dir).normalize_or_zero();
        let mut steer = DVec3::ZERO;
        let mut target_speed;

        // Threats and prey in sight.
        let mut nearest: Option<(usize, f64)> = None;
        let mut threat = DVec3::ZERO;
        for (j, (species, dir, _, eating)) in snapshot.iter().enumerate() {
            if j == i {
                continue;
            }
            let d = metres(a.dir, *dir);
            match (a.species, species) {
                (Species::Grazer, Species::Predator) if d < 70.0 && !eating => {
                    threat -= towards(*dir) * (70.0 - d);
                }
                (Species::Predator, Species::Grazer)
                    if d < 160.0 && nearest.is_none_or(|(_, nd)| d < nd) =>
                {
                    nearest = Some((j, d));
                }
                _ => {}
            }
            // Personal space.
            if *species == a.species && d < 2.5 && d > 1e-3 {
                steer -= towards(*dir) * (2.5 - d);
            }
        }

        if let Activity::Eat(t) = &mut a.activity {
            *t -= dt;
            if *t <= 0.0 {
                a.activity = Activity::Idle(5.0);
            }
            target_speed = 0.0;
        } else if threat.length_squared() > 0.0 {
            a.activity = Activity::Run;
            steer += threat.normalize() * 3.0;
            target_speed = a.species.run_speed();
        } else if let (Species::Predator, Some((j, d))) = (a.species, nearest) {
            a.activity = Activity::Run;
            steer += towards(snapshot[j].1) * 3.0;
            target_speed = a.species.run_speed();
            if d < 1.6 {
                caught.push(j);
                a.activity = Activity::Eat(25.0 + 20.0 * rand(&mut a.rng));
            }
        } else {
            match &mut a.activity {
                Activity::Idle(t) | Activity::Walk(t) => *t -= dt,
                _ => {}
            }
            a.activity = match a.activity {
                Activity::Idle(t) if t <= 0.0 => {
                    // Turn a little and set off.
                    a.facing = (a.facing + random_tangent(a.dir, &mut a.rng) * 0.8).normalize();
                    Activity::Walk(3.0 + 8.0 * rand(&mut a.rng))
                }
                Activity::Walk(t) if t <= 0.0 => Activity::Idle(3.0 + 12.0 * rand(&mut a.rng)),
                Activity::Run => Activity::Walk(4.0),
                other => other,
            };
            target_speed = match a.activity {
                Activity::Walk(_) => a.species.walk_speed(),
                _ => 0.0,
            };
            // Keep with the group.
            if let Some((sum, n)) = group_centre.get(&a.group)
                && *n > 1
            {
                let centre = sum.normalize();
                let d = metres(a.dir, centre);
                let spread = if a.species == Species::Grazer {
                    18.0
                } else {
                    10.0
                };
                if d > spread {
                    steer += towards(centre) * ((d - spread) / spread).min(2.0);
                    target_speed = target_speed.max(a.species.walk_speed());
                }
            }
        }

        // Turn towards the steering direction, accelerate towards the target speed.
        if steer.length_squared() > 1e-12 {
            let want = tangent(steer).normalize_or(a.facing);
            let turn = (dt * 2.5).min(1.0);
            a.facing = (a.facing + (want - a.facing) * turn).normalize_or(a.facing);
        }
        let facing = a.facing - a.dir * a.facing.dot(a.dir);
        a.facing = if facing.length_squared() > 1e-12 {
            facing.normalize()
        } else {
            random_tangent(a.dir, &mut a.rng)
        };
        let accel = if target_speed > a.speed { 6.0 } else { 4.0 };
        a.speed += (target_speed - a.speed).clamp(-accel * dt, accel * dt);

        // Move, unless that would walk into water.
        let next = (a.dir + a.facing * (a.speed * dt / radius)).normalize();
        if ground(next) > 0.3 {
            a.dir = next;
        } else {
            a.facing = -a.facing;
            a.speed *= 0.3;
        }
        a.stride += a.speed * dt / a.species.stride_m() * std::f64::consts::TAU;
    }
    // The caught are eaten.
    caught.sort_unstable();
    caught.dedup();
    for j in caught.into_iter().rev() {
        let a = fauna.animals.swap_remove(j);
        commands.entity(a.entity).despawn();
    }

    // Place them relative to the camera (floating origin), and swing their legs.
    for a in &fauna.animals {
        let h = ground(a.dir).max(0.0) * exaggeration;
        let position = a.dir * (radius + h);
        if let Ok(mut t) = transforms.get_mut(a.entity) {
            t.translation = (position - eye).as_vec3();
            let gallop = (a.speed / a.species.run_speed()).min(1.0);
            // Bob while moving.
            t.translation += a.dir.as_vec3() * (a.stride.sin().abs() as f32 * 0.06 * gallop as f32);
            t.look_to(a.facing.as_vec3(), a.dir.as_vec3());
        }
        let amplitude = (a.speed / a.species.walk_speed()).min(1.0) * 0.45
            + 0.25 * (a.speed / a.species.run_speed()).min(1.0);
        for (k, leg) in a.legs.iter().enumerate() {
            // Diagonal pairs move together.
            let phase = a.stride
                + if k == 0 || k == 3 {
                    0.0
                } else {
                    std::f64::consts::PI
                };
            if let Ok(mut t) = transforms.get_mut(*leg) {
                t.rotation = Quat::from_rotation_x((phase.sin() * amplitude) as f32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cells_cover_the_spawn_radius() {
        let r = 6_371_000.0;
        let cam = DVec3::new(0.3, 0.9, 0.2).normalize() * (r + 2.0);
        let mut cells = Vec::new();
        cells_near(cam, SPAWN_RADIUS_M, r, &mut cells);
        let edge = cells[0].edge_m(r);
        let area = cells.len() as f64 * edge * edge;
        let circle = std::f64::consts::PI * SPAWN_RADIUS_M * SPAWN_RADIUS_M;
        assert!(area > circle, "{area} < {circle}");
        assert!(cells.iter().all(|c| c.level == CELL_LEVEL));
    }

    #[test]
    fn densities_sample_the_right_texel() {
        let res = 4;
        let mut texels = vec![[0.0; 4]; res * res * 6];
        texels[2 * res + 2] = [0.7, 0.1, 0.0, 0.0];
        let d = Densities {
            resolution: res,
            texels,
        };
        // Texel (2, 2) of face 0 spans st 0..0.5 on both axes.
        let inside = crate::planet::surface::face_st_to_dir(0, bevy::math::DVec2::splat(0.25));
        assert_eq!(d.at(inside), (0.7f32 as f64, 0.1f32 as f64));
        let outside = crate::planet::surface::face_st_to_dir(0, bevy::math::DVec2::splat(-0.25));
        assert_eq!(d.at(outside).0, 0.0);
    }
}
