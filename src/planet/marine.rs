//! Sea life near the camera: fish schools, kelp forests and coral reefs.
//!
//! Like the animals on land, sea life is placed around the camera, deterministically per
//! quadtree cell, so the same spot always holds the same reef. What grows where follows the
//! baked maps: kelp in cool shallow water, coral in warm shallows, and fish schools
//! everywhere the water is not too deep, more of them over reefs and kelp. Fish swim on the
//! CPU: each school wanders at a depth between the surface and the seabed, turns back toward
//! home, and its fish keep loose formation around it, wagging their tails.
//!
//! Everything is drawn only while the camera is near the water, since the sea surface is
//! opaque from above.

use bevy::{
    math::DVec3,
    platform::collections::{HashMap, HashSet},
    prelude::*,
};

use super::{
    camera::{PlanetAnchor, PlanetCamera},
    lod::{NodeId, TerrainLod},
    weather::Weather,
};

/// Quadtree level of spawn cells (~150 m).
const CELL_LEVEL: u8 = 16;
const SPAWN_RADIUS_M: f64 = 260.0;
const DESPAWN_RADIUS_M: f64 = 400.0;
/// Sea life is only placed while the camera is this close to sea level.
const NEAR_WATER_M: f64 = 60.0;
/// While diving, a school is brought in when none is this close.
const VISITOR_RANGE_M: f64 = 35.0;

/// Sea-surface temperature above which coral grows, and below which kelp does (°C).
const CORAL_MIN_C: f64 = 20.0;
const KELP_MAX_C: f64 = 18.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Habitat {
    Reef,
    Kelp,
    Open,
}

struct School {
    cell: NodeId,
    /// Planet-centred position of the school's centre, metres.
    centre: DVec3,
    /// Unit tangent the school swims along.
    heading: DVec3,
    home: DVec3,
    speed: f64,
    /// Preferred depth below the surface, as a fraction of the water depth.
    depth_fraction: f64,
    fish: Vec<Fish>,
    rng: u64,
    turn: f64,
}

struct Fish {
    entity: Entity,
    tail: Entity,
    /// Offset in the school's frame (along, across, up), metres.
    offset: Vec3,
    phase: f32,
}

struct Assets3d {
    fish_body: Handle<Mesh>,
    fish_tail: Handle<Mesh>,
    fish_colors: Vec<Handle<StandardMaterial>>,
    kelp_stalk: Handle<Mesh>,
    kelp_leaf: Handle<Mesh>,
    kelp_color: Handle<StandardMaterial>,
    coral_shapes: Vec<Handle<Mesh>>,
    coral_colors: Vec<Handle<StandardMaterial>>,
}

#[derive(Resource, Default)]
pub struct Marine {
    assets: Option<Assets3d>,
    cells: HashSet<NodeId>,
    /// Static things (kelp, coral) per cell.
    statics: HashMap<NodeId, Vec<Entity>>,
    schools: Vec<School>,
    /// Schools spawned next to the camera so far (seeds the next one).
    visitors: u64,
}

impl Marine {
    pub fn fish_count(&self) -> usize {
        self.schools.iter().map(|s| s.fish.len()).sum()
    }

    pub fn static_count(&self) -> usize {
        self.statics.values().map(Vec::len).sum()
    }
}

/// What lives at a spot: water `depth_m` deep with surface temperature `temp_c`.
pub fn habitat(depth_m: f64, temp_c: f64) -> Option<Habitat> {
    if !(1.5..250.0).contains(&depth_m) {
        return None;
    }
    if temp_c > CORAL_MIN_C && depth_m < 30.0 {
        Some(Habitat::Reef)
    } else if temp_c < KELP_MAX_C && temp_c > -1.0 && (3.0..35.0).contains(&depth_m) {
        Some(Habitat::Kelp)
    } else {
        Some(Habitat::Open)
    }
}

fn rand(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
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

/// Rotation taking the model's +Y to `up` and +Z to `forward` (made tangent).
fn orient(up: DVec3, forward: DVec3) -> Quat {
    let f = (forward - up * forward.dot(up)).normalize_or(frame(up).0);
    let right = up.cross(f);
    Quat::from_mat3(&Mat3::from_cols(right.as_vec3(), up.as_vec3(), f.as_vec3()))
}

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

fn build_assets(meshes: &mut Assets<Mesh>, materials: &mut Assets<StandardMaterial>) -> Assets3d {
    let mat = |materials: &mut Assets<StandardMaterial>, c: Color, rough: f32| {
        materials.add(StandardMaterial {
            base_color: c,
            perceptual_roughness: rough,
            ..default()
        })
    };
    Assets3d {
        // A unit-length body along +Z (scaled per fish), and a flat tail fin behind it.
        fish_body: meshes.add(Sphere::new(0.5).mesh().ico(2).unwrap()),
        fish_tail: meshes.add(Cuboid::new(0.02, 0.45, 0.35)),
        fish_colors: [
            Color::srgb(0.75, 0.78, 0.82),
            Color::srgb(0.95, 0.75, 0.15),
            Color::srgb(0.25, 0.45, 0.85),
            Color::srgb(0.95, 0.45, 0.15),
            Color::srgb(0.55, 0.65, 0.60),
        ]
        .into_iter()
        .map(|c| mat(materials, c, 0.35))
        .collect(),
        kelp_stalk: meshes.add(Cylinder::new(0.04, 1.0)),
        kelp_leaf: meshes.add(Cuboid::new(0.25, 1.1, 0.01)),
        kelp_color: mat(materials, Color::srgb(0.35, 0.33, 0.10), 0.8),
        coral_shapes: vec![
            meshes.add(Sphere::new(0.5).mesh().ico(2).unwrap()),
            meshes.add(Cone::new(0.25, 1.0)),
            meshes.add(Cylinder::new(0.12, 1.0)),
            meshes.add(Torus::new(0.2, 0.45)),
        ],
        coral_colors: [
            Color::srgb(0.95, 0.40, 0.55),
            Color::srgb(0.98, 0.60, 0.20),
            Color::srgb(0.60, 0.35, 0.80),
            Color::srgb(0.95, 0.85, 0.35),
            Color::srgb(0.30, 0.75, 0.70),
            Color::srgb(0.85, 0.25, 0.20),
        ]
        .into_iter()
        .map(|c| mat(materials, c, 0.7))
        .collect(),
    }
}

/// Spawns a school of fish over `at` (unit direction) where the seabed is at `floor` m.
fn spawn_school(
    commands: &mut Commands,
    assets: &Assets3d,
    cell: NodeId,
    at: DVec3,
    radius: f64,
    floor: f64,
    rng: &mut u64,
) -> School {
    let depth_fraction = 0.3 + 0.4 * rand(rng);
    let color = (rand(rng) * assets.fish_colors.len() as f64) as usize;
    let size = (0.15 + rand(rng) * 0.35) as f32;
    let count = 12 + (rand(rng) * 30.0) as usize;
    let spread = (size * 6.0).max(1.5);
    let heading = {
        let (n, e) = frame(at);
        let a = rand(rng) * std::f64::consts::TAU;
        n * a.cos() + e * a.sin()
    };
    let centre_pos = at * (radius + floor * depth_fraction);
    let fish = (0..count)
        .map(|_| {
            let offset = Vec3::new(
                (rand(rng) as f32 - 0.5) * spread * 1.6,
                (rand(rng) as f32 - 0.5) * spread * 0.5,
                (rand(rng) as f32 - 0.5) * spread,
            );
            let s = size * (0.85 + 0.3 * rand(rng) as f32);
            let mut tail = Entity::PLACEHOLDER;
            let entity = commands
                .spawn((
                    Name::new("Fish"),
                    Transform::from_scale(Vec3::splat(s)),
                    Visibility::default(),
                    PlanetAnchor(centre_pos),
                ))
                .with_children(|f| {
                    f.spawn((
                        Mesh3d(assets.fish_body.clone()),
                        MeshMaterial3d(assets.fish_colors[color].clone()),
                        Transform::from_scale(Vec3::new(0.28, 0.4, 1.0)),
                    ));
                    tail = f
                        .spawn((
                            Mesh3d(assets.fish_tail.clone()),
                            MeshMaterial3d(assets.fish_colors[color].clone()),
                            Transform::from_xyz(0.0, 0.0, -0.6),
                        ))
                        .id();
                })
                .id();
            Fish {
                entity,
                tail,
                offset,
                phase: rand(rng) as f32 * 6.3,
            }
        })
        .collect();
    School {
        cell,
        centre: centre_pos,
        heading,
        home: at,
        speed: 0.6 + rand(rng) * 0.8,
        depth_fraction,
        fish,
        rng: *rng,
        turn: 0.0,
    }
}

/// Spawns and drops sea life as the camera moves, and swims the fish.
pub fn update_marine(
    mut commands: Commands,
    mut marine: ResMut<Marine>,
    camera: Res<PlanetCamera>,
    lod: Res<TerrainLod>,
    weather: Res<Weather>,
    time: Res<Time>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut anchors: Query<(&mut PlanetAnchor, &mut Transform)>,
    mut parts: Query<&mut Transform, Without<PlanetAnchor>>,
) {
    let Some(surface) = lod.surface().cloned() else {
        return;
    };
    if !weather.ready() {
        return;
    }
    let marine = &mut *marine;
    if marine.assets.is_none() {
        marine.assets = Some(build_assets(&mut meshes, &mut materials));
    }
    let radius = surface.radius_m;
    let eye = camera.position();
    let seabed = |dir: DVec3| surface.height(dir, 1.0);

    // Populate cells near the camera; drop far ones.
    let mut near = Vec::new();
    if camera.altitude_m.abs() < NEAR_WATER_M {
        cells_near(eye, SPAWN_RADIUS_M, radius, &mut near);
    }
    let assets = marine.assets.as_ref().unwrap();
    for cell in near {
        if !marine.cells.insert(cell) {
            continue;
        }
        let mut rng = (cell.face as u64) << 56 ^ (cell.x as u64) << 28 ^ cell.y as u64 ^ 0x5ea;
        rand(&mut rng);
        let centre = cell.center_dir();
        let depth = -seabed(centre);
        let Some(home) = habitat(depth, weather.temperature(centre, 0.0)) else {
            continue;
        };
        let edge = cell.edge_m(radius);
        let (n, e) = frame(centre);
        let place = |rng: &mut u64, around: DVec3, spread: f64| {
            (around + (n * (rand(rng) - 0.5) + e * (rand(rng) - 0.5)) * spread / radius).normalize()
        };
        let mut statics = Vec::new();
        match home {
            Habitat::Kelp => {
                for _ in 0..(20.0 + rand(&mut rng) * 40.0) as usize {
                    let dir = place(&mut rng, centre, edge);
                    let floor = seabed(dir);
                    if floor > -2.5 {
                        continue;
                    }
                    // Up to a metre or two below the surface.
                    let length = ((-floor - 1.0) * (0.6 + 0.4 * rand(&mut rng))) as f32;
                    let base = dir * (radius + floor);
                    let rotation = orient(dir, frame(dir).0)
                        * Quat::from_rotation_y((rand(&mut rng) * 6.3) as f32);
                    let stalk = commands
                        .spawn((
                            Name::new("Kelp"),
                            Transform::from_rotation(rotation),
                            PlanetAnchor(base),
                            Visibility::default(),
                        ))
                        .with_children(|kelp| {
                            // A unit cylinder, stretched to the stalk's length.
                            kelp.spawn((
                                Mesh3d(assets.kelp_stalk.clone()),
                                MeshMaterial3d(assets.kelp_color.clone()),
                                Transform::from_xyz(0.0, length * 0.5, 0.0)
                                    .with_scale(Vec3::new(1.0, length, 1.0)),
                            ));
                            let leaves = (length / 1.2) as usize;
                            for k in 0..leaves {
                                let y = (k as f32 + 0.5) * length / leaves as f32;
                                let a = k as f32 * 2.4;
                                kelp.spawn((
                                    Mesh3d(assets.kelp_leaf.clone()),
                                    MeshMaterial3d(assets.kelp_color.clone()),
                                    Transform::from_xyz(0.15 * a.cos(), y, 0.15 * a.sin())
                                        .with_rotation(
                                            Quat::from_rotation_y(a) * Quat::from_rotation_z(0.5),
                                        ),
                                ));
                            }
                        })
                        .id();
                    statics.push(stalk);
                }
            }
            Habitat::Reef => {
                for _ in 0..(120.0 + rand(&mut rng) * 120.0) as usize {
                    let dir = place(&mut rng, centre, edge);
                    let floor = seabed(dir);
                    if floor > -1.5 {
                        continue;
                    }
                    let shape = (rand(&mut rng) * assets.coral_shapes.len() as f64) as usize;
                    let color = (rand(&mut rng) * assets.coral_colors.len() as f64) as usize;
                    let size = (0.5 + rand(&mut rng) * 1.8) as f32;
                    let rotation = orient(dir, frame(dir).0)
                        * Quat::from_rotation_y((rand(&mut rng) * 6.3) as f32);
                    let stretch = if shape == 0 {
                        Vec3::new(1.0, 0.6, 1.0)
                    } else {
                        Vec3::ONE
                    };
                    statics.push(
                        commands
                            .spawn((
                                Name::new("Coral"),
                                Mesh3d(assets.coral_shapes[shape].clone()),
                                MeshMaterial3d(assets.coral_colors[color].clone()),
                                Transform::from_rotation(rotation).with_scale(stretch * size),
                                PlanetAnchor(dir * (radius + floor + 0.25 * size as f64)),
                            ))
                            .id(),
                    );
                }
            }
            Habitat::Open => {}
        }
        marine.statics.insert(cell, statics);

        // Fish schools: commonest over reefs, then kelp, fewest in open water.
        let chance = match home {
            Habitat::Reef => 1.0,
            Habitat::Kelp => 0.8,
            Habitat::Open => 0.35,
        };
        let schools = (chance * 2.0 * rand(&mut rng)).round() as usize + (chance >= 1.0) as usize;
        for _ in 0..schools {
            let at = place(&mut rng, centre, edge * 0.8);
            let floor = seabed(at);
            if floor > -2.0 {
                continue;
            }
            let school = spawn_school(&mut commands, assets, cell, at, radius, floor, &mut rng);
            marine.schools.push(school);
        }
    }
    // A school passing by: while diving, keep one within sight of the camera, since the
    // per-cell schools are often out of the ~60 m underwater visibility.
    let below = camera.altitude_m < -1.5;
    let (forward, _) = camera.look();
    let in_view = |s: &School| {
        let to = s.centre - eye;
        to.length() < VISITOR_RANGE_M && to.normalize_or_zero().dot(forward) > 0.6
    };
    if below && !marine.schools.iter().any(in_view) {
        let (north, _, up) = camera.frame();
        let mut rng = marine.visitors.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ 0xf15;
        marine.visitors += 1;
        let side = (rand(&mut rng) - 0.5) * 10.0;
        let ahead = forward - up * forward.dot(up);
        let ahead = ahead.normalize_or(north);
        let at = (eye + ahead * (10.0 + rand(&mut rng) * 8.0) + ahead.cross(up) * side).normalize();
        let floor = seabed(at);
        // Owned by the nearest populated cell, so it goes when the camera leaves.
        let cell = marine
            .cells
            .iter()
            .min_by(|a, b| {
                let da = a.center_dir().distance(at);
                let db = b.center_dir().distance(at);
                da.total_cmp(&db)
            })
            .copied();
        if let (Some(cell), true) = (cell, floor < -3.0) {
            let mut school = spawn_school(&mut commands, assets, cell, at, radius, floor, &mut rng);
            // Near the camera's depth, so it swims across the view.
            let depth = -camera.altitude_m;
            school.depth_fraction = (depth / -floor).clamp(0.2, 0.8);
            let altitude = (floor * school.depth_fraction).clamp(floor + 1.0, -1.0);
            school.centre = at * (radius + altitude);
            school.home = at;
            marine.schools.push(school);
        }
    }
    let far: Vec<NodeId> = marine
        .cells
        .iter()
        .filter(|c| {
            camera.altitude_m.abs() > NEAR_WATER_M * 1.5
                || (c.center_dir() * eye.length()).distance(eye) > DESPAWN_RADIUS_M
        })
        .copied()
        .collect();
    for cell in far {
        marine.cells.remove(&cell);
        for e in marine.statics.remove(&cell).unwrap_or_default() {
            commands.entity(e).despawn();
        }
    }
    marine.schools.retain(|s| {
        let keep = marine.cells.contains(&s.cell);
        if !keep {
            for f in &s.fish {
                commands.entity(f.entity).despawn();
            }
        }
        keep
    });

    // Swim.
    let dt = time.delta_secs_f64();
    let t = time.elapsed_secs();
    for school in &mut marine.schools {
        let up = school.centre.normalize();
        // Wander, and turn back toward home beyond ~100 m.
        if rand(&mut school.rng) < dt * 0.3 {
            school.turn = (rand(&mut school.rng) - 0.5) * 0.8;
        }
        let to_home = school.home * radius - school.centre;
        let away = to_home.length();
        let (n, e) = frame(up);
        let mut heading = school.heading
            + (n * e.dot(school.heading) - e * n.dot(school.heading)) * school.turn * dt;
        if away > 100.0 {
            heading += (to_home / away - up * up.dot(to_home / away)) * 0.5 * dt;
        }
        heading = (heading - up * up.dot(heading)).normalize_or(n);
        school.heading = heading;
        let mut next = school.centre + heading * school.speed * dt;
        // Hold the preferred depth, clear of the seabed and the surface.
        let dir = next.normalize();
        let floor = seabed(dir).min(-2.0);
        let target = (floor * school.depth_fraction).clamp(floor + 1.0, -1.0);
        let altitude = next.length() - radius;
        next = dir * (radius + altitude + (target - altitude) * (dt * 0.5).min(1.0));
        school.centre = next;

        let rotation = orient(next.normalize(), heading);
        for fish in &school.fish {
            let wobble = Vec3::new(
                (t * 0.7 + fish.phase).sin() * 0.3,
                (t * 0.5 + fish.phase * 1.3).sin() * 0.15,
                (t * 0.9 + fish.phase * 0.7).sin() * 0.4,
            );
            let offset = rotation * (fish.offset + wobble);
            if let Ok((mut anchor, mut transform)) = anchors.get_mut(fish.entity) {
                anchor.0 = next + offset.as_dvec3();
                transform.rotation = rotation;
            }
            if let Ok(mut tail) = parts.get_mut(fish.tail) {
                tail.rotation = Quat::from_rotation_y((t * 8.0 + fish.phase).sin() * 0.5);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn habitats_follow_depth_and_temperature() {
        assert_eq!(habitat(10.0, 26.0), Some(Habitat::Reef));
        assert_eq!(habitat(10.0, 12.0), Some(Habitat::Kelp));
        assert_eq!(habitat(80.0, 26.0), Some(Habitat::Open));
        assert_eq!(habitat(0.5, 26.0), None);
        assert_eq!(habitat(4_000.0, 26.0), None);
    }

    #[test]
    fn orient_maps_up_and_forward() {
        let up = DVec3::new(0.3, 0.8, -0.2).normalize();
        let q = orient(up, DVec3::X);
        assert!((q * Vec3::Y).distance(up.as_vec3()) < 1e-5);
        assert!((q * Vec3::Z).dot(up.as_vec3()).abs() < 1e-5);
    }
}
