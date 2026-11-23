//! Vegetation near the camera: grass, shrubs and trees as real geometry.
//!
//! Plants are placed on the CPU, per tile of the terrain quadtree at a fixed level per
//! layer, on a jittered grid that is global to each cube face (so near and far tree tiles
//! agree on where every tree stands). The CPU only knows which biomes *can* hold a plant;
//! whether it is there right now is up to the live ecosystem: the vertex shader reads the
//! `plants` field and grows each plant in when the local cover exceeds the plant's random
//! threshold, so forests spread, thin and burn on screen as the simulation runs.
//!
//! Trees come in two layers: detailed meshes up close, and cheap ones out to a few
//! kilometres, switched per tree at [`TREE_SWITCH_M`].

use bevy::{
    asset::RenderAssetUsages,
    light::NotShadowCaster,
    math::{DVec2, DVec3},
    mesh::{Indices, MeshVertexAttribute, MeshVertexBufferLayoutRef, PrimitiveTopology},
    pbr::{MaterialPipeline, MaterialPipelineKey},
    platform::collections::HashMap,
    prelude::*,
    render::render_resource::{
        AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
        VertexFormat,
    },
    shader::ShaderRef,
    tasks::{AsyncComputeTaskPool, Task, block_on, poll_once},
};
use std::sync::Arc;

use super::{
    Biome,
    camera::{PlanetAnchor, PlanetCamera},
    lod::NodeId,
    surface::{CubeField, PlanetSurface, face_st_to_dir, smoothstep},
};

/// Root of the plant this vertex belongs to, relative to the tile centre (xyz), and the
/// plant's random cover threshold (w).
pub const ATTRIBUTE_ROOT: MeshVertexAttribute =
    MeshVertexAttribute::new("VegetationRoot", 0x5157_7a02, VertexFormat::Float32x4);
/// Unit "up" at the root (xyz); w packs the plant code (integer part: kind + 4 * part +
/// 8 * far) and an ambient-occlusion term (fraction).
pub const ATTRIBUTE_PLANT: MeshVertexAttribute =
    MeshVertexAttribute::new("VegetationPlant", 0x5157_7a03, VertexFormat::Float32x4);

/// Distance at which trees switch from their detailed mesh to the cheap one.
pub const TREE_SWITCH_M: f32 = 700.0;

const KIND_GRASS: u32 = 0;
const KIND_SHRUB: u32 = 1;
const KIND_BROADLEAF: u32 = 2;
const KIND_CONIFER: u32 = 3;
const PART_STEM: u32 = 0;
const PART_FOLIAGE: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layer {
    Grass,
    Shrubs,
    TreesNear,
    TreesFar,
}

#[derive(Clone, Copy, Debug)]
pub struct LayerSettings {
    /// Quadtree level of this layer's tiles.
    pub level: u8,
    /// Tiles within this distance of the camera are built and drawn; plants shrink away
    /// towards it (in the shader).
    pub radius_m: f64,
    /// Spacing of the placement grid.
    pub spacing_m: f64,
    /// Placement stream: layers with the same stream share plant positions.
    stream: u64,
}

impl Layer {
    pub const ALL: [Layer; 4] = [
        Layer::Grass,
        Layer::Shrubs,
        Layer::TreesNear,
        Layer::TreesFar,
    ];

    pub fn settings(self) -> LayerSettings {
        match self {
            Layer::Grass => LayerSettings {
                level: 18,
                radius_m: 45.0,
                spacing_m: 0.45,
                stream: 1,
            },
            Layer::Shrubs => LayerSettings {
                level: 16,
                radius_m: 600.0,
                spacing_m: 4.0,
                stream: 2,
            },
            Layer::TreesNear => LayerSettings {
                level: 15,
                radius_m: TREE_SWITCH_M as f64 + 100.0,
                spacing_m: 11.0,
                stream: 3,
            },
            Layer::TreesFar => LayerSettings {
                level: 13,
                radius_m: 3_000.0,
                spacing_m: 11.0,
                stream: 3,
            },
        }
    }
}

/// Where each plant type can grow at all (0..1 cover the biome allows), from the baked
/// biome map, plus the baked temperature for choosing conifers.
pub struct VegetationMaps {
    grass: CubeField,
    shrub: CubeField,
    tree: CubeField,
    temperature: CubeField,
}

impl VegetationMaps {
    pub fn new(resolution: u32, biome: &[f32], temperature: Vec<f32>) -> Self {
        let map = |f: fn(Biome) -> f32| {
            let data = biome
                .iter()
                .map(|&b| {
                    Biome::ALL
                        .get(b.round().max(0.0) as usize)
                        .map_or(0.0, |&b| f(b))
                })
                .collect();
            CubeField::new(resolution, data)
        };
        Self {
            grass: map(|b| max_cover(b)[0]),
            shrub: map(|b| max_cover(b)[1]),
            tree: map(|b| max_cover(b)[2]),
            temperature: CubeField::new(resolution, temperature),
        }
    }
}

/// The most cover of (grass, shrubs, trees) each biome can carry.
fn max_cover(biome: Biome) -> [f32; 3] {
    match biome {
        Biome::Ocean | Biome::Lake | Biome::Ice => [0.0; 3],
        Biome::Tundra => [0.9, 0.5, 0.15],
        Biome::Desert => [0.25, 0.3, 0.05],
        Biome::BorealForest => [0.7, 0.6, 1.0],
        Biome::Grassland => [1.0, 0.5, 0.4],
        Biome::TemperateForest => [0.7, 0.7, 1.0],
        Biome::Shrubland => [0.9, 1.0, 0.4],
        Biome::TropicalForest => [0.6, 0.7, 1.0],
        Biome::Savanna => [1.0, 0.6, 0.55],
    }
}

/// Mesh data of one tile, positions relative to `center`.
struct TileMesh {
    center: DVec3,
    mesh: Option<Mesh>,
}

#[derive(Default)]
struct Builder {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    roots: Vec<[f32; 4]>,
    plants: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

/// One plant being emitted: its frame and the per-vertex data it shares.
#[derive(Clone, Copy)]
struct Plant {
    root: Vec3,
    up: Vec3,
    east: Vec3,
    north: Vec3,
    threshold: f32,
    kind: u32,
    far: bool,
}

impl Plant {
    /// A point in the plant's frame (x east, y up, z north) to tile-relative.
    fn at(&self, p: Vec3) -> Vec3 {
        self.root + self.east * p.x + self.up * p.y + self.north * p.z
    }

    fn dir(&self, d: Vec3) -> Vec3 {
        (self.east * d.x + self.up * d.y + self.north * d.z).normalize_or(self.up)
    }
}

impl Builder {
    fn vertex(&mut self, plant: &Plant, part: u32, p: Vec3, n: Vec3, ao: f32) -> u32 {
        let index = self.positions.len() as u32;
        self.positions.push(plant.at(p).to_array());
        self.normals.push(plant.dir(n).to_array());
        self.roots
            .push([plant.root.x, plant.root.y, plant.root.z, plant.threshold]);
        let code = plant.kind + 4 * part + if plant.far { 8 } else { 0 };
        self.plants.push([
            plant.up.x,
            plant.up.y,
            plant.up.z,
            code as f32 + ao.clamp(0.0, 1.0) * 0.99,
        ]);
        index
    }

    /// A surface of revolution around the plant's axis: `rings` are (height, radius, ao),
    /// bottom to top, each with `sides` vertices. A zero radius closes the shape (one
    /// vertex). Normals point out from `center_y`, scaled vertically by `squash`.
    #[allow(clippy::too_many_arguments)]
    fn lathe(
        &mut self,
        plant: &Plant,
        part: u32,
        rings: &[(f32, f32, f32)],
        sides: u32,
        yaw: f32,
        center_y: f32,
        jitter: &mut Rng,
        wobble: f32,
    ) {
        let mut prev: Vec<u32> = Vec::new();
        for &(y, r, ao) in rings {
            let mut ring = Vec::new();
            if r <= 0.0 {
                let p = Vec3::new(0.0, y, 0.0);
                let n = Vec3::new(0.0, (y - center_y).signum(), 0.0);
                ring.push(self.vertex(plant, part, p, n, ao));
            } else {
                for s in 0..sides {
                    let a = yaw + s as f32 / sides as f32 * std::f32::consts::TAU;
                    let r = r * (1.0 + wobble * (jitter.f32() - 0.5));
                    let p = Vec3::new(a.cos() * r, y, a.sin() * r);
                    // A NaN centre means a cylinder: horizontal normals.
                    let ny = if center_y.is_nan() {
                        0.0
                    } else {
                        (y - center_y) * 0.6
                    };
                    let n = Vec3::new(p.x, ny, p.z);
                    ring.push(self.vertex(plant, part, p, n, ao));
                }
            }
            if !prev.is_empty() {
                self.stitch(&prev, &ring);
            }
            prev = ring;
        }
    }

    /// Triangles between two rings (either may be a single vertex).
    fn stitch(&mut self, a: &[u32], b: &[u32]) {
        let n = a.len().max(b.len());
        for s in 0..n {
            let a0 = a[s % a.len()];
            let a1 = a[(s + 1) % a.len()];
            let b0 = b[s % b.len()];
            let b1 = b[(s + 1) % b.len()];
            if a.len() > 1 {
                self.indices.extend_from_slice(&[a0, a1, b0]);
            }
            if b.len() > 1 {
                self.indices.extend_from_slice(&[a1, b1, b0]);
            }
        }
    }

    fn grass(&mut self, plant: &Plant, height: f32, rng: &mut Rng) {
        let blades = 5;
        for k in 0..blades {
            let yaw = (k as f32 + rng.f32() * 0.6) / blades as f32 * std::f32::consts::TAU;
            let h = height * (0.6 + 0.6 * rng.f32());
            let w = 0.02 + 0.02 * rng.f32();
            let lean = 0.15 + 0.35 * rng.f32();
            let side = Vec3::new(yaw.cos(), 0.0, yaw.sin());
            let out = Vec3::new(-yaw.sin(), 0.0, yaw.cos());
            let offset = out * 0.12 * rng.f32();
            // Blade normals lean up, so grass lights like a canopy rather than cards.
            let n = (out * 0.4 + Vec3::Y).normalize();
            let at =
                |t: f32, x: f32| offset + side * x + Vec3::Y * (t * h) + out * (lean * h * t * t);
            let v0 = self.vertex(plant, PART_FOLIAGE, at(0.0, -w) - Vec3::Y * 0.05, n, 0.0);
            let v1 = self.vertex(plant, PART_FOLIAGE, at(0.0, w) - Vec3::Y * 0.05, n, 0.0);
            let v2 = self.vertex(plant, PART_FOLIAGE, at(0.55, -w * 0.7), n, 0.55);
            let v3 = self.vertex(plant, PART_FOLIAGE, at(0.55, w * 0.7), n, 0.55);
            let v4 = self.vertex(plant, PART_FOLIAGE, at(1.0, 0.0), n, 1.0);
            self.indices
                .extend_from_slice(&[v0, v1, v2, v1, v3, v2, v2, v3, v4]);
        }
    }

    fn shrub(&mut self, plant: &Plant, height: f32, rng: &mut Rng) {
        // Two or three overlapping lobes, so shrubs don't read as boulders.
        let lobes = 2 + (rng.f32() * 2.0) as u32;
        let root = plant.root;
        for k in 0..lobes {
            let h = height * (0.6 + 0.4 * rng.f32()) * if k == 0 { 1.0 } else { 0.8 };
            let r = h * 0.5;
            let a = rng.f32() * 6.3;
            let off = if k == 0 {
                Vec3::ZERO
            } else {
                Vec3::new(a.cos(), 0.0, a.sin()) * height * 0.45
            };
            let lobe = Plant { root, ..*plant };
            let shifted = Plant {
                root: lobe.at(off),
                ..lobe
            };
            let rings = [
                (0.0, 0.0, 0.3),
                (0.15 * h, r * 0.7, 0.4),
                (0.5 * h, r, 0.7),
                (0.82 * h, r * 0.65, 0.9),
                (h, 0.0, 1.0),
            ];
            let start = self.positions.len();
            self.lathe(&shifted, PART_FOLIAGE, &rings, 7, a, 0.45 * h, rng, 0.6);
            // Every lobe grows from the plant's own root.
            for r in &mut self.roots[start..] {
                r[0] = root.x;
                r[1] = root.y;
                r[2] = root.z;
            }
        }
    }

    fn trunk(&mut self, plant: &Plant, top: f32, radius: f32, sides: u32, yaw: f32) {
        // Starts below the ground, so trees on coarse terrain far away don't float.
        let rings = [(-3.0, radius * 1.3, 0.2), (top, radius * 0.6, 0.5)];
        let mut rng = Rng(0);
        self.lathe(
            plant,
            PART_STEM,
            &rings,
            sides,
            yaw,
            f32::NAN,
            &mut rng,
            0.0,
        );
    }

    fn broadleaf(&mut self, plant: &Plant, h: f32, rng: &mut Rng) {
        let yaw = rng.f32() * 6.3;
        let w = h * (0.26 + 0.1 * rng.f32());
        let base = h * (0.3 + 0.1 * rng.f32());
        let mid = (base + h) * 0.5;
        if plant.far {
            let rings = [(base, 0.0, 0.2), (mid, w, 0.6), (h, 0.0, 1.0)];
            self.lathe(plant, PART_FOLIAGE, &rings, 5, yaw, mid, rng, 0.3);
            return;
        }
        self.trunk(plant, mid, h * 0.022 + 0.08, 5, yaw);
        let rings = [
            (base, 0.0, 0.1),
            (base + (h - base) * 0.15, w * 0.75, 0.3),
            (base + (h - base) * 0.45, w, 0.6),
            (base + (h - base) * 0.75, w * 0.85, 0.85),
            (h, 0.0, 1.0),
        ];
        self.lathe(plant, PART_FOLIAGE, &rings, 7, yaw, mid, rng, 0.45);
    }

    fn conifer(&mut self, plant: &Plant, h: f32, rng: &mut Rng) {
        let yaw = rng.f32() * 6.3;
        let w = h * (0.17 + 0.05 * rng.f32());
        if plant.far {
            let rings = [(h * 0.12, w, 0.3), (h, 0.0, 1.0)];
            self.lathe(plant, PART_FOLIAGE, &rings, 5, yaw, h * 0.3, rng, 0.2);
            return;
        }
        self.trunk(plant, h * 0.5, h * 0.016 + 0.06, 4, yaw);
        // Stacked cones, each narrower, like the whorls of a spruce.
        for k in 0..3 {
            let k = k as f32;
            let bottom = h * (0.12 + 0.26 * k);
            let top = (bottom + h * (0.5 - 0.06 * k)).min(h);
            let r = w * (1.0 - 0.27 * k);
            let rings = [(bottom, r, 0.15 + 0.3 * k), (top, 0.0, 0.6 + 0.2 * k)];
            self.lathe(plant, PART_FOLIAGE, &rings, 6, yaw + k, bottom, rng, 0.25);
        }
    }

    fn build(self) -> Option<Mesh> {
        if self.indices.is_empty() {
            return None;
        }
        Some(
            Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::RENDER_WORLD,
            )
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
            .with_inserted_attribute(ATTRIBUTE_ROOT, self.roots)
            .with_inserted_attribute(ATTRIBUTE_PLANT, self.plants)
            .with_inserted_indices(Indices::U32(self.indices)),
        )
    }
}

/// Small deterministic generator (splitmix64).
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut r = Self(seed);
        r.next();
        r
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn f32(&mut self) -> f32 {
        self.f64() as f32
    }
}

/// Builds one tile of a layer.
fn build_tile(
    layer: Layer,
    node: NodeId,
    surface: &PlanetSurface,
    maps: &VegetationMaps,
    exaggeration: f64,
    seed: u32,
) -> TileMesh {
    let settings = layer.settings();
    let radius = surface.radius_m;
    let center = node.center_dir() * radius;
    let mut out = Builder::default();

    // The placement grid is global to the face, so tiles of different levels agree.
    let cells = (radius * std::f64::consts::FRAC_PI_2 / settings.spacing_m).round();
    let (min, size) = node.st_rect();
    let u0 = (min + 1.0) * 0.5;
    let u1 = u0 + DVec2::splat(size * 0.5);
    let (i0, i1) = ((u0.x * cells).ceil() as i64, (u1.x * cells).ceil() as i64);
    let (j0, j1) = ((u0.y * cells).ceil() as i64, (u1.y * cells).ceil() as i64);
    let face = node.face as usize;
    let field = match layer {
        Layer::Grass => &maps.grass,
        Layer::Shrubs => &maps.shrub,
        Layer::TreesNear | Layer::TreesFar => &maps.tree,
    };
    let far = layer == Layer::TreesFar;

    for j in j0..j1 {
        for i in i0..i1 {
            let mut rng = Rng::new(
                (settings.stream << 56)
                    ^ ((face as u64) << 52)
                    ^ ((j as u64) << 26)
                    ^ i as u64
                    ^ ((seed as u64) << 40),
            );
            let u = DVec2::new(i as f64 + rng.f64(), j as f64 + rng.f64()) / cells;
            let threshold = rng.f32();
            let dir = face_st_to_dir(face, u * 2.0 - 1.0);
            let allowed = field.sample(dir).clamp(0.0, 1.0) as f32;
            // The shader shows a plant only where live cover exceeds its threshold, and
            // live cover never exceeds what the biome allows (by much).
            if threshold > allowed + 0.05 {
                continue;
            }
            let h = surface.height(dir, 1.0);
            if h < 0.3 {
                continue;
            }
            let up = dir;
            let mut east = DVec3::Y.cross(up);
            if east.length_squared() < 1e-12 {
                east = DVec3::X.cross(up);
            }
            let east = east.normalize();
            let north = up.cross(east);
            if layer != Layer::Grass {
                // No trees or shrubs on cliffs.
                let d = 2.0 / radius;
                let he = surface.height((dir + east * d).normalize(), 1.0);
                let hn = surface.height((dir + north * d).normalize(), 1.0);
                let slope = DVec2::new(he - h, hn - h).length() / 2.0 * exaggeration;
                if slope > if layer == Layer::Shrubs { 1.0 } else { 0.75 } {
                    continue;
                }
            }
            let root = dir * (radius + h * exaggeration) - center;
            let kind = match layer {
                Layer::Grass => KIND_GRASS,
                Layer::Shrubs => KIND_SHRUB,
                _ => {
                    let temp = maps.temperature.sample(dir);
                    let conifer = smoothstep(17.0, 5.0, temp) * 0.9 + 0.05;
                    if rng.f64() < conifer {
                        KIND_CONIFER
                    } else {
                        KIND_BROADLEAF
                    }
                }
            };
            let plant = Plant {
                root: root.as_vec3(),
                up: up.as_vec3(),
                east: east.as_vec3(),
                north: north.as_vec3(),
                threshold,
                kind,
                far,
            };
            match kind {
                KIND_GRASS => out.grass(&plant, 0.35 + 0.35 * rng.f32(), &mut rng),
                KIND_SHRUB => out.shrub(&plant, 0.7 + 1.6 * rng.f32(), &mut rng),
                KIND_BROADLEAF => out.broadleaf(&plant, 9.0 + 13.0 * rng.f32(), &mut rng),
                _ => out.conifer(&plant, 11.0 + 17.0 * rng.f32(), &mut rng),
            }
        }
    }
    TileMesh {
        center,
        mesh: out.build(),
    }
}

/// The planet the vegetation stands on.
struct Planet {
    surface: Arc<PlanetSurface>,
    maps: Arc<VegetationMaps>,
    material: Handle<VegetationMaterial>,
    exaggeration: f64,
    seed: u32,
}

struct Tile {
    entity: Option<Entity>,
    last_used: u64,
}

/// Vegetation tiles: built, in flight, and wanted this frame.
#[derive(Resource, Default)]
pub struct Vegetation {
    planet: Option<Planet>,
    built: HashMap<(Layer, NodeId), Tile>,
    tasks: HashMap<(Layer, NodeId), Task<TileMesh>>,
    frame: u64,
    missing: usize,
    drawn: usize,
}

impl Vegetation {
    /// Switch to a new planet, dropping every tile of the old one.
    pub fn set_planet(
        &mut self,
        commands: &mut Commands,
        surface: Arc<PlanetSurface>,
        maps: VegetationMaps,
        material: Handle<VegetationMaterial>,
        exaggeration: f64,
        seed: u32,
    ) {
        for (_, tile) in self.built.drain() {
            if let Some(e) = tile.entity {
                commands.entity(e).despawn();
            }
        }
        self.tasks.clear();
        self.planet = Some(Planet {
            surface,
            maps: Arc::new(maps),
            material,
            exaggeration,
            seed,
        });
    }

    /// Every tile the camera wants is built.
    pub fn settled(&self) -> bool {
        self.planet.is_none() || (self.missing == 0 && self.tasks.is_empty())
    }

    /// The most cover of (grass, shrubs, trees) the biome at `dir` allows.
    pub fn allowed_cover(&self, dir: DVec3) -> Option<[f64; 3]> {
        let maps = &self.planet.as_ref()?.maps;
        Some([&maps.grass, &maps.shrub, &maps.tree].map(|f| f.sample(dir).clamp(0.0, 1.0)))
    }

    /// Tiles with plants in them, drawn this frame.
    pub fn drawn(&self) -> usize {
        self.drawn
    }
}

/// The tiles of `layer` near `cam`.
fn select_tiles(layer: Layer, cam: DVec3, radius_m: f64, out: &mut Vec<NodeId>) {
    let settings = layer.settings();
    fn visit(
        node: NodeId,
        level: u8,
        cam: DVec3,
        radius_m: f64,
        reach: f64,
        out: &mut Vec<NodeId>,
    ) {
        let edge = node.edge_m(radius_m);
        // Distance to the node, conservatively: its corners are ~0.71 edges from the centre,
        // which is taken at the camera's radius (tiles near the camera are near its height).
        let d = (node.center_dir() * cam.length()).distance(cam) - edge * 0.75;
        if d > reach {
            return;
        }
        if node.level == level {
            out.push(node);
            return;
        }
        for child in node.children() {
            visit(child, level, cam, radius_m, reach, out);
        }
    }
    for face in 0..6 {
        visit(
            NodeId::root(face),
            settings.level,
            cam,
            radius_m,
            settings.radius_m,
            out,
        );
    }
}

/// Per frame: build the tiles near the camera, show them, and forget old ones.
pub fn update_vegetation(
    mut commands: Commands,
    mut vegetation: ResMut<Vegetation>,
    camera: Res<PlanetCamera>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut tiles: Query<&mut Visibility, With<VegetationTile>>,
) {
    let veg = &mut *vegetation;
    let Some(planet) = &veg.planet else { return };
    veg.frame += 1;
    let frame = veg.frame;

    let done: Vec<(Layer, NodeId)> = veg
        .tasks
        .iter_mut()
        .filter_map(|(key, task)| block_on(poll_once(task)).map(|tile| (*key, tile)))
        .map(|(key, tile)| {
            let entity = tile.mesh.map(|mesh| {
                let mut e = commands.spawn((
                    Name::new(format!("Vegetation {:?} {:?}", key.0, key.1)),
                    Mesh3d(meshes.add(mesh)),
                    MeshMaterial3d(planet.material.clone()),
                    Transform::default(),
                    Visibility::Hidden,
                    PlanetAnchor(tile.center),
                    VegetationTile,
                ));
                // Millions of blades: grass shadows aren't worth it.
                if key.0 == Layer::Grass {
                    e.insert(NotShadowCaster);
                }
                e.id()
            });
            veg.built.insert(
                key,
                Tile {
                    entity,
                    last_used: frame - 1,
                },
            );
            key
        })
        .collect();
    for key in done {
        veg.tasks.remove(&key);
    }

    let cam = camera.position();
    let radius = planet.surface.radius_m;
    let mut want = Vec::new();
    let mut missing = Vec::new();
    // Nothing to draw from high up.
    if camera.height_above_ground() < 4_000.0 {
        for layer in Layer::ALL {
            let mut nodes = Vec::new();
            select_tiles(layer, cam, radius, &mut nodes);
            for node in nodes {
                let key = (layer, node);
                match veg.built.get_mut(&key) {
                    Some(tile) => {
                        tile.last_used = frame;
                        want.push(key);
                    }
                    None => missing.push(key),
                }
            }
        }
    }
    veg.missing = missing.len();

    // Start builds, nearest first, near layers before far ones.
    missing.sort_by(|a, b| {
        let d = |k: &(Layer, NodeId)| {
            (k.1.center_dir() * radius).distance(cam) / k.0.settings().radius_m
        };
        d(a).total_cmp(&d(b))
    });
    let pool = AsyncComputeTaskPool::get();
    for key in missing {
        if veg.tasks.len() >= 32 {
            break;
        }
        if veg.tasks.contains_key(&key) {
            continue;
        }
        let surface = planet.surface.clone();
        let maps = planet.maps.clone();
        let (exaggeration, seed) = (planet.exaggeration, planet.seed);
        veg.tasks.insert(
            key,
            pool.spawn(
                async move { build_tile(key.0, key.1, &surface, &maps, exaggeration, seed) },
            ),
        );
    }

    veg.drawn = 0;
    for tile in veg.built.values() {
        let Some(entity) = tile.entity else { continue };
        let want = if tile.last_used == frame {
            veg.drawn += 1;
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        match tiles.get_mut(entity) {
            Ok(mut vis) => {
                vis.set_if_neq(want);
            }
            Err(_) => {
                commands.entity(entity).insert(want);
            }
        }
    }

    // Forget tiles unused for a while.
    let stale: Vec<(Layer, NodeId)> = veg
        .built
        .iter()
        .filter(|(_, t)| frame - t.last_used > 300)
        .map(|(k, _)| *k)
        .collect();
    for key in stale {
        if let Some(Tile {
            entity: Some(e), ..
        }) = veg.built.remove(&key)
        {
            commands.entity(e).despawn();
        }
    }
}

#[derive(Component)]
pub struct VegetationTile;

/// Draws vegetation tiles; see `shaders/planet/vegetation.wesl`.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct VegetationMaterial {
    #[uniform(0)]
    pub params: VegetationParams,
    #[texture(1, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub plants: Handle<Image>,
    #[texture(2, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub temperature: Handle<Image>,
    #[texture(3, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub moisture: Handle<Image>,
}

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct VegetationParams {
    pub planet_radius_m: f32,
    pub resolution: f32,
    pub tree_switch_m: f32,
    pub _pad: f32,
    /// Draw distance of grass, shrubs and trees (xyz).
    pub fade_end_m: Vec4,
}

impl VegetationParams {
    pub fn new(planet_radius_m: f32, resolution: f32) -> Self {
        let end = |l: Layer| l.settings().radius_m as f32;
        Self {
            planet_radius_m,
            resolution,
            tree_switch_m: TREE_SWITCH_M,
            _pad: 0.0,
            fade_end_m: Vec4::new(
                end(Layer::Grass),
                end(Layer::Shrubs),
                end(Layer::TreesFar),
                0.0,
            ),
        }
    }
}

impl Material for VegetationMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/planet/vegetation.wesl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/planet/vegetation.wesl".into()
    }

    // Plants cast shadows, grown in like in the main pass (see `vegetation_prepass.wesl`).
    fn prepass_vertex_shader() -> ShaderRef {
        "shaders/planet/vegetation_prepass.wesl".into()
    }

    // Only the shadow pass uses the prepass shader; the camera has no depth prepass.
    fn enable_prepass() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            Mesh::ATTRIBUTE_NORMAL.at_shader_location(1),
            ATTRIBUTE_ROOT.at_shader_location(2),
            ATTRIBUTE_PLANT.at_shader_location(3),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        // Blades and open cones are seen from both sides.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::surface::DetailParams;

    fn flat_planet() -> (PlanetSurface, VegetationMaps) {
        let res = 8;
        let n = (res * res * 6) as usize;
        let surface = PlanetSurface {
            radius_m: 6_371_000.0,
            height: CubeField::new(res, vec![100.0; n]),
            detail: DetailParams::default(),
            precipitation: None,
        };
        let biome = vec![Biome::TemperateForest as u32 as f32; n];
        (surface, VegetationMaps::new(res, &biome, vec![12.0; n]))
    }

    fn node_at(level: u8, x: u32, y: u32) -> NodeId {
        NodeId {
            face: 0,
            level,
            x,
            y,
        }
    }

    #[test]
    fn near_and_far_trees_stand_in_the_same_places() {
        let (surface, maps) = flat_planet();
        let roots = |layer: Layer, node: NodeId| -> Vec<[i64; 3]> {
            let tile = build_tile(layer, node, &surface, &maps, 1.0, 7);
            let mesh = tile.mesh.expect("trees");
            let Some(bevy::mesh::VertexAttributeValues::Float32x4(r)) =
                mesh.attribute(ATTRIBUTE_ROOT)
            else {
                panic!("roots")
            };
            let mut out: Vec<[i64; 3]> = r
                .iter()
                .map(|r| {
                    let p = tile.center + DVec3::new(r[0] as f64, r[1] as f64, r[2] as f64);
                    [p.x.round() as i64, p.y.round() as i64, p.z.round() as i64]
                })
                .collect();
            out.sort();
            out.dedup();
            out
        };
        let base = 1u32 << 12;
        let far = node_at(13, base, base);
        let mut near: Vec<[i64; 3]> = far
            .children()
            .into_iter()
            .flat_map(|c| c.children())
            .flat_map(|c| roots(Layer::TreesNear, c))
            .collect();
        near.sort();
        let far = roots(Layer::TreesFar, far);
        assert!(far.len() > 100, "a forest has trees: {}", far.len());
        assert_eq!(near, far);
    }

    #[test]
    fn tiles_near_the_camera_are_selected() {
        let r = 6_371_000.0;
        let cam = DVec3::new(1.0, 0.1, -0.2).normalize() * (r + 2.0);
        let mut tiles = Vec::new();
        select_tiles(Layer::Grass, cam, r, &mut tiles);
        assert!(!tiles.is_empty() && tiles.len() < 40, "{}", tiles.len());
        let edge = tiles[0].edge_m(r);
        for t in &tiles {
            assert!((t.center_dir() * r).distance(cam) < 45.0 + edge);
        }
    }
}
