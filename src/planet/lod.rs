//! Level-of-detail terrain: a quadtree on each cube face, refined around the camera, so the
//! planet can be seen whole from space and walked over at metre scale.
//!
//! Every quadtree node is a chunk of `CHUNK_QUADS²` quads. A node splits when the camera is
//! closer than `split_distance × its edge length`. Chunk meshes are built on the async
//! compute pool from [`PlanetSurface`] in double precision, relative to the chunk centre, so
//! they stay precise at any distance; [`super::camera`] keeps the camera at the origin and
//! moves chunks instead (floating origin). A node only gives way to its children once all
//! four are built, so there are no holes while detail streams in. Skirts hide the cracks
//! between neighbouring levels.

use std::sync::Arc;

use bevy::{
    asset::RenderAssetUsages,
    math::{DVec2, DVec3},
    mesh::{Indices, MeshVertexAttribute, PrimitiveTopology},
    platform::collections::HashMap,
    prelude::*,
    render::render_resource::VertexFormat,
    tasks::{AsyncComputeTaskPool, Task, block_on, futures_lite::future::poll_once},
};

use super::{
    TerrainMaterial,
    camera::PlanetCamera,
    mesh::FACE_AXES,
    surface::{PlanetSurface, face_st_to_dir},
};

/// Quads per chunk edge.
pub const CHUNK_QUADS: u32 = 32;

/// Unit direction (xyz) and height above sea level in metres (w) per vertex, for shading.
pub const ATTRIBUTE_SURFACE: MeshVertexAttribute =
    MeshVertexAttribute::new("Surface", 0x5157_7a01, VertexFormat::Float32x4);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId {
    pub face: u8,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl NodeId {
    pub fn root(face: u8) -> Self {
        Self {
            face,
            level: 0,
            x: 0,
            y: 0,
        }
    }

    /// Face `st` of the node's lower corner and its edge length in `st` units.
    pub fn st_rect(&self) -> (DVec2, f64) {
        let size = 2.0 / (1u64 << self.level) as f64;
        (
            DVec2::new(self.x as f64 * size - 1.0, self.y as f64 * size - 1.0),
            size,
        )
    }

    pub fn children(&self) -> [NodeId; 4] {
        [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dy)| NodeId {
            face: self.face,
            level: self.level + 1,
            x: self.x * 2 + dx,
            y: self.y * 2 + dy,
        })
    }

    pub fn center_dir(&self) -> DVec3 {
        let (min, size) = self.st_rect();
        face_st_to_dir(self.face as usize, min + size * 0.5)
    }

    /// Approximate edge length on the sphere, in metres.
    pub fn edge_m(&self, radius_m: f64) -> f64 {
        radius_m * std::f64::consts::FRAC_PI_2 / (1u64 << self.level) as f64
    }
}

/// A built chunk mesh, positions relative to `center`.
pub struct ChunkMesh {
    pub center: DVec3,
    pub mesh: Mesh,
}

#[derive(Clone, Copy, Debug)]
pub struct LodSettings {
    /// A node splits when the camera is closer than this many edge lengths.
    pub split_distance: f64,
    pub max_level: u8,
    /// Vertical exaggeration of the terrain.
    pub exaggeration: f64,
    /// Chunk builds in flight at once.
    pub max_tasks: usize,
}

impl Default for LodSettings {
    fn default() -> Self {
        Self {
            split_distance: 2.4,
            max_level: 19,
            exaggeration: 1.0,
            max_tasks: 48,
        }
    }
}

/// Builds the mesh of one node: a `(CHUNK_QUADS + 1)²` vertex grid plus edge skirts.
pub fn build_chunk(node: NodeId, surface: &PlanetSurface, exaggeration: f64) -> ChunkMesh {
    let n = CHUNK_QUADS as i32;
    let (min, size) = node.st_rect();
    let face = node.face as usize;
    let spacing = node.edge_m(surface.radius_m) / n as f64;
    let min_wavelength = 2.0 * spacing;

    // Heights on the grid plus a one-vertex border, for normals.
    let w = (n + 3) as usize;
    let mut points = Vec::with_capacity(w * w);
    let mut heights = Vec::with_capacity(w * w);
    let mut dirs = Vec::with_capacity(w * w);
    for j in -1..=n + 1 {
        for i in -1..=n + 1 {
            let st = min + DVec2::new(i as f64, j as f64) / n as f64 * size;
            let dir = face_st_to_dir(face, st);
            let h = surface.height(dir, min_wavelength);
            // Oceans are flat at sea level; the material shades them by depth.
            let r = surface.radius_m + h.max(0.0) * exaggeration;
            points.push(dir * r);
            heights.push(h);
            dirs.push(dir);
        }
    }
    let at = |i: i32, j: i32| ((j + 1) as usize) * w + (i + 1) as usize;
    let center_dir = node.center_dir();
    let center = center_dir
        * (surface.radius_m + surface.height(center_dir, min_wavelength).max(0.0) * exaggeration);

    let verts = ((n + 1) * (n + 1)) as usize;
    let skirt_verts = (4 * (n + 1)) as usize;
    let mut positions = Vec::with_capacity(verts + skirt_verts);
    let mut normals = Vec::with_capacity(verts + skirt_verts);
    let mut surf = Vec::with_capacity(verts + skirt_verts);
    for j in 0..=n {
        for i in 0..=n {
            let k = at(i, j);
            let dir = dirs[k];
            let mut normal = (points[at(i + 1, j)] - points[at(i - 1, j)])
                .cross(points[at(i, j + 1)] - points[at(i, j - 1)])
                .normalize_or(dir);
            if normal.dot(dir) < 0.0 {
                normal = -normal;
            }
            if heights[k] < 0.0 {
                normal = dir;
            }
            positions.push((points[k] - center).as_vec3().to_array());
            normals.push(normal.as_vec3().to_array());
            surf.push([dir.x as f32, dir.y as f32, dir.z as f32, heights[k] as f32]);
        }
    }

    let grid = |i: i32, j: i32| (j * (n + 1) + i) as u32;
    let (normal_axis, u, v) = FACE_AXES[face];
    let flip = u.cross(v).dot(normal_axis) < 0.0;
    let mut indices = Vec::with_capacity((n * n * 6 + 4 * n * 6) as usize);
    for j in 0..n {
        for i in 0..n {
            let (i0, i1, i2, i3) = (
                grid(i, j),
                grid(i + 1, j),
                grid(i, j + 1),
                grid(i + 1, j + 1),
            );
            if flip {
                indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
            } else {
                indices.extend_from_slice(&[i0, i1, i2, i1, i3, i2]);
            }
        }
    }

    // Skirts: each edge copied downwards, deep enough to cover a coarser neighbour.
    let skirt = spacing * 1.5 + 20.0;
    let edges: [Vec<(i32, i32)>; 4] = [
        (0..=n).map(|i| (i, 0)).collect(),
        (0..=n).map(|i| (i, n)).collect(),
        (0..=n).map(|j| (0, j)).collect(),
        (0..=n).map(|j| (n, j)).collect(),
    ];
    for edge in edges {
        let start = positions.len() as u32;
        for &(i, j) in &edge {
            let k = at(i, j);
            let p = points[k] - dirs[k] * skirt;
            positions.push((p - center).as_vec3().to_array());
            normals.push(normals[grid(i, j) as usize]);
            surf.push(surf[grid(i, j) as usize]);
        }
        for e in 0..edge.len() - 1 {
            let (a, b) = (
                grid(edge[e].0, edge[e].1),
                grid(edge[e + 1].0, edge[e + 1].1),
            );
            let (sa, sb) = (start + e as u32, start + e as u32 + 1);
            // The material draws both sides, so winding doesn't matter here.
            indices.extend_from_slice(&[a, b, sa, b, sb, sa]);
        }
    }

    let mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(ATTRIBUTE_SURFACE, surf)
    .with_inserted_indices(Indices::U32(indices));
    ChunkMesh { center, mesh }
}

/// A built chunk in the world.
#[derive(Component)]
pub struct TerrainChunk {
    pub center: DVec3,
}

struct Built {
    entity: Entity,
    last_used: u64,
}

/// The quadtree state: built chunks, builds in flight, and what was drawn last frame.
#[derive(Resource, Default)]
pub struct TerrainLod {
    pub settings: LodSettings,
    surface: Option<Arc<PlanetSurface>>,
    material: Option<Handle<TerrainMaterial>>,
    built: HashMap<NodeId, Built>,
    tasks: HashMap<NodeId, Task<ChunkMesh>>,
    frame: u64,
    /// Nodes wanted last frame that were not built yet.
    missing: usize,
    drawn: usize,
}

impl TerrainLod {
    pub fn new(settings: LodSettings) -> Self {
        Self {
            settings,
            ..default()
        }
    }

    pub fn surface(&self) -> Option<&Arc<PlanetSurface>> {
        self.surface.as_ref()
    }

    /// Whether the terrain shows anything yet.
    pub fn ready(&self) -> bool {
        self.drawn > 0
    }

    /// Every chunk the camera wants is built and drawn.
    pub fn settled(&self) -> bool {
        self.ready() && self.missing == 0 && self.tasks.is_empty()
    }

    pub fn drawn(&self) -> usize {
        self.drawn
    }

    /// Switch to a new planet surface, dropping every chunk of the old one.
    pub fn set_surface(
        &mut self,
        commands: &mut Commands,
        surface: PlanetSurface,
        material: Handle<TerrainMaterial>,
    ) {
        for (_, built) in self.built.drain() {
            commands.entity(built.entity).despawn();
        }
        self.tasks.clear();
        self.surface = Some(Arc::new(surface));
        self.material = Some(material);
        self.drawn = 0;
        self.missing = 0;
    }
}

/// Per frame: choose the nodes to draw, start builds for missing ones, collect finished
/// builds, and show exactly the chosen chunks.
pub fn update_terrain(
    mut commands: Commands,
    mut lod: ResMut<TerrainLod>,
    camera: Res<PlanetCamera>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut chunks: Query<&mut Visibility, With<TerrainChunk>>,
) {
    let lod = &mut *lod;
    let (Some(surface), Some(material)) = (lod.surface.clone(), lod.material.clone()) else {
        return;
    };
    lod.frame += 1;

    // Finished builds become (hidden) chunk entities.
    let done: Vec<NodeId> = lod
        .tasks
        .iter_mut()
        .filter_map(|(id, task)| block_on(poll_once(task)).map(|chunk| (*id, chunk)))
        .map(|(id, chunk)| {
            let entity = commands
                .spawn((
                    Name::new(format!("Terrain {id:?}")),
                    Mesh3d(meshes.add(chunk.mesh)),
                    MeshMaterial3d(material.clone()),
                    Transform::default(),
                    Visibility::Hidden,
                    TerrainChunk {
                        center: chunk.center,
                    },
                ))
                .id();
            lod.built.insert(
                id,
                Built {
                    entity,
                    last_used: lod.frame - 1,
                },
            );
            id
        })
        .collect();
    for id in done {
        lod.tasks.remove(&id);
    }

    // Select.
    let cam = camera.position();
    let mut draw = Vec::new();
    let mut want = Vec::new();
    for face in 0..6 {
        select(
            NodeId::root(face),
            cam,
            &surface,
            &lod.settings,
            &lod.built,
            &mut draw,
            &mut want,
        );
    }
    lod.missing = want.len();

    // Start builds, nearest first.
    want.sort_by(|a, b| {
        let d = |n: &NodeId| (n.center_dir() * surface.radius_m).distance_squared(cam);
        d(a).total_cmp(&d(b))
    });
    let pool = AsyncComputeTaskPool::get();
    for id in want {
        if lod.tasks.len() >= lod.settings.max_tasks {
            break;
        }
        if lod.tasks.contains_key(&id) {
            continue;
        }
        let surface = surface.clone();
        let exaggeration = lod.settings.exaggeration;
        lod.tasks.insert(
            id,
            pool.spawn(async move { build_chunk(id, &surface, exaggeration) }),
        );
    }

    // Show exactly the selected chunks.
    let frame = lod.frame;
    for id in &draw {
        if let Some(built) = lod.built.get_mut(id) {
            built.last_used = frame;
        }
    }
    lod.drawn = draw.len();
    for (_, built) in lod.built.iter() {
        let want = if built.last_used == frame {
            Visibility::Inherited
        } else {
            Visibility::Hidden
        };
        match chunks.get_mut(built.entity) {
            Ok(mut vis) => {
                vis.set_if_neq(want);
            }
            // Spawned this frame: its commands haven't been applied yet.
            Err(_) => {
                commands.entity(built.entity).insert(want);
            }
        }
    }

    // Forget chunks unused for a while once the cache is large.
    if lod.built.len() > 3000 {
        let stale: Vec<NodeId> = lod
            .built
            .iter()
            .filter(|(_, b)| frame - b.last_used > 600)
            .map(|(id, _)| *id)
            .collect();
        for id in stale {
            if let Some(b) = lod.built.remove(&id) {
                commands.entity(b.entity).despawn();
            }
        }
    }
}

/// Whether a node is wholly hidden behind the planet's horizon (allowing for mountains).
fn beyond_horizon(node: &NodeId, cam: DVec3, radius: f64) -> bool {
    let d = cam.length();
    if d <= radius {
        return false;
    }
    let horizon = (radius / d).acos() + (radius / (radius + 9_000.0)).acos();
    let angle = node.center_dir().angle_between(cam / d);
    // Angular radius of the node (half diagonal).
    let extent = node.edge_m(radius) / radius * 0.75;
    angle - extent > horizon
}

fn select(
    node: NodeId,
    cam: DVec3,
    surface: &PlanetSurface,
    settings: &LodSettings,
    built: &HashMap<NodeId, Built>,
    draw: &mut Vec<NodeId>,
    want: &mut Vec<NodeId>,
) {
    let radius = surface.radius_m;
    if node.level > 0 && beyond_horizon(&node, cam, radius) {
        return;
    }
    let edge = node.edge_m(radius);
    let center = node.center_dir() * radius;
    // Distance to the node, less its half diagonal, so big nodes refine early enough.
    let distance = (cam.distance(center) - edge * 0.7).max(0.0);
    let split = node.level < settings.max_level && distance < settings.split_distance * edge;
    if !built.contains_key(&node) {
        want.push(node);
        return;
    }
    if split {
        let children = node.children();
        let missing: Vec<NodeId> = children
            .iter()
            .filter(|c| !built.contains_key(*c))
            .copied()
            .collect();
        if missing.is_empty() {
            for child in children {
                select(child, cam, surface, settings, built, draw, want);
            }
            return;
        }
        want.extend(missing);
    }
    draw.push(node);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::surface::{CubeField, DetailParams};

    fn flat_surface() -> PlanetSurface {
        PlanetSurface {
            radius_m: 6_371_000.0,
            height: CubeField::new(4, vec![100.0; 4 * 4 * 6]),
            detail: DetailParams::default(),
            precipitation: None,
        }
    }

    #[test]
    fn children_tile_the_parent() {
        let node = NodeId {
            face: 2,
            level: 3,
            x: 5,
            y: 1,
        };
        let (min, size) = node.st_rect();
        for c in node.children() {
            let (cmin, csize) = c.st_rect();
            assert!((csize - size / 2.0).abs() < 1e-12);
            assert!(cmin.x >= min.x - 1e-12 && cmin.x + csize <= min.x + size + 1e-12);
            assert!(cmin.y >= min.y - 1e-12 && cmin.y + csize <= min.y + size + 1e-12);
        }
    }

    #[test]
    fn chunk_is_local_and_faces_outward() {
        let surface = flat_surface();
        let node = NodeId {
            face: 4,
            level: 10,
            x: 300,
            y: 700,
        };
        let chunk = build_chunk(node, &surface, 1.0);
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(pos)) =
            chunk.mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            panic!()
        };
        let edge = node.edge_m(surface.radius_m) as f32;
        assert!(pos.iter().all(|p| Vec3::from(*p).length() < edge * 1.5));
        let Some(bevy::mesh::VertexAttributeValues::Float32x3(normals)) =
            chunk.mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!()
        };
        let up = chunk.center.normalize().as_vec3();
        assert!(normals.iter().all(|n| Vec3::from(*n).dot(up) > 0.5));
    }

    #[test]
    fn far_nodes_hide_behind_the_horizon() {
        let r = 6_371_000.0;
        let cam = DVec3::Z * (r + 2.0);
        let near = NodeId {
            face: 4,
            level: 12,
            x: 2048,
            y: 2048,
        };
        let far = NodeId {
            face: 5,
            level: 12,
            x: 2048,
            y: 2048,
        };
        assert!(!beyond_horizon(&near, cam, r));
        assert!(beyond_horizon(&far, cam, r));
    }
}
