//! The planet: declares its simulation fields and passes, and renders the result.

pub mod mesh;

use bevy::{
    light::NotShadowCaster, prelude::*, render::render_resource::AsBindGroup, shader::ShaderRef,
};

use crate::sim::{FieldFormat, PassTrigger, SimAllocate, SimFields, SimGraph, SimPass};

/// Field holding terrain height in metres above the sea-level datum.
pub const HEIGHT: &str = "height";

const TERRAIN_INIT: &str = "terrain_init";
const TERRAIN_RELAX: &str = "terrain_relax";

/// Mirrors `TerrainParams` in `shaders/planet/terrain_params.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TerrainParams {
    pub planet_radius_m: f32,
    pub continent_frequency: f32,
    pub mountain_height_m: f32,
    pub ocean_depth_m: f32,
    /// Maximum stable slope (rise over run) before material slides, measured between
    /// neighbouring texels. At planet scale a texel spans kilometres, so this is far below
    /// the slope of real scree (~0.6).
    pub talus_slope: f32,
    /// Fraction of the excess moved per step; <= 0.25 for stability.
    pub erosion_rate: f32,
    pub _pad: [f32; 2],
}

impl Default for TerrainParams {
    fn default() -> Self {
        Self {
            planet_radius_m: 6_371_000.0,
            continent_frequency: 1.6,
            mountain_height_m: 6_500.0,
            ocean_depth_m: 4_500.0,
            talus_slope: 0.02,
            erosion_rate: 0.2,
            _pad: [0.0; 2],
        }
    }
}

pub struct PlanetPlugin {
    /// Vertices along one cube-face edge of the render mesh.
    pub mesh_subdivisions: u32,
    pub height_exaggeration: f32,
}

impl Default for PlanetPlugin {
    fn default() -> Self {
        Self {
            mesh_subdivisions: 256,
            height_exaggeration: 25.0,
        }
    }
}

#[derive(Resource, Clone, Copy)]
struct PlanetConfig {
    mesh_subdivisions: u32,
    height_exaggeration: f32,
}

impl Plugin for PlanetPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<PlanetMaterial>::default())
            .insert_resource(PlanetConfig {
                mesh_subdivisions: self.mesh_subdivisions,
                height_exaggeration: self.height_exaggeration,
            })
            .add_systems(Startup, declare_simulation)
            .add_systems(PostStartup, spawn_planet.after(SimAllocate));
    }
}

/// The whole planet simulation so far: one field and two passes. New subsystems
/// (hydraulic erosion, climate, vegetation, ...) add their fields and passes here.
fn declare_simulation(mut graph: ResMut<SimGraph>) {
    let params = TerrainParams::default();
    graph
        .add_field(HEIGHT, FieldFormat::R32Float)
        .add_pass(
            SimPass::new(
                TERRAIN_INIT,
                "shaders/planet/terrain_init.wesl",
                "terrain_init",
            )
            .write(HEIGHT)
            .trigger(PassTrigger::OnReset)
            .with_params(params),
        )
        .add_pass(
            SimPass::new(
                TERRAIN_RELAX,
                "shaders/planet/terrain_relax.wesl",
                "terrain_relax",
            )
            .read(HEIGHT)
            .write(HEIGHT)
            .with_params(params),
        );
}

/// Draws the planet directly from the simulation's height field.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct PlanetMaterial {
    #[texture(0, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub height: Handle<Image>,
    #[uniform(1)]
    pub params: PlanetMaterialParams,
}

#[derive(Clone, Copy, Debug, bevy::render::render_resource::ShaderType)]
pub struct PlanetMaterialParams {
    pub planet_radius_m: f32,
    pub height_exaggeration: f32,
    pub resolution: f32,
    pub _pad: f32,
}

impl Material for PlanetMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/planet/planet_material.wesl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/planet/planet_material.wesl".into()
    }
}

fn spawn_planet(
    mut commands: Commands,
    config: Res<PlanetConfig>,
    graph: Res<SimGraph>,
    fields: Res<SimFields>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
) {
    let material = materials.add(PlanetMaterial {
        height: fields.front(HEIGHT),
        params: PlanetMaterialParams {
            planet_radius_m: TerrainParams::default().planet_radius_m,
            height_exaggeration: config.height_exaggeration,
            resolution: graph.resolution as f32,
            _pad: 0.0,
        },
    });
    commands.spawn((
        Name::new("Planet"),
        Mesh3d(meshes.add(mesh::cube_sphere(config.mesh_subdivisions))),
        MeshMaterial3d(material),
        // The shadow pass uses the default vertex shader, which doesn't displace.
        NotShadowCaster,
        Transform::default(),
    ));
}
