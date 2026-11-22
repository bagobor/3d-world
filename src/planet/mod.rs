//! The planet: declares its simulation fields and passes, and renders the result.

pub mod mesh;

use bevy::{
    light::NotShadowCaster, prelude::*, render::render_resource::AsBindGroup, shader::ShaderRef,
};

use crate::sim::{FieldFormat, PassTrigger, SimAllocate, SimFields, SimGraph, SimPass, SimPhase};

/// Terrain height in metres above the sea-level datum.
pub const HEIGHT: &str = "height";
/// Water depth in metres (oceans hold the sea surface at the datum).
pub const WATER: &str = "water";
/// Suspended sediment, in metres of terrain.
pub const SEDIMENT: &str = "sediment";
/// Outflow to the (+x, -x, +y, -y) neighbours, in metres of water per step.
pub const FLUX: &str = "flux";
/// Smoothed inflow per step: high along rivers.
pub const DISCHARGE: &str = "discharge";
/// 0..1, carried inland from oceans, lakes and rivers.
pub const MOISTURE: &str = "moisture";
/// Mean annual temperature, °C (static: there is no climate simulation).
pub const TEMPERATURE: &str = "temperature";
/// Biome class index ([`Biome`]), stored as f32.
pub const BIOME: &str = "biome";

/// Bump when generation changes, so cached planets from older generators are not loaded.
pub const GENERATOR_VERSION: &str = "planet-v1";

/// Biome classes in the `biome` field. Keep in sync with `shaders/planet/env_params.wesl`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Biome {
    Ocean = 0,
    Lake,
    Ice,
    Tundra,
    Desert,
    BorealForest,
    Grassland,
    TemperateForest,
    Shrubland,
    TropicalForest,
    Savanna,
}

impl Biome {
    pub const ALL: [Biome; 11] = [
        Biome::Ocean,
        Biome::Lake,
        Biome::Ice,
        Biome::Tundra,
        Biome::Desert,
        Biome::BorealForest,
        Biome::Grassland,
        Biome::TemperateForest,
        Biome::Shrubland,
        Biome::TropicalForest,
        Biome::Savanna,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Biome::Ocean => "ocean",
            Biome::Lake => "lake",
            Biome::Ice => "ice",
            Biome::Tundra => "tundra",
            Biome::Desert => "desert",
            Biome::BorealForest => "boreal forest",
            Biome::Grassland => "grassland",
            Biome::TemperateForest => "temperate forest",
            Biome::Shrubland => "shrubland",
            Biome::TropicalForest => "tropical forest",
            Biome::Savanna => "savanna",
        }
    }
}

const TERRAIN_INIT: &str = "terrain_init";
const TERRAIN_RELAX: &str = "terrain_relax";
const HYDRO_INIT: &str = "hydro_init";
const HYDRO_FLUX: &str = "hydro_flux";
const HYDRO_WATER: &str = "hydro_water";
const HYDRO_ERODE: &str = "hydro_erode";
const MOISTURE_SPREAD: &str = "moisture_spread";
const GENERATION_FINALIZE: &str = "generation_finalize";

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

/// Mirrors `HydroParams` in `shaders/planet/hydro_params.wesl`. Units are documented there.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct HydroParams {
    pub planet_radius_m: f32,
    pub rain_m: f32,
    pub evaporation: f32,
    pub pipe_gain: f32,
    pub flux_damping: f32,
    pub capacity: f32,
    pub erosion_rate: f32,
    pub deposition_rate: f32,
    pub min_slope: f32,
    pub max_erosion_m: f32,
    pub discharge_smoothing: f32,
    pub _pad: f32,
}

impl Default for HydroParams {
    fn default() -> Self {
        Self {
            planet_radius_m: TerrainParams::default().planet_radius_m,
            rain_m: 0.005,
            evaporation: 0.005,
            pipe_gain: 0.01,
            flux_damping: 0.5,
            capacity: 1000.0,
            erosion_rate: 0.3,
            deposition_rate: 0.3,
            min_slope: 0.001,
            max_erosion_m: 10.0,
            discharge_smoothing: 0.05,
            _pad: 0.0,
        }
    }
}

/// Mirrors `EnvParams` in `shaders/planet/env_params.wesl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct EnvParams {
    pub planet_radius_m: f32,
    pub moisture_length_m: f32,
    pub barrier_height_m: f32,
    pub lake_depth_m: f32,
    pub river_discharge: f32,
    pub equator_temp_c: f32,
    pub pole_temp_c: f32,
    pub lapse_rate_c_per_m: f32,
    pub subtropic_dryness: f32,
    pub _pad: [f32; 3],
}

impl Default for EnvParams {
    fn default() -> Self {
        Self {
            planet_radius_m: TerrainParams::default().planet_radius_m,
            moisture_length_m: 900_000.0,
            barrier_height_m: 1_500.0,
            lake_depth_m: 20.0,
            river_discharge: 0.1,
            equator_temp_c: 28.0,
            pole_temp_c: -30.0,
            lapse_rate_c_per_m: 0.0065,
            subtropic_dryness: 0.6,
            _pad: [0.0; 3],
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
            .add_systems(PostStartup, spawn_planet.after(SimAllocate))
            .add_systems(Update, toggle_biome_view);
    }
}

/// The planet. Generation (run once, then cached) builds terrain, erodes it with water,
/// spreads moisture inland and finally derives static temperature and biome maps. Live
/// passes (ecosystem, ...) will be added with `SimPhase::Live`.
fn declare_simulation(mut graph: ResMut<SimGraph>) {
    let terrain = TerrainParams::default();
    let hydro = HydroParams::default();
    let env = EnvParams::default();
    let generate = |pass: SimPass| pass.phase(SimPhase::Generate);
    graph
        .add_field(HEIGHT, FieldFormat::R32Float)
        .add_field(WATER, FieldFormat::R32Float)
        .add_field(SEDIMENT, FieldFormat::R32Float)
        .add_field(FLUX, FieldFormat::Rgba32Float)
        .add_field(DISCHARGE, FieldFormat::R32Float)
        .add_field(MOISTURE, FieldFormat::R32Float)
        .add_field(TEMPERATURE, FieldFormat::R32Float)
        .add_field(BIOME, FieldFormat::R32Float)
        // Initial state.
        .add_pass(generate(
            SimPass::new(
                TERRAIN_INIT,
                "shaders/planet/terrain_init.wesl",
                "terrain_init",
            )
            .write(HEIGHT)
            .trigger(PassTrigger::OnReset)
            .with_params(terrain),
        ))
        .add_pass(generate(
            SimPass::new(HYDRO_INIT, "shaders/planet/hydro_init.wesl", "hydro_init")
                .read(HEIGHT)
                .write(WATER)
                .write(SEDIMENT)
                .write(FLUX)
                .write(DISCHARGE)
                .write(MOISTURE)
                .trigger(PassTrigger::OnReset)
                .with_params(hydro),
        ))
        // Every generation step: thermal erosion, moisture, then pipe-model hydraulics.
        .add_pass(generate(
            SimPass::new(
                TERRAIN_RELAX,
                "shaders/planet/terrain_relax.wesl",
                "terrain_relax",
            )
            .read(HEIGHT)
            .write(HEIGHT)
            .with_params(terrain),
        ))
        .add_pass(generate(
            SimPass::new(
                MOISTURE_SPREAD,
                "shaders/planet/moisture_spread.wesl",
                "moisture_spread",
            )
            .read(HEIGHT)
            .read(WATER)
            .read(DISCHARGE)
            .read(MOISTURE)
            .write(MOISTURE)
            .with_params(env),
        ))
        .add_pass(generate(
            SimPass::new(HYDRO_FLUX, "shaders/planet/hydro_flux.wesl", "hydro_flux")
                .read(HEIGHT)
                .read(WATER)
                .read(FLUX)
                .write(FLUX)
                .with_params(hydro),
        ))
        .add_pass(generate(
            SimPass::new(
                HYDRO_WATER,
                "shaders/planet/hydro_water.wesl",
                "hydro_water",
            )
            .read(HEIGHT)
            .read(WATER)
            .read(FLUX)
            .read(SEDIMENT)
            .read(DISCHARGE)
            .write(WATER)
            .write(SEDIMENT)
            .write(DISCHARGE)
            .read(MOISTURE)
            .with_params(hydro),
        ))
        .add_pass(generate(
            SimPass::new(
                HYDRO_ERODE,
                "shaders/planet/hydro_erode.wesl",
                "hydro_erode",
            )
            .read(HEIGHT)
            .read(SEDIMENT)
            .read(DISCHARGE)
            .write(HEIGHT)
            .write(SEDIMENT)
            .with_params(hydro),
        ))
        // Once at the end: settle sediment, derive temperature and biomes.
        .add_pass(
            SimPass::new(
                GENERATION_FINALIZE,
                "shaders/planet/generation_finalize.wesl",
                "generation_finalize",
            )
            .read(HEIGHT)
            .read(SEDIMENT)
            .read(WATER)
            .read(MOISTURE)
            .write(HEIGHT)
            .write(TEMPERATURE)
            .write(BIOME)
            .trigger(PassTrigger::AfterGeneration)
            .with_params(env),
        );
    // What a generated planet is: everything the renderer and live simulation read.
    for field in [HEIGHT, WATER, DISCHARGE, MOISTURE, TEMPERATURE, BIOME] {
        graph.bake(field);
    }
}

/// Draws the planet directly from the simulation's fields.
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct PlanetMaterial {
    #[texture(0, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub height: Handle<Image>,
    #[uniform(1)]
    pub params: PlanetMaterialParams,
    #[texture(2, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub water: Handle<Image>,
    #[texture(3, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub discharge: Handle<Image>,
    #[texture(4, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub temperature: Handle<Image>,
    #[texture(5, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub biome: Handle<Image>,
}

#[derive(Clone, Copy, Debug, bevy::render::render_resource::ShaderType)]
pub struct PlanetMaterialParams {
    pub planet_radius_m: f32,
    pub height_exaggeration: f32,
    pub resolution: f32,
    /// Discharge (m of water per step) above which land is drawn as river.
    pub river_discharge: f32,
    /// 1 once the static maps exist (after generation or a cache load).
    pub show_biomes: f32,
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
        water: fields.front(WATER),
        discharge: fields.front(DISCHARGE),
        temperature: fields.front(TEMPERATURE),
        biome: fields.front(BIOME),
        params: PlanetMaterialParams {
            planet_radius_m: TerrainParams::default().planet_radius_m,
            height_exaggeration: config.height_exaggeration,
            resolution: graph.resolution as f32,
            river_discharge: 0.1,
            show_biomes: 0.0,
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

/// Biome colours only exist once generation has finished (or the planet was loaded).
fn toggle_biome_view(
    stats: Res<crate::sim::SimStats>,
    planets: Query<&MeshMaterial3d<PlanetMaterial>>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
) {
    let show = if stats.live() { 1.0 } else { 0.0 };
    for handle in &planets {
        if materials
            .get(&handle.0)
            .is_some_and(|m| m.params.show_biomes != show)
            && let Some(mut material) = materials.get_mut(&handle.0)
        {
            material.params.show_biomes = show;
        }
    }
}
