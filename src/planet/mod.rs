//! The planet: declares its simulation fields and passes, and renders the result.

pub mod camera;
pub mod fauna;
pub mod lod;
pub mod mesh;
pub mod sky;
pub mod surface;
pub mod vegetation;

use bevy::{
    light::NotShadowCaster,
    math::DVec3,
    mesh::MeshVertexBufferLayoutRef,
    pbr::{MaterialPipeline, MaterialPipelineKey},
    prelude::*,
    render::render_resource::{
        AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError,
    },
    shader::ShaderRef,
};

use crate::sim::{
    FieldFormat, PassTrigger, SimAllocate, SimControl, SimFields, SimGraph, SimPass, SimPhase,
    cache::BakedFields,
};
use camera::{PlanetAnchor, PlanetCamera};
use lod::{LodSettings, TerrainLod};
use surface::{CubeField, DetailParams, PlanetSurface, precipitation_field};
use vegetation::{Vegetation, VegetationMaps, VegetationMaterial, VegetationParams};

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
/// Live plant cover: r grass, g shrubs, b trees (0..1 each), a fuel (dead litter).
pub const PLANTS: &str = "plants";
/// Live fire intensity (0..1) burning this step.
pub const FIRE: &str = "fire";
/// Live animal populations: r grazers, g predators (density, ~1 = what the best pasture
/// feeds), b grass and a shrub cover eaten this step.
pub const ANIMALS: &str = "animals";

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
const PLANTS_INIT: &str = "plants_init";
const PLANTS_GROW: &str = "plants_grow";
const FIRE_SPREAD: &str = "fire_spread";
const ANIMALS_INIT: &str = "animals_init";
const ANIMALS_STEP: &str = "animals_step";

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

/// Mirrors `PlantParams` in `shaders/planet/plant_params.wesl`. One live step is about a
/// month.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PlantParams {
    pub growth_grass: f32,
    pub growth_shrub: f32,
    pub growth_tree: f32,
    pub spread: f32,
    pub seed_rain: f32,
    pub litter_rate: f32,
    pub decay_rate: f32,
    pub ignition_rate: f32,
    pub spread_chance: f32,
    pub burn_grass: f32,
    pub burn_shrub: f32,
    pub burn_tree: f32,
    pub subtropic_dryness: f32,
    pub lake_depth_m: f32,
    pub _pad: [f32; 2],
}

impl Default for PlantParams {
    fn default() -> Self {
        let env = EnvParams::default();
        Self {
            growth_grass: 0.3,
            growth_shrub: 0.06,
            growth_tree: 0.025,
            spread: 1.0,
            seed_rain: 0.0005,
            litter_rate: 0.02,
            decay_rate: 0.04,
            ignition_rate: 5e-6,
            spread_chance: 0.7,
            burn_grass: 0.95,
            burn_shrub: 0.8,
            burn_tree: 0.6,
            subtropic_dryness: env.subtropic_dryness,
            lake_depth_m: env.lake_depth_m,
            _pad: [0.0; 2],
        }
    }
}

/// Mirrors `AnimalParams` in `shaders/planet/animal_params.wesl`, where the units are
/// documented. One live step is about a month.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AnimalParams {
    pub graze_rate: f32,
    pub graze_half: f32,
    pub grazer_gain: f32,
    pub grazer_death: f32,
    pub grazer_crowding: f32,
    pub hunt_rate: f32,
    pub hunt_half: f32,
    pub predator_gain: f32,
    pub predator_death: f32,
    pub grazer_diffusion: f32,
    pub grazer_taxis: f32,
    pub predator_diffusion: f32,
    pub predator_taxis: f32,
    pub immigration: f32,
    pub fire_kill: f32,
    pub _pad: f32,
}

impl Default for AnimalParams {
    fn default() -> Self {
        Self {
            graze_rate: 0.12,
            graze_half: 0.2,
            grazer_gain: 1.5,
            grazer_death: 0.05,
            grazer_crowding: 0.08,
            hunt_rate: 0.25,
            hunt_half: 0.3,
            predator_gain: 0.3,
            predator_death: 0.04,
            grazer_diffusion: 0.04,
            grazer_taxis: 0.15,
            predator_diffusion: 0.06,
            predator_taxis: 0.1,
            immigration: 0.0005,
            fire_kill: 0.5,
            _pad: 0.0,
        }
    }
}

pub struct PlanetPlugin {
    /// Vertices along one cube-face edge of the preview mesh shown while generating.
    pub mesh_subdivisions: u32,
    /// Vertical exaggeration of the generation preview.
    pub height_exaggeration: f32,
    /// The level-of-detail terrain shown once the planet is generated.
    pub terrain: LodSettings,
}

impl Default for PlanetPlugin {
    fn default() -> Self {
        Self {
            mesh_subdivisions: 256,
            height_exaggeration: 25.0,
            terrain: LodSettings::default(),
        }
    }
}

#[derive(Resource, Clone, Copy)]
struct PlanetConfig {
    mesh_subdivisions: u32,
    height_exaggeration: f32,
}

/// Systems that move the camera; the floating origin runs after them.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CameraControl;

impl Plugin for PlanetPlugin {
    fn build(&self, app: &mut App) {
        let radius = TerrainParams::default().planet_radius_m as f64;
        app.add_plugins((
            MaterialPlugin::<PlanetMaterial>::default(),
            MaterialPlugin::<TerrainMaterial>::default(),
            MaterialPlugin::<CloudMaterial>::default(),
            MaterialPlugin::<VegetationMaterial>::default(),
        ))
        .init_resource::<Vegetation>()
        .init_resource::<fauna::Fauna>()
        .insert_resource(PlanetConfig {
            mesh_subdivisions: self.mesh_subdivisions,
            height_exaggeration: self.height_exaggeration,
        })
        .insert_resource(TerrainLod::new(self.terrain))
        .insert_resource(PlanetCamera::new(radius))
        .add_systems(Startup, (declare_simulation, sky::spawn_atmosphere))
        .add_systems(PostStartup, spawn_planet.after(SimAllocate))
        .add_systems(
            Update,
            (
                toggle_biome_view,
                show_clouds,
                load_terrain_surface,
                camera::clamp_camera.after(CameraControl),
                lod::update_terrain.after(camera::clamp_camera),
                vegetation::update_vegetation.after(camera::clamp_camera),
                fauna::read_densities,
                fauna::update_fauna.after(camera::clamp_camera),
                camera::apply_floating_origin
                    .after(lod::update_terrain)
                    .after(vegetation::update_vegetation),
                swap_preview_for_terrain.after(lod::update_terrain),
                sky::adapt_atmosphere.after(camera::clamp_camera),
            ),
        );
    }
}

/// The terrain material, shared by every chunk.
#[derive(Resource)]
struct TerrainMaterialHandle(Handle<TerrainMaterial>);

/// The vegetation material, shared by every vegetation tile.
#[derive(Resource)]
struct VegetationMaterialHandle(Handle<VegetationMaterial>);

/// The GPU-displaced sphere shown while the planet is generating.
#[derive(Component)]
struct GenerationPreview;

/// When a planet is baked (generated or loaded), hand its height field to the LOD terrain.
fn load_terrain_surface(
    mut commands: Commands,
    baked: Res<BakedFields>,
    control: Res<SimControl>,
    material: Option<Res<TerrainMaterialHandle>>,
    vegetation_material: Option<Res<VegetationMaterialHandle>>,
    mut lod: ResMut<TerrainLod>,
    mut vegetation: ResMut<Vegetation>,
    mut loaded: Local<u64>,
) {
    let (Some(material), Some(vegetation_material)) = (material, vegetation_material) else {
        return;
    };
    if !baked.ready() || baked.version == *loaded {
        return;
    }
    *loaded = baked.version;
    let Some(height) = baked.f32_field(HEIGHT) else {
        error!("terrain: the baked planet has no `{HEIGHT}` field");
        return;
    };
    let surface = PlanetSurface {
        radius_m: TerrainParams::default().planet_radius_m as f64,
        height: CubeField::new(baked.resolution, height),
        detail: DetailParams {
            seed: control.seed,
            ..default()
        },
        precipitation: baked.f32_field(MOISTURE).map(|moisture| {
            precipitation_field(
                baked.resolution,
                &moisture,
                EnvParams::default().subtropic_dryness as f64,
            )
        }),
    };
    info!("terrain: new planet surface (seed {})", baked.seed);
    lod.set_surface(&mut commands, surface, material.0.clone());
    match (baked.f32_field(BIOME), baked.f32_field(TEMPERATURE)) {
        (Some(biome), Some(temperature)) => vegetation.set_planet(
            &mut commands,
            lod.surface().unwrap().clone(),
            VegetationMaps::new(baked.resolution, &biome, temperature),
            vegetation_material.0.clone(),
            lod.settings.exaggeration,
            control.seed,
        ),
        _ => error!("vegetation: the baked planet has no biome or temperature map"),
    }
}

/// Show the LOD terrain once it has something to draw; until then, the preview sphere.
fn swap_preview_for_terrain(
    lod: Res<TerrainLod>,
    stats: Res<crate::sim::SimStats>,
    mut previews: Query<&mut Visibility, With<GenerationPreview>>,
) {
    let terrain = lod.ready() && stats.live();
    for mut vis in &mut previews {
        vis.set_if_neq(if terrain {
            Visibility::Hidden
        } else {
            Visibility::Inherited
        });
    }
}

/// The planet. Generation (run once, then cached) builds terrain, erodes it with water,
/// spreads moisture inland and finally derives static temperature and biome maps. Live
/// passes run the plant ecosystem (growth, competition, wildfire) for the rest of the session.
fn declare_simulation(mut graph: ResMut<SimGraph>) {
    let terrain = TerrainParams::default();
    let hydro = HydroParams::default();
    let env = EnvParams::default();
    let plants = PlantParams::default();
    let animals = AnimalParams::default();
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
        .add_field(PLANTS, FieldFormat::Rgba32Float)
        .add_field(FIRE, FieldFormat::R32Float)
        .add_field(ANIMALS, FieldFormat::Rgba32Float)
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
        )
        // Live: the plant ecosystem, on top of the baked planet.
        .add_pass(
            SimPass::new(
                PLANTS_INIT,
                "shaders/planet/plants_init.wesl",
                "plants_init",
            )
            .read(TEMPERATURE)
            .read(MOISTURE)
            .read(BIOME)
            .write(PLANTS)
            .write(FIRE)
            .trigger(PassTrigger::OnReset)
            .with_params(plants),
        )
        .add_pass(
            SimPass::new(
                ANIMALS_INIT,
                "shaders/planet/animals_init.wesl",
                "animals_init",
            )
            .read(BIOME)
            .write(ANIMALS)
            .trigger(PassTrigger::OnReset)
            .with_params(animals),
        )
        // Animals feed first; plants_grow then takes what they ate off the cover.
        .add_pass(
            SimPass::new(
                ANIMALS_STEP,
                "shaders/planet/animals_step.wesl",
                "animals_step",
            )
            .read(ANIMALS)
            .read(PLANTS)
            .read(FIRE)
            .read(BIOME)
            .write(ANIMALS)
            .with_params(animals),
        )
        .add_pass(
            SimPass::new(
                PLANTS_GROW,
                "shaders/planet/plants_grow.wesl",
                "plants_grow",
            )
            .read(PLANTS)
            .read(FIRE)
            .read(TEMPERATURE)
            .read(MOISTURE)
            .read(BIOME)
            .read(ANIMALS)
            .write(PLANTS)
            .with_params(plants),
        )
        .add_pass(
            SimPass::new(
                FIRE_SPREAD,
                "shaders/planet/fire_spread.wesl",
                "fire_spread",
            )
            .read(PLANTS)
            .read(FIRE)
            .read(TEMPERATURE)
            .read(MOISTURE)
            .read(BIOME)
            .write(FIRE)
            .with_params(plants),
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
    #[texture(6, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub plants: Handle<Image>,
    #[texture(7, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub fire: Handle<Image>,
    #[texture(8, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub moisture: Handle<Image>,
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

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // The shader file also holds the terrain entry points.
        descriptor.vertex.entry_point = Some("vertex".into());
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.entry_point = Some("fragment".into());
        }
        Ok(())
    }
}

/// Draws LOD terrain chunks: same fields and shading as [`PlanetMaterial`], but positions
/// and normals come from the chunk mesh (see [`lod::build_chunk`]).
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct TerrainMaterial {
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
    #[texture(6, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub plants: Handle<Image>,
    #[texture(7, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub fire: Handle<Image>,
    #[texture(8, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub moisture: Handle<Image>,
}

impl Material for TerrainMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/planet/planet_material.wesl".into()
    }

    // Terrain casts shadows (mountains, hills).
    fn prepass_vertex_shader() -> ShaderRef {
        "shaders/planet/terrain_prepass.wesl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/planet/planet_material.wesl".into()
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
            lod::ATTRIBUTE_SURFACE.at_shader_location(2),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        descriptor.vertex.entry_point = Some("terrain_vertex".into());
        if let Some(fragment) = descriptor.fragment.as_mut() {
            fragment.entry_point = Some("terrain_fragment".into());
        }
        // Skirts are seen from both sides.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// Height of the cloud layer above sea level, in metres.
pub const CLOUD_HEIGHT_M: f32 = 6_000.0;

/// A thin cloud shell around the planet. Coverage is procedural, shaped by latitude bands
/// and the baked moisture map, and drifts slowly; the same function shades cloud shadows on
/// the ground (see `shaders/planet/cloud_field.wesl`).
#[derive(Asset, TypePath, AsBindGroup, Clone, Debug)]
pub struct CloudMaterial {
    #[uniform(0)]
    pub params: CloudParams,
    #[texture(1, dimension = "2d_array", sample_type = "float", filterable = false)]
    pub moisture: Handle<Image>,
}

#[derive(Clone, Copy, Debug, bevy::render::render_resource::ShaderType)]
pub struct CloudParams {
    pub planet_radius_m: f32,
    pub cloud_height_m: f32,
    pub resolution: f32,
    /// 0 hides the clouds (while generating, before moisture exists).
    pub visible: f32,
}

impl Material for CloudMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/planet/clouds.wesl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/planet/clouds.wesl".into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Blend
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Seen from above and from below.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}

/// The cloud shell: shown once the planet is live.
#[derive(Component)]
struct CloudShell;

fn show_clouds(
    stats: Res<crate::sim::SimStats>,
    shells: Query<&MeshMaterial3d<CloudMaterial>, With<CloudShell>>,
    mut materials: ResMut<Assets<CloudMaterial>>,
) {
    let visible = if stats.live() { 1.0 } else { 0.0 };
    for handle in &shells {
        if materials
            .get(&handle.0)
            .is_some_and(|m| m.params.visible != visible)
            && let Some(mut m) = materials.get_mut(&handle.0)
        {
            m.params.visible = visible;
        }
    }
}

fn spawn_planet(
    mut commands: Commands,
    config: Res<PlanetConfig>,
    graph: Res<SimGraph>,
    fields: Res<SimFields>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<PlanetMaterial>>,
    mut terrain_materials: ResMut<Assets<TerrainMaterial>>,
    mut cloud_materials: ResMut<Assets<CloudMaterial>>,
    mut vegetation_materials: ResMut<Assets<VegetationMaterial>>,
) {
    let radius = TerrainParams::default().planet_radius_m;
    let vegetation = vegetation_materials.add(VegetationMaterial {
        params: VegetationParams::new(radius, graph.resolution as f32),
        plants: fields.front(PLANTS),
        temperature: fields.front(TEMPERATURE),
        moisture: fields.front(MOISTURE),
    });
    commands.insert_resource(VegetationMaterialHandle(vegetation));
    let clouds = cloud_materials.add(CloudMaterial {
        params: CloudParams {
            planet_radius_m: radius,
            cloud_height_m: CLOUD_HEIGHT_M,
            resolution: graph.resolution as f32,
            visible: 0.0,
        },
        moisture: fields.front(MOISTURE),
    });
    commands.spawn((
        Name::new("Clouds"),
        Mesh3d(meshes.add(mesh::cube_sphere(192))),
        MeshMaterial3d(clouds),
        NotShadowCaster,
        Transform::from_scale(Vec3::splat(radius + CLOUD_HEIGHT_M)),
        PlanetAnchor(DVec3::ZERO),
        CloudShell,
    ));
    let terrain = terrain_materials.add(TerrainMaterial {
        height: fields.front(HEIGHT),
        water: fields.front(WATER),
        discharge: fields.front(DISCHARGE),
        temperature: fields.front(TEMPERATURE),
        biome: fields.front(BIOME),
        plants: fields.front(PLANTS),
        fire: fields.front(FIRE),
        moisture: fields.front(MOISTURE),
        params: PlanetMaterialParams {
            planet_radius_m: TerrainParams::default().planet_radius_m,
            height_exaggeration: 1.0,
            resolution: graph.resolution as f32,
            river_discharge: 0.1,
            show_biomes: 1.0,
        },
    });
    commands.insert_resource(TerrainMaterialHandle(terrain));
    let material = materials.add(PlanetMaterial {
        height: fields.front(HEIGHT),
        water: fields.front(WATER),
        discharge: fields.front(DISCHARGE),
        temperature: fields.front(TEMPERATURE),
        biome: fields.front(BIOME),
        plants: fields.front(PLANTS),
        fire: fields.front(FIRE),
        moisture: fields.front(MOISTURE),
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
        Transform::from_scale(Vec3::splat(TerrainParams::default().planet_radius_m)),
        PlanetAnchor(DVec3::ZERO),
        GenerationPreview,
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
