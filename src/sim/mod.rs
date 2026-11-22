//! Reusable GPU simulation framework.
//!
//! A simulation is described declaratively in the main world:
//!
//! * **Fields** ([`SimFieldDesc`]) are per-texel planet state stored on the GPU as
//!   cube-sphere texture arrays (6 layers, one per cube face, `resolution × resolution`
//!   texels each). Height, water, temperature, vegetation density, ... are all fields.
//! * **Passes** ([`SimPass`]) are WGSL/WESL compute entry points that read some fields and
//!   write others. Passes run in declaration order, once per simulation step.
//!
//! The framework owns everything a pass would otherwise repeat: allocating the textures,
//! building bind group layouts and bind groups, compiling pipelines, double-buffering any
//! field that a pass both reads and writes, per-step global uniforms, typed per-pass
//! parameters, reset, pause and time-slicing (a fixed number of steps per rendered frame,
//! because wgpu gives us a single queue shared with rendering).
//!
//! ## Shader binding contract
//!
//! Every pass shader sees, in `@group(0)`:
//!
//! | binding | contents |
//! |---|---|
//! | 0 | `var<uniform> globals: SimGlobals` (see `shaders/sim/common.wesl`) |
//! | 1 | `var<uniform> params: <your params struct>` (16 zero bytes if the pass has none) |
//! | 2.. | one binding per entry of [`SimPass::bindings`], in order |
//!
//! A [`FieldAccess::Read`] binding is a `texture_2d_array<f32>` (or `<u32>`/`<i32>` for
//! integer formats) read with `textureLoad`. A [`FieldAccess::Write`] binding is a
//! `texture_storage_2d_array<FORMAT, write>`. A pass that writes a field must write every
//! texel of it.
//!
//! The default dispatch covers every texel: `ceil(res / 8) × ceil(res / 8) × 6`
//! workgroups, so shaders use `@workgroup_size(8, 8, 1)` and `global_invocation_id.z`
//! is the cube face.

mod render;

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use bevy::{
    asset::RenderAssetUsages,
    platform::collections::HashMap,
    prelude::*,
    render::{
        RenderApp,
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        render_resource::{
            Extent3d, TextureDimension, TextureFormat, TextureUsages, TextureViewDescriptor,
            TextureViewDimension,
        },
    },
};

/// Number of cube faces; every field is a texture array with this many layers.
pub const CUBE_FACES: u32 = 6;

/// Workgroup edge used by the default dispatch (`@workgroup_size(8, 8, 1)`).
pub const WORKGROUP_SIZE: u32 = 8;

/// Upper bound on [`SimControl::steps_per_frame`]; sizes the per-step globals buffer.
pub const MAX_STEPS_PER_FRAME: u32 = 64;

/// Adds the GPU simulation runtime. Configure it with [`SimGraph`] before `Startup` ends.
pub struct GpuSimPlugin {
    /// Texels along one edge of each cube face.
    pub resolution: u32,
}

impl Plugin for GpuSimPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(SimGraph::new(self.resolution))
            .init_resource::<SimControl>()
            .init_resource::<SimStats>()
            .add_plugins((
                ExtractResourcePlugin::<SimGraph>::default(),
                ExtractResourcePlugin::<SimFields>::default(),
                ExtractResourcePlugin::<SimControl>::default(),
                ExtractResourcePlugin::<SimStats>::default(),
            ))
            // Fields are allocated after every plugin's Startup systems have declared them.
            .add_systems(PostStartup, allocate_fields.in_set(SimAllocate));

        if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
            render::build_render_app(render_app);
        }
    }
}

/// `PostStartup` set that allocates field textures. Order systems that need
/// [`SimFields`] during start-up `.after(SimAllocate)`.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SimAllocate;

/// Element format of a field. Kept to formats that are valid as write-only storage
/// textures on every wgpu backend we target.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FieldFormat {
    R32Float,
    Rg32Float,
    Rgba32Float,
    Rgba16Float,
    R32Uint,
}

impl FieldFormat {
    pub fn texture_format(self) -> TextureFormat {
        match self {
            FieldFormat::R32Float => TextureFormat::R32Float,
            FieldFormat::Rg32Float => TextureFormat::Rg32Float,
            FieldFormat::Rgba32Float => TextureFormat::Rgba32Float,
            FieldFormat::Rgba16Float => TextureFormat::Rgba16Float,
            FieldFormat::R32Uint => TextureFormat::R32Uint,
        }
    }

    /// Bytes per texel, for memory accounting and readback.
    pub fn texel_bytes(self) -> u32 {
        match self {
            FieldFormat::R32Float | FieldFormat::R32Uint => 4,
            FieldFormat::Rg32Float | FieldFormat::Rgba16Float => 8,
            FieldFormat::Rgba32Float => 16,
        }
    }
}

/// Declares one cube-sphere field.
#[derive(Clone, Debug)]
pub struct SimFieldDesc {
    pub name: &'static str,
    pub format: FieldFormat,
}

/// How a pass binds a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldAccess {
    /// Sampled `texture_2d_array`, read with `textureLoad`.
    Read,
    /// Write-only storage `texture_storage_2d_array`.
    Write,
}

#[derive(Clone, Debug)]
pub struct FieldBinding {
    pub field: &'static str,
    pub access: FieldAccess,
}

/// When a pass runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassTrigger {
    /// Once, at the first step after start-up or after [`SimControl::request_reset`].
    OnReset,
    /// Every simulation step.
    EveryStep,
    /// Every `n`th simulation step (step counter starts at 0 after a reset).
    EveryNSteps(u32),
}

/// One compute pass: a shader entry point plus the fields it touches.
#[derive(Clone, Debug)]
pub struct SimPass {
    /// Unique, human readable; also used for GPU debug labels and profiling scopes.
    pub label: &'static str,
    /// Asset path of the shader, e.g. `"shaders/planet/terrain_init.wesl"`.
    pub shader: &'static str,
    pub entry_point: &'static str,
    pub bindings: Vec<FieldBinding>,
    pub trigger: PassTrigger,
    /// Raw bytes of the params uniform (binding 1). Use [`SimPass::with_params`].
    pub params: Vec<u8>,
}

impl SimPass {
    pub fn new(label: &'static str, shader: &'static str, entry_point: &'static str) -> Self {
        Self {
            label,
            shader,
            entry_point,
            bindings: Vec::new(),
            trigger: PassTrigger::EveryStep,
            params: vec![0; 16],
        }
    }

    pub fn read(mut self, field: &'static str) -> Self {
        self.bindings.push(FieldBinding {
            field,
            access: FieldAccess::Read,
        });
        self
    }

    pub fn write(mut self, field: &'static str) -> Self {
        self.bindings.push(FieldBinding {
            field,
            access: FieldAccess::Write,
        });
        self
    }

    pub fn trigger(mut self, trigger: PassTrigger) -> Self {
        self.trigger = trigger;
        self
    }

    /// Sets the params uniform. `T` must match the WGSL struct layout (uniform address
    /// space rules: keep it to 16-byte aligned vec4/f32/u32 groups).
    pub fn with_params<T: bytemuck::Pod>(mut self, params: T) -> Self {
        self.set_params(params);
        self
    }

    pub fn set_params<T: bytemuck::Pod>(&mut self, params: T) {
        let mut bytes = bytemuck::bytes_of(&params).to_vec();
        bytes.resize(bytes.len().next_multiple_of(16).max(16), 0);
        self.params = bytes;
    }

    fn reads(&self, field: &str) -> bool {
        self.bindings
            .iter()
            .any(|b| b.field == field && b.access == FieldAccess::Read)
    }

    fn writes(&self, field: &str) -> bool {
        self.bindings
            .iter()
            .any(|b| b.field == field && b.access == FieldAccess::Write)
    }
}

/// The declared simulation: fields plus ordered passes.
///
/// Structure (fields, pass list, shaders, bindings) is fixed once `Startup` ends. Pass
/// params may change at any time with [`SimGraph::pass_mut`]; they are re-uploaded every
/// frame.
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct SimGraph {
    pub resolution: u32,
    pub fields: Vec<SimFieldDesc>,
    pub passes: Vec<SimPass>,
}

impl SimGraph {
    fn new(resolution: u32) -> Self {
        assert!(resolution > 0, "simulation resolution must be positive");
        Self {
            resolution,
            fields: Vec::new(),
            passes: Vec::new(),
        }
    }

    pub fn add_field(&mut self, name: &'static str, format: FieldFormat) -> &mut Self {
        assert!(
            self.fields.iter().all(|f| f.name != name),
            "sim field `{name}` declared twice"
        );
        self.fields.push(SimFieldDesc { name, format });
        self
    }

    pub fn add_pass(&mut self, pass: SimPass) -> &mut Self {
        assert!(
            self.passes.iter().all(|p| p.label != pass.label),
            "sim pass `{}` declared twice",
            pass.label
        );
        for b in &pass.bindings {
            assert!(
                self.fields.iter().any(|f| f.name == b.field),
                "sim pass `{}` binds undeclared field `{}`",
                pass.label,
                b.field
            );
        }
        self.passes.push(pass);
        self
    }

    pub fn pass_mut(&mut self, label: &str) -> Option<&mut SimPass> {
        self.passes.iter_mut().find(|p| p.label == label)
    }

    pub fn field(&self, name: &str) -> Option<&SimFieldDesc> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// A field needs a back buffer if any single pass both reads and writes it.
    fn needs_back_buffer(&self, field: &str) -> bool {
        self.passes
            .iter()
            .any(|p| p.reads(field) && p.writes(field))
    }

    /// GPU memory used by all fields, in bytes.
    pub fn memory_bytes(&self) -> u64 {
        let texels = (self.resolution as u64).pow(2) * CUBE_FACES as u64;
        self.fields
            .iter()
            .map(|f| {
                let copies = if self.needs_back_buffer(f.name) { 2 } else { 1 };
                texels * f.format.texel_bytes() as u64 * copies
            })
            .sum()
    }
}

/// GPU textures backing a field.
#[derive(Clone, Debug)]
pub struct SimField {
    /// The current state. Stable for the lifetime of the app, so materials and readbacks
    /// can hold on to it.
    pub front: Handle<Image>,
    /// Write target for passes that read and write the field in one dispatch; copied back
    /// into `front` right after such a pass.
    pub back: Option<Handle<Image>>,
    pub format: FieldFormat,
}

/// Allocated field textures, keyed by field name. Available from `PostStartup` on.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct SimFields(pub HashMap<&'static str, SimField>);

impl SimFields {
    pub fn front(&self, name: &str) -> Handle<Image> {
        self.0
            .get(name)
            .unwrap_or_else(|| panic!("unknown sim field `{name}`"))
            .front
            .clone()
    }
}

/// Runtime controls, written by the app (UI, keys), read by the render world each frame.
#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct SimControl {
    pub paused: bool,
    /// Simulation steps dispatched per rendered frame. This is the time-slicing knob: the
    /// simulation shares the single wgpu queue with rendering, so its GPU cost lands in
    /// every frame.
    pub steps_per_frame: u32,
    /// Seed exposed to shaders as `globals.seed`.
    pub seed: u32,
    /// Simulated time per step, exposed as `globals.dt`.
    pub dt: f32,
    reset_generation: u32,
}

impl Default for SimControl {
    fn default() -> Self {
        Self {
            paused: false,
            steps_per_frame: 1,
            seed: 1,
            dt: 1.0,
            reset_generation: 0,
        }
    }
}

impl SimControl {
    /// Restarts the simulation: `OnReset` passes run again and the step counter returns to 0.
    pub fn request_reset(&mut self) {
        self.reset_generation = self.reset_generation.wrapping_add(1);
    }

    pub fn reset_generation(&self) -> u32 {
        self.reset_generation
    }
}

/// Counters published by the render world, readable from the main world.
#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct SimStats(Arc<SimStatsInner>);

#[derive(Default)]
struct SimStatsInner {
    step: AtomicU64,
    total_steps: AtomicU64,
    ready: AtomicU64,
}

impl SimStats {
    /// Steps since the last reset.
    pub fn step(&self) -> u64 {
        self.0.step.load(Ordering::Relaxed)
    }

    /// Steps since start-up, across resets.
    pub fn total_steps(&self) -> u64 {
        self.0.total_steps.load(Ordering::Relaxed)
    }

    /// True once every pass pipeline has compiled and the simulation is running.
    pub fn ready(&self) -> bool {
        self.0.ready.load(Ordering::Relaxed) != 0
    }

    pub(crate) fn publish(&self, step: u64, total: u64, ready: bool) {
        self.0.step.store(step, Ordering::Relaxed);
        self.0.total_steps.store(total, Ordering::Relaxed);
        self.0.ready.store(ready as u64, Ordering::Relaxed);
    }
}

fn allocate_fields(
    mut commands: Commands,
    graph: Res<SimGraph>,
    mut images: ResMut<Assets<Image>>,
) {
    let res = graph.resolution;
    let mut fields = HashMap::default();
    for desc in &graph.fields {
        let mut make = || images.add(new_field_image(res, desc.format));
        let front = make();
        let back = graph.needs_back_buffer(desc.name).then(make);
        fields.insert(
            desc.name,
            SimField {
                front,
                back,
                format: desc.format,
            },
        );
    }
    info!(
        "sim: {} fields, {} passes, {}x{}x6 texels, {:.1} MiB of field textures",
        graph.fields.len(),
        graph.passes.len(),
        res,
        res,
        graph.memory_bytes() as f64 / (1024.0 * 1024.0)
    );
    commands.insert_resource(SimFields(fields));
}

fn new_field_image(resolution: u32, format: FieldFormat) -> Image {
    let mut image = Image::new_uninit(
        Extent3d {
            width: resolution,
            height: resolution,
            depth_or_array_layers: CUBE_FACES,
        },
        TextureDimension::D2,
        format.texture_format(),
        RenderAssetUsages::RENDER_WORLD,
    );
    image.texture_descriptor.usage = TextureUsages::STORAGE_BINDING
        | TextureUsages::TEXTURE_BINDING
        | TextureUsages::COPY_SRC
        | TextureUsages::COPY_DST;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    image
}
