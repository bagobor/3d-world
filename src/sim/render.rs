//! Render-world side of the GPU simulation: pipelines, bind groups and dispatch.

use std::{borrow::Cow, num::NonZeroU64};

use bevy::app::SubApp;

use bevy::{
    core_pipeline::schedule::camera_driver,
    prelude::*,
    render::{
        Render, RenderSystems,
        render_asset::RenderAssets,
        render_resource::*,
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue},
        texture::GpuImage,
    },
    shader::ShaderCacheError,
};

use super::{
    CUBE_FACES, FieldAccess, MAX_STEPS_PER_FRAME, PassTrigger, SimControl, SimFields, SimGraph,
    SimStats, WORKGROUP_SIZE,
};

/// Size of one per-step globals slot. 256 is the largest
/// `min_uniform_buffer_offset_alignment` any wgpu backend reports.
const GLOBALS_SLOT: u64 = 256;

/// Mirrors `SimGlobals` in `shaders/sim/common.wesl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SimGlobals {
    resolution: u32,
    step: u32,
    seed: u32,
    total_steps: u32,
    dt: f32,
    time: f32,
    inv_resolution: f32,
    _pad: f32,
}

pub(super) fn build_render_app(render_app: &mut SubApp) {
    render_app
        .init_resource::<SimRuntime>()
        .add_systems(
            Render,
            prepare_sim
                .in_set(RenderSystems::PrepareBindGroups)
                .run_if(resource_exists::<SimGraph>)
                .run_if(resource_exists::<SimFields>),
        )
        .add_systems(
            RenderGraph,
            run_sim
                .before(camera_driver)
                .run_if(resource_exists::<SimGpu>),
        );
}

/// Per-pass GPU objects.
struct PassGpu {
    layout: BindGroupLayoutDescriptor,
    pipeline: CachedComputePipelineId,
    params: Buffer,
    bind_group: Option<BindGroup>,
}

/// GPU objects for the whole graph, created on the first frame the graph is available.
#[derive(Resource)]
struct SimGpu {
    passes: Vec<PassGpu>,
    globals: Buffer,
    /// Keeps imported shader modules alive while pipelines compile.
    _shaders: Vec<Handle<Shader>>,
}

/// Simulation clock and this frame's plan, owned by the render world.
#[derive(Resource, Default)]
struct SimRuntime {
    ready: bool,
    failed: bool,
    reset_generation: Option<u32>,
    step: u64,
    total_steps: u64,
    time: f32,
    /// Steps to dispatch this frame: (globals slot, step index since reset).
    plan: Vec<(u32, u64)>,
}

fn prepare_sim(
    mut commands: Commands,
    graph: Res<SimGraph>,
    fields: Res<SimFields>,
    control: Res<SimControl>,
    stats: Res<SimStats>,
    gpu: Option<ResMut<SimGpu>>,
    mut runtime: ResMut<SimRuntime>,
    asset_server: Res<AssetServer>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let Some(mut gpu) = gpu else {
        commands.insert_resource(create_gpu(&graph, &asset_server, &pipeline_cache, &device));
        return;
    };
    runtime.plan.clear();

    // Pipelines: wait for all, so OnReset passes never get skipped.
    if !runtime.ready {
        if runtime.failed {
            return;
        }
        let mut all_ok = true;
        for (pass, pass_gpu) in graph.passes.iter().zip(&gpu.passes) {
            match pipeline_cache.get_compute_pipeline_state(pass_gpu.pipeline) {
                CachedPipelineState::Ok(_) => {}
                CachedPipelineState::Err(ShaderCacheError::ShaderNotLoaded(_)) => all_ok = false,
                CachedPipelineState::Err(err) => {
                    error!(
                        "sim pass `{}` ({}) failed to compile:\n{err}",
                        pass.label, pass.shader
                    );
                    runtime.failed = true;
                    return;
                }
                _ => all_ok = false,
            }
        }
        if !all_ok {
            return;
        }
    }

    // Bind groups: created once, as soon as every field texture exists on the GPU.
    let gpu = &mut *gpu;
    for (pass, pass_gpu) in graph.passes.iter().zip(gpu.passes.iter_mut()) {
        if pass_gpu.bind_group.is_none() {
            pass_gpu.bind_group = create_bind_group(
                pass,
                pass_gpu,
                &gpu.globals,
                &fields,
                &gpu_images,
                &pipeline_cache,
                &device,
            );
        }
    }
    if gpu.passes.iter().any(|p| p.bind_group.is_none()) {
        return;
    }
    if !runtime.ready {
        info!("sim: all {} pass pipelines ready", graph.passes.len());
        runtime.ready = true;
    }

    // Params change at runtime; they are tiny, so upload every frame.
    for (pass, pass_gpu) in graph.passes.iter().zip(&gpu.passes) {
        queue.write_buffer(&pass_gpu.params, 0, &pass.params);
    }

    // Reset handling.
    if runtime.reset_generation != Some(control.reset_generation()) {
        runtime.reset_generation = Some(control.reset_generation());
        runtime.step = 0;
        runtime.time = 0.0;
        // Step 0 always runs on reset, even while paused, so OnReset passes initialise
        // the fields.
        runtime.plan.push((0, 0));
    }

    if !control.paused {
        let steps = control.steps_per_frame.min(MAX_STEPS_PER_FRAME);
        let already = runtime.plan.len() as u64;
        for i in already..(already + steps as u64).min(MAX_STEPS_PER_FRAME as u64) {
            let step = runtime.step + i;
            runtime.plan.push((i as u32, step));
        }
    }

    // Upload one globals slot per planned step.
    if !runtime.plan.is_empty() {
        let mut bytes = vec![0u8; runtime.plan.len() * GLOBALS_SLOT as usize];
        for (n, &(_, step)) in runtime.plan.iter().enumerate() {
            let globals = SimGlobals {
                resolution: graph.resolution,
                step: step as u32,
                seed: control.seed,
                total_steps: (runtime.total_steps + n as u64) as u32,
                dt: control.dt,
                time: runtime.time + control.dt * n as f32,
                inv_resolution: 1.0 / graph.resolution as f32,
                _pad: 0.0,
            };
            let at = n * GLOBALS_SLOT as usize;
            bytes[at..at + size_of::<SimGlobals>()].copy_from_slice(bytemuck::bytes_of(&globals));
        }
        queue.write_buffer(&gpu.globals, 0, &bytes);

        let planned = runtime.plan.len() as u64;
        // The step counter after this frame is the last planned step + 1.
        runtime.step = runtime.plan.last().unwrap().1 + 1;
        runtime.total_steps += planned;
        runtime.time += control.dt * planned as f32;
    }

    stats.publish(runtime.step, runtime.total_steps, runtime.ready);
}

fn create_gpu(
    graph: &SimGraph,
    asset_server: &AssetServer,
    pipeline_cache: &PipelineCache,
    device: &RenderDevice,
) -> SimGpu {
    let globals = device.create_buffer(&BufferDescriptor {
        label: Some("sim globals"),
        size: GLOBALS_SLOT * MAX_STEPS_PER_FRAME as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    // Shared module imported by every pass shader.
    let shaders = vec![asset_server.load::<Shader>("shaders/sim/common.wesl")];

    let passes = graph
        .passes
        .iter()
        .map(|pass| {
            let mut entries = vec![
                uniform_entry(0, true, size_of::<SimGlobals>() as u64),
                uniform_entry(1, false, pass.params.len() as u64),
            ];
            for (i, binding) in pass.bindings.iter().enumerate() {
                let format = graph.field(binding.field).unwrap().format.texture_format();
                let ty = match binding.access {
                    FieldAccess::Read => BindingType::Texture {
                        sample_type: read_sample_type(format),
                        view_dimension: TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    FieldAccess::Write => BindingType::StorageTexture {
                        access: StorageTextureAccess::WriteOnly,
                        format,
                        view_dimension: TextureViewDimension::D2Array,
                    },
                };
                entries.push(BindGroupLayoutEntry {
                    binding: 2 + i as u32,
                    visibility: ShaderStages::COMPUTE,
                    ty,
                    count: None,
                });
            }
            let layout = BindGroupLayoutDescriptor::new(format!("sim {}", pass.label), &entries);
            let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
                label: Some(Cow::Owned(format!("sim {}", pass.label))),
                layout: vec![layout.clone()],
                shader: asset_server.load(pass.shader),
                entry_point: Some(Cow::Borrowed(pass.entry_point)),
                ..default()
            });
            let params = device.create_buffer(&BufferDescriptor {
                label: Some(pass.label),
                size: pass.params.len() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            PassGpu {
                layout,
                pipeline,
                params,
                bind_group: None,
            }
        })
        .collect();

    SimGpu {
        passes,
        globals,
        _shaders: shaders,
    }
}

fn uniform_entry(binding: u32, dynamic: bool, size: u64) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: NonZeroU64::new(size),
        },
        count: None,
    }
}

fn read_sample_type(format: TextureFormat) -> TextureSampleType {
    match format.sample_type(None, None) {
        Some(TextureSampleType::Float { .. }) => TextureSampleType::Float { filterable: false },
        Some(other) => other,
        None => TextureSampleType::Float { filterable: false },
    }
}

fn create_bind_group(
    pass: &super::SimPass,
    pass_gpu: &PassGpu,
    globals: &Buffer,
    fields: &SimFields,
    gpu_images: &RenderAssets<GpuImage>,
    pipeline_cache: &PipelineCache,
    device: &RenderDevice,
) -> Option<BindGroup> {
    // Resolve every texture view first; bail out if any image isn't uploaded yet.
    let mut views = Vec::with_capacity(pass.bindings.len());
    for binding in &pass.bindings {
        let field = &fields.0[binding.field];
        let handle = match binding.access {
            FieldAccess::Read => &field.front,
            // Read+write in one pass: write the back buffer, copied to front afterwards.
            FieldAccess::Write if pass.reads(binding.field) => field.back.as_ref().unwrap(),
            FieldAccess::Write => &field.front,
        };
        views.push(&gpu_images.get(handle)?.texture_view);
    }

    let mut entries = vec![
        BindGroupEntry {
            binding: 0,
            resource: BindingResource::Buffer(BufferBinding {
                buffer: globals,
                offset: 0,
                size: NonZeroU64::new(size_of::<SimGlobals>() as u64),
            }),
        },
        BindGroupEntry {
            binding: 1,
            resource: pass_gpu.params.as_entire_binding(),
        },
    ];
    for (i, view) in views.into_iter().enumerate() {
        entries.push(BindGroupEntry {
            binding: 2 + i as u32,
            resource: BindingResource::TextureView(view),
        });
    }
    Some(device.create_bind_group(
        pass.label,
        &pipeline_cache.get_bind_group_layout(&pass_gpu.layout),
        &entries,
    ))
}

fn run_sim(
    mut render_context: RenderContext,
    graph: Res<SimGraph>,
    fields: Res<SimFields>,
    gpu: Res<SimGpu>,
    runtime: Res<SimRuntime>,
    pipeline_cache: Res<PipelineCache>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    if !runtime.ready || runtime.plan.is_empty() {
        return;
    }
    let groups = graph.resolution.div_ceil(WORKGROUP_SIZE);
    let encoder = render_context.command_encoder();

    for &(slot, step) in &runtime.plan {
        for (pass, pass_gpu) in graph.passes.iter().zip(&gpu.passes) {
            let due = match pass.trigger {
                PassTrigger::OnReset => step == 0,
                PassTrigger::EveryStep => true,
                PassTrigger::EveryNSteps(n) => step.is_multiple_of(n.max(1) as u64),
            };
            if !due {
                continue;
            }
            let (Some(pipeline), Some(bind_group)) = (
                pipeline_cache.get_compute_pipeline(pass_gpu.pipeline),
                pass_gpu.bind_group.as_ref(),
            ) else {
                continue;
            };
            {
                let mut cpass = encoder.begin_compute_pass(&ComputePassDescriptor {
                    label: Some(pass.label),
                    ..default()
                });
                cpass.set_pipeline(pipeline);
                cpass.set_bind_group(0, bind_group, &[slot * GLOBALS_SLOT as u32]);
                cpass.dispatch_workgroups(groups, groups, CUBE_FACES);
            }
            // Publish double-buffered writes so the next pass (and rendering) sees them.
            for binding in &pass.bindings {
                if binding.access != FieldAccess::Write || !pass.reads(binding.field) {
                    continue;
                }
                let field = &fields.0[binding.field];
                let (Some(back), Some(front)) = (
                    field.back.as_ref().and_then(|h| gpu_images.get(h)),
                    gpu_images.get(&field.front),
                ) else {
                    continue;
                };
                encoder.copy_texture_to_texture(
                    back.texture.as_image_copy(),
                    front.texture.as_image_copy(),
                    back.texture.size(),
                );
            }
        }
    }
}
