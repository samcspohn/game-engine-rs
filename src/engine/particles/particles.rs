use std::{
    cell::{Cell, SyncUnsafeCell},
    cmp::Reverse,
    collections::HashMap,
    mem::transmute,
    sync::{
        atomic::{AtomicI32, AtomicU32, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use crate::{
    editor::inspectable::{Inpsect, Ins, Inspectable_},
    engine::{
        atomic_vec::{self, AtomicVec},
        color_gradient::ColorGradient,
        particles::shaders::cs::a,
        project::asset_manager::{Asset, AssetInstance, AssetManager, AssetManagerBase},
        rendering::{
            self,
            camera::CameraViewData,
            component::buffer_usage_all,
            lighting::lighting_compute::{
                cs::li,
                lt::{self, tile},
            },
            texture::{self, Texture, TextureManager},
            vulkan_manager::VulkanManager,
        },
        storage::_Storage,
        time::Time,
        transform_compute::{self, cs::transform, TransformCompute},
        utils::{
            self,
            gpu_perf::{self, GpuPerf},
            PrimaryCommandBuffer,
        },
        world::{component::Component, transform::Transform, Sys, World},
    },
};
// use lazy_static::lazy::Lazy;

use crossbeam::{atomic::AtomicConsume, queue::SegQueue};
use dary_heap::BinaryHeap;
use egui::TextureId;
use egui_winit_vulkano::Gui;
use nalgebra_glm as glm;
use ncollide3d::pipeline;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, MutexGuard};
use rapier3d::na::ComplexField;
use segvec::SegVec;
use segvec::Slice;
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{
        allocator::SubbufferAllocator, view::BufferView, Buffer, BufferContents, BufferCreateInfo,
        Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, DispatchIndirectCommand, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{DescriptorBindingFlags, DescriptorSetLayout, DescriptorSetLayoutCreateInfo},
        DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    format::Format,
    image::{sampler::{Filter, Sampler, SamplerCreateInfo}, view::ImageView},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    padded::Padded,
    pipeline::{
        graphics::{
            color_blend::{
                AttachmentBlend, BlendFactor, BlendOp, ColorBlendAttachmentState, ColorBlendState,
                ColorBlendStateFlags, LogicOp,
            },
            depth_stencil::{CompareOp, DepthState, DepthStencilState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        layout::{PipelineDescriptorSetLayoutCreateInfo, PipelineLayoutCreateInfo},
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{RenderPass, Subpass},
    sync::{self, GpuFuture},
    DeviceSize,
};
// use super::cs::{{emitter_init, particle_template}, self};

use lazy_static::lazy_static;

// pub const MAX_PARTICLES: i32 = SETTINGS.read().get::<i32>("MAX_PARTICLES").unwrap();
// lazy_static! {
//     pub static ref _MAX_PARTICLES: i32 = crate::engine::utils::SETTINGS
//         .read()
//         .get::<i32>("MAX_PARTICLES")
//         .unwrap();
// }

use super::{
    asset::{ParticleTemplateManager, TEMPLATE_UPDATE},
    particle_sort::ParticleSort,
    particle_textures::ParticleTextures,
    shaders::{
        self,
        cs::{self, burst, emitter_init, particle, particle_template, Data},
        gs, scs,
    },
};

pub static mut PARTICLE_COL_OVER_LIFE_TEX: TextureId = TextureId::Managed(0);
pub static mut NUM_TEMPLATES: usize = 1;

pub struct PerformanceCounters {
    pub update_particles: i32,
    pub update_emitters: i32,
    pub init_emitters: i32,
    pub sort_particles: i32,
}
pub struct ParticleBuffers {
    pub particles: Subbuffer<[cs::particle]>,
    pub particle_positions_lifes: Subbuffer<[cs::pos_lif]>,
    // pub pos_life_compressed: Subbuffer<[scs::pos_life_comp]>,
    pub particle_next: Subbuffer<[i32]>,
    pub particle_template_ids: Subbuffer<[i32]>,
    pub alive: Subbuffer<[u32]>,
    pub alive_count: Subbuffer<u32>,
    pub emitters: Mutex<Subbuffer<[cs::emitter]>>,
    emitter_init_dummy: Subbuffer<[cs::emitter_init]>,
    particle_burst_dummy: Subbuffer<[cs::burst]>,
    pub particle_templates: Mutex<Subbuffer<[cs::particle_template]>>,
    pub avail: Subbuffer<[u32]>,
    pub avail_count: Subbuffer<u32>,
    // pub buffer_0: Subbuffer<i32>,
    pub indirect: Subbuffer<[DispatchIndirectCommand]>,
    pub particle_lighting: Subbuffer<[u32]>,
}

pub struct ParticlesSystem {
    pub sort: Mutex<ParticleSort>,
    pub max_curr_particles: AtomicU32,
    pub max_particles: AtomicI32,
    pub emitter_inits: AtomicVec<cs::emitter_init>,
    pub emitter_deinits: AtomicVec<cs::emitter_init>,
    pub particle_deinits: Mutex<BinaryHeap<(Reverse<u64>, u32)>>,
    pub emitter_max_particles: SyncUnsafeCell<Vec<(u64, u32)>>, // precomputed max particles for each emitter
    pub burst_deinits_accum: SyncUnsafeCell<Vec<AtomicU32>>,
    pub particle_burts: AtomicVec<cs::burst>,
    pub particle_templates: Arc<Mutex<_Storage<cs::particle_template>>>,
    pub particle_template_manager: Arc<Mutex<ParticleTemplateManager>>,
    pub particle_buffers: Mutex<ParticleBuffers>,
    pub compute_pipeline: Arc<ComputePipeline>,
    // pub compute_uniforms: Mutex<Vec<SubbufferAllocator>>,
    // pub cycle: SyncUnsafeCell<usize>,
    // pub render_uniforms: Mutex<SubbufferAllocator>,
    pub def_texture: Arc<ImageView>,
    pub def_sampler: Arc<Sampler>,
    pub vk: Arc<VulkanManager>,
    // pub performance: PerformanceCounters,
    pub particle_textures: Arc<Mutex<ParticleTextures>>,
    pub gpu_perf: Arc<GpuPerf>,
    pub start_time: Instant,
}
pub struct ParticleRenderPipeline {
    pub arc: Arc<GraphicsPipeline>,
}
impl ParticleRenderPipeline {
    pub fn new(vk: Arc<VulkanManager>, render_pass: Arc<RenderPass>) -> Self {
        let vs = shaders::vs::load(vk.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = shaders::fs::load(vk.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let gs = shaders::gs::load(vk.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        // let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        // let blend_state = ColorBlendState::new(subpass.num_color_attachments()).blend_alpha();
        let mut depth_stencil_state = DepthStencilState::simple_depth_test();
        depth_stencil_state.depth = Some(DepthState {
            write_enable: false,
            compare_op: CompareOp::Less,
        });

        let render_pipeline = utils::pipeline::graphics_pipeline(
            vk.clone(),
            &[vs.clone(), gs.clone(), fs.clone()],
            &[],
            |g| {
                g.input_assembly_state =
                    Some(InputAssemblyState::new().topology(PrimitiveTopology::PointList));
                g.rasterization_state = Some(RasterizationState::new().cull_mode(CullMode::None));
                g.depth_stencil_state = Some(depth_stencil_state);
                g.color_blend_state = Some(ColorBlendState::new(1).blend_alpha());
                // g.color_blend_state = Some(ColorBlendState {
                //     // logic_op: LogicOp::,
                //     attachments: vec![ColorBlendAttachmentState {
                //         blend: Some(AttachmentBlend {
                //             src_color_blend_factor:  BlendFactor::One,
                //             dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
                //             color_blend_op: BlendOp::Add,
                //             src_alpha_blend_factor: BlendFactor::One,
                //             dst_alpha_blend_factor: BlendFactor::Zero,
                //             alpha_blend_op: BlendOp::Add,
                //         }),
                //         ..Default::default()
                //         // color_write_mask: ,
                //         // color_write_enable: todo!(),
                //     }],
                //     // blend_constants: todo!(),
                //     ..Default::default()
                // });
                let mut layout_create_info = PipelineDescriptorSetLayoutCreateInfo::from_stages(&[
                    PipelineShaderStageCreateInfo::new(vs),
                    PipelineShaderStageCreateInfo::new(gs),
                    PipelineShaderStageCreateInfo::new(fs),
                ]);
                let binding = layout_create_info.set_layouts[0]
                    .bindings
                    .get_mut(&12)
                    .unwrap();
                binding.binding_flags |= DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
                binding.descriptor_count = 16;
                let pipeline_layout = PipelineLayout::new(
                    vk.device.clone(),
                    layout_create_info
                        .into_pipeline_layout_create_info(vk.device.clone())
                        .unwrap(),
                )
                .unwrap();

                g.layout = pipeline_layout;
                // GraphicsPipelineCreateInfo::layout(PipelineLayout::new(vk.device.clone(), PipelineDescriptorSetLayoutCreateInfo::))
            },
            render_pass.clone(),
        );

        Self {
            arc: render_pipeline,
        }
    }
}

pub struct ParticleDebugPipeline {
    pub arc: Arc<GraphicsPipeline>,
}
impl ParticleDebugPipeline {
    pub fn new(vk: Arc<VulkanManager>, render_pass: Arc<RenderPass>) -> Self {
        let vs = shaders::vs::load(vk.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = shaders::fs_d::load(vk.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let gs = shaders::gs_d::load(vk.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let shader1 = utils::pipeline::graphics_pipeline(
            vk.clone(),
            &[vs.clone(), gs.clone(), fs.clone()],
            &[],
            |g| {
                g.input_assembly_state =
                    Some(InputAssemblyState::new().topology(PrimitiveTopology::PointList));
                g.rasterization_state = Some(RasterizationState::new().cull_mode(CullMode::None));
            },
            render_pass.clone(),
        );

        Self { arc: shader1 }
    }
}

impl ParticlesSystem {
    pub fn new(
        vk: Arc<VulkanManager>,
        tex_man: Arc<Mutex<TextureManager>>,
        gpu_perf: Arc<GpuPerf>,
    ) -> ParticlesSystem {
        // let performance = PerformanceCounters {
        //     update_particles: vk.new_query(),
        //     update_emitters: vk.new_query(),
        //     init_emitters: vk.new_query(),
        //     sort_particles: vk.new_query(),
        // };
        let max_particles: i32 = 1;
        let particles = vk
            .buffer_array::<[cs::particle]>(max_particles as u64, MemoryTypeFilter::PREFER_DEVICE);
        // let particles = Buffer::new(&vk.mem_alloc, BufferCreateInfo{ usage: buffer_usage_all()

        // }, allocation_info, layout)
        // let particles = Buffer::<[cs::particle]>::array(
        //     &vk.mem_alloc,
        //     max_particles as vulkano::DeviceSize,
        //     buffer_usage_all(),
        //     vk.device.active_queue_family_indices().iter().copied(),
        // )
        // .unwrap();

        let particle_template_ids = vk.buffer_array(
            max_particles as vulkano::DeviceSize,
            MemoryTypeFilter::PREFER_DEVICE,
        );

        let mut builder = AutoCommandBufferBuilder::primary(
            &vk.comm_alloc,
            vk.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        println!("pos_lif: {}", std::mem::size_of::<cs::pos_lif>());

        // let particle_positions_lifes: Vec<cs::pos_lif> = (0..max_particles)
        //     .map(|_| cs::pos_lif {
        //         pos: [0., 0., 0.],
        //         life: 0.,
        //     })
        //     .collect();
        let copy_buffer = vk.buffer_from_iter((0..max_particles).map(|_| cs::pos_lif {
            pos: [0., 0., 0.],
            life: 0.,
        }));
        let particle_positions_lifes: Subbuffer<[cs::pos_lif]> = vk.buffer_array(
            max_particles as vulkano::DeviceSize,
            MemoryTypeFilter::PREFER_DEVICE,
        );
        // builder.fill_buffer(particle_positions_lifes, 0);

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_buffer,
                particle_positions_lifes.clone(),
            ))
            .unwrap();

        let particle_next = vk.buffer_array(
            max_particles as vulkano::DeviceSize,
            MemoryTypeFilter::PREFER_DEVICE,
        );

        // avail
        let copy_buffer = vk.buffer_from_iter(0..max_particles);
        let avail = vk.buffer_array(
            max_particles as vulkano::DeviceSize,
            MemoryTypeFilter::PREFER_DEVICE,
        );
        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, avail.clone()))
            .unwrap();

        let alive = vk.buffer_array(
            max_particles as vulkano::DeviceSize,
            MemoryTypeFilter::PREFER_DEVICE,
        );

        // avail_count
        let copy_buffer = vk.buffer_from_data(0u32);
        // Buffer::from_data(&vk.mem_alloc, buffer_usage_all(), false, 0i32).unwrap();
        // let avail_count = vk.buffer_from_data(0u32);
        // let alive_count = vk.buffer_from_data(0u32);

        let avail_count = vk.buffer_array(1, MemoryTypeFilter::PREFER_DEVICE);
        let alive_count = vk.buffer_array(1, MemoryTypeFilter::PREFER_DEVICE);

        // buffer 0
        // let buffer_0 = vk.buffer_from_data(0i32);

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_buffer.clone(),
                avail_count.clone(),
            ))
            .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_buffer.clone(),
                alive_count.clone(),
            ))
            .unwrap();
        // builder.fill_buffer(dst_buffer, data)
        // builder
        //     .copy_buffer(CopyBufferInfo::buffers(copy_buffer, buffer_0.clone()))
        //     .unwrap();

        // emitters
        let emitters = vk.buffer_array(1, MemoryTypeFilter::PREFER_DEVICE);

        // indirect
        let copy_buffer = vk.buffer_from_iter([DispatchIndirectCommand { x: 0, y: 1, z: 1 }]);
        let indirect = vk.buffer_array(1, MemoryTypeFilter::PREFER_DEVICE);
        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, indirect.clone()))
            .unwrap();

        unsafe {
            *super::asset::DEFAULT_TEXTURE.get() =
                AssetInstance::<Texture>::new(tex_man.lock().from_file("default/particle.png"));
        }
        let particle_textures = Arc::new(Mutex::new(ParticleTextures::new(
            tex_man,
            ParticleTextures::color_tex(&[[[255u8; 4]; 256]], &vk, &mut builder),
        )));

        let particle_templates = _Storage::new();
        let particle_templates = Arc::new(Mutex::new(particle_templates));
        let particle_template_manager = Arc::new(Mutex::new(ParticleTemplateManager::new(
            (particle_templates.clone(), particle_textures.clone()),
            &["ptem"],
        )));
        particle_template_manager
            .lock()
            .new_asset("res/default.ptem");

        let copy_buffer = vk.buffer_from_iter(particle_templates.lock().data.clone());
        let templates = vk.buffer_array(
            copy_buffer.len() as vulkano::DeviceSize,
            MemoryTypeFilter::PREFER_DEVICE,
        );

        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, templates.clone()))
            .unwrap();

        let (def_texture, def_sampler) =
            texture::texture_from_bytes(vk.clone(), &[255_u8, 255, 255, 255], 1, 1);

        // build buffer
        let command_buffer = builder.build().unwrap();

        let execute = sync::now(vk.device.clone())
            .boxed()
            .then_execute(vk.queue.clone(), command_buffer);

        match execute {
            Ok(execute) => {
                let future = execute.then_signal_fence_and_flush();
                match future {
                    Ok(_) => {}
                    // Err(FlushError::OutOfDate) => {}
                    Err(_e) => {}
                }
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
            }
        };
        let cs = cs::load(vk.device.clone()).unwrap();
        // let compute_pipeline = vulkano::pipeline::ComputePipeline::new(
        //     vk.device.clone(),
        //     cs.entry_point("main").unwrap(),
        //     &(),
        //     None,
        //     |_| {},
        // )
        // .expect("Failed to create compute shader");
        let compute_pipeline = utils::pipeline::compute_pipeline(vk.clone(), cs);

        // let uniforms = Mutex::new((0..2).map(|_| vk.sub_buffer_allocator()).collect());
        // let render_uniforms = Mutex::new(vk.sub_buffer_allocator());
        let emitter_init_dummy =
            vk.buffer_array(1 as vulkano::DeviceSize, MemoryTypeFilter::PREFER_DEVICE);
        let particle_burst_dummy =
            vk.buffer_array(1 as vulkano::DeviceSize, MemoryTypeFilter::PREFER_DEVICE);

        let mut emitter_max_particles = Vec::new();
        let mut burst_deinits_accum = Vec::new();
        // let v = unsafe { &mut *self.emmitter_max_particles.get() };
        // let v2 = unsafe { &mut *self.burst_deinits_accum.get() };
        emitter_max_particles.resize(particle_template_manager.lock().id_gen as usize, (0, 0));
        burst_deinits_accum.resize_with(particle_template_manager.lock().id_gen as usize, || {
            AtomicU32::new(0)
        });

        for (id, a) in particle_template_manager.lock().assets_id.iter() {
            let a = a.lock();
            let time_offset = Duration::from_secs_f32(a.max_lifetime).as_micros() as u64;
            emitter_max_particles[*id as usize] =
                (time_offset, (a.emission_rate * a.max_lifetime) as u32);
        }

        ParticlesSystem {
            sort: Mutex::new(ParticleSort::new(
                vk.clone(),
                gpu_perf.clone(),
                max_particles,
            )),
            emitter_inits: AtomicVec::new(),
            emitter_deinits: AtomicVec::new(),
            particle_deinits: Mutex::new(BinaryHeap::new()),
            max_curr_particles: AtomicU32::new(0u32),
            max_particles: AtomicI32::new(max_particles),
            emitter_max_particles: SyncUnsafeCell::new(emitter_max_particles),
            burst_deinits_accum: SyncUnsafeCell::new(burst_deinits_accum),
            particle_burts: AtomicVec::new(),
            particle_templates,
            particle_template_manager,
            particle_buffers: Mutex::new(ParticleBuffers {
                particles,
                particle_next,
                particle_positions_lifes,
                // pos_life_compressed: vk.buffer_array(max_particles as u64, MemoryTypeFilter::PREFER_DEVICE),
                particle_template_ids,
                emitter_init_dummy,
                particle_burst_dummy,
                emitters: Mutex::new(emitters),
                particle_templates: Mutex::new(templates),
                alive,
                alive_count,
                avail,
                avail_count,
                indirect,
                particle_lighting: vk
                    .buffer_array(65_536 * 32 + 1, MemoryTypeFilter::PREFER_DEVICE),
            }),
            // render_pipeline,
            compute_pipeline,
            // compute_uniforms: uniforms,
            // render_uniforms,
            def_texture,
            def_sampler,
            vk,
            gpu_perf,
            // performance,
            particle_textures,
            start_time: Instant::now(),
            // cycle: SyncUnsafeCell::new(0),
        }
    }
    fn get_descriptors2(
        &self,
        transform: Subbuffer<[transform]>,
        uniform_sub_buffer: Subbuffer<Data>,
    ) -> Arc<PersistentDescriptorSet> {
        let (emitter_init_dummy, particle_burst_dummy) = {
            let pb = self.particle_buffers.lock();
            (
                pb.emitter_init_dummy.clone(),
                pb.particle_burst_dummy.clone(),
            )
        };
        self.get_descriptors(
            transform,
            uniform_sub_buffer,
            emitter_init_dummy,
            particle_burst_dummy,
        )
    }
    fn get_descriptors(
        &self,
        transform: Subbuffer<[transform]>,
        uniform_sub_buffer: Subbuffer<Data>,
        emitter_inits: Subbuffer<impl ?Sized>,
        particle_bursts: Subbuffer<impl ?Sized>,
    ) -> Arc<PersistentDescriptorSet> {
        let pb = self.particle_buffers.lock();

        let descriptor_set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.compute_pipeline
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, transform),
                WriteDescriptorSet::buffer(1, pb.particles.clone()),
                WriteDescriptorSet::buffer(2, pb.particle_positions_lifes.clone()),
                WriteDescriptorSet::buffer(3, pb.particle_next.clone()),
                WriteDescriptorSet::buffer(4, pb.avail.clone()),
                WriteDescriptorSet::buffer(5, pb.emitters.lock().clone()),
                WriteDescriptorSet::buffer(6, pb.avail_count.clone()),
                WriteDescriptorSet::buffer(7, pb.particle_templates.lock().clone()),
                WriteDescriptorSet::buffer(8, uniform_sub_buffer),
                WriteDescriptorSet::buffer(9, pb.particle_template_ids.clone()),
                WriteDescriptorSet::buffer(10, emitter_inits.clone()),
                WriteDescriptorSet::buffer(11, pb.alive.clone()),
                WriteDescriptorSet::buffer(12, pb.alive_count.clone()),
                WriteDescriptorSet::buffer(13, pb.indirect.clone()),
                WriteDescriptorSet::buffer(15, particle_bursts.clone()),
            ],
            [],
        )
        .unwrap();
        descriptor_set
    }
    pub fn update(
        &self,
        builder: &mut utils::PrimaryCommandBuffer,
        particle_init_data: (usize, Vec<emitter_init>, Vec<emitter_init>, Vec<burst>),
        transform_compute: &TransformCompute,
        time: &Time,
        gui: &mut Gui,
    ) {
        {
            let mut pb = self.particle_buffers.lock();
            let m_particles = self.max_curr_particles.load_consume() as u64;
            if pb.particles.len() < m_particles {
                let new_size = m_particles.next_power_of_two();
                self.max_particles.store(new_size as i32, Ordering::Relaxed);
                // increase particle buffer size
                let buf = self
                    .vk
                    .buffer_array(new_size, MemoryTypeFilter::PREFER_DEVICE);
                builder
                    .copy_buffer(CopyBufferInfo::buffers(pb.particles.clone(), buf.clone()))
                    .unwrap();
                pb.particles = buf;

                // increase particle template buffer size
                let buf = self
                    .vk
                    .buffer_array(new_size, MemoryTypeFilter::PREFER_DEVICE);
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        pb.particle_template_ids.clone(),
                        buf.clone(),
                    ))
                    .unwrap();
                pb.particle_template_ids = buf;

                // increase particle position buffer size
                let copy_buf =
                    self.vk
                        .buffer_from_iter((0..new_size as u32).map(|_| cs::pos_lif {
                            pos: [0., 0., 0.],
                            life: 0.,
                        }));
                let buf = self
                    .vk
                    .buffer_array(new_size, MemoryTypeFilter::PREFER_DEVICE);
                builder
                    .copy_buffer(CopyBufferInfo::buffers(copy_buf, buf.clone()))
                    .unwrap(); // todo: write shader to init
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        pb.particle_positions_lifes.clone(),
                        buf.clone(),
                    ))
                    .unwrap();
                pb.particle_positions_lifes = buf;

                // increase particle next buffer size
                let buf: Subbuffer<[i32]> = self
                    .vk
                    .buffer_array(new_size, MemoryTypeFilter::PREFER_DEVICE);
                // builder.fill_buffer(buf.clone(), ~0);
                let buf2: Subbuffer<[u32]> = unsafe { transmute(buf.clone()) };
                let d: u32 = unsafe { transmute(-1i32) };
                builder.fill_buffer(buf2, d).unwrap();
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        pb.particle_next.clone(),
                        buf.clone(),
                    ))
                    .unwrap();
                pb.particle_next = buf;

                // increase particle avail buffer size
                let copy_buf = self.vk.buffer_from_iter(0..new_size as u32);
                let buf = self
                    .vk
                    .buffer_array(new_size, MemoryTypeFilter::PREFER_DEVICE);
                builder
                    .copy_buffer(CopyBufferInfo::buffers(copy_buf, buf.clone()))
                    .unwrap();
                builder
                    .copy_buffer(CopyBufferInfo::buffers(pb.avail.clone(), buf.clone()))
                    .unwrap();
                pb.avail = buf;

                // increase particle alive buffer size
                let buf = self
                    .vk
                    .buffer_array(new_size, MemoryTypeFilter::PREFER_DEVICE);
                builder
                    .copy_buffer(CopyBufferInfo::buffers(pb.alive.clone(), buf.clone()))
                    .unwrap();
                pb.alive = buf;

                self.sort.lock().resize(new_size as i32);
            }
        }

        self.update_templates(builder, gui);
        self.update_emitter_len(builder, particle_init_data.0);
        builder.bind_pipeline_compute(self.compute_pipeline.clone());
        self.particle_bursts(
            builder,
            transform_compute.gpu_transforms.clone(),
            particle_init_data.3,
            time,
        );
        self.emitter_deinit(
            builder,
            transform_compute.gpu_transforms.clone(),
            particle_init_data.2,
            time,
        );

        self.emitter_init(
            builder,
            transform_compute.gpu_transforms.clone(),
            particle_init_data.1,
            time,
        );
        let a = self.gpu_perf.node("emitter update", builder);
        self.emitter_update(
            builder,
            transform_compute.gpu_transforms.clone(),
            particle_init_data.0,
            time,
        );
        a.end(builder);

        let a = self.gpu_perf.node("particle update", builder);
        self.particle_update(builder, transform_compute.gpu_transforms.clone(), time);
        a.end(builder);
    }

    pub fn reduce_particles(&self) {
        // reduce particle count
        let inst = Instant::now();
        let inst_u64 = inst.duration_since(self.start_time.clone()).as_micros() as u64;
        let mut particle_deinits = self.particle_deinits.lock();

        while let Some((Reverse(inst), count)) = particle_deinits.peek() {
            if inst < &inst_u64 {
                let (_, count) = particle_deinits.pop().unwrap();
                self.max_curr_particles.fetch_sub(count, Ordering::Relaxed);
            } else {
                break;
            }
        }
    }

    fn update_templates(&self, builder: &mut PrimaryCommandBuffer, gui: &mut Gui) {
        let pb = self.particle_buffers.lock();
        let mut pt = self.particle_templates.lock();
        if TEMPLATE_UPDATE.load(Ordering::Relaxed) {
            let v = unsafe { &mut *self.emitter_max_particles.get() };
            let v2 = unsafe { &mut *self.burst_deinits_accum.get() };
            v.resize(
                self.particle_template_manager.lock().id_gen as usize,
                (0, 0),
            );
            v2.resize_with(
                self.particle_template_manager.lock().id_gen as usize,
                || AtomicU32::new(0),
            );
            let mut ptex = self.particle_textures.lock();
            let mut colors: Vec<[[u8; 4]; 256]> = Vec::new();
            // colors.push([[255u8;4];256]);
            let mut count = 0;
            for (id, a) in self.particle_template_manager.lock().assets_id.iter() {
                let mut a = a.lock();
                a.index = count;
                let time_offset = Duration::from_secs_f32(a.max_lifetime).as_micros() as u64;
                v[*id as usize] = (
                    time_offset,
                    (a.emission_rate * a.max_lifetime).ceil() as u32,
                );
                if a.trail {
                    v[*id as usize].1 += 1;
                }
                *pt.get_mut(id) = a.gen_particle_template(&mut ptex);
                colors.push(a.color_over_life.to_color_array());
                count += 1;
            }
            println!("colors len: {}", colors.len());
            println!("template len: {}", pt.data.len());

            ptex.color_tex = ParticleTextures::color_tex(colors.as_slice(), &self.vk, builder);
            unsafe {
                PARTICLE_COL_OVER_LIFE_TEX = gui.register_user_image_view(ptex.color_tex.0.clone(), SamplerCreateInfo {
                    mag_filter: Filter::Nearest,
                    min_filter: Filter::Nearest,
                   ..Default::default()
                });
                NUM_TEMPLATES = count
            }
        }

        let copy_buffer = self.vk.buffer_from_iter(pt.data.iter().copied());
        drop(pt);

        if copy_buffer.len() > pb.particle_templates.lock().len() {
            *pb.particle_templates.lock() = self.vk.buffer_array(
                copy_buffer.len() as vulkano::DeviceSize,
                MemoryTypeFilter::PREFER_DEVICE,
            );
        }
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_buffer,
                pb.particle_templates.lock().clone(),
            ))
            .unwrap();
        TEMPLATE_UPDATE.store(false, Ordering::Relaxed);
    }
    fn update_emitter_len(&self, builder: &mut PrimaryCommandBuffer, num_emitters: usize) {
        let pb = self.particle_buffers.lock();
        let max_len = num_emitters.next_power_of_two();
        if pb.emitters.lock().len() < max_len as u64 {
            let buf = self.vk.buffer_array(
                max_len as vulkano::DeviceSize,
                MemoryTypeFilter::PREFER_DEVICE,
            );
            let a: Subbuffer<[u32]> = unsafe { transmute(buf.clone()) };

            builder.fill_buffer(a, 0u32).unwrap();

            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    pb.emitters.lock().clone(),
                    buf.clone(),
                ))
                .unwrap();
            *pb.emitters.lock() = buf;
        }
    }

    fn particle_bursts(
        &self,
        builder: &mut PrimaryCommandBuffer,
        transform: Subbuffer<[transform]>,
        particle_bursts: Vec<cs::burst>,
        time: &Time,
    ) {
        let inst_u64 = Instant::now()
            .duration_since(self.start_time.clone())
            .as_micros() as u64;
        let burst_deinits_accum = unsafe { &*self.burst_deinits_accum.get() };
        for (id, count) in burst_deinits_accum.iter().enumerate() {
            if count.load(Ordering::Relaxed) > 0 {
                let time_offset = { unsafe { &*self.emitter_max_particles.get() } }[id as usize].0;
                self.particle_deinits.lock().push((
                    Reverse(inst_u64 + time_offset),
                    count.load(Ordering::Relaxed),
                ));
                count.store(0, Ordering::Relaxed);
            }
        }

        // let pb = self.particle_buffers.lock();

        if particle_bursts.is_empty() {
            return;
        };
        let len = particle_bursts.len();

        let particle_bursts = self.vk.buffer_from_iter(particle_bursts);
        let max_particles: i32 = self.max_particles.load_consume() as i32;
        let uniform_sub_buffer = self.vk.allocate(cs::Data {
            num_jobs: len as i32,
            dt: time.dt,
            time: time.time as f32,
            stage: 0,
            MAX_PARTICLES: max_particles,
        });

        let dummy = {
            let mut pb = self.particle_buffers.lock();
            pb.emitter_init_dummy.clone()
        };
        let descriptor_set =
            self.get_descriptors(transform, uniform_sub_buffer, dummy, particle_bursts);

        builder.bind_descriptor_sets(
            PipelineBindPoint::Compute,
            self.compute_pipeline.layout().clone(),
            0, // Bind this descriptor set to index 0.
            descriptor_set,
        );
        // self.vk.query(&self.performance.init_emitters, builder);
        builder.dispatch([len as u32 / 1024 + 1, 1, 1]).unwrap();
        // self.vk.end_query(&self.performance.init_emitters, builder)
    }

    pub fn emitter_deinit(
        &self,
        builder: &mut utils::PrimaryCommandBuffer,
        transform: Subbuffer<[transform]>,
        emitter_deinits: Vec<cs::emitter_init>,
        time: &Time,
    ) {
        // let pb = self.particle_buffers.lock();

        if emitter_deinits.is_empty() {
            return;
        };
        let len = emitter_deinits.len();
        let mut emitter_deinit_accum = HashMap::new();
        let v = unsafe { &*self.emitter_max_particles.get() };
        let emitter_deinits: Subbuffer<[cs::emitter_init]> = {
            // let iter = iter.into_iter();
            let buffer: Subbuffer<[emitter_init]> = Buffer::new_unsized(
                self.vk.mem_alloc.clone(),
                BufferCreateInfo {
                    usage: buffer_usage_all(),
                    ..Default::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
                emitter_deinits.len() as vulkano::DeviceSize,
            )
            .unwrap();

            {
                let mut write_guard = buffer.write().unwrap();

                for (o, i) in write_guard.iter_mut().zip(emitter_deinits.into_iter()) {
                    emitter_deinit_accum
                        .entry(i.template_id)
                        .and_modify(|e| *e += v[i.template_id as usize].1)
                        .or_insert(v[i.template_id as usize].1);
                    *o = i;
                }
            }
            buffer
        };
        let inst_u64 = Instant::now()
            .duration_since(self.start_time.clone())
            .as_micros() as u64;
        let mut particle_deinits = self.particle_deinits.lock();
        for (id, count) in emitter_deinit_accum.iter() {
            let time_offset =
                particle_deinits.push((Reverse(inst_u64 + v[*id as usize].0), *count));
        }
        // let emitter_deinits = self.vk.buffer_from_iter(emitter_deinits);
        // let emitter_deinits = self
        //     .vk
        //     .buffer_array::<[emitter_init]>(len as vulkano::DeviceSize, MemoryTypeFilter::PREFER_DEVICE); // TODO: cache

        // builder
        //     .copy_buffer(CopyBufferInfo::buffers(
        //         copy_buffer,
        //         emitter_deinits.clone(),
        //     ))
        //     .unwrap();

        let max_particles: i32 = self.max_particles.load_consume() as i32;
        let uniform_sub_buffer = self.vk.allocate(cs::Data {
            num_jobs: len as i32,
            dt: time.dt,
            time: time.time as f32,
            stage: 1,
            MAX_PARTICLES: max_particles,
        });

        let dummy = {
            let mut pb = self.particle_buffers.lock();
            pb.particle_burst_dummy.clone()
        };
        let descriptor_set =
            self.get_descriptors(transform, uniform_sub_buffer, emitter_deinits, dummy);

        builder.bind_descriptor_sets(
            PipelineBindPoint::Compute,
            self.compute_pipeline.layout().clone(),
            0, // Bind this descriptor set to index 0.
            descriptor_set,
        );
        // self.vk.query(&self.performance.init_emitters, builder);
        builder.dispatch([len as u32 / 1024 + 1, 1, 1]).unwrap();
        // self.vk.end_query(&self.performance.init_emitters, builder)
    }

    pub fn emitter_init(
        &self,
        builder: &mut utils::PrimaryCommandBuffer,
        transform: Subbuffer<[transform]>,
        emitter_inits: Vec<cs::emitter_init>,
        time: &Time,
    ) {
        // let pb = self.particle_buffers.lock();
        if emitter_inits.is_empty() {
            return;
        };
        let len = emitter_inits.len();

        let emitter_inits = self.vk.buffer_from_iter(emitter_inits);
        // let emitter_inits = self
        //     .vk
        //     .buffer_array::<[emitter_init]>(len as vulkano::DeviceSize, MemoryTypeFilter::PREFER_DEVICE); // TODO: cache

        // builder
        //     .copy_buffer(CopyBufferInfo::buffers(copy_buffer, emitter_inits.clone()))
        //     .unwrap();

        let max_particles: i32 = self.max_particles.load_consume() as i32;
        let uniform_sub_buffer = self.vk.allocate(cs::Data {
            num_jobs: len as i32,
            dt: time.dt,
            time: time.time as f32,
            stage: 2,
            MAX_PARTICLES: max_particles,
        });
        let dummy = {
            let mut pb = self.particle_buffers.lock();
            pb.particle_burst_dummy.clone()
        };
        let descriptor_set =
            self.get_descriptors(transform, uniform_sub_buffer, emitter_inits, dummy);

        builder.bind_descriptor_sets(
            PipelineBindPoint::Compute,
            self.compute_pipeline.layout().clone(),
            0, // Bind this descriptor set to index 0.
            descriptor_set,
        );
        // self.vk.query(&self.performance.init_emitters, builder);
        builder.dispatch([len as u32 / 1024 + 1, 1, 1]).unwrap();
        // self.vk.end_query(&self.performance.init_emitters, builder)
    }

    pub fn emitter_update(
        &self,
        builder: &mut utils::PrimaryCommandBuffer,
        transform: Subbuffer<[transform]>,
        emitter_len: usize,
        time: &Time,
    ) {
        // let pb = self.particle_buffers.lock();

        let emitter_len = emitter_len.max(1);
        let max_particles: i32 = self.max_particles.load_consume() as i32;

        let uniform_sub_buffer = self.vk.allocate(cs::Data {
            num_jobs: emitter_len as i32,
            dt: time.dt,
            time: time.time as f32,
            stage: 3,
            MAX_PARTICLES: max_particles,
        });
        let descriptor_set = self.get_descriptors2(transform, uniform_sub_buffer);

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .dispatch([emitter_len as u32 / 1024 + 1, 1, 1])
            .unwrap();
    }

    pub fn particle_update(
        &self,
        builder: &mut utils::PrimaryCommandBuffer,
        transform: Subbuffer<[transform]>,
        time: &Time,
    ) {
        let mut pb = self.particle_buffers.lock();
        let max_particles: i32 = self.max_particles.load_consume() as i32;
        let mut uniform_data = cs::Data {
            num_jobs: max_particles,
            dt: time.dt,
            time: time.time as f32,
            stage: 4,
            MAX_PARTICLES: max_particles,
        };
        let uniform_sub_buffer = self.vk.allocate(uniform_data);

        let alive_count = pb.alive_count.clone();
        let indirect = pb.indirect.clone();
        drop(pb);

        let descriptor_set = self.get_descriptors2(transform.clone(), uniform_sub_buffer);

        builder
            .update_buffer(alive_count, &0u32)
            // .copy_buffer(CopyBufferInfo::buffers(
            //     pb.buffer_0.clone(),
            //     pb.alive_count.clone(),
            // ))
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set,
            )
            .unwrap()
            .dispatch([max_particles as u32 / 1024 + 1, 1, 1])
            .unwrap();
        // set indirect
        uniform_data.num_jobs = 1;
        uniform_data.stage = 5;
        let uniform_sub_buffer = self.vk.allocate(uniform_data);
        // let uniform_sub_buffer = self.compute_uniforms.lock()[unsafe { *self.cycle.get() }]
        //     .allocate_sized()
        //     .unwrap();
        *uniform_sub_buffer.write().unwrap() = uniform_data;
        let descriptor_set = self.get_descriptors2(transform.clone(), uniform_sub_buffer);

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .dispatch([1, 1, 1])
            .unwrap();
        // dispatch indirect particle update
        uniform_data.num_jobs = -1;
        uniform_data.stage = 6;
        let uniform_sub_buffer = self.vk.allocate(uniform_data);

        // let uniform_sub_buffer = self.compute_uniforms.lock()[unsafe { *self.cycle.get() }]
        //     .allocate_sized()
        //     .unwrap();
        *uniform_sub_buffer.write().unwrap() = uniform_data;
        let descriptor_set = self.get_descriptors2(transform.clone(), uniform_sub_buffer);

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .dispatch_indirect(indirect.clone())
            .unwrap();
    }
    pub fn debug_particles(
        &self,
        particle_debug_pipeline: &ParticleDebugPipeline,
        builder: &mut utils::PrimaryCommandBuffer,
        cvd: &CameraViewData,
        transform: Subbuffer<[transform]>,

        lights: Subbuffer<[crate::engine::rendering::lighting::lighting_compute::lt::light]>,
        light_templates: Subbuffer<[crate::engine::rendering::pipeline::fs::lightTemplate]>,
        tiles: Subbuffer<[lt::tile]>,
    ) {
        let pb = self.particle_buffers.lock();
        let uniform_sub_buffer = self.vk.allocate(gs::Data {
            view: cvd.view.into(),
            proj: cvd.proj.into(),
            cam_inv_rot: cvd.inv_rot.into(),
            cam_rot: cvd.cam_rot.coords.into(),
            cam_pos: cvd.cam_pos.into(),
            num_templates: pb.particle_templates.lock().len() as u32,
            screen_dims: [cvd.dimensions[0] as f32, cvd.dimensions[1] as f32],
            // _dummy0: Default::default(),
        });
        let pt = self.particle_textures.lock();
        let set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            particle_debug_pipeline
                .arc
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            [
                // WriteDescriptorSet::buffer(0, pb.pos_life_compressed.clone()),
                WriteDescriptorSet::buffer(0, pb.particle_positions_lifes.clone()),
                WriteDescriptorSet::buffer(3, pb.particle_templates.lock().clone()),
                WriteDescriptorSet::buffer(4, uniform_sub_buffer),
                WriteDescriptorSet::buffer(5, self.sort.lock().a2.clone()),
                WriteDescriptorSet::buffer(6, pb.particle_template_ids.clone()),
                WriteDescriptorSet::buffer(7, pb.particle_next.clone()),
                WriteDescriptorSet::buffer(8, transform),
                WriteDescriptorSet::buffer(9, pb.particles.clone()),
            ],
            [],
        )
        .unwrap();
        // unsafe {
        //     self.vk.query(&*RENDER_QUERY, builder);
        // }
        builder
            .bind_pipeline_graphics(particle_debug_pipeline.arc.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                particle_debug_pipeline.arc.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .draw_indirect(self.sort.lock().draw.clone())
            .unwrap();
        // unsafe {
        //     self.vk.end_query(&*RENDER_QUERY, builder);
        // }
    }

    pub fn render_particles(
        &self,
        particle_render_pipeline: &ParticleRenderPipeline,
        builder: &mut utils::PrimaryCommandBuffer,
        cvd: &CameraViewData,
        transform: Subbuffer<[transform]>,

        lights: Subbuffer<[crate::engine::rendering::lighting::lighting_compute::lt::light]>,
        light_templates: Subbuffer<[crate::engine::rendering::pipeline::fs::lightTemplate]>,
        light_ids: Subbuffer<[u32]>,
        light_blh: Subbuffer<[rendering::lighting::lighting_compute::cs::BoundingLine]>,
        tiles: Subbuffer<[lt::tile]>,
    ) {
        // static mut RENDER_QUERY: Lazy<i32> = Lazy::new(|| -1);
        // unsafe {
        //     if *RENDER_QUERY == -1 {
        //         *RENDER_QUERY = self.vk.new_query();
        //     }
        // }
        // let r = self.gpu_perf.node("particle render", builder);
        let pb = self.particle_buffers.lock();
        builder.fill_buffer(pb.particle_lighting.clone(), 0);
        let uniform_sub_buffer = self.vk.allocate(gs::Data {
            view: cvd.view.into(),
            proj: cvd.proj.into(),
            cam_inv_rot: cvd.inv_rot.into(),
            cam_rot: cvd.cam_rot.coords.into(),
            cam_pos: cvd.cam_pos.into(),
            num_templates: pb.particle_templates.lock().len() as u32,
            screen_dims: cvd.dimensions.into(),
            // _dummy0: Default::default(),
        });
        let pt = self.particle_textures.lock();
        let set = PersistentDescriptorSet::new_variable(
            &self.vk.desc_alloc,
            particle_render_pipeline
                .arc
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            pt.samplers.len() as u32,
            [
                // WriteDescriptorSet::buffer(0, pb.pos_life_compressed.clone()),
                WriteDescriptorSet::buffer(0, pb.particle_positions_lifes.clone()),
                WriteDescriptorSet::buffer(3, pb.particle_templates.lock().clone()),
                WriteDescriptorSet::buffer(4, uniform_sub_buffer),
                WriteDescriptorSet::buffer(5, self.sort.lock().a2.clone()),
                WriteDescriptorSet::buffer(6, pb.particle_template_ids.clone()),
                WriteDescriptorSet::buffer(7, pb.particle_next.clone()),
                WriteDescriptorSet::buffer(8, transform),
                WriteDescriptorSet::buffer(9, pb.particles.clone()),
                WriteDescriptorSet::buffer(10, pb.particle_lighting.clone()),
                WriteDescriptorSet::image_view_sampler(
                    11,
                    pt.color_tex.0.clone(),
                    pt.color_tex.1.clone(),
                ),
                WriteDescriptorSet::image_view_sampler_array(
                    12,
                    0,
                    pt.samplers.iter().map(|a| (a.0.clone() as _, a.1.clone())),
                ),
            ],
            [],
        )
        .unwrap();

        let light_layout = particle_render_pipeline
            .arc
            .layout()
            .set_layouts()
            .get(1)
            .unwrap();
        let light_desc = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            light_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, light_templates),
                WriteDescriptorSet::buffer(1, lights),
                WriteDescriptorSet::buffer(2, tiles),
                // WriteDescriptorSet::buffer(3, light_ids),
                WriteDescriptorSet::buffer(4, light_blh),
            ],
            [],
        )
        .unwrap();

        // unsafe {
        //     self.vk.query(&*RENDER_QUERY, builder);
        // }
        builder
            .bind_pipeline_graphics(particle_render_pipeline.arc.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                particle_render_pipeline.arc.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                particle_render_pipeline.arc.layout().clone(),
                1,
                light_desc,
            )
            .unwrap()
            .draw_indirect(self.sort.lock().draw.clone())
            .unwrap();
        // r.end(builder);
        // unsafe {
        //     self.vk.end_query(&*RENDER_QUERY, builder);
        // }
    }
}
