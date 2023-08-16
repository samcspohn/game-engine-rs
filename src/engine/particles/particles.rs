use std::sync::{
    atomic::{AtomicI32, AtomicUsize, Ordering},
    Arc,
};

use crate::{
    editor::inspectable::{Inpsect, Ins, Inspectable, Inspectable_},
    engine::{
        color_gradient::ColorGradient,
        project::asset_manager::{Asset, AssetInstance, AssetManager, AssetManagerBase},
        rendering::{
            renderer_component::buffer_usage_all,
            texture::{Texture, TextureManager},
            vulkan_manager::VulkanManager,
        },
        storage::_Storage,
        time::Time,
        transform_compute::{self, cs::ty::transform, TransformCompute},
        utils::PrimaryCommandBuffer,
        world::{
            component::{Component, _ComponentID},
            transform::Transform,
            Sys, World,
        },
    },
};
// use lazy_static::lazy::Lazy;

use component_derive::{AssetID, ComponentID};
use crossbeam::queue::SegQueue;
use nalgebra_glm as glm;
use parking_lot::{Mutex, MutexGuard};
use serde::{Deserialize, Serialize};
use sync_unsafe_cell::SyncUnsafeCell;
use vulkano::{
    buffer::{
        cpu_pool::CpuBufferPoolSubbuffer, BufferAccess, CpuAccessibleBuffer, CpuBufferPool,
        DeviceLocalBuffer, TypedBufferAccess,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, DispatchIndirectCommand, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator,
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutCreateInfo, DescriptorSetLayoutCreationError,
        },
        PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
    memory::allocator::{MemoryUsage, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            depth_stencil::{CompareOp, DepthState, DepthStencilState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            rasterization::{CullMode, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        layout::PipelineLayoutCreateInfo,
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout, StateMode,
    },
    render_pass::{RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    sync::{self, FlushError, GpuFuture},
    DeviceSize,
};

// use super::cs::{ty::{emitter_init, particle_template}, self};

use lazy_static::lazy_static;

// pub const MAX_PARTICLES: i32 = SETTINGS.read().get::<i32>("MAX_PARTICLES").unwrap();
lazy_static! {
    pub static ref _MAX_PARTICLES: i32 = crate::engine::utils::SETTINGS
        .read()
        .get::<i32>("MAX_PARTICLES")
        .unwrap();
}

use crate::engine::project::asset_manager::_AssetID;

use super::{
    particle_asset::{ParticleTemplateManager, TEMPLATE_UPDATE},
    particle_sort::ParticleSort,
    particle_textures::ParticleTextures,
    shaders::{
        self,
        cs::{
            self,
            ty::{burst, emitter_init, particle_template, Data},
        },
        gs,
    },
};

pub struct PerformanceCounters {
    pub update_particles: i32,
    pub update_emitters: i32,
    pub init_emitters: i32,
    pub sort_particles: i32,
}
pub struct ParticleBuffers {
    pub particles: Arc<DeviceLocalBuffer<[cs::ty::particle]>>,
    pub particle_positions_lifes: Arc<DeviceLocalBuffer<[cs::ty::pos_lif]>>,
    pub particle_next: Arc<DeviceLocalBuffer<[i32]>>,
    pub particle_template_ids: Arc<DeviceLocalBuffer<[i32]>>,
    pub alive: Arc<DeviceLocalBuffer<[u32]>>,
    pub alive_count: Arc<DeviceLocalBuffer<i32>>,
    pub emitters: Mutex<Arc<DeviceLocalBuffer<[cs::ty::emitter]>>>,
    emitter_init_dummy: Arc<DeviceLocalBuffer<[cs::ty::emitter_init]>>,
    particle_burst_dummy: Arc<DeviceLocalBuffer<[cs::ty::burst]>>,
    pub particle_templates: Mutex<Arc<DeviceLocalBuffer<[cs::ty::particle_template]>>>,
    pub avail: Arc<DeviceLocalBuffer<[u32]>>,
    pub avail_count: Arc<DeviceLocalBuffer<i32>>,
    pub buffer_0: Arc<DeviceLocalBuffer<i32>>,
    pub indirect: Arc<DeviceLocalBuffer<[DispatchIndirectCommand]>>,
    pub alive_b: Arc<DeviceLocalBuffer<[cs::ty::b]>>,
}

pub struct AtomicVec<T: Copy> {
    lock: Mutex<()>,
    data: SyncUnsafeCell<Vec<T>>,
    index: AtomicUsize,
}
impl<T: Copy> AtomicVec<T> {
    pub fn new() -> Self {
        let mut data = Vec::with_capacity(4);
        unsafe {
            data.set_len(4);
        }
        Self {
            lock: Mutex::new(()),
            data: SyncUnsafeCell::new(data),
            index: AtomicUsize::new(0),
        }
    }
    pub fn push_multi<'a>(&mut self, count: usize) -> (MutexGuard<()>, &'a [T]) {
        let _l = self.lock.lock();
        unsafe {
            let index = self.index.load(Ordering::Relaxed);
            (*self.data.get()).reserve(count);
            (*self.data.get()).set_len(index + count);
            self.index.fetch_add(count, Ordering::Relaxed);
            (_l, &(*self.data.get())[index..(index + count)])
        }
    }
    fn try_push(&self, d: T) -> Option<(usize, T)> {
        let index = self.index.fetch_add(1, Ordering::Relaxed);
        if index >= unsafe { (*self.data.get()).len() } {
            Some((index, d))
        } else {
            unsafe { (*self.data.get())[index] = d };
            None
        }
    }
    fn _push(&self, i: usize, d: T) {
        // let index = self.index.fetch_add(1, Ordering::Relaxed);
        unsafe {
            let _l = self.lock.lock();
            if i >= (*self.data.get()).len() {
                let data = &mut (*self.data.get());
                let len = data.len() + 1;
                let new_len = (data.len() + 1).next_power_of_two();
                data.reserve_exact(new_len - len + 1);
                data.set_len(new_len);
            }
            (*self.data.get())[i] = d;
            // drop(l);
        };
    }
    pub fn push(&self, d: T) {
        if let Some((id, d)) = self.try_push(d) {
            // let _ = self.lock.lock();
            self._push(id, d);
        }
    }
    pub fn get_vec(&self) -> Vec<T> {
        let _l = self.lock.lock();
        // let mut v = Vec::new();
        let len = self.index.load(Ordering::Relaxed);
        // unsafe {
        //     let cap = (*self.data.get()).capacity();
        //     std::mem::swap(&mut v, &mut *self.data.get());
        //     v.set_len(len);
        //     *self.data.get() = Vec::with_capacity(cap);
        //     (*self.data.get()).set_len(cap);
        // }
        // v
        let mut v = Vec::with_capacity(len);
        // std::mem::swap(&mut v, unsafe { &mut *self.data.get() });
        // unsafe {
        //     unsafe { &mut *self.data.get() }.set_len(len.next_power_of_two());
        // }
        for i in (0..len).into_iter() {
            unsafe { v.push((*self.data.get())[i]) };
        }
        self.index.store(0, Ordering::Relaxed);
        v
    }
}

pub struct ParticleCompute {
    pub sort: ParticleSort,
    pub emitter_inits: AtomicVec<cs::ty::emitter_init>,
    pub emitter_deinits: AtomicVec<cs::ty::emitter_init>,
    pub particle_burts: AtomicVec<cs::ty::burst>,
    pub particle_templates: Arc<Mutex<_Storage<cs::ty::particle_template>>>,
    pub particle_template_manager: Arc<Mutex<ParticleTemplateManager>>,
    pub particle_buffers: ParticleBuffers,
    pub compute_pipeline: Arc<ComputePipeline>,
    pub compute_uniforms: CpuBufferPool<cs::ty::Data>,
    pub render_uniforms: CpuBufferPool<shaders::gs::ty::Data>,
    pub def_texture: Arc<ImageView<ImmutableImage>>,
    pub def_sampler: Arc<Sampler>,
    pub vk: Arc<VulkanManager>,
    pub performance: PerformanceCounters,
    pub particle_textures: Arc<Mutex<ParticleTextures>>,
}
pub struct ParticleRenderPipeline {
    pub arc: Arc<GraphicsPipeline>,
}
impl ParticleRenderPipeline {
    pub fn new(vk: Arc<VulkanManager>, render_pass: Arc<RenderPass>) -> Self {
        let vs = shaders::vs::load(vk.device.clone()).unwrap();
        let fs = shaders::fs::load(vk.device.clone()).unwrap();
        let gs = shaders::gs::load(vk.device.clone()).unwrap();

        let subpass = Subpass::from(render_pass, 0).unwrap();
        let blend_state = ColorBlendState::new(subpass.num_color_attachments()).blend_alpha();
        let mut depth_stencil_state = DepthStencilState::simple_depth_test();
        depth_stencil_state.depth = Some(DepthState {
            enable_dynamic: false,
            write_enable: StateMode::Fixed(false),
            compare_op: StateMode::Fixed(CompareOp::Less),
        });

        let render_pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            // .input_assembly_state(InputAssemblyState::new())
            .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::PointList))
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .geometry_shader(gs.entry_point("main").unwrap(), ())
            // .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::LineStrip))
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .depth_stencil_state(depth_stencil_state)
            .color_blend_state(blend_state)
            .render_pass(subpass)
            .with_auto_layout(vk.device.clone(), |layout_create_infos| {
                let binding = layout_create_infos[0].bindings.get_mut(&11).unwrap();
                binding.variable_descriptor_count = true;
                binding.descriptor_count = 16;
            })
            // .with_pipeline_layout(vk.device.clone(), layout)
            .unwrap();
        // .build(vk.device.clone())
        // .unwrap();

        Self {
            arc: render_pipeline,
        }
    }
}

impl ParticleCompute {
    pub fn new(vk: Arc<VulkanManager>, tex_man: Arc<Mutex<TextureManager>>) -> ParticleCompute {
        let performance = PerformanceCounters {
            update_particles: vk.new_query(),
            update_emitters: vk.new_query(),
            init_emitters: vk.new_query(),
            sort_particles: vk.new_query(),
        };
        let max_particles: i32 = *_MAX_PARTICLES;
        let particles = DeviceLocalBuffer::<[cs::ty::particle]>::array(
            &vk.mem_alloc,
            max_particles as vulkano::DeviceSize,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        let particle_template_ids = DeviceLocalBuffer::<[i32]>::array(
            &vk.mem_alloc,
            max_particles as vulkano::DeviceSize,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            &vk.comm_alloc,
            vk.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        println!("pos_lif: {}", std::mem::size_of::<cs::ty::pos_lif>());
        println!("b: {}", std::mem::size_of::<cs::ty::b>());

        let particle_positions_lifes: Vec<cs::ty::pos_lif> = (0..max_particles)
            .map(|_| cs::ty::pos_lif {
                pos: [0., 0., 0.],
                life: 0.,
            })
            .collect();
        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &vk.mem_alloc,
            buffer_usage_all(),
            false,
            particle_positions_lifes,
        )
        .unwrap();
        let particle_positions_lifes = DeviceLocalBuffer::<[cs::ty::pos_lif]>::array(
            &vk.mem_alloc,
            max_particles as vulkano::DeviceSize,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_buffer,
                particle_positions_lifes.clone(),
            ))
            .unwrap();

        let particle_next = DeviceLocalBuffer::<[i32]>::array(
            &vk.mem_alloc,
            max_particles as vulkano::DeviceSize,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        // avail
        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &vk.mem_alloc,
            buffer_usage_all(),
            false,
            0..max_particles,
        )
        .unwrap();
        let avail = DeviceLocalBuffer::<[u32]>::array(
            &vk.mem_alloc,
            max_particles as vulkano::DeviceSize,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, avail.clone()))
            .unwrap();

        // alive_b
        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &vk.mem_alloc,
            buffer_usage_all(),
            false,
            (0..max_particles).map(|_| 0),
        )
        .unwrap();
        let alive_b = DeviceLocalBuffer::<[cs::ty::b]>::array(
            &vk.mem_alloc,
            max_particles as vulkano::DeviceSize,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, alive_b.clone()))
            .unwrap();
        let alive = DeviceLocalBuffer::<[u32]>::array(
            &vk.mem_alloc,
            max_particles as vulkano::DeviceSize,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        // avail_count
        let copy_buffer =
            CpuAccessibleBuffer::from_data(&vk.mem_alloc, buffer_usage_all(), false, 0i32).unwrap();
        let avail_count = DeviceLocalBuffer::<i32>::new(
            &vk.mem_alloc,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        let alive_count = DeviceLocalBuffer::<i32>::new(
            &vk.mem_alloc,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        // buffer 0
        let buffer_0 = DeviceLocalBuffer::<i32>::new(
            &vk.mem_alloc,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

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
        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, buffer_0.clone()))
            .unwrap();

        // emitters
        let emitters = DeviceLocalBuffer::<[cs::ty::emitter]>::array(
            &vk.mem_alloc,
            1 as vulkano::DeviceSize,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        // indirect
        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &vk.mem_alloc,
            buffer_usage_all(),
            false,
            [DispatchIndirectCommand { x: 0, y: 1, z: 1 }],
        )
        .unwrap();
        let indirect = DeviceLocalBuffer::<[DispatchIndirectCommand]>::array(
            &vk.mem_alloc,
            1 as DeviceSize,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, indirect.clone()))
            .unwrap();

        unsafe {
            *super::particle_asset::DEFAULT_TEXTURE.get() =
                AssetInstance::<Texture>::new(tex_man.lock().from_file("eng_res/particle.png"));
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

        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &vk.mem_alloc,
            buffer_usage_all(),
            false,
            particle_templates.lock().data.clone(),
        )
        .unwrap();
        let templates = DeviceLocalBuffer::<[cs::ty::particle_template]>::array(
            &vk.mem_alloc,
            copy_buffer.len() as vulkano::DeviceSize,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, templates.clone()))
            .unwrap();

        let def_texture = {
            let dimensions = ImageDimensions::Dim2d {
                width: 1,
                height: 1,
                array_layers: 1,
            };
            let image_data = vec![255_u8, 255, 255, 255];
            let image = ImmutableImage::from_iter(
                &vk.mem_alloc,
                image_data,
                dimensions,
                MipmapsCount::One,
                Format::R8G8B8A8_SRGB,
                &mut builder,
            )
            .unwrap();

            ImageView::new_default(image).unwrap()
        };
        let def_sampler = Sampler::new(
            vk.device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

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
                    Err(FlushError::OutOfDate) => {}
                    Err(_e) => {}
                }
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
            }
        };
        let cs = cs::load(vk.device.clone()).unwrap();
        let compute_pipeline = vulkano::pipeline::ComputePipeline::new(
            vk.device.clone(),
            cs.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create compute shader");

        let uniforms = CpuBufferPool::<cs::ty::Data>::new(
            vk.mem_alloc.clone(),
            buffer_usage_all(),
            MemoryUsage::Upload,
        );
        let render_uniforms = CpuBufferPool::<gs::ty::Data>::new(
            vk.mem_alloc.clone(),
            buffer_usage_all(),
            MemoryUsage::Upload,
        );
        let emitter_init_dummy = DeviceLocalBuffer::<[cs::ty::emitter_init]>::array(
            &vk.mem_alloc,
            1 as vulkano::DeviceSize,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        let particle_burst_dummy = DeviceLocalBuffer::<[cs::ty::burst]>::array(
            &vk.mem_alloc,
            1 as vulkano::DeviceSize,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        ParticleCompute {
            sort: ParticleSort::new(
                vk.device.clone(),
                vk.queue.clone(),
                vk.mem_alloc.clone(),
                &vk.comm_alloc,
                vk.desc_alloc.clone(),
            ),
            emitter_inits: AtomicVec::new(),
            emitter_deinits: AtomicVec::new(),
            particle_burts: AtomicVec::new(),
            particle_templates,
            particle_template_manager,
            particle_buffers: ParticleBuffers {
                particles,
                particle_next,
                particle_positions_lifes,
                particle_template_ids,
                emitter_init_dummy,
                particle_burst_dummy,
                emitters: Mutex::new(emitters),
                particle_templates: Mutex::new(templates),
                alive,
                alive_count,
                avail,
                avail_count,
                buffer_0,
                indirect,
                alive_b,
            },
            // render_pipeline,
            compute_pipeline,
            compute_uniforms: uniforms,
            render_uniforms,
            def_texture,
            def_sampler,
            vk,
            performance,
            particle_textures,
        }
    }
    fn get_descriptors2(
        &self,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        uniform_sub_buffer: Arc<CpuBufferPoolSubbuffer<Data>>,
    ) -> Arc<PersistentDescriptorSet> {
        self.get_descriptors(
            transform,
            uniform_sub_buffer,
            self.particle_buffers.emitter_init_dummy.clone(),
            self.particle_buffers.particle_burst_dummy.clone(),
        )
    }
    fn get_descriptors(
        &self,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        uniform_sub_buffer: Arc<CpuBufferPoolSubbuffer<Data>>,
        emitter_inits: Arc<dyn BufferAccess>,
        particle_bursts: Arc<dyn BufferAccess>,
    ) -> Arc<PersistentDescriptorSet> {
        let pb = &self.particle_buffers;

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
                WriteDescriptorSet::buffer(14, pb.alive_b.clone()),
                WriteDescriptorSet::buffer(15, particle_bursts.clone()),
            ],
        )
        .unwrap();
        descriptor_set
    }
    pub fn update(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        particle_init_data: (usize, Vec<emitter_init>, Vec<emitter_init>, Vec<burst>),
        transform_compute: &TransformCompute,
        time: &Time,
    ) {
        self.update_templates(builder);
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
        self.emitter_update(
            builder,
            transform_compute.gpu_transforms.clone(),
            particle_init_data.0,
            time,
        );
        self.particle_update(builder, transform_compute.gpu_transforms.clone(), time);
    }

    fn update_templates(&self, builder: &mut PrimaryCommandBuffer) {
        let pb = &self.particle_buffers;
        let mut pt = self.particle_templates.lock();
        if TEMPLATE_UPDATE.load(Ordering::Relaxed) {
            let mut ptex = self.particle_textures.lock();
            let mut colors: Vec<[[u8; 4]; 256]> = Vec::new();
            // colors.push([[255u8;4];256]);
            for (id, a) in self.particle_template_manager.lock().assets_id.iter() {
                let a = a.lock();
                *pt.get_mut(id) = a.gen_particle_template(&mut ptex);
                colors.push(a.color_over_life.to_color_array());
            }
            println!("colors len: {}", colors.len());
            println!("template len: {}", pt.data.len());

            ptex.color_tex = ParticleTextures::color_tex(colors.as_slice(), &self.vk, builder);
        }

        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &self.vk.mem_alloc,
            buffer_usage_all(),
            false,
            pt.data.iter().copied(),
        )
        .unwrap();
        drop(pt);

        if copy_buffer.len() > pb.particle_templates.lock().len() {
            *pb.particle_templates.lock() =
                DeviceLocalBuffer::<[cs::ty::particle_template]>::array(
                    &self.vk.mem_alloc,
                    copy_buffer.len() as vulkano::DeviceSize,
                    buffer_usage_all(),
                    self.vk.device.active_queue_family_indices().iter().copied(),
                )
                .unwrap();
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
        let pb = &self.particle_buffers;
        let max_len = num_emitters.next_power_of_two();
        if pb.emitters.lock().len() < max_len as u64 {
            let emitters = DeviceLocalBuffer::<[cs::ty::emitter]>::array(
                &self.vk.mem_alloc,
                max_len as vulkano::DeviceSize,
                buffer_usage_all(),
                self.vk.device.active_queue_family_indices().iter().copied(),
            )
            .unwrap();

            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    pb.emitters.lock().clone(),
                    emitters.clone(),
                ))
                .unwrap();
            *pb.emitters.lock() = emitters;
        }
    }

    fn particle_bursts(
        &self,
        builder: &mut PrimaryCommandBuffer,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        particle_bursts: Vec<cs::ty::burst>,
        time: &Time,
    ) {
        let pb = &self.particle_buffers;

        if particle_bursts.is_empty() {
            return;
        };
        let len = particle_bursts.len();

        let particle_bursts = CpuAccessibleBuffer::from_iter(
            &self.vk.mem_alloc,
            buffer_usage_all(),
            false,
            particle_bursts,
        )
        .unwrap();
        let max_particles: i32 = *_MAX_PARTICLES;
        let uniform_sub_buffer = {
            let uniform_data = cs::ty::Data {
                num_jobs: len as i32,
                dt: time.dt,
                time: time.time,
                stage: 0,
                MAX_PARTICLES: max_particles,
            };
            self.compute_uniforms.from_data(uniform_data).unwrap()
        };
        let descriptor_set = self.get_descriptors(
            transform,
            uniform_sub_buffer,
            pb.emitter_init_dummy.clone(),
            particle_bursts,
        );

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
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        emitter_deinits: Vec<cs::ty::emitter_init>,
        time: &Time,
    ) {
        let pb = &self.particle_buffers;

        if emitter_deinits.is_empty() {
            return;
        };
        let len = emitter_deinits.len();

        let emitter_deinits = CpuAccessibleBuffer::from_iter(
            &self.vk.mem_alloc,
            buffer_usage_all(),
            false,
            emitter_deinits,
        )
        .unwrap();

        let max_particles: i32 = *_MAX_PARTICLES;
        let uniform_sub_buffer = {
            let uniform_data = cs::ty::Data {
                num_jobs: len as i32,
                dt: time.dt,
                time: time.time,
                stage: 1,
                MAX_PARTICLES: max_particles,
            };
            self.compute_uniforms.from_data(uniform_data).unwrap()
        };
        let descriptor_set = self.get_descriptors(
            transform,
            uniform_sub_buffer,
            emitter_deinits,
            pb.particle_burst_dummy.clone(),
        );

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
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        emitter_inits: Vec<cs::ty::emitter_init>,
        time: &Time,
    ) {
        let pb = &self.particle_buffers;
        if emitter_inits.is_empty() {
            return;
        };
        let len = emitter_inits.len();

        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &self.vk.mem_alloc,
            buffer_usage_all(),
            false,
            emitter_inits,
        )
        .unwrap();
        let emitter_inits = DeviceLocalBuffer::<[cs::ty::emitter_init]>::array(
            &self.vk.mem_alloc,
            len as vulkano::DeviceSize,
            buffer_usage_all(),
            self.vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap(); // TODO: cache

        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, emitter_inits.clone()))
            .unwrap();

        let max_particles: i32 = *_MAX_PARTICLES;
        let uniform_sub_buffer = {
            let uniform_data = cs::ty::Data {
                num_jobs: len as i32,
                dt: time.dt,
                time: time.time,
                stage: 2,
                MAX_PARTICLES: max_particles,
            };
            self.compute_uniforms.from_data(uniform_data).unwrap()
        };
        let descriptor_set = self.get_descriptors(
            transform,
            uniform_sub_buffer,
            emitter_inits,
            pb.particle_burst_dummy.clone(),
        );

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
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        emitter_len: usize,
        time: &Time,
    ) {
        let pb = &self.particle_buffers;

        let emitter_len = emitter_len.max(1);
        let max_particles: i32 = *_MAX_PARTICLES;

        let uniform_sub_buffer = {
            let uniform_data = cs::ty::Data {
                num_jobs: emitter_len as i32,
                dt: time.dt,
                time: time.time,
                stage: 3,
                MAX_PARTICLES: max_particles,
            };
            self.compute_uniforms.from_data(uniform_data).unwrap()
        };
        let descriptor_set = self.get_descriptors2(transform, uniform_sub_buffer);

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .dispatch([emitter_len as u32 / 1024 + 1, 1, 1])
            .unwrap();
    }

    pub fn particle_update(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        time: &Time,
    ) {
        let pb = &self.particle_buffers;
        let max_particles: i32 = *_MAX_PARTICLES;

        let mut uniform_data = cs::ty::Data {
            num_jobs: max_particles,
            dt: time.dt,
            time: time.time,
            stage: 4,
            MAX_PARTICLES: max_particles,
        };
        let uniform_sub_buffer = self.compute_uniforms.from_data(uniform_data).unwrap();

        let descriptor_set = self.get_descriptors2(transform.clone(), uniform_sub_buffer);

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                pb.buffer_0.clone(),
                pb.alive_count.clone(),
            ))
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set,
            )
            .dispatch([max_particles as u32 / 1024 + 1, 1, 1])
            .unwrap();
        // set indirect
        uniform_data.num_jobs = 1;
        uniform_data.stage = 5;
        let uniform_sub_buffer = self.compute_uniforms.from_data(uniform_data).unwrap();
        let descriptor_set = self.get_descriptors2(transform.clone(), uniform_sub_buffer);

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .dispatch([1, 1, 1])
            .unwrap();
        // dispatch indirect particle update
        uniform_data.num_jobs = -1;
        uniform_data.stage = 6;
        let uniform_sub_buffer = self.compute_uniforms.from_data(uniform_data).unwrap();
        let descriptor_set = self.get_descriptors2(transform.clone(), uniform_sub_buffer);

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .dispatch_indirect(pb.indirect.clone())
            .unwrap();
    }
    pub fn render_particles(
        &self,
        particle_render_pipeline: &ParticleRenderPipeline,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        view: glm::Mat4,
        proj: glm::Mat4,
        cam_rot: [f32; 4],
        cam_pos: [f32; 3],
        transform: Arc<DeviceLocalBuffer<[transform]>>,
    ) {
        let pb = &self.particle_buffers;
        let uniform_sub_buffer = {
            let uniform_data = gs::ty::Data {
                view: view.into(),
                proj: proj.into(),
                cam_rot,
                cam_pos,
                num_templates: pb.particle_templates.lock().len() as u32,
                _dummy0: Default::default(),
            };
            self.render_uniforms.from_data(uniform_data).unwrap()
        };
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
                WriteDescriptorSet::buffer(0, pb.particle_positions_lifes.clone()),
                WriteDescriptorSet::buffer(3, pb.particle_templates.lock().clone()),
                WriteDescriptorSet::buffer(4, uniform_sub_buffer),
                WriteDescriptorSet::buffer(5, self.sort.a2.clone()),
                WriteDescriptorSet::buffer(6, pb.particle_template_ids.clone()),
                WriteDescriptorSet::buffer(7, pb.particle_next.clone()),
                WriteDescriptorSet::buffer(8, transform),
                WriteDescriptorSet::buffer(9, pb.particles.clone()),
                WriteDescriptorSet::image_view_sampler(
                    10,
                    pt.color_tex.0.clone(),
                    pt.color_tex.1.clone(),
                ),
                WriteDescriptorSet::image_view_sampler_array(
                    11,
                    0,
                    pt.samplers.iter().map(|a| (a.0.clone() as _, a.1.clone())),
                ),
            ],
        )
        .unwrap();
        builder
            .bind_pipeline_graphics(particle_render_pipeline.arc.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                particle_render_pipeline.arc.layout().clone(),
                0,
                set,
            )
            .draw_indirect(self.sort.draw.clone())
            .unwrap();
    }
}
