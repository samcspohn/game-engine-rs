use std::sync::Arc;

use crate::{
    engine::{transform::Transform, Component, Storage, Sys, World, _Storage},
    inspectable::{Inpsect, Ins, Inspectable},
    particle_sort::ParticleSort,
    renderer_component2::buffer_usage_all,
    transform_compute::cs::ty::transform,
};
use nalgebra_glm as glm;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{
        BufferUsage, CpuAccessibleBuffer, CpuBufferPool, DeviceLocalBuffer, TypedBufferAccess,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
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
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, StateMode,
    },
    render_pass::{RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    sync::{self, FlushError, GpuFuture},
};

use self::cs::ty::emitter_init;

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/particle_shaders/particles.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/particle_shaders/particles.vert",
        types_meta: {


            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
pub mod gs {
    vulkano_shaders::shader! {
        ty: "geometry",
        path: "src/particle_shaders/particles.geom",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/particle_shaders/particles.frag"
    }
}

pub const MAX_PARTICLES: i32 = 8 * 1024 * 1024 * 4;
// pub const NUM_EMITTERS: i32 = 1_200_000;

// #[component]
#[derive(Default, Clone, Deserialize, Serialize)]
pub struct ParticleEmitter {
    template: i32,
}
impl Inspectable for ParticleEmitter {
    fn inspect(&mut self, transform: Transform, id: i32, ui: &mut egui::Ui, sys: &mut Sys) {
        // ui.add(egui::DragValue::new(&mut self.template));
        // ui.add(egui::Label::new("Particle Emitter"));
        // egui::CollapsingHeader::new("Particle Emitter")
        //     .default_open(true)
        //     .show(ui, |ui| {
        Ins(&mut self.template).inspect("template", ui, sys);
        // });
    }
}
impl ParticleEmitter {
    pub fn new(template: i32) -> ParticleEmitter {
        ParticleEmitter { template }
    }
}

impl Component for ParticleEmitter {
    // fn assign_transform(&mut self, t: Transform) {
    //     self.t = t;
    // }
    fn init(&mut self, transform: Transform, id: i32, sys: &mut crate::engine::Sys) {
        self.template = id % 2;
        sys.particles
            .emitter_inits
            .lock()
            .push(cs::ty::emitter_init {
                transform_id: transform.id,
                alive: 1,
                template_id: self.template,
                e_id: id,
            });
    }
    fn deinit(&mut self, transform: Transform, id: i32, sys: &mut crate::engine::Sys) {
        sys.particles
            .emitter_inits
            .lock()
            .push(cs::ty::emitter_init {
                transform_id: transform.id,
                alive: 0,
                template_id: self.template,
                e_id: id,
            });
    }
}

pub struct ParticleCompute {
    pub sort: ParticleSort,
    pub particles: Arc<DeviceLocalBuffer<[cs::ty::particle]>>,
    pub particle_positions_lifes: Arc<DeviceLocalBuffer<[cs::ty::pos_lif]>>,
    pub particle_template_ids: Arc<DeviceLocalBuffer<[i32]>>,
    // pub particle_lifes: Arc<DeviceLocalBuffer<[f32]>>,
    pub emitters: Mutex<Arc<DeviceLocalBuffer<[cs::ty::emitter]>>>,
    pub emitter_inits: Arc<Mutex<Vec<cs::ty::emitter_init>>>,
    pub particle_templates: _Storage<cs::ty::particle_template>,
    pub particle_template: Arc<DeviceLocalBuffer<[cs::ty::particle_template]>>,
    pub avail: Arc<DeviceLocalBuffer<[u32]>>,
    pub avail_count: Arc<DeviceLocalBuffer<i32>>,
    pub render_pipeline: Arc<GraphicsPipeline>,
    pub compute_pipeline: Arc<ComputePipeline>,
    pub compute_uniforms: CpuBufferPool<cs::ty::Data>,
    pub render_uniforms: CpuBufferPool<gs::ty::Data>,
    pub def_texture: Arc<ImageView<ImmutableImage>>,
    pub def_sampler: Arc<Sampler>,
}

impl ParticleCompute {
    pub fn new(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        // dimensions: [u32; 2],
        // swapchain: Arc<Swapchain<Window>>,
        queue: Arc<Queue>,
        mem: Arc<StandardMemoryAllocator>,
        command_allocator: &StandardCommandBufferAllocator,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> ParticleCompute {
        let q = glm::Quat::identity();

        // particles
        // let particles: Vec<cs::ty::particle> = (0..MAX_PARTICLES)
        //     .into_iter()
        //     .map(|_| cs::ty::particle {
        //         emitter_id: 0,
        //         rot: q.coords.into(),
        //         template_id: 0,
        //         _dummy0: Default::default(),
        //         vel: [0., 0., 0.],
        //         sorted: -1,
        //     })
        //     .collect();
        // let copy_buffer =
        //     CpuAccessibleBuffer::from_iter(device.clone(), buffer_usage_all(), false, particles)
        //         .unwrap();
        let particles = DeviceLocalBuffer::<[cs::ty::particle]>::array(
            &mem,
            MAX_PARTICLES as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        let particle_template_ids = DeviceLocalBuffer::<[i32]>::array(
            &mem,
            MAX_PARTICLES as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            command_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        // builder.copy_buffer(copy_buffer, particles.clone()).unwrap();
        println!("pos_lif: {}", std::mem::size_of::<cs::ty::pos_lif>());
        // positions
        let particle_positions_lifes: Vec<cs::ty::pos_lif> = (0..MAX_PARTICLES)
            .into_iter()
            .map(|_| cs::ty::pos_lif {
                pos: [0., 0., 0.],
                life: 0.,
                // rot: [1.,0.,0.,0.],
                // template_id: 0,
                // _dummy0: Default::default(),
            })
            .collect();
        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &mem,
            buffer_usage_all(),
            false,
            particle_positions_lifes,
        )
        .unwrap();
        let particle_positions_lifes = DeviceLocalBuffer::<[cs::ty::pos_lif]>::array(
            &mem,
            MAX_PARTICLES as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_buffer,
                particle_positions_lifes.clone(),
            ))
            .unwrap();

        // //lifes
        // let particle_lifes: Vec<f32> = (0..MAX_PARTICLES).into_iter().map(|_| 0f32).collect();

        // let copy_buffer = CpuAccessibleBuffer::from_iter(
        //     &mem,
        //     buffer_usage_all(),
        //     false,
        //     particle_lifes,
        // )
        // .unwrap();
        // let particle_lifes = DeviceLocalBuffer::<[f32]>::array(
        //     &mem,
        //     MAX_PARTICLES as vulkano::DeviceSize,
        //     buffer_usage_all(),
        //     device.active_queue_families(),
        // )
        // .unwrap();

        // builder
        //     .copy_buffer(copy_buffer, particle_lifes.clone())
        //     .unwrap();
        // avail
        let avail: Vec<u32> = (0..MAX_PARTICLES).into_iter().map(|i| i as u32).collect();

        let copy_buffer =
            CpuAccessibleBuffer::from_iter(&mem, buffer_usage_all(), false, avail).unwrap();
        let avail = DeviceLocalBuffer::<[u32]>::array(
            &mem,
            MAX_PARTICLES as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, avail.clone()))
            .unwrap();

        // avail_count
        let copy_buffer =
            CpuAccessibleBuffer::from_data(&mem, buffer_usage_all(), false, 0i32).unwrap();
        let avail_count = DeviceLocalBuffer::<i32>::new(
            &mem,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, avail_count.clone()))
            .unwrap();

        // emitters
        // let emitters: Vec<cs::ty::emitter> = (0..NUM_EMITTERS)
        //     .into_iter()
        //     .map(|i| cs::ty::emitter {
        //         // pos: [
        //         //     (rand::random::<f32>() - 0.5) * 2000.,
        //         //     (rand::random::<f32>() - 0.5) * 2000.,
        //         //     (rand::random::<f32>() - 0.5) * 2000.,
        //         // ],
        //         alive: 0i32,
        //         transform_id: 0i32,
        //         emission: 0f32,
        //         template_id: i % 2,
        //         // _dummy0: Default::default(),
        //     })
        //     .collect();
        // let copy_buffer =
        //     CpuAccessibleBuffer::from_iter(&mem, buffer_usage_all(), false, emitters)
        //         .unwrap();
        let emitters = DeviceLocalBuffer::<[cs::ty::emitter]>::array(
            &mem,
            1 as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        // builder.copy_buffer(copy_buffer, emitters.clone()).unwrap();

        // particle templates
        // let templates: Vec<cs::ty::particle_template> = vec![
        // cs::ty::particle_template {
        //     color: [0f32, 0f32, 1f32],
        //     speed: 5f32,
        //     emission_rate: 40f32,
        //     _dummy0: Default::default(),
        // },
        // cs::ty::particle_template {
        //     color: [1f32, 0f32, 0f32],
        //     speed: 3f32,
        //     emission_rate: 20f32,
        //     _dummy0: Default::default(),
        // },
        // ];
        let mut particle_templates = _Storage::new();
        particle_templates.emplace(cs::ty::particle_template {
            color: [0f32, 0f32, 1f32],
            speed: 5f32,
            emission_rate: 20f32,
            _dummy0: Default::default(),
        });
        particle_templates.emplace(cs::ty::particle_template {
            color: [1f32, 0f32, 0f32],
            speed: 3f32,
            emission_rate: 10f32,
            _dummy0: Default::default(),
        });
        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &mem,
            buffer_usage_all(),
            false,
            particle_templates.data.clone(), // .iter()
                                             // .map(|x| {
                                             //     x
                                             // })
                                             // .collect::<Vec<cs::ty::particle_template>>(),
        )
        .unwrap();
        let templates = DeviceLocalBuffer::<[cs::ty::particle_template]>::array(
            &mem,
            copy_buffer.len() as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
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
            let image_data = vec![255 as u8, 255, 255, 255];
            let image = ImmutableImage::from_iter(
                &mem,
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
            device.clone(),
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

        let execute = sync::now(device.clone())
            .boxed()
            .then_execute(queue.clone(), command_buffer);

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

        let vs = vs::load(device.clone()).unwrap();
        let fs = fs::load(device.clone()).unwrap();
        let gs = gs::load(device.clone()).unwrap();
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let blend_state = ColorBlendState::new(subpass.num_color_attachments()).blend_alpha();
        let mut depth_stencil_state = DepthStencilState::simple_depth_test();
        depth_stencil_state.depth = Some(DepthState {
            enable_dynamic: false,
            write_enable: StateMode::Fixed(false),
            compare_op: StateMode::Fixed(CompareOp::Less),
        });

        // DepthStencilState {
        //     depth: Some(DepthState {
        //         enable_dynamic: false,
        //         write_enable: StateMode::Fixed(false),
        //         compare_op: StateMode::Fixed(CompareOp::Less),
        //     }),
        //     depth_bounds: todo!(),
        //     stencil: todo!(),
        // }
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
            .build(device.clone())
            .unwrap();

        let cs = cs::load(device.clone()).unwrap();
        let compute_pipeline = vulkano::pipeline::ComputePipeline::new(
            device.clone(),
            cs.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create compute shader");

        let uniforms = CpuBufferPool::<cs::ty::Data>::new(
            mem.clone(),
            buffer_usage_all(),
            MemoryUsage::Upload,
        );
        let render_uniforms = CpuBufferPool::<gs::ty::Data>::new(
            mem.clone(),
            buffer_usage_all(),
            MemoryUsage::Upload,
        );

        ParticleCompute {
            sort: ParticleSort::new(
                device.clone(),
                // render_pass.clone(),
                // swapchain.clone(),
                queue.clone(),
                mem,
                command_allocator,
                desc_allocator,
            ),
            particles,
            particle_positions_lifes,
            particle_template_ids,
            emitters: Mutex::new(emitters),
            emitter_inits: Arc::new(Mutex::new(Vec::new())),
            particle_templates,
            particle_template: templates,
            avail,
            avail_count,
            render_pipeline,
            compute_pipeline,
            compute_uniforms: uniforms,
            render_uniforms,
            def_texture,
            def_sampler,
        }
    }

    pub fn emitter_init(
        &self,
        device: Arc<Device>,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        emitter_inits: Vec<cs::ty::emitter_init>,
        emitter_len: usize,
        dt: f32,
        time: f32,
        cam_pos: [f32; 3],
        cam_rot: [f32; 4],
        mem: Arc<StandardMemoryAllocator>,
        command_allocator: &StandardCommandBufferAllocator,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
    ) {
        // let mut emitter_inits = emitter_inits.lock();
        if emitter_inits.len() == 0 {
            return;
        };
        let len = emitter_inits.len();
        // let mut ei = Vec::<emitter_init>::new();
        // std::mem::swap(&mut ei, &mut emitter_inits);

        let copy_buffer =
            CpuAccessibleBuffer::from_iter(&mem, buffer_usage_all(), false, emitter_inits).unwrap();
        let emitter_inits = DeviceLocalBuffer::<[cs::ty::emitter_init]>::array(
            &mem,
            len as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, emitter_inits.clone()))
            .unwrap();

        let max_len = (emitter_len as f32 + 1.).log2().ceil();
        let max_len = (2 as u32).pow(max_len as u32);
        let mut self_emitters = self.emitters.lock();
        if self_emitters.len() < max_len as u64 {
            let emitters = DeviceLocalBuffer::<[cs::ty::emitter]>::array(
                &mem,
                max_len as vulkano::DeviceSize,
                buffer_usage_all(),
                device.active_queue_family_indices().iter().copied(),
            )
            .unwrap();

            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    self_emitters.clone(),
                    emitters.clone(),
                ))
                .unwrap();
            *self_emitters = emitters.clone();
        }

        let uniform_sub_buffer = {
            let uniform_data = cs::ty::Data {
                num_jobs: len as i32,
                dt: dt,
                time,
                stage: 0,
                cam_pos,
                cam_rot,
                MAX_PARTICLES,
                _dummy0: Default::default(),
            };
            self.compute_uniforms.from_data(uniform_data).unwrap()
        };
        let descriptor_set = PersistentDescriptorSet::new(
            &desc_allocator,
            self.compute_pipeline
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, transform.clone()),
                WriteDescriptorSet::buffer(1, self.particles.clone()),
                WriteDescriptorSet::buffer(2, self.particle_positions_lifes.clone()),
                // WriteDescriptorSet::buffer(3, self.particle_lifes.clone()),
                WriteDescriptorSet::buffer(4, self.avail.clone()),
                WriteDescriptorSet::buffer(5, self_emitters.clone()),
                WriteDescriptorSet::buffer(6, self.avail_count.clone()),
                WriteDescriptorSet::buffer(7, self.particle_template.clone()),
                WriteDescriptorSet::buffer(8, uniform_sub_buffer.clone()),
                WriteDescriptorSet::buffer(9, self.particle_template_ids.clone()),
                WriteDescriptorSet::buffer(10, emitter_inits.clone()),
            ],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set.clone(),
            )
            .dispatch([len as u32 / 1024 + 1, 1, 1])
            .unwrap();
    }

    pub fn emitter_update(
        &self,
        device: Arc<Device>,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        emitter_len: usize,
        dt: f32,
        time: f32,
        cam_pos: [f32; 3],
        cam_rot: [f32; 4],
        mem: Arc<StandardMemoryAllocator>,
        command_allocator: &StandardCommandBufferAllocator,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
    ) {
        let emitter_len = emitter_len.max(1);
        let emitter_inits = DeviceLocalBuffer::<[cs::ty::emitter_init]>::array(
            &mem,
            1 as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        let uniform_sub_buffer = {
            let uniform_data = cs::ty::Data {
                num_jobs: emitter_len as i32,
                dt: dt,
                time,
                stage: 1,
                cam_pos,
                cam_rot,
                MAX_PARTICLES,
                _dummy0: Default::default(),
            };
            self.compute_uniforms.from_data(uniform_data).unwrap()
        };
        let descriptor_set = PersistentDescriptorSet::new(
            &desc_allocator,
            self.compute_pipeline
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, transform.clone()),
                WriteDescriptorSet::buffer(1, self.particles.clone()),
                WriteDescriptorSet::buffer(2, self.particle_positions_lifes.clone()),
                // WriteDescriptorSet::buffer(3, self.particle_lifes.clone()),
                WriteDescriptorSet::buffer(4, self.avail.clone()),
                WriteDescriptorSet::buffer(5, self.emitters.lock().clone()),
                WriteDescriptorSet::buffer(6, self.avail_count.clone()),
                WriteDescriptorSet::buffer(7, self.particle_template.clone()),
                WriteDescriptorSet::buffer(8, uniform_sub_buffer.clone()),
                WriteDescriptorSet::buffer(9, self.particle_template_ids.clone()),
                WriteDescriptorSet::buffer(10, emitter_inits.clone()),
            ],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set.clone(),
            )
            .dispatch([emitter_len as u32 / 1024 + 1, 1, 1])
            .unwrap();
    }

    pub fn particle_update(
        &self,
        device: Arc<Device>,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        dt: f32,
        time: f32,
        cam_pos: [f32; 3],
        cam_rot: [f32; 4],
        mem: Arc<StandardMemoryAllocator>,
        command_allocator: &StandardCommandBufferAllocator,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
    ) {
        let emitter_inits = DeviceLocalBuffer::<[cs::ty::emitter_init]>::array(
            &mem,
            1 as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        let uniform_sub_buffer = {
            let uniform_data = cs::ty::Data {
                num_jobs: MAX_PARTICLES,
                dt: dt,
                time,
                stage: 2,
                cam_pos,
                cam_rot,
                MAX_PARTICLES,
                _dummy0: Default::default(),
            };
            self.compute_uniforms.from_data(uniform_data).unwrap()
        };
        let descriptor_set = PersistentDescriptorSet::new(
            &desc_allocator,
            self.compute_pipeline
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, transform.clone()),
                WriteDescriptorSet::buffer(1, self.particles.clone()),
                WriteDescriptorSet::buffer(2, self.particle_positions_lifes.clone()),
                // WriteDescriptorSet::buffer(3, self.particle_lifes.clone()),
                WriteDescriptorSet::buffer(4, self.avail.clone()),
                WriteDescriptorSet::buffer(5, self.emitters.lock().clone()),
                WriteDescriptorSet::buffer(6, self.avail_count.clone()),
                WriteDescriptorSet::buffer(7, self.particle_template.clone()),
                WriteDescriptorSet::buffer(8, uniform_sub_buffer.clone()),
                WriteDescriptorSet::buffer(9, self.particle_template_ids.clone()),
                WriteDescriptorSet::buffer(10, emitter_inits.clone()),
            ],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set.clone(),
            )
            .dispatch([MAX_PARTICLES as u32 / 1024 + 1, 1, 1])
            .unwrap();
    }
    pub fn render_particles(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        view: glm::Mat4,
        proj: glm::Mat4,
        cam_rot: [f32; 4],
        cam_pos: [f32; 3],
        mem: Arc<StandardMemoryAllocator>,
        command_allocator: &StandardCommandBufferAllocator,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
    ) {
        builder.bind_pipeline_graphics(self.render_pipeline.clone());
        let layout = self.render_pipeline.layout().set_layouts().get(0).unwrap();

        let mut descriptors = Vec::new();
        let uniform_sub_buffer = {
            let uniform_data = gs::ty::Data {
                view: view.into(),
                proj: proj.into(),
                cam_rot,
                cam_pos,
                _dummy0: Default::default(),
            };
            self.render_uniforms.from_data(uniform_data).unwrap()
        };

        descriptors.push(WriteDescriptorSet::buffer(
            0,
            self.particle_positions_lifes.clone(),
        ));
        // descriptors.push(WriteDescriptorSet::buffer(2, self.particles.clone()));
        descriptors.push(WriteDescriptorSet::buffer(
            3,
            self.particle_template.clone(),
        ));
        descriptors.push(WriteDescriptorSet::buffer(4, uniform_sub_buffer.clone()));
        descriptors.push(WriteDescriptorSet::buffer(5, self.sort.a2.clone()));
        descriptors.push(WriteDescriptorSet::buffer(
            6,
            self.particle_template_ids.clone(),
        ));

        // if let Some(texture) = mesh.texture.as_ref() {
        //     descriptors.push(WriteDescriptorSet::image_view_sampler(
        //         1,
        //         texture.image.clone(),
        //         texture.sampler.clone(),
        //     ));
        // } else {
        //     descriptors.push(WriteDescriptorSet::image_view_sampler(
        //         1,
        //         self.def_texture.clone(),
        //         self.def_sampler.clone(),
        //     ));
        // }
        // descriptors.push(WriteDescriptorSet::buffer(2, instance_buffer.clone()));
        let set =
            PersistentDescriptorSet::new(&desc_allocator, layout.clone(), descriptors).unwrap();

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.render_pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .draw_indirect(self.sort.draw.clone())
            .unwrap();
        // .draw(MAX_PARTICLES as u32, 1, 0, 0)
        // .unwrap();

        // let temp = self.sort.a1.clone();
        // self.sort.a1 = self.sort.a2.clone();
        // self.sort.a2 = temp.clone();
    }
}
