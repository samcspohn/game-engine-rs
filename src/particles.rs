use std::sync::Arc;

use crate::transform_compute::cs::ty::transform;
use nalgebra_glm as glm;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, DeviceLocalBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
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
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo}, sync::{self, GpuFuture, FlushError}, swapchain::Swapchain,
};
use winit::window::Window;

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/particles.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/particles_geom/particles.vert",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
pub mod gs {
    vulkano_shaders::shader! {
        ty: "geometry",
        path: "src/particles_geom/particles.geom",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/particles_geom/particles.frag"
    }
}

pub const MAX_PARTICLES: i32 = 8 * 1024 * 1024;

pub struct ParticleCompute {
    pub particles: Arc<DeviceLocalBuffer<[cs::ty::particle]>>,
    pub particle_positions: Arc<DeviceLocalBuffer<[[f32; 4]]>>,
    pub render_pipeline: Arc<GraphicsPipeline>,
    pub compute_pipeline: Arc<ComputePipeline>,
    pub compute_uniforms: CpuBufferPool<cs::ty::Data>,
    pub render_uniforms: CpuBufferPool<vs::ty::Data>,
    pub def_texture: Arc<ImageView<ImmutableImage>>,
    pub def_sampler: Arc<Sampler>,
}

impl ParticleCompute {
    pub fn new(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        // dimensions: [u32; 2],
        swapchain: Arc<Swapchain<Window>>,
        queue: Arc<Queue>,
    ) -> ParticleCompute {
        let q = glm::Quat::identity();
        let particles: Vec<cs::ty::particle> = (0..MAX_PARTICLES)
            .into_iter()
            .map(|_| cs::ty::particle {
                emitter_id: 0,
                _dummy0: Default::default(),
                rot: q.coords.into(),
                proto_id: 0,
                _dummy1: Default::default(),
            })
            .collect();
        let copy_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, particles)
                .unwrap();
        let particles = DeviceLocalBuffer::<[cs::ty::particle]>::array(
            device.clone(),
            MAX_PARTICLES as vulkano::DeviceSize,
            BufferUsage::all(),
            // BufferUsage::storage_buffer()
            // | BufferUsage::vertex_buffer_transfer_destination()
            // | BufferUsage::transfer_source(), // Specify use as a storage buffer, vertex buffer, and transfer destination.
            device.active_queue_families(),
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder.copy_buffer(copy_buffer, particles.clone()).unwrap();

        let particle_positions: Vec<[f32; 4]> = (0..MAX_PARTICLES)
            .into_iter()
            .map(|_| {
                [
                    (rand::random::<f32>() - 0.5) * 100.,
                    (rand::random::<f32>() - 0.5) * 100.,
                    (rand::random::<f32>() - 0.5) * 100.,
                    0.,
                ]
            })
            .collect();
        let copy_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            particle_positions,
        )
        .unwrap();
        let particle_positions = DeviceLocalBuffer::<[[f32; 4]]>::array(
            device.clone(),
            MAX_PARTICLES as vulkano::DeviceSize,
            BufferUsage::all(),
            // BufferUsage::storage_buffer()
            // | BufferUsage::vertex_buffer_transfer_destination()
            // | BufferUsage::transfer_source(),
            device.active_queue_families(),
        )
        .unwrap();

        builder
            .copy_buffer(copy_buffer, particle_positions.clone())
            .unwrap();

        let command_buffer = builder.build().unwrap();
        

        let execute = Some(sync::now(device.clone()).boxed())
            .take()
            .unwrap()
            .then_execute(queue.clone(), command_buffer);

        match execute {
            Ok(execute) => {
                let future = execute
                    .then_signal_fence_and_flush();
                match future {
                    Ok(future) => {
                        
                    }
                    Err(FlushError::OutOfDate) => {
                       
                    }
                    Err(e) => {
                        
                    }
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

        let uniforms = CpuBufferPool::<cs::ty::Data>::new(device.clone(), BufferUsage::all());
        let render_uniforms =
            CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::all());

        let def_texture = {
            let dimensions = ImageDimensions::Dim2d {
                width: 1,
                height: 1,
                array_layers: 1,
            };
            let image_data = vec![255 as u8, 255, 255, 255];
            let image = ImmutableImage::from_iter(
                image_data,
                dimensions,
                MipmapsCount::One,
                Format::R8G8B8A8_SRGB,
                queue.clone(),
            )
            .unwrap()
            .0;

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

        ParticleCompute {
            particles,
            particle_positions,
            render_pipeline,
            compute_pipeline,
            compute_uniforms: uniforms,
            render_uniforms,
            def_texture,
            def_sampler,
        }
    }

    pub fn particle_update(
        &self,
        device: Arc<Device>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        dt: f32,
    ) {
        let uniform_sub_buffer = {
            let uniform_data = cs::ty::Data {
                num_jobs: MAX_PARTICLES,
                dt: dt,
            };
            self.compute_uniforms.next(uniform_data).unwrap()
        };
        let descriptor_set = PersistentDescriptorSet::new(
            self.compute_pipeline
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                // WriteDescriptorSet::buffer(0, transform.clone()),
                WriteDescriptorSet::buffer(1, self.particles.clone()),
                WriteDescriptorSet::buffer(2, self.particle_positions.clone()),
                WriteDescriptorSet::buffer(3, uniform_sub_buffer.clone()),
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
            .dispatch([MAX_PARTICLES as u32 / 128 + 1, 1, 1])
            .unwrap();
    }
    pub fn render_particles(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        view: glm::Mat4,
        proj: glm::Mat4,
    ) {
        builder.bind_pipeline_graphics(self.render_pipeline.clone());
        let layout = self.render_pipeline.layout().set_layouts().get(0).unwrap();

        let mut descriptors = Vec::new();
        let uniform_sub_buffer = {
            let uniform_data = vs::ty::Data {
                view: view.into(),
                proj: proj.into(),
            };
            self.render_uniforms.next(uniform_data).unwrap()
        };

        descriptors.push(WriteDescriptorSet::buffer(
            0,
            self.particle_positions.clone(),
        ));
        descriptors.push(WriteDescriptorSet::buffer(1, uniform_sub_buffer.clone()));

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
        let set = PersistentDescriptorSet::new(layout.clone(), descriptors).unwrap();

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.render_pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .draw(MAX_PARTICLES as u32, 1, 0, 0)
            .unwrap();
    }
}
