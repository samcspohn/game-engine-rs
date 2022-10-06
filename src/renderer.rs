use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use vulkano::{
    buffer::{
        cpu_pool::CpuBufferPoolSubbuffer, BufferSlice, CpuAccessibleBuffer, CpuBufferPool,
        DeviceLocalBuffer, TypedBufferAccess,
    },
    command_buffer::{
        AutoCommandBufferBuilder, DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
    impl_vertex,
    memory::pool::StdMemoryPool,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    shader::ShaderModule,
};

use crate::{
    model::{Mesh, Normal, Vertex, UV},
    transform_compute::MVP,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct ModelMat {
    pub pos: [f32; 3],
    pub padding: f32,
}
impl_vertex!(ModelMat, pos);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Id {
    pub id: i32,
}
impl_vertex!(Id, id);

// use self::vs::ty::Data;

pub struct RenderPipeline {
    _vs: Arc<ShaderModule>,
    _fs: Arc<ShaderModule>,
    pipeline: Arc<GraphicsPipeline>,
    pub def_texture: Arc<ImageView<ImmutableImage>>,
    pub def_sampler: Arc<Sampler>,
}

impl RenderPipeline {
    pub fn new(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        dimensions: [u32; 2],
        queue: Arc<Queue>,
    ) -> RenderPipeline {
        let vs = vs::load(device.clone()).unwrap();
        let fs = fs::load(device.clone()).unwrap();
        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(
                BuffersDefinition::new()
                    .vertex::<Vertex>()
                    .vertex::<Normal>()
                    .vertex::<UV>(), // .instance::<Id>(),
            )
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            // .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
            //     Viewport {
            //         origin: [0.0, 0.0],
            //         dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            //         depth_range: 0.0..1.0,
            //     },
            // ]))
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap();

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

        RenderPipeline {
            _vs: vs,
            _fs: fs,
            pipeline,
            def_texture,
            def_sampler,
        }
    }
    pub fn _regen(
        &mut self,
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        dimensions: [u32; 2],
    ) {
        self.pipeline = GraphicsPipeline::start()
            .vertex_input_state(
                BuffersDefinition::new()
                    .vertex::<Vertex>()
                    .vertex::<Normal>()
                    .vertex::<UV>(), // .instance::<Id>(),
            )
            .vertex_shader(self._vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            // .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
            //     Viewport {
            //         origin: [0.0, 0.0],
            //         dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            //         depth_range: 0.0..1.0,
            //     },
            // ]))
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(self._fs.entry_point("main").unwrap(), ())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap();
    }

    pub fn bind_pipeline(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    ) -> &RenderPipeline {
        builder.bind_pipeline_graphics(self.pipeline.clone());

        self
    }

    pub fn bind_mesh(
        &self,
        // device: Arc<Device>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        instance_buffer: Arc<BufferSlice<[i32], CpuAccessibleBuffer<[i32]>>>,
        mvp_buffer: Arc<DeviceLocalBuffer<[MVP]>>,
        mesh: &Mesh,
        indirect_buffer: Arc<
            BufferSlice<
                [DrawIndexedIndirectCommand],
                CpuAccessibleBuffer<[DrawIndexedIndirectCommand]>,
            >,
        >,
    ) -> &RenderPipeline {
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        let mut descriptors = Vec::new();

        descriptors.push(WriteDescriptorSet::buffer(0, mvp_buffer.clone()));

        if let Some(texture) = mesh.texture.as_ref() {
            descriptors.push(WriteDescriptorSet::image_view_sampler(
                1,
                texture.image.clone(),
                texture.sampler.clone(),
            ));
        } else {
            descriptors.push(WriteDescriptorSet::image_view_sampler(
                1,
                self.def_texture.clone(),
                self.def_sampler.clone(),
            ));
        }
        descriptors.push(WriteDescriptorSet::buffer(2, instance_buffer.clone()));
        let set = PersistentDescriptorSet::new(layout.clone(), descriptors).unwrap();

        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                set.clone(),
            )
            .bind_vertex_buffers(
                0,
                (
                    mesh.vertex_buffer.clone(),
                    mesh.normals_buffer.clone(),
                    mesh.uvs_buffer.clone(),
                    // instance_buffer.clone(),
                ),
            )
            // .bind_vertex_buffers(1, transforms_buffer.data.clone())
            .bind_index_buffer(mesh.index_buffer.clone())
            .draw_indexed_indirect(indirect_buffer)
            .unwrap();
        self
    }

    // pub fn bind_finish(
    //     &self,
    //     builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    // ) {
    //     builder
    //         .unwrap();
    // }
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/vert.glsl",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/frag.glsl"
    }
}
