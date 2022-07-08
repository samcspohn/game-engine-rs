use std::sync::Arc;

use bytemuck::{Zeroable, Pod};
use vulkano::{
    buffer::{cpu_pool::CpuBufferPoolSubbuffer, CpuAccessibleBuffer, TypedBufferAccess},
    command_buffer::{AutoCommandBufferBuilder, PrimaryAutoCommandBuffer},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Device,
    memory::pool::StdMemoryPool,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState}, rasterization::{RasterizationState, CullMode},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{RenderPass, Subpass},
    shader::ShaderModule, impl_vertex,
};

use crate::{
    model::{Normal, Vertex, Mesh},
    // ModelMat,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct ModelMat {
    pub pos: [f32; 3],
}
impl_vertex!(ModelMat, pos);

use self::vs::ty::Data;

pub struct RenderPipeline {
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    pipeline: Arc<GraphicsPipeline>,
    // vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    // index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
    // normals_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
    // instance_buffer: Arc<CpuAccessibleBuffer<[ModelMat]>>,
}


impl RenderPipeline {
    pub fn new(
        device: Arc<Device>,
        render_pass: Arc<RenderPass>,
        dimensions: [u32; 2],
        // vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
        // index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
        // normals_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
    ) -> RenderPipeline {
        let vs = vs::load(device.clone()).unwrap();
        let fs = fs::load(device.clone()).unwrap();
        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(
                BuffersDefinition::new()
                    .vertex::<Vertex>()
                    .vertex::<Normal>()
                    .instance::<ModelMat>(),
            )
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                    depth_range: 0.0..1.0,
                },
            ]))
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap();

        RenderPipeline {
            vs,
            fs,
            pipeline,
            // vertex_buffer,
            // index_buffer,
            // normals_buffer,
        }
    }
    pub fn regen(
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
                    .instance::<ModelMat>(),
            )
            .vertex_shader(self.vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                    depth_range: 0.0..1.0,
                },
            ]))
            .fragment_shader(self.fs.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap();
    }

    pub fn bind_pipeline(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        uniform_buffer_subbuffer: &Arc<CpuBufferPoolSubbuffer<Data, Arc<StdMemoryPool>>>,
    ) -> &RenderPipeline {
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            layout.clone(),
            [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer.clone())],
        )
        .unwrap();

        builder
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                set.clone(),
            );
            self
    }

    pub fn bind_mesh(
        &self,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        instance_buffer: Arc<CpuAccessibleBuffer<[ModelMat]>>,
        mesh: &Mesh,
    ) -> &RenderPipeline {
        builder
            .bind_vertex_buffers(
                0,
                (
                    mesh.vertex_buffer.clone(),
                    mesh.normals_buffer.clone(),
                    instance_buffer.clone(),
                ),
            )
            .bind_index_buffer(mesh.index_buffer.clone())
            .draw_indexed(
                mesh.index_buffer.len() as u32,
                instance_buffer.len() as u32,
                0,
                0,
                0,
            )
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
