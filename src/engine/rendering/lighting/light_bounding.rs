use std::sync::Arc;

use nalgebra_glm::Mat4;
use vulkano::{
    buffer::{allocator::SubbufferAllocator, Subbuffer},
    command_buffer::{
        DrawIndexedIndirectCommand, DrawIndirectCommand, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    format::Format,
    image::{ImageAccess, AttachmentImage, ImageUsage, view::ImageView},
    memory::allocator::MemoryUsage,
    pipeline::{
        self,
        graphics::{
            input_assembly::InputAssemblyState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, RenderPass, Subpass, FramebufferCreateInfo},
};

use crate::engine::{
    prelude::{utils, VulkanManager},
    rendering::model::_Vertex,
};

use super::lighting_compute::cs::light;

// use super::vulkan_manager::VulkanManager;

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/cluster_light.vert",
    }
}
pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/cluster_light.frag",
    }
}
pub struct LightBounding {
    pipeline: Arc<pipeline::GraphicsPipeline>,
    uniforms: SubbufferAllocator,
    render_pass: Arc<RenderPass>,
    vk: Arc<VulkanManager>,
    frame_buffer: Arc<Framebuffer>,
    // light_mvps: Subbuffer<[[[f32; 4]; 4]]>,
    // pub image: Arc<dyn ImageAccess>,
    viewport: Viewport,
    // framebuffer: Arc<Framebuffer>,
}
impl LightBounding {
    pub fn new(vk: Arc<VulkanManager>) -> Self {
        let render_pass = vulkano::single_pass_renderpass!(
            vk.device.clone(),
            attachments: {
                color: {
                    load: DontCare,
                    store: Store,
                    format: Format::R8_SINT,
                    // Same here, this has to match.
                    samples: 1,
                }
            },
            pass:
            { color: [color], depth_stencil: {}, }
        )
        .unwrap();
        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [16.0, 9.0],
            depth_range: 0.0..1.0,
        };
        let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
        let vs = vs::load(vk.device.clone()).unwrap();
        let fs = fs::load(vk.device.clone()).unwrap();
        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new().vertex::<_Vertex>())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .input_assembly_state(InputAssemblyState::new())
            .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
                viewport.clone()
            ]))
            // .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
            .render_pass(subpass)
            .build(vk.device.clone())
            .unwrap();
        let image = AttachmentImage::with_usage(
            &vk.mem_alloc,
            [16,9],
            Format::R8_SINT,
            ImageUsage::TRANSIENT_ATTACHMENT,
            // ImageUsage::SAMPLED
            //     | ImageUsage::STORAGE
            //     | ImageUsage::COLOR_ATTACHMENT
            //     | ImageUsage::TRANSFER_SRC, // | ImageUsage::INPUT_ATTACHMENT,
        )
        .unwrap();
        let view = ImageView::new_default(image.clone()).unwrap();
        LightBounding {
            pipeline,
            frame_buffer: Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap(),
            render_pass,
            uniforms: vk.sub_buffer_allocator(),
            // light_mvps: vk.buffer_array(1, MemoryUsage::DeviceOnly),
            vk,
            viewport,
        }
    }
    pub fn render(
        &self,
        builder: &mut utils::PrimaryCommandBuffer,
        visible_lights: Subbuffer<[u32]>,
        lights: Subbuffer<[light]>,
        clusters: Subbuffer<[u32]>,
        draw: Subbuffer<[DrawIndirectCommand]>,
        vp: Mat4,
    ) {
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![None],
                    ..RenderPassBeginInfo::framebuffer(self.frame_buffer.clone())
                },
                SubpassContents::Inline,
            )
            .unwrap();
        builder.set_viewport(0, [self.viewport.clone()]);
        builder.bind_pipeline_graphics(self.pipeline.clone());
        let ub = {
            let u: Subbuffer<vs::UniformBufferObject> = self.uniforms.allocate_sized().unwrap();
            {
                let mut _u = u.write().unwrap();
                *_u = vs::UniformBufferObject { vp: vp.into() };
            }
            u
        };
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();
        let set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            layout.clone(),
            [
                WriteDescriptorSet::buffer(0, ub),
                WriteDescriptorSet::buffer(1, lights.clone()),
                WriteDescriptorSet::buffer(2, visible_lights.clone()),
                WriteDescriptorSet::buffer(3, clusters.clone()),
            ],
        )
        .unwrap();
        builder
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                set,
            )
            .draw_indirect(draw)
            // .draw(6, 2, 0, 0)
            // .bind_vertex_buffers(
            //     0,
            //     (
            //         mesh.vertex_buffer.clone(),
            //         mesh.normals_buffer.clone(),
            //         mesh.uvs_buffer.clone(),
            //         // instance_buffer.clone(),
            //     ),
            // )
            // .bind_vertex_buffers(1, transforms_buffer.data.clone())
            // .bind_index_buffer(mesh.index_buffer.clone())
            // .draw_indexed_indirect(indirect_buffer)
            .unwrap();
    }
}
