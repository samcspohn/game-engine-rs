use std::sync::Arc;

use glium::buffer::Content;
use nalgebra_glm::{Vec3, Vec4};
use vulkano::{
    buffer::Subbuffer, descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet}, memory::allocator::MemoryUsage, padded::Padded, pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            multisample::MultisampleState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    }, render_pass::{RenderPass, Subpass}
};

use crate::engine::{particles::shaders::scs, utils::PrimaryCommandBuffer};

use super::{
    camera::{CameraViewData, _Frustum},
    vulkan_manager::VulkanManager,
};

pub mod gs1 {
    vulkano_shaders::shader! {
        ty: "geometry",
        path: "src/shaders/debug1.geom",
    }
}
pub mod gs2 {
    vulkano_shaders::shader! {
        ty: "geometry",
        path: "src/shaders/debug2.geom",
    }
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/debug1.vert",
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/debug1.frag",
    }
}

pub struct DebugSystem {
    shader1: Arc<GraphicsPipeline>,
    shader2: Arc<GraphicsPipeline>,
    frustums_to_draw: Vec<gs1::DrawFrustum>,
    arrows_to_draw: Vec<gs2::DrawArrow>,
    // frustums_buffer: Subbuffer<[gs::DrawFrustum]>,
    vk: Arc<VulkanManager>,
}
impl DebugSystem {
    pub fn new(vk: Arc<VulkanManager>, render_pass: Arc<RenderPass>) -> Self {
        let vs = vs::load(vk.device.clone()).unwrap();
        let fs = fs::load(vk.device.clone()).unwrap();
        let gs1 = gs1::load(vk.device.clone()).unwrap();
        let gs2 = gs2::load(vk.device.clone()).unwrap();

        let subpass = Subpass::from(render_pass, 0).unwrap();
        // let blend_state = ColorBlendState::new(subpass.num_color_attachments()).blend_alpha();
        // let mut depth_stencil_state = DepthStencilState::simple_depth_test();
        // depth_stencil_state.depth = Some(DepthState {
        //     enable_dynamic: false,
        //     write_enable: StateMode::Fixed(false),
        //     compare_op: StateMode::Fixed(CompareOp::Less),
        // });

        let shader1 = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            // .input_assembly_state(InputAssemblyState::new())
            .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::PointList))
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .geometry_shader(gs1.entry_point("main").unwrap(), ())
            // .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::LineStrip))
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .multisample_state(MultisampleState {
                rasterization_samples: subpass.num_samples().unwrap(),
                ..Default::default()
            })
            // .color_blend_state(blend_state)
            .render_pass(subpass.clone())
            .build(vk.device.clone())
            .unwrap();

        let shader2 = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            // .input_assembly_state(InputAssemblyState::new())
            .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::PointList))
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .geometry_shader(gs2.entry_point("main").unwrap(), ())
            // .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::LineStrip))
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .depth_stencil_state(DepthStencilState::simple_depth_test())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .multisample_state(MultisampleState {
                rasterization_samples: subpass.num_samples().unwrap(),
                ..Default::default()
            })
            // .color_blend_state(blend_state)
            .render_pass(subpass.clone())
            .build(vk.device.clone())
            .unwrap();

        Self {
            shader1,
            shader2,
            frustums_to_draw: Vec::new(),
            arrows_to_draw: Vec::new(),
            // frustums_buffer: vk.buffer_array(1, MemoryUsage::DeviceOnly),
            vk,
        }
    }
    pub fn append_frustum(&mut self, f: gs1::Frustum, color: Vec4) {
        // let f: scs::Frustum = f.into();
        self.frustums_to_draw.push(gs1::DrawFrustum {
            f,
            color: color.into(),
        });
    }
    pub fn append_arrow(&mut self, dir: Vec3, pos: Vec3, size: f32, color: Vec4) {
        self.arrows_to_draw.push(gs2::DrawArrow {
            dir: Padded(dir.into()),
            pos: pos.into(),
            size,
            color: color.into(),
        })
    }
    pub fn draw(&mut self, builder: &mut PrimaryCommandBuffer, cvd: &CameraViewData) {
        {
            let num_frust = self.frustums_to_draw.len();
            let buf: Subbuffer<[gs1::DrawFrustum]> = self.vk.allocate_unsized(self.frustums_to_draw.len() as u64);
            {
                let mut a = buf.write().unwrap();
                let ftd = std::mem::take(&mut self.frustums_to_draw);
                for (df, i) in ftd.into_iter().zip(a.iter_mut()) {
                    *i = df;
                }
            }
            let uni = self.vk.allocate(gs1::Data {
                view: cvd.view.into(),
                proj: cvd.proj.into(),
            });
            let set = PersistentDescriptorSet::new(
                &self.vk.desc_alloc,
                self.shader1.layout().set_layouts().get(0).unwrap().clone(),
                [
                    // WriteDescriptorSet::buffer(0, pb.pos_life_compressed.clone()),
                    WriteDescriptorSet::buffer(0, buf),
                    WriteDescriptorSet::buffer(1, uni),
                ],
            )
            .unwrap();
            match builder
                .bind_pipeline_graphics(self.shader1.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.shader1.layout().clone(),
                    0,
                    set,
                )
                .draw(num_frust as u32, 1, 0, 0)
            {
                Ok(_) => {}
                Err(e) => {
                    println!("{}", e)
                }
            };
        }

        // draw arrows
        {
            let num_arrows = self.arrows_to_draw.len();
            let buf: Subbuffer<[gs2::DrawArrow]> = self.vk.allocate_unsized(num_arrows as u64);
            {
                let mut a = buf.write().unwrap();
                let ftd = std::mem::take(&mut self.arrows_to_draw);
                // for i in 0..8 {
                //     *a[i] = self.arrows_to_draw[i];
                // }
                for (df, i) in ftd.iter().zip(a.iter_mut()) {
                    *i = *df;
                }
            }
            // self.arrows_to_draw.clear();
            let uni = self.vk.allocate(gs2::Data {
                view: cvd.view.into(),
                proj: cvd.proj.into(),
                cam_pos: cvd.cam_pos.into(),
            });
            let set = PersistentDescriptorSet::new(
                &self.vk.desc_alloc,
                self.shader2.layout().set_layouts().get(0).unwrap().clone(),
                [
                    // WriteDescriptorSet::buffer(0, pb.pos_life_compressed.clone()),
                    WriteDescriptorSet::buffer(0, buf),
                    WriteDescriptorSet::buffer(1, uni),
                ],
            )
            .unwrap();
            match builder
                .bind_pipeline_graphics(self.shader2.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.shader2.layout().clone(),
                    0,
                    set,
                )
                .draw(num_arrows as u32, 1, 0, 0)
            {
                Ok(_) => {}
                Err(e) => {
                    println!("{}", e)
                }
            };
        }
    }
}
