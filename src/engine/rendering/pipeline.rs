use std::sync::Arc;

use nalgebra_glm::Vec3;
use vulkano::{
    buffer::{allocator::SubbufferAllocator, Buffer, BufferContents, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, PersistentDescriptorSet,
        WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::{self, Format},
    image::{sampler::Sampler, view::ImageView, Image},
    impl_vertex,
    memory::allocator::StandardMemoryAllocator,
    padded::Padded,
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, RasterizationState},
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{BuffersDefinition, Vertex, VertexDefinition, VertexInputState},
            viewport::ViewportState,
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    render_pass::{RenderPass, Subpass},
    shader::ShaderModule,
};

use crate::engine::{
    rendering::vulkan_manager::VulkanManager,
    transform_compute::cs::{transform, MVP},
    utils,
};

use self::fs::light;

use super::{
    lighting::lighting_compute::{
        cs,
        lt::{self, tile},
    },
    model::{Mesh, Normal, _Vertex, UV},
    texture::{self, TextureManager},
};

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/model.vert",
        // types_meta: {
        //     use bytemuck::{Pod, Zeroable};

        //     #[derive(Clone, Copy, Zeroable, Pod)]
        // },
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/model.frag"
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, BufferContents, Vertex)]
pub struct ModelMat {
    #[format(R32G32B32_SFLOAT)]
    pub pos: [f32; 3],
}
// impl_vertex!(ModelMat, pos);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, BufferContents)]
pub struct Id {
    pub id: i32,
}
// impl_vertex!(Id, id);

// use self::vs::ty::Data;

pub struct RenderPipeline {
    // _vs: Arc<ShaderModule>,
    // _fs: Arc<ShaderModule>,
    pub pipeline: Arc<GraphicsPipeline>,
    // pub uniforms: SubbufferAllocator,
    pub def_texture: Arc<ImageView>,
    pub def_sampler: Arc<Sampler>,
    vk: Arc<VulkanManager>,
}

#[derive(BufferContents, Vertex)]
#[repr(C)]
struct VertU32 {
    #[format(R32_UINT)]
    a: u32,
}
#[derive(BufferContents, Vertex)]
#[repr(C)]
struct VertU32_2 {
    #[format(R32_UINT)]
    a: u32,
}
impl RenderPipeline {
    pub fn new(
        sub_pass_index: u32,
        vk: Arc<VulkanManager>,
        render_pass: Arc<RenderPass>,
        // use_msaa: bool,
    ) -> RenderPipeline {
        // let vs = vs::load(vk.device.clone())
        //     .unwrap()
        //     .entry_point("main")
        //     .unwrap();
        // let fs = fs::load(vk.device.clone())
        //     .unwrap()
        //     .entry_point("main")
        //     .unwrap();

        // let pipeline = utils::pipeline::graphics_pipeline(
        //     vk.clone(),
        //     &[vs, fs],
        //     &[
        //         _Vertex::per_vertex(),
        //         Normal::per_vertex(),
        //         UV::per_vertex(),
        //     ],
        //     |g| {},
        //     render_pass.clone(),
        // );

        let vs = vs::load(vk.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = fs::load(vk.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        let vertex_input_state = [
            _Vertex::per_vertex(),
            Normal::per_vertex(),
            UV::per_vertex(),
        ]
        .definition(&vs.info().input_interface)
        .unwrap();

        let stages = [
            PipelineShaderStageCreateInfo::new(vs),
            PipelineShaderStageCreateInfo::new(fs),
        ];

        let layout = PipelineLayout::new(
            vk.device.clone(),
            PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                .into_pipeline_layout_create_info(vk.device.clone())
                .unwrap(),
        )
        .unwrap();

        let subpass = Subpass::from(render_pass.clone(), sub_pass_index).unwrap();

        let pipeline = GraphicsPipeline::new(
            vk.device.clone(),
            None,
            GraphicsPipelineCreateInfo {
                stages: stages.into_iter().collect(),
                vertex_input_state: Some(vertex_input_state),
                input_assembly_state: Some(InputAssemblyState::default()),
                viewport_state: Some(ViewportState::default()),
                rasterization_state: Some(RasterizationState::default().cull_mode(CullMode::Back)),
                multisample_state: Some(MultisampleState::default()),
                color_blend_state: Some(ColorBlendState::with_attachment_states(
                    subpass.num_color_attachments(),
                    ColorBlendAttachmentState::default(),
                )),
                depth_stencil_state: Some(DepthStencilState::simple_depth_test()),
                dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                subpass: Some(subpass.into()),
                ..GraphicsPipelineCreateInfo::layout(layout)
            },
        )
        .unwrap();

        // let vertex_input_state = [
        //     _Vertex::per_vertex(),
        //     Normal::per_vertex(),
        //     UV::per_vertex(),
        // ]
        // .definition(&vs)
        // .unwrap();
        // // VertexInputState::new().attribute(0, _Vertex::per_vertex().definition(&vs)).attribute(1, Normal::per_vertex().definition(&vs)).attribute(2, UV::per_vertex().definition(&vs));

        // let stages = [
        //     PipelineShaderStageCreateInfo::new(vs),
        //     PipelineShaderStageCreateInfo::new(fs),
        // ];
        // let layout = PipelineLayout::new(
        //     vk.device.clone(),
        //     PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
        //         .into_pipeline_layout_create_info(vk.device.clone())
        //         .unwrap(),
        // )
        // .unwrap();

        // let subpass = PipelineRenderingCreateInfo {
        //     color_attachment_formats: vec![Some(vk.swapchain().image_format())],
        //     ..Default::default()
        // };
        // let pipeline = GraphicsPipeline::new(
        //     vk.device.clone(),
        //     None,
        //     GraphicsPipelineCreateInfo {
        //         stages: stages.into_iter().collect(),
        //         vertex_input_state: Some(vertex_input_state),
        //         input_assembly_state: Some(InputAssemblyState::default()),
        //         viewport_state: Some(ViewportState::default()),
        //         rasterization_state: Some(RasterizationState::default().cull_mode(CullMode::Back)),
        //         multisample_state: Some(MultisampleState::default()),
        //         color_blend_state: Some(ColorBlendState::with_attachment_states(
        //             subpass.color_attachment_formats.len() as u32,
        //             ColorBlendAttachmentState::default(),
        //         )),
        //         dynamic_state: [DynamicState::Viewport].into_iter().collect(),
        //         subpass: Some(subpass.into()),
        //         ..GraphicsPipelineCreateInfo::layout(layout)
        //     },
        // )
        // .unwrap();
        // let pipeline = GraphicsPipeline::start()
        //     .vertex_input_state(
        //         BuffersDefinition::new()
        //             .vertex::<_Vertex>()
        //             .vertex::<Normal>()
        //             .vertex::<UV>(), // .instance::<Id>(),
        //                              // .vertex::<VertU32>()
        //                              // .vertex::<VertU32_2>(),
        //     )
        //     .vertex_shader(vs.entry_point("main").unwrap(), ())
        //     .input_assembly_state(InputAssemblyState::new())
        //     // .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
        //     //     Viewport {
        //     //         origin: [0.0, 0.0],
        //     //         dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        //     //         depth_range: 0.0..1.0,
        //     //     },
        //     // ]))
        //     .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        //     .fragment_shader(fs.entry_point("main").unwrap(), ())
        //     .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        //     .depth_stencil_state(DepthStencilState::simple_depth_test())
        //     .multisample_state(MultisampleState {
        //         rasterization_samples: subpass.num_samples().unwrap(),
        //         ..Default::default()
        //     })
        //     .render_pass(subpass)
        //     .build(vk.device.clone())
        //     .unwrap();
        // let pipeline = if use_msaa {
        //     pipeline.multisample_state(MultisampleState {
        //         rasterization_samples: subpass.num_samples().unwrap(),
        //         ..Default::default()
        //     })
        // } else {
        //     pipeline
        // };
        // let pipeline = pipeline

        let mut builder = AutoCommandBufferBuilder::primary(
            &vk.comm_alloc,
            vk.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let (def_texture, def_sampler) =
            texture::texture_from_bytes(vk.clone(), &vec![255_u8, 255, 255, 255], 1, 1);
        // let def_texture = {
        //     let dimensions = ImageDimensions::Dim2d {
        //         width: 1,
        //         height: 1,
        //         array_layers: 1,
        //     };
        //     let image_data = vec![255_u8, 255, 255, 255];
        //     let image = ImmutableImage::from_iter(
        //         &vk.mem_alloc,
        //         image_data,
        //         dimensions,
        //         MipmapsCount::One,
        //         Format::R8G8B8A8_SRGB,
        //         &mut builder,
        //     )
        //     .unwrap();
        //     ImageView::new_default(image).unwrap()
        // };
        // let _ = builder.build().unwrap().execute(vk.queue.clone()).unwrap();

        // let def_sampler = Sampler::new(
        //     vk.device.clone(),
        //     SamplerCreateInfo {
        //         mag_filter: Filter::Linear,
        //         min_filter: Filter::Linear,
        //         address_mode: [SamplerAddressMode::Repeat; 3],
        //         ..Default::default()
        //     },
        // )
        // .unwrap();

        RenderPipeline {
            // _vs: vs,
            // _fs: fs,
            pipeline,
            def_texture,
            def_sampler,
            // uniforms: vk.sub_buffer_allocator(),
            vk,
        }
    }
    // pub fn _regen(
    //     &mut self,
    //     device: Arc<Device>,
    //     render_pass: Arc<RenderPass>,
    //     _dimensions: [u32; 2],
    // ) {
    //     self.pipeline = GraphicsPipeline::start()
    //         .vertex_input_state(
    //             BuffersDefinition::new()
    //                 .vertex::<_Vertex>()
    //                 .vertex::<Normal>()
    //                 .vertex::<UV>(), // .instance::<Id>(),
    //         )
    //         .vertex_shader(self._vs.entry_point("main").unwrap(), ())
    //         .input_assembly_state(InputAssemblyState::new())
    //         // .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
    //         //     Viewport {
    //         //         origin: [0.0, 0.0],
    //         //         dimensions: [dimensions[0] as f32, dimensions[1] as f32],
    //         //         depth_range: 0.0..1.0,
    //         //     },
    //         // ]))
    //         .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
    //         .fragment_shader(self._fs.entry_point("main").unwrap(), ())
    //         .rasterization_state(RasterizationState::new().cull_mode(CullMode::Front))
    //         .depth_stencil_state(DepthStencilState::simple_depth_test())
    //         .render_pass(Subpass::from(render_pass, 0).unwrap())
    //         .build(device)
    //         .unwrap();
    // }

    pub fn bind_pipeline(&self, builder: &mut utils::PrimaryCommandBuffer) -> &RenderPipeline {
        builder.bind_pipeline_graphics(self.pipeline.clone());

        self
    }

    pub fn bind_mesh(
        &self,
        texture_manager: &TextureManager,
        builder: &mut utils::PrimaryCommandBuffer,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
        instance_buffer: Subbuffer<[[i32; 2]]>,
        mvp_buffer: Subbuffer<[MVP]>,
        ////
        light_len: u32,
        lights: Subbuffer<[lt::light]>,
        light_templates: Subbuffer<[fs::lightTemplate]>,
        tiles: Subbuffer<[tile]>,
        screen_dims: [f32; 2],
        bounding_line_hierarchy: Subbuffer<[cs::BoundingLine]>,
        /////
        transforms: Subbuffer<[transform]>,
        mesh: &Mesh,
        indirect_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,
        cam_pos: Vec3,
        light_list: Subbuffer<[u32]>,
        skeleton: Option<&Subbuffer<[[[f32; 4]; 3]]>>,
        has_skeleton: bool,
        empty: Subbuffer<[i32]>,
        num_bones: i32,
    ) -> &RenderPipeline {
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        let mut descriptors = Vec::new();

        descriptors.push(WriteDescriptorSet::buffer(0, mvp_buffer));

        descriptors.push(WriteDescriptorSet::buffer(1, instance_buffer));
        // descriptors.push(WriteDescriptorSet::buffer(3, transforms));
        let uniform = self.vk.allocate(fs::Data { screen_dims });
        let vs_uniform = self.vk.allocate(vs::UniformBufferObject {
            has_skeleton: if has_skeleton { 1 } else { 0 },
            num_bones,
        });
        // descriptors.push(WriteDescriptorSet::buffer(3, light_templates));
        // descriptors.push(WriteDescriptorSet::buffer(4, lights));
        // descriptors.push(WriteDescriptorSet::buffer(5, tiles));
        // descriptors.push(WriteDescriptorSet::buffer(6, uniform));
        // descriptors.push(WriteDescriptorSet::buffer(7, light_list));
        // descriptors.push(WriteDescriptorSet::buffer(7, mesh.bone_weight_offsets));
        if let Some(skel) = skeleton {
            descriptors.push(WriteDescriptorSet::buffer(2, skel.clone()));
        } else {
            descriptors.push(WriteDescriptorSet::buffer(2, empty.clone()));
        }
        if let Some(buf) = mesh.bone_weights_buffer.as_ref() {
            descriptors.push(WriteDescriptorSet::buffer(3, buf.clone()));
        } else {
            descriptors.push(WriteDescriptorSet::buffer(3, empty.clone()));
        }
        descriptors.push(WriteDescriptorSet::buffer(4, vs_uniform));
        descriptors.push(WriteDescriptorSet::buffer(
            5,
            mesh.bone_weights_offsets_counts_buf.clone(),
        ));
        // descriptors.push(WriteDescriptorSet::buffer(6, bounding_line_hierarchy));

        if let Some(texture) = mesh.texture.as_ref() {
            let texture = texture_manager.get_id(texture).unwrap().lock();
            descriptors.push(WriteDescriptorSet::image_view_sampler(
                7,
                texture.image.clone(),
                texture.sampler.clone(),
            ));
        } else {
            descriptors.push(WriteDescriptorSet::image_view_sampler(
                7,
                self.def_texture.clone(),
                self.def_sampler.clone(),
            ));
        }
        descriptors.push(WriteDescriptorSet::buffer(8, uniform));

        let light_layout = self.pipeline.layout().set_layouts().get(1).unwrap();
        let light_desc = PersistentDescriptorSet::new(
            &desc_allocator,
            light_layout.clone(),
            [
                WriteDescriptorSet::buffer(0, light_templates),
                WriteDescriptorSet::buffer(1, lights),
                WriteDescriptorSet::buffer(2, tiles),
                WriteDescriptorSet::buffer(3, light_list),
                WriteDescriptorSet::buffer(4, bounding_line_hierarchy),
            ],
            [],
        )
        .unwrap();

        // descriptors.push(WriteDescriptorSet::buffer(14, mesh.bone_weights_counts_buf.clone()));
        let pc = Into::<[f32; 3]>::into(cam_pos);
        if let Ok(set) =
            PersistentDescriptorSet::new(&desc_allocator, layout.clone(), descriptors, [])
        {
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    set,
                )
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    1,
                    light_desc,
                )
                .unwrap()
                .bind_vertex_buffers(
                    0,
                    (
                        mesh.vertex_buffer.clone(),
                        mesh.normals_buffer.clone(),
                        mesh.uvs_buffer.clone(),
                        // mesh.bone_weights_offsets_buf.clone(),
                        // mesh.bone_weights_counts_buf.clone(),
                        // instance_buffer.clone(),
                    ),
                )
                .unwrap()
                // .bind_vertex_buffers(1, transforms_buffer.data.clone())
                .bind_index_buffer(mesh.index_buffer.clone())
                .unwrap()
                .push_constants(self.pipeline.layout().clone(), 0, pc)
                .unwrap()
                .draw_indexed_indirect(indirect_buffer)
                .unwrap();
        } else {
            println!("failed to create descriptor set");
        }
        self
    }
}
