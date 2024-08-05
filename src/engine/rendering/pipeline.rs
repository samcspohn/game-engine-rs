use std::sync::Arc;

use nalgebra_glm::Vec3;
use vulkano::{
    buffer::{allocator::SubbufferAllocator, Buffer, BufferContents, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    format::{self, Format},
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
    impl_vertex,
    memory::allocator::StandardMemoryAllocator,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::{BuffersDefinition, Vertex},
            viewport::ViewportState,
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    shader::ShaderModule,
};

use crate::engine::{
    rendering::vulkan_manager::VulkanManager,
    transform_compute::cs::{transform, MVP},
};

use self::fs::light;

use super::{
    lighting::lighting_compute::lt::{self, tile},
    model::{Mesh, Normal, _Vertex, UV},
    texture::TextureManager,
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
    _vs: Arc<ShaderModule>,
    _fs: Arc<ShaderModule>,
    pub pipeline: Arc<GraphicsPipeline>,
    // pub uniforms: SubbufferAllocator,
    pub def_texture: Arc<ImageView<ImmutableImage>>,
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
        render_pass: Arc<RenderPass>,
        _dimensions: [u32; 2],
        sub_pass_index: u32,
        vk: Arc<VulkanManager>,
        // use_msaa: bool,
    ) -> RenderPipeline {
        let subpass = Subpass::from(render_pass, sub_pass_index).unwrap();
        let vs = vs::load(vk.device.clone()).unwrap();
        let fs = fs::load(vk.device.clone()).unwrap();
        let pipeline = GraphicsPipeline::start()
            .vertex_input_state(
                BuffersDefinition::new()
                    .vertex::<_Vertex>()
                    .vertex::<Normal>()
                    .vertex::<UV>(), // .instance::<Id>(),
                                     // .vertex::<VertU32>()
                                     // .vertex::<VertU32_2>(),
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
            .multisample_state(MultisampleState {
                rasterization_samples: subpass.num_samples().unwrap(),
                ..Default::default()
            })
            .render_pass(subpass)
            .build(vk.device.clone())
            .unwrap();
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
        let _ = builder.build().unwrap().execute(vk.queue.clone()).unwrap();

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

        RenderPipeline {
            _vs: vs,
            _fs: fs,
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

    pub fn bind_pipeline(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
    ) -> &RenderPipeline {
        builder.bind_pipeline_graphics(self.pipeline.clone());

        self
    }

    pub fn bind_mesh(
        &self,
        texture_manager: &TextureManager,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
        instance_buffer: Subbuffer<[[i32; 2]]>,
        mvp_buffer: Subbuffer<[MVP]>,
        ////
        light_len: u32,
        lights: Subbuffer<[lt::light]>,
        light_templates: Subbuffer<[fs::lightTemplate]>,
        tiles: Subbuffer<[tile]>,
        screen_dims: [f32; 2],
        /////
        transforms: Subbuffer<[transform]>,
        mesh: &Mesh,
        indirect_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,
        cam_pos: Vec3,
        light_list: Subbuffer<[u32]>,
        visible_lights: Subbuffer<[u32]>,
        visible_lights_count: Subbuffer<u32>,
        skeleton: Option<&Subbuffer<[[[f32; 4]; 3]]>>,
        has_skeleton: bool,
        empty: Subbuffer<[i32]>,
        num_bones: i32,
    ) -> &RenderPipeline {
        let layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        let mut descriptors = Vec::new();

        descriptors.push(WriteDescriptorSet::buffer(0, mvp_buffer));

        if let Some(texture) = mesh.texture.as_ref() {
            let texture = texture_manager.get_id(texture).unwrap().lock();
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
        descriptors.push(WriteDescriptorSet::buffer(2, instance_buffer));
        // descriptors.push(WriteDescriptorSet::buffer(3, transforms));
        let uniform = self.vk.allocate(fs::Data { screen_dims });
        let vs_uniform = self.vk.allocate(vs::UniformBufferObject {
            has_skeleton: if has_skeleton { 1 } else { 0 },
            num_bones,
        });
        descriptors.push(WriteDescriptorSet::buffer(3, light_templates));
        descriptors.push(WriteDescriptorSet::buffer(4, lights));
        descriptors.push(WriteDescriptorSet::buffer(5, tiles));
        descriptors.push(WriteDescriptorSet::buffer(6, uniform));
        descriptors.push(WriteDescriptorSet::buffer(7, light_list));
        // descriptors.push(WriteDescriptorSet::buffer(7, mesh.bone_weight_offsets));
        if let Some(skel) = skeleton {
            descriptors.push(WriteDescriptorSet::buffer(10, skel.clone()));
        } else {
            descriptors.push(WriteDescriptorSet::buffer(10, empty.clone()));
        }
        if let Some(buf) = mesh.bone_weights_buffer.as_ref() {
            descriptors.push(WriteDescriptorSet::buffer(11, buf.clone()));
        } else {
            descriptors.push(WriteDescriptorSet::buffer(11, empty.clone()));
        }
        descriptors.push(WriteDescriptorSet::buffer(12, vs_uniform));
        descriptors.push(WriteDescriptorSet::buffer(
            13,
            mesh.bone_weights_offsets_counts_buf.clone(),
        ));
        // descriptors.push(WriteDescriptorSet::buffer(14, mesh.bone_weights_counts_buf.clone()));

        if let Ok(set) = PersistentDescriptorSet::new(&desc_allocator, layout.clone(), descriptors)
        {
            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    self.pipeline.layout().clone(),
                    0,
                    set,
                )
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
                // .bind_vertex_buffers(1, transforms_buffer.data.clone())
                .bind_index_buffer(mesh.index_buffer.clone())
                .draw_indexed_indirect(indirect_buffer)
                .unwrap();
        }
        self
    }
}
