use std::sync::Arc;

use vulkano::{
    format::Format, pipeline::{
        compute::ComputePipelineCreateInfo,
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState}, depth_stencil::{CompareOp, DepthState, DepthStencilState, DepthStencilStateFlags}, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::{CullMode, RasterizationState}, subpass::PipelineRenderingCreateInfo, vertex_input::{VertexBufferDescription, VertexDefinition}, viewport::ViewportState, GraphicsPipelineCreateInfo
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, DynamicState, GraphicsPipeline, PipelineLayout,
        PipelineShaderStageCreateInfo,
    }, render_pass::{self, RenderPass, Subpass}, shader::{EntryPoint, ShaderInterface, ShaderModule}
};

use crate::engine::{rendering::vulkan_manager::VulkanManager, transform_compute::cs::s};

pub fn compute_pipeline(vk: Arc<VulkanManager>, shader: Arc<ShaderModule>) -> Arc<ComputePipeline> {
    let cs = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let layout = PipelineLayout::new(
        vk.device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(vk.device.clone())
            .unwrap(),
    )
    .unwrap();
    ComputePipeline::new(
        vk.device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}
pub fn graphics_pipeline<D>(
    vk: Arc<VulkanManager>,
    shaders: &[EntryPoint],
    vertex_description: &[VertexBufferDescription],
    mut graphics_options: D,
    render_pass: Arc<RenderPass>,
) -> Arc<GraphicsPipeline>
where
    D: FnOnce(&mut GraphicsPipelineCreateInfo),
{
    let vertex_input_state = vertex_description
    .definition(&shaders[0].info().input_interface)
    .unwrap();
    let stages: Vec<_> = shaders
        .into_iter()
        .map(|shader| PipelineShaderStageCreateInfo::new(shader.clone()))
        .collect();
    let layout = PipelineLayout::new(
        vk.device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(vk.device.clone())
            .unwrap(),
    )
    .unwrap();
    // let color_attachment_formats = render_pass
    //     .attachments()
    //     .iter()
    //     .map(|a| Some(a.format))
    //     .collect();
    // let depth_attachment_format = 
    // let subpass = PipelineRenderingCreateInfo {
    //     color_attachment_formats: vec![Some(Format::R8G8B8A8_UNORM)],
    //     depth_attachment_format: Some(Format::D32_SFLOAT),
    //     ..Default::default()
    // };
    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();
    // let options = GraphicsPipelineCreateInfo {..graphics_options..layout};
    let mut defualt_options = GraphicsPipelineCreateInfo {
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
        depth_stencil_state: Some(DepthStencilState {
            flags: Default::default(),
            depth: Some(DepthState {
                write_enable: true,
                compare_op: CompareOp::Less,
            }),
            // depth_bounds: None,
            // stencil: None,
            ..Default::default()
        }),
        dynamic_state: [DynamicState::Viewport].into_iter().collect(),
        subpass: Some(subpass.clone().into()),
        ..GraphicsPipelineCreateInfo::layout(layout)
    };
    graphics_options(&mut defualt_options);
    let pipeline = GraphicsPipeline::new(vk.device.clone(), None, defualt_options).unwrap();
    pipeline
}

// pub fn graphics_pipeline_w_layout( vk: Arc<VulkanManager>,
//     shaders: &[EntryPoint],
//     vertex_description: &[VertexBufferDescription],
//     mut graphics_options: D,
//     render_pass: Arc<RenderPass>,
// layout) {

// }