use std::sync::Arc;

use vulkano::{
    pipeline::{
        compute::ComputePipelineCreateInfo, graphics::{color_blend::{ColorBlendAttachmentState, ColorBlendState}, input_assembly::InputAssemblyState, multisample::MultisampleState, rasterization::{CullMode, RasterizationState}, subpass::PipelineRenderingCreateInfo, vertex_input::{VertexBufferDescription, VertexDefinition}, viewport::ViewportState, GraphicsPipelineCreateInfo}, layout::PipelineDescriptorSetLayoutCreateInfo, ComputePipeline, DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo
    },
    shader::ShaderModule,
};

use crate::prelude::VulkanManager;

pub fn compute_pipeline(vk: Arc<VulkanManager>, shader: Arc<ShaderModule>) -> Arc<ComputePipeline> {
    let cs = shader.entry_point("main").unwrap();
    let stage = PipelineShaderStageCreateInfo::new(cs);
    let layout = PipelineLayout::new(
        vk.device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(vk.device.clone()),
    )
    .unwrap();
    ComputePipeline::new(
        vk.device.clone(),
        None,
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )
    .unwrap()
}
pub fn graphics_pipeline(
    vk: Arc<VulkanManager>,
    shaders: &[Arc<ShaderModule>],
    vertex_description: &[VertexBufferDescription],
    graphics_options: GraphicsPipelineCreateInfo,
) -> Arc<GraphicsPipeline> {
    let stages = shaders
        .iter()
        .map(|shader| PipelineShaderStageCreateInfo::new(shader))
        .collect();
    let vertex_input_state = vertex_description.definition(&shaders[0]).unwrap();
    let layout = PipelineLayout::new(
        vk.device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(vk.device.clone()),
    )
    .unwrap();
    let subpass = PipelineRenderingCreateInfo {
        color_attachment_formats: vec![Some(vk.swapchain().image_format())],
        ..Default::default()
    };
    let layout = GraphicsPipelineCreateInfo::layout(layout);
    let defualt_options = GraphicsPipelineCreateInfo {
        stages: stages.into_iter().collect(),
        vertex_input_state: Some(vertex_input_state),
        input_assembly_state: Some(InputAssemblyState::default()),
        viewport_state: Some(ViewportState::default()),
        rasterization_state: Some(RasterizationState::default().cull_mode(CullMode::Back)),
        multisample_state: Some(MultisampleState::default()),
        color_blend_state: Some(ColorBlendState::with_attachment_states(
            subpass.color_attachment_formats.len() as u32,
            ColorBlendAttachmentState::default(),
        )),
        dynamic_state: [DynamicState::Viewport].into_iter().collect(),
        subpass: Some(subpass.into()),
        ..layout
    };
    let pipeline = GraphicsPipeline::new(
        vk.device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            ..graphics_options
            ..defualt_options
        },
    )
    .unwrap();
    pipeline
}
