use std::{
    ops::{Div, Mul},
    sync::Arc,
};

use crate::engine::{rendering::vulkan_manager::VulkanManager, utils};

use super::PrimaryCommandBuffer;
use vulkano::{
    buffer::Subbuffer,
    command_buffer::DispatchIndirectCommand,
    descriptor_set::{DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet},
    memory::allocator::MemoryTypeFilter,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    shader::ShaderModule,
};
use vulkano::{
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        RenderingAttachmentInfo, RenderingInfo,
    },
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features,
        QueueCreateInfo, QueueFlags,
    },
    image::{view::ImageView, Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            subpass::PipelineRenderingCreateInfo,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
            GraphicsPipelineCreateInfo,
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
        DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
    },
    render_pass::{AttachmentLoadOp, AttachmentStoreOp},
    swapchain::{
        acquire_next_image, Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
    Validated, Version, VulkanError, VulkanLibrary,
};

pub mod cs1 {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/multi_radixsort_histograms.comp",
    }
}
pub mod cs2 {
    vulkano_shaders::shader! {
        ty: "compute",
        vulkan_version: "1.1",
        spirv_version: "1.5",
        path: "src/shaders/multi_radixsort.comp",
    }
}

// pub mod cs {
//     vulkano_shaders::shader! {
//         ty: "compute",
//         path: "src/shaders/radix_sort.comp",
//     }
// }

pub struct RadixSort {
    histograms_pipeline: Arc<ComputePipeline>,
    radix_pipeline: Arc<ComputePipeline>,
    histograms: Subbuffer<[u32]>,
    // work_group_offset_buffer: Subbuffer<[u32]>,
}

impl RadixSort {
    pub fn new(vk: Arc<VulkanManager>) -> Self {
        let histograms_pipeline =
            utils::pipeline::compute_pipeline(vk.clone(), cs1::load(vk.device.clone()).unwrap());

        let radix_pipeline =
            utils::pipeline::compute_pipeline(vk.clone(), cs2::load(vk.device.clone()).unwrap());

        let histogram_buffer = vk.buffer_array(256, MemoryTypeFilter::PREFER_DEVICE);
        // let prefix_sum_buffer = vk.buffer_array(256, MemoryTypeFilter::PREFER_DEVICE);
        Self {
            histograms_pipeline,
            radix_pipeline,
            histograms: histogram_buffer,
            // work_group_offset_buffer: prefix_sum_buffer,
        }
    }
    // fn get_descriptors(&self, vk: Arc<VulkanManager>) -> Arc<DescriptorSet> {
    //     DescriptorSet::new(
    //         &vk.desc_alloc,
    //         self.pipeline
    //             .layout()
    //             .set_layouts()
    //             .get(0) // 0 is the index of the descriptor set.
    //             .unwrap()
    //             .clone(),
    //         [
    //             WriteDescriptorSet::buffer(4, self.histogram_buffer.clone()),
    //             WriteDescriptorSet::buffer(5, self.prefix_sum_buffer.clone()),
    //         ],
    //     )
    //     .unwrap()
    // }
    pub fn sort(
        &mut self,
        vk: Arc<VulkanManager>,
        max_elements: u32,
        indirect: Subbuffer<[DispatchIndirectCommand]>,
        num_elements: Subbuffer<cs1::PC>,
        keys: &mut Subbuffer<[u32]>,
        payload: &mut Subbuffer<[u32]>,
        keys2: &mut Subbuffer<[u32]>,
        payload2: &mut Subbuffer<[u32]>,
        builder: &mut PrimaryCommandBuffer,
    ) {
        let num_global_counts = max_elements.div_ceil(256 * 4).next_power_of_two().mul(256);
        if num_global_counts > self.histograms.len() as u32 {
            self.histograms =
                vk.buffer_array(num_global_counts as u64, MemoryTypeFilter::PREFER_DEVICE);
        }

        for shift in (0..32).step_by(8) {
            builder.fill_buffer(self.histograms.clone(), 0);

            // histogram
            let push_constants = cs1::PushConstants {
                g_num_blocks_per_workgroup: 4,
                g_shift: shift,
            };
            let push_constants = vk.allocate(push_constants);
            let descriptor_set = PersistentDescriptorSet::new(
                &vk.desc_alloc,
                self.histograms_pipeline
                    .layout()
                    .set_layouts()
                    .get(0) // 0 is the index of the descriptor set.
                    .unwrap()
                    .clone(),
                [
                    WriteDescriptorSet::buffer(0, keys.clone()),
                    WriteDescriptorSet::buffer(1, self.histograms.clone()),
                    WriteDescriptorSet::buffer(2, push_constants.clone()),
                    WriteDescriptorSet::buffer(3, num_elements.clone()),
                ],
                [],
            )
            .unwrap();

            builder.bind_pipeline_compute(self.histograms_pipeline.clone());
            builder.bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.histograms_pipeline.layout().clone(),
                0,
                descriptor_set,
            );
            builder.dispatch_indirect(indirect.clone()).unwrap();

            // radix sort
            let descriptor_set = PersistentDescriptorSet::new(
                &vk.desc_alloc,
                self.radix_pipeline
                    .layout()
                    .set_layouts()
                    .get(0)
                    .unwrap()
                    .clone(),
                [
                    WriteDescriptorSet::buffer(0, keys.clone()),
                    WriteDescriptorSet::buffer(1, payload.clone()),
                    WriteDescriptorSet::buffer(2, keys2.clone()),
                    WriteDescriptorSet::buffer(3, payload2.clone()),
                    WriteDescriptorSet::buffer(4, self.histograms.clone()),
                    WriteDescriptorSet::buffer(5, push_constants),
                    WriteDescriptorSet::buffer(6, num_elements.clone()),
                ],
                [],
            )
            .unwrap();
            // let descriptor_set = get_descriptors(keys, payload, keys2, payload2, push_constants);
            builder.bind_pipeline_compute(self.radix_pipeline.clone());
            builder.bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.radix_pipeline.layout().clone(),
                0,
                descriptor_set,
            );
            builder.dispatch_indirect(indirect.clone()).unwrap();

            std::mem::swap(keys, keys2);
            std::mem::swap(payload, payload2);
        }
        // std::mem::swap(keys, keys2);
        // std::mem::swap(payload, payload2);

        // builder.bind_pipeline_compute(self.pipeline.clone());
        // builder.dispatch([buffer.len() as u32 / 1024, 1, 1]);
    }
}
