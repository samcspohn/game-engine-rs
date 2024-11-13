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
        spirv_version: "1.5",
        path: "shaders/multi_radixsort_histograms.comp",
    }
}
pub mod cs2 {
    vulkano_shaders::shader! {
        ty: "compute",
        spirv_version: "1.5",
        path: "shaders/multi_radixsort.comp",
    }
}

// pub mod cs {
//     vulkano_shaders::shader! {
//         ty: "compute",
//         path: "shaders/radix_sort.comp",
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
        pc: Subbuffer<cs1::PC>,
        elements_in: &mut Subbuffer<[u32]>,
        payloads_in: &mut Subbuffer<[u32]>,
        elements_out: &mut Subbuffer<[u32]>,
        payloads_out: &mut Subbuffer<[u32]>,
        builder: &mut PrimaryCommandBuffer,
    ) {
        const WORKGROUP_SIZE: u32 = 256;
        const NUM_BLOCKS_PER_WORKGROUP: u32 = 32;
        const NUM_BUCKETS: u32 = 256;

        // const NUM_BLOCKS: u32 = 256;

        let global_invocation_size = max_elements.mul(4).div_ceil(NUM_BLOCKS_PER_WORKGROUP);
        let num_workgroups = global_invocation_size.div_ceil(WORKGROUP_SIZE);
        let histogram_size = num_workgroups.mul(NUM_BUCKETS);

        // println!("max_elements: {}", max_elements);
        // println!("global_invocation_size: {}", global_invocation_size);
        // println!("num_workgroups: {}", num_workgroups);
        // println!("histogram_size: {}", histogram_size);

        // let indirect: Subbuffer<[DispatchIndirectCommand]> =
        //     vk.buffer_from_iter([DispatchIndirectCommand {
        //         x: num_workgroups,
        //         y: 1,
        //         z: 1,
        //     }]);
        // let pc = vk.buffer_from_data(cs2::PC {
        //     g_num_elements: max_elements * 4,
        //     g_num_workgroups: num_workgroups,
        // });
        if histogram_size > self.histograms.len() as u32 {
            self.histograms = vk.buffer_array(histogram_size as u64, MemoryTypeFilter::PREFER_DEVICE);
        }

        let hist_layout = self
            .histograms_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();
        let radix_layout = self.radix_pipeline.layout().set_layouts().get(1).unwrap();

        for shift in (0..32).step_by(8) {
            // builder.fill_buffer(self.histograms.clone(), 0);

            // histogram
            let hist_push = cs1::PushConstants {
                g_shift: shift,
                g_num_blocks_per_workgroup: NUM_BLOCKS_PER_WORKGROUP,
            };
            let radix_push = cs2::PushConstants {
                g_shift: shift,
                g_num_blocks_per_workgroup: NUM_BLOCKS_PER_WORKGROUP,
            };
            // let push_constants = vk.allocate(push_constants);
            let hist_set = PersistentDescriptorSet::new(
                &vk.desc_alloc,
                hist_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, elements_in.clone()),
                    WriteDescriptorSet::buffer(1, self.histograms.clone()),
                    WriteDescriptorSet::buffer(5, pc.clone()),
                ],
                [],
            )
            .unwrap();
            // radix sort
            let radix_set = PersistentDescriptorSet::new(
                &vk.desc_alloc,
                radix_layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, elements_in.clone()),
                    WriteDescriptorSet::buffer(1, elements_out.clone()),
                    WriteDescriptorSet::buffer(3, payloads_in.clone()),
                    WriteDescriptorSet::buffer(4, payloads_out.clone()),
                    WriteDescriptorSet::buffer(2, self.histograms.clone()),
                    WriteDescriptorSet::buffer(5, pc.clone()),
                ],
                [],
            )
            .unwrap();

            builder
                .bind_pipeline_compute(self.histograms_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.histograms_pipeline.layout().clone(),
                    0,
                    hist_set,
                )
                .unwrap()
                .push_constants(self.histograms_pipeline.layout().clone(), 0, hist_push)
                .unwrap()
                .dispatch_indirect(indirect.clone())
                .unwrap()
                .bind_pipeline_compute(self.radix_pipeline.clone())
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.radix_pipeline.layout().clone(),
                    1,
                    radix_set,
                )
                .unwrap()
                .push_constants(self.radix_pipeline.layout().clone(), 0, radix_push)
                .unwrap()
                .dispatch_indirect(indirect.clone())
                .unwrap();

            if shift < 24 {
                std::mem::swap(elements_in, elements_out);
                std::mem::swap(payloads_in, payloads_out);
            }
        }

        // builder.bind_pipeline_compute(self.pipeline.clone());
        // builder.dispatch([buffer.len() as u32 / 1024, 1, 1]);
    }
}
