use std::{ops::{Div, Mul}, sync::Arc};

use crate::engine::rendering::vulkan_manager::VulkanManager;

use super::PrimaryCommandBuffer;
use vulkano::{
    buffer::Subbuffer, command_buffer::DispatchIndirectCommand, descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet}, memory::allocator::MemoryUsage, pipeline::{ComputePipeline, Pipeline, PipelineBindPoint}, shader::ShaderModule
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
        path: "src/shaders/multi_radixsort.comp",
    }
}

pub struct RadixSort {
    pipeline: Arc<ComputePipeline>,
    global_count_buffer: Subbuffer<[u32]>,
    work_group_offset_buffer: Subbuffer<[u32]>,
}

impl RadixSort {
    pub fn new(vk: Arc<VulkanManager>) -> Self {
        let pipeline = vulkano::pipeline::ComputePipeline::new(
            vk.device.clone(),
            cs::load(vk.device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create compute shader");
        let histogram_buffer = vk.buffer_array(256, MemoryUsage::DeviceOnly);
        let prefix_sum_buffer = vk.buffer_array(256, MemoryUsage::DeviceOnly);
        Self {
            pipeline,
            global_count_buffer: histogram_buffer,
            work_group_offset_buffer: prefix_sum_buffer,
        }
    }
    // fn get_descriptors(&self, vk: Arc<VulkanManager>) -> Arc<PersistentDescriptorSet> {
    //     PersistentDescriptorSet::new(
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
        num_elements: Subbuffer<cs::PC>,
        keys: &mut Subbuffer<[u32]>,
        payload: &mut Subbuffer<[u32]>,
        keys2: &mut Subbuffer<[u32]>,
        payload2: &mut Subbuffer<[u32]>,
        builder: &mut PrimaryCommandBuffer,
    ) {

        let num_global_counts = max_elements.div_ceil(256).next_power_of_two().mul(256);
        if num_global_counts > self.global_count_buffer.len() as u32 {
            self.global_count_buffer = vk.buffer_array(num_global_counts as u64, MemoryUsage::DeviceOnly);
            self.work_group_offset_buffer = vk.buffer_array(num_global_counts.div_ceil(256) as u64, MemoryUsage::DeviceOnly);
        }

        let get_descriptors =
            |keys: &mut Subbuffer<[u32]>,
             payload: &mut Subbuffer<[u32]>,
             keys2: &mut Subbuffer<[u32]>,
             payload2: &mut Subbuffer<[u32]>,
             push_constants: Subbuffer<cs::PushConstants>| {
                PersistentDescriptorSet::new(
                    &vk.desc_alloc,
                    self.pipeline
                        .layout()
                        .set_layouts()
                        .get(0) // 0 is the index of the descriptor set.
                        .unwrap()
                        .clone(),
                    [
                        WriteDescriptorSet::buffer(0, keys.clone()),
                        WriteDescriptorSet::buffer(1, keys2.clone()),
                        WriteDescriptorSet::buffer(2, payload.clone()),
                        WriteDescriptorSet::buffer(3, payload2.clone()),
                        WriteDescriptorSet::buffer(4, self.global_count_buffer.clone()),
                        WriteDescriptorSet::buffer(5, self.work_group_offset_buffer.clone()),
                        WriteDescriptorSet::buffer(6, num_elements.clone()),
                        WriteDescriptorSet::buffer(7, push_constants),
                    ],
                )
                .unwrap()
            };
        
        builder.bind_pipeline_compute(self.pipeline.clone());
        for shift in 0..4 {
            builder.fill_buffer(self.global_count_buffer.clone(), 0);
            builder.fill_buffer(self.work_group_offset_buffer.clone(), 0);
            // pass 0
            let push_constants = cs::PushConstants {
                // bitShift: shift,
                pass: shift,
            };
            let push_constants = vk.allocate(push_constants);
            let descriptor_set = get_descriptors(keys, payload, keys2, payload2, push_constants);

            builder.bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            );
            builder.dispatch_indirect(indirect.clone()).unwrap();

            // // pass 1 / prefix sum
            // let push_constants = cs::PushConstants {
            //     // bitShift: shift,
            //     pass: 1,
            // };
            // let push_constants = vk.allocate(push_constants);
            // let descriptor_set = get_descriptors(keys, payload, keys2, payload2, push_constants);

            // builder.bind_descriptor_sets(
            //     PipelineBindPoint::Compute,
            //     self.pipeline.layout().clone(),
            //     0,
            //     descriptor_set,
            // );
            // builder.dispatch([1, 1, 1]).unwrap();

            // // pass 2
            // let push_constants = cs::PushConstants {
            //     // bitShift: shift,
            //     pass: 2,
            // };
            // let push_constants = vk.allocate(push_constants);
            // let descriptor_set = get_descriptors(keys, payload, keys2, payload2, push_constants);

            // builder.bind_descriptor_sets(
            //     PipelineBindPoint::Compute,
            //     self.pipeline.layout().clone(),
            //     0,
            //     descriptor_set,
            // );
            // builder.dispatch_indirect(indirect.clone()).unwrap();

            std::mem::swap(keys, keys2);
            std::mem::swap(payload, payload2);

            // unnecessary / swap buffers
            // // pass 3
            // let push_constants = cs::PushConstants {
            //     numElements: num_elements,
            //     bitShift: shift,
            //     pass: 3,
            // };
            // let push_constants = vk.allocate(push_constants);
            // let descriptor_set = self.get_descriptors(vk.clone());

            // builder.bind_descriptor_sets(
            //     PipelineBindPoint::Compute,
            //     self.pipeline.layout().clone(),
            //     0, // Bind this descriptor set to index 0.
            //     descriptor_set,
            // );
            // builder.dispatch([work_groups_x, work_groups_y, 1]).unwrap();
        }
        // std::mem::swap(keys, keys2);
        // std::mem::swap(payload, payload2);

        // builder.bind_pipeline_compute(self.pipeline.clone());
        // builder.dispatch([buffer.len() as u32 / 1024, 1, 1]);
    }
}
