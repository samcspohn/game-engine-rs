use std::sync::Arc;

use crate::engine::rendering::vulkan_manager::VulkanManager;

use super::PrimaryCommandBuffer;
use vulkano::{
    buffer::Subbuffer,
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    memory::allocator::MemoryUsage,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    shader::ShaderModule,
};

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/radix_sort.comp",
    }
}

struct RadixSort {
    pipeline: Arc<ComputePipeline>,
    histogram_buffer: Subbuffer<[u32]>,
    prefix_sum_buffer: Subbuffer<[u32]>,
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
            histogram_buffer,
            prefix_sum_buffer,
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
        &self,
        vk: Arc<VulkanManager>,
        num_elements: u32,
        keys: Subbuffer<[u32]>,
        payload: Subbuffer<[u32]>,
        keys2: Subbuffer<[u32]>,
        payload2: Subbuffer<[u32]>,
        builder: &mut PrimaryCommandBuffer,
    ) {


        let get_descriptors = |push_constants: Subbuffer<cs::PushConstants>| {
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
                    WriteDescriptorSet::buffer(1, payload.clone()),
                    WriteDescriptorSet::buffer(2, keys2.clone()),
                    WriteDescriptorSet::buffer(3, payload2.clone()),
                    WriteDescriptorSet::buffer(4, self.histogram_buffer.clone()),
                    WriteDescriptorSet::buffer(5, self.prefix_sum_buffer.clone()),
                    WriteDescriptorSet::buffer(6, push_constants),
                ],
            )
            .unwrap()
        };
        for shift in (0..32).step_by(8) {
            // pass 0
            let push_constants = cs::PushConstants {
                numElements: num_elements,
                bitShift: shift,
                pass: 0,
            };
            let push_constants = vk.allocate(push_constants);
            let descriptor_set = get_descriptors(push_constants);

            builder.bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set,
            );
            let mut work_groups_x = (num_elements as f32 / 256.0).ceil() as u32;
            let mut work_groups_y = 1;
            if (work_groups_x > 65535) {
                work_groups_y = (work_groups_x as f32 / 65535.0).ceil() as u32;
                work_groups_x = 65535;
            }
            // self.vk.query(&self.performance.init_emitters, builder);
            builder.dispatch([work_groups_x, work_groups_y, 1]).unwrap();

            // pass 1 / prefix sum
            let push_constants = cs::PushConstants {
                numElements: num_elements,
                bitShift: shift,
                pass: 1,
            };
            let push_constants = vk.allocate(push_constants);
            let descriptor_set = get_descriptors(push_constants);

            builder.bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set,
            );
            builder.dispatch([1, 1, 1]);

            // pass 2
            let push_constants = cs::PushConstants {
                numElements: num_elements,
                bitShift: shift,
                pass: 2,
            };
            let push_constants = vk.allocate(push_constants);
            let descriptor_set = get_descriptors(push_constants);

            builder.bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set,
            );
            builder.dispatch([work_groups_x, work_groups_y, 1]).unwrap();

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

        // builder.bind_pipeline_compute(self.pipeline.clone());
        // builder.dispatch([buffer.len() as u32 / 1024, 1, 1]);
    }
}
