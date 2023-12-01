use std::sync::Arc;

use parking_lot::Mutex;
use vulkano::{
    buffer::{allocator::SubbufferAllocator, Subbuffer},
    command_buffer::CopyBufferInfo,
    descriptor_set::{DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet},
    memory::allocator::MemoryUsage,
    pipeline::{ComputePipeline, Pipeline},
};

use crate::engine::{
    prelude::{utils::PrimaryCommandBuffer, VulkanManager},
    transform_compute::cs::transform,
};

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/lighting.comp",
    }
}

pub struct LightingCompute {
    pipeline: Arc<ComputePipeline>,
    uniforms: Mutex<SubbufferAllocator>,
    vk: Arc<VulkanManager>,
    dummy_buffer: Subbuffer<[u8]>,
    pub(crate) buckets: Subbuffer<[u32]>,
    pub(crate) buckets_count: Subbuffer<[u32]>,
    buckets_2: Subbuffer<[u32]>,
    pub(crate) light_ids: Mutex<Subbuffer<[u32]>>,
}
impl LightingCompute {
    pub fn new(vk: Arc<VulkanManager>) -> LightingCompute {
        Self {
            pipeline: vulkano::pipeline::ComputePipeline::new(
                vk.device.clone(),
                cs::load(vk.device.clone())
                    .unwrap()
                    .entry_point("main")
                    .unwrap(),
                &(),
                None,
                |_| {},
            )
            .expect("Failed to create compute shader"),
            uniforms: Mutex::new(vk.sub_buffer_allocator()),
            dummy_buffer: vk.buffer_array(1, MemoryUsage::DeviceOnly),
            buckets: vk.buffer_array(65536, MemoryUsage::DeviceOnly),
            buckets_count: vk.buffer_array(65536, MemoryUsage::DeviceOnly),
            buckets_2: vk.buffer_array(65536, MemoryUsage::DeviceOnly),
            light_ids: Mutex::new(vk.buffer_array(1, MemoryUsage::DeviceOnly)),
            vk: vk,
        }
    }
    fn get_descriptors<U, T>(
        &self,
        num_jobs: i32,
        stage: i32,
        lights: Subbuffer<[cs::light]>,
        deinits: Subbuffer<[U]>,
        inits: Subbuffer<[T]>,
        transforms: Subbuffer<[transform]>,
        light_ids: Subbuffer<[u32]>,
    ) -> Arc<PersistentDescriptorSet> {
        let uniforms = {
            let uniform_data = cs::Data {
                num_jobs: num_jobs as i32,
                stage: stage.into(),
                // _dummy0: Default::default(),
            };
            let ub = self.uniforms.lock().allocate_sized().unwrap();
            *ub.write().unwrap() = uniform_data;
            ub
            // self.uniforms.from_data(uniform_data).unwrap()
        };
        PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.pipeline
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, uniforms),
                WriteDescriptorSet::buffer(1, lights),
                WriteDescriptorSet::buffer(2, deinits),
                WriteDescriptorSet::buffer(3, inits),
                WriteDescriptorSet::buffer(4, transforms),
                WriteDescriptorSet::buffer(5, self.buckets.clone()),
                WriteDescriptorSet::buffer(6, self.buckets_count.clone()),
                WriteDescriptorSet::buffer(7, self.buckets_2.clone()),
                WriteDescriptorSet::buffer(8, light_ids),
            ],
        )
        .unwrap()
    }
    pub fn update_lights(
        &self,
        builder: &mut PrimaryCommandBuffer,
        deinits: Option<Subbuffer<[cs::light_deinit]>>,
        inits: Option<Subbuffer<[cs::light_init]>>,
        lights: Subbuffer<[cs::light]>,
        transforms: Subbuffer<[transform]>,
    ) {
        let mut light_ids = self.light_ids.lock();
        if lights.len() * 9 > light_ids.len() {
            let buf = self
                .vk
                // .buffer_array(8_000_000, MemoryUsage::DeviceOnly);
                .buffer_array(
                    (lights.len() * 9).next_power_of_two(),
                    MemoryUsage::DeviceOnly,
                );
            *light_ids = buf;
        }
        builder.bind_pipeline_compute(self.pipeline.clone());
        // builder.fill_buffer(self.buckets.clone(), 0).unwrap();
        builder.fill_buffer(self.buckets_count.clone(), 0).unwrap();
        // builder.fill_buffer(self.buckets_2.clone(), 0).unwrap();
        builder.fill_buffer(light_ids.clone(), 0).unwrap();
        if let Some(deinits) = deinits {
            let descriptor_set = self.get_descriptors(
                deinits.len() as i32,
                0,
                lights.clone(),
                deinits.clone(),
                self.dummy_buffer.clone(),
                transforms.clone(),
                light_ids.clone(),
            );
            builder
                .bind_descriptor_sets(
                    self.pipeline.bind_point(),
                    self.pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .dispatch([deinits.len() as u32 / 256 + 1, 1, 1])
                .unwrap();
        }
        if let Some(inits) = inits {
            let descriptor_set = self.get_descriptors(
                inits.len() as i32,
                1,
                lights.clone(),
                self.dummy_buffer.clone(),
                inits.clone(),
                transforms.clone(),
                light_ids.clone(),
            );
            builder
                .bind_descriptor_sets(
                    self.pipeline.bind_point(),
                    self.pipeline.layout().clone(),
                    0,
                    descriptor_set,
                )
                .dispatch([inits.len() as u32 / 256 + 1, 1, 1])
                .unwrap();
        }
        // calc light hash
        let descriptor_set = self.get_descriptors(
            lights.len() as i32,
            2,
            lights.clone(),
            self.dummy_buffer.clone(),
            self.dummy_buffer.clone(),
            transforms.clone(),
            light_ids.clone(),
        );
        builder
            .bind_descriptor_sets(
                self.pipeline.bind_point(),
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .dispatch([lights.len() as u32 / 256 + 1, 1, 1])
            .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.buckets_count.clone(),
                self.buckets.clone(),
            ))
            .unwrap();
        // prefix sum
        let descriptor_set = self.get_descriptors(
            128,
            3,
            lights.clone(),
            self.dummy_buffer.clone(),
            self.dummy_buffer.clone(),
            transforms.clone(),
            light_ids.clone(),
        );
        builder
            .bind_descriptor_sets(
                self.pipeline.bind_point(),
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .dispatch([1, 1, 1])
            .unwrap();
        builder.copy_buffer(CopyBufferInfo::buffers(
            self.buckets.clone(),
            self.buckets_2.clone(),
        ));
        let descriptor_set = self.get_descriptors(
            lights.len() as i32,
            4,
            lights.clone(),
            self.dummy_buffer.clone(),
            self.dummy_buffer.clone(),
            transforms.clone(),
            light_ids.clone(),
        );
        builder
            .bind_descriptor_sets(
                self.pipeline.bind_point(),
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .dispatch([lights.len() as u32 / 256 + 1, 1, 1])
            .unwrap();
    }
}
