use std::sync::Arc;

use parking_lot::Mutex;
use vulkano::{
    buffer::{allocator::SubbufferAllocator, Subbuffer},
    command_buffer::CopyBufferInfo,
    descriptor_set::{self, DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet},
    memory::allocator::MemoryUsage,
    pipeline::{ComputePipeline, Pipeline},
};

use crate::engine::{
    prelude::utils,
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
    light_hashes: Mutex<Subbuffer<[[u32; 8]]>>,
    pub(crate) light_ids: Mutex<Subbuffer<[u32]>>,
}
const NUM_BUCKETS: u64 = 65536;
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
            buckets: vk.buffer_array(NUM_BUCKETS + 256, MemoryUsage::DeviceOnly),
            buckets_count: vk.buffer_array(NUM_BUCKETS, MemoryUsage::DeviceOnly),
            buckets_2: vk.buffer_array(NUM_BUCKETS, MemoryUsage::DeviceOnly),
            light_hashes: Mutex::new(vk.buffer_array(1, MemoryUsage::DeviceOnly)),
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
        let light_hashes = self.light_hashes.lock();
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
                // WriteDescriptorSet::buffer(9, light_hashes.clone()),
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
        if lights.len() > light_ids.len() {
            let buf = self
                .vk
                // .buffer_array(8_000_000, MemoryUsage::DeviceOnly);
                .buffer_array(lights.len(), MemoryUsage::DeviceOnly);
            *light_ids = buf;
            // let buf = self
            //     .vk
            //     .buffer_array(lights.len().next_power_of_two(), MemoryUsage::DeviceOnly);
            // *self.light_hashes.lock() = buf;
        }
        builder.bind_pipeline_compute(self.pipeline.clone());
        builder.fill_buffer(self.buckets_count.clone(), 0).unwrap();
        builder.fill_buffer(light_ids.clone(), 0).unwrap();
        // let _builder = builder;
        let mut build_stage =
            |builder: &mut utils::PrimaryCommandBuffer,
             num_jobs: i32,
             stage: i32,
             inits: Option<Subbuffer<[cs::light_init]>>,
             deinits: Option<Subbuffer<[cs::light_deinit]>>| {
                let descriptor_set = if let Some(deinit) = deinits {
                    self.get_descriptors(
                        num_jobs,
                        stage,
                        lights.clone(),
                        deinit,
                        self.dummy_buffer.clone(),
                        transforms.clone(),
                        light_ids.clone(),
                    )
                } else if let Some(init) = inits {
                    self.get_descriptors(
                        num_jobs,
                        stage,
                        lights.clone(),
                        self.dummy_buffer.clone(),
                        init,
                        transforms.clone(),
                        light_ids.clone(),
                    )
                } else {
                    self.get_descriptors(
                        num_jobs,
                        stage,
                        lights.clone(),
                        self.dummy_buffer.clone(),
                        self.dummy_buffer.clone(),
                        transforms.clone(),
                        light_ids.clone(),
                    )
                };
                builder
                    .bind_descriptor_sets(
                        self.pipeline.bind_point(),
                        self.pipeline.layout().clone(),
                        0,
                        descriptor_set,
                    )
                    .dispatch([(num_jobs as u32).div_ceil(128), 1, 1])
                    .unwrap();
            };

        if let Some(deinits) = deinits {
            build_stage(builder, deinits.len() as i32, 0, None, Some(deinits));
        }
        // build_stage();
        if let Some(inits) = inits {
            build_stage(builder, inits.len() as i32, 1, Some(inits), None);
        }
        // calc light hash
        build_stage(builder, lights.len() as i32, 2, None, None);

        // prefix sum
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                self.buckets_count.clone(),
                self.buckets.clone().slice(0..NUM_BUCKETS),
            ))
            .unwrap();
        build_stage(builder, 256, 3, None, None);
        build_stage(builder, 1, 5, None, None);
        build_stage(builder, 256, 6, None, None);
        builder.copy_buffer(CopyBufferInfo::buffers(
            self.buckets.clone().slice(0..NUM_BUCKETS),
            self.buckets_2.clone(),
        ));
        //////
        build_stage(builder, lights.len() as i32, 4, None, None);
    }
}
