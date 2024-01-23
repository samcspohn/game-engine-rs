use std::sync::Arc;

use nalgebra_glm::{Mat4, Vec3};
use parking_lot::Mutex;
use rapier3d::na::ComplexField;
use vulkano::{
    buffer::{allocator::SubbufferAllocator, Subbuffer},
    command_buffer::{CopyBufferInfo, DrawIndirectCommand},
    descriptor_set::{self, DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet},
    memory::allocator::MemoryUsage,
    padded::Padded,
    pipeline::{ComputePipeline, Pipeline},
};

use crate::engine::{
    prelude::utils, prelude::{utils::PrimaryCommandBuffer, VulkanManager}, rendering::pipeline::fs, transform_compute::cs::transform
};
pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/lighting.comp",
    }
}
pub mod lt {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/light_tile.comp",
    }
}

pub struct LightingCompute {
    pipeline: Arc<ComputePipeline>,
    pipeline2: Arc<ComputePipeline>,
    uniforms: Mutex<SubbufferAllocator>,
    vk: Arc<VulkanManager>,
    dummy_buffer: Subbuffer<[u8]>,
    pub(crate) visible_lights: Mutex<Subbuffer<[u32]>>,
    pub(crate) visible_lights_index: Subbuffer<u32>,
    pub(crate) draw_indirect: Subbuffer<[DrawIndirectCommand]>,
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
            pipeline2: vulkano::pipeline::ComputePipeline::new(
                vk.device.clone(),
                lt::load(vk.device.clone())
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
            visible_lights: Mutex::new(vk.buffer_array(1, MemoryUsage::DeviceOnly)),
            visible_lights_index: vk.buffer_from_data(0),
            // tiles: Mutex::new(vk.buffer_array(1, MemoryUsage::DeviceOnly)),
            draw_indirect: vk.buffer_from_iter([DrawIndirectCommand {
                vertex_count: 6,
                instance_count: 0,
                first_instance: 0,
                first_vertex: 0,
            }]),
            vk: vk,
        }
    }
    fn get_descriptors<U, T>(
        &self,
        num_jobs: i32,
        stage: i32,
        lights: Subbuffer<[lt::light]>,
        deinits: Subbuffer<[U]>,
        inits: Subbuffer<[T]>,
        transforms: Subbuffer<[transform]>,
        light_templates: Subbuffer<[fs::lightTemplate]>,
    ) -> Arc<PersistentDescriptorSet> {
        // let visble_lights = self.visible_lights.lock();
        let uniforms = {
            let uniform_data = cs::Data {
                num_jobs: num_jobs as i32,
                stage: stage.into(),
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
                WriteDescriptorSet::buffer(6, light_templates.clone()),
            ],
        )
        .unwrap()
    }

    pub fn update_lights_1(
        &self,
        builder: &mut PrimaryCommandBuffer,
        deinits: Option<Subbuffer<[cs::light_deinit]>>,
        inits: Option<Subbuffer<[cs::light_init]>>,
        lights: Subbuffer<[lt::light]>,
        transforms: Subbuffer<[transform]>,
        light_templates: Subbuffer<[fs::lightTemplate]>,
        // screen_dims: [u32; 2],
    ) {
        builder.bind_pipeline_compute(self.pipeline.clone());
        // builder.fill_buffer(self.buckets_count.clone(), 0).unwrap();
        // builder.fill_buffer(visible_lights.clone(), 0).unwrap();
        builder.update_buffer(self.visible_lights_index.clone(), &0);
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
                        light_templates.clone(),
                    )
                } else if let Some(init) = inits {
                    self.get_descriptors(
                        num_jobs,
                        stage,
                        lights.clone(),
                        self.dummy_buffer.clone(),
                        init,
                        transforms.clone(),
                        light_templates.clone(),
                    )
                } else {
                    self.get_descriptors(
                        num_jobs,
                        stage,
                        lights.clone(),
                        self.dummy_buffer.clone(),
                        self.dummy_buffer.clone(),
                        transforms.clone(),
                        light_templates.clone(),
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
        if let Some(inits) = inits {
            build_stage(builder, inits.len() as i32, 1, Some(inits), None);
        }
        build_stage(builder, lights.len() as i32, 2, None, None);
        // // prefix sum
        // builder
        //     .copy_buffer(CopyBufferInfo::buffers(
        //         self.buckets_count.clone(),
        //         self.buckets.clone().slice(0..NUM_BUCKETS),
        //     ))
        //     .unwrap();
        // build_stage(builder, 256, 3, None, None);
        // build_stage(builder, 1, 5, None, None);
        // build_stage(builder, 256, 6, None, None);
        // builder.copy_buffer(CopyBufferInfo::buffers(
        //     self.buckets.clone().slice(0..NUM_BUCKETS),
        //     self.buckets_2.clone(),
        // ));
        // //////
        // build_stage(builder, lights.len() as i32, 4, None, None);
    }

    pub fn update_lights_2(
        &self,
        builder: &mut PrimaryCommandBuffer,
        lights: Subbuffer<[lt::light]>,
        view: Mat4,
        proj: Mat4,
        cam_pos: Vec3,
        screen_dims: [f32; 2],
        tiles: &Mutex<Subbuffer<[lt::tile]>>
    ) {
        let tiles_curr_len = { tiles.lock().len() };
        let tiles_should_be_len = (((screen_dims[0] / 16.).ceil() + 1.) * ((screen_dims[1].abs() / 16.).ceil() + 1.)).max(1.) as u64;
        if tiles_curr_len != tiles_should_be_len {
            let buf = self.vk.buffer_array(
                tiles_should_be_len,
                MemoryUsage::DeviceOnly,
            );
            *tiles.lock() = buf;
        }

        let mut uni = lt::Data {
            // num_jobs: 0,
            numThreads: [(screen_dims[0] / 16.).ceil() as u32, (screen_dims[1].abs() / 16.).ceil() as u32],
            vp: { proj * view }.into(),
            cam_pos: cam_pos.into(),
            num_lights: lights.len() as i32,
            screen_dims,
        };
        builder.bind_pipeline_compute(self.pipeline2.clone());
        let uniforms = {
            let ub = self.uniforms.lock().allocate_sized().unwrap();
            *ub.write().unwrap() = uni;
            ub
            // self.uniforms.from_data(uniform_data).unwrap()
        };
        let set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.pipeline2
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, uniforms),
                WriteDescriptorSet::buffer(1, lights.clone()),
                // WriteDescriptorSet::buffer(4, transforms),
                WriteDescriptorSet::buffer(2, tiles.lock().clone()),
            ],
        )
        .unwrap();
        builder
            .bind_descriptor_sets(
                self.pipeline2.bind_point(),
                self.pipeline2.layout().clone(),
                0,
                set,
            )
            .dispatch([
                ((screen_dims[0] / 16.).ceil() as u32),
                ((screen_dims[1].abs() / 16.).ceil() as u32),
                1,
            ])
            .unwrap();
                
        // // prefix sum
        // builder
        //     .copy_buffer(CopyBufferInfo::buffers(
        //         self.buckets_count.clone(),
        //         self.buckets.clone().slice(0..NUM_BUCKETS),
        //     ))
        //     .unwrap();
        // build_stage(builder, 256, 3, None, None);
        // build_stage(builder, 1, 5, None, None);
        // build_stage(builder, 256, 6, None, None);
        // builder.copy_buffer(CopyBufferInfo::buffers(
        //     self.buckets.clone().slice(0..NUM_BUCKETS),
        //     self.buckets_2.clone(),
        // ));
        // //////
        // build_stage(builder, lights.len() as i32, 4, None, None);
    }
}
