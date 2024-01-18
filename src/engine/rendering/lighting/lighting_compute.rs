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

use self::cs::cluster;
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
    pub(crate) clusters: Mutex<Subbuffer<[cluster]>>,
    screen_dims: Mutex<[f32; 2]>,
    // pub(crate) buckets: Subbuffer<[u32]>,
    // pub(crate) buckets_count: Subbuffer<[u32]>,
    // buckets_2: Subbuffer<[u32]>,
    // light_hashes: Mutex<Subbuffer<[[u32; 8]]>>,
    // pub(crate) light_ids: Mutex<Subbuffer<[u32]>>,
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
            clusters: Mutex::new(vk.buffer_array(1, MemoryUsage::DeviceOnly)),
            draw_indirect: vk.buffer_from_iter([DrawIndirectCommand {
                vertex_count: 6,
                instance_count: 0,
                first_instance: 0,
                first_vertex: 0,
            }]),
            // buckets: vk.buffer_array(NUM_BUCKETS + 256, MemoryUsage::DeviceOnly),
            // buckets_count: vk.buffer_array(NUM_BUCKETS, MemoryUsage::DeviceOnly),
            // buckets_2: vk.buffer_array(NUM_BUCKETS, MemoryUsage::DeviceOnly),
            // light_hashes: Mutex::new(vk.buffer_array(1, MemoryUsage::DeviceOnly)),
            // light_ids: Mutex::new(vk.buffer_array(1, MemoryUsage::DeviceOnly)),
            vk: vk,
            screen_dims: Mutex::new([1., 1.]),
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
        visible_lights: Subbuffer<[u32]>,
        view: Mat4,
        proj: Mat4,
        cam_pos: Vec3,
        // cam_forw: Vec3,
        // cam_up: Vec3,
        // cam_right: Vec3,
    ) -> Arc<PersistentDescriptorSet> {
        // let visble_lights = self.visible_lights.lock();
        let uniforms = {
            let uniform_data = cs::Data {
                num_jobs: num_jobs as i32,
                stage: stage.into(),
                vp: { proj * view }.into(),
                view: view.into(),
                proj: proj.into(),
                cam_pos: cam_pos.into(), // cam_forw: cam_forw.into(),
                                         // cam_up: Padded(cam_up.into()),
                                         // cam_right: Padded(cam_right.into()),
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
                WriteDescriptorSet::buffer(5, self.clusters.lock().clone()),
                // WriteDescriptorSet::buffer(5, visible_lights.clone()),
                // WriteDescriptorSet::buffer(6, self.visible_lights_index.clone()),
                // WriteDescriptorSet::buffer(7, self.draw_indirect.clone()),
                // WriteDescriptorSet::buffer(5, self.buckets.clone()),
                // WriteDescriptorSet::buffer(6, self.buckets_count.clone()),
                // WriteDescriptorSet::buffer(7, self.buckets_2.clone()),
                // WriteDescriptorSet::buffer(8, light_ids),
                // WriteDescriptorSet::buffer(9, light_hashes.clone()),
            ],
        )
        .unwrap()
    }

    pub fn update_lights_1(
        &self,
        builder: &mut PrimaryCommandBuffer,
        deinits: Option<Subbuffer<[cs::light_deinit]>>,
        inits: Option<Subbuffer<[cs::light_init]>>,
        lights: Subbuffer<[cs::light]>,
        transforms: Subbuffer<[transform]>,
        // screen_dims: [u32; 2],
    ) {
        // let cam_forw = Vec3::new(0., 0., 0.);
        // let cam_up = Vec3::new(0., 0., 0.);
        // let cam_right = Vec3::new(0., 0., 0.);
        let mut visible_lights = self.visible_lights.lock();
        if lights.len() > visible_lights.len() {
            let buf = self
                .vk
                // .buffer_array(8_000_000, MemoryUsage::DeviceOnly);
                .buffer_array(lights.len(), MemoryUsage::DeviceOnly);
            *visible_lights = buf;
            // let buf = self
            //     .vk
            //     .buffer_array(lights.len().next_power_of_two(), MemoryUsage::DeviceOnly);
            // *self.light_hashes.lock() = buf;
        }

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
                        visible_lights.clone(),
                        Mat4::default(),
                        Mat4::default(),
                        Vec3::default(),
                    )
                } else if let Some(init) = inits {
                    self.get_descriptors(
                        num_jobs,
                        stage,
                        lights.clone(),
                        self.dummy_buffer.clone(),
                        init,
                        transforms.clone(),
                        visible_lights.clone(),
                        Mat4::default(),
                        Mat4::default(),
                        Vec3::default(),
                    )
                } else {
                    self.get_descriptors(
                        num_jobs,
                        stage,
                        lights.clone(),
                        self.dummy_buffer.clone(),
                        self.dummy_buffer.clone(),
                        transforms.clone(),
                        visible_lights.clone(),
                        Mat4::default(),
                        Mat4::default(),
                        Vec3::default(),
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
        lights: Subbuffer<[cs::light]>,
        transforms: Subbuffer<[transform]>,
        view: Mat4,
        proj: Mat4,
        cam_pos: Vec3,
        screen_dims: [f32; 2],
    ) {
        let mut visible_lights = self.visible_lights.lock();
        if lights.len() > visible_lights.len() {
            let buf = self
                .vk
                // .buffer_array(8_000_000, MemoryUsage::DeviceOnly);
                .buffer_array(lights.len(), MemoryUsage::DeviceOnly);
            *visible_lights = buf;
            // let buf = self
            //     .vk
            //     .buffer_array(lights.len().next_power_of_two(), MemoryUsage::DeviceOnly);
            // *self.light_hashes.lock() = buf;
        }
        if screen_dims != *self.screen_dims.lock() {
            let buf = self.vk.buffer_array(
                (((screen_dims[0] / 16.).ceil() + 1.) * ((screen_dims[1].abs() / 16.).ceil() + 1.)).max(1.) as u64,
                MemoryUsage::DeviceOnly,
            );
            *self.screen_dims.lock() = screen_dims;
            *self.clusters.lock() = buf;
        }

        builder.bind_pipeline_compute(self.pipeline.clone());
        // builder.fill_buffer(self.buckets_count.clone(), 0).unwrap();
        // builder.fill_buffer(visible_lights.clone(), 0).unwrap();
        builder.update_buffer(self.visible_lights_index.clone(), &0);
        // let _builder = builder;
        let mut build_stage =
            |builder: &mut utils::PrimaryCommandBuffer, num_jobs: i32, stage: i32| {
                let descriptor_set = self.get_descriptors(
                    num_jobs,
                    stage,
                    lights.clone(),
                    self.dummy_buffer.clone(),
                    self.dummy_buffer.clone(),
                    transforms.clone(),
                    visible_lights.clone(),
                    view,
                    proj,
                    cam_pos,
                );
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

        // if let Some(deinits) = deinits {
        //     build_stage(builder, deinits.len() as i32, 0, None, Some(deinits));
        // }
        // if let Some(inits) = inits {
        //     build_stage(builder, inits.len() as i32, 1, Some(inits), None);
        // }
        // calc light hash
        let cluster_len = { self.clusters.lock().len() } as i32;
        build_stage(builder, cluster_len, 2);
        build_stage(builder, lights.len() as i32, 3);
        // let screen_dims = [screen_dims[0] as f32, screen_dims[1] as f32];

        let mut uni = lt::Data {
            // num_jobs: 0,
            numThreads: [(screen_dims[0] / 16.).ceil() as u32, (screen_dims[1].abs() / 16.).ceil() as u32],
            stage: 0,
            vp: { proj * view }.into(),
            view: view.into(),
            proj: proj.into(),
            cam_pos: Padded(cam_pos.into()),
            num_lights: lights.len() as i32,
            InverseProjection: nalgebra_glm::inverse(&(proj * view)).into(),
            ScreenDimensions: screen_dims,
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
                WriteDescriptorSet::buffer(2, self.clusters.lock().clone()),
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
