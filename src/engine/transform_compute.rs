use std::{
    cell::{SyncUnsafeCell, UnsafeCell},
    ptr,
    sync::Arc,
    time::Instant,
};

use crate::engine::{
    rendering::component::buffer_usage_all,
    world::transform::{CacheVec, TransformData, POS_U, ROT_U, SCL_U},
};

use force_send_sync::SendSync;
use nalgebra_glm as glm;
use parking_lot::Mutex;
use puffin_egui::puffin;
use rayon::prelude::*;
use vulkano::{
    buffer::{allocator::SubbufferAllocator, BufferContents, BufferUsage, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CopyBufferInfo,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, DescriptorSet, PersistentDescriptorSet,
        WriteDescriptorSet,
    },
    device::Device,
    memory::allocator::{MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    DeviceSize,
};

use self::cs::{transform, Data, MVP};

use super::{perf::Perf, rendering::vulkan_manager::VulkanManager, utils, world::transform::TransformBuf};

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/transform.comp",
    }
}

pub struct TransformCompute {
    pub gpu_transforms: Subbuffer<[transform]>,
    // pub(crate) update_data_alloc: Mutex<Vec<SubbufferAllocator>>,
    // pub cycle: usize,
    pub mvp: Subbuffer<[MVP]>,
    pub(crate) vk: Arc<VulkanManager>,
    // update_count: (Option<u32>, Option<u32>, Option<u32>),
    update_data: (
        Option<(Subbuffer<[i32]>, Subbuffer<[[f32; 4]]>)>,
        Option<(Subbuffer<[i32]>, Subbuffer<[[f32; 4]]>)>,
        Option<(Subbuffer<[i32]>, Subbuffer<[[f32; 4]]>)>,
    ),
    // uniforms: Mutex<SubbufferAllocator>,
    compute: Arc<ComputePipeline>,
}

impl TransformCompute {
    pub fn new(vk: Arc<VulkanManager>) -> Self {
        let num_images = vk.images.len() as u32;

        let num_transforms = 2;
        let gpu_transforms = vk.buffer_array(
            num_transforms as vulkano::DeviceSize,
            MemoryTypeFilter::PREFER_DEVICE,
        );
        let mvp = vk.buffer_array(
            num_transforms as vulkano::DeviceSize,
            MemoryTypeFilter::PREFER_DEVICE,
        );

        TransformCompute {
            gpu_transforms,
            mvp,
            // update_data_alloc: Mutex::new(
            //     (0..2)
            //         .into_iter()
            //         .map(|_| vk.sub_buffer_allocator_with_usage(BufferUsage::STORAGE_BUFFER))
            //         .collect(),
            // ),
            // cycle: 0,
            update_data: (None, None, None),
            // uniforms: Mutex::new(vk.sub_buffer_allocator()),
            compute: {
                let cs = cs::load(vk.device.clone())
                    .unwrap()
                    .entry_point("main")
                    .unwrap();
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
            },
            vk: vk.clone(),
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    pub(crate) fn alloc_buffers(&mut self, len: u64) -> TransformBuf {
        // let alloc = self.update_data_alloc.lock();
        // let cycle = self.cycle;
        let pos_i = self.vk.allocate_unsized((len / 32 + 1));
        let pos = self.vk.allocate_unsized(len);
        let rot_i = self.vk.allocate_unsized((len / 32 + 1));
        let rot = self.vk.allocate_unsized(len);
        let scl_i = self.vk.allocate_unsized((len / 32 + 1));
        let scl = self.vk.allocate_unsized(len);
        // self.cycle = (cycle + 1) % alloc.len();
        // drop(alloc);
        (pos_i, pos, rot_i, rot, scl_i, scl)
    }

    pub(crate) fn update_data(
        &mut self,
        builder: &mut utils::PrimaryCommandBuffer,
        // image_num: u32,
        transform_data: TransformBuf,
        perf: &Perf,
    ) {
        {
            // let write_to_buffer = perf.node("write to buffer");
            // self._get_update_data(&transform_data, image_num, perf);
        }
        {
            let write_to_buffer = perf.node("transform update");
            puffin::profile_scope!("transform update compute");
            self.__update_data(builder, transform_data);
        }
    }

    //////////////////////////////////////////////////////////////////
    ///
    pub fn __update_data(
        &self,
        builder: &mut utils::PrimaryCommandBuffer,
        data: TransformBuf,
    ) {
        // stage 0
        puffin::profile_scope!("update positions");

        // if let Some(num_jobs) = update_count {
        let num_jobs = data.1.len();
        let transforms_sub_buffer = self.vk.allocate(cs::Data {
            num_jobs: num_jobs as i32,
            stage: 0.into(),
            view: Default::default(),
            proj: Default::default(),
            // _dummy0: Default::default(),
        });
        // {
        //     let uniform_data = cs::Data {
        //         num_jobs: num_jobs as i32,
        //         stage: 0.into(),
        //         view: Default::default(),
        //         proj: Default::default(),
        //         // _dummy0: Default::default(),
        //     };
        //     let ub = self.uniforms.lock().allocate_sized().unwrap();
        //     *ub.write().unwrap() = uniform_data;
        //     ub
        //     // self.uniforms.from_data(uniform_data).unwrap()
        // };

        let descriptor_set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.compute
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, transforms_sub_buffer),
                WriteDescriptorSet::buffer(1, self.gpu_transforms.clone()),
                WriteDescriptorSet::buffer(2, self.mvp.clone()),
                WriteDescriptorSet::buffer(3, data.0.clone()),
                WriteDescriptorSet::buffer(4, data.1.clone()),
                WriteDescriptorSet::buffer(5, data.2.clone()),
                WriteDescriptorSet::buffer(6, data.3.clone()),
                WriteDescriptorSet::buffer(7, data.4.clone()),
                WriteDescriptorSet::buffer(8, data.5.clone()),
                // WriteDescriptorSet::buffer(3, ids.clone()),
                // WriteDescriptorSet::buffer(4, transforms_sub_buffer),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.compute.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set,
            )
            .unwrap()
            .dispatch([num_jobs as u32 / 128 + 1, 1, 1])
            .unwrap();
        // }
    }
    //////////////////////////////////////////////////////////////////
    pub fn _update_gpu_transforms(
        &mut self,
        builder: &mut utils::PrimaryCommandBuffer,
        transform_len: usize,
    ) {
        let len = transform_len;
        // let mut max_len = ((len as f32).log2() + 1.).ceil();
        let max_len = len.next_power_of_two();

        if self.gpu_transforms.len() < len as u64 {
            let device_local_buffer = self.vk.buffer_array(
                max_len as vulkano::DeviceSize,
                MemoryTypeFilter::PREFER_DEVICE,
            );
            let copy_buffer = self.gpu_transforms.clone();

            self.gpu_transforms = device_local_buffer;
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    copy_buffer,
                    self.gpu_transforms.clone(),
                ))
                .unwrap();

            let device_local_buffer = self.vk.buffer_array(
                max_len as vulkano::DeviceSize,
                MemoryTypeFilter::PREFER_DEVICE,
            );

            let copy_buffer = self.mvp.clone();

            self.mvp = device_local_buffer;
            builder
                .copy_buffer(CopyBufferInfo::buffers(copy_buffer, self.mvp.clone()))
                .unwrap();
        }
    }
    pub fn update_mvp(
        &self,
        builder: &mut utils::PrimaryCommandBuffer,
        view: glm::Mat4,
        proj: glm::Mat4,
        // transforms_len: i32,
        data: TransformBuf,
    ) {
        puffin::profile_scope!("update mvp");
        // stage 3
        let uniform = self.vk.allocate(cs::Data {
            num_jobs: data.1.len() as i32,
            stage: 3.into(),
            view: view.into(),
            proj: proj.into(),
            // _dummy0: Default::default(),
        });
        // {
        //     let uniform_data = cs::Data {
        //         num_jobs: data.1.len() as i32,
        //         stage: 3.into(),
        //         view: view.into(),
        //         proj: proj.into(),
        //         // _dummy0: Default::default(),
        //     };
        //     let ub = self.uniforms.lock().allocate_sized().unwrap();
        //     *ub.write().unwrap() = uniform_data;
        //     ub
        // };

        let descriptor_set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.compute
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                // WriteDescriptorSet::buffer(
                //     0,
                //     // TODO: make static
                //     self.vk.buffer_from_iter(vec![0]),
                // ),
                WriteDescriptorSet::buffer(0, uniform),
                WriteDescriptorSet::buffer(1, self.gpu_transforms.clone()),
                WriteDescriptorSet::buffer(2, self.mvp.clone()),
                WriteDescriptorSet::buffer(3, data.0.clone()),
                WriteDescriptorSet::buffer(4, data.1.clone()),
                WriteDescriptorSet::buffer(5, data.2.clone()),
                WriteDescriptorSet::buffer(6, data.3.clone()),
                WriteDescriptorSet::buffer(7, data.4.clone()),
                WriteDescriptorSet::buffer(8, data.5.clone()),
                // WriteDescriptorSet::buffer(
                //     3,
                //     // TODO: make static
                //     self.vk.buffer_from_iter(vec![0]),
                // ),
            ],
            [],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.compute.clone())
            .unwrap()
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute.layout().clone(),
                0,
                descriptor_set,
            )
            .unwrap()
            .dispatch([data.1.len() as u32 / 128 + 1, 1, 1])
            .unwrap();
    }
}
