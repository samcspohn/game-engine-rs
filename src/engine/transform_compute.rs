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
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    memory::allocator::{MemoryUsage, StandardMemoryAllocator},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    DeviceSize,
};

use self::cs::{transform, Data, MVP};

use super::{perf::Perf, rendering::vulkan_manager::VulkanManager};

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/transform.comp",
    }
}

pub struct TransformCompute {
    // TODO: replace cache with subbufferallocator
    pub gpu_transforms: Subbuffer<[transform]>,
    pub(crate) update_data_alloc: Mutex<SubbufferAllocator>,
    // pub position_cache: Vec<Subbuffer<[[f32; 4]]>>,
    // pub position_id_cache: Vec<Subbuffer<[i32]>>,
    // pub rotation_cache: Vec<Subbuffer<[[f32; 4]]>>,
    // pub rotation_id_cache: Vec<Subbuffer<[i32]>>,
    // pub scale_cache: Vec<Subbuffer<[[f32; 4]]>>,
    // pub scale_id_cache: Vec<Subbuffer<[i32]>>,
    pub mvp: Subbuffer<[MVP]>,
    pub(crate) vk: Arc<VulkanManager>,
    // update_count: (Option<u32>, Option<u32>, Option<u32>),
    update_data: (
        Option<(Subbuffer<[i32]>, Subbuffer<[[f32; 4]]>)>,
        Option<(Subbuffer<[i32]>, Subbuffer<[[f32; 4]]>)>,
        Option<(Subbuffer<[i32]>, Subbuffer<[[f32; 4]]>)>,
    ),
    uniforms: Mutex<SubbufferAllocator>,
    compute: Arc<ComputePipeline>,
}

impl TransformCompute {
    pub fn new(vk: Arc<VulkanManager>) -> Self {
        let num_images = vk.images.len() as u32;

        let num_transforms = 2;
        let gpu_transforms = vk.buffer_array(
            num_transforms as vulkano::DeviceSize,
            MemoryUsage::DeviceOnly,
        );
        let mvp = vk.buffer_array(
            num_transforms as vulkano::DeviceSize,
            MemoryUsage::DeviceOnly,
        );

        TransformCompute {
            gpu_transforms,
            mvp,
            update_data_alloc: Mutex::new(
                vk.sub_buffer_allocator_with_usage(BufferUsage::STORAGE_BUFFER),
            ),
            // position_cache: (0..num_images)
            //     .map(|_| vk.buffer_array(1, MemoryUsage::Upload))
            //     .collect(),
            // position_id_cache: (0..num_images)
            //     .map(|_| vk.buffer_array(1, MemoryUsage::Upload))
            //     .collect(),
            // // rotation
            // rotation_cache: (0..num_images)
            //     .map(|_| vk.buffer_array(1, MemoryUsage::Upload))
            //     .collect(),
            // rotation_id_cache: (0..num_images)
            //     .map(|_| vk.buffer_array(1, MemoryUsage::Upload))
            //     .collect(),
            // // scale
            // scale_cache: (0..num_images)
            //     .map(|_| vk.buffer_array(1, MemoryUsage::Upload))
            //     .collect(),
            // scale_id_cache: (0..num_images)
            //     .map(|_| vk.buffer_array(1, MemoryUsage::Upload))
            //     .collect(),
            update_data: (None, None, None),
            uniforms: Mutex::new(vk.sub_buffer_allocator()),
            compute: vulkano::pipeline::ComputePipeline::new(
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
            vk: vk.clone(),
        }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////
    fn __get_update_data<T: Copy + Send + Sync>(
        &mut self,
        ids: &Vec<CacheVec<i32>>,
        data: &Vec<CacheVec<T>>,
        // ids_buf: &mut Vec<Subbuffer<[i32]>>,
        // data_buf: &mut Vec<Subbuffer<[T]>>,
        vk: Arc<VulkanManager>,
        image_num: u32,
    ) -> Option<(Subbuffer<[i32]>, Subbuffer<[T]>)>
    where
        [T]: BufferContents,
    {
        let transform_ids_len_pos: u64 = ids.iter().map(|x| x.len() as u64).sum();

        if transform_ids_len_pos == 0 {
            return None;
        }

        puffin::profile_scope!("transform_ids_buffer");
        unsafe {

        //     let ids_buf: Subbuffer<[i32]> = self
        //     .update_data_alloc
        //     .get_mut()
        //     .allocate_unsized(transform_ids_len_pos)
        //     .unwrap();
        // let mut offset = 0;
        // {
        //     let mut u_w = ids_buf.write().unwrap();
        //     // if let Ok(mut mapping) = u_w {
        //     for i in ids {
        //         let m_slice = &mut u_w[offset..offset + i.len()];
        //         m_slice.copy_from_slice((*i.v.get()).as_slice());
        //         offset += i.len();
        //     }
        // }
        // let data_buf: Subbuffer<[T]> = self
        //     .update_data_alloc
        //     .get_mut()
        //     .allocate_unsized(transform_ids_len_pos)
        //     .unwrap();
        // let mut offset = 0;
        // {
        //     let mut u_w = data_buf.write().unwrap();
        //     // if let Ok(mut mapping) = u_w {
        //     for i in data {
        //         let m_slice = &mut u_w[offset..offset + i.len()];
        //         m_slice.copy_from_slice((*i.v.get()).as_slice());
        //         offset += i.len();
        //     }
        // }


            let ids_buf: Subbuffer<[i32]> = self
                .update_data_alloc
                .get_mut()
                .allocate_unsized(transform_ids_len_pos)
                .unwrap();
            let mut offset = 0;
            {
                let mut u_w = SyncUnsafeCell::new(ids_buf.write().unwrap());
                rayon::scope(|s| {
                    for i in ids {
                        let offs = offset;
                        let u_w = &u_w;
                        s.spawn(move |s| {
                            let m_slice = &mut (*u_w.get())[offs..offs + i.len()];
                            m_slice.copy_from_slice((*i.v.get()).as_slice());
                        });
                        offset += i.len();
                    }
                });
            }
            let data_buf: Subbuffer<[T]> = self
                .update_data_alloc
                .get_mut()
                .allocate_unsized(transform_ids_len_pos)
                .unwrap();
            let mut offset = 0;
            {
                let mut u_w = SyncUnsafeCell::new(data_buf.write().unwrap());
                rayon::scope(|s| {
                    for i in data {
                        let offs = offset;
                        let u_w = &u_w;
                        s.spawn(move |s| {
                            let m_slice = &mut (*u_w.get())[offs..offs + i.len()];
                            m_slice.copy_from_slice((*i.v.get()).as_slice());
                        });
                        offset += i.len();
                    }
                });
            }
            Some((ids_buf, data_buf))
        }
    }

    fn get_position_update_data(
        &mut self,
        transform_data: &TransformData,
        image_num: u32,
        mem: Arc<StandardMemoryAllocator>,
    ) -> Option<(Subbuffer<[i32]>, Subbuffer<[[f32; 4]]>)> {
        self.__get_update_data(
            &transform_data.pos_id,
            &transform_data.pos_data,
            // &mut self.position_id_cache,
            // &mut self.position_cache,
            self.vk.clone(),
            image_num,
        )
    }
    fn get_rotation_update_data(
        &mut self,
        transform_data: &TransformData,
        image_num: u32,
        mem: Arc<StandardMemoryAllocator>,
    ) -> Option<(Subbuffer<[i32]>, Subbuffer<[[f32; 4]]>)> {
        self.__get_update_data(
            &transform_data.rot_id,
            &transform_data.rot_data,
            // &mut self.rotation_id_cache,
            // &mut self.rotation_cache,
            self.vk.clone(),
            image_num,
        )
    }
    fn get_scale_update_data(
        &mut self,
        transform_data: &TransformData,
        image_num: u32,
        mem: Arc<StandardMemoryAllocator>,
    ) -> Option<(Subbuffer<[i32]>, Subbuffer<[[f32; 4]]>)> {
        self.__get_update_data(
            &transform_data.scl_id,
            &transform_data.scl_data,
            // &mut self.scale_id_cache,
            // &mut self.scale_cache,
            self.vk.clone(),
            image_num,
        )
    }
    pub(crate) fn _get_update_data(&mut self, transform_data: &TransformData, image_num: u32) {
        self.update_data = {
            puffin::profile_scope!("buffer transform data");
            let position_update_data = self.get_position_update_data(
                &transform_data,
                image_num,
                self.vk.mem_alloc.clone(),
            );

            let rotation_update_data = self.get_rotation_update_data(
                &transform_data,
                image_num,
                self.vk.mem_alloc.clone(),
            );

            let scale_update_data =
                self.get_scale_update_data(&transform_data, image_num, self.vk.mem_alloc.clone());
            (
                position_update_data,
                rotation_update_data,
                scale_update_data,
            )
        };
    }

    
    pub(crate) fn update_data(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        image_num: u32,
        transform_data: &TransformData,
        perf: &Perf,
    ) {
        // {
        //     let write_to_buffer = perf.node("write to buffer");
        //     self._get_update_data(&transform_data, image_num);
        // }
        {
            let write_to_buffer = perf.node("transform update");
            puffin::profile_scope!("transform update compute");
            self.__update_data(builder, image_num, transform_data.extent);
        }
    }

    //////////////////////////////////////////////////////////////////

    pub fn _update_data<T: Copy + Send + Sync>(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        stage: i32,
        update_count: u32,
        data: Subbuffer<[T]>,
        ids: Subbuffer<[i32]>,
    ) where
        [T]: BufferContents,
    {
        // stage 0
        puffin::profile_scope!("update positions");

        // if let Some(num_jobs) = update_count {
        let num_jobs = update_count;
        let transforms_sub_buffer = {
            let uniform_data = cs::Data {
                num_jobs: num_jobs as i32,
                stage: stage.into(),
                view: Default::default(),
                proj: Default::default(),
                // _dummy0: Default::default(),
            };
            let ub = self.uniforms.lock().allocate_sized().unwrap();
            *ub.write().unwrap() = uniform_data;
            ub
            // self.uniforms.from_data(uniform_data).unwrap()
        };

        let descriptor_set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.compute
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, data.clone()),
                WriteDescriptorSet::buffer(1, self.gpu_transforms.clone()),
                WriteDescriptorSet::buffer(2, self.mvp.clone()),
                WriteDescriptorSet::buffer(3, ids.clone()),
                WriteDescriptorSet::buffer(4, transforms_sub_buffer),
            ],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.compute.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set,
            )
            .dispatch([num_jobs as u32 / 128 + 1, 1, 1])
            .unwrap();
        // }
    }
    //////////////////////////////////////////////////////////////////
    pub fn _update_gpu_transforms(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform_len: usize,
    ) {
        let len = transform_len;
        // let mut max_len = ((len as f32).log2() + 1.).ceil();
        let max_len = len.next_power_of_two();

        if self.gpu_transforms.len() < len as u64 {
            let device_local_buffer = self
                .vk
                .buffer_array(max_len as vulkano::DeviceSize, MemoryUsage::DeviceOnly);
            let copy_buffer = self.gpu_transforms.clone();

            self.gpu_transforms = device_local_buffer;
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    copy_buffer,
                    self.gpu_transforms.clone(),
                ))
                .unwrap();

            let device_local_buffer = self
                .vk
                .buffer_array(max_len as vulkano::DeviceSize, MemoryUsage::DeviceOnly);

            let copy_buffer = self.mvp.clone();

            self.mvp = device_local_buffer;
            builder
                .copy_buffer(CopyBufferInfo::buffers(copy_buffer, self.mvp.clone()))
                .unwrap();
        }
    }

    fn update_positions(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        image_num: u32,
    ) {
        // stage 0
        puffin::profile_scope!("update positions");
        let update_data = &self.update_data.0;
        if let Some(position_update_data) = update_data {
            self._update_data(
                builder,
                0,
                position_update_data.0.len() as u32,
                position_update_data.1.clone(),
                position_update_data.0.clone(),
                // self.update_count.0,
                // self.position_cache[image_num as usize].clone(),
                // self.position_id_cache[image_num as usize].clone(),
            )
        }
    }
    fn update_rotations(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        image_num: u32,
    ) {
        // stage 1
        puffin::profile_scope!("update rotations");
        let update_data = &self.update_data.1;
        if let Some(rotation_update_data) = update_data {
            self._update_data(
                builder,
                1,
                rotation_update_data.0.len() as u32,
                rotation_update_data.1.clone(),
                rotation_update_data.0.clone(),
                // self.update_count.0,
                // self.position_cache[image_num as usize].clone(),
                // self.position_id_cache[image_num as usize].clone(),
            )
        }
    }
    fn update_scales(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        image_num: u32,
    ) {
        // stage 2
        puffin::profile_scope!("update scales");
        let update_data = &self.update_data.2;
        if let Some(scale_update_data) = update_data {
            self._update_data(
                builder,
                2,
                scale_update_data.0.len() as u32,
                scale_update_data.1.clone(),
                scale_update_data.0.clone(),
                // self.update_count.0,
                // self.position_cache[image_num as usize].clone(),
                // self.position_id_cache[image_num as usize].clone(),
            )
        }
    }

    pub fn __update_data(
        &mut self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        image_num: u32,
        transform_len: usize,
    ) {
        self._update_gpu_transforms(builder, transform_len);
        self.update_positions(builder, image_num);
        self.update_rotations(builder, image_num);
        self.update_scales(builder, image_num);
    }
    pub fn update_mvp(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        view: glm::Mat4,
        proj: glm::Mat4,
        transforms_len: i32,
    ) {
        puffin::profile_scope!("update mvp");
        // stage 3
        let transforms_sub_buffer = {
            let uniform_data = cs::Data {
                num_jobs: transforms_len,
                stage: 3.into(),
                view: view.into(),
                proj: proj.into(),
                // _dummy0: Default::default(),
            };
            let ub = self.uniforms.lock().allocate_sized().unwrap();
            *ub.write().unwrap() = uniform_data;
            ub
        };

        let descriptor_set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.compute
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(
                    0,
                    // TODO: make static
                    self.vk.buffer_from_iter(vec![0]),
                ),
                WriteDescriptorSet::buffer(1, self.gpu_transforms.clone()),
                WriteDescriptorSet::buffer(2, self.mvp.clone()),
                WriteDescriptorSet::buffer(
                    3,
                    // TODO: make static
                    self.vk.buffer_from_iter(vec![0]),
                ),
                WriteDescriptorSet::buffer(4, transforms_sub_buffer),
            ],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.compute.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute.layout().clone(),
                0,
                descriptor_set,
            )
            .dispatch([transforms_len as u32 / 128 + 1, 1, 1])
            .unwrap();
    }
}
