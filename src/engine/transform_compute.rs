use std::{cell::UnsafeCell, ptr, sync::Arc, time::Instant};

use crate::engine::{
    rendering::renderer_component::buffer_usage_all,
    world::transform::{CacheVec, TransformData, POS_U, ROT_U, SCL_U},
};

use nalgebra_glm as glm;
use puffin_egui::puffin;
use rayon::prelude::*;
use sync_unsafe_cell::SyncUnsafeCell;
use vulkano::{
    buffer::{
        BufferContents, BufferUsage, CpuAccessibleBuffer, CpuBufferPool, DeviceLocalBuffer,
        TypedBufferAccess,
    },
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

use self::cs::ty::{transform, Data, MVP};

use super::{perf::Perf, rendering::vulkan_manager::VulkanManager};

// #[repr(C)]
// #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
// pub struct MVP {
//     pub mvp: [[f32; 4]; 4],
// }
// impl_vertex!(MVP, mvp);

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/transform.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub struct TransformCompute {
    pub gpu_transforms: Arc<DeviceLocalBuffer<[transform]>>,
    pub position_cache: Vec<Arc<CpuAccessibleBuffer<[[f32; 3]]>>>,
    pub position_id_cache: Vec<Arc<CpuAccessibleBuffer<[i32]>>>,
    pub rotation_cache: Vec<Arc<CpuAccessibleBuffer<[[f32; 4]]>>>,
    pub rotation_id_cache: Vec<Arc<CpuAccessibleBuffer<[i32]>>>,
    pub scale_cache: Vec<Arc<CpuAccessibleBuffer<[[f32; 3]]>>>,
    pub scale_id_cache: Vec<Arc<CpuAccessibleBuffer<[i32]>>>,
    pub mvp: Arc<DeviceLocalBuffer<[MVP]>>,
    pub(crate) vk: Arc<VulkanManager>,
    update_count: (Option<u32>, Option<u32>, Option<u32>),
    uniforms: CpuBufferPool<Data>,
    compute: Arc<ComputePipeline>,
}

impl TransformCompute {
    pub fn new(vk: Arc<VulkanManager>) -> Self {
        let num_images = vk.images.len() as u32;

        let num_transforms = 2;
        let gpu_transforms = DeviceLocalBuffer::<[transform]>::array(
            &vk.mem_alloc,
            num_transforms as vulkano::DeviceSize,
            BufferUsage {
                storage_buffer: true,
                transfer_dst: true,
                transfer_src: true,
                ..Default::default()
            },
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        let mvp = DeviceLocalBuffer::<[MVP]>::array(
            &vk.mem_alloc,
            num_transforms as vulkano::DeviceSize,
            BufferUsage {
                storage_buffer: true,
                transfer_dst: true,
                transfer_src: true,
                ..Default::default()
            },
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        TransformCompute {
            gpu_transforms,
            mvp,
            position_cache: (0..num_images)
                .map(|_| {
                    let uninitialized = unsafe {
                        CpuAccessibleBuffer::<[[f32; 3]]>::uninitialized_array(
                            &vk.mem_alloc,
                            1 as DeviceSize,
                            buffer_usage_all(),
                            false,
                        )
                        .unwrap()
                    };
                    uninitialized
                })
                .collect(),
            position_id_cache: (0..num_images)
                .map(|_| {
                    let uninitialized = unsafe {
                        CpuAccessibleBuffer::<[i32]>::uninitialized_array(
                            &vk.mem_alloc,
                            1 as DeviceSize,
                            buffer_usage_all(),
                            false,
                        )
                        .unwrap()
                    };
                    uninitialized
                })
                .collect(),
            // rotation
            rotation_cache: (0..num_images)
                .map(|_| {
                    let uninitialized = unsafe {
                        CpuAccessibleBuffer::<[[f32; 4]]>::uninitialized_array(
                            &vk.mem_alloc,
                            1 as DeviceSize,
                            buffer_usage_all(),
                            false,
                        )
                        .unwrap()
                    };
                    uninitialized
                })
                .collect(),
            rotation_id_cache: (0..num_images)
                .map(|_| {
                    let uninitialized = unsafe {
                        CpuAccessibleBuffer::<[i32]>::uninitialized_array(
                            &vk.mem_alloc,
                            1 as DeviceSize,
                            buffer_usage_all(),
                            false,
                        )
                        .unwrap()
                    };
                    uninitialized
                })
                .collect(),
            // scale
            scale_cache: (0..num_images)
                .map(|_| {
                    let uninitialized = unsafe {
                        CpuAccessibleBuffer::<[[f32; 3]]>::uninitialized_array(
                            &vk.mem_alloc,
                            1 as DeviceSize,
                            buffer_usage_all(),
                            false,
                        )
                        .unwrap()
                    };
                    uninitialized
                })
                .collect(),
            scale_id_cache: (0..num_images)
                .map(|_| {
                    let uninitialized = unsafe {
                        CpuAccessibleBuffer::<[i32]>::uninitialized_array(
                            &vk.mem_alloc,
                            1 as DeviceSize,
                            buffer_usage_all(),
                            false,
                        )
                        .unwrap()
                    };
                    uninitialized
                })
                .collect(),
            update_count: (None, None, None),
            uniforms: CpuBufferPool::<cs::ty::Data>::new(
                vk.mem_alloc.clone(),
                buffer_usage_all(),
                MemoryUsage::Download,
            ),
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
        ids: &Vec<CacheVec<i32>>,
        data: &Vec<CacheVec<T>>,
        ids_buf: &mut Vec<Arc<CpuAccessibleBuffer<[i32]>>>,
        data_buf: &mut Vec<Arc<CpuAccessibleBuffer<[T]>>>,
        mem: Arc<StandardMemoryAllocator>,
        image_num: u32,
    ) -> Option<u32>
    where
        [T]: BufferContents,
    {
        let transform_ids_len_pos: u64 = ids.iter().map(|x| x.len() as u64).sum();

        if transform_ids_len_pos == 0 {
            return None;
        }

        puffin::profile_scope!("transform_ids_buffer");
        if ids_buf[image_num as usize].len() < transform_ids_len_pos.next_power_of_two()
            || data_buf[image_num as usize].len() < transform_ids_len_pos.next_power_of_two()
        {
            // let num_images = ids_buff.len();
            ids_buf[image_num as usize] = unsafe {
                CpuAccessibleBuffer::<[i32]>::uninitialized_array(
                    &mem,
                    transform_ids_len_pos.next_power_of_two() as DeviceSize,
                    buffer_usage_all(),
                    false,
                )
                .unwrap()
            };
            data_buf[image_num as usize] = unsafe {
                CpuAccessibleBuffer::<[T]>::uninitialized_array(
                    &mem,
                    transform_ids_len_pos.next_power_of_two() as DeviceSize,
                    buffer_usage_all(),
                    false,
                )
                .unwrap()
            };
        }
        unsafe {
            loop {
                let uninitialized = ids_buf[image_num as usize].clone();
                let mut offset = 0;
                let u_w = uninitialized.write();
                if let Ok(mut mapping) = u_w {
                    for i in ids {
                        let m_slice = &mut mapping[offset..offset + i.len()];
                        m_slice.copy_from_slice((*i.v.get()).as_slice());
                        offset += i.len();
                    }
                    break;
                } else {
                    ids_buf[image_num as usize] =
                        CpuAccessibleBuffer::<[i32]>::uninitialized_array(
                            &mem,
                            (transform_ids_len_pos + 1).next_power_of_two() as DeviceSize,
                            buffer_usage_all(),
                            false,
                        )
                        .unwrap();
                }
            }
        }
        // };
        // let position_updates_buffer = unsafe {
        puffin::profile_scope!("position_updates_buffer");
        unsafe {
            loop {
                let uninitialized = data_buf[image_num as usize].clone();
                let mut offset = 0;
                let u_w = uninitialized.write();
                if let Ok(mut mapping) = u_w {
                    for i in data {
                        let m_slice = &mut mapping[offset..offset + i.len()];
                        m_slice.copy_from_slice((*i.v.get()).as_slice());
                        offset += i.len();
                    }
                    break;
                } else {
                    data_buf[image_num as usize] = CpuAccessibleBuffer::<[T]>::uninitialized_array(
                        &mem,
                        transform_ids_len_pos.next_power_of_two() as DeviceSize,
                        buffer_usage_all(),
                        false,
                    )
                    .unwrap();
                }
            }
        }
        Some(transform_ids_len_pos.try_into().unwrap())
    }

    fn get_position_update_data(
        &mut self,
        transform_data: &TransformData,
        image_num: u32,
        mem: Arc<StandardMemoryAllocator>,
    ) -> Option<u32> {
        Self::__get_update_data(
            &transform_data.pos_id,
            &transform_data.pos_data,
            &mut self.position_id_cache,
            &mut self.position_cache,
            mem,
            image_num,
        )
    }
    fn get_rotation_update_data(
        &mut self,
        transform_data: &TransformData,
        image_num: u32,
        mem: Arc<StandardMemoryAllocator>,
    ) -> Option<u32> {
        Self::__get_update_data(
            &transform_data.rot_id,
            &transform_data.rot_data,
            &mut self.rotation_id_cache,
            &mut self.rotation_cache,
            mem,
            image_num,
        )
    }
    fn get_scale_update_data(
        &mut self,
        transform_data: &TransformData,
        image_num: u32,
        mem: Arc<StandardMemoryAllocator>,
    ) -> Option<u32> {
        Self::__get_update_data(
            &transform_data.scl_id,
            &transform_data.scl_data,
            &mut self.scale_id_cache,
            &mut self.scale_cache,
            mem,
            image_num,
        )
    }
    fn _get_update_data(&mut self, transform_data: &TransformData, image_num: u32) {
        self.update_count = {
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
        {
            let write_to_buffer = perf.node("write to buffer");
            self._get_update_data(&transform_data, image_num);
        }
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
        update_count: Option<u32>,
        data: Arc<CpuAccessibleBuffer<[T]>>,
        ids: Arc<CpuAccessibleBuffer<[i32]>>,
    ) where
        [T]: BufferContents,
    {
        // stage 0
        puffin::profile_scope!("update positions");

        if let Some(num_jobs) = update_count {
            let transforms_sub_buffer = {
                let uniform_data = cs::ty::Data {
                    num_jobs: num_jobs as i32,
                    stage,
                    view: Default::default(),
                    proj: Default::default(),
                    _dummy0: Default::default(),
                };
                self.uniforms.from_data(uniform_data).unwrap()
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
        }
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
            let device_local_buffer = DeviceLocalBuffer::<[transform]>::array(
                &self.vk.mem_alloc,
                max_len as vulkano::DeviceSize,
                BufferUsage {
                    storage_buffer: true,
                    transfer_dst: true,
                    transfer_src: true,
                    ..Default::default()
                },
                self.vk.device.active_queue_family_indices().iter().copied(),
            )
            .unwrap();
            let copy_buffer = self.gpu_transforms.clone();

            self.gpu_transforms = device_local_buffer;
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    copy_buffer,
                    self.gpu_transforms.clone(),
                ))
                .unwrap();

            let device_local_buffer = DeviceLocalBuffer::<[MVP]>::array(
                &self.vk.mem_alloc,
                max_len as vulkano::DeviceSize,
                BufferUsage {
                    storage_buffer: true,
                    transfer_dst: true,
                    transfer_src: true,
                    ..Default::default()
                },
                self.vk.device.active_queue_family_indices().iter().copied(),
            )
            .unwrap();

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
        self._update_data(
            builder,
            0,
            self.update_count.0,
            self.position_cache[image_num as usize].clone(),
            self.position_id_cache[image_num as usize].clone(),
        )
    }
    fn update_rotations(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        image_num: u32,
    ) {
        // stage 0
        puffin::profile_scope!("update positions");
        self._update_data(
            builder,
            1,
            self.update_count.1,
            self.rotation_cache[image_num as usize].clone(),
            self.rotation_id_cache[image_num as usize].clone(),
        )
    }
    fn update_scales(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        image_num: u32,
    ) {
        // stage 0
        puffin::profile_scope!("update positions");
        self._update_data(
            builder,
            2,
            self.update_count.2,
            self.scale_cache[image_num as usize].clone(),
            self.scale_id_cache[image_num as usize].clone(),
        )
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
            let uniform_data = cs::ty::Data {
                num_jobs: transforms_len,
                stage: 3,
                view: view.into(),
                proj: proj.into(),
                _dummy0: Default::default(),
            };
            self.uniforms.from_data(uniform_data).unwrap()
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
                    CpuAccessibleBuffer::from_iter(
                        // TODO: make static
                        &self.vk.mem_alloc,
                        buffer_usage_all(),
                        false,
                        vec![0],
                    )
                    .unwrap(),
                ),
                WriteDescriptorSet::buffer(1, self.gpu_transforms.clone()),
                WriteDescriptorSet::buffer(2, self.mvp.clone()),
                WriteDescriptorSet::buffer(
                    3,
                    CpuAccessibleBuffer::from_iter(
                        // TODO: make static
                        &self.vk.mem_alloc,
                        buffer_usage_all(),
                        false,
                        vec![0],
                    )
                    .unwrap(),
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
