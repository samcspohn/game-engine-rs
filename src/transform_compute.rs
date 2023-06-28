use std::{cell::UnsafeCell, ptr, sync::Arc};

use crate::{
    renderer_component::buffer_usage_all, engine::world::transform::{POS_U, ROT_U, SCL_U, CacheVec, TransformData},
};

use nalgebra_glm as glm;
use puffin_egui::puffin;
use rayon::prelude::*;
use sync_unsafe_cell::SyncUnsafeCell;
use vulkano::{
    buffer::{
        BufferUsage, CpuAccessibleBuffer, CpuBufferPool, DeviceLocalBuffer, TypedBufferAccess, BufferContents,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CopyBufferInfo,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    memory::allocator::StandardMemoryAllocator,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    DeviceSize,
};

use self::cs::ty::{transform, Data, MVP};

// #[repr(C)]
// #[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
// pub struct MVP {
//     pub mvp: [[f32; 4]; 4],
// }
// impl_vertex!(MVP, mvp);

pub struct TransformCompute {
    pub transform: Arc<DeviceLocalBuffer<[transform]>>,
    pub position_cache: Vec<Arc<CpuAccessibleBuffer<[[f32; 3]]>>>,
    pub position_id_cache: Vec<Arc<CpuAccessibleBuffer<[i32]>>>,
    pub rotation_cache: Vec<Arc<CpuAccessibleBuffer<[[f32; 4]]>>>,
    pub rotation_id_cache: Vec<Arc<CpuAccessibleBuffer<[i32]>>>,
    pub scale_cache: Vec<Arc<CpuAccessibleBuffer<[[f32; 3]]>>>,
    pub scale_id_cache: Vec<Arc<CpuAccessibleBuffer<[i32]>>>,
    pub mvp: Arc<DeviceLocalBuffer<[MVP]>>,
}

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

impl TransformCompute {
    pub fn update(
        &mut self,
        device: Arc<Device>,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        positions_len: usize,
        mem: Arc<StandardMemoryAllocator>,
        _command_allocator: &StandardCommandBufferAllocator,
    ) {
        let len = positions_len;
        // let mut max_len = ((len as f32).log2() + 1.).ceil();
        let max_len = (len as f32 + 1.).log2().ceil();
        let max_len = 2_u32.pow(max_len as u32);

        let transform_data = self;

        if transform_data.transform.len() < len as u64 {
            let device_local_buffer = DeviceLocalBuffer::<[transform]>::array(
                &mem,
                max_len as vulkano::DeviceSize,
                BufferUsage {
                    storage_buffer: true,
                    transfer_dst: true,
                    transfer_src: true,
                    ..Default::default()
                },
                // BufferUsage::storage_buffer()
                //     | BufferUsage::vertex_buffer_transfer_destination()
                //     | BufferUsage::transfer_source(),
                device.active_queue_family_indices().iter().copied(),
            )
            .unwrap();
            let copy_buffer = transform_data.transform.clone();

            transform_data.transform = device_local_buffer;
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    copy_buffer,
                    transform_data.transform.clone(),
                ))
                .unwrap();

            let device_local_buffer = DeviceLocalBuffer::<[MVP]>::array(
                &mem,
                max_len as vulkano::DeviceSize,
                BufferUsage {
                    storage_buffer: true,
                    transfer_dst: true,
                    transfer_src: true,
                    ..Default::default()
                },
                // BufferUsage::storage_buffer()
                //     | BufferUsage::vertex_buffer_transfer_destination()
                //     | BufferUsage::transfer_source(),
                device.active_queue_family_indices().iter().copied(),
            )
            .unwrap();

            let copy_buffer = transform_data.mvp.clone();

            transform_data.mvp = device_local_buffer;
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    copy_buffer,
                    transform_data.mvp.clone(),
                ))
                .unwrap();
        }
    }
///////////////////////////////////////////////////////////////////////////////////////////////
    pub fn get_update_data<T: Copy>(
        ids: &Vec<CacheVec<i32>>, 
        data: &Vec<CacheVec<T>>,
        ids_buf: &mut Vec<Arc<CpuAccessibleBuffer<[i32]>>>,
        data_buf: &mut Vec<Arc<CpuAccessibleBuffer<[T]>>>,
        mem: Arc<StandardMemoryAllocator>,
        image_num: u32,
    ) -> Option<u32> where [T]:BufferContents {
        let transform_ids_len_pos: u64 = ids
            .iter()
            .map(|x| x.len() as u64)
            .sum();

        if transform_ids_len_pos > 0 {
            puffin::profile_scope!("transform_ids_buffer");
            if ids_buf[image_num as usize].len()
                < transform_ids_len_pos.next_power_of_two()
                || data_buf[image_num as usize].len()
                    < transform_ids_len_pos.next_power_of_two()
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
                            // let j = &i.v;
                            let j_iter = i.iter();
                            let m_iter = mapping[offset..offset + i.len()].iter_mut();
                            j_iter.zip(m_iter).for_each(|(j, m)| {
                                // for  in slice {
                                ptr::write(m, *j);
                                // }
                            });
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
                            // let j = &i.v;
                            let j_iter = i.iter();
                            let m_iter = mapping[offset..offset + i.len()].iter_mut();
                            j_iter.zip(m_iter).for_each(|(j, m)| {
                                // for  in slice {
                                ptr::write(m, *j);
                                // }
                            });
                            offset += i.len();
                        }
                        break;
                    } else {
                        data_buf[image_num as usize] =
                            CpuAccessibleBuffer::<[T]>::uninitialized_array(
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
        } else {
            None
        }
    }

    pub fn get_position_update_data(&mut self, transform_data: &TransformData, image_num: u32, mem: Arc<StandardMemoryAllocator>) -> Option<u32> {
        Self::get_update_data(&transform_data.pos_id, &transform_data.pos_data, &mut self.position_id_cache, &mut self.position_cache, mem, image_num)
    }
    pub fn get_rotation_update_data(&mut self, transform_data: &TransformData, image_num: u32, mem: Arc<StandardMemoryAllocator>) -> Option<u32> {
        Self::get_update_data(&transform_data.rot_id, &transform_data.rot_data, &mut self.rotation_id_cache, &mut self.rotation_cache, mem, image_num)
    }
    pub fn get_scale_update_data(&mut self, transform_data: &TransformData, image_num: u32, mem: Arc<StandardMemoryAllocator>) -> Option<u32> {
        Self::get_update_data(&transform_data.scl_id, &transform_data.scl_data, &mut self.scale_id_cache, &mut self.scale_cache, mem, image_num)
    }

    pub fn update_positions(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform_uniforms: &CpuBufferPool<Data>,
        compute_pipeline: Arc<ComputePipeline>,
        position_update_data: Option<u32>,
        _mem: Arc<StandardMemoryAllocator>,
        _command_allocator: &StandardCommandBufferAllocator,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
        image_num: u32,
    ) {
        // stage 0
        puffin::profile_scope!("update positions");

        if let Some(num_jobs) = position_update_data {
            let transforms_sub_buffer = {
                let uniform_data = cs::ty::Data {
                    num_jobs: num_jobs as i32,
                    stage: 0,
                    view: Default::default(),
                    proj: Default::default(),
                    _dummy0: Default::default(),
                };
                transform_uniforms.from_data(uniform_data).unwrap()
            };

            let descriptor_set = PersistentDescriptorSet::new(
                &desc_allocator,
                compute_pipeline
                    .layout()
                    .set_layouts()
                    .get(0) // 0 is the index of the descriptor set.
                    .unwrap()
                    .clone(),
                [
                    WriteDescriptorSet::buffer(0, self.position_cache[image_num as usize].clone()),
                    WriteDescriptorSet::buffer(1, self.transform.clone()),
                    WriteDescriptorSet::buffer(2, self.mvp.clone()),
                    WriteDescriptorSet::buffer(
                        3,
                        self.position_id_cache[image_num as usize].clone(),
                    ),
                    WriteDescriptorSet::buffer(4, transforms_sub_buffer),
                ],
            )
            .unwrap();

            builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    compute_pipeline.layout().clone(),
                    0, // Bind this descriptor set to index 0.
                    descriptor_set,
                )
                .dispatch([num_jobs as u32 / 128 + 1, 1, 1])
                .unwrap();
        }
    }
    pub fn update_rotations(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform_uniforms: &CpuBufferPool<Data>,
        compute_pipeline: Arc<ComputePipeline>,
        rotation_update_data: Option<u32>,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
        image_num: u32,
    ) {
        puffin::profile_scope!("update rotations");
        // stage 1
        if let Some(num_jobs) = rotation_update_data {
            let transforms_sub_buffer = {
                let uniform_data = cs::ty::Data {
                    num_jobs: num_jobs as i32,
                    stage: 1,
                    view: Default::default(),
                    proj: Default::default(),
                    _dummy0: Default::default(),
                };
                transform_uniforms.from_data(uniform_data).unwrap()
            };

            let descriptor_set = PersistentDescriptorSet::new(
                &desc_allocator,
                compute_pipeline
                    .layout()
                    .set_layouts()
                    .get(0) // 0 is the index of the descriptor set.
                    .unwrap()
                    .clone(),
                [
                    WriteDescriptorSet::buffer(0, self.rotation_cache[image_num as usize].clone()),
                    WriteDescriptorSet::buffer(1, self.transform.clone()),
                    WriteDescriptorSet::buffer(2, self.mvp.clone()),
                    WriteDescriptorSet::buffer(
                        3,
                        self.rotation_id_cache[image_num as usize].clone(),
                    ),
                    WriteDescriptorSet::buffer(4, transforms_sub_buffer),
                ],
            )
            .unwrap();

            builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    compute_pipeline.layout().clone(),
                    0, // Bind this descriptor set to index 0.
                    descriptor_set,
                )
                .dispatch([num_jobs as u32 / 128 + 1, 1, 1])
                .unwrap();
        }
    }
    pub fn update_scales(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform_uniforms: &CpuBufferPool<Data>,
        compute_pipeline: Arc<ComputePipeline>,
        scale_update_data: Option<u32>,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
        image_num: u32,
    ) {
        puffin::profile_scope!("update scales");
        // stage 2
        if let Some(num_jobs) = scale_update_data {
            let transforms_sub_buffer = {
                let uniform_data = cs::ty::Data {
                    num_jobs: num_jobs as i32,
                    stage: 2,
                    view: Default::default(),
                    proj: Default::default(),
                    _dummy0: Default::default(),
                };
                transform_uniforms.from_data(uniform_data).unwrap()
            };

            let descriptor_set = PersistentDescriptorSet::new(
                &desc_allocator,
                compute_pipeline
                    .layout()
                    .set_layouts()
                    .get(0) // 0 is the index of the descriptor set.
                    .unwrap()
                    .clone(),
                [
                    WriteDescriptorSet::buffer(0, self.scale_cache[image_num as usize].clone()),
                    WriteDescriptorSet::buffer(1, self.transform.clone()),
                    WriteDescriptorSet::buffer(2, self.mvp.clone()),
                    WriteDescriptorSet::buffer(3, self.scale_id_cache[image_num as usize].clone()),
                    WriteDescriptorSet::buffer(4, transforms_sub_buffer),
                ],
            )
            .unwrap();

            builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    compute_pipeline.layout().clone(),
                    0, // Bind this descriptor set to index 0.
                    descriptor_set,
                )
                .dispatch([num_jobs as u32 / 128 + 1, 1, 1])
                .unwrap();
        }
    }
    pub fn update_mvp(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        _device: Arc<Device>,
        view: glm::Mat4,
        proj: glm::Mat4,
        transform_uniforms: &CpuBufferPool<Data>,
        compute_pipeline: Arc<ComputePipeline>,
        transforms_len: i32,
        mem: Arc<StandardMemoryAllocator>,
        _command_allocator: &StandardCommandBufferAllocator,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
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
            transform_uniforms.from_data(uniform_data).unwrap()
        };

        let descriptor_set = PersistentDescriptorSet::new(
            &desc_allocator,
            compute_pipeline
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(
                    0,
                    CpuAccessibleBuffer::from_iter(&mem, buffer_usage_all(), false, vec![0])
                        .unwrap(),
                ),
                WriteDescriptorSet::buffer(1, self.transform.clone()),
                WriteDescriptorSet::buffer(2, self.mvp.clone()),
                WriteDescriptorSet::buffer(
                    3,
                    CpuAccessibleBuffer::from_iter(&mem, buffer_usage_all(), false, vec![0])
                        .unwrap(),
                ),
                WriteDescriptorSet::buffer(4, transforms_sub_buffer),
            ],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set,
            )
            .dispatch([transforms_len as u32 / 128 + 1, 1, 1])
            .unwrap();
    }
}

pub fn transform_buffer_init(
    device: Arc<Device>,
    // builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    // queue: Arc<Queue>,
    positions: Vec<transform>,
    mem: Arc<StandardMemoryAllocator>,
    _command_allocator: &StandardCommandBufferAllocator,
    _desc_allocator: Arc<StandardDescriptorSetAllocator>,
    num_images: u32,
) -> TransformCompute {
    // Apply scoped logic to create `DeviceLocalBuffer` initialized with vertex data.
    // let len = 2_000_000;
    let len = positions.len();
    // let mut max_len = (len as f32).log2().ceil();
    let max_len = (len as f32 + 1.).log2().ceil();
    let max_len = 2_u32.pow(max_len as u32);

    // Create a buffer array on the GPU with enough space for `PARTICLE_COUNT` number of `Vertex`.
    let device_local_buffer = DeviceLocalBuffer::<[transform]>::array(
        &mem,
        max_len as vulkano::DeviceSize,
        BufferUsage {
            storage_buffer: true,
            transfer_dst: true,
            transfer_src: true,
            ..Default::default()
        },
        device.active_queue_family_indices().iter().copied(),
    )
    .unwrap();
    let pos = device_local_buffer;

    let device_local_buffer = DeviceLocalBuffer::<[MVP]>::array(
        &mem,
        max_len as vulkano::DeviceSize,
        BufferUsage {
            storage_buffer: true,
            transfer_dst: true,
            transfer_src: true,
            ..Default::default()
        },
        device.active_queue_family_indices().iter().copied(),
    )
    .unwrap();
    let mvp = device_local_buffer;

    TransformCompute {
        transform: pos,
        mvp,
        position_cache: (0..num_images)
            .map(|_| {
                let uninitialized = unsafe {
                    CpuAccessibleBuffer::<[[f32; 3]]>::uninitialized_array(
                        &mem,
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
                        &mem,
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
                        &mem,
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
                        &mem,
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
                        &mem,
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
                        &mem,
                        1 as DeviceSize,
                        buffer_usage_all(),
                        false,
                    )
                    .unwrap()
                };
                uninitialized
            })
            .collect(),
    }
}
