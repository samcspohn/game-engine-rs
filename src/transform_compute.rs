use std::{ptr, sync::Arc};

use bytemuck::{Pod, Zeroable};
use nalgebra_glm as glm;
use puffin_egui::puffin;
use vulkano::{
    buffer::{
        BufferUsage, CpuAccessibleBuffer, CpuBufferPool, DeviceLocalBuffer, TypedBufferAccess,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CopyBufferInfo,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    impl_vertex,
    memory::allocator::StandardMemoryAllocator,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    DeviceSize,
};

use crate::{
    engine::transform::{POS_U, ROT_U, SCL_U},
    renderer_component2::buffer_usage_all,
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
    pub mvp: Arc<DeviceLocalBuffer<[MVP]>>,
}

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/transform.comp",
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
        command_allocator: &StandardCommandBufferAllocator,
    ) {
        let len = positions_len;
        // let mut max_len = ((len as f32).log2() + 1.).ceil();
        let max_len = (len as f32 + 1.).log2().ceil();
        let max_len = (2 as u32).pow(max_len as u32);

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

            transform_data.transform = device_local_buffer.clone();
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

            transform_data.mvp = device_local_buffer.clone();
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    copy_buffer,
                    transform_data.mvp.clone(),
                ))
                .unwrap();
        }
    }
    pub fn get_position_update_data(
        // tc: &TransformCompute,
        &self,
        device: Arc<Device>,
        transform_data: Arc<(
            usize,
            Vec<Arc<(Vec<Vec<i32>>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>)>>,
        )>,
        mem: Arc<StandardMemoryAllocator>,
        command_allocator: &StandardCommandBufferAllocator,
    ) -> Option<(
        Arc<CpuAccessibleBuffer<[i32]>>,
        Arc<CpuAccessibleBuffer<[[f32; 3]]>>,
    )> {
        let transform_ids_len_pos: u64 = transform_data
            .1
            .iter()
            .map(|x| x.0[POS_U].len() as u64)
            .sum();

        if transform_ids_len_pos > 0 {
            let transform_ids_buffer_pos = unsafe {
                puffin::profile_scope!("transform_ids_buffer");
                // let inst = Instant::now();
                let uninitialized = CpuAccessibleBuffer::<[i32]>::uninitialized_array(
                    &mem,
                    transform_ids_len_pos as DeviceSize,
                    buffer_usage_all(),
                    false,
                )
                .unwrap();
                {
                    let mut mapping = uninitialized.write().unwrap();
                    let mut offset = 0;
                    for i in &transform_data.1 {
                        let j = &i.0[POS_U];
                        let j_iter = j.iter();
                        let m_iter = mapping[offset..offset + j.len()].iter_mut();
                        j_iter.zip(m_iter).for_each(|(j, m)| {
                            // for  in slice {
                            ptr::write(m, *j);
                            // }
                        });
                        offset += j.len();
                    }
                }
                uninitialized
            };
            let position_updates_buffer = unsafe {
                puffin::profile_scope!("position_updates_buffer");
                // let inst = Instant::now();
                let uninitialized = {
                    puffin::profile_scope!("position_updates_buffer: alloc");
                    CpuAccessibleBuffer::<[[f32; 3]]>::uninitialized_array(
                        &mem,
                        transform_ids_len_pos as DeviceSize,
                        buffer_usage_all(),
                        false,
                    )
                    .unwrap()
                };
                {
                    let mut mapping = uninitialized.write().unwrap();
                    let mut offset = 0;
                    for i in &transform_data.1 {
                        let j = &i.1;
                        let j_iter = j.iter();
                        let m_iter = mapping[offset..offset + j.len()].iter_mut();
                        j_iter.zip(m_iter).for_each(|(j, m)| {
                            // for  in slice {
                            ptr::write(m, *j);
                            // }
                        });
                        offset += j.len()
                    }
                }
                uninitialized
            };
            Some((transform_ids_buffer_pos, position_updates_buffer))
        } else {
            None
        }
    }
    pub fn get_rotation_update_data(
        // tc: &TransformCompute,
        &self,
        device: Arc<Device>,
        transform_data: Arc<(
            usize,
            Vec<Arc<(Vec<Vec<i32>>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>)>>,
        )>,
        mem: Arc<StandardMemoryAllocator>,
        command_allocator: &StandardCommandBufferAllocator,
    ) -> Option<(
        Arc<CpuAccessibleBuffer<[i32]>>,
        Arc<CpuAccessibleBuffer<[[f32; 4]]>>,
    )> {
        let transform_ids_len_rot: u64 = transform_data
            .1
            .iter()
            .map(|x| x.0[ROT_U].len() as u64)
            .sum();
        if transform_ids_len_rot > 0 {
            let transform_ids_buffer_rot = unsafe {
                puffin::profile_scope!("transform_ids_buffer");
                // let inst = Instant::now();
                let uninitialized = CpuAccessibleBuffer::<[i32]>::uninitialized_array(
                    &mem,
                    transform_ids_len_rot as DeviceSize,
                    buffer_usage_all(),
                    false,
                )
                .unwrap();
                {
                    let mut mapping = uninitialized.write().unwrap();
                    let mut offset = 0;
                    for i in &transform_data.1 {
                        let j = &i.0[ROT_U];
                        let j_iter = j.iter();
                        let m_iter = mapping[offset..offset + j.len()].iter_mut();
                        j_iter.zip(m_iter).for_each(|(j, m)| {
                            // for  in slice {
                            ptr::write(m, *j);
                            // }
                        });
                        offset += j.len();
                    }
                }
                uninitialized
            };
            let rotation_updates_buffer = unsafe {
                puffin::profile_scope!("position_updates_buffer");
                // let inst = Instant::now();
                let uninitialized = {
                    puffin::profile_scope!("position_updates_buffer: alloc");
                    CpuAccessibleBuffer::<[[f32; 4]]>::uninitialized_array(
                        &mem,
                        transform_ids_len_rot as DeviceSize,
                        buffer_usage_all(),
                        false,
                    )
                    .unwrap()
                };
                {
                    let mut mapping = uninitialized.write().unwrap();
                    let mut offset = 0;
                    for i in &transform_data.1 {
                        let j = &i.2;
                        let j_iter = j.iter();
                        let m_iter = mapping[offset..offset + j.len()].iter_mut();
                        j_iter.zip(m_iter).for_each(|(j, m)| {
                            // for  in slice {
                            ptr::write(m, *j);
                            // }
                        });
                        offset += j.len()
                    }
                }
                uninitialized
            };
            Some((transform_ids_buffer_rot, rotation_updates_buffer))
        } else {
            None
        }
    }
    pub fn get_scale_update_data(
        // tc: &TransformCompute,
        &self,
        device: Arc<Device>,
        transform_data: Arc<(
            usize,
            Vec<Arc<(Vec<Vec<i32>>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>)>>,
        )>,
        mem: Arc<StandardMemoryAllocator>,
        command_allocator: &StandardCommandBufferAllocator,
    ) -> Option<(
        Arc<CpuAccessibleBuffer<[i32]>>,
        Arc<CpuAccessibleBuffer<[[f32; 3]]>>,
    )> {
        let transform_ids_len_scl: u64 = transform_data
            .1
            .iter()
            .map(|x| x.0[SCL_U].len() as u64)
            .sum();
        if transform_ids_len_scl > 0 {
            let transform_ids_buffer_scale = unsafe {
                puffin::profile_scope!("transform_ids_buffer");
                // let inst = Instant::now();
                let uninitialized = CpuAccessibleBuffer::<[i32]>::uninitialized_array(
                    &mem,
                    transform_ids_len_scl as DeviceSize,
                    buffer_usage_all(),
                    false,
                )
                .unwrap();
                {
                    let mut mapping = uninitialized.write().unwrap();
                    let mut offset = 0;
                    for i in &transform_data.1 {
                        let j = &i.0[SCL_U];
                        let j_iter = j.iter();
                        let m_iter = mapping[offset..offset + j.len()].iter_mut();
                        j_iter.zip(m_iter).for_each(|(j, m)| {
                            // for  in slice {
                            ptr::write(m, *j);
                            // }
                        });
                        offset += j.len();
                    }
                }
                uninitialized
            };

            let scale_updates_buffer = unsafe {
                puffin::profile_scope!("position_updates_buffer");

                // let inst = Instant::now();
                let uninitialized = {
                    puffin::profile_scope!("position_updates_buffer: alloc");
                    CpuAccessibleBuffer::<[[f32; 3]]>::uninitialized_array(
                        &mem,
                        transform_ids_len_scl as DeviceSize,
                        buffer_usage_all(),
                        false,
                    )
                    .unwrap()
                };
                {
                    let mut mapping = uninitialized.write().unwrap();
                    let mut offset = 0;
                    for i in &transform_data.1 {
                        let j = &i.3;
                        let j_iter = j.iter();
                        let m_iter = mapping[offset..offset + j.len()].iter_mut();
                        j_iter.zip(m_iter).for_each(|(j, m)| {
                            // for  in slice {
                            ptr::write(m, *j);
                            // }
                        });
                        offset += j.len()
                    }
                }
                uninitialized
            };
            Some((transform_ids_buffer_scale, scale_updates_buffer))
        } else {
            None
        }
    }

    pub fn update_positions(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        view: glm::Mat4,
        proj: glm::Mat4,
        transform_uniforms: &CpuBufferPool<Data>,
        compute_pipeline: Arc<ComputePipeline>,
        position_update_data: Option<(
            Arc<CpuAccessibleBuffer<[i32]>>,
            Arc<CpuAccessibleBuffer<[[f32; 3]]>>,
        )>,
        mem: Arc<StandardMemoryAllocator>,
        command_allocator: &StandardCommandBufferAllocator,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
    ) {
        // stage 0
        puffin::profile_scope!("update positions");

        if let Some((transform_ids_buffer, updates_buffer)) = position_update_data {
            let transforms_sub_buffer = {
                let uniform_data = cs::ty::Data {
                    num_jobs: transform_ids_buffer.len() as i32,
                    stage: 0,
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
                    WriteDescriptorSet::buffer(0, updates_buffer.clone()),
                    WriteDescriptorSet::buffer(1, self.transform.clone()),
                    WriteDescriptorSet::buffer(2, self.mvp.clone()),
                    WriteDescriptorSet::buffer(3, transform_ids_buffer.clone()),
                    WriteDescriptorSet::buffer(4, transforms_sub_buffer.clone()),
                ],
            )
            .unwrap();

            builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    compute_pipeline.layout().clone(),
                    0, // Bind this descriptor set to index 0.
                    descriptor_set.clone(),
                )
                .dispatch([transform_ids_buffer.len() as u32 / 128 + 1, 1, 1])
                .unwrap();
        }
    }
    pub fn update_rotations(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        view: glm::Mat4,
        proj: glm::Mat4,
        transform_uniforms: &CpuBufferPool<Data>,
        compute_pipeline: Arc<ComputePipeline>,
        rotation_update_data: Option<(
            Arc<CpuAccessibleBuffer<[i32]>>,
            Arc<CpuAccessibleBuffer<[[f32; 4]]>>,
        )>,
        mem: Arc<StandardMemoryAllocator>,
        command_allocator: &StandardCommandBufferAllocator,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
    ) {
        puffin::profile_scope!("update rotations");
        // stage 1
        if let Some((transform_ids_buffer, updates_buffer)) = rotation_update_data {
            let transforms_sub_buffer = {
                let uniform_data = cs::ty::Data {
                    num_jobs: transform_ids_buffer.len() as i32,
                    stage: 1,
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
                    WriteDescriptorSet::buffer(0, updates_buffer.clone()),
                    WriteDescriptorSet::buffer(1, self.transform.clone()),
                    WriteDescriptorSet::buffer(2, self.mvp.clone()),
                    WriteDescriptorSet::buffer(3, transform_ids_buffer.clone()),
                    WriteDescriptorSet::buffer(4, transforms_sub_buffer.clone()),
                ],
            )
            .unwrap();

            builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    compute_pipeline.layout().clone(),
                    0, // Bind this descriptor set to index 0.
                    descriptor_set.clone(),
                )
                .dispatch([transform_ids_buffer.len() as u32 / 128 + 1, 1, 1])
                .unwrap();
        }
    }
    pub fn update_scales(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        view: glm::Mat4,
        proj: glm::Mat4,
        transform_uniforms: &CpuBufferPool<Data>,
        compute_pipeline: Arc<ComputePipeline>,
        scale_update_data: Option<(
            Arc<CpuAccessibleBuffer<[i32]>>,
            Arc<CpuAccessibleBuffer<[[f32; 3]]>>,
        )>,
        mem: Arc<StandardMemoryAllocator>,
        command_allocator: &StandardCommandBufferAllocator,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
    ) {
        puffin::profile_scope!("update scales");
        // stage 2
        if let Some((transform_ids_buffer, updates_buffer)) = scale_update_data {
            let transforms_sub_buffer = {
                let uniform_data = cs::ty::Data {
                    num_jobs: transform_ids_buffer.len() as i32,
                    stage: 2,
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
                    WriteDescriptorSet::buffer(0, updates_buffer.clone()),
                    WriteDescriptorSet::buffer(1, self.transform.clone()),
                    WriteDescriptorSet::buffer(2, self.mvp.clone()),
                    WriteDescriptorSet::buffer(3, transform_ids_buffer.clone()),
                    WriteDescriptorSet::buffer(4, transforms_sub_buffer.clone()),
                ],
            )
            .unwrap();

            builder
                .bind_pipeline_compute(compute_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    compute_pipeline.layout().clone(),
                    0, // Bind this descriptor set to index 0.
                    descriptor_set.clone(),
                )
                .dispatch([transform_ids_buffer.len() as u32 / 128 + 1, 1, 1])
                .unwrap();
        }
    }
    pub fn update_mvp(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        device: Arc<Device>,
        view: glm::Mat4,
        proj: glm::Mat4,
        transform_uniforms: &CpuBufferPool<Data>,
        compute_pipeline: Arc<ComputePipeline>,
        transforms_len: i32,
        mem: Arc<StandardMemoryAllocator>,
        command_allocator: &StandardCommandBufferAllocator,
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
                WriteDescriptorSet::buffer(4, transforms_sub_buffer.clone()),
            ],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                compute_pipeline.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set.clone(),
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
    command_allocator: &StandardCommandBufferAllocator,
    desc_allocator: Arc<StandardDescriptorSetAllocator>,
) -> TransformCompute {
    // Apply scoped logic to create `DeviceLocalBuffer` initialized with vertex data.
    // let len = 2_000_000;
    let len = positions.len();
    // let mut max_len = (len as f32).log2().ceil();
    let max_len = (len as f32 + 1.).log2().ceil();
    let max_len = (2 as u32).pow(max_len as u32);

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
        mvp: mvp,
    }
}
