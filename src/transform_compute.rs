use std::sync::Arc;

use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer, BufferContents},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer},
    device::{Device, Queue},
    sync::GpuFuture,
};

use crate::ModelMat;

pub struct GPUVector<T> where [T]: BufferContents {
     
    pub data: Arc<DeviceLocalBuffer<[T]>>,
    pub len: u32
}

pub fn transform_buffer_init(
    device: Arc<Device>,
    queue: Arc<Queue>,
    positions: Vec<ModelMat>,
) -> GPUVector<ModelMat> {
    // Apply scoped logic to create `DeviceLocalBuffer` initialized with vertex data.
    let len = 2_000_000;
    // let len = positions.len();


    let buffer = {
        // // Initialize vertex data as an iterator.
        // let vertices = (0..PARTICLE_COUNT).map(|i| {
        //     let f = i as f32 / (PARTICLE_COUNT / 10) as f32;
        //     Vertex {
        //         pos: [2. * f.fract() - 1., 0.2 * f.floor() - 1.],
        //         vel: [0.; 2],
        //     }
        // });

        // Create a CPU accessible buffer initialized with the vertex data.
        // let temporary_accessible_buffer = CpuAccessibleBuffer::from_iter(
        //     device.clone(),
        //     BufferUsage::transfer_source(), // Specify this buffer will be used as a transfer source.
        //     false,
        //     positions,
        // )
        // .unwrap();

        // Create a buffer array on the GPU with enough space for `PARTICLE_COUNT` number of `Vertex`.
        let device_local_buffer = DeviceLocalBuffer::<[ModelMat]>::array(
            device.clone(),
            len as vulkano::DeviceSize,
            BufferUsage::storage_buffer() | BufferUsage::vertex_buffer_transfer_destination(), // Specify use as a storage buffer, vertex buffer, and transfer destination.
            device.active_queue_families(),
        )
        .unwrap();

        // // Create one-time command to copy between the buffers.
        // let mut cbb = AutoCommandBufferBuilder::primary(
        //     device.clone(),
        //     queue.family(),
        //     CommandBufferUsage::OneTimeSubmit,
        // )
        // .unwrap();
        // cbb.copy_buffer(temporary_accessible_buffer, device_local_buffer.clone())
        //     .unwrap();
        // let cb = cbb.build().unwrap();

        // // Execute copy and wait for copy to complete before proceeding.
        // cb.execute(queue.clone())
        //     .unwrap()
        //     .then_signal_fence_and_flush()
        //     .unwrap()
        //     .wait(None /* timeout */)
        //     .unwrap();

        device_local_buffer
    };
    GPUVector {
        data: buffer,
        len: len as u32
    }
}


pub fn transform_buffer(
    device: Arc<Device>,
    queue: Arc<Queue>,
    transforms_buffer:  &mut GPUVector<ModelMat>,
    positions: Vec<ModelMat>,
) {
    // Apply scoped logic to create `DeviceLocalBuffer` initialized with vertex data.
    let len = positions.len();

    // let buffer = {
        // // Initialize vertex data as an iterator.
        // let vertices = (0..PARTICLE_COUNT).map(|i| {
        //     let f = i as f32 / (PARTICLE_COUNT / 10) as f32;
        //     Vertex {
        //         pos: [2. * f.fract() - 1., 0.2 * f.floor() - 1.],
        //         vel: [0.; 2],
        //     }
        // });

        // Create a CPU accessible buffer initialized with the vertex data.
        let temporary_accessible_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::transfer_source(), // Specify this buffer will be used as a transfer source.
            false,
            positions,
        )
        .unwrap();

        if transforms_buffer.len < len as u32 {

            // Create a buffer array on the GPU with enough space for `PARTICLE_COUNT` number of `Vertex`.
            let device_local_buffer = DeviceLocalBuffer::<[ModelMat]>::array(
                device.clone(),
                len as vulkano::DeviceSize,
                BufferUsage::storage_buffer() | BufferUsage::vertex_buffer_transfer_destination(), // Specify use as a storage buffer, vertex buffer, and transfer destination.
                device.active_queue_families(),
            )
            .unwrap();
            transforms_buffer.data = device_local_buffer;
            transforms_buffer.len = len as u32;
        }

        // Create one-time command to copy between the buffers.
        let mut cbb = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        cbb.copy_buffer(temporary_accessible_buffer, transforms_buffer.data.clone())
            .unwrap();
        let cb = cbb.build().unwrap();

        // Execute copy and wait for copy to complete before proceeding.
        cb.execute(queue.clone())
            .unwrap()
            .then_signal_fence_and_flush()
            .unwrap()
            .wait(None /* timeout */)
            .unwrap();

        // transforms_buffer
    // };
    // buffer
}
