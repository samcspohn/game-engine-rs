use std::sync::Arc;

use vulkano::{
    buffer::{BufferContents, BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer},
    device::{Device, Queue},
    sync::GpuFuture,
};

use crate::ModelMat;

pub struct GPUVector<T>
where
    [T]: BufferContents,
{
    pub data: Arc<DeviceLocalBuffer<[T]>>,
    pub len: u32,
    pub max_len: u32,
}

pub fn transform_buffer_init(
    device: Arc<Device>,
    queue: Arc<Queue>,
    positions: Vec<ModelMat>,
) -> GPUVector<ModelMat> {
    // Apply scoped logic to create `DeviceLocalBuffer` initialized with vertex data.
    // let len = 2_000_000;
    let len = positions.len();
    // let mut max_len = (len as f32).log2().ceil();
    let max_len = (len as f32 + 1.).log2().ceil();
    let max_len = (2 as u32).pow(max_len as u32);

    // Create a buffer array on the GPU with enough space for `PARTICLE_COUNT` number of `Vertex`.
    let device_local_buffer = DeviceLocalBuffer::<[ModelMat]>::array(
        device.clone(),
        max_len as vulkano::DeviceSize,
        BufferUsage::storage_buffer()
            | BufferUsage::vertex_buffer_transfer_destination()
            | BufferUsage::transfer_source(), // Specify use as a storage buffer, vertex buffer, and transfer destination.
        device.active_queue_families(),
    )
    .unwrap();

    GPUVector {
        data: device_local_buffer,
        len: len as u32,
        max_len: max_len as u32,
    }
}

pub fn transform_buffer(
    device: Arc<Device>,
    queue: Arc<Queue>,
    transforms_buffer: &mut GPUVector<ModelMat>,
    positions: Vec<ModelMat>,
) {
    // Apply scoped logic to create `DeviceLocalBuffer` initialized with vertex data.
    let len = positions.len();
    // let mut max_len = ((len as f32).log2() + 1.).ceil();
    let max_len = (len as f32 + 1.).log2().ceil();
    let max_len = (2 as u32).pow(max_len as u32);

    if transforms_buffer.max_len < len as u32 {
        // Create a buffer array on the GPU with enough space for `PARTICLE_COUNT` number of `Vertex`.
        let device_local_buffer = DeviceLocalBuffer::<[ModelMat]>::array(
            device.clone(),
            max_len as vulkano::DeviceSize,
            BufferUsage::storage_buffer()
                | BufferUsage::vertex_buffer_transfer_destination()
                | BufferUsage::transfer_source(), // Specify use as a storage buffer, vertex buffer, and transfer destination.
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
        // cbb.copy_buffer(transforms_buffer.data.clone(), device_local_buffer.clone())
        //     .unwrap();
        // let cb = cbb.build().unwrap();

        // // Execute copy and wait for copy to complete before proceeding.
        // cb.execute(queue.clone())
        //     .unwrap()
        //     .then_signal_fence_and_flush()
        //     .unwrap()
        //     .wait(None /* timeout */)
        //     .unwrap();

        transforms_buffer.data = device_local_buffer;
        transforms_buffer.len = len as u32;
        transforms_buffer.max_len = max_len as u32;
    }
}
