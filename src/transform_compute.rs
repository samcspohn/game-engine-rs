use std::sync::Arc;

use bytemuck::{Zeroable, Pod};
use vulkano::{
    buffer::{BufferContents, BufferUsage, DeviceLocalBuffer},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, PrimaryCommandBuffer, PrimaryAutoCommandBuffer},
    device::{Device, Queue},
    sync::GpuFuture, impl_vertex,
};

use crate::ModelMat;


#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct MVP {
    pub mvp: [[f32; 4];4],
}
impl_vertex!(MVP, mvp);

pub struct TransformData {
    pub positions: GPUVector<ModelMat>,
    pub mvp: GPUVector<MVP>,
}


impl TransformData {
    pub fn update(&mut self, 
        device: Arc<Device>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
        positions_len: usize) {
              // Apply scoped logic to create `DeviceLocalBuffer` initialized with vertex data.
    let len = positions_len;
    // let mut max_len = ((len as f32).log2() + 1.).ceil();
    let max_len = (len as f32 + 1.).log2().ceil();
    let max_len = (2 as u32).pow(max_len as u32);

    let transform_data = self;

    if transform_data.positions.max_len < len as u32 {
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
        transform_data.positions.len = len as u32;
        transform_data.positions.max_len = max_len as u32;

        transform_data.mvp.len = len as u32;
        transform_data.mvp.max_len = max_len as u32;

        let copy_buffer = transform_data.positions.data.clone();

        transform_data.positions.data = device_local_buffer.clone();
        builder
        .copy_buffer(copy_buffer, transform_data.positions.data.clone())
        .unwrap();

        let device_local_buffer = DeviceLocalBuffer::<[MVP]>::array(
            device.clone(),
            max_len as vulkano::DeviceSize,
            BufferUsage::storage_buffer()
                | BufferUsage::vertex_buffer_transfer_destination()
                | BufferUsage::transfer_source(), // Specify use as a storage buffer, vertex buffer, and transfer destination.
            device.active_queue_families(),
        )
        .unwrap();

        let copy_buffer = transform_data.mvp.data.clone();

        transform_data.mvp.data = device_local_buffer.clone();
        builder
        .copy_buffer(copy_buffer, transform_data.mvp.data.clone())
        .unwrap();

    } else {
        transform_data.positions.len = len as u32;
        transform_data.mvp.len = len as u32;
    }
        }
}
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
    // builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer>,
    // queue: Arc<Queue>,
    positions: Vec<ModelMat>,
) -> TransformData {
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
    let pos = GPUVector {
        data: device_local_buffer,
        len: len as u32,
        max_len: max_len as u32,
    };


    let device_local_buffer = DeviceLocalBuffer::<[MVP]>::array(
        device.clone(),
        max_len as vulkano::DeviceSize,
        BufferUsage::storage_buffer()
            | BufferUsage::vertex_buffer_transfer_destination()
            | BufferUsage::transfer_source(), // Specify use as a storage buffer, vertex buffer, and transfer destination.
        device.active_queue_families(),
    )
    .unwrap();
    let mvp = GPUVector {
        data: device_local_buffer,
        len: len as u32,
        max_len: max_len as u32,
    };

    TransformData { positions: pos, mvp: mvp }
}