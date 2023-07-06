

use std::sync::Arc;

use crossbeam::queue::SegQueue;
use force_send_sync::SendSync;
use serde::{Deserialize, Serialize};
use sync_unsafe_cell::SyncUnsafeCell;

use rayon::prelude::*;

use parking_lot::Mutex;
use parking_lot::RwLock;
use vulkano::buffer::CpuAccessibleBuffer;
// use spin::Mutex;
use vulkano::{
    buffer::DeviceLocalBuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        PrimaryAutoCommandBuffer,
    },
    pipeline::graphics::viewport::Viewport,
};

// use crate::{physics::Physics};

use self::project::Project;
use self::rendering::pipeline::RenderPipeline;
use self::rendering::texture::TextureManager;
use self::rendering::vulkan_manager::VulkanManager;
use self::transform_compute::cs::ty::MVP;
use self::transform_compute::cs::ty::transform;
use self::world::World;

pub mod storage;
pub mod component;
pub mod world;
pub mod game_object;
pub mod project;
pub mod input;
pub mod rendering;
pub mod time;
pub mod transform_compute;
pub mod runtime_compilation;
// mod render_pipeline;
pub mod physics;
pub mod particles;
pub mod main_loop;

#[repr(C)]
pub struct RenderJobData<'a> {
    pub builder: &'a mut AutoCommandBufferBuilder<
        PrimaryAutoCommandBuffer,
        Arc<StandardCommandBufferAllocator>,
    >,
    pub transforms: Arc<DeviceLocalBuffer<[transform]>>,
    pub mvp: Arc<DeviceLocalBuffer<[MVP]>>,
    pub view: &'a nalgebra_glm::Mat4,
    pub proj: &'a nalgebra_glm::Mat4,
    pub pipeline: &'a RenderPipeline,
    pub viewport: &'a Viewport,
    pub texture_manager: &'a parking_lot::Mutex<TextureManager>,
    pub vk: Arc<VulkanManager>,
}

// pub struct ComponentRenderData {
//     pub vertex_buffer: Arc<Vec<Vertex>>,
//     pub normals_buffer: Arc<Vec<Normal>>,
//     pub uvs_buffer: Arc<Vec<UV>>,
//     pub index_buffer: Arc<Vec<u32>>,
//     pub texture: Option<i32>,
//     pub instance_buffer: Arc<Vec<i32>>
// }
pub struct Defer {
    work: SegQueue<Box<dyn FnOnce(&mut World) + Send + Sync>>,
}

impl Defer {
    pub fn append<T: 'static>(&self, f: T)
    where
        T: FnOnce(&mut World) + Send + Sync,
    {
        self.work.push(Box::new(f));
    }
    pub fn do_defered(&self, wrld: &mut World) {
        while let Some(w) = self.work.pop() {
            w(wrld);
        }
    }
    pub fn new() -> Defer {
        Defer {
            work: SegQueue::new(),
        }
    }
}

pub(crate) struct Engine {
    world: World,
    project: Project,
}

impl Engine {
    fn new() -> Self {
        todo!()
        // Engine {

        // }
    }
}