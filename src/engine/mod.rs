

use std::sync::Arc;

use crate::model::{Vertex, Normal, UV};
use crate::{
    physics,
};
use force_send_sync::SendSync;
use serde::{Deserialize, Serialize};
use sync_unsafe_cell::SyncUnsafeCell;

use rayon::prelude::*;

use crossbeam::{queue::SegQueue};

use parking_lot::Mutex;
use parking_lot::RwLock;
use thincollections::thin_map::{ThinMap};
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

use crate::{

    particles::{ParticleCompute, ParticleEmitter},
    renderer::RenderPipeline,

    texture::TextureManager,

};

use crate::{physics::Physics};

use self::project::Project;
use self::world::World;

pub mod storage;
pub mod component;
pub mod world;
pub mod game_object;
pub mod project;
pub mod input;
pub mod rendering;

#[repr(C)]
pub struct RenderJobData<'a> {
    pub builder: &'a mut AutoCommandBufferBuilder<
        PrimaryAutoCommandBuffer,
        Arc<StandardCommandBufferAllocator>,
    >,
    pub transforms: Arc<DeviceLocalBuffer<[crate::transform_compute::cs::ty::transform]>>,
    pub mvp: Arc<DeviceLocalBuffer<[crate::transform_compute::cs::ty::MVP]>>,
    pub view: &'a nalgebra_glm::Mat4,
    pub proj: &'a nalgebra_glm::Mat4,
    pub pipeline: &'a RenderPipeline,
    pub viewport: &'a Viewport,
    pub texture_manager: &'a parking_lot::Mutex<TextureManager>,
    pub vk: Arc<crate::vulkan_manager::VulkanManager>,
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