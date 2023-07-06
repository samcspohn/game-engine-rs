use std::sync::Arc;

use crossbeam::queue::SegQueue;
use force_send_sync::SendSync;
use parking_lot::{Mutex, RwLock};
use vulkano::command_buffer::{AutoCommandBufferBuilder, SecondaryAutoCommandBuffer, allocator::StandardCommandBufferAllocator, PrimaryAutoCommandBuffer};


use super::{
    input::Input,
    project::asset_manager::{AssetManagerBase, AssetsManager},
    world::{transform::Transform, Sys, World},
    Defer, RenderJobData, physics::Physics, rendering::{renderer_component::RendererManager, vulkan_manager::VulkanManager, model::ModelRenderer},
};

pub type SecondaryCommandBuffer = AutoCommandBufferBuilder<SecondaryAutoCommandBuffer, Arc<StandardCommandBufferAllocator>>;
pub type PrimaryCommandBuffer = AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, Arc<StandardCommandBufferAllocator>>;
pub type GPUWork = SegQueue<SendSync<Box<dyn FnOnce(&mut PrimaryCommandBuffer)>>>;
pub struct System<'a> {
    pub physics: &'a Physics,
    pub defer: &'a Defer,
    pub input: &'a Input,
    pub rendering: &'a RwLock<RendererManager>,
    pub assets: &'a AssetsManager,
    pub vk: Arc<VulkanManager>,
    pub gpu_work: &'a GPUWork,
}
impl<'a> System<'a> {
    pub fn get_model_manager(&self) -> Arc<Mutex<dyn AssetManagerBase + Send + Sync>> {
        let b = &self.assets;
        let a = b.get_manager::<ModelRenderer>().clone();
        a
    }
    pub fn enque_gpu_work<T: 'static>(&self, gpu_job: T) where T: FnOnce(&mut PrimaryCommandBuffer) {
        self.gpu_work.push(unsafe{ SendSync::new(Box::new(gpu_job)) });
    }
}

pub trait Component {
    // fn assign_transform(&mut self, t: Transform);
    fn init(&mut self, _transform: &Transform, _id: i32, _sys: &Sys) {}
    fn deinit(&mut self, _transform: &Transform, _id: i32, _sys: &Sys) {}
    fn on_start(&mut self, _transform: &Transform, _sys: &System) {} // TODO implement call
    fn on_destroy(&mut self, _transform: &Transform, _sys: &System) {} // TODO implement call
    fn update(&mut self, _transform: &Transform, _sys: &System, world: &World) {}
    fn late_update(&mut self, _transform: &Transform, _sys: &System) {}
    fn editor_update(&mut self, _transform: &Transform, _sys: &System) {}
    fn on_render(&mut self, _t_id: i32) -> Box<dyn Fn(&mut RenderJobData)> {
        Box::new(|_rd: &mut RenderJobData| {})
    }
    // fn as_any(&self) -> &dyn Any;
}

pub trait _ComponentID {
    const ID: u64;
}

#[derive(Clone, Copy)]
pub struct GameObject {
    pub t: i32,
}
