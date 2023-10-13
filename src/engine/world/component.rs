use std::sync::Arc;

use crossbeam::queue::SegQueue;
use force_send_sync::SendSync;
use nalgebra_glm::{Quat, Vec3};
use parking_lot::{Mutex, RwLock};
use rapier3d::prelude::RigidBodyHandle;
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, PrimaryAutoCommandBuffer,
    SecondaryAutoCommandBuffer,
};

use crate::engine::{
    input::Input,
    particles::{particle_asset::ParticleTemplate, particles::ParticleCompute},
    physics::{Physics, collider::_ColliderType},
    project::asset_manager::{AssetInstance, AssetManagerBase, AssetsManager},
    rendering::{component::RendererManager, model::ModelRenderer, vulkan_manager::VulkanManager},
    time::Time,
    utils::{GPUWork, PrimaryCommandBuffer},
    world::{transform::Transform, Sys, World},
    Defer, RenderJobData,
};

pub struct System<'a> {
    pub physics: &'a Physics,
    pub defer: &'a Defer,
    pub input: &'a Input,
    pub time: &'a Time,
    pub rendering: &'a RwLock<RendererManager>,
    pub assets: &'a AssetsManager,
    pub vk: Arc<VulkanManager>,
    pub gpu_work: &'a GPUWork,
    pub(crate) particle_system: &'a ParticleCompute,
    pub(crate) new_rigid_bodies: &'a SegQueue<(_ColliderType,Vec3,Quat,force_send_sync::SendSync<*mut RigidBodyHandle>)>
}
impl<'a> System<'a> {
    pub fn get_model_manager(&self) -> Arc<Mutex<dyn AssetManagerBase + Send + Sync>> {
        let b = &self.assets;
        let a = b.get_manager::<ModelRenderer>().clone();
        a
    }
    pub fn enque_gpu_work<T: 'static>(&self, gpu_job: T)
    where
        T: FnOnce(&mut PrimaryCommandBuffer, Arc<VulkanManager>),
    {
        self.gpu_work
            .push(unsafe { SendSync::new(Box::new(gpu_job)) });
    }
    pub fn emitter_burst(
        &self,
        template: &AssetInstance<ParticleTemplate>,
        count: u32,
        position: Vec3,
        direction: Vec3,
    ) {
        // let r = rotation;
        let burst = crate::engine::particles::shaders::cs::burst {
            pos: position.into(),
            template_id: template.id,
            dir: direction.into(),
            count,
        };

        self.particle_system.particle_burts.push(burst);
    }
}

pub trait Component {
    // fn assign_transform(&mut self, t: Transform);
    fn init(&mut self, transform: &Transform, id: i32, sys: &Sys) {}
    fn deinit(&mut self, transform: &Transform, _id: i32, sys: &Sys) {}
    fn on_start(&mut self, transform: &Transform, sys: &System) {} // TODO implement call
    fn on_destroy(&mut self, transform: &Transform, sys: &System) {} // TODO implement call
    fn update(&mut self, transform: &Transform, sys: &System, world: &World) {}
    fn late_update(&mut self, transform: &Transform, sys: &System) {}
    fn editor_update(&mut self, transform: &Transform, sys: &System) {}
    fn on_render(&mut self, _t_id: i32) -> Box<dyn Fn(&mut RenderJobData) + Send + Sync> {
        Box::new(|_rd: &mut RenderJobData| {})
    }
    fn inspect(&mut self, transform: &Transform, id: i32, ui: &mut egui::Ui, sys: &Sys);
    // fn as_any(&self) -> &dyn Any;
}

pub trait _ComponentID {
    const ID: u64;
}
