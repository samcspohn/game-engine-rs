use std::{cell::SyncUnsafeCell, sync::Arc};

use crossbeam::queue::SegQueue;
use force_send_sync::SendSync;
use kira::manager::AudioManager;
use nalgebra_glm::{Quat, Vec3};
use parking_lot::{Mutex, RwLock};
use rapier3d::prelude::{QueryPipeline, RigidBodyHandle};
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, PrimaryAutoCommandBuffer,
    SecondaryAutoCommandBuffer,
};

use crate::engine::{
    audio::{self, asset::AudioAsset, system::AudioSystem},
    input::Input,
    particles::{asset::ParticleTemplate, particles::ParticlesSystem},
    physics::{collider::_ColliderType, Physics, PhysicsData},
    project::asset_manager::{AssetInstance, AssetManager, AssetManagerBase, AssetsManager},
    rendering::{component::RendererManager, model::ModelRenderer, vulkan_manager::VulkanManager},
    time::Time,
    utils::{GPUWork, PrimaryCommandBuffer},
    world::{transform::Transform, Sys, World},
    Defer, RenderJobData,
};

use super::NewRigidBody;

pub struct System<'a> {
    pub audio: &'a AudioSystem,
    pub physics: &'a PhysicsData,
    pub defer: &'a Defer,
    pub input: &'a Input,
    pub time: &'a Time,
    pub rendering: &'a RwLock<RendererManager>,
    pub assets: &'a AssetsManager,
    pub vk: Arc<VulkanManager>,
    pub gpu_work: &'a GPUWork,
    pub(crate) particle_system: &'a ParticlesSystem,
    pub(crate) new_rigid_bodies: &'a SegQueue<NewRigidBody>,
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
    pub fn play_sound(&self, template: &AssetInstance<AudioAsset>) {
        let b = &self.assets;
        let a = b.get_manager::<AudioAsset>().clone();
        unsafe {
            let c = a.lock();
            let d = c
                .as_any()
                .downcast_ref_unchecked::<AssetManager<audio::asset::Param, AudioAsset>>();
            self.audio.m.lock().play(
                d.assets_id
                    .get(&template.id)
                    .unwrap()
                    .lock()
                    .d
                    .assume_init_ref()
                    .clone(),
            );
        }
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
