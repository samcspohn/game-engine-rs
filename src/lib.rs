// mod engine;
#![allow(warnings, unused)]
#![feature(downcast_unchecked)]
#![feature(sync_unsafe_cell)]
#![feature(ptr_from_ref)]
#![feature(closure_lifetime_binder)]
#![feature(non_lifetime_binders)]
pub mod editor;
pub mod engine;
pub use component_derive;

pub use egui;
pub use force_send_sync;
pub use nalgebra_glm as glm;
pub use noise;
pub use parking_lot;
pub use rapier3d;
pub use rayon;
pub use serde;
pub use vulkano;
pub use winit;
pub mod prelude {
    // pub use serde;
    pub use crate::engine::particles::component::ParticleEmitter;
    pub use crate::engine::rendering::component::Renderer;
    pub use component_derive::AssetID;
    pub use component_derive::ComponentID;
    pub use engine::{
        input, utils,
        world::{
            component::{Component, System, _ComponentID},
            transform::{Transform, TransformRef, _Transform},
            Sys, World,
        },
        RenderJobData,
    };
    pub use inspectable::{Inpsect, Ins};
    pub use lazy_static;
    pub use rand;

    pub use crate::engine::rendering::vulkan_manager::VulkanManager;
    pub use crate::{editor::inspectable, engine};
    // pub use crate::engine::utils;
}
#[cfg(target_os = "windows")]
pub mod win_alloc {
    pub use mimalloc::MiMalloc;

    #[global_allocator]
    static GLOBAL: MiMalloc = MiMalloc;
}
