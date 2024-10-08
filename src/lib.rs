// mod engine;
#![allow(warnings, unused)]
#![feature(downcast_unchecked)]
#![feature(sync_unsafe_cell)]
#![feature(ptr_from_ref)]
#![feature(closure_lifetime_binder)]
#![feature(non_lifetime_binders)]
pub mod editor;
pub mod engine;

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
pub use id;
pub mod prelude {
    // pub use serde;
    pub use crate::engine::particles::component::ParticleEmitter;
    pub use crate::engine::rendering::component::Renderer;
    pub use id::*;
    pub use engine::{
        input, utils,
        world::{
            component::{Component, System},
            transform::{Transform, TransformRef, _Transform},
            Sys, World,
        },
        RenderData,
    };
    pub use inspectable::{Inpsect, Ins};
    pub use lazy_static;
    pub use rand;

    pub use crate::engine::rendering::vulkan_manager::VulkanManager;
    pub use crate::{editor::inspectable, engine};
    // pub use crate::engine::utils;
}