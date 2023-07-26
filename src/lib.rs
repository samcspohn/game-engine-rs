// mod engine;
#![allow(warnings, unused)]

pub mod editor;
pub mod engine;
pub use component_derive;

pub use vulkano;
pub use rayon;
pub use noise;
pub use parking_lot;
pub use force_send_sync;
pub use nalgebra_glm as glm;
pub mod prelude {
    pub use component_derive::ComponentID;
    pub use component_derive::AssetID;
    pub use engine::{
        world::{
            component::{Component, System, _ComponentID},
            transform::{Transform, _Transform},
            Sys, World,
        },
        input,
        RenderJobData,
        utils,
    };
    pub use inspectable::{Inpsect, Ins, Inspectable};
    pub use crate::engine::particles::particles::ParticleEmitter;
    pub use crate::engine::rendering::renderer_component::Renderer;

    pub use crate::engine::rendering::vulkan_manager::VulkanManager;
    pub use crate::{engine, editor::inspectable};
}
#[cfg(target_os = "windows")]
pub mod win_alloc {
    pub use mimalloc::MiMalloc;
    
    #[global_allocator]
    static GLOBAL: MiMalloc = MiMalloc;
}
