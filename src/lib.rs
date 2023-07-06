// mod engine;
#![allow(warnings, unused)]


pub mod perf;
pub mod color_gradient;
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
        component::{Component, System, _ComponentID},
        world::{
            transform::{Transform, _Transform},
            Sys, World,
        },
        input,
        RenderJobData,
    };
    pub use inspectable::{Inpsect, Ins, Inspectable};
    pub use crate::engine::particles::particles::ParticleEmitter;
    pub use crate::engine::rendering::renderer_component::Renderer;

    use crate::{engine, editor::inspectable};
}
#[cfg(target_os = "windows")]
pub mod win_alloc {
    pub use mimalloc::MiMalloc;
    
    #[global_allocator]
    static GLOBAL: MiMalloc = MiMalloc;
}
