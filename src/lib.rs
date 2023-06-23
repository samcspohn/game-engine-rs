// mod engine;
#![allow(warnings, unused)]

pub mod model;
pub mod perf;
pub mod renderer;
pub mod camera;
pub mod color_gradient;
pub mod editor;
pub mod engine;
pub mod game;
pub mod particle_sort;
pub mod particles;
pub mod physics;
pub mod render_pipeline;
pub mod renderer_component;
pub mod texture;
pub mod time;
pub mod transform_compute;
pub mod vulkan_manager;
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
    pub use particles::ParticleEmitter;
    pub use renderer_component::Renderer;

    use crate::{engine, editor::inspectable, particles, renderer_component};
}
