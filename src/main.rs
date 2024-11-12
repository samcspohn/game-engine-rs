#![allow(warnings)]
#![feature(downcast_unchecked)]
#![feature(sync_unsafe_cell)]
#![feature(ptr_from_ref)]
#![feature(closure_lifetime_binder)]
#![feature(iter_map_windows)]
mod editor;
mod engine;
// mod terrain_eng;

use crate::engine::input::Input;
use crate::engine::particles::{component::ParticleEmitter, particles::ParticlesSystem};
use crate::engine::project::asset_manager::{AssetManagerBase, AssetsManager};
use crate::engine::project::{file_watcher, Project};
use crate::engine::rendering::camera::{Camera, CameraData};
use crate::engine::rendering::component::{buffer_usage_all, Renderer};
use crate::engine::rendering::model::{ModelManager, ModelRenderer};
use crate::engine::rendering::texture::TextureManager;
use crate::engine::rendering::vulkan_manager::VulkanManager;
use crate::engine::transform_compute::{cs, TransformCompute};
use crate::engine::world::World;
use crate::engine::{particles, runtime_compilation, transform_compute, Engine};

use crossbeam::channel::{Receiver, Sender};
use crossbeam::queue::SegQueue;
// use egui::plot::{HLine, Line, Plot, Value, Values};
use egui::TextureId;

use egui_winit_vulkano::GuiConfig;
// use egui_winit_vulkano::GuiConfig;
use glm::{vec4, Vec3};
use nalgebra_glm as glm;
use notify::{RecursiveMode, Watcher};
use parking_lot::{Mutex, RwLock};
use puffin_egui::*;
use std::{
    collections::{BTreeMap, HashMap},
    env, fs,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self},
    time::{Duration, Instant},
};
use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, ModifiersState, MouseButton,
        VirtualKeyCode, WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};


fn main() {
    println!("here");
    env::set_var("RUST_BACKTRACE", "1");
    // env::set_var("RUSTFLAGS", format!("-Z threads={}",thread::max_concurrency()));
    crate::engine::utils::SETTINGS
        .read()
        .get::<i32>("MAX_PARTICLES")
        .unwrap();
    let args: Vec<String> = env::args().collect();
    // dbg!(args);
    let engine_dir = env::current_dir().ok().unwrap();
    assert!(env::set_current_dir(&Path::new(&args[1])).is_ok()); // TODO move to top. procedurally generate cube/move cube to built in assets

    let path = "runtime";
    if let Ok(_) = fs::remove_dir_all(path) {}
    fs::create_dir(path).unwrap();

    let mut engine = engine::Engine::new(&engine_dir, &args[1], false);
    // let (mut engine, event_loop) = engine::Engine::new(&engine_dir, &args[1], false);
    engine.init();

    // engine.run(event_loop);
    // engine.run();
    while !engine.update_sim() {}
    engine.end();
}
