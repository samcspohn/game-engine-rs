use crossbeam::queue::SegQueue;
use force_send_sync::SendSync;
use glm::{vec3, Vec3};
use nalgebra_glm as glm;
use num_integer::Roots;
use parking_lot::Mutex;
use puffin_egui::puffin;
use rapier3d::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, CommandBufferInheritanceInfo};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender},
        Arc,
    },
    time::Instant,
};

use winit::{event::VirtualKeyCode, window::Window};
// use rapier3d::{na::point, prelude::InteractionGroups};

use crate::{
    camera::{Camera, CameraData},
    editor::editor_ui::PLAYING_GAME,
    editor::inspectable::{Inpsect, Ins, Inspectable},
    engine::input::Input,
    engine::{world::{transform::TransformData, World}, component::{SecondaryCommandBuffer, GPUWork}},
    particles::{cs::ty::emitter_init, ParticleEmitter},
    perf::Perf,
    renderer_component::RendererData,
};

type GameComm = (
    Sender<RenderingData>,
    Receiver<(Input, bool)>,
    // Sender<Terrain>,
);
// transform_data, cam_datas, main_cam_id, renderer_data, emitter_inits
#[repr(C)]
pub struct RenderingData {
    pub transform_data: TransformData,
    pub cam_datas: std::vec::Vec<Arc<Mutex<CameraData>>>,
    pub main_cam_id: i32,
    pub renderer_data: RendererData,
    pub emitter_inits: (usize, std::vec::Vec<emitter_init>),
    pub gpu_work: GPUWork,
}

pub fn game_thread_fn(world: Arc<Mutex<World>>, coms: GameComm, running: Arc<AtomicBool>) {
    let gravity = vector![0.0, -9.81, 0.0];
    let mut perf = Perf {
        data: BTreeMap::new(),
    };
    let mut phys_time = 0f32;
    let phys_step = 1. / 30.;
    println!("game thread id: {:?}", std::thread::current().id());
    println!(
        "game thread priority: {:?}",
        thread_priority::get_current_thread_priority().ok().unwrap()
    );
    while running.load(Ordering::SeqCst) {
        let (input, playing_game) = coms.1.recv().unwrap();
        let mut world = world.lock();
        let gpu_work = SegQueue::new();
        if playing_game {
            puffin::profile_scope!("game loop");
            let inst = Instant::now();
            {
                puffin::profile_scope!("world update");
                if phys_time >= phys_step {
                    // let mut physics = world.sys.physics.lock();
                    let len = world.sys.physics.lock().rigid_body_set.len();
                    let num_threads = (len / (num_cpus::get().sqrt())).max(1).min(num_cpus::get());
                    // let physics = Arc::new(&mut physics);
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(num_threads)
                        .build_scoped(
                            |thread| thread.run(),
                            |pool| {
                                pool.install(|| {
                                    world.sys.physics.lock().step(&gravity, &mut perf);
                                })
                            },
                        )
                        .unwrap();
                    // drop(physics);
                    phys_time -= phys_step;
                }
                phys_time += input.time.dt;
                world._update(&input, &gpu_work);
            }
            {
                puffin::profile_scope!("defered");
                world.do_defered();
                world._destroy();
                world.defer_instantiate();
            }

            perf.update("world sim".into(), Instant::now() - inst);
        } else {
            world.editor_update(&input, &gpu_work);
        }
        let inst = Instant::now();
        let transform_data = world.transforms.get_transform_data_updates();
        perf.update("get transform data".into(), Instant::now() - inst);

        let inst = Instant::now();
        let renderer_data = world.sys.renderer_manager.write().get_renderer_data();
        perf.update("get renderer data".into(), Instant::now() - inst);
        let emitter_len = world.get_emitter_len();
        let v = world.sys.particles_system.emitter_inits.get_vec();
        let (main_cam_id, cam_datas) = world.get_cam_datas();
        drop(world);
        let data = RenderingData {
            transform_data,
            cam_datas,
            main_cam_id,
            renderer_data,
            emitter_inits: (emitter_len, v),
            gpu_work,
        };
        let res = coms.0.send(data);
        if res.is_err() {
            println!("ohno");
        }
    }
    perf.print();
}
