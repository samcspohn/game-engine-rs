use crossbeam::{
    channel::{Receiver, Sender},
    queue::SegQueue,
};
use force_send_sync::SendSync;
use glm::{vec3, Vec3};
use nalgebra_glm as glm;
use num_integer::Roots;
use parking_lot::Mutex;
use puffin_egui::puffin;
use rapier3d::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Instant,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferInheritanceInfo, CommandBufferUsage,
};

use winit::{event::VirtualKeyCode, window::Window};

use crate::{
    editor::editor_ui::PLAYING_GAME,
    editor::inspectable::{Inpsect, Ins, Inspectable},
    engine::rendering::camera::CameraData,
    engine::world::{transform::TransformData, World},
    engine::{input::Input, perf::Perf},
};

use super::{
    particles::shaders::cs::ty::{burst, emitter_init},
    rendering::renderer_component::RendererData,
    utils::GPUWork,
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
    pub cam_datas: Vec<Arc<Mutex<CameraData>>>,
    pub main_cam_id: i32,
    pub renderer_data: RendererData,
    pub emitter_inits: (usize, Vec<emitter_init>, Vec<emitter_init>, Vec<burst>),
    pub gpu_work: GPUWork,
}

pub fn main_loop(world: Arc<Mutex<World>>, coms: GameComm, running: Arc<AtomicBool>) {
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
        let inst = Instant::now();
        if playing_game {
            puffin::profile_scope!("game loop");
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
                let inst = Instant::now();
                world._update(&input, &gpu_work);
                perf.update("world update".into(), Instant::now() - inst);
            }
            {
                puffin::profile_scope!("defered");
                let inst = Instant::now();
                world.do_defered();
                perf.update("world do defered".into(), Instant::now() - inst);

                let inst = Instant::now();
                world._destroy();
                perf.update("world _destroy".into(), Instant::now() - inst);

                let inst = Instant::now();                
                world.defer_instantiate();
                perf.update("world instantiate".into(), Instant::now() - inst);

            }
        } else {
            world._destroy();
            world.editor_update(&input, &gpu_work);
        }
        perf.update("world sim".into(), Instant::now() - inst);
        let inst = Instant::now();
        let transform_data = world.transforms.get_transform_data_updates();
        perf.update("get transform data".into(), Instant::now() - inst);

        let inst = Instant::now();
        let renderer_data = world.sys.renderer_manager.write().get_renderer_data();
        perf.update("get renderer data".into(), Instant::now() - inst);
        let emitter_len = world.get_emitter_len();
        let emitter_inits = world.sys.particles_system.emitter_inits.get_vec();
        let emitter_deinits = world.sys.particles_system.emitter_deinits.get_vec();
        let particle_bursts = world.sys.particles_system.particle_burts.get_vec();
        let (main_cam_id, cam_datas) = world.get_cam_datas();
        drop(world);
        let data = RenderingData {
            transform_data,
            cam_datas,
            main_cam_id,
            renderer_data,
            emitter_inits: (emitter_len, emitter_inits, emitter_deinits, particle_bursts),
            gpu_work,
        };
        let res = coms.0.send(data);
        if res.is_err() {
            println!("ohno");
        }
    }
    perf.print();
}
