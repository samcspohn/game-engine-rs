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
    engine::input::Input,
    editor::inspectable::{Inpsect, Ins, Inspectable},
    particles::{cs::ty::emitter_init, ParticleEmitter},
    perf::Perf,
    renderer_component::RendererData, engine::world::World,
};

// // #[component]
// #[derive(Default, Clone, Serialize, Deserialize)]
// pub struct Bomb {
//     pub vel: glm::Vec3,
// }
// impl Component for Bomb {
//     // fn assign_transform(&mut self, t: Transform) {
//     //     self.t = t;
//     // }
//     fn update(&mut self, transform: &Transform, sys: &System, world: &World) {
//         let pos = transform.get_position();
//         let vel = self.vel;
//         let dt = sys.input.time.dt.min(1. / 20.);
//         // let dt = 1. / 100.;
//         // let dir = vel * (1.0 / 100.0);
//         let ray = rapier3d::prelude::Ray {
//             origin: point![pos.x, pos.y, pos.z],
//             dir: vel,
//         };
//         if let Some((_handle, _hit)) = sys.physics.query_pipeline.cast_ray_and_get_normal(
//             &sys.physics.rigid_body_set,
//             &sys.physics.collider_set,
//             &ray,
//             dt,
//             true,
//             QueryFilter::new(),
//         ) {
//             // println!("impact");
//             world.destroy(transform.id);
//             // let g = GameObject { t: transform.id };
//             // sys.defer.append(move |world| {
//             //     world.delete(g);
//             // });
//             // self.vel = glm::reflect_vec(&vel, &_hit.normal);
//         }
//         // if pos.y <= 0. {
//         //     self.vel = glm::reflect_vec(&vel, &glm::vec3(0.,1.,0.));
//         //     pos.y += 0.1;
//         //     trans.set_position(self.t, pos);
//         // }
//         // sys.trans.rotate(self.t, &glm::Vec3::y(), 3.0 * dt);
//         transform._move(self.vel * dt);
//         self.vel += glm::vec3(0.0, -9.81, 0.0) * dt;

//         // *pos += vel * (1.0 / 60.0);
//     }
// }

// impl Inspectable for Bomb {
//     fn inspect(&mut self, _transform: &Transform, _id: i32, ui: &mut egui::Ui, sys: &Sys) {
//         // ui.add(egui::Label::new("Bomb"));
//         // egui::CollapsingHeader::new("Bomb")
//         //     .default_open(true)
//         //     .show(ui, |ui| {
//         Ins(&mut self.vel).inspect("vel", ui, sys);
//         // });
//     }
// }

// #[derive(Default, Clone, Serialize, Deserialize)]
// #[serde(default)]
// pub struct Player {
//     rof: f32,
//     speed: f32,
//     grab_mode: bool,
//     cursor_vis: bool,
// }
// impl Component for Player {
//     fn update(&mut self, transform: &Transform, sys: &System, world: &World) {
//         let input = &sys.input;
//         let speed = self.speed * input.time.dt;

//         if input.get_key_press(&VirtualKeyCode::G) {
//             let _er = sys
//                 .vk
//                 .surface
//                 .object()
//                 .unwrap()
//                 .downcast_ref::<Window>()
//                 .unwrap()
//                 .set_cursor_grab(match self.grab_mode {
//                     true => winit::window::CursorGrabMode::Confined,
//                     false => winit::window::CursorGrabMode::None,
//                 });
//             self.grab_mode = !self.grab_mode;
//         }
//         // if input.get_key_press(&VirtualKeyCode::J) {
//         //     lock_cull = !lock_cull;
//         //     // lock_cull.
//         // }

//         if input.get_key(&VirtualKeyCode::H) {
//             sys.vk
//                 .surface
//                 .object()
//                 .unwrap()
//                 .downcast_ref::<Window>()
//                 .unwrap()
//                 .set_cursor_visible(self.cursor_vis);
//             self.cursor_vis = !self.cursor_vis;
//         }

//         if input.get_key_press(&VirtualKeyCode::R) {
//             self.speed *= 1.5;
//         }
//         if input.get_key_press(&VirtualKeyCode::F) {
//             self.speed /= 1.5;
//         }

//         // forward/backward
//         if input.get_key(&VirtualKeyCode::W) {
//             transform.translate(vec3(0., 0., 1.) * -speed);
//         }
//         if input.get_key(&VirtualKeyCode::S) {
//             transform.translate(vec3(0., 0., 1.) * speed);
//         }
//         //left/right
//         if input.get_key(&VirtualKeyCode::A) {
//             transform.translate(vec3(1., 0., 0.) * -speed);
//         }
//         if input.get_key(&VirtualKeyCode::D) {
//             transform.translate(vec3(1., 0., 0.) * speed);
//         }
//         // up/down
//         if input.get_key(&VirtualKeyCode::Space) {
//             transform.translate(vec3(0., 1., 0.) * -speed);
//         }
//         if input.get_key(&VirtualKeyCode::LShift) {
//             transform.translate(vec3(0., 1., 0.) * speed);
//         }

//         if input.get_mouse_button(&2) {
//             let mut cam_rot = transform.get_rotation();
//             cam_rot = glm::quat_rotate(
//                 &cam_rot,
//                 input.get_mouse_delta().0 as f32 * 0.01,
//                 &(glm::inverse(&glm::quat_to_mat3(&cam_rot)) * Vec3::y()),
//             );
//             cam_rot = glm::quat_rotate(
//                 &cam_rot,
//                 input.get_mouse_delta().1 as f32 * 0.01,
//                 &Vec3::x(),
//             );
//             transform.set_rotation(cam_rot);
//             // transform.rotate(&Vec3::y(), input.get_mouse_delta().0 as f32 * 0.01);
//             // transform.rotate(&Vec3::x(), input.get_mouse_delta().1 as f32 * -0.01);
//             // transform.set_rotation(glm::mat3_to_quat(&glm::mat4_to_mat3(&glm::look_at(
//             //     &vec3(0., 0., 0.),
//             //     &-transform.forward(),
//             //     &Vec3::y(),
//             // ))));
//         }

//         const ALOT: f32 = 10_000_000. / 60.;
//         if input.get_mouse_button(&0) && input.get_mouse_button(&2) {
//             let len = (ALOT * input.time.dt.min(1.0 / 30.0)) as usize;
//             world
//                 .instantiate_many(len as i32, 0, 0)
//                 .with_transform(|| {
//                     _Transform {
//                         // position: _cam_pos
//                         //     + glm::quat_to_mat3(&_cam_rot)
//                         //         * (glm::Vec3::y() * 10. - glm::Vec3::z() * 25.)
//                         //     + glm::vec3(
//                         //         rand::random::<f32>() - 0.5,
//                         //         rand::random::<f32>() - 0.5,
//                         //         rand::random::<f32>() - 0.5,
//                         //     ) * 18.,
//                         position: glm::vec3(
//                             (rand::random::<f32>() - 0.5) * 1500f32,
//                             100f32,
//                             (rand::random::<f32>() - 0.5) * 1500f32,
//                         ),
//                         ..Default::default()
//                     }
//                 })
//                 .with_com(&|| Bomb {
//                     vel: glm::Vec3::y() * 100.
//                         + glm::vec3(
//                             rand::random::<f32>() - 0.5,
//                             rand::random::<f32>() - 0.5,
//                             rand::random::<f32>() - 0.5,
//                         ) * 70.,
//                 })
//                 .with_com::<ParticleEmitter>(&|| ParticleEmitter::new(1))
//                 .build();
//         }
//     }
// }

// impl Inspectable for Player {
//     fn inspect(&mut self, _transform: &Transform, _id: i32, ui: &mut egui::Ui, sys: &Sys) {
//         Ins(&mut self.rof).inspect("rof", ui, sys);
//         Ins(&mut self.speed).inspect("speed", ui, sys);
//     }
// }

// #[component]
// pub struct Maker {}
// impl Component for Maker {
//     // fn assign_transform(&mut self, t: Transform) {
//     //     self.t = t;
//     // }
//     // fn init(&mut self, t: Transform, _sys: &mut Sys) {
//     //     self.t = t;
//     // }
//     fn update(&mut self, _transform: Transform, sys: &System) {
//         sys.defer.append(|world| {
//             let g = world.instantiate();
//             world.add_component(
//                 g,
//                 Bomb {
//                     vel: glm::vec3(rand::random(), rand::random(), rand::random()),
//                 },
//             );
//             // world.get_component::<Bomb, _>(g, |b| {
//             //     if let Some(b) = b {
//             //         *b.vel = glm::vec3(rand::random(), rand::random(), rand::random());
//             //     }
//             // });
//         });
//     }
// }

type GameComm = (
    Sender<RenderingData>,
    Receiver<(Input, bool)>,
    // Sender<Terrain>,
);
// transform_data, cam_datas, main_cam_id, renderer_data, emitter_inits
#[repr(C)]
pub struct RenderingData {
    pub transform_data: Arc<(
        usize,
        std::vec::Vec<
            Arc<(
                std::vec::Vec<std::vec::Vec<i32>>,
                std::vec::Vec<[f32; 3]>,
                std::vec::Vec<[f32; 4]>,
                std::vec::Vec<[f32; 3]>,
            )>,
        >,
    )>,
    pub cam_datas: std::vec::Vec<Arc<Mutex<CameraData>>>,
    pub main_cam_id: i32,
    pub renderer_data: RendererData,
    pub emitter_inits: (usize, std::vec::Vec<emitter_init>),
}

// #[no_mangle]
pub fn game_loop(
    world: &mut World,
    perf: &mut Perf,
    playing_game: bool,
    input: &Input,
) -> RenderingData {
    let gravity = vector![0.0, -9.81, 0.0];
    static mut phys_time: f32 = 0f32;
    let phys_step = 1. / 30.;

    if playing_game {
        puffin::profile_scope!("game loop");
        let inst = Instant::now();
        {
            puffin::profile_scope!("world update");
            if unsafe { phys_time >= phys_step } {
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
                                world.sys.physics.lock().step(&gravity, perf);
                            })
                        },
                    )
                    .unwrap();
                // drop(physics);
                unsafe {
                    phys_time -= phys_step;
                }
            }
            unsafe {
                phys_time += input.time.dt;
            }
            world._update(&input);
        }
        {
            puffin::profile_scope!("defered");
            world.do_defered();
            world._destroy();
            world.defer_instantiate();
        }

        perf.update("world sim".into(), Instant::now() - inst);
    } else {
        world.editor_update(&input);
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
    // drop(world);
    let inst = Instant::now();

    RenderingData {
        transform_data: transform_data.clone(),
        cam_datas,
        main_cam_id,
        renderer_data,
        emitter_inits: (emitter_len, v),
    }
    // (
    //     transform_data.clone(),
    //     cam_datas,
    //     main_cam_id,
    //     renderer_data,
    //     (emitter_len, v),
    // )
}
pub fn game_thread_fn(
    world: Arc<Mutex<World>>,
    coms: GameComm,
    running: Arc<AtomicBool>,
    // func: &dyn Fn(&mut World, &mut Perf, bool, &Input) -> RenderingData,
) {
    let gravity = vector![0.0, -9.81, 0.0];
    let mut perf = Perf {
        data: BTreeMap::new(),
    };
    let mut phys_time = 0f32;
    let phys_step = 1. / 30.;
    while running.load(Ordering::SeqCst) {
        let (input, playing_game) = coms.1.recv().unwrap();
        let mut world = world.lock();
        // std::thread::sleep(std::time::Duration::from_millis(10));
        // {

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
                world._update(&input);
            }
            {
                puffin::profile_scope!("defered");
                world.do_defered();
                world._destroy();
                world.defer_instantiate();
            }

            perf.update("world sim".into(), Instant::now() - inst);
        } else {
            world.editor_update(&input);
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
            transform_data: transform_data.clone(),
            cam_datas,
            main_cam_id,
            renderer_data,
            emitter_inits: (emitter_len, v),
        };
        // }

        // let data = func(&mut world, &mut perf, playing_game, &input);
        // drop(world);
        let inst = Instant::now();
        let res = coms.0.send(data);
        perf.update("send data".into(), Instant::now() - inst);
        if res.is_err() {
            println!("ohno");
        }
    }
    perf.print();
}
