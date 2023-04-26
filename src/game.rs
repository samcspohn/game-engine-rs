use glm::{vec3, vec4, Mat4, Quat, Vec3};
use nalgebra_glm as glm;
use num_integer::Roots;
use parking_lot::{Mutex, RwLock};
use puffin_egui::puffin;
use rapier3d::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender},
        Arc,
    },
    time::Instant,
};
use vulkano::device::Device;
use winit::event::VirtualKeyCode;
// use rapier3d::{na::point, prelude::InteractionGroups};

use crate::{
    camera::{Camera, CameraData},
    editor_ui::PLAYING_GAME,
    engine::{
        physics::Physics,
        transform::{self, Transform, _Transform},
        Component, Defer, GameObject, Storage, Sys, System, World,
    },
    input::Input,
    inspectable::{self, Inpsect, Ins, Inspectable},
    model::ModelManager,
    particles::{cs::ty::emitter_init, ParticleCompute, ParticleEmitter},
    perf::Perf,
    renderer_component2::{Renderer, RendererData, RendererManager},
    terrain::Terrain,
};

// #[component]
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Bomb {
    pub vel: glm::Vec3,
}
impl Component for Bomb {
    // fn assign_transform(&mut self, t: Transform) {
    //     self.t = t;
    // }
    fn update(&mut self, transform: Transform, sys: &System) {
        let pos = transform.get_position();
        let vel = self.vel;
        let dt = sys.input.time.dt.min(1. / 20.);
        // let dt = 1. / 100.;
        // let dir = vel * (1.0 / 100.0);
        let ray = rapier3d::prelude::Ray {
            origin: point![pos.x, pos.y, pos.z],
            dir: vel,
        };
        if let Some((_handle, _hit)) = sys.physics.query_pipeline.cast_ray_and_get_normal(
            &sys.physics.rigid_body_set,
            &&sys.physics.collider_set,
            &ray,
            dt,
            true,
            QueryFilter::new(),
        ) {
            // let g = GameObject { t: transform.id };
            // sys.defer.append(move |world| {
            //     world.delete(g);
            // });
            self.vel = glm::reflect_vec(&vel, &&_hit.normal);
        }
        // if pos.y <= 0. {
        //     self.vel = glm::reflect_vec(&vel, &glm::vec3(0.,1.,0.));
        //     pos.y += 0.1;
        //     trans.set_position(self.t, pos);
        // }
        // sys.trans.rotate(self.t, &glm::Vec3::y(), 3.0 * dt);
        transform._move(self.vel * dt);
        self.vel += glm::vec3(0.0, -9.81, 0.0) * dt;

        // *pos += vel * (1.0 / 60.0);
    }
}

impl Inspectable for Bomb {
    fn inspect(&mut self, _transform: Transform, _id: i32, ui: &mut egui::Ui, sys: &mut Sys) {
        // ui.add(egui::Label::new("Bomb"));
        // egui::CollapsingHeader::new("Bomb")
        //     .default_open(true)
        //     .show(ui, |ui| {
        Ins(&mut self.vel).inspect("vel", ui, sys);
        // });
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Player {
    rof: f32,
    speed: f32,
}
impl Component for Player {
    fn update(&mut self, transform: Transform, sys: &System) {
        let input = &sys.input;
        let speed = self.speed * input.time.dt;
        if !input.get_key(&VirtualKeyCode::LControl) {
            // forward/backward
            if input.get_key(&VirtualKeyCode::W) {
                transform.translate((vec3(0., 0., 1.) * -speed));
                // cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * -speed;
            }
            if input.get_key(&VirtualKeyCode::S) {
                transform.translate((vec3(0., 0., 1.) * speed));
                // cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * speed;
            }
            //left/right
            if input.get_key(&VirtualKeyCode::A) {
                transform.translate(vec3(1., 0., 0.) * -speed);
                // cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(1.0, 0.0, 0.0, 1.0)).xyz() * -speed;
            }
            if input.get_key(&VirtualKeyCode::D) {
                transform.translate(vec3(1., 0., 0.) * speed);
                // cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(1.0, 0.0, 0.0, 1.0)).xyz() * speed;
            }
            // up/down
            if input.get_key(&VirtualKeyCode::Space) {
                transform.translate(vec3(0., 1., 0.) * -speed);
                // cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 1.0, 0.0, 1.0)).xyz() * -speed;
            }
            if input.get_key(&VirtualKeyCode::LShift) {
                transform.translate(vec3(0., 1., 0.) * speed);
                // cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 1.0, 0.0, 1.0)).xyz() * speed;
            }

            if input.get_mouse_button(&2) {
                let mut cam_rot = transform.get_rotation();
                cam_rot = glm::quat_rotate(
                    &cam_rot,
                    input.get_mouse_delta().0 as f32 * 0.01,
                    &(glm::inverse(&glm::quat_to_mat3(&cam_rot)) * Vec3::y()),
                );
                cam_rot = glm::quat_rotate(
                    &cam_rot,
                    input.get_mouse_delta().1 as f32 * 0.01,
                    &Vec3::x(),
                );
                transform.set_rotation(cam_rot);
                // transform.rotate(&Vec3::y(), input.get_mouse_delta().0 as f32 * 0.01);
                // transform.rotate(&Vec3::x(), input.get_mouse_delta().1 as f32 * -0.01);
                // transform.set_rotation(glm::mat3_to_quat(&glm::mat4_to_mat3(&glm::look_at(
                //     &vec3(0., 0., 0.),
                //     &-transform.forward(),
                //     &Vec3::y(),
                // ))));
            }

            const ALOT: f32 = 10_000_000. / 60.;
            if input.get_mouse_button(&0) && input.get_mouse_button(&2) {
                let len = (ALOT * input.time.dt.min(1.0 / 30.0)) as usize;
                sys.defer.append(move |world| {
                    // let len =
                    // let chunk_size =  (len / (64 * 64)).max(1);
                    (0..len)
                        .into_iter()
                        // .chunks(chunk_size)
                        .for_each(|_| {
                            let g = world.instantiate_with_transform(_Transform {
                                // position: _cam_pos
                                //     + glm::quat_to_mat3(&_cam_rot)
                                //         * (glm::Vec3::y() * 10. - glm::Vec3::z() * 25.)
                                //     + glm::vec3(
                                //         rand::random::<f32>() - 0.5,
                                //         rand::random::<f32>() - 0.5,
                                //         rand::random::<f32>() - 0.5,
                                //     ) * 18.,
                                position: glm::vec3(
                                    (rand::random::<f32>() - 0.5) * 1000f32,
                                    100f32,
                                    (rand::random::<f32>() - 0.5) * 1000f32,
                                ),
                                ..Default::default()
                            });
                            world.add_component(
                                g,
                                Bomb {
                                    vel: glm::Vec3::y() * 50.
                                        + glm::vec3(
                                            rand::random::<f32>() - 0.5,
                                            rand::random::<f32>() - 0.5,
                                            rand::random::<f32>() - 0.5,
                                        ) * 40.,
                                },
                            );
                            // world.add_component(g, Renderer::new(0));
                            world.add_component(g, ParticleEmitter::new(1));
                        });
                });
            }
        }
    }
}

impl Inspectable for Player {
    fn inspect(&mut self, _transform: Transform, _id: i32, ui: &mut egui::Ui, sys: &mut Sys) {
        Ins(&mut self.rof).inspect("rof", ui, sys);
        Ins(&mut self.speed).inspect("speed", ui, sys);
    }
}

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
    Sender<(
        Arc<(
            usize,
            Vec<Arc<(Vec<Vec<i32>>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>)>>,
        )>,
        Vec<Arc<Mutex<CameraData>>>,
        i32,
        RendererData,
        (usize, Vec<crate::particles::cs::ty::emitter_init>),
    )>,
    Receiver<Input>,
    // Sender<Terrain>,
);

pub fn game_thread_fn(world: Arc<Mutex<World>>, coms: GameComm, running: Arc<AtomicBool>) {
    let gravity = vector![0.0, -9.81, 0.0];
    let defer = Defer::new();

    ////////////////////////////////////////////////

    let mut perf = Perf {
        data: BTreeMap::new(),
    };
    let mut phys_time = 0f32;

    while running.load(Ordering::SeqCst) {
        // println!("waiting for input");
        let input = coms.1.recv().unwrap();
        // println!("input recvd");
        let (transform_data, renderer_data, emitter_len, v, cam_datas, main_cam_id) = {
            let mut world = world.lock();
            if unsafe { PLAYING_GAME } {
                puffin::profile_scope!("game loop");
                let inst = Instant::now();
                {
                    puffin::profile_scope!("world update");
                    if phys_time > 1.0 / 30.0 {
                        let sys = world.sys.lock();
                        let len = sys.physics.rigid_body_set.len();
                        let num_threads =
                            (len / (num_cpus::get().sqrt())).max(1).min(num_cpus::get());
                        drop(sys);
                        rayon::ThreadPoolBuilder::new()
                            .num_threads(num_threads)
                            .build_scoped(
                                |thread| thread.run(),
                                |pool| {
                                    pool.install(|| {
                                        world.sys.lock().physics.step(&gravity, &mut perf);
                                    })
                                },
                            )
                            .unwrap();
                        phys_time -= 1.0 / 30.0;
                    }
                    phys_time += input.time.dt;
                    world.update(&defer, &input);
                }
                {
                    puffin::profile_scope!("defered");
                    defer.do_defered(&mut world);
                }

                perf.update("world".into(), Instant::now() - inst);
            } else {
                world.editor_update(&defer, &input);
            }
            let inst = Instant::now();
            let transform_data = {
                puffin::profile_scope!("get transform data");
                world.transforms.write().get_transform_data_updates()
            };
            perf.update("get transform data".into(), Instant::now() - inst);

            let inst = Instant::now();
            let _sys = world.sys.clone();
            let sys = _sys.lock();
            let mut rm = sys.renderer_manager.write();
            let renderer_data = {
                // let a = rm.model_indirect.read();
                // let b = a.deref();
                let renderer_data = RendererData {
                    model_indirect: rm
                        .model_indirect
                        .read()
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                    indirect_model: rm
                        .indirect_model
                        .read()
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                    updates: rm
                        .updates
                        .iter()
                        .flat_map(|(id, t)| {
                            vec![id.clone(), t.indirect_id.clone(), t.transform_id.clone()]
                                .into_iter()
                        })
                        .collect(),
                    transforms_len: rm.transforms.data.len() as i32,
                };
                rm.updates.clear();
                renderer_data
            };

            let emitter_len = world
                .get_components::<ParticleEmitter>()
                .unwrap()
                .read()
                .as_any()
                .downcast_ref::<Storage<ParticleEmitter>>()
                .unwrap()
                .data
                .len();
            let v = {
                // let sys = world.sys.lock();
                let mut emitter_inits = sys.particles.emitter_inits.lock();
                let mut v = Vec::<emitter_init>::new();
                std::mem::swap(&mut v, &mut emitter_inits);
                v
            };
            // *lock = Arc::new(Mutex::new(Vec::new()));

            let camera_components = world.get_components::<Camera>().unwrap().read();
            let camera_storage = camera_components
                .as_any()
                .downcast_ref::<Storage<Camera>>()
                .unwrap();
            let mut main_cam_id = -1;
            let cam_datas = camera_storage
                .valid
                .iter()
                .zip(camera_storage.data.iter())
                .map(|(v, d)| {
                    if v.load(Ordering::Relaxed) {
                        let d = d.lock();
                        main_cam_id = d.0;
                        d.1.get_data()
                    } else {
                        None
                    }
                })
                .filter(|a| a.is_some())
                .map(|a| a.unwrap())
                .collect();
            perf.update("get renderer data".into(), Instant::now() - inst);
            (
                transform_data,
                renderer_data,
                emitter_len,
                v,
                cam_datas,
                main_cam_id,
            )
        };

        // std::thread::sleep(Duration::from_millis(5));
        let inst = Instant::now();

        let res = coms.0.send((
            transform_data.clone(),
            cam_datas,
            main_cam_id,
            renderer_data,
            (emitter_len, v),
            // render_data,
            // terr_chunks,
        ));
        perf.update("send data".into(), Instant::now() - inst);
        if res.is_err() {
            println!("ohno");
        }
    }
    perf.print();
    // let p = perf.iter();
    // for (k, x) in p {
    //     for (k, x) in p {
    //         let len = x.len();
    //         println!("{}: {:?}", k, (x.into_iter().map(|a| a).sum::<Duration>() / len as u32));
    //     }
    // }
}
