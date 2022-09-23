use component_derive::component;
use core::time;
use crossbeam::queue::SegQueue;
use glm::{vec4, Vec3};
use nalgebra_glm as glm;
use parking_lot::{Mutex, RwLock};
use puffin_egui::puffin;
use rapier3d::prelude::*;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    ops::Deref,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{Receiver, Sender},
        Arc,
    },
    time::{Duration, Instant},
};
use vulkano::{buffer::CpuAccessibleBuffer, device::Device};
use winit::event::VirtualKeyCode;
// use rapier3d::{na::point, prelude::InteractionGroups};

use crate::{
    engine::{
        physics::{self, Physics},
        transform::{Transform, Transforms, _Transform},
        Component, GameObject, LazyMaker, Sys, System, World,
    },
    input::Input,
    model::ModelManager,
    perf::Perf,
    renderer::{Id, ModelMat},
    renderer_component2::{Offset, Renderer, RendererData, RendererManager},
    terrain::Terrain,
    texture::TextureManager,
    transform_compute::cs::ty::transform,
};

#[component]
pub struct Bomb {
    pub vel: glm::Vec3,
}
impl Component for Bomb {
    fn init(&mut self, t: Transform, sys: &mut Sys) {
        self.t = t;
    }
    fn update(&mut self, sys: &System) {
        let pos = sys.trans.get_position(self.t);
        let vel = self.vel;
        let dt = sys.input.time.dt.min(1. / 20.);
        // let dt = 1. / 100.;
        // let dir = vel * (1.0 / 100.0);
        let ray = rapier3d::prelude::Ray {
            origin: point![pos.x, pos.y, pos.z],
            dir: vel,
        };
        if let Some((_handle, _hit)) = sys.physics.query_pipeline.cast_ray_and_get_normal(
            &&sys.physics.collider_set,
            &ray,
            dt,
            true,
            InteractionGroups::all(),
            None,
        ) {
            let g = GameObject { t: self.t };
            sys.defer.append(move |world| {
                world.delete(g);
            });
            // self.vel = glm::reflect_vec(&vel, &&_hit.normal);
        }
        // if pos.y <= 0. {
        //     self.vel = glm::reflect_vec(&vel, &glm::vec3(0.,1.,0.));
        //     pos.y += 0.1;
        //     trans.set_position(self.t, pos);
        // }
        // sys.trans.rotate(self.t, &glm::Vec3::y(), 3.0 * dt);
        sys.trans._move(self.t, self.vel * dt);
        self.vel += glm::vec3(0.0, -9.81, 0.0) * dt;

        // *pos += vel * (1.0 / 60.0);
    }
}
#[component]
pub struct Maker {}
impl Component for Maker {
    fn init(&mut self, t: Transform, sys: &mut Sys) {
        self.t = t;
    }
    fn update(&mut self, sys: &System) {
        sys.defer.append(|world| {
            let g = world.instantiate();
            world.add_component(
                g,
                Bomb {
                    t: Transform(-1),
                    vel: glm::vec3(rand::random(), rand::random(), rand::random()),
                },
            );
            // world.get_component::<Bomb, _>(g, |b| {
            //     if let Some(b) = b {
            //         *b.vel = glm::vec3(rand::random(), rand::random(), rand::random());
            //     }
            // });
        });
    }
}

pub fn game_thread_fn(
    device: Arc<Device>,
    queue: Arc<vulkano::device::Queue>,
    model_manager: Arc<Mutex<ModelManager>>,
    renderer_manager: Arc<RwLock<RendererManager>>,
    // texture_manager: Arc<TextureManager>,
    coms: (
        Sender<(
            Arc<(
                usize,
                Vec<Arc<(Vec<Vec<i32>>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>)>>,
            )>,
            glm::Vec3,
            glm::Quat,
            RendererData,
            // Arc<(Vec<Offset>, Vec<Id>)>,
            // Arc<&HashMap<i32, HashMap<i32, Mesh>>>,
        )>,
        Receiver<Input>,
        // Sender<Terrain>,
    ),
    running: Arc<AtomicBool>,
) {
    let mut physics = Physics::new();
    // rayon::ThreadPoolBuilder::new().num_threads(num_cpus::get() - 1).build_global().unwrap();
    /* Create the ground. */
    // let collider = ColliderBuilder::cuboid(100.0, 0.1, 100.0).build();
    // physics.collider_set.insert(collider);

    /* Create the bounding ball. */
    // let rigid_body = RigidBodyBuilder::dynamic()
    //     .translation(vector![0.0, 10.0, 0.0])
    //     .build();
    // let collider = ColliderBuilder::ball(0.5).restitution(0.7).build();
    // let ball_body_handle = physics.rigid_body_set.insert(rigid_body);
    // physics.collider_set.insert_with_parent(
    //     collider,
    //     ball_body_handle,
    //     &mut physics.rigid_body_set,
    // );

    /* Create other structures necessary for the simulation. */
    let gravity = vector![0.0, -9.81, 0.0];

    let lazy_maker = LazyMaker::new();
    let mut world = World::new(model_manager, renderer_manager.clone(), physics);
    world.register::<Bomb>(true);
    world.register::<Maker>(true);
    world.register::<Renderer>(false);
    world.register::<Terrain>(true);

    // let _root = world.instantiate();

    // let sys = System {trans: &world.transforms.read(), physics: &&physics, defer: &lazy_maker, input: &&input};
    // use rand::Rng;

    let ter = world.instantiate();

    world.add_component(
        ter,
        Terrain {
            chunks: Arc::new(Mutex::new(HashMap::new())),
            terrain_size: 33,
            chunk_range: 9,
            ..Default::default()
        },
    );

    {
        // let mut renderer_manager = world.renderer_manager.lock();
        for _ in 0..1  {
            // bombs
            let g = world.instantiate();
            world.add_component(
                g,
                Bomb {
                    t: Transform(-1),
                    vel: glm::vec3(
                        rand::random::<f32>() - 0.5,
                        rand::random::<f32>() - 0.5,
                        rand::random::<f32>() - 0.5,
                    ) * 5.0,
                },
            );
            world.add_component(g, Renderer::new(g.t, 0));
            world.transforms.read()._move(
                g.t,
                glm::vec3(
                    rand::random::<f32>() * 100. - 50.,
                    50.0 + rand::random::<f32>() * 100.,
                    rand::random::<f32>() * 100. - 50.,
                ),
            );
        }
    }
    // {
    //     // maker
    //     let g = world.instantiate();
    //     world.add_component(g, Maker { t: Transform(-1) });
    // }

    ////////////////////////////////////////////////
    let mut cam_pos = glm::vec3(0.0 as f32, 0.0, -1.0);
    let mut cam_rot = glm::quat(1.0, 0.0, 0.0, 0.0);

    let mut perf = Perf {
        data: HashMap::new(),
    };

    while running.load(Ordering::SeqCst) {
        // println!("waiting for input");
        let input = coms.1.recv().unwrap();
        // println!("input recvd");

        {
            puffin::profile_scope!("game loop");
            let inst = Instant::now();
            {
                puffin::profile_scope!("world update");
                world.sys.physics.step(&gravity);
                world.update(&lazy_maker, &input);
            }

            let speed = 20.0 * input.time.dt;
            if !input.get_key(&VirtualKeyCode::LControl) {
                // forward/backward
                if input.get_key(&VirtualKeyCode::W) {
                    cam_pos +=
                        (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * -speed;
                }
                if input.get_key(&VirtualKeyCode::S) {
                    cam_pos +=
                        (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * speed;
                }
                //left/right
                if input.get_key(&VirtualKeyCode::A) {
                    cam_pos +=
                        (glm::quat_to_mat4(&cam_rot) * vec4(1.0, 0.0, 0.0, 1.0)).xyz() * -speed;
                }
                if input.get_key(&VirtualKeyCode::D) {
                    cam_pos +=
                        (glm::quat_to_mat4(&cam_rot) * vec4(1.0, 0.0, 0.0, 1.0)).xyz() * speed;
                }
                // up/down
                if input.get_key(&VirtualKeyCode::Space) {
                    cam_pos +=
                        (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 1.0, 0.0, 1.0)).xyz() * -speed;
                }
                if input.get_key(&VirtualKeyCode::LShift) {
                    cam_pos +=
                        (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 1.0, 0.0, 1.0)).xyz() * speed;
                }

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
                // const ALOT: f32 = 10_000_000. / 60.;
                if input.get_mouse_button(&0) {
                    let _cam_rot = cam_rot.clone();
                    let _cam_pos = cam_pos.clone();
                    // let rm = renderer_manager.clone();
                    lazy_maker.append(move |world| {
                        let len = 1;//(ALOT * input.time.dt.min(1.0 / 30.0)) as usize;
                        // let chunk_size =  (len / (64 * 64)).max(1);
                        (0..len)
                            .into_iter()
                            // .chunks(chunk_size)
                            .for_each(|_| {
                                let g = world.instantiate_with_transform(_Transform {
                                    position: _cam_pos
                                        + glm::quat_to_mat3(&_cam_rot)
                                            * (glm::Vec3::y() * 4. - glm::Vec3::z() * 6.),
                                    ..Default::default()
                                });
                                world.add_component(
                                    g,
                                    Bomb {
                                        t: Transform(-1),
                                        vel: glm::quat_to_mat3(&_cam_rot) * -glm::Vec3::z() * 50.
                                            + glm::vec3(
                                                rand::random(),
                                                rand::random(),
                                                rand::random(),
                                            ) * 18.,
                                    },
                                );
                                world.add_component(g, Renderer::new(g.t, 0));
                            });
                    });
                }
            }

            {
                puffin::profile_scope!("defered");
                lazy_maker.do_defered(&mut world);
            }

            perf.update("world".into(), Instant::now() - inst);
        }

        // let positions_to_buffer = {
        //     puffin::profile_scope!("prepare transforms");

        //     let pos_len = { world.transforms.read().positions.len() };
        //     let mut positions_to_buffer: Vec<[f32;3]> = { Vec::with_capacity(pos_len) };

        //     unsafe {
        //         positions_to_buffer.set_len(pos_len);
        //     }
        //     {
        //         let positions = &mut world.transforms.write().positions;
        //         let p_iter = positions.par_iter_mut();
        //         let m_iter = positions_to_buffer.par_iter_mut();

        //         let chunk_size = (pos_len / (64 * 64)).max(1);
        //         p_iter.zip_eq(m_iter).chunks(chunk_size).for_each(|slice| {
        //             for (x, y) in slice {
        //                 let x = x.get_mut();
        //                 // let x = x.lock();
        //                 *y = [x.x, x.y, x.z];
        //             }
        //         });
        //     }

        //     Arc::new(positions_to_buffer)
        // };
        let inst = Instant::now();
        let positions_to_buffer = {
            puffin::profile_scope!("get transform data");
            world.transforms.write().get_transform_data_updates()
        };
        perf.update("get transform data".into(), Instant::now() - inst);

        let mut rm = renderer_manager.write();
        // let a = rm.model_indirect.read();
        // let b = a.deref();
        let inst = Instant::now();
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
                    vec![id.clone(), t.indirect_id.clone(), t.transform_id.clone()].into_iter()
                })
                .collect(),
            transforms_len: rm.transforms.data.len() as i32,
        };
        rm.updates.clear();
        perf.update("get renderer data".into(), Instant::now() - inst);

        // std::thread::sleep(Duration::from_millis(5));
        let inst = Instant::now();

        let res = coms.0.send((
            positions_to_buffer.clone(),
            cam_pos.clone(),
            cam_rot.clone(),
            renderer_data,
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
