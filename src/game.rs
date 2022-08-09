use component_derive::component;
use glm::{vec4, Vec3};
use nalgebra_glm as glm;
use parking_lot::Mutex;
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
use vulkano::{device::Device, buffer::CpuAccessibleBuffer};
use winit::event::VirtualKeyCode;
// use rapier3d::{na::point, prelude::InteractionGroups};

use crate::{
    engine::{
        physics::{self, Physics},
        transform::{Transform, Transforms, _Transform},
        Component, LazyMaker, System, World,
    },
    input::Input,
    model::ModelManager,
    renderer::{ModelMat, Id},
    renderer_component::{Renderer, RendererManager, Offset},
    terrain::Terrain,
    texture::TextureManager,
};

#[component]
pub struct Bomb {
    pub vel: glm::Vec3,
}
impl Component for Bomb {
    fn init(&mut self, t: Transform) {
        self.t = t;
    }
    fn update(&mut self, sys: &System) {
        let pos = sys.trans.get_position(self.t);
        let vel = self.vel;
        // let dt = sys.2.time.dt.min(1./100.);
        let dt = 1. / 100.;
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
            self.vel = glm::reflect_vec(&vel, &&_hit.normal);
        }
        // if pos.y <= 0. {
        //     self.vel = glm::reflect_vec(&vel, &glm::vec3(0.,1.,0.));
        //     pos.y += 0.1;
        //     trans.set_position(self.t, pos);
        // }
        sys.trans._move(self.t, self.vel * dt);
        self.vel += glm::vec3(0.0, -9.81, 0.0) * dt;

        // *pos += vel * (1.0 / 60.0);
    }
}
#[component]
pub struct Maker {}
impl Component for Maker {
    fn init(&mut self, t: Transform) {
        self.t = t;
    }
    fn update(&mut self, sys: &System) {
        sys.defer.append(|world, _rm, _mm, ph| {
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
    // texture_manager: Arc<TextureManager>,
    coms: (
        Sender<(
            Arc<Vec<ModelMat>>,
            glm::Vec3,
            glm::Quat,
            Arc<(Vec<Offset>, Vec<Id>)>,
            // Arc<&HashMap<i32, HashMap<i32, Mesh>>>,
        )>,
        Receiver<Input>,
        // Sender<Terrain>,
    ),
    running: Arc<AtomicBool>,
) {
    let mut physics = Physics::new();

    /* Create the ground. */
    // let collider = ColliderBuilder::cuboid(100.0, 0.1, 100.0).build();
    // physics.collider_set.insert(collider);

    /* Create the bounding ball. */
    let rigid_body = RigidBodyBuilder::dynamic()
        .translation(vector![0.0, 10.0, 0.0])
        .build();
    let collider = ColliderBuilder::ball(0.5).restitution(0.7).build();
    let ball_body_handle = physics.rigid_body_set.insert(rigid_body);
    physics.collider_set.insert_with_parent(
        collider,
        ball_body_handle,
        &mut physics.rigid_body_set,
    );

    /* Create other structures necessary for the simulation. */
    let gravity = vector![0.0, -9.81, 0.0];

    let lazy_maker = LazyMaker::new();
    let mut renderer_manager = Mutex::new(RendererManager {
        ..Default::default()
    });
    let mut world = World::new();
    world.register::<Bomb>();
    world.register::<Maker>();
    world.register::<Renderer>();
    world.register::<Terrain>();

    let _root = world.instantiate();

    // let sys = System {trans: &world.transforms.read(), physics: &&physics, defer: &lazy_maker, input: &&input};
    // use rand::Rng;
    {
        let mut renderer_manager = renderer_manager.lock();
        for _ in 0..1_000_000 {
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
        world.add_component(g, Renderer::new(g.t, 0, &mut renderer_manager));
        world.transforms.read()._move(
            g.t,
            glm::vec3(
                rand::random::<f32>() * 100. - 50.,
                50.0,
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
    let ter = world.instantiate();

    world.add_component(ter, Terrain {
        chunks: Arc::new(Mutex::new(HashMap::new())),
        terrain_size: 33,
        ..Default::default()
    });

    ////////////////////////////////////////////////
    let mut cam_pos = glm::vec3(0.0 as f32, 0.0, -1.0);
    let mut cam_rot = glm::quat(1.0, 0.0, 0.0, 0.0);

    // let mut input = Input {
    //     ..Default::default()
    // };

    let mut perf = HashMap::<String, Duration>::new();

    let mut update_perf = |k: String, v: Duration| {
        if let Some(dur) = perf.get_mut(&k) {
            *dur += v;
        } else {
            perf.insert(k, v);
        }
    };
    let mut loops = 0;
    while running.load(Ordering::SeqCst) {
        loops += 1;
        // println!("waiting for input");
        let input = coms.1.recv().unwrap();
        // println!("input recvd");

        let speed = 20.0 * input.time.dt;
        // forward/backward
        if input.get_key(&VirtualKeyCode::W) {
            cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * -speed;
        }
        if input.get_key(&VirtualKeyCode::S) {
            cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * speed;
        }
        //left/right
        if input.get_key(&VirtualKeyCode::A) {
            cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(1.0, 0.0, 0.0, 1.0)).xyz() * -speed;
        }
        if input.get_key(&VirtualKeyCode::D) {
            cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(1.0, 0.0, 0.0, 1.0)).xyz() * speed;
        }
        // up/down
        if input.get_key(&VirtualKeyCode::Space) {
            cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 1.0, 0.0, 1.0)).xyz() * -speed;
        }
        if input.get_key(&VirtualKeyCode::LShift) {
            cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 1.0, 0.0, 1.0)).xyz() * speed;
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

        let inst = Instant::now();
        physics.step(&gravity);
        world.update(&physics, &lazy_maker, &input, &model_manager, &renderer_manager);

        const ALOT: f32 = 10_000_000. / 60.;

        if input.get_mouse_button(&0) {
            let _cam_rot = cam_rot.clone();
            let _cam_pos = cam_pos.clone();
            // let rm = renderer_manager.clone();
            lazy_maker.append(move |world, rm, mm, ph| {
                let len = (ALOT * input.time.dt) as usize;
                // let chunk_size =  (len / (64 * 64)).max(1);
                (0..len)
                    .into_iter()
                    // .chunks(chunk_size)
                    .for_each(|_| {
                        let g = world.instantiate_with_transform(_Transform {
                            position: _cam_pos - glm::Vec3::y() * 2.,
                            ..Default::default()
                        });
                        world.add_component(
                            g,
                            Bomb {
                                t: Transform(-1),
                                vel: glm::quat_to_mat3(&_cam_rot) * -glm::Vec3::z() * 50.
                                    + glm::vec3(rand::random(), rand::random(), rand::random())
                                        * 18.,
                            },
                        );
                        world.add_component(g, Renderer::new(g.t, 0, rm));
                        // world.add_component(g, d)
                        // world
                        //     .transforms
                        //     .read()
                        //     .set_position(g.t, _cam_pos - glm::Vec3::y() * 2.);
                    });
            });
        }

        {
            lazy_maker.init(&mut world, &mut renderer_manager.lock(), &mut model_manager.lock(), &mut physics);
        }
            
        update_perf("world".into(), Instant::now() - inst);

        // dur += Instant::now() - inst;

        let inst = Instant::now();

        let positions = &world.transforms.read().positions;

        let mut cube_models: Vec<ModelMat> = Vec::with_capacity(positions.len());

        // cube_models.reserve(positions.len());
        unsafe {
            cube_models.set_len(positions.len());
        }
        let p_iter = positions.par_iter();
        let m_iter = cube_models.par_iter_mut();

        let chunk_size = (positions.len() / (64 * 64)).max(1);
        p_iter.zip_eq(m_iter).chunks(chunk_size).for_each(|slice| {
            for (x, y) in slice {
                let x = x.lock();
                *y = ModelMat {
                    pos: [x.x, x.y, x.z],
                    ..Default::default()
                };
            }
        });
        {
            let mut mm = model_manager.lock();
            if let Some(id) = mm.models.get("src/cube/cube.obj") {
                let id = id.clone();
                if let Some(mr) = mm.models_ids.get_mut(&id) {
                    mr.count = cube_models.len() as u32;
                }
            }
        }

        update_perf("get cube models".into(), Instant::now() - inst);
        let cube_models = Arc::new(cube_models);
        // let terr_chunks = Arc::new(&ter.chunks);
        // println!("sending models");
        // let render_data = { let x = renderer_manager.lock().getInstances(device.clone()).clone() };
        let inst = Instant::now();
        let render_data = renderer_manager.lock().get_instances();
        update_perf("get instances".into(), Instant::now() - inst);
        let res = coms.0.send((
            cube_models.clone(),
            cam_pos.clone(),
            cam_rot.clone(),
            render_data,
            // terr_chunks,
        ));
        if res.is_err() {
            println!("ohno");
        }
    }
    let p = perf.iter();
    for (k, x) in p {
        println!("{}: {:?}", k, (*x / loops));
    }
}
