use std::{time::{Instant, Duration}, sync::{Arc, mpsc::{Sender, Receiver}, atomic::{AtomicBool, Ordering}}, collections::HashMap};
use component_derive::component;
use nalgebra_glm as glm;
use glm::{ vec4, Vec3, };
use rapier3d::prelude::*;
use vulkano::device::Device;
use winit::event::VirtualKeyCode;
use rayon::prelude::*;
// use rapier3d::{na::point, prelude::InteractionGroups};

use crate::{renderer::ModelMat, engine::{transform::{Transform, Transforms}, Component, LazyMaker, physics::{self, Physics}, World}, input::Input, texture::TextureManager, terrain::Terrain};

#[component]
pub struct Bomb {
    pub vel: glm::Vec3,
}
impl Component for Bomb {
    fn init(&mut self, t: Transform) {
        self.t = t;
    }
    fn update(&mut self, trans: &Transforms, sys: (&physics::Physics, &LazyMaker, &Input)) {
        let pos = trans.get_position(self.t);
        let vel = self.vel;
        // let dt = sys.2.time.dt.min(1./100.);
        let dt = 1./100.;
        // let dir = vel * (1.0 / 100.0);
        let ray = rapier3d::prelude::Ray {
            origin: point![pos.x, pos.y, pos.z],
            dir: vel,
        };
        if let Some((_handle, _hit)) = sys.0.query_pipeline.cast_ray_and_get_normal(
            &&sys.0.collider_set,
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
        trans._move(self.t, self.vel * dt);
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
    fn update(&mut self, trans: &Transforms, sys: (&physics::Physics, &LazyMaker, &Input)) {
        sys.1.append(|world| {
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
    texture_manager: Arc<TextureManager>,
    coms: (
        Sender<(
            Arc<Vec<ModelMat>>,
            Arc<Vec<ModelMat>>,
            glm::Vec3,
            glm::Quat,
            // Arc<&HashMap<i32, HashMap<i32, Mesh>>>,
        )>,
        Receiver<Input>,
        Sender<Terrain>,
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

    let mut world = World::new();
    world.register::<Bomb>();
    world.register::<Maker>();

    let _root = world.instantiate();

    // use rand::Rng;
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
        world.transforms.read()._move(
            g.t,
            glm::vec3(
                rand::random::<f32>() * 100. - 50.,
                50.0,
                rand::random::<f32>() * 100. - 50.,
            ),
        );
    }
    // {
    //     // maker
    //     let g = world.instantiate();
    //     world.add_component(g, Maker { t: Transform(-1) });
    // }
    let lazy_maker = LazyMaker::new();

    let mut ter = Terrain {
        chunks: HashMap::new(),
        device: device.clone(),
        texture_manager: texture_manager.clone(),
        // queue: queue.clone(),
        terrain_size: 33,
    };
    ter.generate(&mut physics.collider_set);

    let res = coms.2.send(ter.clone());
    if res.is_err() {
        // println!("ohno");
    }

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
        world.update(&physics, &lazy_maker, &input);

        const ALOT: f32 = 10_000_000. / 60.;

        if input.get_mouse_button(&0) {
            let _cam_rot = cam_rot.clone();
            let _cam_pos = cam_pos.clone();
            lazy_maker.append(move |world| {
                let len = (ALOT * input.time.dt) as usize;
                // let chunk_size =  (len / (64 * 64)).max(1);
                (0..len)
                    .into_iter()
                    // .chunks(chunk_size)
                    .for_each(|_| {
                        let g = world.instantiate();
                        world.add_component(
                            g,
                            Bomb {
                                t: Transform(-1),
                                vel: glm::quat_to_mat3(&_cam_rot) * -glm::Vec3::z() * 50.
                                    + glm::vec3(rand::random(), rand::random(), rand::random())
                                        * 18.,
                            },
                        );
                        world
                            .transforms
                            .read()
                            .set_position(g.t, _cam_pos - glm::Vec3::y() * 2.);
                    });
            });
        }

        lazy_maker.init(&mut world);

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

        let chunk_size = (positions.len()  / (64 * 64)).max(1);
        p_iter.zip_eq(m_iter).chunks(chunk_size).for_each(|slice| {
            for (x, y) in slice {
                let x = x.lock();
                *y = ModelMat {
                    pos: [x.x, x.y, x.z],
                    ..Default::default()
                };
            }
        });

        update_perf("get cube models".into(), Instant::now() - inst);
        let terrain_models: Vec<ModelMat> = vec![ModelMat {
            pos: glm::vec3(0., 0., 0.).into(),
            ..Default::default()
        }];
        let terrain_models = Arc::new(terrain_models);
        let cube_models = Arc::new(cube_models);
        // let terr_chunks = Arc::new(&ter.chunks);
        // println!("sending models");
        let res = coms.0.send((
            terrain_models.clone(),
            cube_models.clone(),
            cam_pos.clone(),
            cam_rot.clone(),
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
