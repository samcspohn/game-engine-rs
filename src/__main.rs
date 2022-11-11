use glam::Vec3;
use std::{
    ops::DerefMut,
    sync::{atomic::AtomicI32, Arc},
};

use rand::prelude::*;
use rapier3d::prelude::*;
use rayon::prelude::*;
// use nalgebra::*;

use crossbeam::queue::SegQueue;
use parking_lot::{Mutex, RwLock};

// use specs::{prelude::*, storage::HashMapStorage, WorldExt};

// F: FnOnce(&mut World) + 'static,

#[derive(Clone, Copy)]
struct GameObject {
    t: i32,
}

// #[derive(Default)]
struct World {
    // entities: Vec<Option<Mutex<GameObject>>>,
    poss: RwLock<Vec<Mutex<glam::Vec3>>>,
    bombs: Vec<Option<Mutex<Bomb>>>,
    makers: Vec<Option<Mutex<Maker>>>,
}

impl World {
    fn new() -> World {
        World {
            // entities: Vec::new(),
            poss: RwLock::new(Vec::new()),
            bombs: Vec::new(),
            makers: Vec::new(),
        }
    }
    fn instantiate(&mut self) -> GameObject {
        let mut poss = self.poss.write();
        let ret = GameObject {
            t: poss.len() as i32,
        };
        // self.entities.push(Some(Mutex::new(ret.clone())));
        poss.push(Mutex::new(Vec3::new(0.0, 0.0, 0.0)));
        ret
    }
    fn add_component_bomb(&mut self, g: &GameObject, mut b: Bomb) {
        b.t = g.t;
        self.bombs.push(Some(Mutex::new(b)));
    }
    fn add_component_maker(&mut self, g: &GameObject, mut m: Maker) {
        m.t = g.t;
        self.makers.push(Some(Mutex::new(m)));
    }
}

struct LazyMaker {
    work: SegQueue<Box<dyn FnOnce(&mut World) + Send + Sync>>,
}

impl LazyMaker {
    fn append(&self, f: Box<dyn FnOnce(&mut World) + Send + Sync>) {
        self.work.push(f);
    }
    fn init(&self, wrld: &mut World) {
        while let Some(w) = self.work.pop() {
            w(wrld);
        }
    }
}

#[component]
struct Bomb {
    vel: glam::Vec3,
}
impl Bomb {
    fn update(&mut self, sys: (&Vec<Mutex<glam::Vec3>>, &ColliderSet, &QueryPipeline)) {
        let mut pos = sys.0[self.t as usize].lock();
        let mut vel = self.vel;
        let ray = rapier3d::prelude::Ray {
            origin: point![pos.x, pos.y, pos.z],
            dir: vector![vel.x, vel.y, vel.z],
        };
        if let Some((handle, toi)) =
            sys.2
                .cast_ray(&sys.1, &ray, 10.0, false, InteractionGroups::all(), None)
        {
            let hit_point = ray.point_at(toi); // Same as: `ray.origin + ray.dir * toi`
            pos.x = hit_point.x;
            pos.y = hit_point.y;
            pos.z = hit_point.z;
            // *b = true;
            // println!("Collider {:?} hit at point {}", handle, hit_point);
            vel.y = -vel.y;
        }
        vel += glam::vec3(0.0, -9.81, 0.0) * 1.0 / 60.0;
        *pos += vel * (1.0 / 60.0);
    }
}

#[component]
struct Maker {}

impl Maker {
    fn update(
        &mut self,
        sys: (
            &Vec<Mutex<glam::Vec3>>,
            &ColliderSet,
            &QueryPipeline,
            &LazyMaker,
        ),
    ) {
        sys.3.append(Box::new(|world| {
            let g = world.instantiate();
            world.add_component_bomb(
                &g,
                Bomb {
                    t: -1,
                    vel: glam::vec3(rand::random(), rand::random(), rand::random()),
                },
            );
        }));
    }
}

fn main() {
    let mut rigid_body_set = RigidBodySet::new();
    let mut collider_set = ColliderSet::new();

    /* Create the ground. */
    let collider = ColliderBuilder::cuboid(100.0, 0.1, 100.0).build();
    collider_set.insert(collider);

    /* Create the bounding ball. */
    let rigid_body = RigidBodyBuilder::dynamic()
        .translation(vector![0.0, 10.0, 0.0])
        .build();
    let collider = ColliderBuilder::ball(0.5).restitution(0.7).build();
    let ball_body_handle = rigid_body_set.insert(rigid_body);
    collider_set.insert_with_parent(collider, ball_body_handle, &mut rigid_body_set);

    /* Create other structures necessary for the simulation. */
    let gravity = vector![0.0, -9.81, 0.0];
    let integration_parameters = IntegrationParameters::default();
    let mut physics_pipeline = PhysicsPipeline::new();
    let mut island_manager = IslandManager::new();
    let mut broad_phase = BroadPhase::new();
    let mut narrow_phase = NarrowPhase::new();
    let mut impulse_joint_set = ImpulseJointSet::new();
    let mut multibody_joint_set = MultibodyJointSet::new();
    let mut ccd_solver = CCDSolver::new();
    let mut query_pipeline = QueryPipeline::new();
    let physics_hooks = ();
    let event_handler = ();

    /* Run the game loop, stepping the simulation once per frame. */

    let mut world = World::new();

    for _ in 0..1_000_000 {
        // bombs
        let g = world.instantiate();
        world.add_component_bomb(
            &g,
            Bomb {
                t: -1,
                vel: glam::vec3(rand::random(), rand::random(), rand::random()),
            },
        );
    }
    // { // maker
    //     let g = world.instantiate();
    //     world.add_component_maker(&g, Maker { t: -1 });

    // }
    let lazy_maker = LazyMaker {
        work: SegQueue::new(),
    };

    use std::time::Instant;
    let now = Instant::now();
    for _ in 0..1000 {
        physics_pipeline.step(
            &gravity,
            &integration_parameters,
            &mut island_manager,
            &mut broad_phase,
            &mut narrow_phase,
            &mut rigid_body_set,
            &mut collider_set,
            &mut impulse_joint_set,
            &mut multibody_joint_set,
            &mut ccd_solver,
            &physics_hooks,
            &event_handler,
        );
        query_pipeline.update(&island_manager, &rigid_body_set, &collider_set);

        {
            let poss = world.poss.read();
            (0..world.bombs.len())
                .into_par_iter()
                .chunks(64 * 64)
                .for_each(|slice| {
                    for i in slice {
                        if let Some(d) = &world.bombs[i] {
                            d.lock().update((&poss, &collider_set, &query_pipeline));
                        }
                    }
                });

            (0..world.makers.len())
                .into_par_iter()
                .chunks(64 * 64)
                .for_each(|slice| {
                    for i in slice {
                        if let Some(m) = &world.makers[i] {
                            m.lock()
                                .update((&poss, &collider_set, &query_pipeline, &lazy_maker));
                        }
                    }
                });
        }

        lazy_maker.init(&mut world);
    }
    // println!("{:?}", vals);
    let elapsed = now.elapsed() / 1000;
    println!("Elapsed: {:.2?}", elapsed);
}
