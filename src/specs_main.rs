
// use physx_sys::*;
use rand::prelude::*;

use rayon::iter::ParallelIterator;

use specs::{prelude::*, storage::HashMapStorage, WorldExt};

use rapier3d::prelude::*;

const TAU: f32 = 2. * std::f32::consts::PI;

#[derive(Debug)]
struct ClusterBomb {
    fuse: usize,
}
impl Component for ClusterBomb {
    // This uses `HashMapStorage`, because only some entities are cluster bombs.
    type Storage = HashMapStorage<Self>;
}

#[derive(Debug)]
struct Shrapnel {
    durability: usize,
}
impl Component for Shrapnel {
    // This uses `HashMapStorage`, because only some entities are shrapnels.
    type Storage = HashMapStorage<Self>;
}

#[derive(Debug, Clone)]
struct Pos(f32, f32);
impl Component for Pos {
    // This uses `VecStorage`, because all entities have a position.
    type Storage = VecStorage<Self>;
}


#[derive(Debug, Clone)]
struct Hit(bool);
impl Component for Hit {
    // This uses `VecStorage`, because all entities have a position.
    type Storage = VecStorage<Self>;
}


#[derive(Debug)]
struct Vel(f32, f32);
impl Component for Vel {
    // This uses `DenseVecStorage`, because nearly all entities have a velocity.
    type Storage = DenseVecStorage<Self>;
}

struct ClusterBombSystem;
impl<'a> System<'a> for ClusterBombSystem {
    type SystemData = (
        Entities<'a>,
        WriteStorage<'a, ClusterBomb>,
        ReadStorage<'a, Pos>,
        // Allows lazily adding and removing components to entities
        // or executing arbitrary code with world access lazily via `execute`.
        Read<'a, LazyUpdate>,
    );

    fn run(&mut self, (entities, mut bombs, positions, updater): Self::SystemData) {
        use rand::distributions::Uniform;

        let durability_range = Uniform::new(10, 20);
        let update_position = |(entity, bomb, position): (Entity, &mut ClusterBomb, &Pos)| {
            let mut rng = rand::thread_rng();

            if bomb.fuse == 0 {
                let _ = entities.delete(entity);
                for _ in 0..1_000_000 {
                    let shrapnel = entities.create();
                    updater.insert(
                        shrapnel,
                        Shrapnel {
                            durability: durability_range.sample(&mut rng),
                        },
                    );
                    updater.insert(shrapnel, position.clone());
                    let angle: f32 = rng.gen::<f32>() * TAU;
                    updater.insert(shrapnel, Vel(angle.sin() * 10f32, angle.cos() * 10f32));
                    updater.insert(shrapnel,Hit(false));
                }
            } else {
                bomb.fuse -= 1;
            }
        };

        // Join components in potentially parallel way using rayon.
        {
            (&entities, &mut bombs, &positions)
                .par_join()
                .for_each(update_position);
        }
    }
}

struct PhysicsSystem;
impl<'a> System<'a> for PhysicsSystem {
    type SystemData = (WriteStorage<'a, Pos>, WriteStorage<'a, Vel>, );

    fn run(&mut self, (mut pos, mut vel): Self::SystemData) {
        (&mut pos, &mut vel).par_join().for_each(|(pos, vel)| {
            pos.0 += vel.0;
            pos.1 += vel.1;

            vel.1 += (1.0 / 60.0) * -9.81;
        });
    }
}

struct PhysicsSystem2 {
    rigid_body_set: RigidBodySet,
    collider_set: ColliderSet,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
    physics_hooks: (),
    event_handler: (),

}

impl<'a> System<'a> for PhysicsSystem2 {
    type SystemData = (WriteStorage<'a, Pos>, WriteStorage<'a, Vel>, WriteStorage<'a, Hit>);

    fn run(&mut self, (mut pos, mut vel, mut hit): Self::SystemData) {
        let gravity = vector![0.0, -9.81, 0.0];
        self.physics_pipeline.step(
            &gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            &mut self.ccd_solver,
            &self.physics_hooks,
            &self.event_handler,
          );
          self.query_pipeline.update(&self.island_manager, &self.rigid_body_set, &self.collider_set);
      
        //   let ray = rapier3d::prelude::Ray {origin: point![0.0,2.0,0.0], dir: vector![0.0,1.0,0.0]};
        //   if let Some((handle, toi)) = self.query_pipeline.cast_ray(&self.collider_set, &ray, 10.0, false, InteractionGroups::all(),None) {
        //     let hit_point = ray.point_at(toi); // Same as: `ray.origin + ray.dir * toi`
        //     println!("Collider {:?} hit at point {}", handle, hit_point);
        //   }
      
      
        //   let ball_body = &self.rigid_body_set[ball_body_handle];
        //   println!(
        //     "Ball altitude: {}",
        //     ball_body.translation().y
        //   );



        (&mut pos, &mut vel, &mut hit).par_join().for_each(|(pos, vel, hit)| {
            pos.0 += vel.0;
            pos.1 += vel.1;

            vel.1 += (1.0 / 60.0) * -9.81;

        let ray = rapier3d::prelude::Ray {origin: point![pos.0,pos.1,0.0], dir: vector![vel.0,vel.1,0.0]};
        if let Some((handle, toi)) = self.query_pipeline.cast_ray(&self.collider_set, &ray, 10.0, false, InteractionGroups::all(),None) {
            let hit_point = ray.point_at(toi); // Same as: `ray.origin + ray.dir * toi`
            hit.0 = true;
            // println!("Collider {:?} hit at point {}", handle, hit_point);
        }
        
        });
    }
}

struct ShrapnelSystem;
impl<'a> System<'a> for ShrapnelSystem {
    type SystemData = (Entities<'a>, WriteStorage<'a, Shrapnel>);

    fn run(&mut self, (entities, mut shrapnels): Self::SystemData) {
        (&entities, &mut shrapnels)
            .par_join()
            .for_each(|(entity, shrapnel)| {
                if shrapnel.durability == 0 {
                    let _ = entities.delete(entity);
                } else {
                    // shrapnel.durability -= 1;
                }
            });
    }
}

fn main() {

    ////////////////////////

    let mut rigid_body_set = RigidBodySet::new();
    let mut collider_set = ColliderSet::new();
  
    // let collider = ColliderBuilder::trimesh(vertices, indices);
    
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

    // ///////////////////////////////////////////////////



    let mut world = World::new();

    let mut dispatcher = DispatcherBuilder::new()
        // .with(PhysicsSystem, "physics", &[])
        .with(ClusterBombSystem, "cluster_bombs", &[])
        .with(ShrapnelSystem, "shrapnels", &[])
        .with(PhysicsSystem2 {rigid_body_set: rigid_body_set,
            collider_set: collider_set,
            integration_parameters: IntegrationParameters::default(),
            physics_pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: BroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            impulse_joint_set: ImpulseJointSet::new(),
            multibody_joint_set: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            physics_hooks: (),
            event_handler: ()
        }, "physics2", &[])
        .build();

    dispatcher.setup(&mut world);

    world
        .create_entity()
        .with(Pos(0., 0.))
        .with(ClusterBomb { fuse: 0 })
        .build();

    let mut step = 0;


    dispatcher.dispatch(&world);

    // Maintain dynamically added and removed entities in dispatch.
    // This is what actually executes changes done by `LazyUpdate`.
    world.maintain();

    use std::time::Instant;
    let now = Instant::now();
    loop {
        step += 1;
        // let mut entities = 0;
        {

            // // Simple console rendering
            // let positions = world.read_storage::<Pos>();
            // const WIDTH: usize = 10;
            // const HEIGHT: usize = 10;
            // const SCALE: f32 = 1. / 4.;
            // let mut screen = [[0; WIDTH]; HEIGHT];
            // for entity in world.entities().join() {
            //     if let Some(pos) = positions.get(entity) {
            //         let x = (pos.0 * SCALE + WIDTH as f32 / 2.).floor() as usize;
            //         let y = (pos.1 * SCALE + HEIGHT as f32 / 2.).floor() as usize;
            //         if x < WIDTH && y < HEIGHT {
            //             screen[y][x] += 1;
            //         }
            //     }
            //     entities += 1;
            // }
            // println!("Step: {}, Entities: {}", step, entities);
            // for row in &screen {
            //     for cell in row {
            //         print!("{} ", cell);
            //     }
            //     println!();
            // }
            // println!();
        }
        if step >= 1000 {
            break;
        }

        dispatcher.dispatch(&world);

        // Maintain dynamically added and removed entities in dispatch.
        // This is what actually executes changes done by `LazyUpdate`.
        world.maintain();
    }
    let elapsed = now.elapsed() / 1000;
    println!("Elapsed: {:.2?}", elapsed);
}