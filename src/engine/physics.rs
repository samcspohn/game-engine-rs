use std::time::Instant;

use num_integer::Roots;
use rapier3d::prelude::*;

use super::perf::Perf;

pub struct Physics {
    pub rigid_body_set: RigidBodySet,
    pub collider_set: ColliderSet,
    integration_parameters: IntegrationParameters,
    physics_pipeline: PhysicsPipeline,
    island_manager: IslandManager,
    broad_phase: BroadPhase,
    narrow_phase: NarrowPhase,
    impulse_joint_set: ImpulseJointSet,
    multibody_joint_set: MultibodyJointSet,
    ccd_solver: CCDSolver,
    pub query_pipeline: QueryPipeline,
    physics_hooks: (),
    event_handler: (),
    gravity: Vector<f32>,
}

impl Physics {
    pub fn new() -> Physics {
        Physics {
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
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
            event_handler: (),
            gravity: vector![0.0, -9.81, 0.0],
        }
    }
    pub fn step(&mut self, perf: &Perf) {
        let physics_step = perf.node("physics step");
        let len = self.rigid_body_set.len();
        let num_threads = (len / (num_cpus::get().sqrt())).max(1).min(num_cpus::get());
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_scoped(
                |thread| thread.run(),
                |pool| {
                    pool.install(|| {
                        // self.step(&self.gravity, &perf);

                        self.physics_pipeline.step(
                            &self.gravity,
                            &self.integration_parameters,
                            &mut self.island_manager,
                            &mut self.broad_phase,
                            &mut self.narrow_phase,
                            &mut self.rigid_body_set,
                            &mut self.collider_set,
                            &mut self.impulse_joint_set,
                            &mut self.multibody_joint_set,
                            &mut self.ccd_solver,
                            Some(&mut self.query_pipeline),
                            &self.physics_hooks,
                            &self.event_handler,
                        );
                    })
                },
            )
            .unwrap();

        drop(physics_step);

        // TODO: investigate if necessary
        let physics_update = perf.node("physics update");
        self.query_pipeline.update(
            // &self.island_manager,
            &self.rigid_body_set,
            &self.collider_set,
        );
    }
    pub fn get_counters() {}
    pub fn remove_collider(&mut self, handle: ColliderHandle) {
        if let Some(_) = self.collider_set.remove(
            handle,
            &mut self.island_manager,
            &mut self.rigid_body_set,
            true,
        ) {}
    }
    pub fn remove_rigid_body(&mut self, handle: RigidBodyHandle) {
        if let Some(_) = self.rigid_body_set.remove(
            handle,
            &mut self.island_manager,
            &mut self.collider_set,
            &mut self.impulse_joint_set,
            &mut self.multibody_joint_set,
            true,
        ) {}
    }
    pub fn add_collider_to_rigid_body(&mut self, collider: Collider, handle: RigidBodyHandle) {
        self.collider_set
            .insert_with_parent(collider, handle, &mut self.rigid_body_set);
    }
    pub fn clear(&mut self) {
        *self = Physics::new();
    }
}
