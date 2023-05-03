use std::time::Instant;

use rapier3d::prelude::*;

use crate::perf::Perf;

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
        }
    }
    pub fn step(&mut self, gravity: &Vector<Real>, perf: &mut Perf) {
        let inst = Instant::now();
        self.physics_pipeline.step(
            gravity,
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
        perf.update("physics step".into(), Instant::now() - inst);
        
        // TODO: investigate if necessary
        let inst = Instant::now();
        self.query_pipeline.update(
            // &self.island_manager,
            &self.rigid_body_set,
            &self.collider_set,
        );
        perf.update("physics update".into(), Instant::now() - inst);
    }
    pub fn get_counters() {
        
    }
    pub fn remove_collider(&mut self, handle: ColliderHandle) {
        if let Some(_) = self.collider_set.remove(handle, &mut self.island_manager, &mut self.rigid_body_set, true) {}

    }
    pub fn clear(&mut self) {
        *self = Physics::new();
    }
}
