use std::mem::transmute;

// use component_derive::ComponentID;
use super::collider::_ColliderType;
use crate::engine::prelude::*;
use crate::engine::project::asset_manager::AssetInstance;
use force_send_sync::SendSync;
use nalgebra_glm::{quat_euler_angles, vec3, Quat, Vec3};
use rapier3d::{na::UnitQuaternion, prelude::*};
use serde::{Deserialize, Serialize};

#[derive(ComponentID, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct _RigidBody {
    pub _type: _ColliderType,
    // #[serde(skip_serializing, skip_deserializing)]
    // handle: ColliderHandle,
    #[serde(skip_serializing, skip_deserializing)]
    rb_handle: RigidBodyHandle,
}
impl Component for _RigidBody {
    fn init(&mut self, _transform: &Transform, _id: i32, _sys: &Sys) {
        _sys.new_rigid_bodies.push((
            self._type.clone(),
            _transform.get_position(),
            _transform.get_rotation(),
            unsafe { SendSync::new(&mut self.rb_handle) },
        ));
    }
    fn deinit(&mut self, _transform: &Transform, _id: i32, _sys: &Sys) {
        if self.rb_handle != RigidBodyHandle::invalid() {
            _sys.physics.lock().remove_rigid_body(self.rb_handle);
        }
    }
    fn late_update(&mut self, _transform: &Transform, _sys: &System) {
        if self.rb_handle != RigidBodyHandle::invalid() {
            let rb = unsafe { _sys.physics.get_rigid_body(self.rb_handle).unwrap() };
            if !rb.is_sleeping() {
                _transform.set_position(rb.translation());
                _transform.set_rotation(rb.rotation());
            }
        } else {
            _sys.new_rigid_bodies.push((
                self._type.clone(),
                _transform.get_position(),
                _transform.get_rotation(),
                unsafe { SendSync::new(&mut self.rb_handle) },
            ));
        }
    }
    fn inspect(&mut self, transform: &Transform, id: i32, ui: &mut egui::Ui, sys: &Sys) {
        ui.menu_button("shape", |ui| {
            if ui.button("Cuboid").clicked() {
                self._type = _ColliderType::Cuboid(Vec3::default());
            }
            if ui.button("Ball").clicked() {
                self._type = _ColliderType::Ball(1.0);
            }
        });

        if match &mut self._type {
            _ColliderType::Cuboid(v) => Ins(v).inspect("Dimensions", ui, sys),
            _ColliderType::Ball(f) => Ins(f).inspect("radius", ui, sys),
            _ColliderType::TriMesh((_,_)) => false,
        } {
            // let phys = sys.physics.lock();
            // phys.get_collider(self.handle).unwrap().set_shape(shape)
            self.deinit(transform, id, sys);
            self.init(transform, id, sys);
        }
        if self.rb_handle != RigidBodyHandle::invalid() {

        let mut phys = sys.physics.lock();
        let rb = unsafe { phys.get_rigid_body_mut(self.rb_handle).unwrap() };
        rb.set_translation(transform.get_position().into(), false);
        rb.set_rotation(
            UnitQuaternion::new_unchecked(transform.get_rotation()),
            false,
        );
        }
    }
}
impl _RigidBody {
    pub fn new(_type: _ColliderType) -> Self {
        Self {
            _type,
            ..Default::default()
        }
    }
    pub fn new_w_vel(_type: _ColliderType, velocity: Vec3) -> Self {
        Self {
            _type,
            ..Default::default()
        }
    }
}
