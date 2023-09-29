use std::mem::transmute;

// use component_derive::ComponentID;
use crate::engine::prelude::*;
use crate::engine::project::asset_manager::AssetInstance;
use nalgebra_glm::{vec3, Vec3, quat_euler_angles};
use rapier3d::{na::UnitQuaternion, prelude::*};
use serde::{Deserialize, Serialize};
#[derive(Clone, Deserialize, Serialize)]
enum ColliderType {
    Cuboid(Vec3),
    Ball(f32),
}
impl Default for ColliderType {
    fn default() -> Self {
        ColliderType::Cuboid(vec3(1., 1., 1.))
    }
}
#[derive(ComponentID, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct _Collider {
    _type: ColliderType,
    #[serde(skip_serializing, skip_deserializing)]
    handle: ColliderHandle,
}
impl Component for _Collider {
    fn init(&mut self, _transform: &Transform, _id: i32, _sys: &Sys) {
        let col = match self._type {
            ColliderType::Cuboid(v) => ColliderBuilder::cuboid(v.x, v.y, v.z),
            ColliderType::Ball(r) => ColliderBuilder::ball(r),
        }
        .position(_transform.get_position().into())
        .rotation(quat_euler_angles(&_transform.get_rotation()))
        .build();
        self.handle = _sys.physics.lock().add_collider(col);
    }
    fn deinit(&mut self, _transform: &Transform, _id: i32, _sys: &Sys) {
        _sys.physics.lock().remove_collider(self.handle);
    }
    fn late_update(&mut self, _transform: &Transform, _sys: &System) {
        let col = unsafe { _sys.physics.get_collider(self.handle).unwrap() };
        col.set_translation(_transform.get_position().into());
        col.set_rotation(UnitQuaternion::new_unchecked(_transform.get_rotation()));
    }
}
impl Inspectable for _Collider {
    fn inspect(&mut self, transform: &Transform, id: i32, ui: &mut egui::Ui, sys: &Sys) {
        ui.menu_button("shape", |ui| {
            if ui.button("Cuboid").clicked() {
                self._type = ColliderType::Cuboid(Vec3::default());
            }
            if ui.button("Ball").clicked() {
                self._type = ColliderType::Ball(1.0);
            }
        });

        if match &mut self._type {
            ColliderType::Cuboid(v) => Ins(v).inspect("Dimensions", ui, sys),
            ColliderType::Ball(f) => Ins(f).inspect("radius", ui, sys),
        } {
        // let phys = sys.physics.lock();
        // phys.get_collider(self.handle).unwrap().set_shape(shape)
            self.deinit(transform, id, sys);
            self.init(transform, id, sys);
        }
        let phys = sys.physics.lock();
        let col = unsafe { phys.get_collider(self.handle).unwrap() };
        col.set_translation(transform.get_position().into());
        col.set_rotation(UnitQuaternion::new_unchecked(transform.get_rotation()));
    }
}
impl _Collider {
    pub fn a() {
        let b = ColliderBuilder::ball(3.).build();
    }
}
