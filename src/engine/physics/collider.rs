use std::mem::transmute;

// use component_derive::ComponentID;
use crate::engine::prelude::*;
use crate::engine::project::asset_manager::AssetInstance;
use force_send_sync::SendSync;
use nalgebra_glm::{quat_euler_angles, vec3, Vec3};
use rapier3d::{
    na::{Point3, UnitQuaternion},
    prelude::*,
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
pub enum _ColliderType {
    Cuboid(Vec3),
    Ball(f32),
    TriMesh((Vec<Point3<f32>>, Vec<[u32; 3]>)),
}
impl Default for _ColliderType {
    fn default() -> Self {
        _ColliderType::Cuboid(vec3(1., 1., 1.))
    }
}
impl _ColliderType {
    pub fn get_collider(&self) -> ColliderBuilder {
        match self {
            _ColliderType::Cuboid(v) => ColliderBuilder::cuboid(v.x, v.y, v.z),
            _ColliderType::Ball(r) => ColliderBuilder::ball(*r),
            _ColliderType::TriMesh((verts, indices)) => {
                ColliderBuilder::trimesh(verts.clone(), indices.clone())
            }
        }
    }
}
#[derive(ComponentID, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct _Collider {
    _type: _ColliderType,
    #[serde(skip_serializing, skip_deserializing)]
    handle: ColliderHandle,
}
impl Component for _Collider {
    fn init(&mut self, _transform: &Transform, _id: i32, _sys: &Sys) {
        _sys.new_colliders.push((
            self._type.clone(),
            _transform.get_position(),
            _transform.get_rotation(),
            unsafe { SendSync::new(&mut self.handle) },
        ));
    }
    fn deinit(&mut self, _transform: &Transform, _id: i32, _sys: &Sys) {
        if self.handle != ColliderHandle::invalid() {
            _sys.physics.lock().remove_collider(self.handle);
        }
    }
    fn late_update(&mut self, _transform: &Transform, _sys: &System) {
        // let col = unsafe { _sys.physics.get_collider(self.handle).unwrap() };
        // col.set_translation(_transform.get_position().into());
        // col.set_rotation(UnitQuaternion::new_unchecked(_transform.get_rotation()));
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
            _ColliderType::TriMesh((_, _)) => false,
        } {
            // let phys = sys.physics.lock();
            // phys.get_collider(self.handle).unwrap().set_shape(shape)
            self.deinit(transform, id, sys);
            self.init(transform, id, sys);
        }
        if self.handle != ColliderHandle::invalid() {
            let mut phys = sys.physics.lock();
            let col = unsafe { phys.get_collider_mut(self.handle).unwrap() };
            col.set_translation(transform.get_position().into());
            col.set_rotation(UnitQuaternion::new_unchecked(transform.get_rotation()));
        }
    }
}
impl _Collider {
    pub fn a() {
        let b = ColliderBuilder::ball(3.).build();
    }
}
