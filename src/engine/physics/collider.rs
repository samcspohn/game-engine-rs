use std::cell::SyncUnsafeCell;
use std::sync::Arc;
use std::{collections::HashMap, mem::transmute};

use crate::engine::rendering::model::ModelManager;
// use component_derive::ComponentID;
use crate::engine::project::asset_manager::AssetInstance;
use crate::engine::world::NewCollider;
use crate::engine::{prelude::*, project::asset_manager::drop_target};
use force_send_sync::SendSync;
use lazy_static::lazy_static;
use nalgebra_glm::{quat_euler_angles, vec3, Vec3};
use once_cell::sync::Lazy;
use rapier3d::{
    na::{Point3, UnitQuaternion},
    prelude::*,
};
use serde::{Deserialize, Serialize};

pub struct PhysMesh {
    pub verts: Vec<Point3<f32>>,
    pub indices: Vec<[u32; 3]>,
}
// lazy_static! {
// pub static mut MESH_MAP: *mut HashMap<i32, ColliderBuilder> = std::ptr::null_mut();
// pub static mut PROC_MESH_ID: *mut i32 = std::ptr::null_mut();
// }

#[derive(Clone, Deserialize, Serialize)]
pub enum _ColliderType {
    Cuboid(Vec3),
    Ball(f32),
    TriMesh(i32),
    TriMeshUnint((Arc<[Point3<f32>]>, Arc<[[u32; 3]]>, i32)),
}
impl Default for _ColliderType {
    fn default() -> Self {
        _ColliderType::Cuboid(vec3(1., 1., 1.))
    }
}
impl _ColliderType {
    pub fn get_collider(&mut self, sys: &Sys) -> ColliderBuilder {
        match self {
            _ColliderType::Cuboid(v) => ColliderBuilder::cuboid(v.x, v.y, v.z),
            _ColliderType::Ball(r) => ColliderBuilder::ball(*r),
            _ColliderType::TriMesh(id) => {
                let mut mesh_map = sys.mesh_map.lock();
                unsafe {
                    if !mesh_map.contains_key(&id) {
                        let model_manager_mutex = sys.get_model_manager();
                        let model_manager_lock = model_manager_mutex.lock();
                        let model_manager = model_manager_lock
                            .as_any()
                            .downcast_ref::<ModelManager>()
                            .unwrap();
                        if let Some(_id) = model_manager.assets_id.get(&id) {
                            unsafe {
                                let model = model_manager.assets_id.get(id).unwrap().lock();
                                let verts = model.model.meshes[0]
                                    .vertices
                                    .iter()
                                    .map(|f| point![f.position[0], f.position[1], f.position[2]])
                                    .collect();
                                let indices = model.model.meshes[0]
                                    .indeces
                                    .chunks(3)
                                    .map(|f| [f[0], f[1], f[2]])
                                    .collect::<Vec<[u32; 3]>>();
                                mesh_map.insert(*id, ColliderBuilder::trimesh(verts, indices));
                            }
                        }
                    }
                }
                let mesh = mesh_map.get(id).unwrap();
                mesh.clone()
                // ColliderBuilder::trimesh(mesh.verts.clone(), mesh.indices.clone())
            }
            _ColliderType::TriMeshUnint((v, i, id)) => unsafe {
                // println!("{}", v.len());
                // let id = sys.proc_mesh_id.fetch_add(-1, std::sync::atomic::Ordering::Relaxed);
                let id: i32 = *id;
                let mut mesh_map = sys.mesh_map.lock();
                mesh_map.insert(
                    id,
                    ColliderBuilder::trimesh(
                        v.iter().cloned().collect(),
                        i.iter().cloned().collect(),
                    ),
                );
                *self = _ColliderType::TriMesh(id);
                mesh_map.get(&id).unwrap().clone()
            },
        }
    }
}
#[derive(ComponentID, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
#[repr(C)]
pub struct _Collider {
    pub _type: _ColliderType,
    #[serde(skip_serializing, skip_deserializing)]
    pub handle: ColliderHandle,
}
impl Component for _Collider {
    fn init(&mut self, _transform: &Transform, _id: i32, _sys: &Sys) {
        _sys.new_colliders.push(NewCollider {
            ct: unsafe { SendSync::new(&mut self._type) },
            pos: _transform.get_position(),
            rot: _transform.get_rotation(),
            tid: _transform.id,
            ch: unsafe { SendSync::new(&mut self.handle) },
        });
    }
    fn deinit(&mut self, _transform: &Transform, _id: i32, _sys: &Sys) {
        if self.handle != ColliderHandle::invalid() {
            _sys.to_remove_colliders.push(self.handle);
            // _sys.physics.lock().remove_collider(self.handle);
        }
        self.handle = ColliderHandle::invalid();
    }
    fn inspect(&mut self, transform: &Transform, id: i32, ui: &mut egui::Ui, sys: &Sys) {
        ui.menu_button("shape", |ui| {
            if ui.button("Cuboid").clicked() {
                self._type = _ColliderType::Cuboid(Vec3::default());
            }
            if ui.button("Ball").clicked() {
                self._type = _ColliderType::Ball(1.0);
            }
            if ui.button("TriMesh").clicked() {
                self._type = _ColliderType::TriMesh((-1));
            }
        });

        if match &mut self._type {
            _ColliderType::Cuboid(v) => Ins(v).inspect("Dimensions", ui, sys),
            _ColliderType::Ball(f) => Ins(f).inspect("radius", ui, sys),
            _ColliderType::TriMesh((id)) => {
                // let label = id.to_string();
                let mut ret = false;
                let drop_data = sys.assets_manager.drag_drop_data.lock().clone();
                let model_manager_mutex = sys.get_model_manager();
                let model_manager_lock = model_manager_mutex.lock();
                let model_manager = model_manager_lock
                    .as_any()
                    .downcast_ref::<ModelManager>()
                    .unwrap();
                let label: String = match model_manager.assets_r.get(&id) {
                    Some(file) => file.clone(),
                    None => "".into(),
                };
                let can_accept_drop_data = match drop_data.rfind(".obj") {
                    Some(_) => true,
                    None => false,
                };
                ui.horizontal(|ui| {
                    ui.add(egui::Label::new("mesh"));
                    drop_target(ui, can_accept_drop_data, |ui| {
                        let response = ui.add(egui::Label::new(label.as_str()));
                        if response.hovered() && ui.input(|i| i.pointer.any_released()) {
                            let model_file: String = drop_data.to_string();

                            if let Some(_id) = model_manager.assets.get(&model_file) {
                                *id = *_id;
                                ret = true;
                                unsafe {
                                    let mut mesh_map = sys.mesh_map.lock();
                                    if !mesh_map.contains_key(_id) {
                                        let model =
                                            model_manager.assets_id.get(_id).unwrap().lock();
                                        let verts = model.model.meshes[0]
                                            .vertices
                                            .iter()
                                            .map(|f| {
                                                point![f.position[0], f.position[1], f.position[2]]
                                            })
                                            .collect();
                                        let indices = model.model.meshes[0]
                                            .indeces
                                            .chunks(3)
                                            .map(|f| [f[0], f[1], f[2]])
                                            .collect::<Vec<[u32; 3]>>();
                                        mesh_map
                                            .insert(*id, ColliderBuilder::trimesh(verts, indices));
                                    }
                                }
                            }
                        }
                    });
                });
                ret
            }
            _ColliderType::TriMeshUnint(_) => {
                unreachable!()
            }
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
