use std::{mem::transmute, sync::Arc};

use crate::{
    editor::editor_cam::EditorCam,
    engine::{
        physics::PhysicsData,
        prelude::{Inspectable_, VulkanManager},
        rendering::component::Renderer,
        rendering::model::ModelManager,
        world::transform::TRANSFORMS,
    },
};
use egui::{Key, Rect, Ui};
use egui_gizmo::{Gizmo, GizmoMode};
use egui_winit_vulkano::Gui;
use nalgebra_glm::{self as glm, quat_euler_angles, Vec4};
use nalgebra_glm::{Mat4, Vec3};
use parking_lot::Mutex;
use rapier3d::{
    na::{Point3, UnitQuaternion},
    prelude::*,
};
use russimp::mesh;
use vulkano::pipeline::graphics::viewport;

use super::{
    entity_inspector::{GameObjectInspector, ROTATION_EULER, _SELECTED},
    EditorWindow,
};
// a map of mesh ids to rapier collision meshes
static SCENE_COLLISION_MESH_MAP: parking_lot::Mutex<
    Option<std::collections::HashMap<i32, Vec<rapier3d::geometry::ColliderBuilder>>>,
> = parking_lot::Mutex::new(None);
pub(super) struct SceneWindow {
    image: egui::TextureId,
    window_dims: [u32; 2],
    cam: EditorCam,
    vk: Arc<VulkanManager>,
    gizmo_mode: GizmoMode,
    orientation: egui_gizmo::GizmoOrientation,
    // gizmo: Gizmo,
    scene_collision: PhysicsData,
}

impl SceneWindow {
    pub fn new(vk: Arc<VulkanManager>) -> SceneWindow {
        SceneWindow {
            image: egui::TextureId::default(),
            window_dims: [1, 1],
            cam: EditorCam::new(vk.clone()),
            vk,
            gizmo_mode: GizmoMode::Translate,
            orientation: egui_gizmo::GizmoOrientation::Global,
            scene_collision: PhysicsData::new(),
            // gizmo: Gizmo::new("scene_gizmo"),
        }
    }
}
impl SceneWindow {
    fn pointer_ray(&self, ui: &Ui, vp: &Mat4, viewport: &Rect) -> Option<Ray> {
        let hover = ui.input(|i| i.pointer.hover_pos())?;

        let x = ((hover.x - viewport.min.x) / viewport.width()) * 2.0 - 1.0;
        let y = ((hover.y - viewport.min.y) / viewport.height()) * 2.0 - 1.0;

        let screen_to_world = glm::inverse(&vp);
        let mut origin = screen_to_world * Vec4::new(x, -y, -1.0, 1.0);
        let origin = origin.xyz() / origin.w;
        let mut target = screen_to_world * Vec4::new(x, -y, 1.0, 1.0);

        // w is zero when far plane is set to infinity
        if target.w.abs() < 1e-7 {
            target.w = 1e-7;
        }

        let target = target.xyz() / target.w;

        let direction = (target - origin).normalize();

        Some(Ray {
            origin: origin.into(),
            dir: direction,
        })
    }
}
impl EditorWindow for SceneWindow {
    fn update(
        &mut self,
        editor_args: &mut super::EditorArgs,
        inspectable: &mut Option<Arc<Mutex<dyn Inspectable_>>>,
        is_focused: bool,
    ) {
        self.scene_collision = PhysicsData::new();
        if is_focused {
            self.cam
                .update(&editor_args.world.input, &editor_args.world.time);
            editor_args
                .world
                .sys
                .assets_manager
                .get_manager2(|models: &ModelManager| {
                    let mut scene_collsion_mesh_map = SCENE_COLLISION_MESH_MAP.lock();
                    if scene_collsion_mesh_map.is_none() {
                        *scene_collsion_mesh_map = Some(std::collections::HashMap::new());
                    }
                    let scene_collsion_mesh_map = &mut *scene_collsion_mesh_map.as_mut().unwrap();
                    editor_args
                        .world
                        .get_component_storage::<Renderer, _, _>(|renderers| {
                            let iter = renderers.valid.iter().zip(renderers.data.iter());
                            for (v, (id, renderer)) in iter {
                                if !unsafe { **v.get() } {
                                    continue;
                                }
                                let model_id = renderer.lock().get_model().id;
                                let transform = unsafe { (*TRANSFORMS).get(*id.get()).unwrap() };

                                if !scene_collsion_mesh_map.contains_key(&model_id) {
                                    if let Some(model) = models.get_id(&model_id) {
                                        let model = model.lock();
                                        let mesh_colliders = model
                                            .model
                                            .meshes
                                            .iter()
                                            .map(|mesh| {
                                                let collider =
                                                    rapier3d::geometry::ColliderBuilder::trimesh(
                                                        mesh.vertices
                                                            .iter()
                                                            .cloned()
                                                            .map(|v| {
                                                                point![
                                                                    v.position[0],
                                                                    v.position[1],
                                                                    v.position[2]
                                                                ]
                                                            })
                                                            .collect(),
                                                        mesh.indices
                                                            .chunks(3)
                                                            .map(|i| {
                                                                [
                                                                    i[0] as u32,
                                                                    i[1] as u32,
                                                                    i[2] as u32,
                                                                ]
                                                            })
                                                            .collect(),
                                                    );
                                                collider
                                            })
                                            .collect();
                                        scene_collsion_mesh_map.insert(model_id, mesh_colliders);
                                    }
                                }
                                let mesh_colliders =
                                    scene_collsion_mesh_map.get(&model_id).unwrap();
                                for mesh in mesh_colliders {
                                    let mut col = mesh.clone();
                                    let pos = transform.get_position();
                                    let rot = transform.get_rotation();
                                    let scl = transform.get_scale();
                                    let col = col
                                        .user_data(transform.id as u128)
                                        .position(pos.into())
                                        .rotation(quat_euler_angles(&rot))
                                        .build();
                                    // let coll = mesh.build();
                                    self.scene_collision.collider_set.insert(col);
                                }
                            }
                        })
                });
            self.scene_collision.query_pipeline.update(
                // &self.island_manager,
                &self.scene_collision.rigid_body_set,
                &self.scene_collision.collider_set,
            );
        }
        self.cam.camera.get(|c| {
            c.is_visible = false;
        })
    }

    fn draw(
        &mut self,
        ui: &mut egui::Ui,
        editor_args: &mut super::EditorArgs,
        inspectable: &mut Option<
            std::sync::Arc<parking_lot::Mutex<dyn crate::engine::prelude::Inspectable_>>,
        >,
        rec: egui::Rect,
        id: egui::Id,
        gui: &mut Gui,
    ) {
        let mut projection = Mat4::identity();
        let mut view = Mat4::identity();
        self.cam.camera.get(|c| {
            c.is_visible = true;
            c.update(
                self.cam.pos,
                self.cam.rot,
                0.01f32,
                10_000f32,
                70f32,
                false,
                1,
            );
            projection = c.camera_view_data.front().map(|cvd| cvd.proj).unwrap();
            view = c.camera_view_data.front().map(|cvd| cvd.view).unwrap();
        });
        let snapping = ui.input(|input| input.modifiers.ctrl);

        ui.horizontal_top(|ui| {
            if editor_args.playing_game {
                if ui.button("Stop").clicked() {
                    println!("stop game");
                    editor_args.playing_game = false;
                }
            } else if ui.button("Play").clicked() {
                println!("play game");
                editor_args.playing_game = true;
                editor_args.world.begin_play();
            }
        });

        let a = ui.available_size();
        let viewport = ui.available_rect_before_wrap();
        if ui.input(|r| r.pointer.primary_released()) {
            if let Some(pointer_ray) = self.pointer_ray(ui, &(projection * view), &viewport) {
                if let Some((handle, _)) = self.scene_collision.query_pipeline.cast_ray(
                    &self.scene_collision.rigid_body_set,
                    &self.scene_collision.collider_set,
                    &pointer_ray,
                    1000.0,
                    true,
                    QueryFilter::default(),
                ) {
                    let t_id = self
                        .scene_collision
                        .collider_set
                        .get(handle)
                        .unwrap()
                        .user_data as i32;
                    // let transform = (unsafe { &*TRANSFORMS }).get(t_id).unwrap();
                    unsafe {
                        _SELECTED = Some(t_id);
                        *inspectable = Some(Arc::new(Mutex::new(GameObjectInspector{})));    
                    };
                }
            }
        }

        self.window_dims = [a[0] as u32, a[1] as u32];
        self.cam
            .camera
            .get(|c| c.resize(self.window_dims, self.vk.clone(), gui));
        // let mut tex_id = self.image;
        self.cam.camera.get(|c| {
            if let Some(tex) = c.texture_id {
                // tex_id = tex;
                self.image = tex;
            }
        });
        ui.image(self.image, a);

        ui.input(|input| {
            if !input.pointer.secondary_down() {
                if input.key_released(Key::T) {
                    self.gizmo_mode = GizmoMode::Translate;
                } else if input.key_released(Key::R) {
                    self.gizmo_mode = GizmoMode::Rotate;
                } else if input.key_released(Key::S) {
                    self.gizmo_mode = GizmoMode::Scale;
                }

                if input.key_released(Key::L) {
                    self.orientation = egui_gizmo::GizmoOrientation::Local;
                } else if input.key_released(Key::G) {
                    self.orientation = egui_gizmo::GizmoOrientation::Global;
                }
            }
        });

        let mut model = Mat4::identity();
        if let Some(ref selected) = unsafe { _SELECTED } {
            let transform = unsafe { &*TRANSFORMS }.get(*selected);
            if let Some(t) = transform {
                model = t.get_matrix();

                let gizmo = Gizmo::new("scene_gizmo")
                    .viewport(viewport)
                    .orientation(self.orientation)
                    .mode(self.gizmo_mode)
                    .snapping(snapping)
                    .view_matrix(view)
                    .projection_matrix(projection)
                    .model_matrix(model);

                if let Some(result) = gizmo.interact(ui) {
                    match self.gizmo_mode {
                        GizmoMode::Translate => {
                            let translation = nalgebra_glm::Vec3::new(
                                result.translation.x,
                                result.translation.y,
                                result.translation.z,
                            );
                            t.set_position(&translation);
                        }
                        GizmoMode::Rotate => {
                            let rotation = nalgebra_glm::quat(
                                result.rotation.x,
                                result.rotation.y,
                                result.rotation.z,
                                result.rotation.w,
                            );
                            unsafe {
                                ROTATION_EULER = Some(glm::quat_euler_angles(&rotation));
                            }
                            t.set_rotation(&rotation);
                        }
                        GizmoMode::Scale => {
                            let scale = nalgebra_glm::Vec3::new(
                                result.scale.x,
                                result.scale.y,
                                result.scale.z,
                            );
                            t.set_scale(scale);
                        }
                    }
                }
            }
        }
    }

    fn get_name(&self) -> &str {
        "Scene"
    }
}
