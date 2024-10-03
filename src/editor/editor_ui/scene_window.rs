use std::{mem::transmute, sync::Arc};

use crate::{
    editor::editor_cam::EditorCam,
    engine::{
        particles::shaders::cs::transform,
        physics::PhysicsData,
        prelude::{Inspectable_, VulkanManager},
        rendering::{
            component::{ur::o, Renderer},
            model::ModelManager,
        },
        world::transform::TRANSFORMS,
    },
};
use egui::{Key, Rect, Ui};
use egui_gizmo::{Gizmo, GizmoMode};
use egui_winit_vulkano::Gui;
use nalgebra_glm::{self as glm, quat_euler_angles, vec3, vec4, Mat3, Quat, Vec4};
use nalgebra_glm::{Mat4, Vec3};
use ncollide3d::{
    math::Isometry,
    na::{point, Matrix3, Point3, Translation, Translation3, UnitQuaternion, Vector3, Vector4},
    partitioning::{BVHImpl, BVH},
    query::{visitors::RayInterferencesCollector, Ray, RayCast},
};
use parking_lot::Mutex;
use russimp::mesh;
use vulkano::pipeline::graphics::viewport;

use super::{
    entity_inspector::{GameObjectInspector, ROTATION_EULER, _SELECTED},
    EditorWindow,
};
// a map of mesh ids to rapier collision meshes
static SCENE_COLLISION_MESH_MAP: parking_lot::Mutex<
    Option<
        std::collections::HashMap<
            i32,
            Vec<((Point3<f32>, Point3<f32>), ncollide3d::shape::TriMesh<f32>)>,
        >,
    >,
> = parking_lot::Mutex::new(None);
pub(super) struct SceneWindow {
    image: egui::TextureId,
    window_dims: [u32; 2],
    cam: EditorCam,
    vk: Arc<VulkanManager>,
    gizmo_mode: GizmoMode,
    orientation: egui_gizmo::GizmoOrientation,
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
            // scene_collision: PhysicsData::new(),
            // gizmo: Gizmo::new("scene_gizmo"),
        }
    }
}
impl SceneWindow {
    fn pointer_ray(&self, ui: &Ui, vp: &Mat4, viewport: &Rect) -> Option<Ray<f32>> {
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
            origin: Point3::new(origin.x, origin.y, origin.z),
            dir: Vector3::new(direction.x, direction.y, direction.z),
        })
    }
}

// Function to rotate an AABB and return the new axis-aligned AABB
fn rotate_aabb(
    aabb: (Point3<f32>, Point3<f32>),
    rotation: Matrix3<f32>,
) -> (Point3<f32>, Point3<f32>) {
    let (min, max) = aabb;

    // Extract the corners of the original AABB
    let corners = [
        min,
        Point3::new(min.x, min.y, max.z),
        Point3::new(min.x, max.y, min.z),
        Point3::new(min.x, max.y, max.z),
        Point3::new(max.x, min.y, min.z),
        Point3::new(max.x, min.y, max.z),
        Point3::new(max.x, max.y, min.z),
        max,
    ];

    // Apply the rotation to each corner
    let rotated_corners: Vec<Point3<f32>> =
        corners.iter().map(|&corner| rotation * corner).collect();

    // Compute the new AABB by finding the min and max coordinates
    let mut new_min = rotated_corners[0];
    let mut new_max = rotated_corners[0];

    for corner in rotated_corners.iter().skip(1) {
        new_min = Point3::new(
            new_min.x.min(corner.x),
            new_min.y.min(corner.y),
            new_min.z.min(corner.z),
        );
        new_max = Point3::new(
            new_max.x.max(corner.x),
            new_max.y.max(corner.y),
            new_max.z.max(corner.z),
        );
    }

    (new_min, new_max)
}

impl EditorWindow for SceneWindow {
    fn update(
        &mut self,
        editor_args: &mut super::EditorArgs,
        inspectable: &mut Option<Arc<Mutex<dyn Inspectable_>>>,
        is_focused: bool,
    ) {
        if is_focused {
            self.cam
                .update(&editor_args.world.input, &editor_args.world.time);
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
        let mut bvs = Vec::new();
        // let bvh = ncollide3d::partitioning::BVT::new_balanced(leaves)
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
                                            let collider = ncollide3d::shape::TriMesh::new(
                                                mesh.vertices
                                                    .iter()
                                                    .cloned()
                                                    .map(|v| {
                                                        Point3::new(
                                                            v.position[0],
                                                            v.position[1],
                                                            v.position[2],
                                                        )
                                                    })
                                                    .collect(),
                                                mesh.indices
                                                    .chunks(3)
                                                    .map(|i| {
                                                        Point3::new(
                                                            i[0] as usize,
                                                            i[1] as usize,
                                                            i[2] as usize,
                                                        )
                                                    })
                                                    .collect(),
                                                None,
                                            );
                                            let aabb = (
                                                Point3::new(
                                                    mesh.aabb.0[0],
                                                    mesh.aabb.0[1],
                                                    mesh.aabb.0[2],
                                                ),
                                                Point3::new(
                                                    mesh.aabb.1[0],
                                                    mesh.aabb.1[1],
                                                    mesh.aabb.1[2],
                                                ),
                                            );
                                            (aabb, collider)
                                        })
                                        .collect();
                                    scene_collsion_mesh_map.insert(model_id, mesh_colliders);
                                }
                            }
                            let mesh_colliders = scene_collsion_mesh_map.get(&model_id).unwrap();
                            for (i, mesh) in mesh_colliders.iter().enumerate() {
                                // let mut col = mesh.clone();
                                let mut aabb = mesh.0;
                                //scale aabb
                                let scl = transform.get_scale();
                                aabb.0.coords[0] *= scl.x;
                                aabb.0.coords[1] *= scl.y;
                                aabb.0.coords[2] *= scl.z;
                                aabb.1.coords[0] *= scl.x;
                                aabb.1.coords[1] *= scl.y;
                                aabb.1.coords[2] *= scl.z;

                                //rotate aabb
                                let rot = transform.get_rotation();
                                let rot: Mat3 = glm::quat_to_mat3(&rot);
                                let rot: Matrix3<f32> =
                                    Matrix3::from_iterator(rot.as_slice().iter().cloned());
                                let mut aabb = rotate_aabb(aabb, rot);
                                //translate aabb
                                let pos = transform.get_position();
                                aabb.0.coords[0] += pos.x;
                                aabb.0.coords[1] += pos.y;
                                aabb.0.coords[2] += pos.z;
                                aabb.1.coords[0] += pos.x;
                                aabb.1.coords[1] += pos.y;
                                aabb.1.coords[2] += pos.z;

                                // aabb.1.coords += transform.get_position();
                                bvs.push((
                                    (unsafe { *id.get() }, model_id, i),
                                    ncollide3d::bounding_volume::AABB::new(aabb.0, aabb.1),
                                ));
                                // let pos = transform.get_position();
                                // let rot = transform.get_rotation();
                                // let scl = transform.get_scale();
                                // let col = col
                                //     .user_data(transform.id as u128)
                                //     .position(pos.into())
                                //     .rotation(quat_euler_angles(&rot))
                                //     .build();
                                // let coll = mesh.build();
                                // self.scene_collision.collider_set.insert(col);
                            }
                        }
                    })
            });
        let bvh = ncollide3d::partitioning::BVT::new_balanced(bvs);

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
                let curr_gizmo_mode = self.gizmo_mode;
                if input.key_released(Key::T) {
                    self.gizmo_mode = GizmoMode::Translate;
                } else if input.key_released(Key::R) {
                    self.gizmo_mode = GizmoMode::Rotate;
                } else if input.key_released(Key::S) {
                    self.gizmo_mode = GizmoMode::Scale;
                } else if input.key_released(Key::L) {
                    self.orientation = egui_gizmo::GizmoOrientation::Local;
                } else if input.key_released(Key::G) {
                    self.orientation = egui_gizmo::GizmoOrientation::Global;
                }

                if self.gizmo_mode == curr_gizmo_mode
                    && (input.key_released(Key::T)
                        || input.key_released(Key::R)
                        || input.key_released(Key::S))
                {
                    self.orientation = match self.orientation {
                        egui_gizmo::GizmoOrientation::Local => egui_gizmo::GizmoOrientation::Global,
                        egui_gizmo::GizmoOrientation::Global => egui_gizmo::GizmoOrientation::Local,
                    };
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
        if ui.input(|r| {
            !r.pointer.is_decidedly_dragging()
                && r.pointer.primary_released()
                && viewport.contains(r.pointer.hover_pos().unwrap())
        }) {
            if let Some(ray) = self.pointer_ray(ui, &(projection * view), &viewport) {
                let mut collector_hit = Vec::new();

                let mut visitor_hit =
                    RayInterferencesCollector::new(&ray, 10_000.0, &mut collector_hit);

                bvh.visit(&mut visitor_hit);
                let mut closest = None;
                for (id, model_id, mesh_idx) in collector_hit.iter() {
                    let scene_collision_mesh_map = SCENE_COLLISION_MESH_MAP.lock();
                    if let Some(model) = scene_collision_mesh_map
                        .as_ref()
                        .and_then(|f| f.get(model_id))
                        .and_then(|f| f.get(*mesh_idx))
                    {
                        let mesh = &model.1;
                        let transform = unsafe { &*TRANSFORMS }.get(*id).unwrap();
                        let inv_trans = transform.get_matrix().try_inverse().unwrap();
                        let inv_rot_scl = (glm::quat_to_mat4(&transform.get_rotation())
                            * glm::scaling(&transform.get_scale()))
                        .try_inverse()
                        .unwrap();

                        let origin =
                            inv_trans * vec4(ray.origin.x, ray.origin.y, ray.origin.z, 1.0);
                        let dir = inv_rot_scl * vec4(ray.dir.x, ray.dir.y, ray.dir.z, 1.0);

                        let ray = Ray::new(
                            Point3::new(origin.x, origin.y, origin.z),
                            Vector3::new(dir.x, dir.y, dir.z),
                        );
                        mesh.toi_with_ray(&Isometry::identity(), &ray, 10_000.0, false)
                            .map(|toi| {
                                closest = (closest.unwrap_or((*id, toi)).1 > toi)
                                    .then(|| (*id, toi))
                                    .or(Some(closest.unwrap_or((*id, toi))));
                            });
                    }
                }
                if let Some((id, toi)) = closest {
                    unsafe { _SELECTED = Some(id) };
                    inspectable.replace(Arc::new(Mutex::new(GameObjectInspector {})));
                }
            }
        }

        // println!("bvs: {:?}", bvs);
        // println!("bvs len: {}", bvs.len());
        // self.cam.camera.get(|c| {
        //     c.debug.append_aabb(
        //         vec3(-10., -10., -10.),
        //         vec3(10., 10., 10.),
        //         0.2,
        //         vec4(0.0, 1.0, 0.0, 1.0),
        //     );
        //     for a in &bvs {
        //         let min = a.1.mins();
        //         let max = a.1.maxs();
        //         c.debug.append_aabb(
        //             vec3(min.x, min.y, min.z),
        //             vec3(max.x, max.y, max.z),
        //             0.2,
        //             glm::vec4(1.0, 0.0, 0.0, 1.0),
        //         );
        //     }
        // });
    }

    fn get_name(&self) -> &str {
        "Scene"
    }
}
