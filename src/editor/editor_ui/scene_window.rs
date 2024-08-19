use std::{mem::transmute, sync::Arc};

use crate::{
    editor::editor_cam::EditorCam,
    engine::{
        prelude::{Inspectable_, VulkanManager},
        world::transform::TRANSFORMS,
    },
};
use egui_gizmo::{Gizmo, GizmoMode};
use egui_winit_vulkano::Gui;
use nalgebra_glm::Mat4;
use parking_lot::Mutex;
use vulkano::pipeline::graphics::viewport;

use super::{
    entity_inspector::{GameObjectInspector, _SELECTED},
    EditorWindow,
};

pub(super) struct SceneWindow {
    image: egui::TextureId,
    window_dims: [u32; 2],
    cam: EditorCam,
    vk: Arc<VulkanManager>,
    // gizmo: Gizmo,
}

impl SceneWindow {
    pub fn new(vk: Arc<VulkanManager>) -> SceneWindow {
        SceneWindow {
            image: egui::TextureId::default(),
            window_dims: [1, 1],
            cam: EditorCam::new(vk.clone()),
            vk,
            // gizmo: Gizmo::new("scene_gizmo"),
        }
    }
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
        let viewport = rec;
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

        // self.gizmo.(GizmoConfig {
        //     view_matrix: view_matrix.into(),
        //     projection_matrix: projection_matrix.into(),
        //     viewport,
        //     modes: GizmoMode::TRANSLATE | GizmoMode::ROTATE | GizmoMode::SCALE,
        //     orientation: self.gizmo_orientation,
        //     snapping,
        //     ..Default::default()
        // });
        if let Some(ref mut i) = inspectable {
            let mut i = i.lock();
            let a: &dyn Inspectable_ = &*i;
            if let Some(i) = a.as_any().downcast_ref::<GameObjectInspector>() {
                let mut model = Mat4::identity();
                if let Some(ref selected) = unsafe { _SELECTED } {
                    let transform = unsafe { &*TRANSFORMS }.get(*selected);
                    if let Some(t) = transform {
                        model = t.get_matrix();

                        let gizmo = Gizmo::new("scene_gizmo")
                            .view_matrix(view)
                            .projection_matrix(projection)
                            .model_matrix(model)
                            .mode(GizmoMode::Translate);

                        if let Some(result) = gizmo.interact(ui) {
                            let translation = nalgebra_glm::Vec3::new(result.translation.x, result.translation.y, result.translation.z);
                            t.translate(translation);
                        }
                    }
                }
            }
        }
        // .viewport(egui::emath::Rect::from(viewport))
        // .orientation();

        // let viewport = ui.clip_rect();

        ui.horizontal_top(|ui| {
            if editor_args.playing_game {
                if ui.button("Stop").clicked() {
                    println!("stop game");
                    editor_args.playing_game = false;

                    // {
                    //     // let mut world = self.world;
                    //     self.world.clear();
                    //     serialize::deserialize(&mut self.world);
                    // }
                }
            } else if ui.button("Play").clicked() {
                println!("play game");
                editor_args.playing_game = true;
                editor_args.world.begin_play();
                // {
                //     // let mut world = self.world;
                //     self.world.clear();
                //     serialize::deserialize(&mut self.world);
                // }
            }
        });
        let a = ui.available_size();
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
    }

    fn get_name(&self) -> &str {
        "Scene"
    }
}
