use std::sync::Arc;

use egui_winit_vulkano::Gui;
use parking_lot::Mutex;

use crate::{
    editor::editor_cam::EditorCam,
    engine::prelude::{Inspectable_, VulkanManager},
};

use super::EditorWindow;

pub(super) struct SceneWindow {
    image: egui::TextureId,
    window_dims: [u32; 2],
    cam: EditorCam,
    vk: Arc<VulkanManager>
}

impl SceneWindow {
    pub fn new(vk: Arc<VulkanManager>) -> SceneWindow {
        SceneWindow {
            image: egui::TextureId::default(),
            window_dims: [1, 1],
            cam: EditorCam::new(vk.clone()),
            vk,
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
            )
        });

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
        self.cam.camera.get(|c| {
            c.resize(self.window_dims, self.vk.clone(), gui)
        });
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
