use crate::{
    editor::{editor_ui::entity_inspector::_selected, inspectable::Inspectable_},
    engine::{
        input::Input,
        prelude::VulkanManager,
        project::{
            asset_manager::AssetsManager,
            file_watcher::{self, FileWatcher},
            serialize, Project,
        },
        time::Time,
        utils,
        world::{transform::Transform, Sys, World},
        Engine,
    },
};
use console::ConsoleWindow;
use egui::{menu, Color32, Context, Layout, Pos2, Rect, Rounding, ScrollArea, Sense, Ui};
use egui_winit_vulkano::Gui;
use entity_inspector::{Inspector, LAST_ACTIVE};
use game_window::GameWindow;
use hierarchy::Hierarchy;
use nalgebra_glm as glm;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, MutexGuard};
use project::ProjectWindow;
use puffin_egui::puffin;
use rayon::str::EncodeUtf16;
use rfd::FileDialog;
use scene_window::SceneWindow;
use serde::de::IntoDeserializer;
use std::{
    any::TypeId,
    borrow::BorrowMut,
    cell::Cell,
    collections::{HashMap, VecDeque},
    fs,
    path::{self, PathBuf},
    rc::Rc,
    sync::Arc,
};

use egui_dock::{DockArea, NodeIndex, Style, Tree};

mod asset_browser;
mod console;
mod entity_inspector;
mod game_window;
mod hierarchy;
mod project;
mod scene_window;
mod top_menu;
mod util;
enum TransformDrag {
    DragToTransform(i32, i32),
    DragBetweenTransform(i32, i32, bool),
}
lazy_static::lazy_static! {

    pub(crate) static ref DRAGGED_TRANSFORM: Mutex<i32> = Mutex::new(0);
    static ref TRANSFORM_DRAG: Mutex<Option<TransformDrag>> = Mutex::new(None);
}
enum GameObjectContextMenu {
    NewGameObject(i32),
    CopyGameObject(i32),
    DeleteGameObject(i32),
}
pub static EDITOR_WINDOW_DIM: Lazy<Mutex<[u32; 2]>> = Lazy::new(|| Mutex::new([1920, 1080]));

struct TabViewer<'a, 'b> {
    // image: egui::TextureId,
    editor_args: &'a mut EditorArgs<'b>,
    // editor: &'a mut Editor,
    // goi: &'a GameObjectInspector<'b>,
    // views: &'a mut Vec<Box<dyn EditorWindow>>,
    gui: &'a mut Gui,
    active: *const (),
    inspectable: &'a mut Option<Arc<Mutex<dyn Inspectable_>>>,
}
// pub(crate) static mut PLAYING_GAME: bool = false;
impl egui_dock::TabViewer for TabViewer<'_, '_> {
    type Tab = Mutex<Box<dyn EditorWindow>>;

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        let rec = ui.max_rect();
        // let id = ui.id();
        // if tab.as_ref() as *const dyn EditorWindow as *const () == self.active {
        //     println!("active tab is {}", tab.get_name());
        // }
        let mut tab = tab.lock();
        let id = egui::Id::new(format!(
            "{}: {:?}",
            tab.get_name(),
            tab.as_ref() as *const dyn EditorWindow as *const ()
        ));
        tab.draw(
            ui,
            &mut self.editor_args,
            self.inspectable,
            rec,
            id,
            self.gui,
        );
    }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        let tab = tab.lock();
        format!(
            "{}: {:?}",
            tab.get_name(),
            tab.as_ref() as *const dyn EditorWindow as *const ()
        )
        .into()
        // (&*tab).into()
    }
}

struct ModelInspector {
    file: String,
}

impl Inspectable_ for ModelInspector {
    fn inspect(&mut self, ui: &mut egui::Ui, _world: &mut World) -> bool {
        ui.add(egui::Label::new(self.file.as_str()));
        true
    }
}

pub struct Editor {
    vk: Arc<VulkanManager>,
    editor_window_dim: [u32; 2],
    dragged_transform: Mutex<i32>,
    transform_drag: Mutex<Option<TransformDrag>>,
    inspectable: Option<Arc<Mutex<dyn Inspectable_>>>,
    dock: Tree<Mutex<Box<dyn EditorWindow>>>,
    views: Vec<Box<dyn EditorWindow>>,
    play_game: bool,
}

pub struct EditorArgs<'a> {
    pub world: &'a mut World,
    pub project: &'a mut Project,
    pub assets_manager: Arc<AssetsManager>,
    pub file_watcher: &'a FileWatcher,
    pub playing_game: bool,
}

impl Editor {
    pub fn new(vk: Arc<VulkanManager>) -> Self {
        let mut tree: Tree<Mutex<Box<dyn EditorWindow>>> = Tree::new(vec![
            Mutex::new(Box::new(SceneWindow::new(vk.clone()))),
            Mutex::new(Box::new(GameWindow::new(vk.clone()))),
        ]);

        // You can modify the tree before constructing the dock
        let [a, b] = tree.split_left(
            NodeIndex::root(),
            0.15,
            vec![Mutex::new(Box::new(Hierarchy::new()))],
        );
        let ins = Box::new(Inspector::new());
        unsafe { LAST_ACTIVE = ins.as_ref() };
        let [c, _] = tree.split_right(a, 0.8, vec![Mutex::new(ins)]);
        let [_, _] = tree.split_below(b, 0.5, vec![Mutex::new(Box::new(ProjectWindow::new()))]);
        let [_, _] = tree.split_below(c, 0.7, vec![Mutex::new(Box::new(ConsoleWindow::new()))]);
        Self {
            vk,
            editor_window_dim: [1920, 1080],
            dragged_transform: Mutex::new(-1),
            transform_drag: Mutex::new(None),
            inspectable: None,
            dock: tree,
            views: vec![Box::new(Hierarchy::new())],
            play_game: false,
        }
    }
    pub fn editor_ui(
        &mut self,
        mut editor_args: EditorArgs,
        egui_ctx: &Context,
        gui: &mut Gui,
        curr_playing: bool,
    ) -> bool {
        {
            let active = self
                .dock
                .find_active_focused()
                .and_then(|p| {
                    let a: *const dyn EditorWindow = &(*p.1.lock().as_ref());
                    let b: *const () = a as *const ();
                    Some(b)
                })
                .unwrap_or(std::ptr::null());

            self.dock.tabs().for_each(|tab| {
                let mut tab = tab.lock();
                let is_active = tab.as_ref() as *const dyn EditorWindow as *const () == active;
                tab.update(&mut editor_args, &mut self.inspectable, is_active);
            });

            // self.dock.set_active_tab(node_index, tab_index)
            // self.dock.tabs().for_each(|tab| {
            //     // if active.is_none() {
            //     //     tab.update(&editor_args.world.input, &editor_args.world.time, false);
            //     // } else {
            //     //     let active = active.unwrap();
            //     tab.update(
            //         &editor_args.world.input,
            //         &editor_args.world.time,
            //         tab.as_ref() as *const dyn EditorWindow as *const () == active,
            //     );
            //     // }
            // });
            {
                // let world = editor_args.world;
                let assets_manager = editor_args.assets_manager.clone();
                let file_watcher = &editor_args.file_watcher;

                egui::TopBottomPanel::top("game engine").show(egui_ctx, |ui| {
                    menu::bar(ui, |ui| {
                        ui.menu_button("File", |ui| {
                            if ui.button("New Project").clicked() {
                                ui.close_menu();
                            }
                            ui.separator();
                            if ui.button("New scene").clicked() {
                                let files = FileDialog::new()
                                    .add_filter("scene", &["scene"])
                                    .set_directory(std::env::current_dir().unwrap())
                                    .save_file();
                                if let Some(file) = files {
                                    let file = file
                                        .strip_prefix(std::env::current_dir().unwrap())
                                        .unwrap()
                                        .as_os_str()
                                        .to_str()
                                        .unwrap();
                                    serialize::serialize_new(file);
                                    serialize::deserialize(editor_args.world, file);
                                    // utils::path_format(file);
                                    editor_args.project.working_scene = file.into();
                                }
                                ui.close_menu();
                            }
                            if ui.button("Open").clicked() {
                                // project?
                                let files = FileDialog::new()
                                    .add_filter("scene", &["scene"])
                                    .set_directory(std::env::current_dir().unwrap())
                                    .pick_file();
                                if let Some(file) = files {
                                    let file = file
                                        .strip_prefix(std::env::current_dir().unwrap())
                                        .unwrap()
                                        .as_os_str()
                                        .to_str()
                                        .unwrap();
                                    serialize::deserialize(editor_args.world, file);
                                    editor_args.project.working_scene = file.into();
                                }
                                ui.close_menu();
                            }
                            if ui.button("Save").clicked() {
                                serialize::serialize(
                                    &editor_args.world,
                                    &editor_args.project.working_scene,
                                );
                                editor_args.project.save_project(
                                    file_watcher,
                                    &editor_args.world,
                                    assets_manager.clone(),
                                );
                                // assets_manager.serialize();
                                ui.close_menu();
                            }
                        });
                        ui.menu_button("View", |ui| {
                            for v in [
                                "Scene",
                                "Hierarchy",
                                "Inspector",
                                "Project",
                                "Console",
                                "Game",
                            ] {
                                if ui.button(v).clicked() {
                                    let a: Mutex<Box<dyn EditorWindow>> = Mutex::new(match v {
                                        "Scene" => Box::new(SceneWindow::new(self.vk.clone())),
                                        "Hierarchy" => Box::new(Hierarchy::new()),
                                        "Inspector" => Box::new(Inspector::new()),
                                        "Project" => Box::new(ProjectWindow::new()),
                                        "Console" => Box::new(ConsoleWindow::new()),
                                        "Game" => Box::new(GameWindow::new(self.vk.clone())),
                                        _ => {
                                            unreachable!()
                                        }
                                    });
                                    unsafe { self.dock.push_to_focused_leaf(a) };
                                    ui.close_menu();
                                }
                            }
                        })
                    });
                });
            }
            self.play_game = curr_playing;

            {
                let mut tab_viewer = TabViewer {
                    // image: frame_color,
                    editor_args: &mut editor_args,
                    active,
                    gui,
                    // views: &mut self.views,
                    inspectable: &mut self.inspectable,
                };
                DockArea::new(&mut self.dock)
                    .style(Style::from_egui(egui_ctx.style().as_ref()))
                    .show(egui_ctx, &mut tab_viewer);
            }
            self.play_game = editor_args.playing_game;
            editor_args.playing_game

            // egui::Window::new("Project")
            //     .default_size([200.0, 600.0])
            //     .vscroll(true)
            //     .show(&egui_ctx, |ui: &mut egui::Ui| {
            //         render_dir(ui, cur_dir, &world.sys.lock());
            //         // for entry in WalkDir::new(".") {
            //         //     ui.label(format!("{}", entry.unwrap().path().display()));
            //         // }
            //     });
        }
    }
}

pub trait EditorWindow {
    fn update(
        &mut self,
        editor_args: &mut EditorArgs,
        inspectable: &mut Option<Arc<Mutex<dyn Inspectable_>>>,
        is_focused: bool,
    ) {
    }
    fn draw(
        &mut self,
        ui: &mut Ui,
        editor_args: &mut EditorArgs,
        inspectable: &mut Option<Arc<Mutex<dyn Inspectable_>>>,
        rec: Rect,
        id: egui::Id,
        gui: &mut Gui,
    );
    fn get_name(&self) -> &str;
}
