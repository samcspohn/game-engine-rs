use crate::{
    editor::{editor_ui::entity_inspector::_selected, inspectable::Inspectable_},
    engine::{
        project::{
            asset_manager::AssetsManager,
            file_watcher::{self, FileWatcher},
            serialize, Project,
        },
        utils,
        world::{transform::Transform, Sys, World},
        Engine,
    },
};
use egui::{menu, Color32, Context, Layout, Pos2, Rect, Rounding, ScrollArea, Sense, Ui};
use hierarchy::Hierarchy;
use nalgebra_glm as glm;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, MutexGuard};
use puffin_egui::puffin;
use rayon::str::EncodeUtf16;
use rfd::FileDialog;
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
    image: egui::TextureId,
    editor_args: &'a mut EditorArgs<'b>,
    // editor: &'a mut Editor,
    // goi: &'a GameObjectInspector<'b>,
    views: &'a mut Vec<Box<dyn EditorWindow>>,
    inspectable: &'a mut Option<Arc<Mutex<dyn Inspectable_>>>,
}
// pub(crate) static mut PLAYING_GAME: bool = false;
impl egui_dock::TabViewer for TabViewer<'_, '_> {
    type Tab = String;

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        if tab != "Game" {
            // ui.label(format!("Content of {tab}"));
            let rec = ui.max_rect();
            let id = ui.id();
            match tab.as_str() {
                "Hierarchy" => {
                    // self.editor.draw(0, self.engine, ui, rec, id);
                    // let mut view = self.editor.views[0].get_mut();
                    // // let _v = view.borrow_mut();
                    // view.draw(ui, self.engine, self.editor, rec, id);
                    self.views[0].draw(ui, &mut self.editor_args, self.inspectable, rec, id);
                }
                "Inspector" => {
                    if let Some(ins) = &mut self.inspectable {
                        ins.lock().inspect(ui, self.editor_args.world);
                    } else {
                    }
                }
                "Project" => {
                    let assets_manager = self.editor_args.assets_manager.clone();
                    let file_watcher = &self.editor_args.file_watcher;
                    project::project(
                        ui,
                        self.editor_args.world,
                        self.inspectable,
                        assets_manager,
                        file_watcher,
                        rec,
                        id,
                    )
                }
                _ => {}
            }
        } else {
            ui.horizontal_top(|ui| {
                if self.editor_args.playing_game {
                    if ui.button("Stop").clicked() {
                        println!("stop game");
                        self.editor_args.playing_game = false;

                        // {
                        //     // let mut world = self.world;
                        //     self.world.clear();
                        //     serialize::deserialize(&mut self.world);
                        // }
                    }
                } else if ui.button("Play").clicked() {
                    println!("play game");
                    self.editor_args.playing_game = true;
                    self.editor_args.world.begin_play();
                    // {
                    //     // let mut world = self.world;
                    //     self.world.clear();
                    //     serialize::deserialize(&mut self.world);
                    // }
                }
            });
            let a = ui.available_size();
            *EDITOR_WINDOW_DIM.lock() = [a[0] as u32, a[1] as u32];
            ui.image(self.image, a);
        }
    }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        (&*tab).into()
    }
}

struct ModelInspector {
    file: String,
}

impl Inspectable_ for ModelInspector {
    fn inspect(&mut self, ui: &mut egui::Ui, _world: &mut World) {
        ui.add(egui::Label::new(self.file.as_str()));
    }
}

pub struct Editor {
    editor_window_dim: [u32; 2],
    dragged_transform: Mutex<i32>,
    transform_drag: Mutex<Option<TransformDrag>>,
    inspectable: Option<Arc<Mutex<dyn Inspectable_>>>,
    dock: Tree<String>,
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
    pub fn new() -> Self {
        let mut tree = Tree::new(vec!["Game".to_owned()]);

        // You can modify the tree before constructing the dock
        let [a, b] = tree.split_left(
            NodeIndex::root(),
            0.15,
            vec!["Hierarchy".to_owned(), "tabx".to_owned()],
        );
        let [c, _] = tree.split_right(a, 0.8, vec!["Inspector".to_owned()]);
        let [_, _] = tree.split_below(b, 0.5, vec!["Project".to_owned()]);
        let [_, _] = tree.split_below(c, 0.7, vec!["console".to_owned()]);
        Self {
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
        frame_color: egui::TextureId,
        curr_playing: bool,
    ) -> bool {
        {
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
                                    .set_directory(".")
                                    .save_file();
                                if let Some(file) = files {
                                    let file = file.as_os_str().to_str().unwrap();
                                    serialize::serialize_new(file);
                                    serialize::deserialize(editor_args.world, file);
                                    editor_args.project.working_scene = file.into();
                                }
                                ui.close_menu();
                            }
                            if ui.button("Open").clicked() {
                                // project?
                                let files = FileDialog::new()
                                    .add_filter("scene", &["scene"])
                                    .set_directory(".")
                                    .pick_file();
                                if let Some(file) = files {
                                    let file = file.as_os_str().to_str().unwrap();
                                    serialize::deserialize(editor_args.world, file);
                                    editor_args.project.working_scene = file.into();
                                }
                                ui.close_menu();
                            }
                            if ui.button("Save").clicked() {
                                serialize::serialize(&editor_args.world, &editor_args.project.working_scene);
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
                            for v in ["Game", "Hierarchy", "Inspector", "Project", "console"] {
                                if ui.button(v).clicked() {
                                    unsafe { self.dock.push_to_focused_leaf(v.into()) };
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
                    image: frame_color,
                    editor_args: &mut editor_args,
                    views: &mut self.views,
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
    fn draw(
        &mut self,
        ui: &mut Ui,
        editor_args: &mut EditorArgs,
        inspectable: &mut Option<Arc<Mutex<dyn Inspectable_>>>,
        rec: Rect,
        id: egui::Id,
    );
}
