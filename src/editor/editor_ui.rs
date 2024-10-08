use crate::{
    editor::{editor_ui::entity_inspector::_SELECTED, inspectable::Inspectable_},
    engine::{
        audio::component::{AudioListener, AudioSource},
        input::Input,
        particles::{component::ParticleEmitter, particles::ParticlesSystem},
        physics::{collider::_Collider, rigid_body::_RigidBody},
        prelude::VulkanManager,
        project::{asset_manager::AssetsManager, file_watcher::FileWatcher, serialize, Project},
        rendering::{
            camera::Camera,
            component::{Renderer, RendererManager},
            lighting::lighting::{Light, LightingSystem},
        },
        world::{
            transform::{TRANSFORMS, TRANSFORM_MAP},
            World,
        },
    },
};
use console::ConsoleWindow;
use egui::{menu, Color32, Context, Layout, Pos2, Rect, Rounding, ScrollArea, Sense, Ui};
use egui_winit_vulkano::Gui;
use entity_inspector::{Inspector, LAST_ACTIVE, QUAT_PTR, ROTATION_EULER};
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
    env, fs,
    path::{self, Path, PathBuf},
    process::Command,
    rc::Rc,
    sync::Arc,
};
use winit::event::VirtualKeyCode;

use egui_dock::{DockArea, NodeIndex, Style, TabIndex, Tree};

mod asset_browser;
mod console;
mod entity_inspector;
mod game_window;
mod hierarchy;
mod key_shortcuts;
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
    type Tab = Box<dyn EditorWindow>;

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        let rec = ui.max_rect();
        // let id = ui.id();
        // if tab.as_ref() as *const dyn EditorWindow as *const () == self.active {
        //     println!("active tab is {}", tab.get_name());
        // }
        // let mut tab = tab.lock();
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
        // let tab = tab.lock();
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub struct Editor {
    vk: Arc<VulkanManager>,
    editor_window_dim: [u32; 2],
    dragged_transform: Mutex<i32>,
    transform_drag: Mutex<Option<TransformDrag>>,
    inspectable: Option<Arc<Mutex<dyn Inspectable_>>>,
    dock: Tree<Box<dyn EditorWindow>>,
    views: Vec<Box<dyn EditorWindow>>,
    play_game: bool,
    scene_window: NodeIndex,
    game_window: NodeIndex,
    pub keyboard_shortcuts: Arc<key_shortcuts::KeyShortcuts>,
}

pub struct EditorArgs<'a> {
    pub engine_dir: String,
    pub world: &'a mut World,
    pub project: &'a mut Project,
    pub assets_manager: Arc<AssetsManager>,
    pub file_watcher: &'a mut FileWatcher,
    pub particle_system: &'a Arc<ParticlesSystem>,
    pub light_system: &'a Arc<LightingSystem>,
    pub shortcuts: Arc<key_shortcuts::KeyShortcuts>,
    pub input: &'a Input,
    // pub render_system: &'a mut RendererManager,
    pub playing_game: bool,
}

impl Editor {
    pub fn new(vk: Arc<VulkanManager>) -> Self {
        let mut tree: Tree<Box<dyn EditorWindow>> = Tree::new(vec![
            Box::new(SceneWindow::new(vk.clone())),
            Box::new(GameWindow::new(vk.clone())),
        ]);

        // You can modify the tree before constructing the dock
        let [a, b] = tree.split_left(NodeIndex::root(), 0.15, vec![Box::new(Hierarchy::new())]);
        let ins = Box::new(Inspector::new());
        unsafe { LAST_ACTIVE = ins.as_ref() };
        let [c, _] = tree.split_right(a, 0.8, vec![ins]);
        let [_, _] = tree.split_below(b, 0.5, vec![Box::new(ProjectWindow::new())]);
        let [_, _] = tree.split_below(c, 0.7, vec![Box::new(ConsoleWindow::new())]);
        let mut shortcuts = key_shortcuts::KeyShortcuts::new();
        shortcuts.add_key_shortcut("Play", VirtualKeyCode::F5, vec![]);
        shortcuts.add_key_shortcut("Stop", VirtualKeyCode::F5, vec![VirtualKeyCode::LShift]);
        //
        shortcuts.add_key_shortcut("Save", VirtualKeyCode::S, vec![VirtualKeyCode::LControl]);
        shortcuts.add_key_shortcut(
            "New Scene",
            VirtualKeyCode::N,
            vec![VirtualKeyCode::LControl],
        );
        shortcuts.add_key_shortcut(
            "Open Scene",
            VirtualKeyCode::O,
            vec![VirtualKeyCode::LControl],
        );
        //
        shortcuts.add_key_shortcut("Copy", VirtualKeyCode::C, vec![VirtualKeyCode::LControl]);
        shortcuts.add_key_shortcut("Paste", VirtualKeyCode::V, vec![VirtualKeyCode::LControl]);
        shortcuts.add_key_shortcut("Delete", VirtualKeyCode::Delete, vec![]);
        shortcuts.add_key_shortcut(
            "Duplicate",
            VirtualKeyCode::D,
            vec![VirtualKeyCode::LControl],
        );

        Self {
            vk,
            editor_window_dim: [1920, 1080],
            dragged_transform: Mutex::new(-1),
            transform_drag: Mutex::new(None),
            inspectable: None,
            dock: tree,
            views: vec![Box::new(Hierarchy::new())],
            play_game: false,
            scene_window: NodeIndex(1),
            game_window: NodeIndex(2),
            keyboard_shortcuts: Arc::new(shortcuts),
        }
    }
    fn save(editor_args: &mut EditorArgs) {
        while editor_args.project.working_scene == "" {
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
                editor_args.project.working_scene = file.into();
            }
        }
        serialize::serialize(&editor_args.world, &editor_args.project.working_scene);
        editor_args.project.save_project(
            &editor_args.file_watcher,
            &editor_args.world,
            editor_args.assets_manager.clone(),
        );
    }
    fn open(editor_args: &mut EditorArgs) {
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
    }
    fn new_scene(editor_args: &mut EditorArgs) {
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
    }
    pub fn editor_ui(
        &mut self,
        mut editor_args: EditorArgs,
        egui_ctx: &Context,
        gui: &mut Gui,
        curr_playing: bool,
    ) -> bool {
        {
            let _active = self
                .dock
                .find_active_focused()
                .and_then(|p| {
                    let a: *const dyn EditorWindow = &(*p.1.as_ref());
                    let b: *const () = a as *const ();
                    Some(b)
                })
                .unwrap_or(std::ptr::null());

            for node in self.dock.iter_mut() {
                if let egui_dock::Node::Leaf { tabs, .. } = node {
                    for tab in tabs {
                        let is_active =
                            tab.as_ref() as *const dyn EditorWindow as *const () == _active;
                        tab.update(&mut editor_args, &mut self.inspectable, is_active);
                    }
                }
            }
            {
                // let world = editor_args.world;
                let assets_manager = editor_args.assets_manager.clone();
                // let file_watcher = editor_args.file_watcher;

                egui::TopBottomPanel::top("game engine").show(egui_ctx, |ui| {
                    menu::bar(ui, |ui| {
                        ui.menu_button("File", |ui| {
                            ui.menu_button("Project", |ui| {
                                if ui.button("New Project").clicked() {
                                    let folder = FileDialog::new()
                                        .add_filter("scene", &["scene"])
                                        .set_directory(std::env::current_dir().unwrap())
                                        .pick_folder();
                                    if let Some(folder) = folder {
                                        let folder = folder
                                            // .strip_prefix(std::env::current_dir().unwrap())
                                            // .unwrap()
                                            .as_os_str()
                                            .to_str()
                                            .unwrap();
                                        env::set_current_dir(folder).unwrap();
                                        let project_name = folder.split("/").last().unwrap();
                                        let mut com = Command::new("cargo")
                                            .args(&["init", "--lib", project_name])
                                            // .env("RUSTFLAGS", "-Z threads=16")
                                            .spawn()
                                            .unwrap();
                                        let _result = com.wait().unwrap();
                                        let mut com = Command::new("cargo")
                                            .args([
                                                "add",
                                                "game_engine",
                                                "--path",
                                                editor_args.engine_dir.as_str(),
                                            ])
                                            .spawn()
                                            .unwrap();
                                        let _result = com.wait().unwrap();
                                        fs::write(
                                            format!("{}/config.toml", folder).as_str(),
                                            "MAX_PARTICLES = 10_000_000",
                                        )
                                        .unwrap();
                                        // let project = Project::default();
                                        // let engine_dir = std::fs::canonicalize(&editor_args.engine_dir).unwrap();
                                        println!("{:?}", editor_args.engine_dir);
                                        let project_dir = std::env::current_dir().unwrap();
                                        assert!(env::set_current_dir(&Path::new(
                                            &editor_args.engine_dir
                                        ))
                                        .is_ok()); // procedurally generate cube/move cube to built in assets
                                        let com = Command::new(
                                            format!(
                                                "{}/target/release/game-engine-rs.exe",
                                                editor_args.engine_dir
                                            )
                                            .as_str(),
                                        )
                                        .arg(folder)
                                        .spawn();
                                        assert!(
                                            env::set_current_dir(&Path::new(&project_dir)).is_ok()
                                        );

                                        // let project = Project::default();
                                        // editor_args.file_watcher.files = project.files.clone();
                                        // unsafe { editor_args.assets_manager.clear() };
                                        // editor_args.assets_manager.deserialize(&project.assets);
                                        // editor_args.file_watcher.init(editor_args.assets_manager.clone());
                                        // *editor_args.project = project;
                                        // editor_args.light_system.regen();
                                        // editor_args.particle_system.regen();
                                        // editor_args.render_system.regen();

                                        // editor_args.world.regen();
                                        // unsafe {
                                        //     TRANSFORMS = &mut editor_args.world.transforms;
                                        //     TRANSFORM_MAP = &mut editor_args.world.transform_map;
                                        // }
                                        // {

                                        //     let world = &mut editor_args.world;
                                        //     world.register::<Renderer>(false, false, false);
                                        //     world.register::<ParticleEmitter>(false, false, false);
                                        //     world.register::<Camera>(true, false, false);
                                        //     world.register::<_Collider>(false, false, false);
                                        //     world.register::<_RigidBody>(false, false, false);
                                        //     world.register::<Light>(false, false, false);
                                        //     world.register::<AudioSource>(true, false, false);
                                        //     world.register::<AudioListener>(true, false, false);
                                        //     //
                                        //     // world.register::<TerrainEng>(true, false, true);
                                        // }
                                        // // *editor_args.file_watcher =
                                        // //     file_watcher::FileWatcher::new(folder);
                                        // // unsafe {
                                        // //     editor_args.assets_manager.clear();
                                        // // }
                                        // // editor_args.world.regen();
                                        // editor_args.project.save_project(
                                        //     &editor_args.file_watcher,
                                        //     &editor_args.world,
                                        //     assets_manager.clone(),
                                        // );
                                        unsafe {
                                            _SELECTED = None;
                                            ROTATION_EULER = None;
                                            QUAT_PTR = None;
                                        }

                                        // let mut project = Project::new(folder.into());
                                        // project.save_project(
                                        //     file_watcher,
                                        //     &editor_args.world,
                                        //     assets_manager.clone(),
                                        // );
                                        // editor_args.project.load_project(editor_args.file_watcher, editor_args.world, assets_manager.clone());
                                    }
                                    ui.close_menu();
                                }
                                if ui.button("Open Project").clicked() {
                                    let folder = FileDialog::new()
                                        .set_directory(std::env::current_dir().unwrap())
                                        .pick_folder();
                                    if let Some(folder) = folder {
                                        let folder = folder.as_os_str().to_str().unwrap();
                                        // let project = Project::default();
                                        // let engine_dir = std::fs::canonicalize(&editor_args.engine_dir).unwrap();
                                        println!("{:?}", editor_args.engine_dir);
                                        let project_dir = std::env::current_dir().unwrap();
                                        assert!(env::set_current_dir(&Path::new(
                                            &editor_args.engine_dir
                                        ))
                                        .is_ok()); // procedurally generate cube/move cube to built in assets
                                        let com = Command::new(
                                            format!(
                                                "{}/target/release/game-engine-rs.exe",
                                                editor_args.engine_dir
                                            )
                                            .as_str(),
                                        )
                                        .arg(folder)
                                        .spawn();
                                        assert!(
                                            env::set_current_dir(&Path::new(&project_dir)).is_ok()
                                        );
                                        // env::set_current_dir(folder).unwrap();
                                        // editor_args.project.load_project(
                                        //     editor_args.file_watcher,
                                        //     editor_args.world,
                                        //     assets_manager.clone(),
                                        // );
                                    }
                                    ui.close_menu();
                                }
                            });

                            ui.separator();
                            if ui.button("New scene").clicked() {
                                Editor::new_scene(&mut editor_args);
                                ui.close_menu();
                            }
                            if ui.button("Open").clicked() {
                                Editor::open(&mut editor_args);
                                ui.close_menu();
                            }
                            if ui.button("Save").clicked() {
                                Editor::save(&mut editor_args);
                                ui.close_menu();
                            }
                        });
                        ui.menu_button("View", |ui| {
                            for view in [
                                (
                                    "Scene",
                                    Box::new(SceneWindow::new(self.vk.clone()))
                                        as Box<dyn EditorWindow>,
                                ),
                                ("Hierarchy", Box::new(Hierarchy::new())),
                                ("Inspector", Box::new(Inspector::new())),
                                ("Project", Box::new(ProjectWindow::new())),
                                ("Console", Box::new(ConsoleWindow::new())),
                                ("Game", Box::new(GameWindow::new(self.vk.clone()))),
                            ] {
                                if ui.button(view.0).clicked() {
                                    unsafe { self.dock.push_to_focused_leaf(view.1) };
                                    ui.close_menu();
                                }
                            }
                        })
                    });
                });
            }

            self.play_game = editor_args.playing_game;

            {
                let mut tab_viewer = TabViewer {
                    // image: frame_color,
                    editor_args: &mut editor_args,
                    active: _active,
                    gui,
                    // views: &mut self.views,
                    inspectable: &mut self.inspectable,
                };
                DockArea::new(&mut self.dock)
                    .style(Style::from_egui(egui_ctx.style().as_ref()))
                    .show(egui_ctx, &mut tab_viewer);
            }
            if editor_args
                .shortcuts
                .get_key_shortcut("Stop", &editor_args.input)
            {
                editor_args.playing_game = false;
                // *self.vk.hide_cursor.lock() = false;
                // *self.vk.cursor_pos.lock() = None;
            }
            else if editor_args
                .shortcuts
                .get_key_shortcut("Play", &editor_args.input)
            {
                editor_args.playing_game = true;
                editor_args.world.begin_play();
            }
            if editor_args
                .shortcuts
                .get_key_shortcut("Save", &editor_args.input)
            {
                Editor::save(&mut editor_args);
            } else if editor_args
                .shortcuts
                .get_key_shortcut("Open", &editor_args.input)
            {
                Editor::open(&mut editor_args);
            } else if editor_args
                .shortcuts
                .get_key_shortcut("New Scene", &editor_args.input)
            {
                Editor::new_scene(&mut editor_args);
            }
            // if !editor_args.playing_game {
            //     match self
            //         .vk
            //         .window()
            //         .set_cursor_grab(winit::window::CursorGrabMode::None)
            //     {
            //         Ok(_) => {}
            //         Err(e) => {}
            //     }
            //     self.vk.window().set_cursor_visible(true);
            //     // println!("stop game");
            // }
            if editor_args.playing_game != self.play_game {

                if let Some((node_idx, t_idx)) =
                    self.dock.iter_mut().enumerate().find_map(|(n_idx, node)| {
                        if let egui_dock::Node::Leaf { tabs, .. } = node {
                            tabs.iter().enumerate().find_map(|(tab_idx, tab)| {
                                if (editor_args.playing_game && tab.get_name() == "Game")
                                    || (!editor_args.playing_game && tab.get_name() == "Scene")
                                {
                                    Some((NodeIndex(n_idx), TabIndex(tab_idx)))
                                } else {
                                    None
                                }
                            })
                        } else {
                            None
                        }
                    })
                {
                    self.dock.set_active_tab(node_idx, t_idx);
                }
            }

            self.play_game = editor_args.playing_game;
            editor_args.playing_game
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
