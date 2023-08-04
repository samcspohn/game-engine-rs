use egui::{Color32, Context, Rounding, Ui, Layout, Pos2, ScrollArea};
use nalgebra_glm as glm;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use puffin_egui::puffin;
use std::{
    any::TypeId,
    collections::{HashMap, VecDeque},
    fs,
    path::{PathBuf, self},
    sync::{Arc},
};


use crate::{
    editor::{inspectable::{Inspectable, Inspectable_}, drag_drop::drag_source, editor_ui::entity_inspector::_selected}, engine::{world::{World, transform::Transform, Sys }, project::{serialize, asset_manager::AssetsManager}},
};
use egui_dock::{DockArea, NodeIndex, Style, Tree};

mod entity_inspector;
mod util;
enum TransformDrag {
    DragToTransform(i32, i32),
    DragBetweenTransform(i32, i32, bool),
}
lazy_static::lazy_static! {

    static ref DRAGGED_TRANSFORM: Mutex<i32> = Mutex::new(0);
    static ref TRANSFORM_DRAG: Mutex<Option<TransformDrag>> = Mutex::new(None);
}
enum GameObjectContextMenu {
    NewGameObject(i32),
    CopyGameObject(i32),
    DeleteGameObject(i32),
}
pub static EDITOR_ASPECT_RATIO: Lazy<Mutex<[u32;2]>> = Lazy::new(|| {Mutex::new([1920,1080])});

struct TabViewer<'a> {
    image: egui::TextureId,
    world: &'a Mutex<World>,
    fps: &'a mut VecDeque<f32>,
    // goi: &'a GameObjectInspector<'b>,
    inspectable: &'a mut Option<Arc<Mutex<dyn Inspectable_>>>,
    assets_manager: Arc<AssetsManager>,
    func: Box<dyn Fn(&str, &mut egui::Ui, &Mutex<World>, &mut VecDeque<f32>, &mut Option<Arc<Mutex<dyn Inspectable_>>>, Arc<AssetsManager>)>,
}
pub(crate) static mut PLAYING_GAME: bool = false;
impl egui_dock::TabViewer for TabViewer<'_> {
    type Tab = String;

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        if tab != "Game" {
            // ui.label(format!("Content of {tab}"));
            (self.func)(tab, ui, self.world, self.fps, self.inspectable, self.assets_manager.clone());
        } else {
            ui.horizontal_top(|ui| {
                if unsafe {PLAYING_GAME} {
                    if ui.button("Stop").clicked() {
                        println!("stop game");
                        unsafe { PLAYING_GAME = false; }
                        {
                            let mut world = self.world.lock();
                            world.clear();
                            serialize::deserialize(&mut world);
                        }
                    }    
                } else if ui.button("Play").clicked() {
                    println!("play game");
                    unsafe { PLAYING_GAME = true; }
                }
            });
            let a = ui.available_size();
            *EDITOR_ASPECT_RATIO.lock() = [a[0] as u32, a[1] as u32];
            ui.image(self.image, a);
        }
    }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        (&*tab).into()
    }
}


struct ModelInspector {
    file: String
}

impl Inspectable_ for ModelInspector {
    fn inspect(&mut self, ui: &mut egui::Ui, _world: &Mutex<World>) {
        ui.add(egui::Label::new(self.file.as_str()));
    }
}

pub fn editor_ui(
    world: &Mutex<World>,
    fps_queue: &mut VecDeque<f32>,
    egui_ctx: &Context,
    frame_color: egui::TextureId,
    assets_manager: Arc<AssetsManager>
) -> bool {
    {
        static mut _SELECTED_TRANSFORMS: Lazy<HashMap<i32, bool>> =
            Lazy::new(HashMap::<i32, bool>::new);

        static mut CONTEXT_MENU: Option<GameObjectContextMenu> = None;
        unsafe { CONTEXT_MENU = None };
        static mut INSPECTABLE: Option<Arc<Mutex<dyn Inspectable_>>> = None;
        static mut DOCK: Lazy<Tree<String>> = Lazy::new(|| {
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
            tree
        });
        unsafe {
            DockArea::new(&mut DOCK)
                .style(Style::from_egui(egui_ctx.style().as_ref()))
                .show(egui_ctx, &mut TabViewer {image: frame_color, world, fps: fps_queue, inspectable: &mut INSPECTABLE, assets_manager, func:
                    Box::new(|tab, ui, world: &Mutex<World>, fps_queue: &mut VecDeque<f32>, _ins: &mut Option<Arc<Mutex<dyn Inspectable_>>>, assets_manager: Arc<AssetsManager>| {
                        let assets_manager = assets_manager;
                        match tab {
                            "Hierarchy" => {

                                // let resp = {
                                    let _w = &world;
                                    let mut world = world.lock();
                                    let resp = ui.scope(|ui| {

                                    let transforms = &mut world.transforms;

                                    let hierarchy_ui = |ui: &mut egui::Ui| {
                                        if !fps_queue.is_empty() {
                                            let fps: f32 = fps_queue.iter().sum::<f32>() / fps_queue.len() as f32;
                                            ui.label(format!("fps: {}", 1.0 / fps));
                                        }
                                        ui.label(format!("entities: {}",transforms.active()));

                                        fn transform_hierarchy_ui(
                                            selected_transforms: &mut HashMap<i32, bool>,
                                            t: Transform,
                                            ui: &mut egui::Ui,
                                            count: &mut i32,
                                            selected: &mut Option<i32>,
                                        ) {
                                            let id = ui.make_persistent_id(format!("{}", t.id));
                                            egui::collapsing_header::CollapsingState::load_with_default_open(
                                                ui.ctx(),
                                                id,
                                                true,
                                            )
                                            .show_header(ui, |ui| {
                                                ui.horizontal(|ui| {
                                                    let label = egui::SelectableLabel::new(
                                                        *selected_transforms.get(&t.id).unwrap_or(&false),
                                                        format!("game object {}", t.id),
                                                    );
                                                    let resp = ui.add(label);

                                                    if resp.clicked() {
                                                        selected_transforms.clear();
                                                        selected_transforms.insert(t.id, true);
                                                        *selected = Some(t.id);
                                                        unsafe { INSPECTABLE = Some(Arc::new(Mutex::new(entity_inspector::GameObjectInspector {}))); }
                                                    }

                                                    let id = resp.id;
                                                    let is_being_dragged = ui.memory().is_being_dragged(id);
                                                    let pointer_released = ui.input().pointer.any_released();

                                                    let between = if ui.memory().is_anything_being_dragged() {
                                                        const HEIGHT: f32 = 7.0;
                                                        let width = resp.rect.right() - resp.rect.left();
                                                        const OFFSET: f32 = 6.0;
                                                        let between_rect = egui::Rect::from_min_size(
                                                            egui::pos2(resp.rect.left(), resp.rect.bottom() + OFFSET - HEIGHT),
                                                            egui::vec2(width, HEIGHT),
                                                        );
                                                        let between = ui.rect_contains_pointer(between_rect);
                                                        let between_rect_fill = egui::Shape::Rect(egui::epaint::RectShape::filled(
                                                            between_rect,
                                                            Rounding::same(1.0),
                                                            Color32::from_rgba_unmultiplied(255, 255, 255, 50),
                                                        ));
                                                        if between && ui.memory().is_anything_being_dragged() {
                                                            ui.painter().add(between_rect_fill);
                                                        } else if ui.rect_contains_pointer(resp.rect)
                                                            && ui.memory().is_anything_being_dragged()
                                                        {
                                                            let a = egui::Shape::Rect(egui::epaint::RectShape::filled(
                                                                resp.rect,
                                                                Rounding::same(1.0),
                                                                Color32::from_rgba_unmultiplied(255, 255, 255, 50),
                                                            ));
                                                            ui.painter().add(a);
                                                        }
                                                        between
                                                } else {
                                                    false
                                                };


                                                    if TRANSFORM_DRAG.lock().is_none() {
                                                        if between && pointer_released {
                                                            let d_t_id: i32 = *DRAGGED_TRANSFORM.lock();

                                                            if d_t_id != t.id && d_t_id != 0 {
                                                                *TRANSFORM_DRAG.lock() =
                                                                    Some(TransformDrag::DragBetweenTransform(
                                                                        d_t_id, t.id, false,
                                                                    ));
                                                                // place as sibling
                                                                println!(
                                                                    "transform {} dropped below transform {}",
                                                                    d_t_id, t.id
                                                                );
                                                            }
                                                        } else if resp.hovered() && pointer_released {
                                                            let d_t_id: i32 = *DRAGGED_TRANSFORM.lock();
                                                            if d_t_id != t.id && d_t_id != 0 {
                                                                *TRANSFORM_DRAG.lock() =
                                                                    Some(TransformDrag::DragToTransform(d_t_id, t.id));
                                                                // place as child
                                                                println!(
                                                                    "transform {} dropped on transform {}",
                                                                    d_t_id, t.id
                                                                );
                                                            }
                                                        }
                                                    }
                                                    // });

                                                    // drag source
                                                    let resp = ui.interact(resp.rect, id, egui::Sense::drag());
                                                    if resp.drag_started() {
                                                        *DRAGGED_TRANSFORM.lock() = t.id;
                                                    }
                                                    if is_being_dragged {
                                                        ui.output().cursor_icon = egui::CursorIcon::Grabbing;
                                                        // Paint the body to a new layer:
                                                        let layer_id = egui::LayerId::new(egui::Order::Tooltip, id);
                                                        let response = ui
                                                            .with_layer_id(layer_id, |ui| {
                                                                ui.label(format!("game object {}", t.id))
                                                            })
                                                            .response;
                                                        if let Some(pointer_pos) = ui.ctx().pointer_interact_pos() {
                                                            let delta = pointer_pos - response.rect.center();
                                                            ui.ctx().translate_layer(layer_id, delta);
                                                        }
                                                    }

                                                    resp.context_menu(|ui: &mut Ui| {
                                                        // let resp = ui.menu_button("Add Child", |ui| {});
                                                        if ui.menu_button("Add Child", |_ui| {}).response.clicked() {
                                                            unsafe {
                                                                CONTEXT_MENU =
                                                                    Some(GameObjectContextMenu::NewGameObject(t.id))
                                                            };
                                                            ui.close_menu();
                                                        }
                                                        if ui
                                                            .menu_button("Copy Game Object", |_| {})
                                                            .response
                                                            .clicked()
                                                        {
                                                            unsafe {
                                                                CONTEXT_MENU =
                                                                    Some(GameObjectContextMenu::CopyGameObject(t.id))
                                                            };
                                                            ui.close_menu();
                                                        }
                                                        if ui
                                                            .menu_button("Delete Game Object", |_ui| {})
                                                            .response
                                                            .clicked()
                                                        {
                                                            unsafe {
                                                                CONTEXT_MENU =
                                                                    Some(GameObjectContextMenu::DeleteGameObject(t.id))
                                                            }
                                                            ui.close_menu();
                                                        }
                                                    });
                                                });
                                                // });
                                            })
                                            .body(|ui| {
                                                if let Some(d) = &mut *TRANSFORM_DRAG.lock() {
                                                    match d {
                                                        TransformDrag::DragBetweenTransform(_, t_id, header_open) => {
                                                            if *t_id == t.id
                                                                && t.get_meta().children.len() > 0
                                                            {
                                                                *header_open = true;
                                                            }
                                                        }
                                                        _ => {}
                                                    }
                                                }

                                                for child in t.get_children() {
                                                    // let child = transforms.get_transform(*child_id);
                                                    transform_hierarchy_ui(
                                                        // transforms,
                                                        selected_transforms,
                                                        child,
                                                        ui,
                                                        count,
                                                        selected,
                                                    );
                                                    *count += 1;
                                                    if *count > 10_000 {
                                                        return;
                                                    }
                                                }
                                                // ui.label("The body is always custom");
                                            });
                                        }

                                        let root = transforms.get(0);
                                        let mut count = 0;

                                        // unsafe {
                                            egui::ScrollArea::both()
                                                .auto_shrink([false, false])
                                                .show(ui, |ui| {
                                                    transform_hierarchy_ui(
                                                        // transforms,
                                                        &mut _SELECTED_TRANSFORMS,
                                                        root,
                                                        ui,
                                                        &mut count,
                                                        &mut _selected,
                                                    );
                                                });
                                        // }
                                    };

                                    // windows

                                    *TRANSFORM_DRAG.lock() = None;
                                    puffin::profile_function!();
                                    // let mut open = true;
                                    // let resp = egui::Window::new("Hierarchy")
                                    //     .default_size([200.0, 200.0])
                                    //     .open(&mut open)
                                    //     // .vscroll(true)
                                    //     // .hscroll(true)
                                    //     .show(&egui_ctx, hierarchy_ui);
                                    hierarchy_ui(ui);

                                    let d = &*TRANSFORM_DRAG.lock();
                                    if let Some(d) = d {
                                        // (d_t_id, t_id)
                                        match d {
                                            TransformDrag::DragToTransform(d_t_id, t_id) => {
                                                transforms.adopt(*t_id, *d_t_id);
                                            }
                                            TransformDrag::DragBetweenTransform(d_t_id, t_id, header_open) => {
                                                transforms.change_place_in_hier(*t_id, *d_t_id, *header_open)
                                            }
                                        }
                                    }
                                });
                                    // resp
                                // };
                                if let Some(cm) = &CONTEXT_MENU {
                                    let e = match cm {
                                        GameObjectContextMenu::NewGameObject(t_id) => {
                                            let _t = world.transforms.get(*t_id).get_transform();
                                            let e = world
                                                .instantiate_with_transform_with_parent(*t_id, _t);
                                            println!("add game object");
                                            e
                                        }
                                        GameObjectContextMenu::CopyGameObject(t_id) => {
                                            let e = world.copy_game_object(*t_id);
                                            println!("copy game object");
                                            e
                                        }
                                        GameObjectContextMenu::DeleteGameObject(t_id) => {
                                            world.destroy(*t_id);
                                            -1
                                        }
                                    };
                                    if e >= 0 {
                                        // unsafe {
                                            _selected = Some(e);
                                            _SELECTED_TRANSFORMS.clear();
                                            _SELECTED_TRANSFORMS.insert(e, true);
                                        // }
                                    } else {
                                        // unsafe {
                                            _selected = None;
                                            _SELECTED_TRANSFORMS.clear();
                                        // }
                                    }
                                }
                                resp.response.context_menu(|ui: &mut Ui| {
                                    let resp = ui.menu_button("Add Game Object", |_ui| {});
                                    if resp.response.clicked() {
                                        let e = world.instantiate();
                                        unsafe {
                                            _selected = Some(e);
                                            _SELECTED_TRANSFORMS.clear();
                                            _SELECTED_TRANSFORMS.insert(e, true);
                                        }
                                        println!("add game object");
                                        ui.close_menu();
                                    }
                                    if ui.menu_button("Save", |_ui| {}).response.clicked() {
                                        serialize::serialize(&world);
                                        ui.close_menu();
                                    }
                                    if ui.menu_button("Load", |_ui| {}).response.clicked() {
                                        serialize::deserialize(&mut world);
                                        ui.close_menu();
                                    }
                                });
                            },
                            "Inspector" => {
                                
                                if let Some(ins) = &mut INSPECTABLE {
                                    // let mut world = world.lock();
                                    ins.lock().inspect(ui, world);
                                } else {
                                }
                            },
                            "Project" => {
                                let world = world.lock();
                                use substring::Substring;
                                let cur_dir: PathBuf = "./test_project_rs".into();
                                fn render_dir(ui: &mut egui::Ui, cur_dir: PathBuf, sys: &Sys, assets_manager: &Arc<AssetsManager>) {
                                    // let label = format!("{:?}", cur_dir);
                                    let label: String = cur_dir.clone().into_os_string().into_string().unwrap();
                                    let id = ui.make_persistent_id(label.clone());
                                    if let Ok(it) = fs::read_dir(cur_dir) {
                                        egui::collapsing_header::CollapsingState::load_with_default_open(
                                            ui.ctx(),
                                            id,
                                            false,
                                        )
                                        .show_header(ui, |ui| {
                                            if let Some(last_slash) = label.rfind(path::MAIN_SEPARATOR_STR) {
                                                let _label = label.substring(last_slash + 1, label.len());
                                                let resp = ui.label(_label);//.sense(Sense::click());
                                                resp.context_menu(|ui| {
                                                    ui.menu_button("new particle template", |ui| {
                                                        static mut TEXT: String = String::new();

                                                        unsafe { let resp2 = ui.text_edit_singleline(&mut TEXT); 
                                                            if resp2.lost_focus() && ui.input().key_pressed(egui::Key::Enter) {
                                                                ui.close_menu();
                                                                assets_manager.new_asset(format!("{label}/{TEXT}.ptem").as_str());
                                                                TEXT = "".into();
                                                            }
                                                        }
                                                    });
                                                });
                                            }
                                        })
                                        .body(|ui| {
                                            let paths: Vec<_> = it.map(|r| r.unwrap()).collect();
                                            let mut dirs: Vec<_> = paths
                                                .iter()
                                                .filter(|x| fs::read_dir(x.path()).is_ok())
                                                .collect();
                                            let mut files: Vec<_> = paths
                                                .iter()
                                                .filter(|x| fs::read_dir(x.path()).is_err())
                                                .collect();
                                            dirs.sort_by_key(|dir| dir.path());
                                            files.sort_by_key(|dir| dir.path());
                                            for entry in dirs {
                                                render_dir(ui, entry.path(), sys, assets_manager)
                                            }
                                            for entry in files {
                                                render_dir(ui, entry.path(), sys, assets_manager)
                                            }

                                            // ui.label("The body is always custom");
                                        });
                                    } else {
                                        if let Some(last_slash) = label.rfind(path::MAIN_SEPARATOR_STR) {
                                            let _label: String = label.substring(last_slash + 1, label.len()).into();
                                            let item_id = egui::Id::new(label.clone());
                                                let resp = drag_source(ui, item_id, label.clone(), move |ui| {
                                                    ui.add(egui::Label::new(_label.clone()).sense(egui::Sense::click()));
                                                });
                                                if resp.clicked() {
                                                    
                                                    unsafe { INSPECTABLE = assets_manager.inspect(label.as_str()) };
                                                }
                                            // }
                                        }
                                    }
                                }
                                render_dir(ui, cur_dir, &world.sys, &assets_manager);
                            }
                            _ => {}
                        }
                    })
                });
        }
        unsafe { PLAYING_GAME }

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
