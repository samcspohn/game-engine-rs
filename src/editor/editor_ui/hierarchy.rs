use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use egui::{Color32, Rect, Rounding, Sense, Ui};
use egui_winit_vulkano::Gui;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use puffin_egui::puffin;

use crate::{
    editor::editor_ui::{
        entity_inspector::{self, _selected},
        GameObjectContextMenu, TransformDrag, DRAGGED_TRANSFORM, TRANSFORM_DRAG,
    },
    engine::{
        prelude::Inspectable_,
        prelude::{Ins, Transform},
        project::{asset_manager::AssetsManager, file_watcher::FileWatcher},
        world::World,
    },
};

use super::{EditorArgs, EditorWindow};

pub(super) struct Hierarchy {
    _selected_transforms: HashMap<i32, bool>,
    context_menu: Option<GameObjectContextMenu>,
    // inspectable: Option<Arc<Mutex<dyn Inspectable_>>>,
    fps_queue: VecDeque<f32>,
}
impl Hierarchy {
    pub fn new() -> Self {
        Self { _selected_transforms: HashMap::new(), context_menu: None, fps_queue: VecDeque::new() }
    }
}

impl EditorWindow for Hierarchy {
    fn draw(
        &mut self,
        ui: &mut Ui,
        editor_args: &mut EditorArgs,
        inspectable: &mut Option<Arc<Mutex<dyn Inspectable_>>>,
        rec: Rect,
        id: egui::Id,
        gui: &mut Gui,
    ) {
        let mut world = &mut editor_args.world;
        self.fps_queue.push_back(world.time.dt);
        if self.fps_queue.len() > 100 {
            self.fps_queue.pop_front();
        }
        let resp = ui.interact(rec, id, Sense::click());
        resp.context_menu(|ui: &mut Ui| {
            // println!("here");
            let resp = ui.button("Add Game Object");
            if resp.clicked() {
                let e = world.create();
                unsafe {
                    _selected = Some(e);
                    self._selected_transforms.clear();
                    self._selected_transforms.insert(e, true);
                }
                println!("add game object");
                ui.close_menu();
            }
        });

        let mut clicked = false;
        let mut resp = ui.scope(|ui| {
            let mut hierarchy_ui = |ui: &mut egui::Ui| {
                fn transform_hierarchy_ui(
                    _self: &mut Hierarchy,
                    inspectable: &mut Option<Arc<Mutex<dyn Inspectable_>>>,
                    // selected_transforms: &mut HashMap<i32, bool>,
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
                                *_self._selected_transforms.get(&t.id).unwrap_or(&false),
                                format!("game object {}", t.id),
                            );
                            let resp = ui.add(label);

                            if resp.clicked() {
                                _self._selected_transforms.clear();
                                _self._selected_transforms.insert(t.id, true);
                                *selected = Some(t.id);
                                *inspectable = Some(Arc::new(Mutex::new(
                                    entity_inspector::GameObjectInspector {},
                                )));
                            }

                            let id = resp.id;
                            let is_being_dragged = ui.memory(|m| m.is_being_dragged(id));
                            let pointer_released = ui.input(|i| i.pointer.any_released());

                            let between = if ui.memory(|m| m.is_anything_being_dragged()) {
                                const HEIGHT: f32 = 7.0;
                                let width = resp.rect.right() - resp.rect.left();
                                const OFFSET: f32 = 6.0;
                                let between_rect = egui::Rect::from_min_size(
                                    egui::pos2(
                                        resp.rect.left(),
                                        resp.rect.bottom() + OFFSET - HEIGHT,
                                    ),
                                    egui::vec2(width, HEIGHT),
                                );
                                let between = ui.rect_contains_pointer(between_rect);
                                let between_rect_fill =
                                    egui::Shape::Rect(egui::epaint::RectShape::filled(
                                        between_rect,
                                        Rounding::same(1.0),
                                        Color32::from_rgba_unmultiplied(255, 255, 255, 50),
                                    ));
                                if between && ui.memory(|m| m.is_anything_being_dragged()) {
                                    ui.painter().add(between_rect_fill);
                                } else if ui.rect_contains_pointer(resp.rect)
                                    && ui.memory(|m| m.is_anything_being_dragged())
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
                                ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::Grabbing);
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
                                if ui.button("Add Child").clicked() {
                                    unsafe {
                                        _self.context_menu =
                                            Some(GameObjectContextMenu::NewGameObject(t.id))
                                    };
                                    ui.close_menu();
                                }
                                if ui.button("Copy Game Object").clicked() {
                                    unsafe {
                                        _self.context_menu =
                                            Some(GameObjectContextMenu::CopyGameObject(t.id))
                                    };
                                    ui.close_menu();
                                }
                                if ui.button("Delete Game Object").clicked() {
                                    unsafe {
                                        _self.context_menu =
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
                                    if *t_id == t.id && t.get_meta().children.len() > 0 {
                                        *header_open = true;
                                    }
                                }
                                _ => {}
                            }
                        }

                        for child in t.get_children() {
                            // let child = transforms.get_transform(*child_id);
                            transform_hierarchy_ui(_self, inspectable, child, ui, count, selected);
                            *count += 1;
                            if *count > 10_000 {
                                return;
                            }
                        }
                        // ui.label("The body is always custom");
                    });
                }

                let root = world.transforms.get(0).unwrap();
                let mut count = 0;

                // unsafe {
                let widg = egui::ScrollArea::both()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        if !self.fps_queue.is_empty() {
                            let fps: f32 =
                                self.fps_queue.iter().sum::<f32>() / self.fps_queue.len() as f32;
                            ui.label(format!("fps: {}", 1.0 / fps));
                        }
                        ui.label(format!("entities: {}", world.transforms.active()));
                        transform_hierarchy_ui(
                            // transforms,
                            self,
                            inspectable,
                            // &mut self._selected_transforms,
                            root,
                            ui,
                            &mut count,
                            unsafe { &mut _selected },
                        );
                    });
                // }
                widg
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
            world.sys.dragged_transform = *DRAGGED_TRANSFORM.lock();
            let d = &*TRANSFORM_DRAG.lock();
            if let Some(d) = d {
                // (d_t_id, t_id)
                // if let _d = *DRAGGED_TRANSFORM.lock() {
                // }
                match d {
                    TransformDrag::DragToTransform(d_t_id, t_id) => {
                        world.transforms.adopt(*t_id, *d_t_id);
                    }
                    TransformDrag::DragBetweenTransform(d_t_id, t_id, header_open) => world
                        .transforms
                        .change_place_in_hier(*t_id, *d_t_id, *header_open),
                }
            }
        });
        // resp.response.sense.click = true;
        // ui.add(resp).sense(Sense::click())
        // };
        if let Some(cm) = unsafe { &self.context_menu } {
            let e = match cm {
                GameObjectContextMenu::NewGameObject(t_id) => {
                    let _t = world.transforms.get(*t_id).unwrap().get_transform();
                    let e = world.create_with_transform_with_parent(*t_id, _t);
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
                unsafe {
                    _selected = Some(e);
                    self._selected_transforms.clear();
                    self._selected_transforms.insert(e, true);
                }
            } else {
                unsafe {
                    _selected = None;
                    self._selected_transforms.clear();
                }
            }
            self.context_menu = None;
        }
    }
    
    fn get_name(&self) -> &str {
        "Hierarchy"
    }
}
// pub(crate) fn hierarchy(
//     ui: &mut egui::Ui,
//     world: &mut World,
//     fps_queue: &mut VecDeque<f32>,
//     inspectable: &mut Option<Arc<Mutex<dyn Inspectable_>>>,
//     assets_manager: Arc<AssetsManager>,
//     file_watcher: &FileWatcher,
//     rec: Rect,
//     id: egui::Id,
// ) {
//     static mut _SELECTED_TRANSFORMS: Lazy<HashMap<i32, bool>> =
//         Lazy::new(HashMap::<i32, bool>::new);
//     static mut CONTEXT_MENU: Option<GameObjectContextMenu> = None;
//     static mut INSPECTABLE: Option<Arc<Mutex<dyn Inspectable_>>> = None;
//     unsafe { INSPECTABLE = inspectable.clone() };
//     unsafe { CONTEXT_MENU = None };
//     let resp = ui.interact(rec, id, Sense::click());
//     resp.context_menu(|ui: &mut Ui| {
//         // println!("here");
//         let resp = ui.button("Add Game Object");
//         if resp.clicked() {
//             let e = world.create();
//             unsafe {
//                 _selected = Some(e);
//                 _SELECTED_TRANSFORMS.clear();
//                 _SELECTED_TRANSFORMS.insert(e, true);
//             }
//             println!("add game object");
//             ui.close_menu();
//         }
//     });

//     let mut clicked = false;
//     let mut resp = ui.scope(|ui| {
//         let transforms = &mut world.transforms;

//         let hierarchy_ui = |ui: &mut egui::Ui| {
//             if !fps_queue.is_empty() {
//                 let fps: f32 = fps_queue.iter().sum::<f32>() / fps_queue.len() as f32;
//                 ui.label(format!("fps: {}", 1.0 / fps));
//             }
//             ui.label(format!("entities: {}", transforms.active()));

//             fn transform_hierarchy_ui(
//                 selected_transforms: &mut HashMap<i32, bool>,
//                 t: Transform,
//                 ui: &mut egui::Ui,
//                 count: &mut i32,
//                 selected: &mut Option<i32>,
//             ) {
//                 let id = ui.make_persistent_id(format!("{}", t.id));
//                 egui::collapsing_header::CollapsingState::load_with_default_open(
//                     ui.ctx(),
//                     id,
//                     true,
//                 )
//                 .show_header(ui, |ui| {
//                     ui.horizontal(|ui| {
//                         let label = egui::SelectableLabel::new(
//                             *selected_transforms.get(&t.id).unwrap_or(&false),
//                             format!("game object {}", t.id),
//                         );
//                         let resp = ui.add(label);

//                         if resp.clicked() {
//                             selected_transforms.clear();
//                             selected_transforms.insert(t.id, true);
//                             *selected = Some(t.id);
//                             unsafe {
//                                 INSPECTABLE = Some(Arc::new(Mutex::new(
//                                     entity_inspector::GameObjectInspector {},
//                                 )));
//                             }
//                         }

//                         let id = resp.id;
//                         let is_being_dragged = ui.memory(|m| m.is_being_dragged(id));
//                         let pointer_released = ui.input(|i| i.pointer.any_released());

//                         let between = if ui.memory(|m| m.is_anything_being_dragged()) {
//                             const HEIGHT: f32 = 7.0;
//                             let width = resp.rect.right() - resp.rect.left();
//                             const OFFSET: f32 = 6.0;
//                             let between_rect = egui::Rect::from_min_size(
//                                 egui::pos2(resp.rect.left(), resp.rect.bottom() + OFFSET - HEIGHT),
//                                 egui::vec2(width, HEIGHT),
//                             );
//                             let between = ui.rect_contains_pointer(between_rect);
//                             let between_rect_fill =
//                                 egui::Shape::Rect(egui::epaint::RectShape::filled(
//                                     between_rect,
//                                     Rounding::same(1.0),
//                                     Color32::from_rgba_unmultiplied(255, 255, 255, 50),
//                                 ));
//                             if between && ui.memory(|m| m.is_anything_being_dragged()) {
//                                 ui.painter().add(between_rect_fill);
//                             } else if ui.rect_contains_pointer(resp.rect)
//                                 && ui.memory(|m| m.is_anything_being_dragged())
//                             {
//                                 let a = egui::Shape::Rect(egui::epaint::RectShape::filled(
//                                     resp.rect,
//                                     Rounding::same(1.0),
//                                     Color32::from_rgba_unmultiplied(255, 255, 255, 50),
//                                 ));
//                                 ui.painter().add(a);
//                             }
//                             between
//                         } else {
//                             false
//                         };

//                         if TRANSFORM_DRAG.lock().is_none() {
//                             if between && pointer_released {
//                                 let d_t_id: i32 = *DRAGGED_TRANSFORM.lock();

//                                 if d_t_id != t.id && d_t_id != 0 {
//                                     *TRANSFORM_DRAG.lock() = Some(
//                                         TransformDrag::DragBetweenTransform(d_t_id, t.id, false),
//                                     );
//                                     // place as sibling
//                                     println!(
//                                         "transform {} dropped below transform {}",
//                                         d_t_id, t.id
//                                     );
//                                 }
//                             } else if resp.hovered() && pointer_released {
//                                 let d_t_id: i32 = *DRAGGED_TRANSFORM.lock();
//                                 if d_t_id != t.id && d_t_id != 0 {
//                                     *TRANSFORM_DRAG.lock() =
//                                         Some(TransformDrag::DragToTransform(d_t_id, t.id));
//                                     // place as child
//                                     println!("transform {} dropped on transform {}", d_t_id, t.id);
//                                 }
//                             }
//                         }
//                         // });

//                         // drag source
//                         let resp = ui.interact(resp.rect, id, egui::Sense::drag());
//                         if resp.drag_started() {
//                             *DRAGGED_TRANSFORM.lock() = t.id;
//                         }
//                         if is_being_dragged {
//                             ui.output_mut(|o| o.cursor_icon = egui::CursorIcon::Grabbing);
//                             // Paint the body to a new layer:
//                             let layer_id = egui::LayerId::new(egui::Order::Tooltip, id);
//                             let response = ui
//                                 .with_layer_id(layer_id, |ui| {
//                                     ui.label(format!("game object {}", t.id))
//                                 })
//                                 .response;
//                             if let Some(pointer_pos) = ui.ctx().pointer_interact_pos() {
//                                 let delta = pointer_pos - response.rect.center();
//                                 ui.ctx().translate_layer(layer_id, delta);
//                             }
//                         }

//                         resp.context_menu(|ui: &mut Ui| {
//                             // let resp = ui.menu_button("Add Child", |ui| {});
//                             if ui.button("Add Child").clicked() {
//                                 unsafe {
//                                     CONTEXT_MENU = Some(GameObjectContextMenu::NewGameObject(t.id))
//                                 };
//                                 ui.close_menu();
//                             }
//                             if ui.button("Copy Game Object").clicked() {
//                                 unsafe {
//                                     CONTEXT_MENU = Some(GameObjectContextMenu::CopyGameObject(t.id))
//                                 };
//                                 ui.close_menu();
//                             }
//                             if ui.button("Delete Game Object").clicked() {
//                                 unsafe {
//                                     CONTEXT_MENU =
//                                         Some(GameObjectContextMenu::DeleteGameObject(t.id))
//                                 }
//                                 ui.close_menu();
//                             }
//                         });
//                     });
//                     // });
//                 })
//                 .body(|ui| {
//                     if let Some(d) = &mut *TRANSFORM_DRAG.lock() {
//                         match d {
//                             TransformDrag::DragBetweenTransform(_, t_id, header_open) => {
//                                 if *t_id == t.id && t.get_meta().children.len() > 0 {
//                                     *header_open = true;
//                                 }
//                             }
//                             _ => {}
//                         }
//                     }

//                     for child in t.get_children() {
//                         // let child = transforms.get_transform(*child_id);
//                         transform_hierarchy_ui(
//                             // transforms,
//                             selected_transforms,
//                             child,
//                             ui,
//                             count,
//                             selected,
//                         );
//                         *count += 1;
//                         if *count > 10_000 {
//                             return;
//                         }
//                     }
//                     // ui.label("The body is always custom");
//                 });
//             }

//             let root = transforms.get(0).unwrap();
//             let mut count = 0;

//             // unsafe {
//             let widg = egui::ScrollArea::both()
//                 .auto_shrink([false, false])
//                 .show(ui, |ui| {
//                     transform_hierarchy_ui(
//                         // transforms,
//                         unsafe { &mut _SELECTED_TRANSFORMS },
//                         root,
//                         ui,
//                         &mut count,
//                         unsafe { &mut _selected },
//                     );
//                 });
//             // }
//             widg
//         };

//         // windows

//         *TRANSFORM_DRAG.lock() = None;
//         puffin::profile_function!();
//         // let mut open = true;
//         // let resp = egui::Window::new("Hierarchy")
//         //     .default_size([200.0, 200.0])
//         //     .open(&mut open)
//         //     // .vscroll(true)
//         //     // .hscroll(true)
//         //     .show(&egui_ctx, hierarchy_ui);
//         hierarchy_ui(ui);

//         world.sys.dragged_transform = *DRAGGED_TRANSFORM.lock();
//         let d = &*TRANSFORM_DRAG.lock();
//         if let Some(d) = d {
//             // (d_t_id, t_id)
//             // if let _d = *DRAGGED_TRANSFORM.lock() {
//             // }
//             match d {
//                 TransformDrag::DragToTransform(d_t_id, t_id) => {
//                     transforms.adopt(*t_id, *d_t_id);
//                 }
//                 TransformDrag::DragBetweenTransform(d_t_id, t_id, header_open) => {
//                     transforms.change_place_in_hier(*t_id, *d_t_id, *header_open)
//                 }
//             }
//         }
//     });
//     // resp.response.sense.click = true;
//     // ui.add(resp).sense(Sense::click())
//     // };
//     if let Some(cm) = unsafe { &CONTEXT_MENU } {
//         let e = match cm {
//             GameObjectContextMenu::NewGameObject(t_id) => {
//                 let _t = world.transforms.get(*t_id).unwrap().get_transform();
//                 let e = world.create_with_transform_with_parent(*t_id, _t);
//                 println!("add game object");
//                 e
//             }
//             GameObjectContextMenu::CopyGameObject(t_id) => {
//                 let e = world.copy_game_object(*t_id);
//                 println!("copy game object");
//                 e
//             }
//             GameObjectContextMenu::DeleteGameObject(t_id) => {
//                 world.destroy(*t_id);
//                 -1
//             }
//         };
//         if e >= 0 {
//             unsafe {
//                 _selected = Some(e);
//                 _SELECTED_TRANSFORMS.clear();
//                 _SELECTED_TRANSFORMS.insert(e, true);
//             }
//         } else {
//             unsafe {
//                 _selected = None;
//                 _SELECTED_TRANSFORMS.clear();
//             }
//         }
//     }
//     *inspectable = unsafe { INSPECTABLE.clone() };
// }
