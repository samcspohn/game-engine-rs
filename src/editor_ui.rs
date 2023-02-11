use egui::{Color32, Context, LayerId, Rounding, Ui};
use nalgebra_glm as glm;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use puffin_egui::puffin;
use spin::lazy;
use vulkano::image::{view::ImageView, AttachmentImage};
use std::{
    any::TypeId,
    collections::{HashMap, VecDeque},
    fs,
    path::PathBuf,
    sync::{atomic::Ordering, Arc},
};
use winit::platform::unix::x11::ffi::LastExtensionError;

use crate::{
    drag_drop::drag_source,
    engine::{
        transform::{Transform, Transforms},
        GameObject, Sys, World,
    },
};
use egui_dock::{DockArea, NodeIndex, Style, Tree};

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

struct TabViewer<'a>{image: egui::TextureId, world:&'a Mutex<World>, fps: &'a mut VecDeque<f32>, func: Box<dyn Fn(&str, &mut egui::Ui, &Mutex<World>, &mut VecDeque<f32>) -> ()> }

impl egui_dock::TabViewer for TabViewer<'_> {
    type Tab = String;

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
        if tab != "game" {
            // ui.label(format!("Content of {tab}"));
            (self.func)(tab,ui,self.world, self.fps);
        } else {
            ui.image(self.image, ui.available_size());
        }
    }

    fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
        (&*tab).into()
    }
}

pub fn editor_ui(world: &Mutex<World>, fps_queue: &mut VecDeque<f32>, egui_ctx: &Context, frame_color: egui::TextureId) {
    {
        static mut _selected_transforms: Lazy<HashMap<i32, bool>> =
            Lazy::new(|| HashMap::<i32, bool>::new());

        static mut _selected: Option<i32> = None;
        static mut context_menu: Option<GameObjectContextMenu> = None;
        unsafe { context_menu = None };
        // let mut world = world.lock();

        static mut dock: Lazy<Tree<String>> = Lazy::new(|| {
            let mut tree = Tree::new(vec!["game".to_owned()]);

            // You can modify the tree before constructing the dock
            let [a, b] = tree.split_left(NodeIndex::root(), 0.15, vec!["Hierarchy".to_owned(), "tabx".to_owned()]);
            let [c, _] = tree.split_right(a, 0.8, vec!["Inspector".to_owned()]);
            let [_, _] = tree.split_below(b, 0.5, vec!["Project".to_owned()]);
            let [_,_] = tree.split_below(c, 0.7, vec!["console".to_owned()]);
            tree
        });
        // static mut CUR_EULER_ANGLES: Option<glm::Vec3> = None;
        unsafe {
            // for x in dock.iter_mut() {
            //     x.
            // }
            DockArea::new(&mut dock)
                .style(Style::from_egui(egui_ctx.style().as_ref()))
                .show(egui_ctx, &mut TabViewer {image: frame_color.clone(), world, fps: fps_queue, func:
                    Box::new(|tab, ui, world: &Mutex<World>, fps_queue: &mut VecDeque<f32>| {
                        match tab {
                            "Hierarchy" => {
                                // let resp = {
                                    let mut world = world.lock();
                                    let resp = ui.scope(|ui| {

                                    let mut transforms = world.transforms.write();
                        
                                    let hierarchy_ui = |ui: &mut egui::Ui| {
                                        if fps_queue.len() > 0 {
                                            let fps: f32 = fps_queue.iter().sum::<f32>() / fps_queue.len() as f32;
                                            ui.label(format!("fps: {}", 1.0 / fps));
                                        }
                        
                                        fn transform_hierarchy_ui(
                                            transforms: &Transforms,
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
                                                // ui.scope(|ui| {
                        
                                                // ui.set_max_width(100.0);
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
                                                        // unsafe {
                                                        //     CUR_EULER_ANGLES = None;
                                                        // };
                                                    }
                        
                                                    // let resp = ui.toggle_value(
                                                    //     &mut selected_transforms
                                                    //         .get_mut(&t.id)
                                                    //         .unwrap_or(&mut this_selection),
                                                    //     format!("game object {}", t.id),
                                                    // );
                                                    let id = resp.id;
                                                    let is_being_dragged = ui.memory().is_being_dragged(id);
                                                    let pointer_released = ui.input().pointer.any_released();
                        
                                                    // drop target
                                                    // on transform
                                                    // between transforms
                                                    let height = 7.0;
                                                    let width = resp.rect.right() - resp.rect.left();
                                                    let offset = 6.0;
                                                    let between_rect = egui::Rect::from_min_size(
                                                        egui::pos2(resp.rect.left(), resp.rect.bottom() + offset - height),
                                                        egui::vec2(width, height),
                                                    );
                                                    let between = ui.rect_contains_pointer(between_rect.clone());
                                                    let a = egui::Shape::Rect(egui::epaint::RectShape::filled(
                                                        between_rect,
                                                        Rounding::same(1.0),
                                                        Color32::from_rgba_unmultiplied(255, 255, 255, 50),
                                                    ));
                                                    if between && ui.memory().is_anything_being_dragged() {
                                                        ui.painter().add(a);
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
                        
                                                    // if pointer_released { println!("id: {2}, top: {0}, botom: {1}, height: {height}", resp.rect.top(),resp.rect.bottom(), t.id); }
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
                                                        if ui.menu_button("Add Child", |ui| {}).response.clicked() {
                                                            unsafe {
                                                                context_menu =
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
                                                                context_menu =
                                                                    Some(GameObjectContextMenu::CopyGameObject(t.id))
                                                            };
                                                            ui.close_menu();
                                                        }
                                                        if ui
                                                            .menu_button("Delete Game Object", |ui| {})
                                                            .response
                                                            .clicked()
                                                        {
                                                            unsafe {
                                                                context_menu =
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
                                                                && transforms.meta[t.id as usize].lock().children.len() > 0
                                                            {
                                                                *header_open = true;
                                                            }
                                                        }
                                                        _ => {}
                                                    }
                                                }
                        
                                                for child_id in t.get_meta().lock().children.iter() {
                                                    let child = Transform {
                                                        id: *child_id,
                                                        transforms: &&transforms,
                                                    };
                                                    transform_hierarchy_ui(
                                                        transforms,
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
                        
                                        let root = Transform {
                                            id: 0,
                                            transforms: &&transforms,
                                        };
                                        let mut count = 0;
                        
                                        // unsafe {
                                            egui::ScrollArea::both()
                                                .auto_shrink([false, false])
                                                .show(ui, |ui| {
                                                    transform_hierarchy_ui(
                                                        &transforms,
                                                        &mut _selected_transforms,
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
                                if let Some(cm) = &context_menu {
                                    let g = match cm {
                                        GameObjectContextMenu::NewGameObject(t_id) => {
                                            let g = world
                                                .instantiate_with_transform_with_parent(*t_id, world.get_transform(*t_id));
                                            println!("add game object");
                                            g
                                        }
                                        GameObjectContextMenu::CopyGameObject(t_id) => {
                                            let g = world.copy_game_object(*t_id);
                                            println!("copy game object");
                                            g
                                        }
                                        GameObjectContextMenu::DeleteGameObject(t_id) => {
                                            world.delete(GameObject { t: *t_id });
                                            GameObject { t: -1 }
                                        }
                                    };
                                    if g.t >= 0 {
                                        // unsafe {
                                            _selected = Some(g.t);
                                            _selected_transforms.clear();
                                            _selected_transforms.insert(g.t, true);
                                        // }
                                    } else {
                                        // unsafe {
                                            _selected = None;
                                            _selected_transforms.clear();
                                        // }
                                    }
                                } 
                                // else if let Some(resp) = resp {
                                    resp.response.context_menu(|ui: &mut Ui| {
                                        let resp = ui.menu_button("Add Game Object", |ui| {});
                                        if resp.response.clicked() {
                                            let g = world.instantiate();
                                            unsafe {
                                                _selected = Some(g.t);
                                                _selected_transforms.clear();
                                                _selected_transforms.insert(g.t, true);
                                            }
                                            println!("add game object");
                                            ui.close_menu();
                                        }
                                        if ui.menu_button("Save", |ui| {}).response.clicked() {
                                            crate::serialize::serialize(&world);
                                            ui.close_menu();
                                        }
                                        if ui.menu_button("Load", |ui| {}).response.clicked() {
                                            crate::serialize::deserialize(&mut world);
                                            ui.close_menu();
                                        }
                                    });
                                // }
                            },
                            "Inspector" => {
                                let mut rmv: Option<(GameObject, std::any::TypeId, i32)> = None;
                                // let resp = egui::Window::new("Inspector")
                                //     .default_size([200.0, 600.0])
                                //     .vscroll(true)
                                //     .show(&egui_ctx, |ui: &mut egui::Ui| {
                                    let mut world = world.lock();
                                        if let Some(t_id) = unsafe { _selected } {
                                            let entities = world.entities.write();
                                            if let Some(ent) = &entities[t_id as usize] {
                                                // let t = Transform { id: t_id, transforms: &*world.transforms.read() };
                                                let t = &*world.transforms.read();
                                                egui::CollapsingHeader::new("Transform")
                                                    .default_open(true)
                                                    .show(ui, |ui| {
                                                        let mut pos = *t.positions[t_id as usize].lock();
                                                        let prev_pos = pos.clone();
                                                        // Ins(&mut pos).inspect("Postition", ui);
                                                        ui.horizontal(|ui| {
                                                            ui.add(egui::Label::new("Position"));
                                                            ui.add(egui::DragValue::new(&mut pos.x).speed(0.1));
                                                            ui.add(egui::DragValue::new(&mut pos.y).speed(0.1));
                                                            ui.add(egui::DragValue::new(&mut pos.z).speed(0.1));
                                                        });
                                                        if pos != prev_pos {
                                                            t.move_child(t_id, pos - prev_pos);
                                                        }
                                                        let mut rot = *t.rotations[t_id as usize].lock();
                                                        // if unsafe { CUR_EULER_ANGLES.is_none() } {
                                                        //     unsafe {
                                                        //         let rot = glm::quat_euler_angles(&rot);
                                                        //         CUR_EULER_ANGLES = Some(rot);
                                                        //     };
                                                        // }
                                                        let prev_rot = rot.clone();
                                                        // static mut CURR_OFFSETS: Option<glm::Vec3> = None;
                                                        let mut changed_rot = false;
                                                        ui.vertical(|ui| {
                                                            ui.horizontal(|ui| {
                                                                // ui.add(egui::Label::new("Rotation"));
                                                                ui.add(egui::Label::new("Rotation"));
                                                                let x = ui.add(
                                                                    egui::Button::new("< X >").sense(egui::Sense::drag()),
                                                                );
                                                                // x.sense.drag = true;
                                                                if x.dragged() && x.drag_delta().x != 0. {
                                                                    t.rotate(
                                                                        t_id,
                                                                        &glm::vec3(1., 0., 0.),
                                                                        x.drag_delta().x / 10.,
                                                                    );
                                                                }
                                                                let y = ui.add(
                                                                    egui::Button::new("< Y >").sense(egui::Sense::drag()),
                                                                );
                                                                // y.sense.drag = true;
                                                                if y.dragged() && y.drag_delta().x != 0. {
                                                                    t.rotate(
                                                                        t_id,
                                                                        &glm::vec3(0., 1., 0.),
                                                                        y.drag_delta().x / 10.,
                                                                    );
                                                                }
                                                                let z = ui.add(
                                                                    egui::Button::new("< Z >").sense(egui::Sense::drag()),
                                                                );
                                                                // z.sense.drag = true;
                                                                if z.dragged() && z.drag_delta().x != 0. {
                                                                    t.rotate(
                                                                        t_id,
                                                                        &glm::vec3(0., 0., 1.),
                                                                        z.drag_delta().x / 10.,
                                                                    );
                                                                }
                                                                // ui.add(egui::DragValue::new(&mut rot.coords.x).speed(0.1));
                                                                // ui.add(egui::DragValue::new(&mut rot.coords.y).speed(0.1));
                                                                // ui.add(egui::DragValue::new(&mut rot.coords.z).speed(0.1));
                                                            });
                                                            ui.horizontal(|ui| {
                                                                ui.add(egui::DragValue::new(&mut rot.coords.w).speed(0.1));
                                                                ui.add(egui::DragValue::new(&mut rot.coords.x).speed(0.1));
                                                                ui.add(egui::DragValue::new(&mut rot.coords.y).speed(0.1));
                                                                ui.add(egui::DragValue::new(&mut rot.coords.z).speed(0.1));
                                                            });
                                                        });
                        
                                                        // Ins(&mut rot).inspect("Rotation", ui);
                                                        // if let Some(rot) = unsafe { &mut CUR_EULER_ANGLES } {
                                                        //     if changed_rot {
                                                        //         // vec<3, T, Q> c = glm::cos(eulerAngle * T(0.5));
                                                        //         // vec<3, T, Q> s = glm::sin(eulerAngle * T(0.5));
                        
                                                        //         // this->x = s.x * c.y * c.z - c.x * s.y * s.z;
                                                        //         // this->y = c.x * s.y * c.z + s.x * c.y * s.z;
                                                        //         // this->z = c.x * c.y * s.z - s.x * s.y * c.z;
                                                        //         // this->w = c.x * c.y * c.z + s.x * s.y * s.z;
                        
                                                        //         // let c = glm::cos(&(*rot * 0.5));
                                                        //         // let s = glm::sin(&(*rot * 0.5));
                                                        //         // let q = glm::Quat {
                                                        //         //     coords: [
                                                        //         //         s.x * c.y * c.z - c.x * s.y * s.z, // x
                                                        //         //         c.x * s.y * c.z + s.x * c.y * s.z, // y
                                                        //         //         c.x * c.y * s.z - s.x * s.y * c.z, // z
                                                        //         //         c.x * c.y * c.z + s.x * s.y * s.z, // w
                                                        //         //     ].into(),
                                                        //         // };
                        
                                                        //         // let mut q = glm::Mat4::identity();
                                                        //         // q = glm::rotate(&q, rot.z, &glm::vec3(0., 0., 1.));
                                                        //         // q = glm::rotate(&q, rot.y, &glm::vec3(0., 1., 0.));
                                                        //         // q = glm::rotate(&q, rot.x, &glm::vec3(1., 0., 0.));
                                                        //         // let q = glm::to_quat(&q);
                                                        //         // let x = glm::quat_angle_axis(rot.x, &glm::vec3(1., 0., 0.));
                                                        //         // let y = glm::quat_angle_axis(rot.y, &glm::vec3(0., 1., 0.));
                                                        //         // let z = glm::quat_angle_axis(rot.z, &glm::vec3(0., 0., 1.));
                                                        //         // let q = x * y * z;
                                                        //         // *t.rotations[t_id as usize].lock() = q;
                                                        //         // t.updates[t_id as usize][1].store(true, Ordering::Relaxed);
                                                        //         t.set_rotation(t_id, q);
                                                        //     }
                                                        // }
                                                        let mut scl = *t.scales[t_id as usize].lock();
                                                        let prev_scl = scl.clone();
                                                        // Ins(&mut scl).inspect("Scale", ui);
                                                        ui.horizontal(|ui| {
                                                            ui.add(egui::Label::new("Scale"));
                                                            ui.add(egui::DragValue::new(&mut scl.x).speed(0.1));
                                                            ui.add(egui::DragValue::new(&mut scl.y).speed(0.1));
                                                            ui.add(egui::DragValue::new(&mut scl.z).speed(0.1));
                                                        });
                                                        if prev_scl != scl {
                                                            t.set_scale(t_id, scl);
                                                        }
                                                    });
                                                let mut components = ent.write();
                                                for (c_type, id) in components.iter_mut() {
                                                    if let Some(c) = world.components.get(&c_type) {
                                                        let c = c.read();
                                                        // let name: String = c.get_name().into();
                                                        ui.separator();
                                                        let _id = ui.make_persistent_id(format!("{:?}:{}", c_type, *id));
                                                        egui::collapsing_header::CollapsingState::load_with_default_open(
                                                            ui.ctx(),
                                                            _id,
                                                            true,
                                                        )
                                                        .show_header(ui, |ui| {
                                                            ui.horizontal(|ui| {
                                                                ui.heading(c.get_name());
                                                                if ui.button("delete").clicked() {
                                                                    println!("delete component");
                                                                    let g = GameObject { t: t_id };
                                                                    // world.remove_component(g, *c_type, *id)
                                                                    rmv = Some((g, *c_type, *id));
                                                                }
                                                            });
                                                        })
                                                        .body(|ui| {
                                                            let transform = Transform {
                                                                id: t_id,
                                                                transforms: &world.transforms.read(),
                                                            };
                                                            c.inspect(transform, *id, ui, &mut world.sys.lock());
                                                        });
                                                    }
                                                }
                                            }
                                        }
                                    // });
                                if let Some((g, c_type, id)) = rmv {
                                    world.remove_component(g, c_type, id)
                                }
                            },
                            "Project" => {
                                let world = world.lock();
                                use substring::Substring;
                                let cur_dir: PathBuf = "./test_project_rs".into();
                                fn render_dir(ui: &mut egui::Ui, cur_dir: PathBuf, sys: &Sys) {
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
                                            if let Some(last_slash) = label.rfind("/") {
                                                let label = label.substring(last_slash + 1, label.len());
                                                ui.label(label);
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
                                                .filter(|x| !fs::read_dir(x.path()).is_ok())
                                                .collect();
                                            dirs.sort_by_key(|dir| dir.path());
                                            files.sort_by_key(|dir| dir.path());
                                            for entry in dirs {
                                                render_dir(ui, entry.path(), sys)
                                            }
                                            for entry in files {
                                                render_dir(ui, entry.path(), sys)
                                            }
                        
                                            // ui.label("The body is always custom");
                                        });
                                    } else {
                                        if let Some(last_slash) = label.rfind("/") {
                                            let _label: String = label.substring(last_slash + 1, label.len()).into();
                                            // drag_source(ui, id, body)
                                            // let a = ui.label(label);
                                            // a.
                                            // let label = egui::Label::new(label);
                        
                                            // let a = ui.add(label);
                                            // a.id
                                            if let Some(last_dot) = label.rfind(".") {
                                                let file_ext = label.substring(last_dot, label.len());
                                                if file_ext == ".obj" {
                                                    let mut mm = sys.model_manager.lock();
                                                    if !mm.models.contains_key(&label) {
                                                        mm.from_file(&label);
                                                    }
                                                }
                                            }
                        
                                            let item_id = egui::Id::new(label.clone());
                        
                                            drag_source(ui, item_id, label, |ui| {
                                                ui.add(egui::Label::new(_label.clone()).sense(egui::Sense::click()));
                                            });
                                        }
                                    }
                                }
                                render_dir(ui, cur_dir, &world.sys.lock());
                                // if let Some(resp) = resp {
                                //     let mut compoenent_init: Option<(TypeId, i32)> = None;
                                //     resp.response.context_menu(|ui: &mut Ui| {
                                //         let resp = ui.menu_button("Add Component", |ui| {
                                //             // ui.add(egui::Button::new("text"));
                                //             for (k, c) in &world.components {
                                //                 let mut c = c.write();
                                //                 let resp = ui.add(egui::Button::new(c.get_name()));
                                //                 if resp.clicked() {
                                //                     if let Some(t_id) = unsafe { _selected } {
                                //                         let c_id = c.new_default(t_id);
                                //                         let key = c.get_type();
                                //                         compoenent_init = Some((key, c_id));
                                //                     }
                                //                     ui.close_menu();
                                //                 }
                                //             }
                                //             if let (Some(t_id), Some((key, c_id))) = (unsafe { _selected }, compoenent_init)
                                //             {
                                //                 let g = GameObject { t: t_id };
                                //                 world.add_component_id(g, key, c_id)
                                //             }
                                //         });
                                //     });
                                // }
                            }
                            _ => {}
                        }
                    })
                });
        }

      

       

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
