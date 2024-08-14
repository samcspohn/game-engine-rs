use std::sync::Arc;

use crate::{
    editor::inspectable::Inspectable_,
    engine::{
        prelude::utils::euler_to_quat,
        world::{entity, World},
    },
};
use egui_winit_vulkano::Gui;
use glm::{Quat, Vec3};
use nalgebra_glm as glm;
use parking_lot::Mutex;

use super::EditorWindow;

pub(crate) struct GameObjectInspector {
    // world: &'a Mutex<World>,
}
pub static mut _selected: Option<i32> = None;

impl Inspectable_ for GameObjectInspector {
    fn inspect(&mut self, ui: &mut egui::Ui, world: &mut World) -> bool {
        // let mut world = world.lock();
        let mut rmv: Option<(i32, u64, i32)> = None;
        static mut ROTATION_EULER: Option<Vec3> = None;
        static mut QUAT_PTR: Option<*mut Quat> = None;
        let mut ret = true;
        let resp = ui.scope(|ui| {
            egui::ScrollArea::both().auto_shrink([false,true]).show(ui, |ui| {

                if let Some(t_id) = unsafe { _selected } { // TODO: fix: if object is destroyed in play, fix panic on "stop"
                    if let Some(_t) = world.transforms.get(t_id) {
                        let ent = _t.entity();
                        egui::CollapsingHeader::new("Transform")
                            .default_open(true)
                            .show(ui, |ui| {
                                // let mut pos = *t.positions[t_id as usize].lock();
                                let mut pos = _t.get_position();
                                let prev_pos = pos;
                                // Ins(&mut pos).inspect("Postition", ui);
                                ui.horizontal(|ui| {
                                    ui.add(egui::Label::new("Position"));
                                    ui.add(egui::DragValue::new(&mut pos.x).speed(0.1));
                                    ui.add(egui::DragValue::new(&mut pos.y).speed(0.1));
                                    ui.add(egui::DragValue::new(&mut pos.z).speed(0.1));
                                });
                                if pos != prev_pos {
                                    _t.move_child(pos - prev_pos);
                                }
                                let rot = _t.get_rotation();
                                let mut rot = glm::quat_euler_angles(&rot);
                                if let Some(ptr) = unsafe { &mut QUAT_PTR } {
                                    if *ptr != world.transforms.rotations[t_id as usize].get() {
                                        unsafe { ROTATION_EULER = Some(rot); }
                                        {*ptr = world.transforms.rotations[t_id as usize].get(); }
                                    }
                                } else {
                                    unsafe {
                                        QUAT_PTR = Some(world.transforms.rotations[t_id as usize].get());
                                        ROTATION_EULER = Some(rot);
                                    }
                                }
                                ui.vertical(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.add(egui::Label::new("Rotation"));
                                        if let Some(rot) = unsafe { &mut ROTATION_EULER } {
                                            let x = ui.drag_angle(&mut rot.z);
                                            let y = ui.drag_angle(&mut rot.y);
                                            let z = ui.drag_angle(&mut rot.x);
                                            _t.set_rotation(&euler_to_quat(&rot));
                                        }
                                    });
                                });
                                let mut scl = _t.get_scale();
                                let prev_scl = scl;
                                // Ins(&mut scl).inspect("Scale", ui);
                                ui.horizontal(|ui| {
                                    ui.add(egui::Label::new("Scale"));
                                    ui.add(egui::DragValue::new(&mut scl.x).speed(0.1));
                                    ui.add(egui::DragValue::new(&mut scl.y).speed(0.1));
                                    ui.add(egui::DragValue::new(&mut scl.z).speed(0.1));
                                });
                                if prev_scl != scl {
                                    _t.set_scale(scl);
                                }
                            });
                        // let mut components = ent.;
                        // let components = &mut ent;
                        let mut comp_rend = |world: &World, c_type: &u64,id: i32| {
                            if let Some(c) = world.components.get(c_type) {
                                let c = c.1.write();
                                // let name: String = c.get_name().into();
                                ui.separator();
                                let _id = ui.make_persistent_id(format!("{:?}:{}", c_type, id));
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
                                            // let g = t_id;
                                            // world.remove_component(g, *c_type, *id)
                                            rmv = Some((t_id, c_type.clone(), id));
                                        }
                                    });
                                })
                                .body(|ui| {
                                    c.inspect(&_t, id, ui, &world.sys);
                                });
                            }
                        };
                        for (c_type, comps) in ent.components.iter_mut() {
                            match comps {
                                entity::Components::Id(id) => {
                                    comp_rend(&world,c_type,*id);
                                }
                                entity::Components::V(v) => {
                                    for id in v {
                                        comp_rend(&world,c_type,*id);
                                    }
                                }
                            }
                        }
                    } else {
                        ret = false;
                    }
                } else {
                    ret = false;
                }
            });
        });

        if let Some((g, c_type, id)) = rmv {
            world.remove_component(g, c_type, id)
        }
        if unsafe { _selected.is_some() } {
            let mut component_init: Option<(u64, i32)> = None;

            ui.menu_button("add component", |ui| {
                for (_k, c) in world.components.iter() {
                    let mut c = c.1.write();
                    let resp = ui.add(egui::Button::new(c.get_name()));
                    if resp.clicked() {
                        if let Some(t_id) = unsafe { _selected } {
                            let c_id = c.new_default(t_id);
                            let key = c.get_id();
                            component_init = Some((key, c_id));
                        }
                        ui.close_menu();
                    }
                }
            });
            if let (Some(t_id), Some((key, c_id))) = (unsafe { _selected }, component_init) {
                // let g = GameObject { t: t_id };
                world.add_component_id(t_id, key, c_id)
            }
        }
        ret
    }
}

pub struct Inspector {
    inspectable: Option<Arc<Mutex<dyn Inspectable_>>>,
}

impl Inspector {
    pub fn new() -> Inspector {
        Inspector { inspectable: None }
    }
}

pub(super) static mut LAST_ACTIVE: *const Inspector = std::ptr::null();
impl EditorWindow for Inspector {
    fn update(
        &mut self,
        editor_args: &mut super::EditorArgs,
        inspectable: &mut Option<Arc<Mutex<dyn Inspectable_>>>,
        is_focused: bool,
    ) {
        if is_focused {
            unsafe {
                LAST_ACTIVE = self;
            }
        }
        if unsafe { LAST_ACTIVE == self } {
            // if is_focused {
            if inspectable.is_some() {
                self.inspectable = inspectable.take();
            }
            // }
        }
    }
    fn draw(
        &mut self,
        ui: &mut egui::Ui,
        editor_args: &mut super::EditorArgs,
        inspectable: &mut Option<Arc<Mutex<dyn Inspectable_>>>,
        rec: egui::Rect,
        id: egui::Id,
        gui: &mut Gui,
    ) {
        if let Some(ins) = &mut self.inspectable {
            if !ins.lock().inspect(ui, editor_args.world) {
                self.inspectable = None;
            }
        } else {
        }
    }

    fn get_name(&self) -> &str {
        "Inspector"
    }
}
