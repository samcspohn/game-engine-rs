use crate::{editor::inspectable::Inspectable_, engine::world::{World, entity}};
use nalgebra_glm as glm;
use parking_lot::Mutex;

pub(crate) struct GameObjectInspector {
    // world: &'a Mutex<World>,
}
pub static mut _selected: Option<i32> = None;

impl Inspectable_ for GameObjectInspector {
    fn inspect(&mut self, ui: &mut egui::Ui, world: &Mutex<World>) {
        let mut world = world.lock();
        let mut rmv: Option<(i32, u64, i32)> = None;

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
                                ui.vertical(|ui| {
                                    ui.horizontal(|ui| {
                                        ui.add(egui::Label::new("Rotation"));
                                        let x = ui.drag_angle(&mut rot.z);
                                        if x.dragged() && x.drag_delta().x != 0. {
                                            _t.rotate(
                                                &glm::vec3(1., 0., 0.),
                                                x.drag_delta().x / 10.,
                                            );
                                        }
                                        let y = ui.drag_angle(&mut rot.y);
                                        // y.sense.drag = true;
                                        if y.dragged() && y.drag_delta().x != 0. {
                                            _t.rotate(
                                                &glm::vec3(0., 1., 0.),
                                                y.drag_delta().x / 10.,
                                            );
                                        }
                                        let z = ui.drag_angle(&mut rot.x);
                                        // z.sense.drag = true;
                                        if z.dragged() && z.drag_delta().x != 0. {
                                            _t.rotate(
                                                &glm::vec3(0., 0., 1.),
                                                z.drag_delta().x / 10.,
                                            );
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
                    }
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
    }
}
