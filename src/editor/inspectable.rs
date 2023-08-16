
use nalgebra_glm as glm;
use parking_lot::Mutex;

use crate::engine::world::{transform::Transform, Sys, World};

pub trait Inspectable {
    fn inspect(&mut self, transform: &Transform, id: i32, ui: &mut egui::Ui, sys: &Sys);
}

pub trait Inspectable_ {
    fn inspect(&mut self, ui: &mut egui::Ui, world: &Mutex<World>);
}
pub struct Ins<'a, T>(pub &'a mut T);
pub trait Inpsect {
    fn inspect(&mut self, name: &str, ui: &mut egui::Ui, sys: &Sys) -> bool;
}

impl<'a> Inpsect for Ins<'a, bool> {
    fn inspect(&mut self, name: &str, ui: &mut egui::Ui, _sys: &Sys) -> bool{
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(name));
            ui.add(egui::Checkbox::new(self.0, ""));
        }).response.changed()
    }
}

impl<'a> Inpsect for Ins<'a, i32> {
    fn inspect(&mut self, name: &str, ui: &mut egui::Ui, _sys: &Sys)  -> bool {
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(name));
            ui.add(egui::DragValue::new(self.0));
        }).response.changed()
    }
}

impl<'a> Inpsect for Ins<'a, f32> {
    fn inspect(&mut self, name: &str, ui: &mut egui::Ui, _sys: &Sys) -> bool {
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(name));
            ui.add(egui::DragValue::new(self.0).speed(0.1));
        }).response.changed()
    }
}
impl<'a> Inpsect for Ins<'a, glm::Vec2> {
    fn inspect(&mut self, name: &str, ui: &mut egui::Ui, _sys: &Sys) -> bool {
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(name));
            ui.add(egui::DragValue::new(&mut self.0.x).speed(0.1));
            ui.add(egui::DragValue::new(&mut self.0.y).speed(0.1));
        }).response.changed()
    }
}
impl<'a> Inpsect for Ins<'a, glm::Vec3> {
    fn inspect(&mut self, name: &str, ui: &mut egui::Ui, _sys: &Sys) -> bool {
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(name));
            ui.add(egui::DragValue::new(&mut self.0.x).speed(0.1));
            ui.add(egui::DragValue::new(&mut self.0.y).speed(0.1));
            ui.add(egui::DragValue::new(&mut self.0.z).speed(0.1));
        }).response.changed()
    }
}
impl<'a> Inpsect for Ins<'a, glm::Quat> {
    fn inspect(&mut self, name: &str, ui: &mut egui::Ui, _sys: &Sys) -> bool {
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(name));
            ui.add(egui::DragValue::new(&mut self.0.coords.w).speed(0.1));
            ui.add(egui::DragValue::new(&mut self.0.coords.x).speed(0.1));
            ui.add(egui::DragValue::new(&mut self.0.coords.y).speed(0.1));
            ui.add(egui::DragValue::new(&mut self.0.coords.z).speed(0.1));
        }).response.changed()
    }
}
// impl<'a> Inpsect for Wrapper<'a, f32> {
//     fn inspect(&mut self, name: &str, ui: &mut egui::Ui) {
//         ui.add(egui::Label::new(name));
//         ui.add(egui::DragValue::new(self.0));
//     }
// }
// impl<'a> Inpsect for Wrapper<'a, f32> {
//     fn inspect(&mut self, name: &str, ui: &mut egui::Ui) {
//         ui.add(egui::Label::new(name));
//         ui.add(egui::DragValue::new(self.0));
//     }
// }
// pub fn inspect<T = bool>(ui: &mut Ui,name: &str, t: &mut T) {
//     ui.horizontal(|ui| {
//         // ui.add(egui::Label::new(name));
//         ui.add(egui::Checkbox::new(t, name));
//     });
// }
// pub fn inspect<T: Numeric>(ui: &mut Ui,name: &str, t: &mut T) {
//     ui.horizontal(|ui| {
//         ui.add(egui::Label::new(name));
//         ui.add(egui::DragValue::new(t));
//     });
// }

// pub trait Inspect {
//     // type Data;
//     fn inspect(&mut self, name: &str, ui: &mut egui::Ui);
// }

// impl<'a> Inspect for Wrapper<'a, bool> {

//     fn inspect(&mut self, name: &str, ui: &mut egui::Ui) {
//         ui.horizontal(|ui| {
//             ui.add(egui::Label::new(name));
//             ui.add(egui::Checkbox::new(&mut *self));
//         });
//     }
// }
// impl<'a, T> Inspect for Wrapper<'a, T> where T: Numeric {
//     // type Data = T;
//     fn inspect(&mut self, name: &str, ui: &mut egui::Ui) {
//         ui.horizontal(|ui| {
//             ui.add(egui::Label::new(name));
//             ui.add(egui::DragValue::new(&mut *self));
//         });
//     }
// }
