use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Serialize, Deserialize};

use crate::engine::{color_gradient::ColorGradient, project::asset_manager::{Asset, AssetManager}, storage::_Storage, prelude::Inspectable_, world::World};

use super::shaders::cs::{ty::particle_template, self};
use crate::engine::project::asset_manager::_AssetID;
use component_derive::AssetID;

#[derive(AssetID, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ParticleTemplate {
    color: [f32; 4],
    emission_rate: f32,
    emission_radius: f32,
    dispersion: f32,
    min_speed: f32,
    max_speed: f32,
    min_lifetime: f32,
    max_lifetime: f32,
    size: f32,
    color_over_life: ColorGradient,
    trail: bool,
}

impl Default for ParticleTemplate {
    fn default() -> Self {
        Self {
            color: [1.; 4],
            emission_rate: 10.,
            emission_radius: 0.,
            min_speed: 1.,
            max_speed: 1.,
            min_lifetime: 1.,
            max_lifetime: 1.,
            size: 1.,
            color_over_life: ColorGradient::new(),
            trail: false,
            dispersion: 1.,
        }
    }
}

impl ParticleTemplate {
    pub fn gen_particle_template(&self) -> particle_template {
        particle_template {
            color: self.color,
            emission_rate: self.emission_rate,
            emission_radius: self.emission_radius,
            dispersion: self.dispersion,
            min_speed: self.min_speed,
            max_speed: self.max_speed,
            min_lifetime: self.min_lifetime,
            max_lifetime: self.max_lifetime,
            color_life: self.color_over_life.to_color_array(),
            trail: if self.trail { 1 } else { 0 },
            _dummy0: Default::default(),
            size: self.size,
        }
    }
}

fn field<F>(ui: &mut egui::Ui, name: &str, func: F)
where
    F: FnOnce(&mut egui::Ui),
{
    ui.horizontal(|ui| {
        ui.add(egui::Label::new(name));
        func(ui);
        // ui.add(egui::DragValue::new(self.0));s
    });
}
impl Inspectable_ for ParticleTemplate {
    fn inspect(&mut self, ui: &mut egui::Ui, _world: &Mutex<World>) {
        field(ui, "color", |ui| {
            ui.color_edit_button_rgba_premultiplied(&mut self.color);
        });
        field(ui, "emission rate", |ui| {
            ui.add(egui::DragValue::new(&mut self.emission_rate));
        });
        field(ui, "emission radius", |ui| {
            ui.add(egui::DragValue::new(&mut self.emission_radius));
        });
        field(ui, "dispersion", |ui| {
            ui.add(egui::DragValue::new(&mut self.dispersion));
        });
        field(ui, "life time", |ui| {
            ui.add(egui::DragValue::new(&mut self.min_lifetime));
        });
        field(ui, "life time", |ui| {
            ui.add(egui::DragValue::new(&mut self.max_lifetime));
        });
        field(ui, "min speed", |ui| {
            ui.add(egui::DragValue::new(&mut self.min_speed));
        });
        field(ui, "max speed", |ui| {
            ui.add(egui::DragValue::new(&mut self.max_speed));
        });
        field(ui, "size", |ui| {
            ui.add(egui::DragValue::new(&mut self.size));
        });
        field(ui, "color over life", |ui| {
            self.color_over_life.edit(ui);
        });
        ui.checkbox(&mut self.trail, "trail");
    }
}

impl Asset<ParticleTemplate, Arc<Mutex<_Storage<cs::ty::particle_template>>>> for ParticleTemplate {
    fn from_file(file: &str, params: &Arc<Mutex<_Storage<particle_template>>>) -> ParticleTemplate {
        let mut t = ParticleTemplate::default();
        if let Ok(s) = std::fs::read_to_string(file) {
            t = serde_yaml::from_str(s.as_str()).unwrap();
        }
        let p_t = t.gen_particle_template();
        params.lock().emplace(p_t);
        t
    }

    fn reload(&mut self, file: &str, _params: &Arc<Mutex<_Storage<particle_template>>>) {
        if let Ok(s) = std::fs::read_to_string(file) {
            *self = serde_yaml::from_str(s.as_str()).unwrap();
        }
    }
    fn save(&mut self, file: &str, _params: &Arc<Mutex<_Storage<particle_template>>>) {
        if let Ok(s) = serde_yaml::to_string(self) {
            match std::fs::write(file, s.as_bytes()) {
                Ok(_) => (),
                Err(a) => {
                    println!("{}: failed for file: {}", a, file)
                }
            }
        }
    }
    fn new(
        _file: &str,
        params: &Arc<Mutex<_Storage<cs::ty::particle_template>>>,
    ) -> Option<ParticleTemplate> {
        let t = ParticleTemplate::default();
        let p_t = t.gen_particle_template();
        params.lock().emplace(p_t);
        Some(t)
    }
}

pub type ParticleTemplateManager =
    AssetManager<Arc<Mutex<_Storage<cs::ty::particle_template>>>, ParticleTemplate>;
