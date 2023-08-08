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
    speed: f32,
    emission_rate: f32,
    life_time: f32,
    color_over_life: ColorGradient,
    trail: bool,
}

impl Default for ParticleTemplate {
    fn default() -> Self {
        Self {
            color: [1.; 4],
            speed: 1.,
            emission_rate: 10.,
            life_time: 1.,
            color_over_life: ColorGradient::new(),
            trail: false,
        }
    }
}

impl ParticleTemplate {
    pub fn gen_particle_template(&self) -> particle_template {
        particle_template {
            color: self.color,
            speed: self.speed,
            emission_rate: self.emission_rate,
            life_time: self.life_time,
            color_life: self.color_over_life.to_color_array(),
            trail: if self.trail { 1 } else { 0 },
            _dummy0: Default::default(),
            size: 1f32,
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
        field(ui, "speed", |ui| {
            ui.add(egui::DragValue::new(&mut self.speed));
        });
        field(ui, "life time", |ui| {
            ui.add(egui::DragValue::new(&mut self.life_time));
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
