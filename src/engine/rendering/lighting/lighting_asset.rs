use std::sync::Arc;

use glm::{vec3, Vec3};
use id::*;
use nalgebra_glm as glm;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{allocator::SubbufferAllocator, BufferUsage, Subbuffer},
    memory::allocator::MemoryTypeFilter,
    DeviceSize,
};

use crate::engine::{
    prelude::*,
    project::asset_manager::{Asset, AssetManager},
    rendering::{pipeline::fs, vulkan_manager::VulkanManager},
    storage::_Storage,
};

#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Attenuation {
    constant: f32,
    linear: f32,
    exponential: f32,
    brightness: f32,
}
#[derive(ID, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LightTemplate {
    color: [f32; 3],
    atten: Attenuation,
    id: i32,
}

impl Default for LightTemplate {
    fn default() -> Self {
        Self {
            color: [1.0, 1.0, 1.0],
            atten: Attenuation {
                constant: 0.5,
                linear: 0.5,
                exponential: 0.5,
                brightness: 1.0,
            },
            id: 0,
        }
    }
}
impl LightTemplate {
    fn gen_light(&self) -> fs::lightTemplate {
        fs::lightTemplate {
            color: self.color,
            p1: 0,
            atten: fs::attenuation {
                constant: self.atten.constant,
                linear: self.atten.linear,
                exponential: self.atten.exponential,
                brightness: self.atten.brightness,
            },
        }
    }
}
impl Inspectable_ for LightTemplate {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn inspect(&mut self, ui: &mut egui::Ui, world: &mut crate::engine::world::World) -> bool{
        ui.horizontal(|ui| {
            ui.label("color");
            ui.color_edit_button_rgb(&mut self.color);
        });

        // Ins(&mut self.color).inspect("color", ui, &world.sys);
        Ins(&mut self.atten.brightness).inspect("brightness", ui, &world.sys);
        ui.add(egui::Label::new("attenuation"));
        ui.separator();
        Ins(&mut self.atten.constant).inspect("constant", ui, &world.sys);
        Ins(&mut self.atten.linear).inspect("linear", ui, &world.sys);
        Ins(&mut self.atten.exponential).inspect("exponential", ui, &world.sys);
        *(world
            .sys
            .lighting_system
            .light_templates
            .lock()
            .get_mut(&self.id)) = self.gen_light();
        true
    }
}
pub type Param = (Arc<Mutex<_Storage<fs::lightTemplate>>>);
impl Asset<LightTemplate, Param> for LightTemplate {
    fn from_file(file: &str, params: &Param) -> LightTemplate {
        let mut l = LightTemplate::default();
        if let Ok(s) = std::fs::read_to_string(file) {
            l = serde_yaml::from_str(s.as_str()).unwrap();
        }
        let l_t = l.gen_light();
        l.id = params.lock().emplace(l_t);
        l
    }

    fn reload(&mut self, file: &str, params: &Param) {
        if let Ok(s) = std::fs::read_to_string(file) {
            *self = serde_yaml::from_str(s.as_str()).unwrap();
        }
    }
    fn save(&mut self, file: &str, _params: &Param) {
        if let Ok(s) = serde_yaml::to_string(self) {
            match std::fs::write(file, s.as_bytes()) {
                Ok(_) => (),
                Err(a) => {
                    println!("{}: failed for file: {}", a, file)
                }
            }
        }
    }
    fn new(_file: &str, params: &Param) -> Option<LightTemplate> {
        let mut l = LightTemplate::default();
        let l_t = l.gen_light();
        l.id = params.lock().emplace(l_t);
        Some(l)
    }
}

pub type LightTemplateManager = AssetManager<Param, LightTemplate>;
