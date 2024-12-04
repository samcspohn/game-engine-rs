use std::{
    cell::SyncUnsafeCell,
    marker::PhantomData,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use id::*;
use lazy_static::lazy_static;
use nalgebra_glm::{vec2, Vec2};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::{
    editor::inspectable::{Inpsect, Ins},
    engine::{
        color_gradient::ColorGradient, prelude::Inspectable_, project::asset_manager::{Asset, AssetInstance, AssetManager}, rendering::texture::{Texture, TextureManager}, storage::_Storage, utils::gradient::Gradient, world::World
    },
};

use super::{
    particle_textures::ParticleTextures,
    shaders::cs::{self, particle_template},
};

#[derive(ID, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ParticleTemplate {
    color: [f32; 4],
    pub emission_rate: f32,
    emission_radius: f32,
    dispersion: f32,
    min_speed: f32,
    max_speed: f32,
    min_lifetime: f32,
    pub max_lifetime: f32,
    size: Vec2,
    pub(super) color_over_life: ColorGradient,
    pub trail: bool,
    billboard: bool,
    align_vel: bool,
    texture: AssetInstance<Texture>,
    recieve_lighting: bool,
    size_over_life: Gradient,
}

lazy_static! {
    pub(super) static ref DEFAULT_TEXTURE: SyncUnsafeCell<AssetInstance::<Texture>> =
        SyncUnsafeCell::new(AssetInstance::<Texture>::new(0));
    pub(super) static ref TEMPLATE_UPDATE: AtomicBool = AtomicBool::new(true);
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
            size: vec2(1., 1.),
            color_over_life: ColorGradient::new(),
            size_over_life: Gradient::new(),
            trail: false,
            billboard: true,
            align_vel: false,
            dispersion: 1.,
            texture: unsafe { *DEFAULT_TEXTURE.get() },
            recieve_lighting: false,
        }
    }
}

impl ParticleTemplate {
    pub fn gen_particle_template(&self, textures: &mut ParticleTextures) -> particle_template {
        particle_template {
            color: self.color,
            emission_rate: self.emission_rate,
            emission_radius: self.emission_radius,
            dispersion: self.dispersion,
            min_speed: self.min_speed,
            max_speed: self.max_speed,
            min_lifetime: self.min_lifetime,
            max_lifetime: self.max_lifetime,
            // color_life: self.color_over_life.to_color_array(),
            trail: if self.trail { 1 } else { 0 },
            align_vel: if self.align_vel { 1 } else { 0 },
            billboard: if self.billboard { 1 } else { 0 },
            scale: self.size.into(),
            tex_id: textures.get_tex_id(&self.texture),
            recieve_lighting: if self.recieve_lighting { 1 } else { 0 },
            padding2: 0,
            padding3: 0,
            // _dummy0: Default::default(),
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
lazy_static! {
    static ref GRADIENT_TEST: Mutex<Gradient> = Mutex::new(Gradient::new());
}
// static mut GRADIENT_TEST: Gradient = Gradient::new();
impl Inspectable_ for ParticleTemplate {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn inspect(&mut self, ui: &mut egui::Ui, _world: &mut World) -> bool {
        field(ui, "color", |ui| {
            if ui
                .color_edit_button_rgba_premultiplied(&mut self.color)
                .changed()
            {
                TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
            }
        });
        field(ui, "emission rate", |ui| {
            if ui
                .add(egui::DragValue::new(&mut self.emission_rate))
                .changed()
            {
                TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
            }
        });
        field(ui, "emission radius", |ui| {
            if ui
                .add(egui::DragValue::new(&mut self.emission_radius))
                .changed()
            {
                TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
            }
        });
        field(ui, "dispersion", |ui| {
            if ui.add(egui::DragValue::new(&mut self.dispersion)).changed() {
                TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
            }
        });
        field(ui, "min lifetime", |ui| {
            if ui
                .add(egui::DragValue::new(&mut self.min_lifetime).speed(0.1))
                .changed()
            {
                TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
            }
            if self.min_lifetime > self.max_lifetime {
                self.max_lifetime = self.min_lifetime;
            }
        });
        field(ui, "max lifetime", |ui| {
            if ui
                .add(egui::DragValue::new(&mut self.max_lifetime).speed(0.1))
                .changed()
            {
                TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
            }
            if self.max_lifetime < self.min_lifetime {
                self.min_lifetime = self.max_lifetime;
            }
        });
        field(ui, "min speed", |ui| {
            if ui.add(egui::DragValue::new(&mut self.min_speed)).changed() {
                TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
            }
            if self.min_speed > self.max_speed {
                self.max_speed = self.min_speed;
            }
        });
        field(ui, "max speed", |ui| {
            if ui.add(egui::DragValue::new(&mut self.max_speed)).changed() {
                TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
            }
            if self.max_speed < self.min_speed {
                self.min_speed = self.max_speed;
            }
        });
        if Ins(&mut self.size).inspect("scale", ui, &_world.sys) {
            TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
        }
        // field(ui, "size", |ui| {
        //     if ui.add(egui::DragValue::new(&mut self.size)).changed() {
        //         TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
        //     }
        // });
        field(ui, "color over life", |ui| {
            if self.color_over_life.edit(ui).changed() {
                TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
            }
        });
        field(ui, "size over life", |ui| {
            GRADIENT_TEST.lock().edit(ui);
        });
        if Ins(&mut self.texture).inspect("texture", ui, &_world.sys) {
            TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
        }
        if ui.checkbox(&mut self.trail, "trail").changed() {
            TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
        }
        if ui.checkbox(&mut self.billboard, "billboard").changed() {
            TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
        }
        if ui
            .checkbox(&mut self.align_vel, "align to velocity")
            .changed()
        {
            TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
        }
        if ui
            .checkbox(&mut self.recieve_lighting, "recieve lighting")
            .changed()
        {
            TEMPLATE_UPDATE.store(true, Ordering::Relaxed);
        }
        true
    }
}

type Param = (
    Arc<Mutex<_Storage<cs::particle_template>>>,
    Arc<Mutex<ParticleTextures>>,
);
impl Asset<ParticleTemplate, Param> for ParticleTemplate {
    fn from_file(file: &str, params: &Param) -> ParticleTemplate {
        let mut t = ParticleTemplate::default();
        if let Ok(s) = std::fs::read_to_string(file) {
            t = serde_yaml::from_str(s.as_str()).unwrap();
        }
        let p_t = t.gen_particle_template(&mut params.1.lock());
        params.0.lock().emplace(p_t);
        t
    }

    fn reload(&mut self, file: &str, _params: &Param) {
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
    fn new(_file: &str, params: &Param) -> Option<ParticleTemplate> {
        let t = ParticleTemplate::default();
        let p_t = t.gen_particle_template(&mut params.1.lock());
        params.0.lock().emplace(p_t);
        Some(t)
    }
}

pub type ParticleTemplateManager = AssetManager<Param, ParticleTemplate>;
