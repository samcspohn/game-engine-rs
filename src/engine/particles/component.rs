use component_derive::ComponentID;
use serde::{Deserialize, Serialize};

use crate::engine::{prelude, project::asset_manager::AssetInstance};
use prelude::*;

use super::{particle_asset::ParticleTemplate, shaders::cs};

#[derive(ComponentID, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ParticleEmitter {
    template: AssetInstance<ParticleTemplate>,
}
impl Default for ParticleEmitter {
    fn default() -> Self {
        Self {
            template: AssetInstance::new(0),
        }
    }
}

impl Inspectable for ParticleEmitter {
    fn inspect(&mut self, _transform: &Transform, _id: i32, ui: &mut egui::Ui, sys: &Sys) {
        Ins(&mut self.template).inspect("template", ui, sys);
    }
}
impl ParticleEmitter {
    pub fn new(template: i32) -> ParticleEmitter {
        let inst = AssetInstance::<ParticleTemplate>::new(template);
        ParticleEmitter { template: inst }
    }
}

impl Component for ParticleEmitter {
    fn init(&mut self, transform: &Transform, id: i32, sys: &Sys) {
        let d = cs::ty::emitter_init {
            transform_id: transform.id,
            alive: 1,
            template_id: self.template.id,
            e_id: id,
        };
        match sys.particles_system.emitter_inits.try_push(d) {
            None => {}
            Some(i) => {
                sys.particles_system.emitter_inits.push(i, d);
            }
        }
    }
    fn deinit(&mut self, transform: &Transform, id: i32, sys: &Sys) {
        let d = cs::ty::emitter_init {
            transform_id: transform.id,
            alive: 0,
            template_id: self.template.id,
            e_id: id,
        };

        match sys.particles_system.emitter_deinits.try_push(d) {
            None => {}
            Some(i) => {
                sys.particles_system.emitter_deinits.push(i, d);
            }
        }
    }
}
