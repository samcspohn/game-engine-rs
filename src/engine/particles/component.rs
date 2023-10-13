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

impl ParticleEmitter {
    pub fn new(template: i32) -> ParticleEmitter {
        let inst = AssetInstance::<ParticleTemplate>::new(template);
        ParticleEmitter { template: inst }
    }
}

impl Component for ParticleEmitter {
    fn init(&mut self, transform: &Transform, id: i32, sys: &Sys) {
        let d = cs::emitter_init {
            transform_id: transform.id,
            alive: 1,
            template_id: self.template.id,
            e_id: id,
        };

        sys.particles_system.emitter_inits.push(d);
    }
    fn deinit(&mut self, transform: &Transform, id: i32, sys: &Sys) {
        let d = cs::emitter_init {
            transform_id: transform.id,
            alive: 0,
            template_id: self.template.id,
            e_id: id,
        };

        sys.particles_system.emitter_deinits.push(d);
    }
    fn inspect(&mut self, transform: &Transform, id: i32, ui: &mut egui::Ui, sys: &Sys) {
        let mut temp = self.template;
        if Ins(&mut temp).inspect("template", ui, sys) {
            self.deinit(transform, id, sys);
            self.template = temp;
            self.init(transform, id, sys);
        }
    }
}
