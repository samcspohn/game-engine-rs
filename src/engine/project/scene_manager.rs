use std::sync::Arc;

use id::*;
use parking_lot::Mutex;

use crate::engine::{prelude::Inspectable_, world::World};

use super::{asset_manager::{Asset, AssetManager}, serialize};



#[derive(ID)]
pub struct SceneAsset {
    file: String,
}

impl Asset<SceneAsset, ()> for SceneAsset {
    fn from_file(file: &str, params: &()) -> SceneAsset {
        SceneAsset { file: file.into() }
        // do nothing
    }

    fn reload(&mut self, file: &str, params: &()) {
        *self = Self::from_file(file, params);
    }
}

impl Inspectable_ for SceneAsset {
    fn inspect(&mut self, ui: &mut egui::Ui, world: &mut World) -> bool {
        ui.label(self.file.as_str());
        ui.separator();
        if ui.button("Load").clicked() {
            serialize::deserialize(world, &self.file);
        }
        true
    }
}

pub type SceneManager = AssetManager<(), SceneAsset>;