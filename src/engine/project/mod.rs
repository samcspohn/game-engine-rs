use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, sync::Arc};

use crate::engine::world::World;

use self::{asset_manager::AssetsManager, file_watcher::FileWatcher};

pub mod asset_manager;
pub mod file_watcher;
pub mod serialize;

#[derive(Serialize, Deserialize, Default)]
pub struct Project {
    pub files: BTreeMap<String, u64>,
    pub assets: BTreeMap<String, serde_yaml::Value>,
    // model_manager: serde_yaml::Value,
    // texture_manager: serde_yaml::Value,
    working_file: String,
}

pub fn save_project(file_watcher: &FileWatcher, world: &World, assets_manager: Arc<AssetsManager>) {
    // let _sys = world.sys.lock();
    let files = file_watcher.files.clone();
    // let model_manager = serde_yaml::to_value(&*sys.model_manager.lock()).unwrap();

    // let texture_manager = serde_yaml::to_value(&*sys
    //     .model_manager
    //     .lock()
    //     .const_params.1.lock()).unwrap();
    let working_file = "test.yaml".into();
    let assets = {
        let a = &assets_manager;
        a.save_assets();
        a.serialize()
    };
    let project = Project {
        files,
        assets,
        working_file,
    };
    std::fs::write("project.yaml", serde_yaml::to_string(&project).unwrap()).unwrap();
}

pub fn load_project(
    file_watcher: &mut FileWatcher,
    world: Arc<Mutex<World>>,
    assets_manager: Arc<AssetsManager>,
) {
    if let Ok(s) = std::fs::read_to_string("project.yaml") {
        {
            let project: Project = serde_yaml::from_str(s.as_str()).unwrap();
            file_watcher.files = project.files;
            assets_manager.deserialize(&project.assets);
        }
        serialize::deserialize(&mut world.lock(), "test.yaml");
    }
}
