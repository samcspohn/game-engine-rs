use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap},
    sync::Arc,
};

use crate::{
    asset_manager::{AssetManagerBase, AssetsManager},
    engine::World,
    file_watcher::FileWatcher,
    serialize,
};

#[derive(Serialize, Deserialize)]
pub struct Project {
    pub files: BTreeMap<String, u64>,
    pub assets: BTreeMap<String, serde_yaml::Value>,
    // model_manager: serde_yaml::Value,
    // texture_manager: serde_yaml::Value,
    working_file: String,
}

pub fn save_project(
    file_watcher: &FileWatcher,
    world: &World,
    assets_manager: Arc<AssetsManager>,
) {
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
    // serialize::serialize(world);
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
            assets_manager.deserialize(project.assets);
            // let sys = world.sys.lock();
            // let mut mm = sys.model_manager.lock();
            // mm.regen(project.model_manager);
            // let mut tm = mm.const_params.1.lock();
            // tm.regen(project.texture_manager);
            // let a = serde_yaml::from_value(project.texture_manager).unwrap();
            // println!("{:?}", project.texture_manager);
            // panic!();
            // mm.const_params.1.assets = a;
            // mm.regen(project.models);
            // mm.id_gen = project.model_id_gen;
        }
        serialize::deserialize(&mut world.lock());
    }
}
