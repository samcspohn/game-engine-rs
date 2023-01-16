use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::{engine::World, file_watcher::FileWatcher, serialize};

#[derive(Serialize, Deserialize)]
struct Project {
    files: HashMap<String, u64>,
    models: HashMap<String, i32>,
    model_id_gen: i32,
    textures: HashSet<String>,
    working_file: String,
}

pub fn save_project(file_watcher: &FileWatcher, world: &World) {
    let sys = world.sys.lock();
    let files = file_watcher.files.clone();
    let models = sys.model_manager.lock().models.clone();
    let models_id_gen = sys.model_manager.lock().model_id_gen;
    let textures = sys
        .model_manager
        .lock()
        .texture_manager
        .textures
        .read()
        .iter()
        .map(|(f, _)| f.clone())
        .collect();
    let working_file = "test.ron".into();

    let project = Project {
        files,
        models,
        model_id_gen: models_id_gen,
        textures,
        working_file,
    };
    std::fs::write("project.ron", ron::to_string(&project).unwrap()).unwrap();
    // serialize::serialize(world);
}

pub fn load_project(file_watcher: &mut FileWatcher, world: &mut World) {
    if let Ok(s) = std::fs::read_to_string("project.ron") {
        {
            let project: Project = ron::from_str(s.as_str()).unwrap();
            file_watcher.files = project.files;
            let sys = world.sys.lock();
            let mut mm = sys.model_manager.lock();
            mm.texture_manager.regen(project.textures);
            mm.regen(project.models);
            mm.model_id_gen = project.model_id_gen;
        }
        serialize::deserialize(world);
    }
}
