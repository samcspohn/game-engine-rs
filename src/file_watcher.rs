use notify::{
    event::{AccessKind, AccessMode, ModifyKind, RenameMode},
    *,
};
use parking_lot::Mutex;
use std::{
    collections::{BTreeMap},
    path::Path,
    sync::{
        mpsc::{Receiver},
        Arc,
    },
};
use substring::Substring;
use walkdir::WalkDir;
// use relative_path;

use crate::{
    asset_manager::{AssetManagerBase, AssetsManager},
};

pub struct FileWatcher {
    pub(crate) files: BTreeMap<String, u64>,
    path: String,
    rx: Receiver<Result<Event>>,
    watcher: Box<dyn Watcher>,
}
// fn get_files(file_map: &mut HashMap<String, u64>) -> Result<()> {
//     for entry in fs::read_dir(path)? {
//         let entry = entry?;
//         let path = entry.path();

//         let metadata = fs::metadata(&path)?;
//         let last_modified = metadata.modified()?.elapsed()?.as_secs();

//         if last_modified < 24 * 3600 && metadata.is_file() {
//             println!(
//             "Last modified: {:?} seconds, is read only: {:?}, size: {:?} bytes, filename: {:?}",
//             last_modified,
//             metadata.permissions().readonly(),
//             metadata.len(),
//             path.file_name().ok_or("No filename")?
//         );
//         }
//     }
//     Ok(())
// }
impl FileWatcher {
    pub fn new(path: &str) -> FileWatcher {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = Box::new(RecommendedWatcher::new(tx, Config::default()).unwrap());
        watcher
            .watch(Path::new(path), RecursiveMode::Recursive)
            .unwrap();
        let files = BTreeMap::new();
        FileWatcher {
            files,
            path: path.into(),
            rx,
            watcher,
        }
    }
    pub fn init(&mut self, assets_manager: Arc<Mutex<AssetsManager>>) {
        for entry in WalkDir::new(&self.path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| !e.file_type().is_dir())
        {
            let f_name = String::from(entry.path().to_string_lossy());
            // let ext = entry.path().extension();
            assets_manager.lock().load(f_name.as_str());
            // let sys = world.sys.lock();
            // let mut mm = sys.model_manager.lock();
            // if let Some(dot) = f_name.rfind(".") {
            //     if f_name.substring(dot, f_name.len()) == ".obj" {
            //         mm.from_file(f_name.as_str());
            //     }
            // }
            self.files.entry(f_name).and_modify(|_e| {}).or_insert(
                entry
                    .metadata()
                    .unwrap()
                    .modified()
                    .unwrap()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            );
        }
    }
    pub fn get_updates(&self, assets_manager: Arc<Mutex<AssetsManager>>) {
        while let Ok(e) = self.rx.try_recv() {
            println!("{:?}", e);
            if let Ok(e) = e {
                match e.kind {
                    EventKind::Create(_) => {
                        // let ext = e.paths[0].extension();
                        let p = e.paths[0].to_string_lossy();
                        assets_manager.lock().load(p.to_string().as_str());
                        // let p = p.substring(p.find("/./").unwrap(), p.len());
                        // let sys = world.sys.lock();
                        // let mut mm = sys.model_manager.lock();
                        // if let Some(dot) = p.rfind(".") {
                        //     if p.substring(dot, p.len()) == ".obj" {
                        //         mm.from_file(p);
                        //     }
                        // }
                    }
                    EventKind::Remove(_) => {
                        let p = e.paths[0].to_string_lossy();
                        assets_manager.lock().remove(p.to_string().as_str());
                        // let p = p.substring(p.find("/./").unwrap(), p.len());
                        // let sys = world.sys.lock();
                        // let mut mm = sys.model_manager.lock();
                        // if let Some(dot) = p.rfind(".") {
                        //     if p.substring(dot, p.len()) == ".obj" {
                        //         mm.remove(p);
                        //     }
                        // }
                    }
                    EventKind::Access(a) => {
                        if a == AccessKind::Close(AccessMode::Write) {
                            let p = e.paths[0].to_string_lossy();
                            assets_manager.lock().reload(p.to_string().as_str());
                            // let p = p.substring(p.find("/./").unwrap(), p.len());
                            // let sys = world.sys.lock();
                            // let mut mm = sys.model_manager.lock();
                            // if let Some(dot) = p.rfind(".") {
                            //     if p.substring(dot, p.len()) == ".obj" {
                            //         mm.reload(p)
                            //     }
                            // }
                        }
                    }
                    EventKind::Modify(ModifyKind::Name(RenameMode::Both)) => {
                        if e.paths.len() == 2 {
                            let p: String = e.paths[0].as_path().to_str().unwrap().to_owned();
                            let p = p.substring(p.find("/./").unwrap() + 1, p.len());
                            let p2: String = e.paths[1].as_path().to_str().unwrap().to_owned();
                            let p2 = p2.substring(p2.find("/./").unwrap() + 1, p2.len());
                            assets_manager
                                .lock()
                                .move_file(p, p2);
                        }
                    }
                    EventKind::Any | EventKind::Modify(_) | EventKind::Other => {}
                }
            }
        }
    }
}
