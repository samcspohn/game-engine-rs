use notify::{
    event::{AccessKind, AccessMode, ModifyKind, RenameMode},
    *,
};
use parking_lot::Mutex;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{
        mpsc::{Receiver, Sender},
        Arc,
    },
};
use substring::Substring;
use walkdir::WalkDir;
// use relative_path;

use crate::asset_manager::{AssetManagerBase, AssetsManager};

pub struct FileWatcher {
    pub(crate) files: BTreeMap<String, u64>,
    // dirs: BTreeSet<String>,
    path: String,
    watchers: BTreeMap<String, (Receiver<Result<Event>>, Box<dyn Watcher>)>,
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
        // let config = notify::Config::default();
        watcher
            .watch(Path::new(path), RecursiveMode::NonRecursive)
            .unwrap();
        let mut a: BTreeMap<String, (Receiver<Result<Event>>, Box<dyn Watcher>)> = BTreeMap::new();
        a.insert(String::from(path), (rx, watcher));
        let files = BTreeMap::new();
        FileWatcher {
            files,
            path: path.into(),
            // rx,
            // tx,
            watchers: a,
            // dirs: BTreeSet::new(),
        }
    }
    pub fn init(&mut self, assets_manager: Arc<AssetsManager>) {
        let mut do_later = Vec::new();
        for entry in WalkDir::new(&self.path).into_iter().filter_map(|e| e.ok())
        // .filter(|e| !e.file_type().is_dir())
        {
            if entry.file_type().is_dir() {
                let p: String = entry.path().to_str().unwrap().into();
                if !self.watchers.contains_key(&p) && !p.contains("/target") {
                    let (tx, rx) = std::sync::mpsc::channel();
                    let mut watcher =
                        Box::new(RecommendedWatcher::new(tx, Config::default()).unwrap());
                    // let config = notify::Config::default();
                    watcher
                        .watch(Path::new(p.as_str()), RecursiveMode::Recursive)
                        .unwrap();
                    self.watchers.insert(p, (rx, watcher));
                } 
                // else if !self.watchers.contains_key(&p)
                //     && (p == format!("{}/target/release", self.path)
                //         || p == format!("{}/target/debug", self.path))
                // {
                //     let (tx, rx) = std::sync::mpsc::channel();
                //     let mut watcher =
                //         Box::new(RecommendedWatcher::new(tx, Config::default()).unwrap());
                //     // let config = notify::Config::default();
                //     watcher
                //         .watch(Path::new(p.as_str()), RecursiveMode::NonRecursive)
                //         .unwrap();
                //     self.watchers.insert(p, (rx, watcher));
                // }
            } else {
                let f_name = String::from(entry.path().to_string_lossy());
                let ext = entry.path().extension().unwrap_or_default();
                // #[cfg(debug)]
                let sep = std::path::MAIN_SEPARATOR;
                if !f_name.contains(format!("{0}target{0}", sep).as_str())
                    // || if let Some(a) = entry.path().parent() {
                    //     a.to_str().unwrap()
                    // } else {
                    //     ""
                    // } == format!("{}/target/release", self.path).as_str()
                {
                    println!("{:?}", ext);
                    if ext == "rs" {
                        do_later.push(entry);
                    } else {
                        assets_manager.load(f_name.as_str());
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
            }
        }
        for entry in do_later {
            let f_name = String::from(entry.path().to_string_lossy());
            let ext = entry.path().extension().unwrap_or_default();
            assets_manager.load(f_name.as_str());
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
            // (f.0)(f.1);
        }
    }
    fn remove_base(p: &Path) -> String {
        let p: String = p.to_str().unwrap().to_owned();
        let p = p.substring(p.find("/./").unwrap() + 1, p.len());
        p.into()
    }
    pub fn get_updates(&self, assets_manager: Arc<AssetsManager>) {
        for (_, (rx, _)) in &self.watchers {
            while let Ok(e) = rx.try_recv() {
                if let Ok(e) = e {
                    println!("{:?}", e);
                    match e.kind {
                        EventKind::Create(_) => {
                            let p = Self::remove_base(e.paths[0].as_path());
                            assets_manager.load(p.as_str());
                        }
                        EventKind::Remove(_) => {
                            let p = Self::remove_base(e.paths[0].as_path());
                            assets_manager.remove(p.as_str());
                        }
                        EventKind::Access(a) => {
                            if a == AccessKind::Close(AccessMode::Write) {
                                let p = Self::remove_base(e.paths[0].as_path());
                                assets_manager.reload(p.as_str());
                            }
                        }
                        EventKind::Modify(ModifyKind::Name(RenameMode::Both)) => {
                            if e.paths.len() == 2 {
                                let p1 = Self::remove_base(e.paths[0].as_path());
                                let p2 = Self::remove_base(e.paths[1].as_path());
                                assets_manager.move_file(p1.as_str(), p2.as_str());
                            }
                        }
                        EventKind::Any | EventKind::Modify(_) | EventKind::Other => {}
                    }
                }
            }
        }
    }
}
