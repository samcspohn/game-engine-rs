use crossbeam::queue::SegQueue;
use notify_debouncer_full::{
    new_debouncer,
    notify::{
        event::{AccessKind, AccessMode, ModifyKind, RenameMode},
        *,
    },
    DebounceEventResult, Debouncer, FileIdMap,
};
use parking_lot::Mutex;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{
        mpsc::{Receiver, Sender},
        Arc,
    },
    time::Duration,
};
use substring::Substring;
use walkdir::WalkDir;
// use relative_path;

use crate::asset_manager::{AssetManagerBase, AssetsManager};

pub struct FileWatcher {
    pub(crate) files: BTreeMap<String, u64>,
    // dirs: BTreeSet<String>,
    path: String,
    events_queue: Arc<SegQueue<DebounceEventResult>>,
    watchers: BTreeMap<String, Debouncer<ReadDirectoryChangesWatcher, FileIdMap>>,
}
const sep: char = std::path::MAIN_SEPARATOR;

impl FileWatcher {
    pub fn new(path: &str) -> FileWatcher {
        // let (tx, rx) = std::sync::mpsc::channel();
        // let mut watcher = Box::new(RecommendedWatcher::new(tx, Config::default()).unwrap());
        // watcher
        //     .watch(Path::new(path), RecursiveMode::NonRecursive)
        //     .unwrap();
        // let mut a: BTreeMap<String, (Receiver<Result<Event>>, Box<dyn Watcher>)> = BTreeMap::new();
        // a.insert(String::from(path), (rx, watcher));
        let events_queue = Arc::new(SegQueue::new());
        let e_q = events_queue.clone();
        let mut debouncer = new_debouncer(
            Duration::from_secs(1),
            None,
            move |res: DebounceEventResult| {
                e_q.push(res);
            },
        )
        .unwrap();
        debouncer
            .watcher()
            .watch(Path::new(path), RecursiveMode::Recursive)
            .unwrap();
        debouncer
            .cache()
            .add_root(Path::new(path), RecursiveMode::Recursive);

        let mut watchers = BTreeMap::new();
        watchers.insert(path.into(), debouncer);

        let files = BTreeMap::new();
        FileWatcher {
            files,
            path: path.into(),
            events_queue,
            watchers,
            // watchers: a,
        }
    }
    fn filter(&self, path: &Path) -> bool {
        let f_name = String::from(path.to_string_lossy());
        let ext = path.extension().unwrap_or_default();

        // let a = path.to_str().unwrap();
        // let a = path.strip_prefix(&self.path).unwrap();
        let p_str = path.to_str().unwrap();
        let a = p_str.substring(p_str.find(&self.path).unwrap() + self.path.len(), p_str.len()).replace(sep, "/");
        let parent = a.substring(0, a.rfind("/").unwrap_or_else(|| {
            a.len()
        }));
        if parent == "/target/release" {
            // println!("{:?}", a);
            return true;
        }

        // // only check files in root build directory
        // if let Ok(a) = path.strip_prefix(&self.path) {
        //     if let Some(a) = a.parent() {
        //         let a = a.to_str().unwrap();
        //         println!("{} == {}", a, format!("target{0}release", sep).as_str());
        //         if a == format!("target{0}release", sep).as_str() {
        //             return true;
        //         }
        //     }
        // }
        !f_name.contains(format!("target{0}", sep).as_str()) && !f_name.contains("runtime")
    }
    pub fn init(&mut self, assets_manager: Arc<AssetsManager>) {
        let mut do_later = Vec::new();

        for entry in WalkDir::new(&self.path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| !e.file_type().is_dir())
        {
            //     if entry.file_type().is_dir() {
            //         let p: String = entry.path().to_str().unwrap().into();
            //         if !self.watchers.contains_key(&p)
            //             && !p.contains(format!("{}target", sep).as_str())
            //             && !p.contains(format!("{}runtime", sep).as_str())
            //         {
            //             // let (tx, rx) = std::sync::mpsc::channel();
            //             // let mut watcher =
            //             //     Box::new(RecommendedWatcher::new(tx, Config::default()).unwrap());
            //             // // let config = notify::Config::default();
            //             // watcher
            //             //     .watch(Path::new(p.as_str()), RecursiveMode::Recursive)
            //             //     .unwrap();

            //             let e_q = self.events_queue.clone();
            //             let mut debouncer = new_debouncer(
            //                 Duration::from_secs(1),
            //                 None,
            //                 move |res: DebounceEventResult| {
            //                     e_q.push(res);
            //                 },
            //             )
            //             .unwrap();

            //             self.watchers.insert(p, debouncer);
            //         } else if !self.watchers.contains_key(&p)
            //             && (p == format!("{0}{1}target{1}release", self.path, sep)
            //                 || p == format!("{0}{1}target{1}debug", self.path, sep))
            //         {
            //             println!("watch target/release");
            //             let (tx, rx) = std::sync::mpsc::channel();
            //             let mut watcher =
            //                 Box::new(RecommendedWatcher::new(tx, Config::default()).unwrap());
            //             // let config = notify::Config::default();
            //             watcher
            //                 .watch(Path::new(p.as_str()), RecursiveMode::NonRecursive)
            //                 .unwrap();
            //             self.watchers.insert(p, (rx, watcher));
            //         }
            //     } else {
            let f_name = String::from(entry.path().to_string_lossy());
            let ext = entry.path().extension().unwrap_or_default();
            // #[cfg(debug)]
            if self.filter(entry.path()) {
                println!("{:?}", ext);
                if ext == "so" || ext == "dll" || ext == "rs" {
                    println!("load later: {:?}", ext);
                    do_later.push(entry);
                } else {
                    let f_name = f_name.replace(sep, "/");
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
            // }
        }
        for entry in do_later {
            let f_name = String::from(entry.path().to_string_lossy());
            let f_name = f_name.replace(sep, "/");
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
    fn remove_base(&self, p: &Path) -> String {
        // p.strip_prefix(&self.path).unwrap();
        let p: String = p.to_str().unwrap().to_owned();
        let p = p.replace(sep, "/");
        let p = p.substring(p.find("/./").unwrap() + 1, p.len());
        p.into()
    }
    pub fn get_updates(&mut self, assets_manager: Arc<AssetsManager>) {
        while let Some(e) = self.events_queue.pop() {
            if let Ok(e) = e {
                for e in e.iter() {
                    // println!("{:?}", e);
                    if !self.filter(&e.paths[0]) {
                        // println!("pass: {:?}", e);
                        continue;
                    }
                    match e.kind {
                        EventKind::Create(_) => {
                            let p = self.remove_base(e.paths[0].as_path());
                            assets_manager.load(p.as_str());
                        }
                        EventKind::Remove(_) => {
                            let p = self.remove_base(e.paths[0].as_path());
                            self.files.remove(&p);
                            assets_manager.remove(p.as_str());
                        }
                        EventKind::Access(a) => {
                            if a == AccessKind::Close(AccessMode::Write) {
                                let p = self.remove_base(e.paths[0].as_path());
                                assets_manager.reload(p.as_str());
                            }
                        }
                        EventKind::Modify(ModifyKind::Any) => {
                            let p = self.remove_base(e.paths[0].as_path());
                            assets_manager.reload(p.as_str());
                        }
                        EventKind::Modify(ModifyKind::Name(RenameMode::Both)) => {
                            if e.paths.len() == 2 {
                                let p1 = self.remove_base(e.paths[0].as_path());
                                let p2 = self.remove_base(e.paths[1].as_path());
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
