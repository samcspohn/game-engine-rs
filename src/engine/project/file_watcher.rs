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

use super::asset_manager::AssetsManager;
// use relative_path;

pub struct FileWatcher {
    pub(crate) files: BTreeMap<String, u64>,
    // dirs: BTreeSet<String>,
    path: String,
    events_queue: Arc<SegQueue<DebounceEventResult>>,
    watcher: Debouncer<RecommendedWatcher, FileIdMap>,
}
const sep: char = std::path::MAIN_SEPARATOR;

impl FileWatcher {
    pub fn new(path: &str) -> FileWatcher {
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

        let files = BTreeMap::new();
        FileWatcher {
            files,
            path: path.into(),
            events_queue,
            watcher: debouncer,
        }
    }
    fn filter(&self, path: &Path) -> bool {
        let f_name = String::from(path.to_string_lossy());
        // let ext = path.extension().unwrap_or_default();

        let p_str = path.to_str().unwrap();
        let a = p_str
            .substring(
                p_str.find(&self.path).unwrap() + self.path.len(),
                p_str.len(),
            )
            .replace(sep, "/");
        let parent = a.substring(0, a.rfind("/").unwrap_or_else(|| a.len()));
        if parent == "/target/release" {
            return true;
        }
        !f_name.contains(format!("target{0}", sep).as_str()) && !f_name.contains("runtime")
    }
    fn process(&mut self, assets_manager: &Arc<AssetsManager>, entry: walkdir::DirEntry) {
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
    }
    pub fn init(&mut self, assets_manager: Arc<AssetsManager>) {
        let mut do_later = Vec::new();
        for entry in WalkDir::new(&self.path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| !e.file_type().is_dir())
        {
            let ext = entry.path().extension().unwrap_or_default();
            if self.filter(entry.path()) {
                if ext == "so" || ext == "dll" || ext == "rs" {
                    do_later.push(entry);
                } else {
                    self.process(&assets_manager, entry);
                }
            }
        }
        for entry in do_later {
            self.process(&assets_manager, entry);
        }
    }
    fn remove_base(&self, p: &Path) -> String {
        let p: String = p.to_str().unwrap().to_owned();
        let p = p.replace(sep, "/");
        let p = p.substring(p.find("/./").unwrap() + 1, p.len());
        p.into()
    }
    pub fn get_updates(&mut self, assets_manager: Arc<AssetsManager>) {
        while let Some(e) = self.events_queue.pop() {
            if let Ok(e) = e {
                for e in e.iter() {
                    if !self.filter(&e.paths[0]) {
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
