use std::sync::atomic::AtomicBool;
use std::{fs, process::Command, sync::Arc, time::Duration};

use component_derive::AssetID;
use lazy_static::lazy_static;
use libloading::Library;
use parking_lot::Mutex;

use crate::engine::project::asset_manager::_AssetID;
use crate::{
    editor::inspectable::Inspectable_,
    engine::{
        project::{
            asset_manager::{Asset, AssetManager},
            serialize,
        },
        world::World,
    },
};

#[derive(AssetID)]
pub struct RSFile {
    path: String,
}

lazy_static! {
    static ref LIB: Mutex<Option<Library>> = Mutex::new(None);
    static ref ID: Mutex<usize> = Mutex::new(0);
}
impl RSFile {}
impl Asset<RSFile, (Arc<AtomicBool>)> for RSFile {
    fn from_file(file: &str, params: &(Arc<AtomicBool>)) -> RSFile {
        params.store(true, std::sync::atomic::Ordering::Relaxed);
        // let mut args = vec!["build"];
        #[cfg(not(debug_assertions))]
        {
            println!("compiling {} for release", file);
            //     args.push("-r");
        }
        #[cfg(debug_assertions)]
        {
            println!("compiling {} for debug", file);
        }
        // Command::new("cargo")
        //     .args(args.as_slice())
        //     .status()
        //     .unwrap();

        RSFile { path: file.into() }
    }

    fn reload(&mut self, file: &str, params: &(Arc<AtomicBool>)) {
        *self = Self::from_file(file, params);
    }
}
impl Inspectable_ for RSFile {
    fn inspect(&mut self, ui: &mut egui::Ui, world: &mut World) {
        ui.add(egui::Label::new(self.path.as_str()));
    }
}
pub type RSManager = AssetManager<(Arc<AtomicBool>), RSFile>;

#[derive(AssetID)]
pub struct Lib {}

impl Asset<Lib, (Arc<Mutex<World>>)> for Lib {
    fn from_file(file: &str, params: &(Arc<Mutex<World>>)) -> Lib {
        if let Some(lib) = &*LIB.lock() {
            let func: libloading::Symbol<unsafe extern "C" fn(&mut World)> =
                unsafe { lib.get(b"unregister").unwrap() };

            let mut world = params.lock();
            world.clear();
            unsafe {
                func(&mut world);
            }
        }
        *LIB.lock() = None;

        #[cfg(target_os = "windows")]
        let ext = "dll";
        #[cfg(not(target_os = "windows"))]
        let ext = "so";
        // DELETE OLD LIB
        let id = *ID.lock();
        *ID.lock() += 1;
        let so_file = format!("runtime/lib{}.{}", id, ext);
        match fs::remove_file(&so_file) {
            Ok(_) => {}
            Err(a) => println!("{:?}", a),
        }

        // RELOCATE NEW LIB --- avoid dylib caching
        let id = id + 1;
        let so_file = format!("runtime/lib{}.{}", id, ext);
        if let Ok(_) = fs::copy(file, &so_file) {
            // LOAD NEW LIB
            let lib = unsafe { libloading::Library::new(&so_file).unwrap() };
            let func: libloading::Symbol<unsafe extern "C" fn(&mut World)> =
                unsafe { lib.get(b"register").unwrap() };

            let mut world = params.lock();
            world.clear();
            unsafe {
                func(&mut world);
            }
            serialize::deserialize(&mut world);
            *LIB.lock() = Some(lib);
        }

        Lib {}
    }

    fn reload(&mut self, file: &str, params: &(Arc<Mutex<World>>)) {
        *self = Self::from_file(file, params);
        println!("reload {}", file);
    }
}

impl Inspectable_ for Lib {
    fn inspect(&mut self, ui: &mut egui::Ui, world: &mut World) {
        // ui.add(egui::Label::new(self.path));
    }
}
pub type LibManager = AssetManager<(Arc<Mutex<World>>), Lib>;

// pub(crate) fn compile_thread() {
//     let mut args = vec!["build"];
//     #[cfg(not(debug_assertions))]
//     {
//         println!("compiling for release");
//         args.push("-r");
//     }
//     // args.push("-r");
//     Command::new("cargo")
//         .args(args.as_slice())
//         .status()
//         .unwrap();
// }
