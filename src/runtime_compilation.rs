use std::{process::Command, sync::Arc, time::Duration};

use lazy_static::lazy_static;
use libloading::Library;
use parking_lot::Mutex;

use crate::{
    asset_manager::{Asset, AssetManager},
    engine::World,
    inspectable::Inspectable_,
};

// struct RuntimeCompilation {

// }

pub struct RSFile {
    path: String,
}

lazy_static! {
    static ref LIB: Mutex<Option<Library>> = Mutex::new(None);
    static ref ID: Mutex<usize> = Mutex::new(0);
}
impl RSFile {}
impl Asset<RSFile, (Arc<Mutex<World>>)> for RSFile {
    fn from_file(file: &str, params: &(Arc<Mutex<World>>)) -> RSFile {
        // Command::new("cargo").args(&["clean", "--manifest-path=test_project_rs/Cargo.toml"]).status().unwrap();
        Command::new("cargo")
            .args(&["build", "--manifest-path=test_project_rs/Cargo.toml", "-r"])
            .status()
            .unwrap();

        if let Some(lib) = &*LIB.lock() {
            let func: libloading::Symbol<unsafe extern "C" fn(&mut World)> =
                unsafe { lib.get(b"unregister").unwrap() };

            let mut world = params.lock();
            world.clear();
            unsafe {
                func(&mut world);
            }
        }
        // *LIB.lock() = None;

        // DELETE OLD LIB
        let id = *ID.lock();
        *ID.lock() += 1;
        let so_file = format!("test_project_rs/runtime/libtest_project_rs{}.so", id);
        Command::new("rm")
            .args(&[
                &so_file,
            ])
            .status()
            .unwrap();
        
        // RELOCATE NEW LIB --- avoid dylib caching
        let id = id + 1;
        let so_file = format!("test_project_rs/runtime/libtest_project_rs{}.so", id);
        Command::new("mv")
            .args(&[
                "test_project_rs/target/release/libtest_project_rs.so",
                &so_file,
            ])
            .status()
            .unwrap();

        // LOAD NEW LIB
        let lib = unsafe { libloading::Library::new(so_file).unwrap() };
        let func: libloading::Symbol<unsafe extern "C" fn(&mut World)> =
            unsafe { lib.get(b"register").unwrap() };

        let mut world = params.lock();
        world.clear();
        unsafe {
            func(&mut world);
        }
        crate::serialize::deserialize(&mut world);
        *LIB.lock() = Some(lib);

        RSFile { path: file.into() }
    }

    fn reload(&mut self, file: &str, params: &(Arc<Mutex<World>>)) {
        *self = Self::from_file(file, params);
        // Command::new("cargo")
        //     .args(&["rustc", "--manifest-path=test_project_rs/Cargo.toml", "-r"])
        //     .status()
        //     .unwrap();
        // Command::new("cp")
        //     .args(&[
        //         "test_project_rs/target/release/libtest_project_rs.so",
        //         "test_project_rs/runtime",
        //         "-f",
        //     ])
        //     .status()
        //     .unwrap();
    }
}
impl Inspectable_ for RSFile {
    fn inspect(&mut self, ui: &mut egui::Ui, world: &Mutex<World>) {
        ui.add(egui::Label::new(self.path.as_str()));
    }
}
pub type RSManager = AssetManager<(Arc<Mutex<World>>), RSFile>;

impl Asset<Library, (Arc<Mutex<World>>)> for Library {
    fn from_file(file: &str, params: &(Arc<Mutex<World>>)) -> Library {
        // std::thread::sleep(Duration::from_millis(10));
        // #[cfg(target_os = "windows")]
        // let lib = unsafe { libloading::Library::new(file).unwrap() };
        // #[cfg(target_os = "linux")]
        // let lib = unsafe { libloading::Library::from(libloading::os::unix::Library::open(Some(file), libloading::os::unix::RTLD_LAZY).unwrap()) };

        let lib = unsafe { libloading::Library::new(file).unwrap() };

        let func: libloading::Symbol<unsafe extern "C" fn(&mut World)> =
            unsafe { lib.get(b"register").unwrap() };

        let mut world = params.lock();
        world.clear();
        unsafe {
            func(&mut world);
        }
        crate::serialize::deserialize(&mut world);
        lib

        // todo!()
    }

    fn unload(&mut self, file: &str, params: &(Arc<Mutex<World>>)) {
        let func: libloading::Symbol<unsafe extern "C" fn(&mut World)> =
            unsafe { self.get(b"unregister").unwrap() };

        let mut world = params.lock();
        world.clear();
        unsafe {
            func(&mut world);
        }
    }

    fn reload(&mut self, file: &str, params: &(Arc<Mutex<World>>)) {
        *self = Self::from_file(file, params);
        println!("reload {}", file);
    }
}

impl Inspectable_ for Library {
    fn inspect(&mut self, ui: &mut egui::Ui, world: &Mutex<World>) {
        // ui.add(egui::Label::new(self.path));
    }
}
pub type LibManager = AssetManager<(Arc<Mutex<World>>), Library>;
