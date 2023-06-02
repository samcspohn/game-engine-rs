use std::{process::Command, sync::Arc, time::Duration};

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

impl RSFile {
    // const ID: Mutex<usize> = Mutex::new(0);
}
impl Asset<RSFile, ()> for RSFile {
    fn from_file(file: &str, params: &()) -> RSFile {
        // Command::new("cargo").args(&["clean", "--manifest-path=test_project_rs/Cargo.toml"]).status().unwrap();
        Command::new("cargo")
            .args(&["build", "--manifest-path=test_project_rs/Cargo.toml", "-r"])
            .status()
            .unwrap();
        // Command::new("cp").args(&["test_project_rs/target/release/libtest_project_rs.so","test_project_rs/runtime", "-f"]).status().unwrap();
        RSFile { path: file.into() }
    }

    fn reload(&mut self, file: &str, params: &()) {
        Command::new("cargo")
        .args(&["rustc", "--manifest-path=test_project_rs/Cargo.toml", "-r"])
        .status()
        .unwrap();
    // Command::new("cp").args(&["test_project_rs/target/release/libtest_project_rs.so",format!("test_project_rs/target/release/libtest_project_rs{}.so", *Self::ID.lock()).as_str()]).status().unwrap();
    // Command::new("rm").args(&["test_project_rs/target/release/libtest_project_rs.so"]).status().unwrap();
        // *Self::ID.lock() += 1;
    }
}
impl Inspectable_ for RSFile {
    fn inspect(&mut self, ui: &mut egui::Ui, world: &Mutex<World>) {
        ui.add(egui::Label::new(self.path.as_str()));
    }
}
pub type RSManager = AssetManager<(), RSFile>;

impl Asset<Library, (Arc<Mutex<World>>)> for Library {
    fn from_file(file: &str, params: &(Arc<Mutex<World>>)) -> Library {
        // std::thread::sleep(Duration::from_millis(10));
        #[cfg(target_os = "windows")]
        let lib = unsafe { libloading::Library::new(file).unwrap() };
        #[cfg(target_os = "linux")]
        let lib = unsafe { libloading::Library::from(libloading::os::unix::Library::open(Some(file), libloading::os::unix::RTLD_LAZY).unwrap()) };

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
