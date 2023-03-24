use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use egui::Ui;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize, ser::{SerializeStruct, SerializeMap}};
use substring::Substring;

use crate::{engine::World, inspectable::Inspectable_};

pub trait Asset<T, P> {
    fn from_file(file: &str, params: &P) -> T;
    fn reload(&mut self, file: &str, params: &P);
    fn save(&mut self, file: &str, params: &P) {}
    // fn from_file_id(file: &str, id: i32) -> T;
}

struct TestAsset {}

// impl Asset<TestAsset> for TestAsset {
//     fn from_file(file: &str) -> TestAsset {
//         TestAsset {  }
//     }
// }

#[derive(Serialize, Deserialize)]
pub struct AssetManager<P, T: Inspectable_ + Asset<T, P>> {
    pub assets: HashMap<std::path::PathBuf, i32>,
    #[serde(skip_serializing, skip_deserializing)]
    pub assets_id: HashMap<i32, Arc<Mutex<T>>>,
    pub id_gen: i32,
    #[serde(skip_serializing, skip_deserializing)]
    pub const_params: P,
}

// use serde;
// impl <P, T: 'static + Inspectable_ + Asset<T, P>> Serialize for AssetManager<P, T> {
//     fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
//     where
//         S: serde::Serializer {
        
//             let mut m = serializer.serialize_map(Some(self.assets.len()))?;
//             for (p,i) in &self.assets {
//                 let path = std::path::Path::new(p.as_str());
//                 m.serialize_entry(path, &i)?;
//             }
//             m.end();
//             let mut s = serializer.serialize_struct("asset_manager", 2)?;
//             s.serialize_field("assets", &m)?;
//         // s.
//         // // s.serialize_field("key", "value")?;
//         // s.end()
//         // // s.(Some(self.assets.len()))?;
//         // // s.end()
//     }
// }

impl<P, T: Inspectable_ + Asset<T, P>> AssetManager<P, T> {
    pub fn new(const_params: P) -> Self {
        Self {
            assets: HashMap::new(),
            assets_id: HashMap::new(),
            id_gen: 0,
            const_params,
        }
    }
    pub fn get_id(&self, id: &i32) -> Option<&Arc<Mutex<T>>> {
        self.assets_id.get(id)
    }
}

impl<P, T: 'static + Inspectable_ + Asset<T, P>> AssetManagerBase for AssetManager<P, T> {
    fn inspect(&mut self, file: &str, ui: &mut Ui, world: &Mutex<World>) {
        let file = std::path::Path::new(file);
        if let Some(a) = self.assets.get(file) {
            if let Some(a) = self.assets_id.get_mut(a) {
                a.lock().inspect(ui, world);
            }
        }
    }

    fn from_file(&mut self, file: &str) -> i32 {
        let _file = std::path::Path::new(file);
        if let Some(id) = self.assets.get_mut(_file) {
            *id
        } else {
            let id = self.id_gen;
            self.id_gen += 1;
            self.assets.insert(file.into(), id);
            self.assets_id.insert(
                id,
                Arc::new(Mutex::new(<T as Asset<T, P>>::from_file(file, &self.const_params))),
            );
            id
        }
    }
    fn regen(&mut self, meta: serde_yaml::Value) {
        let m = meta.get("assets").unwrap().as_mapping().unwrap();
        for i in m.into_iter() {
            println!("{:?}", i);
            // let (f, id) = i;
            let f: String = i.0.as_str().unwrap().into();
            let id = i.1.as_i64().unwrap() as i32;
            self.assets_id.insert(
                id,
                Arc::new(Mutex::new(<T as Asset<T, P>>::from_file(&f, &self.const_params))),
            );
            self.assets.insert(std::path::Path::new(&f).to_path_buf(), id);
        }
        self.id_gen = meta.get("id_gen").unwrap().as_i64().unwrap() as i32;
    }
    fn reload(&mut self, path: &str) {
        let _file = std::path::Path::new(path);
        if let Some(id) = self.assets.get(_file) {
            if let Some(a) = self.assets_id.get_mut(id) {
                a.lock().reload(path, &self.const_params);
                // let mesh = Mesh::load_model(
                //     path,
                //     self.device.clone(),
                //     self.texture_manager.clone(),
                //     &self.allocator,
                // );
                // m.mesh = mesh;
            }
        }
    }
    fn remove(&mut self, path: &str) {
        let _file = std::path::Path::new(path);
        if let Some(id) = self.assets.get(_file) {
            self.assets_id.remove(id);
            self.assets.remove(_file);
        }
    }

    fn inspectable(&mut self, file: &str) -> Option<Arc<Mutex<dyn Inspectable_>>> {
        let file = std::path::Path::new(file);
        if let Some(a) = self.assets.get(file) {
            if let Some(a) = self.assets_id.get(a) {
                let a = a.clone();
                Some(a)
            }else {
                None
            }
        }else {
            None
        }
    }
}

pub trait AssetManagerBase {
    fn inspect(&mut self, file: &str, ui: &mut Ui, world: &Mutex<World>);
    fn from_file(&mut self, file: &str) -> i32;
    fn regen(&mut self, meta: serde_yaml::Value);
    fn reload(&mut self, path: &str);
    fn remove(&mut self, path: &str);
    fn inspectable(&mut self, file: &str) ->  Option<Arc<Mutex<dyn Inspectable_>>> ;
    // fn new(&mut self, const_params: P);
}

pub struct AssetsManager {
    pub asset_managers: HashMap<String, Arc<Mutex<dyn AssetManagerBase>>>,
}

impl AssetsManager {
    // pub fn get(&self, ext: &str) -> Option<Arc<dyn AssetManagerBase>> {
    //     if let Some(o) = self.asset_managers.get(ext) {
    //         Some(o.clone())
    //     } else {
    //         None
    //     }
    // }
    pub fn new() -> Self {
        Self {
            asset_managers: HashMap::new(),
        }
    }
    pub fn add_asset_manager(&mut self, ext: &str, am: Arc<Mutex<dyn AssetManagerBase>>) {
        self.asset_managers.insert(ext.into(), am);
    }
    pub fn inspect(&mut self, file: &str) -> Option<Arc<Mutex<dyn Inspectable_>>>  {
        if let Some(dot) = file.rfind(".") {
            let ext = file.substring(dot, file.len());
            if let Some(o) = self.asset_managers.get_mut(ext) {
                o.lock().inspectable(file)
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn load(&mut self, file: &str) {
        if let Some(dot) = file.rfind(".") {
            let ext = file.substring(dot, file.len());
            if let Some(o) = self.asset_managers.get_mut(ext) {
                o.lock().from_file(file);
            } else {
            }
        }
    }
}
