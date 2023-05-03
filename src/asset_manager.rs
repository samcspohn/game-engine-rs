use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use egui::Ui;
use parking_lot::Mutex;
use serde::{
    Deserialize, Serialize,
};


use crate::{engine::World, inspectable::Inspectable_};

pub trait Asset<T, P> {
    fn from_file(file: &str, params: &P) -> T;
    fn reload(&mut self, file: &str, params: &P);
    fn save(&mut self, _file: &str, _params: &P) {}
    fn new(_file: &str, _params: &P) -> Option<T> {None}
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
    pub assets: BTreeMap<String, i32>,
    #[serde(skip_serializing, skip_deserializing)]
    pub assets_id: HashMap<i32, Arc<Mutex<T>>>,
    pub id_gen: i32,
    #[serde(skip_serializing, skip_deserializing)]
    pub const_params: P,
    // #[serde(skip_serializing, skip_deserializing)]
    // createable_asset: bool,
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
            assets: BTreeMap::new(),
            assets_id: HashMap::new(),
            id_gen: 0,
            const_params,
            // createable_asset,
        }
    }
    pub fn get_id(&self, id: &i32) -> Option<&Arc<Mutex<T>>> {
        self.assets_id.get(id)
    }
}

impl<P, T: 'static + Inspectable_ + Asset<T, P>> AssetManagerBase for AssetManager<P, T> {
    fn inspect(&mut self, file: &str, ui: &mut Ui, world: &Mutex<World>) {
        // let file = std::path::Path::new(file);
        if let Some(a) = self.assets.get(file) {
            if let Some(a) = self.assets_id.get_mut(a) {
                a.lock().inspect(ui, world);
            }
        }
    }

    fn from_file(&mut self, file: &str) -> i32 {
        // let _file = std::path::Path::new(file);
        if let Some(id) = self.assets.get_mut(file) {
            *id
        } else {
            let id = self.id_gen;
            self.id_gen += 1;
            self.assets.insert(file.into(), id);
            self.assets_id.insert(
                id,
                Arc::new(Mutex::new(<T as Asset<T, P>>::from_file(
                    file,
                    &self.const_params,
                ))),
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
                Arc::new(Mutex::new(<T as Asset<T, P>>::from_file(
                    &f,
                    &self.const_params,
                ))),
            );
            self.assets.insert(f, id);
        }
        self.id_gen = meta.get("id_gen").unwrap().as_i64().unwrap() as i32;
    }
    fn reload(&mut self, path: &str) {
        // let _file = std::path::Path::new(path);
        if let Some(id) = self.assets.get(path) {
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
        // let _file = std::path::Path::new(path);
        if let Some(id) = self.assets.get(path) {
            self.assets_id.remove(id);
            self.assets.remove(path);
        }
    }

    fn inspectable(&mut self, file: &str) -> Option<Arc<Mutex<dyn Inspectable_>>> {
        // let file = std::path::Path::new(file);
        if let Some(a) = self.assets.get(file) {
            if let Some(a) = self.assets_id.get(a) {
                let a = a.clone();
                Some(a)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn new_asset(&mut self, file: &str) -> i32 {
        if let Some(id) = self.assets.get(file) {
            *id
        } else if let Some(mut asset) = <T as Asset<T, P>>::new(
            file,
            &self.const_params,
        ) {
            let id = self.id_gen;
            self.id_gen += 1;
            asset.save(file, &self.const_params);

            self.assets.insert(file.into(), id);
            self.assets_id.insert(
                id,
                Arc::new(Mutex::new(asset)),
            );
            id
        } else {
            -1
        }
    }

    fn serialize(&self) -> serde_yaml::Value {
        serde_yaml::to_value(self).unwrap()
    }

    fn save(&self) {
        for (n, i) in &self.assets {
            if let Some(a) = self.assets_id.get(i) {
                a.lock().save(n, &self.const_params);
            }
        }
    }

    fn move_file(&mut self, from: &str, to: &str) {
        let asset_id = *self.assets.get(from).unwrap();
        self.assets.remove(from);
        self.assets.insert(to.to_owned(), asset_id);
    }
}

pub trait AssetManagerBase {
    fn inspect(&mut self, file: &str, ui: &mut Ui, world: &Mutex<World>);
    fn from_file(&mut self, file: &str) -> i32;
    fn regen(&mut self, meta: serde_yaml::Value);
    fn reload(&mut self, path: &str);
    fn remove(&mut self, path: &str);
    fn inspectable(&mut self, file: &str) -> Option<Arc<Mutex<dyn Inspectable_>>>;
    fn new_asset(&mut self,file: &str) -> i32;
    fn serialize(&self) -> serde_yaml::Value;
    fn save(&self);
    fn move_file(&mut self, from: &str, to: &str);
}

pub struct AssetsManager {
    pub asset_managers_names: HashMap<String, Arc<Mutex<dyn AssetManagerBase>>>,
    pub asset_managers_ext: HashMap<String, Arc<Mutex<dyn AssetManagerBase>>>
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
            asset_managers_names: HashMap::new(),
            asset_managers_ext: HashMap::new(),
        }
    }
    pub fn add_asset_manager(&mut self, name: &str, ext: &[&str], am: Arc<Mutex<dyn AssetManagerBase>>) {
        self.asset_managers_names.insert(name.into(), am.clone());
        for ext in ext {
            self.asset_managers_ext.insert((*ext).into(), am.clone());
        }
    }
    pub fn inspect(&mut self, file: &str) -> Option<Arc<Mutex<dyn Inspectable_>>> {
        let path = std::path::Path::new(file);
        // if let Some(dot) = file.rfind(".") {
        if let Some(ext) = path.extension() {
            let ext = ext.to_str().unwrap();
            if let Some(o) = self.asset_managers_ext.get_mut(ext) {
                o.lock().inspectable(file)
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn load(&mut self, file: &str) {
        let path = std::path::Path::new(file);
        // if let Some(dot) = file.rfind(".") {
        if let Some(ext) = path.extension() {
            if let Some(o) = self.asset_managers_ext.get_mut(ext.to_str().unwrap()) {
                o.lock().from_file(file);
            } else {
            }
        }
    }
    pub fn new_asset(&mut self, file: &str) -> i32 {
        println!("new asset {file}");
        let path = std::path::Path::new(file);
        if let Some(ext) = path.extension() {
            let ext = ext.to_str().unwrap();
            if let Some(o) = self.asset_managers_ext.get_mut(ext) {
                o.lock().new_asset(file)
            } else {
                -1
            }
        }else {
            -1
        }
    }
    pub fn remove(&mut self, file: &str) {
        // println!("remove asset {file}");
        let path = std::path::Path::new(file);
        if let Some(ext) = path.extension() {
            let ext = ext.to_str().unwrap();
            if let Some(o) = self.asset_managers_ext.get_mut(ext) {
                o.lock().remove(file);
            }
        }
    }
    pub fn reload(&mut self, file: &str) {
        // println!("remove asset {file}");
        let path = std::path::Path::new(file);
        if let Some(ext) = path.extension() {
            let ext = ext.to_str().unwrap();
            if let Some(o) = self.asset_managers_ext.get_mut(ext) {
                o.lock().reload(file);
            }
        }
    }
    pub fn move_file(&mut self, from: &str, to: &str) {
        // println!("remove asset {file}");
        let path = std::path::Path::new(from);
        if let Some(ext) = path.extension() {
            let ext = ext.to_str().unwrap();
            if let Some(o) = self.asset_managers_ext.get_mut(ext) {
                o.lock().move_file(from, to);
            }
        }
    }
    pub fn serialize(&self) -> BTreeMap<String,serde_yaml::Value> {
        let mut r = BTreeMap::new();
        for (n,am) in &self.asset_managers_names {
            r.insert(n.to_owned(), am.lock().serialize());
        }
        r
    }
    pub fn deserialize(&mut self, val: BTreeMap<String,serde_yaml::Value>) {
        for (n, v) in val {
            let a = self.asset_managers_names.get(&n).unwrap();
            a.lock().regen(v);
        }
    }
    pub fn save_assets(&self) {
        for (_,a) in &self.asset_managers_names {
            a.lock().save();
        }
    }
}
