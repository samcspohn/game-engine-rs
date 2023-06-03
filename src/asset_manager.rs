use egui::Ui;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{
    any::{Any, TypeId},
    collections::{BTreeMap, HashMap},
    marker::PhantomData,
    sync::Arc,
};
use sync_unsafe_cell::SyncUnsafeCell;

use crate::{
    drag_drop::{self, drop_target},
    engine::World,
    inspectable::{Inpsect, Ins, Inspectable, Inspectable_},
};

#[derive(Deserialize, Serialize)]
#[serde(default)]
pub struct AssetInstance<T> {
    pub id: i32,
    #[serde(skip_serializing, skip_deserializing)]
    _pd: PhantomData<T>,
}
impl<T> AssetInstance<T> {
    pub fn new(id: i32) -> Self {
        Self {
            id,
            _pd: PhantomData,
        }
    }
}

impl<T> Default for AssetInstance<T> {
    fn default() -> Self {
        Self {
            id: 0,
            _pd: PhantomData,
        }
    }
}
impl<T> Clone for AssetInstance<T> {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            _pd: PhantomData,
        }
    }
}
impl<T> Copy for AssetInstance<T> {}

impl<'a, T: 'static> Inpsect for Ins<'a, AssetInstance<T>> {
    fn inspect(&mut self, name: &str, ui: &mut egui::Ui, sys: &crate::engine::Sys) {
        let drop_data = drag_drop::DRAG_DROP_DATA.lock();
        sys.assets_manager
            .inspect_instance::<T>(name, &drop_data, &mut self.0.id, ui)

        // let asset_manager = sys.assets_manager.asset_managers_type.get(&TypeId::of::<T>()).unwrap();

        // let model: String = match asset_manager.lock().assets_id.get(&self.0.id) {
        //     Some(model) => model.lock().file.clone(),
        //     None => "".into(),
        // };
        // let can_accept_drop_data = match drop_data.rfind(".obj") {
        //     Some(_) => true,
        //     None => false,
        // };
        // // println!("can accept drop data:{}",can_accept_drop_data);
        // ui.horizontal(|ui| {
        //     ui.add(egui::Label::new(name));
        //     drop_target(ui, can_accept_drop_data, |ui| {
        //         let response = ui.add(egui::Label::new(model.as_str()));
        //         if response.hovered() && ui.input().pointer.any_released() {
        //             let model_file: String = drop_data.clone();

        //             if let Some(id) = asset_manager.lock().assets.get(&model_file) {
        //                 self.0.id = *id;
        //             }
        //         }
        //     });
        // });
    }
    // fn inspect(
    //     &mut self,
    //     transform: &crate::engine::transform::Transform,
    //     id: i32,
    //     ui: &mut egui::Ui,
    //     sys: &crate::engine::Sys,
    // ) {
    //     let drop_data = drag_drop::DRAG_DROP_DATA.lock();

    //     let model: String = match sys.model_manager.lock().assets_id.get(&self.0.id) {
    //         Some(model) => model.lock().file.clone(),
    //         None => "".into(),
    //     };
    //     let can_accept_drop_data = match drop_data.rfind(".obj") {
    //         Some(_) => true,
    //         None => false,
    //     };
    //     // println!("can accept drop data:{}",can_accept_drop_data);
    //     ui.horizontal(|ui| {
    //         ui.add(egui::Label::new(name));
    //         drop_target(ui, can_accept_drop_data, |ui| {
    //             // let model_name = sys.model_manager.lock().models.get(k)
    //             let response = ui.add(egui::Label::new(model.as_str()));
    //             if response.hovered() && ui.input().pointer.any_released() {
    //                 let model_file: String = drop_data.clone();

    //                 if let Some(id) = sys.model_manager.lock().assets.get(&model_file) {
    //                     self.0.id = *id;
    //                 }
    //             }
    //         });
    //     });
    // }
}
pub trait _AssetInstance<T> {
    const ASSET: T;
}

impl<Model> AssetInstance<Model> {}

impl<T> AssetInstance<T> {
    // const ASSET: T = T;
    // pub fn inspect(&mut self) {
    //     fn inspect(&mut self, name: &str, ui: &mut egui::Ui, sys: &Sys) {
    //         let drop_data = drag_drop::DRAG_DROP_DATA.lock();

    //         let model: String = match sys.model_manager.lock().assets_id.get(&self.0.id) {
    //             Some(model) => model.lock().file.clone(),
    //             None => "".into(),
    //         };
    //         let can_accept_drop_data = match drop_data.rfind(".obj") {
    //             Some(_) => true,
    //             None => false,
    //         };
    //         // println!("can accept drop data:{}",can_accept_drop_data);
    //         ui.horizontal(|ui| {
    //             ui.add(egui::Label::new(name));
    //             drop_target(ui, can_accept_drop_data, |ui| {
    //                 // let model_name = sys.model_manager.lock().models.get(k)
    //                 let response = ui.add(egui::Label::new(model.as_str()));
    //                 if response.hovered() && ui.input().pointer.any_released() {
    //                     let model_file: String = drop_data.clone();

    //                     if let Some(id) = sys.model_manager.lock().assets.get(&model_file) {
    //                         self.0.id = *id;
    //                     }
    //                 }
    //             });
    //         });
    //     }
    // }
}

pub trait Asset<T, P> {
    fn from_file(file: &str, params: &P) -> T;
    fn reload(&mut self, file: &str, params: &P);
    fn unload(&mut self, file: &str, params: &P) {}
    fn save(&mut self, _file: &str, _params: &P) {}
    fn new(_file: &str, _params: &P) -> Option<T> {
        None
    }
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
    pub assets_r: BTreeMap<i32, String>,
    #[serde(skip_serializing, skip_deserializing)]
    pub assets_id: HashMap<i32, Arc<Mutex<T>>>,
    pub id_gen: i32,
    #[serde(skip_serializing, skip_deserializing)]
    pub const_params: P,
    #[serde(skip_serializing, skip_deserializing)]
    ext: Vec<String>,
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

impl<P: 'static, T: Inspectable_ + Asset<T, P>> AssetManager<P, T> {
    pub fn new(const_params: P, ext: &[&str]) -> Self {
        Self {
            assets: BTreeMap::new(),
            assets_r: BTreeMap::new(),
            assets_id: HashMap::new(),
            id_gen: 0,
            const_params,
            ext: ext.iter().map(|s| s.to_string()).collect(),
            // createable_asset,
        }
    }
    pub fn get_id(&self, id: &i32) -> Option<&Arc<Mutex<T>>> {
        self.assets_id.get(id)
    }
}

impl<P: 'static, T: 'static + Inspectable_ + Asset<T, P>> AssetManagerBase for AssetManager<P, T> {
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
            self.assets_r.insert(id, file.into());
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
            self.assets.insert(f.clone(), id);
            self.assets_r.insert(id, f);
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
            let id = *id;
            self.assets_id.get(&id).unwrap().lock().unload(path, &self.const_params);
            self.assets_id.remove(&id);
            self.assets.remove(path);
            self.assets_r.remove(&id);
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
        } else if let Some(mut asset) = <T as Asset<T, P>>::new(file, &self.const_params) {
            let id = self.id_gen;
            self.id_gen += 1;
            asset.save(file, &self.const_params);

            self.assets.insert(file.into(), id);
            self.assets_r.insert(id, file.into());
            self.assets_id.insert(id, Arc::new(Mutex::new(asset)));
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

    fn get_type(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn inspect_instance(&self, name: &str, path: &str, id: &mut i32, ui: &mut Ui) {
        // let drop_data = drag_drop::DRAG_DROP_DATA.lock();

        // let asset_manager = sys.assets_manager.asset_managers_type.get(&TypeId::of::<T>()).unwrap();

        // let label = id.to_string();
        let label: String = match self.assets_r.get(&id) {
            Some(file) => file.clone(),
            None => "".into(),
        };
        let can_accept_drop_data = self
            .ext
            .iter()
            .map(|e| match path.rfind(format!(".{}", e).as_str()) {
                Some(_) => true,
                None => false,
            })
            .any(|b| b);
        // println!("can accept drop data:{}",can_accept_drop_data);
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(name));
            drop_target(ui, can_accept_drop_data, |ui| {
                let response = ui.add(egui::Label::new(label.as_str()));
                if response.hovered() && ui.input().pointer.any_released() {
                    let model_file: String = path.to_string();

                    if let Some(_id) = self.assets.get(&model_file) {
                        *id = *_id;
                    }
                }
            });
        });
    }
    fn get_ext(&self) -> &[String] {
        self.ext.as_slice()
    }
    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
}

pub trait AssetManagerBase {
    fn inspect(&mut self, file: &str, ui: &mut Ui, world: &Mutex<World>);
    fn from_file(&mut self, file: &str) -> i32;
    fn regen(&mut self, meta: serde_yaml::Value);
    fn reload(&mut self, path: &str);
    fn remove(&mut self, path: &str);
    fn inspectable(&mut self, file: &str) -> Option<Arc<Mutex<dyn Inspectable_>>>;
    fn new_asset(&mut self, file: &str) -> i32;
    fn serialize(&self) -> serde_yaml::Value;
    fn save(&self);
    fn move_file(&mut self, from: &str, to: &str);
    fn get_type(&self) -> TypeId;
    fn inspect_instance(&self, name: &str, path: &str, id: &mut i32, ui: &mut Ui);
    fn get_ext(&self) -> &[String];
    fn as_any(&self) -> &dyn std::any::Any;
}

pub struct AssetsManager {
    pub asset_managers_names:
        SyncUnsafeCell<HashMap<String, Arc<Mutex<dyn AssetManagerBase + Send + Sync>>>>,
    pub asset_managers_ext:
        SyncUnsafeCell<HashMap<String, Arc<Mutex<dyn AssetManagerBase + Send + Sync>>>>,
    pub asset_managers_type:
        SyncUnsafeCell<HashMap<TypeId, Arc<Mutex<dyn AssetManagerBase + Send + Sync>>>>,
}

impl AssetsManager {
    // pub fn get(&self, ext: &str) -> Option<Arc<dyn AssetManagerBase>> {
    //     if let Some(o) = self.asset_managers.get(ext) {
    //         Some(o.clone())
    //     } else {
    //         None
    //     }
    // }
    pub fn get_manager<T: 'static>(
        &self,
    ) -> Arc<Mutex<dyn AssetManagerBase + Send + Sync>> {
        unsafe { &*self.asset_managers_type.get() }
            .get(&TypeId::of::<T>())
            .unwrap()
            .clone()
    }
    // pub fn get_manager_type<T: 'static>(
    //     &self,
    //     id: &AssetInstance<T>,
    // ) -> Arc<Mutex<dyn AssetManagerBase + Send + Sync>> {
    //     unsafe { &*self.asset_managers_type.get() }
    //         .get(&TypeId::of::<T>())
    //         .unwrap()
    //         .clone()
    // }
    pub fn inspect_instance<T: 'static>(&self, name: &str, path: &str, id: &mut i32, ui: &mut Ui) {
        let asset_manager = unsafe { &*self.asset_managers_type.get() }
            .get(&TypeId::of::<T>())
            .unwrap();
        asset_manager.lock().inspect_instance(name, path, id, ui);
    }
    pub fn new() -> Self {
        Self {
            asset_managers_names: SyncUnsafeCell::new(HashMap::new()),
            asset_managers_ext: SyncUnsafeCell::new(HashMap::new()),
            asset_managers_type: SyncUnsafeCell::new(HashMap::new()),
        }
    }
    pub unsafe fn add_asset_manager(
        &self,
        name: &str,
        // ext: &[&str],
        am: Arc<Mutex<dyn AssetManagerBase + Send + Sync>>,
    ) {
        (*self.asset_managers_names.get()).insert(name.into(), am.clone());
        for ext in am.lock().get_ext() {
            (*self.asset_managers_ext.get()).insert(ext.clone(), am.clone());
        }
        let t = am.lock().get_type();
        (*self.asset_managers_type.get()).insert(t, am);
    }
    pub fn inspect(&self, file: &str) -> Option<Arc<Mutex<dyn Inspectable_>>> {
        let path = std::path::Path::new(file);
        // if let Some(dot) = file.rfind(".") {
        if let Some(ext) = path.extension() {
            let ext = ext.to_str().unwrap();
            if let Some(o) = unsafe { &*self.asset_managers_ext.get() }.get(ext) {
                o.lock().inspectable(file)
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn load(&self, file: &str) {
        let path = std::path::Path::new(file);
        // if let Some(dot) = file.rfind(".") {
        if let Some(ext) = path.extension() {
            if let Some(o) = unsafe { &*self.asset_managers_ext.get() }.get(ext.to_str().unwrap()) {
                o.lock().from_file(file);
            } else {
            }
        }
    }
    pub fn new_asset(&self, file: &str) -> i32 {
        println!("new asset {file}");
        let path = std::path::Path::new(file);
        if let Some(ext) = path.extension() {
            let ext = ext.to_str().unwrap();
            if let Some(o) = unsafe { &*self.asset_managers_ext.get() }.get(ext) {
                o.lock().new_asset(file)
            } else {
                -1
            }
        } else {
            -1
        }
    }
    pub fn remove(&self, file: &str) {
        // println!("remove asset {file}");
        let path = std::path::Path::new(file);
        if let Some(ext) = path.extension() {
            let ext = ext.to_str().unwrap();
            if let Some(o) = unsafe { &*self.asset_managers_ext.get() }.get(ext) {
                o.lock().remove(file);
            }
        }
    }
    pub fn reload(&self, file: &str) {
        // println!("remove asset {file}");
        let path = std::path::Path::new(file);
        if let Some(ext) = path.extension() {
            let ext = ext.to_str().unwrap();
            if let Some(o) = unsafe { &*self.asset_managers_ext.get() }.get(ext) {
                o.lock().reload(file);
            }
        }
    }
    pub fn move_file(&self, from: &str, to: &str) {
        // println!("remove asset {file}");
        let path = std::path::Path::new(from);
        if let Some(ext) = path.extension() {
            let ext = ext.to_str().unwrap();
            if let Some(o) = unsafe { &*self.asset_managers_ext.get() }.get(ext) {
                o.lock().move_file(from, to);
            }
        }
    }
    pub fn serialize(&self) -> BTreeMap<String, serde_yaml::Value> {
        let mut r = BTreeMap::new();
        for (n, am) in unsafe { &*self.asset_managers_names.get() } {
            r.insert(n.to_owned(), am.lock().serialize());
        }
        r
    }
    pub fn deserialize(&self, val: BTreeMap<String, serde_yaml::Value>) {
        for (n, v) in &val {
            if n == "lib" {
                continue;
            }
            let a = unsafe { &*self.asset_managers_names.get() }.get(n).unwrap();
            a.lock().regen(v.clone());
        }
        // let a = unsafe { &*self.asset_managers_names.get() }.get("lib").unwrap();
        //     a.lock().regen(val["lib"].clone());
    }
    pub fn save_assets(&self) {
        for (_, a) in unsafe { &*self.asset_managers_names.get() } {
            a.lock().save();
        }
    }
}
