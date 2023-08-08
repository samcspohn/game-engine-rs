pub mod component;
pub mod entity;
pub mod transform;

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicI32, Ordering},
        Arc,
    },
};

use crossbeam::queue::SegQueue;
use force_send_sync::SendSync;
use parking_lot::{MappedRwLockReadGuard, Mutex, RwLock, RwLockWriteGuard};
use rayon::{
    prelude::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator},
    Scope,
};
use serde::{Deserialize, Serialize};
use thincollections::thin_map::ThinMap;

use crate::editor::inspectable::Inspectable;

use self::{
    component::{Component, System, _ComponentID},
    entity::{Entity, EntityParBuilder, _EntityParBuilder},
    transform::{Transform, Transforms},
};

use super::{
    input::Input,
    particles::{component::ParticleEmitter, particles::ParticleCompute},
    physics::Physics,
    project::asset_manager::{AssetManagerBase, AssetsManager},
    rendering::{
        camera::{Camera, CameraData},
        model::ModelRenderer,
        renderer_component::RendererManager,
        vulkan_manager::VulkanManager,
    },
    storage::{Storage, StorageBase},
    utils::GPUWork,
    Defer, RenderJobData,
};

pub struct Sys {
    // pub model_manager: Arc<parking_lot::Mutex<ModelManager>>,
    pub renderer_manager: Arc<RwLock<RendererManager>>,
    pub assets_manager: Arc<AssetsManager>,
    pub physics: Arc<Mutex<Physics>>,
    pub particles_system: Arc<ParticleCompute>,
    pub vk: Arc<VulkanManager>,
    pub defer: Defer,
}

impl Sys {
    pub fn get_model_manager(&self) -> Arc<Mutex<dyn AssetManagerBase + Send + Sync>> {
        let b = &self.assets_manager;
        let a = b.get_manager::<ModelRenderer>().clone();
        a
    }
}

pub struct World {
    pub(crate) entities: RwLock<Vec<Mutex<Option<Entity>>>>,
    pub(crate) transforms: Transforms,
    pub(crate) components:
        SendSync<ThinMap<u64, Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>>>,
    pub(crate) components_names:
        HashMap<String, Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>>,
    pub(crate) root: i32,
    pub sys: Sys,
    pub(crate) to_destroy: Mutex<Vec<i32>>,
    pub(crate) to_instantiate: Mutex<Vec<_EntityParBuilder>>,
}

//  T_ = 'static
//     + Send
//     + Sync
//     + Component
//     + Inspectable
//     + Default
//     + Clone
//     + Serialize
//     + for<'a> Deserialize<'a>;
#[allow(dead_code)]
impl World {
    pub fn new(
        particles: Arc<ParticleCompute>,
        vk: Arc<VulkanManager>,
        assets_manager: Arc<AssetsManager>,
    ) -> World {
        let mut trans = Transforms::new();
        let root = trans.new_root();
        World {
            entities: RwLock::new(vec![Mutex::new(None)]),
            transforms: trans,
            components: unsafe { SendSync::new(ThinMap::new()) },
            components_names: HashMap::new(),
            root,
            sys: Sys {
                renderer_manager: Arc::new(RwLock::new(RendererManager::new(
                    vk.device.clone(),
                    vk.mem_alloc.clone(),
                ))),
                assets_manager,
                physics: Arc::new(Mutex::new(Physics::new())),
                particles_system: particles,
                vk: vk,
                defer: Defer::new(),
            },
            to_destroy: Mutex::new(Vec::new()),
            to_instantiate: Mutex::new(Vec::new()),
        }
    }
    pub fn defer_instantiate(&mut self) {
        let mut a = self.to_instantiate.lock();
        let mut to_instantiate = Vec::new();
        std::mem::swap(a.as_mut(), &mut to_instantiate);
        drop(a);
        for a in to_instantiate.into_iter() {
            if let Some(t_func) = a.t_func {
                let t = self
                    .transforms
                    .multi_transform_with(a.parent, a.count, t_func);

                let mut ent = self.entities.write();

                for i in &t {
                    if (*i as usize) < ent.len() {
                        *ent[*i as usize].lock() = Some(Entity::new());
                    } else {
                        break;
                    }
                }
                if let Some(last) = t.last() {
                    while ent.len() <= *last as usize {
                        ent.push(Mutex::new(Some(Entity::new())));
                    }
                }
                drop(ent);

                for b in a.comp_funcs.iter() {
                    b(self, &t);
                }
            }
        }
        // to_instantiate.clear();
    }
    pub fn instantiate_many(&self, count: i32, chunk: i32, parent: i32) -> EntityParBuilder {
        EntityParBuilder::new(parent, count, chunk, &self)
    }
    pub fn instantiate(&mut self) -> i32 {
        let ret = self.transforms.new_transform(self.root);
        {
            let entities = &mut self.entities.write();
            if ret as usize >= entities.len() {
                entities.push(Mutex::new(Some(Entity::new())));
            } else {
                entities[ret as usize] = Mutex::new(Some(Entity::new()));
            }
        }
        ret
    }
    pub fn instantiate_with_transform(&mut self, transform: transform::_Transform) -> i32 {
        let ret = self.transforms.new_transform_with(self.root, transform);
        {
            let entities = &mut self.entities.write();
            if ret as usize >= entities.len() {
                entities.push(Mutex::new(Some(Entity::new())));
            } else {
                entities[ret as usize] = Mutex::new(Some(Entity::new()));
            }
        }
        ret
    }
    fn copy_game_object_child(&mut self, t: i32, new_parent: i32) {
        let tr = self.transforms.get(t).get_transform();
        let g = self.instantiate_with_transform_with_parent(new_parent, tr);
        let entities = self.entities.read();
        let src = entities[t as usize].lock();
        let mut dest = entities[g as usize].lock();
        if let (Some(src_ent), Some(dest_ent)) = (&mut src.as_ref(), &mut dest.as_mut()) {
            for c in src_ent.components.iter() {
                dest_ent.components.insert(
                    c.0.clone(),
                    self.copy_component_id(&self.transforms.get(g), c.0.clone(), *c.1),
                );
            }
            let children: Vec<i32> = self
                .transforms
                .get(t)
                .get_meta()
                .children
                .iter()
                .copied()
                .collect();
            drop(src);
            drop(dest);
            drop(entities);
            for c in children {
                self.copy_game_object_child(c, g);
            }
        } else {
            panic!("copy object not valid");
        }
    }
    pub fn copy_game_object(&mut self, t: i32) -> i32 {
        let trans = self.transforms.get(t);
        let parent = { trans.get_parent().id };
        let tr = trans.get_transform();
        drop(trans);
        let g = self.instantiate_with_transform_with_parent(parent, tr);
        let entities = self.entities.read();
        let src = entities[t as usize].lock();
        let mut dest = entities[g as usize].lock();
        if let (Some(src_ent), Some(dest_ent)) = (&mut src.as_ref(), &mut dest.as_mut()) {
            for c in src_ent.components.iter() {
                dest_ent.components.insert(
                    c.0.clone(),
                    self.copy_component_id(&self.transforms.get(g), c.0.clone(), *c.1),
                );
            }
            let children: Vec<i32> = self
                .transforms
                .get(t)
                .get_meta()
                .children
                .iter()
                .copied()
                .collect();
            drop(src);
            drop(dest);
            drop(entities);
            for c in children {
                self.copy_game_object_child(c, g);
            }
        } else {
            panic!("copy object not valid");
        }
        g
    }
    fn copy_component_id(&self, t: &Transform, key: u64, c_id: i32) -> i32 {
        if let Some(stor) = self.components.get(&key) {
            let mut stor = stor.write();
            let c = stor.copy(t.id, c_id);
            stor.init(&t, c, &self.sys);
            c
        } else {
            panic!("no component storage for key");
        }
    }
    pub fn instantiate_with_transform_with_parent(
        &mut self,
        parent: i32,
        transform: transform::_Transform,
    ) -> i32 {
        // let mut trans = self.transforms.;
        let ret = self.transforms.new_transform_with(parent, transform);
        {
            let entities = &mut self.entities.write();
            if ret as usize >= entities.len() {
                entities.push(Mutex::new(Some(Entity::new())));
            } else {
                entities[ret as usize] = Mutex::new(Some(Entity::new()));
            }
        }
        ret
    }
    pub fn add_component<
        T: 'static
            + Send
            + Sync
            + Component
            + _ComponentID
            + Inspectable
            + Default
            + Clone
            + Serialize
            + for<'a> Deserialize<'a>,
    >(
        &mut self,
        g: i32,
        d: T,
    ) {
        // d.assign_transform(g);
        let key = T::ID;
        if let Some(stor) = self.components.get(&key) {
            let stor: &mut Storage<T> = unsafe { std::mem::transmute(&mut stor.write()) };
            let c_id = stor.emplace(g, d);
            let trans = self.transforms.get(g);
            stor.init(&trans, c_id, &self.sys);
            if let Some(ent) = self.entities.read()[g as usize].lock().as_mut() {
                ent.components.insert(key, c_id);
            }
        } else {
            panic!("no type key?")
        }
    }
    pub fn add_component_id(&mut self, g: i32, key: u64, c_id: i32) {
        if let Some(ent) = self.entities.read()[g as usize].lock().as_mut() {
            ent.components.insert(key.clone(), c_id);
            if let Some(stor) = self.components.get(&key) {
                let trans = self.transforms.get(g);
                stor.write().init(&trans, c_id, &self.sys);
            }
        }
    }
    pub fn deserialize(&mut self, g: i32, key: String, s: serde_yaml::Value) {
        if let Some(stor) = self.components_names.get(&key) {
            let mut stor = stor.write();
            let c_id = stor.deserialize(g, s);
            let trans = self.transforms.get(g);
            stor.init(&trans, c_id, &self.sys);
            if let Some(ent) = self.entities.read()[g as usize].lock().as_mut() {
                ent.components.insert(stor.get_id(), c_id);
            }
        } else {
            panic!("no type key: {}", key);
        }
    }
    pub fn remove_component(&mut self, g: i32, key: u64, c_id: i32) {
        if let Some(ent) = self.entities.read()[g as usize].lock().as_mut() {
            if let Some(stor) = self.components.get(&key) {
                let trans = self.transforms.get(g);
                stor.write().deinit(&trans, c_id, &self.sys);
                stor.write().erase(c_id);
            }
            ent.components.remove(&key);
        }
    }
    pub fn register<
        T: 'static
            + Send
            + Sync
            + Component
            + _ComponentID
            + Inspectable
            + Default
            + Clone
            + Serialize
            + for<'a> Deserialize<'a>,
    >(
        &mut self,
        has_update: bool,
        has_late_update: bool,
        has_render: bool,
    ) {
        let key = T::ID;
        let data = Storage::<T>::new(has_update, has_late_update, has_render);
        let component_storage: Arc<RwLock<Box<dyn StorageBase + Send + Sync + 'static>>> =
            Arc::new(RwLock::new(Box::new(data)));
        self.components.insert(key, component_storage.clone());
        self.components_names.insert(
            component_storage.read().get_name().to_string(),
            component_storage.clone(),
        );
    }

    pub fn unregister<
        T: 'static
            + Send
            + Sync
            + Component
            + _ComponentID
            + Inspectable
            + Default
            + Clone
            + Serialize
            + for<'a> Deserialize<'a>,
    >(
        &mut self,
    ) {
        // let key: TypeId = TypeId::of::<T>();
        let key = T::ID;
        self.components.remove(&key);
        self.components_names
            .remove(std::any::type_name::<T>().split("::").last().unwrap());
    }
    pub fn destroy(&self, g: i32) {
        self.to_destroy.lock().push(g);
    }

    // fn __destroy<'a>(
    //     _self: &'a World,
    //     trans: Transform,
    //     entities: &'a RwLockWriteGuard<Vec<Mutex<Option<Entity>>>>,
    //     s: &Scope<'a>,
    // ) {
    //     let g = trans.id;
    //     let mut ent = entities[g as usize].lock();
    //     if let Some(ent_mut) = ent.as_mut() {
    //         for (t, id) in ent_mut.components.iter() {
    //             let stor = &mut _self.components.get(t).unwrap().write();

    //             stor.deinit(&trans, *id, &_self.sys);
    //             stor.erase(*id);
    //         }
    //         // remove entity
    //         *ent = None; // todo make read()
    //     }

    //     let children: Vec<i32> = trans.get_meta().children.iter().copied().collect();
    //     for t in children {
    //         let _self = _self.clone();
    //         s.spawn(move |s| {
    //             let t = _self.transforms.get(t);
    //             Self::__destroy(_self, t, entities, s);
    //         });
    //     }

    //     // remove transform
    //     _self.transforms.remove(trans);
    // }
    // pub(crate) fn _destroy(&mut self) {
    //     let ent = &self.entities.write();
    //     let _self = Arc::new(&self);
    //     rayon::scope(|s| {
    //         while let Some(t) = _self.to_destroy.pop() {
    //             let trans = _self.transforms.get(t);
    //             Self::__destroy(&_self, trans, &ent, &s);
    //         }
    //     });
    //     self.transforms.reduce_last();
    //     for (id, c) in self.components.iter() {
    //         c.read().reduce_last();
    //     }
    //     // remove/deinit components
    // }

    fn __destroy<'a>(
        _self: &'a World,
        trans: Transform<'a>,
        entities: &'a parking_lot::RwLockReadGuard<Vec<Mutex<Option<Entity>>>>,
        // s: &Scope<'a>,
    ) {
        let g = trans.id;
        let mut ent = entities[g as usize].lock();
        if let Some(ent_mut) = ent.as_mut() {
            for (t, id) in ent_mut.components.iter() {
                let stor = &_self.components.get(t).unwrap().read();

                stor.deinit(&trans, *id, &_self.sys);
                stor.erase(*id);
            }
            // remove entity
            *ent = None; // todo make read()
        }

        let children: Vec<i32> = trans.get_meta().children.iter().copied().collect();
        for t in children {
            // _self.to_destroy.push(t);
            // let _self = _self.clone();
            let t = _self.transforms.get(t);
            Self::__destroy(_self, t, entities);
        }

        // remove transform
        _self.transforms.remove(trans);
    }
    pub(crate) fn _destroy(&mut self) {
        {
            let ent = &self.entities.read();
            let _self = Arc::new(&self);
            let mut to_destroy = _self.to_destroy.lock();
            to_destroy.par_iter().for_each(|t| {
                let trans = _self.transforms.get(*t);
                Self::__destroy(&_self, trans, &ent);
            });
            to_destroy.clear();
        }
        self.transforms.reduce_last();
        for (id, c) in self.components.iter() {
            c.read().reduce_last();
        }
        // remove/deinit components
    }
    // pub fn get_component<T: 'static + Send + Sync + Component, F>(&self, g: i32, f: F)
    // where
    //     F: FnOnce(&Mutex<T>),
    // {
    //     let key: TypeId = TypeId::of::<T>();

    //     if let Some(components) = &self.entities.read()[g.0 as usize] {
    //         if let Some(id) = &components.read().get(&key) {
    //             if let Some(stor_base) = &self.components.get(&key) {
    //                 if let Some(stor) = stor_base.write().as_any().downcast_ref::<Storage<T>>() {
    //                     f(stor.get(id));
    //                 }
    //             }
    //         }
    //     }
    // }
    pub fn get_components<T: 'static + Send + Sync + Component + _ComponentID>(
        &self,
    ) -> Option<&Arc<RwLock<Box<dyn StorageBase + Send + Sync>>>> {
        let key = T::ID;
        self.components.get(&key)
    }

    pub fn do_defered(&mut self) {
        while let Some(w) = self.sys.defer.work.pop() {
            w(self);
        }
        // self.sys.defer.do_defered(self);
    }
    pub(crate) fn _update(&mut self, input: &Input, gpu_work: &GPUWork) {
        {
            let sys = &self.sys;
            let sys = System {
                physics: &sys.physics.lock(),
                defer: &sys.defer,
                input,
                rendering: &sys.renderer_manager,
                assets: &sys.assets_manager,
                vk: sys.vk.clone(),
                gpu_work,
                particle_system: &sys.particles_system,
            };
            for (_, stor) in self.components.iter() {
                stor.write().update(&self.transforms, &sys, &self);
            }
            for (_, stor) in self.components.iter() {
                stor.write().late_update(&self.transforms, &sys);
            }
        }
        self.update_cameras();
    }
    fn update_cameras(&mut self) {
        let camera_components = self.get_components::<Camera>().unwrap().read();
        let camera_storage = camera_components
            .as_any()
            .downcast_ref::<Storage<Camera>>()
            .unwrap();
        camera_storage
            .valid
            .iter()
            .zip(camera_storage.data.iter())
            .for_each(|(v, d)| {
                if unsafe { *v.get() } {
                    let mut d = d.lock();
                    let id: i32 = d.0;
                    d.1._update(&self.transforms.get(id));
                }
            });
    }
    pub(crate) fn editor_update(&mut self, input: &Input, gpu_work: &GPUWork) {
        let sys = &self.sys;
        let sys = System {
            physics: &sys.physics.lock(),
            defer: &sys.defer,
            input,
            rendering: &sys.renderer_manager,
            assets: &sys.assets_manager,
            vk: sys.vk.clone(),
            gpu_work,
            particle_system: &sys.particles_system,
        };
        for (_, stor) in self.components.iter() {
            stor.write().editor_update(&self.transforms, &sys, input);
        }
    }
    pub fn render(&self) -> Vec<Box<dyn Fn(&mut RenderJobData)>> {
        let mut render_jobs = vec![];
        for (_, stor) in self.components.iter() {
            stor.write().on_render(&mut render_jobs);
        }
        render_jobs
    }
    pub(crate) fn get_cam_datas(&mut self) -> (i32, Vec<Arc<Mutex<CameraData>>>) {
        let camera_components = self.get_components::<Camera>().unwrap().read();
        let camera_storage = camera_components
            .as_any()
            .downcast_ref::<Storage<Camera>>()
            .unwrap();
        let mut main_cam_id = -1;
        let cam_datas = camera_storage
            .valid
            .iter()
            .zip(camera_storage.data.iter())
            .map(|(v, d)| {
                if unsafe { *v.get() } {
                    let d = d.lock();
                    main_cam_id = d.0;
                    d.1.get_data()
                } else {
                    None
                }
            })
            .flatten()
            .collect();
        (main_cam_id, cam_datas)
    }
    pub(crate) fn get_emitter_len(&self) -> usize {
        self.get_components::<ParticleEmitter>()
            .unwrap()
            .read()
            .as_any()
            .downcast_ref::<Storage<ParticleEmitter>>()
            .unwrap()
            .data
            .len()
    }

    pub fn clear(&mut self) {
        self.destroy(self.root);
        self._destroy();
        // self.entities.write().clear();
        // self.transforms.clear();

        self.sys.renderer_manager.write().clear();
        self.sys.physics.lock().clear();

        self.root = self.transforms.new_root();
        self.entities.write().push(Mutex::new(None));
    }
}
