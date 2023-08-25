pub mod component;
pub mod entity;
pub mod transform;

use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicI32, Ordering},
        Arc,
    },
    time::Instant,
};

use crossbeam::queue::SegQueue;
use force_send_sync::SendSync;
use parking_lot::{MappedRwLockReadGuard, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use rayon::{
    prelude::{
        IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
    },
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
    perf::Perf,
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
    pub(crate) components: HashMap<
        u64,
        Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>,
        nohash_hasher::BuildNoHashHasher<i32>,
    >,
    pub(crate) components_names:
        HashMap<String, Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>>,
    pub(crate) root: i32,
    pub sys: Sys,
    pub(crate) to_destroy: Mutex<Vec<i32>>,
    pub(crate) to_instantiate: Mutex<Vec<_EntityParBuilder>>,
    active_ent: i32,
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
            components: HashMap::default(),
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
            active_ent: 1,
        }
    }
    pub fn defer_instantiate(&mut self, perf: &Perf) {
        let mut a = self.to_instantiate.lock();
        let mut to_instantiate = Vec::new();
        std::mem::swap(a.as_mut(), &mut to_instantiate);
        drop(a);
        for a in to_instantiate.into_iter() {
            


            if let Some(t_func) = a.t_func {
                let inst = Instant::now();
                let world_instantiate_transforms = perf.node("world instantiate transforms");
                let t = self
                    .transforms
                    .multi_transform_with(a.parent, a.count, t_func);
                drop(world_instantiate_transforms);
                let world_instantiate_ent = perf.node("world instantiate entities");
                let mut ent = self.entities.write();

                for i in t.get() {
                    if (*i as usize) < ent.len() {
                        *ent[*i as usize].lock() = Some(Entity::new());
                    } else {
                        break;
                    }
                }
                if let Some(last) = t.get().last() {
                    while ent.len() <= *last as usize {
                        ent.push(Mutex::new(Some(Entity::new())));
                    }
                }


                drop(ent);
                drop(world_instantiate_ent);
                let instantiate_components = perf.node("world instantiate _ components");
                a.comp_funcs.par_iter().for_each(|b| {
                    b(self, t.get(), perf);
                });
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
            let c_id = stor.insert(g, d);
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
                stor.write().remove(c_id);
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

    fn __destroy<'a>(
        transforms: &Transforms,
        components: &HashMap<
            u64,
            RwLockReadGuard<Box<dyn StorageBase + Send + Sync>>,
            nohash_hasher::BuildNoHashHasher<i32>,
        >,
        sys: &Sys,
        _t: i32,
        entities: &'a RwLockWriteGuard<Vec<Mutex<Option<Entity>>>>,
    ) {
        let g = _t;

        // delete children first
        {
            let children: Vec<i32> = transforms.get(_t).get_meta().children.iter().copied().collect();
            for t in children {
                Self::__destroy(transforms, components, sys, t, entities);
            }
        }
        {
            let trans = transforms.get(_t);
            let mut ent = entities[g as usize].lock();
            if let Some(ent_mut) = ent.as_mut() {
                for (t, id) in ent_mut.components.iter() {
                    let stor = &mut components.get(t).unwrap();

                    stor.deinit(&trans, *id, &sys);
                    stor.remove(*id);
                }
                // remove entity
                *ent = None;
            }
        }

        // remove transform
        transforms.remove(_t);
    }
    pub(crate) fn _destroy(&mut self, perf: &Perf) {
        {
            // let inst = Instant::now();
            let world_destroy_todestroy = perf.node("world _destroy to_destroy");
            let mut ent = self.entities.write();
            // let _self = Arc::new(&self);
            let mut to_destroy = self.to_destroy.lock();

            let mut unlocked: HashMap<
                u64,
                RwLockReadGuard<Box<dyn StorageBase + 'static + Sync + Send>>,
                nohash_hasher::BuildNoHashHasher<i32>,
            > = HashMap::default();
            self.components.iter().for_each(|c| {
                unlocked.insert(*c.0, c.1.read());
            });
            // to_destroy.sort();

            to_destroy.par_iter().for_each(|t| {
                Self::__destroy(&self.transforms, &unlocked, &self.sys, *t, &ent);
            });
            to_destroy.clear();
        }
        let world_destroy_reduce = perf.node("world _destroy reduce");
        rayon::scope(|s| {
            s.spawn(|s| {
                self.transforms.reduce_last();
            });
            self.components.iter().for_each(|(id, c)| {
                s.spawn(|s| {
                    c.write().reduce_last();
                });
            });
        });

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
    pub(crate) fn _update(&mut self, input: &Input, gpu_work: &GPUWork, perf: &Perf) {
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
                let world_update = perf.node("world update");
                let mut stor = stor.write();
                stor.update(&self.transforms, &sys, &self);
            }
            for (_, stor) in self.components.iter() {
                let world_update = perf.node("world late_update");
                let mut stor = stor.write();
                stor.late_update(&self.transforms, &sys);
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
        let mut tperf = Perf::new();
        self._destroy(&mut tperf);

        let mut unlocked: HashMap<
            u64,
            RwLockReadGuard<Box<dyn StorageBase + 'static + Sync + Send>>,
            nohash_hasher::BuildNoHashHasher<i32>,
        > = HashMap::default();
        self.components.iter().for_each(|c| {
            unlocked.insert(*c.0, c.1.read());
        });

        for (_t, i) in self.entities.read().iter().enumerate() {
            let mut a = i.lock();
            if let Some(ent) = a.as_mut() {
                let trans = self.transforms.get(_t as i32);
                for (t, id) in ent.components.iter() {
                    let stor = &mut unlocked.get(t).unwrap();
                    stor.deinit(&trans, *id, &self.sys);
                    stor.remove(*id);
                }
            }
            // remove entity
            *a = None;
        }

        self.sys.renderer_manager.write().clear();
        self.sys.physics.lock().clear();

        self.transforms.clean();
        self.root = self.transforms.new_root();
        self.entities.write().push(Mutex::new(None));
    }
}
