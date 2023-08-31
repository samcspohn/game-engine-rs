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
use thincollections::{thin_map::ThinMap, thin_vec::ThinVec};

use crate::editor::inspectable::Inspectable;

use self::{
    component::{Component, System, _ComponentID},
    entity::{Entity, EntityParBuilder, _EntityParBuilder},
    transform::{CacheVec, Transform, Transforms, VecCache, _Transform},
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
        }
    }

    pub fn instantiate_many(&self, count: i32, chunk: i32, parent: i32) -> EntityParBuilder {
        EntityParBuilder::new(parent, count, chunk, &self)
    }
    pub fn instantiate(&mut self) -> i32 {
        self.instantiate_with_transform(_Transform::default())
    }
    pub fn instantiate_with_transform(&mut self, transform: transform::_Transform) -> i32 {
        self.instantiate_with_transform_with_parent(self.root, transform)
    }
    pub fn instantiate_with_transform_with_parent(
        &mut self,
        parent: i32,
        transform: transform::_Transform,
    ) -> i32 {
        self.transforms.new_transform_with(parent, transform)
    }
    fn copy_game_object_child(&mut self, t: i32, new_parent: i32) -> i32 {
        let tr = self.transforms.get(t).unwrap().get_transform();
        let g = self.instantiate_with_transform_with_parent(new_parent, tr);
        let children =
            if let (Some(src), Some(dest)) = (self.transforms.get(t), self.transforms.get(g)) {
                let src_ent = src.entity();
                let dest_ent = dest.entity();
                for c in src_ent.components.iter() {
                    dest_ent
                        .components
                        .insert(c.0.clone(), self.copy_component_id(&src, c.0.clone(), *c.1));
                }
                let children: Vec<i32> = src.get_children().map(|t| t.id).collect();
                drop(src);
                drop(dest);
                children
            } else {
                Vec::new()
            };
        for c in children {
            self.copy_game_object_child(c, g);
        }
        g
    }
    pub fn copy_game_object(&mut self, t: i32) -> i32 {
        let trans = self.transforms.get(t).unwrap();
        let parent = { trans.get_parent().id };
        drop(trans);
        self.copy_game_object_child(t, parent)
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

    // pub fn add_component<
    //     T: 'static
    //         + Send
    //         + Sync
    //         + Component
    //         + _ComponentID
    //         + Inspectable
    //         + Default
    //         + Clone
    //         + Serialize
    //         + for<'a> Deserialize<'a>,
    // >(
    //     &mut self,
    //     g: i32,
    //     d: T,
    // ) {
    //     // d.assign_transform(g);
    //     let key = T::ID;
    //     if let Some(stor) = self.components.get(&key) {
    //         let stor: &mut Storage<T> = unsafe { std::mem::transmute(&mut stor.write()) };
    //         let c_id = stor.insert(g, d);
    //         let trans = self.transforms.get(g);
    //         stor.init(&trans, c_id, &self.sys);
    //         if let Some(ent) = self.entities.read()[g as usize].lock().as_mut() {
    //             ent.components.insert(key, c_id);
    //         }
    //     } else {
    //         panic!("no type key?")
    //     }
    // }
    pub fn add_component_id(&mut self, g: i32, key: u64, c_id: i32) {
        let trans = self.transforms.get(g).unwrap();
        let ent = trans.entity();
        ent.components.insert(key.clone(), c_id);
        if let Some(stor) = self.components.get(&key) {
            stor.write().init(&trans, c_id, &self.sys);
        }
    }
    pub fn deserialize(&mut self, g: i32, key: String, s: serde_yaml::Value) {
        if let Some(stor) = self.components_names.get(&key) {
            let mut stor = stor.write();
            let c_id = stor.deserialize(g, s);
            let trans = self.transforms.get(g).unwrap();
            stor.init(&trans, c_id, &self.sys);
            let ent = trans.entity();
            ent.components.insert(stor.get_id(), c_id);
        } else {
            panic!("no type key: {}", key);
        }
    }
    pub fn remove_component(&mut self, g: i32, key: u64, c_id: i32) {
        if let Some(stor) = self.components.get(&key) {
            let trans = self.transforms.get(g).unwrap();
            let ent = trans.entity();
            stor.write().deinit(&trans, c_id, &self.sys);
            stor.write().remove(c_id);
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
        let key = T::ID;
        self.components.remove(&key);
        self.components_names
            .remove(std::any::type_name::<T>().split("::").last().unwrap());
    }
    pub fn defer_instantiate(&mut self, perf: &Perf) {
        let mut to_instantiate = self.to_instantiate.lock();
        if to_instantiate.len() == 0 {
            return;
        }
        let v = VecCache::new();
        // to parallelize
        let mut component_transform_offsets: HashMap<
            u64,
            (
                i32,                // offset counter 0
                i32,                // c_id 1
                i32,                // count 2
                ThinVec<i32>,       // 3
                Arc<CacheVec<i32>>, // 4
            ),
        > = self
            .components
            .iter()
            .map(|(id, c)| (*id, (0, 0, 0, ThinVec::new(), Arc::new(v.get_vec(0)))))
            .collect();
        let mut transform_funcs: Vec<(
            i32,
            i32,
            Option<&Box<dyn Fn() -> _Transform + Send + Sync>>,
        )> = Vec::new();
        let mut offset = 0;
        let mut offsets = Vec::new();

        // parallel
        for a in to_instantiate.iter() {
            if let Some(t_func) = &a.t_func {
                transform_funcs.push((a.parent, a.count, Some(t_func)));
            } else {
                transform_funcs.push((a.parent, a.count, None));
            }
            for b in a.comp_funcs.iter() {
                let c = component_transform_offsets.get_mut(&b.0).unwrap();
                c.2 += a.count;
                c.3.push(offset);
            }
            offsets.push(offset);
            offset += a.count;
        }
        let world_instantiate_transforms = perf.node("world instantiate transforms");
        let t = self
            .transforms
            .multi_transform_with(offset as usize, transform_funcs, &offsets);

        drop(world_instantiate_transforms);
        let instantiate_components = perf.node("world instantiate _ components");
        self.components.iter().for_each(|(id, c)| {
            let offsets = component_transform_offsets.get_mut(&id).unwrap();
            offsets.4 = Arc::new(c.write().allocate(offsets.2 as usize));
        });
        let new_transforms = &t.get();
        let _self = &self;
        let mut t_offset = 0;
        rayon::scope(|s| {
            for (i, a) in to_instantiate.iter().enumerate() {
                for b in a.comp_funcs.iter() {
                    let c = component_transform_offsets.get_mut(&b.0).unwrap();
                    let c_offset = c.0;
                    c.0 += a.count;
                    let c_ids = c.4.clone();
                    s.spawn(move |s| {
                        b.1(
                            &_self,
                            &new_transforms,
                            &perf,
                            t_offset,
                            c_offset as usize,
                            &c_ids.get(),
                        );
                    });
                    c.1 += 1;
                }
                t_offset += a.count as usize;
            }
        });
        to_instantiate.clear();
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
    ) {
        let g = _t;
        {
            let children: Vec<i32> = transforms
                .get(_t)
                .unwrap()
                .get_children()
                .map(|t| t.id)
                .collect();
            for t in children {
                Self::__destroy(transforms, components, sys, t);
            }
        }
        {
            let trans = transforms.get(_t).unwrap();
            for (t, id) in trans.entity().components.iter() {
                let stor = &mut components.get(t).unwrap();

                stor.deinit(&trans, *id, &sys);
                stor.remove(*id);
            }
            // remove entity
            // *ent = None;
        }
        // }

        // remove transform
        transforms.remove(_t);
    }
    pub(crate) fn _destroy(&mut self, perf: &Perf) {
        {
            let world_destroy_todestroy = perf.node("world _destroy to_destroy");
            let mut to_destroy = self.to_destroy.lock();

            let mut unlocked: HashMap<
                u64,
                RwLockReadGuard<Box<dyn StorageBase + 'static + Sync + Send>>,
                nohash_hasher::BuildNoHashHasher<i32>,
            > = HashMap::default();
            self.components.iter().for_each(|c| {
                unlocked.insert(*c.0, c.1.read());
            });

            to_destroy.par_iter().for_each(|t| {
                Self::__destroy(&self.transforms, &unlocked, &self.sys, *t);
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
                    d.1._update(&self.transforms.get(id).unwrap());
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

        for i in 0..self.transforms.valid.len() {
            if let Some(trans) = self.transforms.get(i as i32) {
                let ent = trans.entity();
                for (t, id) in ent.components.iter() {
                    let stor = &mut unlocked.get(t).unwrap();
                    stor.deinit(&trans, *id, &self.sys);
                    stor.remove(*id);
                }
                self.transforms.remove(i as i32);
            }
        }

        self.sys.renderer_manager.write().clear();
        self.sys.physics.lock().clear();

        self.transforms.clean();
        self.root = self.transforms.new_root();
        // self.entities.write().push(Mutex::new(None));
    }
}
