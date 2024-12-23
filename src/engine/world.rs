pub mod component;
pub mod entity;
pub mod transform;

use component::__Component;
use crossbeam::queue::SegQueue;
use force_send_sync::SendSync;
use id::ID_trait;
use kira::manager::{backend::DefaultBackend, AudioManager, AudioManagerSettings};
use nalgebra_glm::{quat_euler_angles, Quat, Vec3};
use parking_lot::{MappedRwLockReadGuard, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use rapier3d::prelude::vector;
use rapier3d::prelude::*;
use rayon::{
    prelude::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    },
    Scope,
};
use serde::{Deserialize, Serialize};
use std::{
    cell::{RefCell, SyncUnsafeCell},
    collections::HashMap,
    sync::{
        atomic::{AtomicI32, Ordering},
        Arc,
    },
    time::Instant,
};
use thincollections::{thin_map::ThinMap, thin_vec::ThinVec};

use self::{
    component::{Component, System},
    entity::{
        Entity, EntityBuilder, EntityParBuilder, Unlocked, _EntityBuilder, _EntityParBuilder,
    },
    transform::{CacheVec, Transform, Transforms, VecCache, _Transform, TRANSFORMS, TRANSFORM_MAP},
};

use super::{
    atomic_vec::AtomicVec,
    audio::system::AudioSystem,
    input::Input,
    particles::{component::ParticleEmitter, particles::ParticlesSystem},
    utils::perf::Perf,
    physics::{
        collider::{_Collider, _ColliderType},
        Physics, PhysicsData,
    },
    project::asset_manager::AssetsManager,
    rendering::{
        camera::Camera,
        component::RendererManager,
        lighting::lighting::LightingSystem,
        model::Skeleton,
        vulkan_manager::VulkanManager,
    },
    storage::{Storage, StorageBase, _Storage},
    time::Time,
    utils::GPUWork,
    Defer, RenderData,
};

pub(crate) struct NewRigidBody {
    pub ct: SendSync<*mut _ColliderType>,
    pub pos: Vec3,
    pub rot: Quat,
    pub vel: Vec3,
    pub tid: i32,
    pub rb: SendSync<*mut RigidBodyHandle>,
}
pub struct NewCollider {
    pub ct: SendSync<*mut _ColliderType>,
    pub pos: Vec3,
    pub rot: Quat,
    pub tid: i32,
    pub ch: SendSync<*mut ColliderHandle>,
}

pub struct Sys {
    // pub model_manager: Arc<parking_lot::Mutex<ModelManager>>,
    pub audio_manager: AudioSystem,
    pub renderer_manager: Arc<RwLock<RendererManager>>,
    pub skeletons_manager: Arc<RwLock<HashMap<i32, _Storage<Mutex<Skeleton>>>>>,
    pub assets_manager: Arc<AssetsManager>,
    // physics
    pub physics: Arc<Mutex<Physics>>,
    pub physics2: Arc<Mutex<PhysicsData>>,
    pub mesh_map: Arc<Mutex<HashMap<i32, ColliderBuilder>>>,
    pub proc_mesh_id: AtomicI32,
    pub proc_colliders: Arc<Mutex<HashMap<i32, Arc<Mutex<_Collider>>>>>,
    //
    pub particles_system: Arc<ParticlesSystem>,
    pub lighting_system: Arc<LightingSystem>,
    pub vk: Arc<VulkanManager>,
    pub defer: Defer,
    pub(crate) dragged_transform: i32,
    pub(crate) new_rigid_bodies: SegQueue<NewRigidBody>,
    pub new_colliders: SegQueue<NewCollider>,
    pub(crate) to_remove_rigid_bodies: SegQueue<RigidBodyHandle>,
    pub to_remove_colliders: SegQueue<ColliderHandle>,
}

// impl Sys {

// }

pub struct World {
    pub phys_time: f32,
    pub phys_step: f32,
    pub transforms: Transforms,
    pub mesh_map: HashMap<i32, ColliderBuilder>,
    pub proc_mesh_id: i32,
    pub(crate) transform_map: HashMap<i32, i32>,
    // pub(super) dragged_transform: i32,
    pub(crate) components: HashMap<
        u64,
        (
            AtomicI32,
            Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>,
        ),
        nohash_hasher::BuildNoHashHasher<i32>,
    >,
    pub(crate) component_updates: HashMap<
        u64,
        Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>,
        nohash_hasher::BuildNoHashHasher<i32>,
    >,
    pub(crate) component_late_updates: HashMap<
        u64,
        Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>,
        nohash_hasher::BuildNoHashHasher<i32>,
    >,
    pub(crate) component_editor_updates: HashMap<
        u64,
        Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>,
        nohash_hasher::BuildNoHashHasher<i32>,
    >,
    pub(crate) component_on_render: HashMap<
        u64,
        Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>,
        nohash_hasher::BuildNoHashHasher<i32>,
    >,
    pub(crate) components_names:
        HashMap<String, Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>>,
    pub(crate) root: i32,
    pub sys: Sys,
    pub(crate) to_destroy: Mutex<Vec<i32>>,
    pub(crate) to_instantiate_multi: AtomicVec<_EntityParBuilder>,
    pub(crate) to_instantiate: AtomicVec<_EntityBuilder>,
    pub(crate) to_instantiate_count_trans: AtomicI32,
    v: VecCache<i32>,
    pub(crate) gpu_work: GPUWork,
    pub(crate) input: Input,
    pub(crate) time: Time,
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
pub struct Registation<'a, T> {
    pub(crate) key: u64,
    pub(crate) _phantom: std::marker::PhantomData<T>,
    pub(crate) _world: &'a mut World,
}
impl<'a, T> Registation<'a, T>
where
    T: 'static
        + Send
        + Sync
        + Component
        + ID_trait
        + Default
        + Clone
        + Serialize
        + for<'b> Deserialize<'b>,
{
    pub fn new(world: &'a mut World) -> Self {
        let key = T::ID;
        // world.register::<T>();
        Self {
            key,
            _phantom: std::marker::PhantomData,
            _world: world,
        }
    }
    pub fn update(mut self) -> Self {
        self._world.component_updates.insert(self.key, self._world.components.get(&self.key).unwrap().1.clone());
        self
    }
    pub fn late_update(mut self) -> Self {
        self._world.component_late_updates.insert(self.key, self._world.components.get(&self.key).unwrap().1.clone());
        self
    }
    pub fn on_render(mut self)  -> Self {
        self._world.component_on_render.insert(self.key, self._world.components.get(&self.key).unwrap().1.clone());
        self
    }
    pub fn editor_update(mut self) -> Self {
        self._world.component_editor_updates.insert(self.key, self._world.components.get(&self.key).unwrap().1.clone());
        self
    }
}
#[allow(dead_code)]
impl World {
    pub fn new(
        particles: Arc<ParticlesSystem>,
        lighting: Arc<LightingSystem>,
        renderer_manager: Arc<RwLock<RendererManager>>,
        vk: Arc<VulkanManager>,
        assets_manager: Arc<AssetsManager>,
    ) -> World {
        let mut trans = Transforms::new();
        let root = trans.new_root();

        let mut w = World {
            phys_time: 0f32,
            phys_step: 1. / 30.,
            transforms: trans,
            mesh_map: HashMap::new(),
            proc_mesh_id: -1,
            transform_map: HashMap::new(),
            components: HashMap::default(),
            component_updates: HashMap::default(),
            component_late_updates: HashMap::default(),
            component_editor_updates: HashMap::default(),
            component_on_render: HashMap::default(),
            components_names: HashMap::new(),
            root,
            sys: Sys {
                audio_manager: AudioSystem::new(),
                renderer_manager,
                skeletons_manager: Arc::new(RwLock::new(HashMap::new())),
                assets_manager,
                physics: Arc::new(Mutex::new(Physics::new())),
                physics2: Arc::new(Mutex::new(PhysicsData::new())),
                mesh_map: Default::default(),
                proc_mesh_id: AtomicI32::new(-1),
                proc_colliders: Default::default(),
                particles_system: particles,
                lighting_system: lighting,
                vk: vk,
                defer: Defer::new(),
                dragged_transform: -1,
                new_rigid_bodies: SegQueue::new(),
                new_colliders: SegQueue::new(),
                to_remove_colliders: SegQueue::new(),
                to_remove_rigid_bodies: SegQueue::new(),
            },
            to_destroy: Mutex::new(Vec::new()),
            to_instantiate_multi: AtomicVec::new(),
            to_instantiate: AtomicVec::new(),
            to_instantiate_count_trans: AtomicI32::new(0),
            v: VecCache::new(),
            gpu_work: SegQueue::new(),
            input: Input::default(),
            time: Time::default(),
        };
        // unsafe {
        //     TRANSFORMS = &mut w.transforms;
        //     TRANSFORM_MAP = &mut w.transform_map;
        // }
        w
    }
    pub fn instantiate(&self, parent: i32) -> EntityBuilder {
        EntityBuilder::new(parent, &self)
    }

    pub fn instantiate_many(&self, count: i32, parent: i32) -> EntityParBuilder {
        EntityParBuilder::new(parent, count, 0, &self)
    }
    pub fn create(&mut self) -> i32 {
        self.create_with_transform(_Transform::default())
    }
    pub fn create_with_transform(&mut self, transform: transform::_Transform) -> i32 {
        self.create_with_transform_with_parent(self.root, transform)
    }
    pub fn create_with_transform_with_parent(
        &mut self,
        parent: i32,
        transform: transform::_Transform,
    ) -> i32 {
        self.transforms.new_transform_with(parent, transform)
    }
    fn copy_game_object_child(&mut self, t: i32, new_parent: i32) -> i32 {
        let tr = self.transforms.get(t).unwrap().get_transform();
        let g = self.create_with_transform_with_parent(new_parent, tr);
        let children =
            if let (Some(src), Some(dest)) = (self.transforms.get(t), self.transforms.get(g)) {
                let src_ent = src.entity();
                let dest_ent = dest.entity();
                for (hash, c) in src_ent.components.iter() {
                    // let stor = &mut unlocked.get(t).unwrap();
                    match c {
                        entity::Components::Id(id) => {
                            dest_ent.insert(*hash, self.copy_component_id(&dest, *hash, *id));
                        }
                        entity::Components::V(v) => {
                            for id in v {
                                dest_ent.insert(*hash, self.copy_component_id(&dest, *hash, *id));
                            }
                        }
                    }
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
            let mut stor = stor.1.write();
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
    pub fn get_components<T: 'static + Component + ID_trait, U>(&self, g_id: i32, func: U)
    where
        U: FnOnce(Vec<&Mutex<T>>),
    {
        let Some(ent) = self.transforms.entity.get(g_id as usize) else {
            return;
        };
        let Some(c) = unsafe { &*ent.get() }.components.get(&T::ID) else {
            return;
        };
        let Some(stor) = self.components.get(&T::ID) else {
            return;
        };
        let a = stor.1.read();
        let b = unsafe { a.as_any().downcast_ref_unchecked::<Storage<T>>() };
        let mut _v: Vec<&Mutex<T>> = vec![];
        match c {
            entity::Components::Id(id) => {
                _v.push(&b.data.get(*id as usize).unwrap().1);
            }
            entity::Components::V(v) => {
                for id in v {
                    _v.push(&b.data.get(*id as usize).unwrap().1);
                }
            }
        }
        func(_v);
    }
    pub fn add_component_id(&mut self, g: i32, key: u64, c_id: i32) {
        let trans = self.transforms.get(g).unwrap();
        let ent = trans.entity();
        ent.insert(key.clone(), c_id);
        if let Some(stor) = self.components.get(&key) {
            stor.1.write().init(&trans, c_id, &self.sys);
        }
    }
    pub fn deserialize(&mut self, g: i32, key: &String, s: &serde_yaml::Value) {
        if let Some(stor) = self.components_names.get(key) {
            let mut stor = stor.write();
            let c_id = stor.deserialize(g, s.clone());
            let trans = self.transforms.get(g).unwrap();
            stor.init(&trans, c_id, &self.sys);
            let ent = trans.entity();
            ent.insert(stor.get_id(), c_id);
        } else {
            panic!("no type key: {}", key);
        }
    }
    pub fn remove_component(&mut self, g: i32, key: u64, c_id: i32) {
        if let Some(stor) = self.components.get(&key) {
            let trans = self.transforms.get(g).unwrap();
            let ent = trans.entity();
            stor.1.write().deinit(&trans, c_id, &self.sys);
            stor.1.write().remove(c_id);
            ent.remove(key, c_id);
        }
    }
    pub fn remove_component2(&mut self, g: i32, key: u64) {
        if let Some(stor) = self.components.get(&key) {
            let trans = self.transforms.get(g).unwrap();
            let ent = trans.entity();
            if let Some(c) = ent.components.get(&key) {
                match c {
                    entity::Components::Id(c_id) => {
                        stor.1.write().deinit(&trans, *c_id, &self.sys);
                        stor.1.write().remove(*c_id);
                        ent.remove(key, *c_id);
                    }
                    entity::Components::V(v) => {
                        for c_id in v.clone() {
                            stor.1.write().deinit(&trans, c_id, &self.sys);
                            stor.1.write().remove(c_id);
                            ent.remove(key, c_id);
                        }
                    }
                }
            }
        }
    }
    pub fn register<
        T: 'static
            + Send
            + Sync
            + Component
            + ID_trait
            + Default
            + Clone
            + Serialize
            + for<'a> Deserialize<'a>,
    >(
        &mut self,
    ) -> Registation<T> {
        let key = T::ID;
        let data = Storage::<T>::new();
        let component_storage: Arc<RwLock<Box<dyn StorageBase + Send + Sync + 'static>>> =
            Arc::new(RwLock::new(Box::new(data)));
        self.components
            .insert(key, (AtomicI32::new(0), component_storage.clone()));

        self.components_names.insert(
            component_storage.read().get_name().to_string(),
            component_storage.clone(),
        );
        
        return Registation::new(self);

        // println!("{} registered", std::any::type_name::<T>());
        // // println!("T::update: {}", T::update as usize);
        // // println!("T::editor_update: {}", T::editor_update as usize);
        // // println!("T::late_update: {}", T::late_update as usize);
        // // println!("T::on_render: {}", T::on_render as usize);
        // // check to see if T overrides the default update function
        // if T::update as usize != __Component::update as usize {
        //     println!("{} has update", std::any::type_name::<T>());
        //     self.component_updates
        //         .insert(key, component_storage.clone());
        // }
        // if T::editor_update as usize != __Component::editor_update as usize {
        //     println!("{} has editor update", std::any::type_name::<T>());
        //     self.component_editor_updates
        //         .insert(key, component_storage.clone());
        // }
        // if T::late_update as usize != __Component::late_update as usize {
        //     println!("{} has late update", std::any::type_name::<T>());
        //     self.component_late_updates
        //         .insert(key, component_storage.clone());
        // }
        // if T::on_render as usize != __Component::on_render as usize {
        //     println!("{} has on render", std::any::type_name::<T>());
        //     self.component_on_render
        //         .insert(key, component_storage.clone());
        // }
    }
    pub fn re_init(&mut self) {
        unsafe {
            TRANSFORMS = &mut self.transforms;
            TRANSFORM_MAP = &mut self.transform_map;
        }
        self.sys.physics = Arc::new(Mutex::new(Physics::new()));
    }

    pub fn unregister<
        T: 'static
            + Send
            + Sync
            + Component
            + ID_trait
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
        self.component_updates.remove(&key);
        self.component_late_updates.remove(&key);
        self.component_editor_updates.remove(&key);
        self.component_on_render.remove(&key);
    }
    pub fn init_colls_rbs(&self, perf: &Perf) {
        let mut phys = self.sys.physics.lock();
        while let Some(col) = self.sys.to_remove_colliders.pop() {
            phys.remove_collider(col);
        }
        while let Some(rb) = self.sys.to_remove_rigid_bodies.pop() {
            phys.remove_rigid_body(rb);
        }
        while let Some(new_rb) = self.sys.new_rigid_bodies.pop() {
            let NewRigidBody {
                mut ct,
                pos,
                rot,
                vel,
                tid,
                mut rb,
            } = new_rb;
            if unsafe { **rb != RigidBodyHandle::invalid() } {
                unsafe {
                    phys.remove_rigid_body(unsafe { **rb });
                }
            }
            let _rb = RigidBodyBuilder::dynamic()
                .ccd_enabled(true)
                .translation(pos.into())
                .rotation(quat_euler_angles(&rot))
                .user_data(tid as u128)
                .build();
            unsafe {
                **rb = phys.add_rigid_body(_rb);
                let col = (**ct)
                    .get_collider(&self.sys)
                    .user_data(tid as u128)
                    .build();
                phys.add_collider_to_rigid_body(col, unsafe { **rb });
            }
        }
        while let Some(new_col) = self.sys.new_colliders.pop() {
            let NewCollider {
                mut ct,
                pos,
                rot,
                tid,
                mut ch,
            } = new_col;
            if unsafe { **ch != ColliderHandle::invalid() } {
                unsafe {
                    phys.remove_collider(**ch);
                }
            }
            unsafe {
                let col = (**ct)
                    .get_collider(&self.sys)
                    .user_data(tid as u128)
                    .position(pos.into())
                    .rotation(quat_euler_angles(&rot))
                    .build();
                **ch = phys.add_collider(col);
            }
        }

        // drop(phys);

        phys.dup_query_pipeline(&perf, &mut self.sys.physics2.lock());

        rayon::scope(|s| {
            // TODO: only update if moved
            let update_colliders_rbs = perf.node("update colliders and rbs");
            // update collider positions
            let mut colliders: Vec<&mut Collider> = phys
                .collider_set
                .iter_mut()
                .filter(|a| a.1.parent().is_none())
                .map(|a| a.1)
                .collect();
            colliders.iter_mut().for_each(|col| {
                let i = col.user_data as i32;
                if let Some(t) = self.transforms.get(i) {
                    col.set_translation(t.get_position());
                    col.set_rotation(nalgebra::UnitQuaternion::from_quaternion(t.get_rotation()));
                }
            });
            // update kinematic bodies
            let mut rb: Vec<&mut RigidBody> = phys
                .rigid_body_set
                .iter_mut()
                // .filter(|a| a.1.is_kinematic())
                .map(|a| a.1)
                .collect();
            rb.par_iter_mut().for_each(|rb| {
                let i = rb.user_data as i32;
                if let Some(t) = self.transforms.get(i) {
                    if rb.is_kinematic() {
                        rb.set_translation(t.get_position(), true);
                        rb.set_rotation(
                            nalgebra::UnitQuaternion::from_quaternion(t.get_rotation()),
                            true,
                        );
                    }
                    if rb.is_moving() {
                        // if let Some(t) = self.transforms.get(rb.user_data as i32) {
                        t.set_position(rb.translation());
                        t.set_rotation(rb.rotation());
                        // }
                    }
                }
            });
            // // update positions of rigidbodies
            // phys.island_manager
            //     .active_dynamic_bodies()
            //     .par_iter()
            //     .chain(phys.island_manager.active_kinematic_bodies().par_iter())
            //     .for_each(|a| {
            //         let rb = unsafe { phys.get_rigid_body(*a).unwrap() };
            //         if let Some(t) = self.transforms.get(rb.user_data as i32) {
            //             t.set_position(rb.translation());
            //             t.set_rotation(rb.rotation());
            //         }
            //     });
        });
    }
    pub fn defer_instantiate(&mut self, perf: &Perf) {
        let world_alloc_transforms = perf.node("world allocate transforms");

        let sys = &self.sys;
        let syst = System {
            audio: &sys.audio_manager,
            // skeleton_manager: &sys.skeletons_manager.read(),
            mesh_map: sys.mesh_map.clone(),
            proc_collider: &sys.proc_colliders,
            proc_mesh_id: &sys.proc_mesh_id,
            physics: &sys.physics2.lock(),
            defer: &sys.defer,
            input: &self.input,
            time: &self.time,
            rendering: &sys.renderer_manager,
            assets: &sys.assets_manager,
            vk: sys.vk.clone(),
            gpu_work: &self.gpu_work,
            particle_system: &sys.particles_system,
            new_rigid_bodies: &sys.new_rigid_bodies,
        };

        let mut trans_count = self.to_instantiate_count_trans.as_ptr();
        let t = self
            .transforms
            ._allocate(self.to_instantiate.len() + unsafe { *trans_count } as usize);
        unsafe { *trans_count = 0 };
        drop(world_alloc_transforms);

        let alloc_components = perf.node("world allocate _ components");

        let mut comp_ids: HashMap<
            u64,
            SyncUnsafeCell<CacheVec<i32>>,
            nohash_hasher::BuildNoHashHasher<u64>,
        > = self
            .components
            .iter()
            .map(|(id, c)| (*id, unsafe { SyncUnsafeCell::new(self.v.get_vec(0)) }))
            .collect();
        self.components.par_iter().for_each(|(id, c)| {
            let ids = comp_ids.get(&id).unwrap();
            unsafe {
                *ids.get() = c.1.write().allocate(unsafe { *c.0.as_ptr() } as usize);
                *c.0.as_ptr() = 0;
            }
        });
        drop(alloc_components);
        let mut unlocked_ = unsafe { SendSync::new(ThinMap::new()) };
        self.components.iter().for_each(|(id, c)| {
            unlocked_.insert(*id, (AtomicI32::new(0), c.1.read()));
        });
        {
            let mut unlocked = unsafe { SendSync::new(ThinMap::new()) };
            for u in unlocked_.iter() {
                unlocked.insert(u.0.clone(), (&u.1 .0, u.1 .1.as_ref()));
            }

            self._defer_instantiate_single(perf, &t, &comp_ids, &unlocked, &syst);
            self._defer_instantiate_multi(perf, &t, &comp_ids, &unlocked, &syst);
        }
        self.to_instantiate_count_trans.store(0, Ordering::Relaxed);
        self.to_instantiate.clear();
        self.to_instantiate_multi.clear();
        // drop(syst);
    }

    pub fn _defer_instantiate_single(
        &self,
        perf: &Perf,
        // syst: &System,
        trans: &CacheVec<i32>,
        comp_ids: &HashMap<
            u64,
            SyncUnsafeCell<CacheVec<i32>>,
            nohash_hasher::BuildNoHashHasher<u64>,
        >,
        unlocked: &SendSync<ThinMap<u64, (&AtomicI32, &(dyn StorageBase + Send + Sync))>>,
        sys: &System,
    ) {
        if self.to_instantiate.len() == 0 {
            return;
        }

        let _toi = self.to_instantiate.get();
        let to_instantiate = unsafe { force_send_sync::SendSync::new(&_toi) };
        let t_default: Box<dyn Fn() -> _Transform + std::marker::Send + std::marker::Sync> =
            Box::new(|| _Transform::default());
        (0..self.to_instantiate.len())
            .into_par_iter()
            .for_each(|i| {
                let a = unsafe { &to_instantiate[i].assume_init_ref() };
                let t_id = i;
                let t = trans.get()[t_id as usize];
                let t_func = a.t_func.as_ref().unwrap_or(&t_default);
                let trans = self.transforms.write_transform(t, t_func());
                for b in &a.comp_funcs {
                    let comp = &unlocked.get_mut(&b.key).unwrap();
                    let c_id = comp.0.fetch_add(1, Ordering::Relaxed);
                    (b.comp_func)(
                        &self,
                        comp.1,
                        (unsafe { &*comp_ids.get(&b.key).unwrap().get() }).get()[c_id as usize],
                        &trans,
                    );
                }
                let ent = trans.entity();
                for c in ent.components.iter_mut() {
                    let comp = &unlocked.get_mut(c.0).unwrap();
                    match c.1 {
                        entity::Components::Id(id) => {
                            comp.1.on_start(&trans, *id, sys);
                        }
                        entity::Components::V(v) => {
                            for id in v {
                                comp.1.on_start(&trans, *id, sys);
                            }
                        }
                    }
                }
            });
        self.to_instantiate_count_trans
            .store(self.to_instantiate.len() as i32, Ordering::SeqCst);
    }

    pub fn _defer_instantiate_multi(
        &self,
        perf: &Perf,
        // syst: &System,
        trans: &CacheVec<i32>,
        comp_ids: &HashMap<
            u64,
            SyncUnsafeCell<CacheVec<i32>>,
            nohash_hasher::BuildNoHashHasher<u64>,
        >,
        unlocked: &SendSync<ThinMap<u64, (&AtomicI32, &(dyn StorageBase + Send + Sync))>>,
        sys: &System,
    ) {
        if self.to_instantiate_multi.len() == 0 {
            return;
        }
        let calc_offsets = perf.node("world calc instantiate multi offsets");
        let mut t_offset = self.to_instantiate_count_trans.load(Ordering::SeqCst);
        self.to_instantiate_multi.get().iter().for_each(|a| {
            let a = unsafe { a.assume_init_ref() };
            unsafe {
                *a.t_func.0.get() = t_offset;
            }
            t_offset += a.count;
            for b in &a.comp_funcs {
                let comp = &unlocked.get_mut(&b.key).unwrap();
                unsafe {
                    *b.offset.get() = *comp.0.as_ptr();
                    *comp.0.as_ptr() += a.count
                };
            }
        });
        drop(calc_offsets);
        let _toi = self.to_instantiate_multi.get();
        let to_instantiate = unsafe { force_send_sync::SendSync::new(&_toi) };
        let _trans = trans.get();
        let t_default: Box<dyn Fn() -> _Transform + std::marker::Send + std::marker::Sync> =
            Box::new(|| _Transform::default());
        (0..self.to_instantiate_multi.len())
            .into_par_iter()
            .for_each(|i| {
                let a = unsafe { &to_instantiate[i].assume_init_ref() };
                let t_func = a.t_func.1.as_ref().unwrap_or(&t_default);
                let t_id = unsafe { *a.t_func.0.get() };
                (0..a.count).into_par_iter().for_each(|i| {
                    let t = _trans[(t_id + i) as usize];
                    let trans = self.transforms.write_transform(t, t_func());
                    for b in a.comp_funcs.iter() {
                        let comp = &unlocked.get(&b.key).unwrap();
                        let c_id = unsafe { *b.offset.get() };
                        let stor = comp.1;
                        let cto = unsafe { (*comp_ids.get(&b.key).unwrap().get()).get() };
                        (b.comp_func)(&self, stor, cto[(c_id + i) as usize], &trans);
                    }
                    let ent = trans.entity();
                    for c in ent.components.iter_mut() {
                        let comp = &unlocked.get_mut(c.0).unwrap();
                        match c.1 {
                            entity::Components::Id(id) => {
                                comp.1.on_start(&trans, *id, sys);
                            }
                            entity::Components::V(v) => {
                                for id in v {
                                    comp.1.on_start(&trans, *id, sys);
                                }
                            }
                        }
                    }
                });
            });
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
        let children = if let Some(trans) = transforms.get(_t) {
            // should always be valid unless destroy is called twice
            trans.get_children().map(|t| t.id).collect()
        } else {
            println!("failed to get transform {} for deallocation", _t);
            vec![]
        };
        for t in children {
            Self::__destroy(transforms, components, sys, t);
        }
        if let Some(trans) = transforms.get(_t) {
            // let trans = transforms.get(_t).unwrap();
            for (t, c) in trans.entity().components.iter() {
                let stor = &mut components.get(t).unwrap();
                match c {
                    entity::Components::Id(id) => {
                        stor.deinit(&trans, *id, &sys);
                        stor.remove(*id);
                    }
                    entity::Components::V(v) => {
                        for id in v {
                            stor.deinit(&trans, *id, &sys);
                            stor.remove(*id);
                        }
                    }
                }
            }
            // remove entity
            // *ent = None;
        }

        // }

        // remove transform
        transforms.remove(_t);
    }
    pub fn _destroy(&mut self, perf: &Perf) {
        // {
        let world_destroy_todestroy = perf.node("world _destroy to_destroy");
        let mut to_destroy = self.to_destroy.lock();

        let mut unlocked: HashMap<
            u64,
            RwLockReadGuard<Box<dyn StorageBase + 'static + Sync + Send>>,
            nohash_hasher::BuildNoHashHasher<i32>,
        > = HashMap::default();
        self.components.iter().for_each(|c| {
            unlocked.insert(*c.0, c.1 .1.read());
        });

        to_destroy.par_iter().for_each(|t| {
            Self::__destroy(&self.transforms, &unlocked, &self.sys, *t);
        });
        drop(world_destroy_todestroy);
        drop(unlocked);
        // }
        let world_destroy_reduce = perf.node("world _destroy reduce");
        if to_destroy.len() < 1000 {
            self.transforms.reduce_last();
            self.components.iter().for_each(|(id, c)| {
                c.1.write().reduce_last();
            });
        } else {
            rayon::scope(|s| {
                s.spawn(|s| {
                    self.transforms.reduce_last();
                });
                self.components.par_iter().for_each(|(id, c)| {
                    c.1.write().reduce_last();
                });
            });
        }
        to_destroy.clear();
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
    pub fn get_component_storage<T: 'static + Send + Sync + Component + ID_trait, U, I>(
        &self,
        func: U,
    ) -> I
    where
        U: FnOnce(&Storage<T>) -> I,
    {
        let key = T::ID;

        let stor = self.components.get(&key).unwrap().1.write();
        let _stor = unsafe { stor.as_any().downcast_ref_unchecked::<Storage<T>>() };
        func(&_stor)

        // let key = T::ID;
        // self.components.get(&key)
    }

    pub fn do_defered(&mut self) {
        while let Some(w) = self.sys.defer.work.pop() {
            w(self);
        }
    }
    pub(crate) fn begin_frame(&mut self, input: Input, time: Time) {
        self.gpu_work = SegQueue::new();
        self.input = input;
        self.time = time;
    }
    pub fn _update(&mut self, perf: &Perf) {
        {
            let sys = &self.sys;
            let sys = System {
                audio: &sys.audio_manager,
                // skeleton_manager: &sys.skeletons_manager.read(),
                mesh_map: sys.mesh_map.clone(),
                proc_collider: &sys.proc_colliders,
                proc_mesh_id: &sys.proc_mesh_id,
                physics: &sys.physics2.lock(),
                defer: &sys.defer,
                input: &self.input,
                time: &self.time,
                rendering: &sys.renderer_manager,
                assets: &sys.assets_manager,
                vk: sys.vk.clone(),
                gpu_work: &self.gpu_work,
                particle_system: &sys.particles_system,
                new_rigid_bodies: &sys.new_rigid_bodies,
            };
            {
                let world_update = perf.node("world update");
                self.component_updates.iter().for_each(|(_, stor)| {
                    let mut stor = stor.read();
                    if stor.len() == 0 {
                        return;
                    }
                    let world_update = perf.node(&format!("update-{}", stor.get_name()));
                    stor.update(&self.transforms, &sys, &self);
                });
            }
            {
                let world_update = perf.node("world late_update");
                self.component_late_updates.iter().for_each(|(_, stor)| {
                    let mut stor = stor.read();
                    if stor.len() == 0 {
                        return;
                    }
                    let world_update =
                        perf.node(&format!("late_update-{}", stor.get_name()));
                    stor.late_update(&self.transforms, &sys);
                });
            }
        }
        // self.update_cameras();
    }
    pub fn update_cameras(&mut self) {
        // let mut ret = Vec::new();
        self.get_component_storage::<Camera, _, _>(|camera_storage| {
            camera_storage.for_each(|t_id, cam| {
                cam._update(&self.transforms.get(t_id).unwrap());
                // ret.push(cvd)
            })
        });
        // ret
    }
    pub(crate) fn editor_update(&mut self) {
        let sys = &self.sys;
        let sys = System {
            audio: &sys.audio_manager,
            // skeleton_manager: &sys.skeletons_manager.read(),
            mesh_map: sys.mesh_map.clone(),
            proc_collider: &sys.proc_colliders,
            proc_mesh_id: &sys.proc_mesh_id,
            physics: &sys.physics2.lock(),
            defer: &sys.defer,
            input: &self.input,
            time: &self.time,
            rendering: &sys.renderer_manager,
            assets: &sys.assets_manager,
            vk: sys.vk.clone(),
            gpu_work: &self.gpu_work,
            particle_system: &sys.particles_system,
            new_rigid_bodies: &sys.new_rigid_bodies,
        };
        for (_, stor) in self.component_editor_updates.iter() {
            stor.write()
                .editor_update(&self.transforms, &sys, &self.input);
        }
    }
    pub fn render(&self, rd: &mut RenderData) {
        // let mut render_jobs = vec![];
        for (_, stor) in self.component_on_render.iter() {
            stor.write().on_render(rd);
        }
        // render_jobs
    }
    // pub(crate) fn get_cam_datas(&mut self) -> (i32, Vec<Arc<Mutex<CameraData>>>) {
    //     self.get_component_storage::<Camera, _, _>(|camera_storage| {
    //         let mut main_cam_id = -1;
    //         let mut cam_datas = Vec::new();
    //         camera_storage.for_each(|t_id, cam| {
    //             main_cam_id = t_id;
    //             if let Some(data) = cam.get_data() {
    //                 cam_datas.push(data);
    //             }
    //         });
    //         (main_cam_id, cam_datas)
    //     })
    // }
    pub(crate) fn get_emitter_len(&self) -> usize {
        self.get_component_storage::<ParticleEmitter, _, _>(|x| x.len())
    }
    // pub fn regen(&mut self) {
    //     *self = Self::new(
    //         self.sys.particles_system.clone(),
    //         self.sys.lighting_system.clone(),
    //         self.sys.renderer_manager.clone(),
    //         self.sys.vk.clone(),
    //         self.sys.assets_manager.clone(),
    //     );
    // }
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
            unlocked.insert(*c.0, c.1 .1.read());
        });
        // destroy entities not part of hierarchy
        for i in 0..self.transforms.valid.len() {
            if let Some(trans) = self.transforms.get(i as i32) {
                let ent = trans.entity();

                for (t, c) in ent.components.iter() {
                    let stor = &mut unlocked.get(t).unwrap();
                    match c {
                        entity::Components::Id(id) => {
                            stor.deinit(&trans, *id, &self.sys);
                            stor.remove(*id);
                        }
                        entity::Components::V(v) => {
                            for id in v {
                                stor.deinit(&trans, *id, &self.sys);
                                stor.remove(*id);
                            }
                        }
                    }
                }
                self.transforms.remove(i as i32);
            }
        }

        self.sys.renderer_manager.write().clear();
        *self.sys.physics.lock() = Physics::new();
        *self.sys.physics2.lock() = PhysicsData::new();
        self.sys.to_remove_colliders = SegQueue::new();
        self.sys.to_remove_rigid_bodies = SegQueue::new();
        self.sys.new_colliders = SegQueue::new();
        self.sys.new_rigid_bodies = SegQueue::new();
        *self.sys.proc_mesh_id.get_mut() = -1;
        self.sys.proc_colliders.lock().clear();

        self.transforms.clean();
        self.root = self.transforms.new_root();
        unsafe {
            self.mesh_map.clear();
            self.proc_mesh_id = -1;
        }
        // self.entities.write().push(Mutex::new(None));
    }

    pub(crate) fn begin_play(&self) {
        let sys = &self.sys;
        let sys = System {
            audio: &sys.audio_manager,
            // skeleton_manager: &sys.skeletons_manager.read(),
            mesh_map: sys.mesh_map.clone(),
            proc_collider: &sys.proc_colliders,
            proc_mesh_id: &sys.proc_mesh_id,
            physics: &sys.physics2.lock(),
            defer: &sys.defer,
            input: &self.input,
            time: &self.time,
            rendering: &sys.renderer_manager,
            assets: &sys.assets_manager,
            vk: sys.vk.clone(),
            gpu_work: &self.gpu_work,
            particle_system: &sys.particles_system,
            new_rigid_bodies: &sys.new_rigid_bodies,
        };

        let mut unlocked: HashMap<
            u64,
            RwLockReadGuard<Box<dyn StorageBase + 'static + Sync + Send>>,
            nohash_hasher::BuildNoHashHasher<i32>,
        > = HashMap::default();
        self.components.iter().for_each(|c| {
            unlocked.insert(*c.0, c.1 .1.read());
        });
        // destroy entities not part of hierarchy
        (0..self.transforms.valid.len())
            .into_par_iter()
            .for_each(|i| {
                if let Some(trans) = self.transforms.get(i as i32) {
                    let ent = trans.entity();

                    for (t, c) in ent.components.iter() {
                        let stor = &mut unlocked.get(t).unwrap();
                        match c {
                            entity::Components::Id(id) => {
                                stor.on_start(&trans, *id, &sys);
                                // stor.deinit(&trans, *id, &self.sys);
                                // stor.remove(*id);
                            }
                            entity::Components::V(v) => {
                                for id in v {
                                    stor.on_start(&trans, *id, &sys);
                                    // stor.deinit(&trans, *id, &self.sys);
                                    // stor.remove(*id);
                                }
                            }
                        }
                    }
                    // self.transforms.remove(i as i32);
                }
            });
    }
}
