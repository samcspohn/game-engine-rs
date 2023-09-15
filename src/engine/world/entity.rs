use std::{
    cell::SyncUnsafeCell,
    collections::HashMap,
    sync::atomic::{AtomicI32, Ordering},
    time::Instant,
};

use force_send_sync::SendSync;
use once_cell::sync::Lazy;
use parking_lot::{RawRwLock, RwLockReadGuard};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use thincollections::{thin_map::ThinMap, thin_vec::ThinVec};

use crate::{
    editor::inspectable::Inspectable,
    engine::{
        perf::Perf,
        storage::{Storage, StorageBase},
    },
};

use super::{
    component::{Component, _ComponentID},
    transform::{CacheVec, Transforms},
    {transform::_Transform, World},
};

pub enum Components {
    Id(i32),
    V(ThinVec<i32>),
}
pub struct Entity {
    pub components: SendSync<ThinMap<u64, Components>>,
}
impl Entity {
    pub fn new() -> Self {
        Self {
            components: unsafe { SendSync::new(ThinMap::with_capacity(2)) },
        }
    }
    pub fn insert(&mut self, hash: u64, id: i32) {
        match self.components.entry(hash) {
            thincollections::thin_map::Entry::Occupied(mut a) => {
                let b = a.get_mut();
                match b {
                    Components::Id(c) => {
                        let mut d = ThinVec::new();
                        d.push(*c);
                        d.push(id);
                        *b = Components::V(d);
                    }
                    Components::V(v) => v.push(id),
                }
            }
            thincollections::thin_map::Entry::Vacant(_) => {
                self.components.insert(hash, Components::Id(id));
            }
        }
    }
    pub fn remove(&mut self, hash: u64, id: i32) {
        match self.components.entry(hash) {
            thincollections::thin_map::Entry::Occupied(mut a) => {
                let b = a.get_mut();
                match b {
                    Components::Id(c) => {
                        assert!(*c == id);
                        self.components.remove(&hash);
                        // let mut d = ThinVec::new();
                        // d.push(*c);
                        // d.push(id);
                        // *b = Components::V(d);
                    }
                    Components::V(v) => {
                        if let Some((i, _)) = v.iter().enumerate().find(|(i, _id)| **_id == id) {
                            v.remove(i);
                        }
                    }
                }
            }
            thincollections::thin_map::Entry::Vacant(_) => {}
        }
    }
}

type CompFunc = (
    u64,
    Box<dyn Fn(&World, &(dyn StorageBase + Send + Sync), i32, i32) + Send + Sync>,
);

pub(crate) struct _EntityBuilder {
    pub(crate) parent: i32,
    pub(crate) t_func: Option<Box<dyn Fn() -> _Transform + Send + Sync>>,
    pub(crate) comp_funcs: ThinVec<CompFunc>,
}
unsafe impl Send for _EntityBuilder {}
unsafe impl Sync for _EntityBuilder {}

impl _EntityBuilder {
    fn from(g: EntityBuilder) -> Self {
        Self {
            parent: g.parent,
            t_func: g.transform_func,
            comp_funcs: g.comp_funcs,
        }
    }
}
pub struct EntityBuilder<'a> {
    world: &'a World,
    parent: i32,
    transform_func: Option<Box<dyn Fn() -> _Transform + Send + Sync>>,
    comp_funcs: ThinVec<CompFunc>,
}
impl<'a> EntityBuilder<'a> {
    pub fn new(parent: i32, world: &'a World) -> Self {
        EntityBuilder {
            world,
            parent: parent,
            comp_funcs: ThinVec::new(),
            transform_func: None,
        }
    }
    pub fn with_transform<D: 'static>(mut self, f: D) -> Self
    where
        D: Fn() -> _Transform + Send + Sync,
    {
        self.transform_func = Some(Box::new(f));
        self
    }
    pub fn with_com<
        D,
        T: 'static
            + Send
            + Sync
            + Component
            + _ComponentID
            + Inspectable
            + Default
            + Clone
            + Serialize
            + for<'b> Deserialize<'b>,
    >(
        mut self,
        f: D,
    ) -> Self
    where
        D: Fn() -> T + 'static + Send + Sync,
    {
        self.world
            .components
            .get(&T::ID)
            .unwrap()
            .0
            .fetch_add(1, Ordering::Relaxed);
        self.comp_funcs.push((
            T::ID,
            Box::new(
                move |world: &World, stor: &(dyn StorageBase + Send + Sync), id: i32, t: i32| {
                    let stor: &Storage<T> = unsafe { stor.as_any().downcast_ref_unchecked() };
                    stor.insert_exact(id, t, &world.transforms, &world.sys, &f);
                },
            ),
        ));
        self
    }
    pub fn build(mut self) {
        // self.world.to_instantiate_count_trans.fetch_add(1, Ordering::Relaxed);
        self.world.to_instantiate.push(_EntityBuilder::from(self));
    }
}
pub type Unlocked<'a> = force_send_sync::SendSync<
    ThinMap<u64, RwLockReadGuard<'a, Box<dyn StorageBase + Send + Sync>>>,
>;
type CompFuncMulti = (
    u64,
    SyncUnsafeCell<i32>,
    Box<dyn Fn(&World, &(dyn StorageBase + Send + Sync), i32, i32) + Send + Sync>,
);

pub(crate) struct _EntityParBuilder {
    pub(crate) count: i32,
    // pub(crate) chunk: i32,
    pub(crate) parent: i32,
    pub(crate) t_func: (
        SyncUnsafeCell<i32>,
        Option<Box<dyn Fn() -> _Transform + Send + Sync>>,
    ),
    pub(crate) comp_funcs: Vec<CompFuncMulti>,
    // pub(crate) instantiate_func: &'static (dyn Fn(
    //     &_EntityParBuilder,
    //     &Vec<i32>,
    //     &HashMap<u64, SyncUnsafeCell<CacheVec<i32>>, nohash_hasher::BuildNoHashHasher<u64>>,
    //     &World,
    //     &SendSync<
    //         ThinMap<
    //             u64,
    //             (
    //                 AtomicI32,
    //                 RwLockReadGuard<'_, Box<dyn StorageBase + Send + Sync>>,
    //             ),
    //         >,
    //     >,
    // ) + Send
    //               + Sync), // pub(crate) c_offsets: ThinMap<u64, i32>,
}

unsafe impl Send for _EntityParBuilder {}
unsafe impl Sync for _EntityParBuilder {}

impl _EntityParBuilder {
    fn from(e: EntityParBuilder) -> Self {
        Self {
            count: e.count,
            parent: e.parent,
            t_func: e.transform_func,
            comp_funcs: e.comp_funcs,
            // instantiate_func: if e.count > 16 {
            //     &high_count
            // } else {
            //     &low_count
            // },
        }
    }
}

// static T_DEFAULT: Box<dyn Fn() -> _Transform + Send + Sync> = Box::new(|| _Transform::default());
static T_DEFAULT: Lazy<Box<dyn Fn() -> _Transform + Send + Sync>> =
    Lazy::new(|| Box::new(|| _Transform::default()));

fn low_count(
    a: &_EntityParBuilder,
    _trans: &Vec<i32>,
    comp_ids: &HashMap<u64, SyncUnsafeCell<CacheVec<i32>>, nohash_hasher::BuildNoHashHasher<u64>>,
    world: &World,
    unlocked: &SendSync<
        ThinMap<
            u64,
            (
                AtomicI32,
                RwLockReadGuard<'_, Box<dyn StorageBase + Send + Sync>>,
            ),
        >,
    >,
) {
    let t_func = a.t_func.1.as_ref().unwrap_or(&T_DEFAULT);
    let t_id = unsafe { *a.t_func.0.get() };
    (0..a.count).into_iter().for_each(|i| {
        let t = _trans[(t_id + i) as usize];
        world.transforms.write_transform(t, t_func());
    });
    a.comp_funcs.iter().for_each(|b| {
        let comp = &unlocked.get(&b.0).unwrap();
        let c_id = unsafe { *b.1.get() };
        let stor = comp.1.as_ref();
        let cto = unsafe { (*comp_ids.get(&b.0).unwrap().get()).get() };
        (0..a.count).into_iter().for_each(|i| {
            let t = _trans[(t_id + i) as usize];
            b.2(&world, stor, cto[(c_id + i) as usize], t);
        });
    });
}
fn high_count(
    a: &_EntityParBuilder,
    _trans: &Vec<i32>,
    comp_ids: &HashMap<u64, SyncUnsafeCell<CacheVec<i32>>, nohash_hasher::BuildNoHashHasher<u64>>,
    world: &World,
    unlocked: &SendSync<
        ThinMap<
            u64,
            (
                AtomicI32,
                RwLockReadGuard<'_, Box<dyn StorageBase + Send + Sync>>,
            ),
        >,
    >,
) {
    let t_func = a.t_func.1.as_ref().unwrap_or(&T_DEFAULT);
    let t_id = unsafe { *a.t_func.0.get() };
    (0..a.count).into_par_iter().for_each(|i| {
        let t = _trans[(t_id + i) as usize];
        world.transforms.write_transform(t, t_func());
    });
    a.comp_funcs.par_iter().for_each(|b| {
        let comp = &unlocked.get(&b.0).unwrap();
        let c_id = unsafe { *b.1.get() };
        let stor = comp.1.as_ref();
        let cto = unsafe { (*comp_ids.get(&b.0).unwrap().get()).get() };
        (0..a.count).into_par_iter().for_each(|i| {
            let t = _trans[(t_id + i) as usize];
            b.2(world, stor, cto[(c_id + i) as usize], t);
        });
    });
}

pub struct EntityParBuilder<'a> {
    world: &'a World,
    count: i32,
    // chunk: i32,
    parent: i32,
    transform_func: (
        SyncUnsafeCell<i32>,
        Option<Box<dyn Fn() -> _Transform + Send + Sync>>,
    ),
    comp_funcs: Vec<CompFuncMulti>,
}
impl<'a> EntityParBuilder<'a> {
    pub fn new(parent: i32, count: i32, chunk: i32, world: &'a World) -> Self {
        EntityParBuilder {
            world,
            count,
            // chunk,
            parent: parent,
            comp_funcs: Vec::new(),
            transform_func: (SyncUnsafeCell::new(-1), None),
        }
    }
    pub fn with_transform<D: 'static>(mut self, f: D) -> Self
    where
        D: Fn() -> _Transform + Send + Sync,
    {
        self.transform_func = (SyncUnsafeCell::new(-1), Some(Box::new(f)));
        self
    }
    pub fn with_com<
        D,
        T: 'static
            + Send
            + Sync
            + Component
            + _ComponentID
            + Inspectable
            + Default
            + Clone
            + Serialize
            + for<'b> Deserialize<'b>,
    >(
        mut self,
        f: D,
    ) -> Self
    where
        D: Fn() -> T + 'static + Send + Sync,
    {
        self.world
            .components
            .get(&T::ID)
            .unwrap()
            .0
            .fetch_add(self.count, Ordering::Relaxed);
        self.comp_funcs.push((
            T::ID,
            SyncUnsafeCell::new(-1),
            Box::new(
                move |world: &World, stor: &(dyn StorageBase + Send + Sync), id: i32, t: i32| {
                    let stor: &Storage<T> = unsafe { stor.as_any().downcast_ref_unchecked() };
                    stor.insert_exact(id, t, &world.transforms, &world.sys, &f);
                },
            ),
        ));
        // self.comp_funcs.push((
        //     T::ID,
        //     Box::new(
        //         move |unlocked: &Unlocked,
        //               world: &World,
        //               new_transforms: &Vec<i32>,
        //               perf: &Perf,
        //               t_offset: usize,
        //               c_offset: usize,
        //               c_ids: &[i32]| {
        //             let key = T::ID;
        //             if let Some(stor) = unlocked.get(&key) {
        //                 // let mut stor_lock = stor.read();
        //                 let stor: &Storage<T> = unsafe { stor.as_any().downcast_ref_unchecked() };
        //                 let world_instantiate =
        //                     perf.node(&format!("world instantiate {}", stor.get_name()));
        //                 // assert!(self.count as usize == t_.len());
        //                 stor.insert_multi(
        //                     self.count as usize,
        //                     &world.transforms,
        //                     &world.sys,
        //                     &new_transforms,
        //                     t_offset,
        //                     perf,
        //                     c_offset,
        //                     c_ids,
        //                     &f,
        //                 );
        //             } else {
        //                 panic!("no type key?")
        //             }
        //         },
        //     ),
        // ));
        self
    }
    pub fn build(mut self) {
        self.world
            .to_instantiate_count_trans
            .fetch_add(self.count, Ordering::Relaxed);
        self.world
            .to_instantiate_multi
            .push(_EntityParBuilder::from(self));
    }
}
