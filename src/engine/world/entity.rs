use std::time::Instant;

use force_send_sync::SendSync;
use parking_lot::{lock_api::RwLockReadGuard, RawRwLock};
use serde::{Deserialize, Serialize};
use thincollections::thin_map::ThinMap;

use crate::{
    editor::inspectable::Inspectable,
    engine::{
        perf::Perf,
        storage::{Storage, StorageBase},
    },
};

use super::{
    component::{Component, _ComponentID},
    transform::CacheVec,
    {transform::_Transform, World},
};

pub struct Entity {
    pub components: SendSync<ThinMap<u64, i32>>,
}
impl Entity {
    pub fn new() -> Self {
        Self {
            components: unsafe { SendSync::new(ThinMap::with_capacity(2)) },
        }
    }
}

type CompFunc = (
    u64,
    Box<
        dyn Fn(
            &World,
            &RwLockReadGuard<'_, RawRwLock, Box<dyn StorageBase + Send + Sync>>,
            i32,
            i32
        ),
    >,
);
pub(crate) struct _EntityBuilder {
    pub(crate) parent: i32,
    pub(crate) t_func: Option<Box<dyn Fn() -> _Transform + Send + Sync>>,
    pub(crate) comp_funcs: Vec<CompFunc>,
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
    comp_funcs: Vec<CompFunc>,
}
impl<'a> EntityBuilder<'a> {
    pub fn new(parent: i32, world: &'a World) -> Self {
        EntityBuilder {
            world,
            parent: parent,
            comp_funcs: Vec::new(),
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
        self.comp_funcs.push((
            T::ID,
            Box::new(
                move |world: &World,
                 stor: &RwLockReadGuard<'_, RawRwLock, Box<dyn StorageBase + Send + Sync>>,
                 id: i32,
                 t: i32| {
                    let stor: &Storage<T> = unsafe { stor.as_any().downcast_ref_unchecked() };
                    stor.insert_exact(id, t, &world.transforms, &world.sys, &f);
                },
            ),
        ));
        self
    }
    pub fn build(mut self) {
        self.world
            .to_instantiate
            .lock()
            .push(_EntityBuilder::from(self));
    }
}
pub type Unlocked<'a> = force_send_sync::SendSync<
    ThinMap<u64, RwLockReadGuard<'a, RawRwLock, Box<dyn StorageBase + Send + Sync>>>,
>;
type CompFuncMulti = (
    u64,
    Box<dyn Fn(&Unlocked, &World, &Vec<i32>, &Perf, usize, usize, &[i32]) + Send + Sync>,
);
pub(crate) struct _EntityParBuilder {
    pub(crate) count: i32,
    pub(crate) chunk: i32,
    pub(crate) parent: i32,
    pub(crate) t_func: Option<Box<dyn Fn() -> _Transform + Send + Sync>>,
    pub(crate) comp_funcs: Vec<CompFuncMulti>,
}

impl _EntityParBuilder {
    fn from(g: EntityParBuilder) -> Self {
        Self {
            count: g.count,
            chunk: g.chunk,
            parent: g.parent,
            t_func: g.transform_func,
            comp_funcs: g.comp_funcs,
        }
    }
}
pub struct EntityParBuilder<'a> {
    world: &'a World,
    count: i32,
    chunk: i32,
    parent: i32,
    transform_func: Option<Box<dyn Fn() -> _Transform + Send + Sync>>,
    comp_funcs: Vec<CompFuncMulti>,
}
impl<'a> EntityParBuilder<'a> {
    pub fn new(parent: i32, count: i32, chunk: i32, world: &'a World) -> Self {
        EntityParBuilder {
            world,
            count,
            chunk,
            parent: parent,
            comp_funcs: Vec::new(),
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
        self.comp_funcs.push((
            T::ID,
            Box::new(
                move |unlocked: &Unlocked,
                      world: &World,
                      new_transforms: &Vec<i32>,
                      perf: &Perf,
                      t_offset: usize,
                      c_offset: usize,
                      c_ids: &[i32]| {
                    let key = T::ID;
                    if let Some(stor) = unlocked.get(&key) {
                        // let mut stor_lock = stor.read();
                        let stor: &Storage<T> = unsafe { stor.as_any().downcast_ref_unchecked() };
                        let world_instantiate =
                            perf.node(&format!("world instantiate {}", stor.get_name()));
                        // assert!(self.count as usize == t_.len());
                        stor.insert_multi(
                            self.count as usize,
                            &world.transforms,
                            &world.sys,
                            &new_transforms,
                            t_offset,
                            perf,
                            c_offset,
                            c_ids,
                            &f,
                        );
                    } else {
                        panic!("no type key?")
                    }
                },
            ),
        ));
        self
    }
    pub fn build(mut self) {
        self.world
            .to_instantiate_multi
            .lock()
            .push(_EntityParBuilder::from(self));
    }
}
