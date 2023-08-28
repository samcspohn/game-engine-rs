use std::{
    any::{Any, TypeId},
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap},
    sync::atomic::{AtomicBool, AtomicI32, Ordering}, cell::SyncUnsafeCell,
};

use bitvec::vec::BitVec;
use crossbeam::queue::SegQueue;
use parking_lot::Mutex;
use rayon::prelude::*;
use segvec::SegVec;
use serde::{Deserialize, Serialize};
use crate::editor::inspectable::Inspectable;

use super::{
    input::Input,
    particles::particles::AtomicVec,
    world::{
        component::{Component, System, _ComponentID},
        entity::Entity,
        transform::{Transform, Transforms, VecCache},
        Sys, World,
    },
    RenderJobData,
};

pub struct _Storage<T> {
    pub data: Vec<T>,
    pub valid: Vec<AtomicBool>,
    avail: pqueue::Queue<Reverse<i32>>,
    extent: i32,
}
impl<T: 'static> _Storage<T> {
    pub fn clear(&mut self) {
        self.data.clear();
        self.avail = pqueue::Queue::new();
        self.extent = 0;
        self.valid.clear();
    }
    pub fn emplace(&mut self, d: T) -> i32 {
        match self.avail.pop() {
            Some(Reverse(i)) => {
                self.data[i as usize] = d;
                i
            }
            None => {
                self.data.push(d);
                self.extent += 1;
                self.extent - 1
            }
        }
    }
    pub fn erase(&mut self, id: i32) {
        self.avail.push(Reverse(id));
    }
    pub fn get(&self, i: &i32) -> &T {
        &self.data[*i as usize]
    }
    pub fn get_mut(&mut self, i: &i32) -> &mut T {
        &mut self.data[*i as usize]
    }
    pub fn new() -> _Storage<T> {
        _Storage::<T> {
            data: Vec::new(),
            valid: Vec::new(),
            avail: pqueue::Queue::new(),
            extent: 0,
        }
    }
}

pub trait StorageBase {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn update(&mut self, transforms: &Transforms, sys: &System, world: &World);
    fn late_update(&mut self, transforms: &Transforms, sys: &System);
    fn editor_update(&mut self, transforms: &Transforms, sys: &System, input: &Input);
    fn on_render(&mut self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData)>>);
    fn copy(&mut self, t: i32, i: i32) -> i32;
    fn remove(&self, i: i32);
    fn deinit(&self, transform: &Transform, i: i32, sys: &Sys);
    fn init(&self, transform: &Transform, i: i32, sys: &Sys);
    fn inspect(&self, transform: &Transform, i: i32, ui: &mut egui::Ui, sys: &Sys);
    fn get_name(&self) -> &'static str;
    fn get_id(&self) -> u64;
    fn get_type(&self) -> TypeId;
    fn new_default(&mut self, t: i32) -> i32;
    fn reduce_last(&mut self);
    // fn new(
    //     &mut self,
    //     t: i32,
    //     f: &Box<dyn Fn() -> (impl Component) + 'static + Sync + Send>,
    // ) -> i32;
    fn serialize(&self, i: i32) -> serde_yaml::Value;
    fn deserialize(&mut self, transform: i32, d: serde_yaml::Value) -> i32;
    fn clear(&mut self, transforms: &Transforms, sys: &Sys);
}
use dary_heap::DaryHeap;
// use pqueue::Queue;

pub struct Avail {
    pub data: DaryHeap<Reverse<i32>, 4>,
    new_ids: AtomicVec<Reverse<i32>>,
}
impl Avail {
    pub fn new() -> Self {
        Self {
            data: DaryHeap::new(),
            new_ids: AtomicVec::new(),
        }
    }
    pub fn commit(&mut self) {
        // let mut a = self.new_ids.lock();
        // let mut a = SegQueue::new();
        // std::mem::swap(&mut a, &mut self.new_ids);
        self.new_ids.get().iter().for_each(|i| {
            self.data.push(*i);
        });
        self.new_ids.clear();

        // a.clear();
    }
    pub fn push(&self, i: i32) {
        self.new_ids.push(Reverse(i));
        // self.data.push(Reverse(i));
    }
    pub fn pop(&mut self) -> Option<i32> {
        match self.data.pop() {
            Some(Reverse(a)) => Some(a),
            None => None,
        }
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
}
#[repr(C)]
pub struct Storage<T> {
    pub data: SegVec<Mutex<(i32, T)>>,
    pub valid: SegVec<SyncUnsafeCell<bool>>,
    new_ids_cache: VecCache<i32>,
    avail: Avail,
    last: i32,
    extent: i32,
    has_update: bool,
    has_render: bool,
    has_late_update: bool,
}
impl<
        T: 'static
            + Component
            + _ComponentID
            + Inspectable
            + Send
            + Sync
            + Default
            + Clone
            + Serialize
            + for<'a> Deserialize<'a>,
    > Storage<T>
{
    pub fn get_next_id(&mut self) -> (bool, i32) {
        if let Some(i) = self.avail.pop() {
            self.last = self.last.max(i);
            (true, i)
        } else {
            let i = self.extent;
            self.extent += 1;
            self.last = i;
            (false, i)
        }
    }
    fn write_t(&self, id: i32, transform: i32, d: T) {
        unsafe {
            *self.valid[id as usize].get() = true;
        };
        *self.data[id as usize].lock() = (transform, d);
    }
    fn push_t(&mut self, transform: i32, d: T) {
        self.data.push(Mutex::new((transform, d)));
        self.valid.push(SyncUnsafeCell::new(true));
    }
    pub fn insert(&mut self, transform: i32, d: T) -> i32 {
        let (write, id) = self.get_next_id();
        if write {
            self.write_t(id, transform, d);
        } else {
            self.push_t(transform, d);
        }
        id
    }
    pub fn insert_multi<D>(
        &mut self,
        count: i32,
        _t: &[i32],
        transforms: &Transforms,
        sys: &Sys,
        entities: &Vec<Mutex<Option<Entity>>>,
        d: &D,
    ) where
        D: Fn() -> T + Send + Sync,
    {
        let c = self.avail.data.len().min(count as usize);
        let mut max = 0;
        let ids = self.new_ids_cache.get_vec(c as usize);
        for _ in 0..c {
            if let Some(i) = self.avail.pop() {
                max = i;
                ids.push(i);
            }
        }
        self.last = self.last.max(max);
        ids.get()
            .par_iter()
            .zip_eq(_t[0..c].par_iter())
            .for_each(|(id, transform)| {
                let trans = transforms.get(*transform);
                self.write_t(*id, *transform, d());
                self.init(&trans, *id, sys);
                if let Some(ent) = entities[*transform as usize].lock().as_mut() {
                    ent.components.insert(T::ID, *id);
                }
            });

        for t in _t[c..count as usize].iter() {
            let trans = transforms.get(*t);
            let id = self.insert(*t, d());
            self.init(&trans, id, sys);
            if let Some(ent) = entities[*t as usize].lock().as_mut() {
                ent.components.insert(T::ID, id);
            }
            // ids.push(i);
        }
        // ids.get()
        //     .par_iter()
        //     .zip_eq(_t.par_iter())
        //     .for_each(|(id, t)| {
        //         let trans = transforms.get(*t);
        //         self.init(&trans, *id, sys);
        //         if let Some(ent) = entities[*t as usize].lock().as_mut() {
        //             ent.components.insert(T::ID, *id);
        //         }
        //     });

        // let c_id = stor.insert(*g, f());
        // let trans = world.transforms.get(*g);
        // stor.init(&trans, c_id, &mut world.sys);
        // if let Some(ent) = entities[*g as usize].lock().as_mut() {
        //     ent.components.insert(key.clone(), c_id);
        // }
    }
    fn _reduce_last(&mut self) {
        self.avail.commit();
        let mut id = self.last;
        while id >= 0 && !*self.valid[id as usize].get_mut() {
            // not thread safe!
            id -= 1;
        }
        self.last = id;
    }
    // pub fn clean(
    //     &mut self,
    //     transforms: &Transforms,
    //     sys: &Sys,
    //     _t: i32,
    //     entities: &Vec<Mutex<Option<Entity>>>,
    // ) {
    //     self.valid.iter().enumerate().for_each(|(i, mut a)| {
    //         if unsafe { *a.get() } {
    //             self.avail.push(i as i32);
    //         }
    //         unsafe {
    //             *a.get() = false;
    //         }
    //         // a.set(false)
    //     });
    //     self.last = -1;
    // }
    pub fn _erase(&self, id: i32) {
        // self.avail = self.avail.min(id);
        self.avail.push(id);
        unsafe {
            *self.valid[id as usize].get() = false;
        }

        // self.reduce_last(id);
    }
    // pub fn get(&self, i: &i32) -> &Mutex<T> {
    //     &self.data[*i as usize]
    // }
    pub fn new(has_update: bool, has_late_update: bool, has_render: bool) -> Storage<T> {
        Storage::<T> {
            data: SegVec::new(),
            valid: SegVec::new(),
            avail: Avail::new(),
            new_ids_cache: VecCache::new(),
            last: -1,
            extent: 0,
            has_update,
            has_late_update,
            has_render,
        }
    }
}

impl<
        T: 'static
            + Component
            + _ComponentID
            + Inspectable
            + Send
            + Sync
            + Default
            + Clone
            + Serialize
            + for<'a> Deserialize<'a>,
    > StorageBase for Storage<T>
{
    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self as &mut dyn Any
    }
    fn update(&mut self, transforms: &Transforms, sys: &System, world: &World) {
        if !self.has_update {
            return;
        }
        let last = (self.last + 1) as usize;
        (0..last).into_par_iter().for_each(|i| {
            if unsafe { *self.valid[i].get() } {
                let mut d = self.data[i].lock();
                let trans = transforms.get(d.0);
                d.1.update(&trans, &sys, world);
            }
        });
    }
    fn late_update(&mut self, transforms: &Transforms, sys: &System) {
        if !self.has_late_update {
            return;
        }
        let last = (self.last + 1) as usize;
        (0..last).into_par_iter().for_each(|i| {
            if unsafe { *self.valid[i].get() } {
                let mut d = self.data[i].lock();
                let trans = transforms.get(d.0);
                d.1.late_update(&trans, &sys);
            }
        });
    }
    fn remove(&self, i: i32) {
        self._erase(i);
    }
    fn deinit(&self, transform: &Transform, i: i32, sys: &Sys) {
        self.data[i as usize].lock().1.deinit(transform, i, sys);
    }
    fn init(&self, transform: &Transform, i: i32, sys: &Sys) {
        self.data[i as usize].lock().1.init(transform, i, sys);
    }
    fn inspect(&self, transform: &Transform, i: i32, ui: &mut egui::Ui, sys: &Sys) {
        self.data[i as usize]
            .lock()
            .1
            .inspect(transform, i, ui, sys);
    }

    fn get_name(&self) -> &'static str {
        std::any::type_name::<T>().split("::").last().unwrap()
    }

    fn new_default(&mut self, t: i32) -> i32 {
        let def: T = Default::default();
        self.insert(t, def)
    }

    fn get_type(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn copy(&mut self, t: i32, i: i32) -> i32 {
        let p = self.data[i as usize].lock().1.clone();

        self.insert(t, p)
    }

    fn serialize(&self, i: i32) -> serde_yaml::Value {
        serde_yaml::to_value(&self.data[i as usize].lock().1).unwrap()
    }
    fn clear(&mut self, transforms: &Transforms, sys: &Sys) {
        let last = (self.last + 1) as usize;
        (0..last).into_par_iter().for_each(|i| {
            if unsafe { *self.valid[i].get() } {
                let mut d = self.data[i].lock();
                let id: i32 = d.0;
                let trans = transforms.get(d.0);
                d.1.deinit(&trans, id, sys);
            }
        });
        // *self = Self::new(self.has_update, self.has_late_update, self.has_render);
    }

    fn deserialize(&mut self, transform: i32, d: serde_yaml::Value) -> i32 {
        let d: T = serde_yaml::from_value(d).unwrap();
        self.insert(transform, d)
    }

    fn on_render(&mut self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData)>>) {
        if !self.has_render {
            return;
        }
        let last = (self.last + 1) as usize;
        (0..last).into_iter().for_each(|i| {
            if unsafe { *self.valid[i].get() } {
                let mut d = self.data[i].lock();
                let t_id = d.0;
                render_jobs.push(d.1.on_render(t_id));
            }
        });
    }

    fn editor_update(&mut self, transforms: &Transforms, sys: &System, input: &Input) {
        if !self.has_update {
            return;
        }

        let last = (self.last + 1) as usize;
        (0..last).into_par_iter().for_each(|i| {
            if unsafe { *self.valid[i].get() } {
                let mut d = self.data[i].lock();
                let trans = transforms.get(d.0);
                d.1.editor_update(&trans, &sys);
            }
            // }
        });
    }
    fn get_id(&self) -> u64 {
        T::ID
    }
    fn reduce_last(&mut self) {
        self._reduce_last();
    }
}
