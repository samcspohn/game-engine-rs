use std::{
    any::{Any, TypeId},
    cell::SyncUnsafeCell,
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap},
    ops::Div,
    ptr::slice_from_raw_parts,
    sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering},
};

use bitvec::vec::BitVec;
use crossbeam::{epoch::Atomic, queue::SegQueue};
use force_send_sync::SendSync;
use id::ID_trait;
use parking_lot::Mutex;
use rayon::prelude::*;
use segvec::SegVec;
use serde::{Deserialize, Serialize};
use thincollections::thin_vec::ThinVec;
use vulkano::padded::Padded;

use super::{
    atomic_vec::AtomicVec,
    input::Input,
    perf::Perf,
    world::{
        component::{Component, System},
        entity::Entity,
        transform::{CacheVec, Transform, Transforms, VecCache},
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
                self.valid[i as usize].store(true, Ordering::Relaxed);
                i
            }
            None => {
                self.data.push(d);
                self.valid.push(AtomicBool::new(true));
                self.extent += 1;
                self.extent - 1
            }
        }
    }
    pub fn erase(&mut self, id: i32) {
        self.avail.push(Reverse(id));
        self.valid[id as usize].store(false, Ordering::Relaxed);
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
    fn update(&self, transforms: &Transforms, sys: &System, world: &World);
    fn late_update(&self, transforms: &Transforms, sys: &System);
    fn editor_update(&mut self, transforms: &Transforms, sys: &System, input: &Input);
    fn on_render(&mut self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData) + Send + Sync>>);
    fn copy(&mut self, t: i32, i: i32) -> i32;
    fn remove(&self, i: i32);
    fn deinit(&self, transform: &Transform, i: i32, sys: &Sys);
    fn init(&self, transform: &Transform, i: i32, sys: &Sys);
    fn on_start(&self, transform: &Transform, i: i32, sys: &System);
    fn on_destroy(&self, transform: &Transform, i: i32, sys: &System);
    fn inspect(&self, transform: &Transform, i: i32, ui: &mut egui::Ui, sys: &Sys);
    fn get_name(&self) -> &'static str;
    fn get_id(&self) -> u64;
    fn get_type(&self) -> TypeId;
    fn new_default(&mut self, t: i32) -> i32;
    fn reduce_last(&mut self);
    fn allocate(&mut self, count: usize) -> CacheVec<i32>;
    fn serialize(&self, i: i32) -> serde_yaml::Value;
    fn deserialize(&mut self, transform: i32, d: serde_yaml::Value) -> i32;
    fn clear(&mut self, transforms: &Transforms, sys: &Sys);
    fn len(&self) -> usize;
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
        self.new_ids.get().iter().for_each(|i| unsafe {
            self.data.push(i.assume_init());
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
    pub data: SegVec<(SyncUnsafeCell<i32>, Mutex<T>)>,
    pub valid: SegVec<SyncUnsafeCell<Padded<bool,3>>>,
    new_ids_cache: VecCache<i32>,
    new_offsets_cache: VecCache<i32>,
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
            + ID_trait
            + Send
            + Sync
            + Default
            + Clone
            + Serialize
            + for<'a> Deserialize<'a>,
    > Storage<T>
{
    // pub fn iter(&self) -> impl Iterator<Item = &(SyncUnsafeCell<i32>, Mutex<T>)> {
    //     let a = self
    //         .data
    //         .iter()
    //         .zip(self.valid.iter())
    //         .filter(|(_, v)| unsafe { *v.get() })
    //         .map(|(d, v)| d);
    //     a
    // }
    pub fn for_each<D>(&self, mut f: D)
    where
        D: FnMut(i32, &mut T) + Send + Sync,
    {
        self.data.iter().zip(self.valid.iter()).for_each(|(d, v)| {
            if unsafe { **v.get() } {
                let id: i32 = unsafe { *d.0.get() };
                let mut d = d.1.lock();
                f(id, &mut d);
            }
        });
    }
    pub fn par_for_each<D>(&self, f: D)
    where
        D: Fn(i32, &mut T) + Send + Sync,
    {
        let last = (self.last + 1) as usize;
        // let leading = 31 - last.leading_zeros();
        // let leading = last.log2();
        // let data = unsafe { SendSync::new(self.data.slice(0..last)) };
        // let valid = unsafe { SendSync::new(self.valid.slice(0..last)) };
        // rayon::scope(|s| {
        //     data.segmented_iter()
        //         .zip(valid.segmented_iter())
        //         .for_each(|p| {
        //             s.spawn(|_| {
        //                 p.0.par_iter().zip_eq(p.1.par_iter()).for_each(|(d, v)| {
        //                     if unsafe { *v.get() } {
        //                         let t_id = unsafe { *d.0.get() };
        //                         let mut d = d.1.lock();
        //                         f(t_id, &mut d);
        //                     }
        //                 })
        //             })
        //         })
        // });

        // let index = AtomicUsize::new(0);
        // let num = num_cpus::get();
        // let chunk_size = last.div(num).max(1).min(256);
        // // (0..num_cpus::get()).into_par_iter().for_each(|thread_id| {
        // rayon::scope(|s| {
        //     for _ in (0..num_cpus::get()) {
        //         s.spawn(|_| {
        //             let mut start = index.fetch_add(chunk_size, Ordering::SeqCst);
        //             let mut end = (start + chunk_size).min(last);
        //             while start < last {
        //                 for i in start..end {
        //                     if unsafe { *self.valid[i].get() } {
        //                         let mut d = self.data[i].1.lock();
        //                         let t_id = unsafe { *self.data[i].0.get() };
        //                         f(t_id, &mut d);
        //                     }
        //                 }
        //                 start = index.fetch_add(chunk_size, Ordering::SeqCst);
        //                 end = (start + chunk_size).min(last);
        //             }
        //         });
        //     }
        // })
        // });

        (0..last).into_par_iter().for_each(|i| {
            if unsafe { **self.valid[i].get() } {
                let mut d = self.data[i].1.lock();
                let t_id = unsafe { *self.data[i].0.get() };
                f(t_id,&mut d);
            }
        });
        // let data = unsafe { &*slice_from_raw_parts(self.data.as_ptr(), last) };
        // let valid = unsafe { &*slice_from_raw_parts(self.valid.as_ptr(), last) };

        // data.par_iter()
        //     .zip_eq(valid.par_iter())
        //     .filter(|(_, v)| unsafe { *v.get() })
        //     .for_each(|(d, _)| {
        //         // if unsafe { *v.get() } {
        //         let t_id = unsafe { *d.0.get() };
        //         let mut d = d.1.lock();
        //         f(t_id, &mut d);
        //         // }
        //     });
    }
    pub fn len(&self) -> usize {
        self.valid.len()
    }
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
            *self.valid[id as usize].get() = Padded(true);
            *self.data[id as usize].0.get() = transform;
        };
        *self.data[id as usize].1.lock() = d;
    }
    fn push_t(&mut self, transform: i32, d: T) {
        self.data
            .push((SyncUnsafeCell::new(transform), Mutex::new(d)));
        self.valid.push(SyncUnsafeCell::new(Padded(true)));
    }
    fn reserve(&mut self, count: usize) {
        let c = self.avail.len().min(count);
        let c = count - c;
        if c > 0 {
            let c = c + self.len();
            self.data
                .resize_with(c, || (SyncUnsafeCell::new(-1), Mutex::new(T::default())));
            self.valid.resize_with(c, || SyncUnsafeCell::new(Padded(false)));
        }
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
    pub(crate) fn _allocate(&mut self, count: usize) -> CacheVec<i32> {
        let mut r = self.new_ids_cache.get_vec(count);
        let c = self.avail.len().min(count as usize);
        self.reserve(count);
        let mut max = -1;
        for _ in 0..c {
            if let Some(i) = self.avail.pop() {
                max = i;
                r.push(i);
            }
        }
        self.last = self.last.max(max);
        for i in (c..count) {
            self.last += 1;
            r.get().push(self.last);
        }
        self.extent = self.extent.max(self.last + 1);
        r
    }
    pub fn insert_exact<D>(&self, id: i32, trans: &Transform<'_>, sys: &Sys, f: &D)
    where
        D: Fn() -> T + Send + Sync,
    {
        self.write_t(id, trans.id, f());
        self.init(&trans, id, sys);
        trans.entity().insert(T::ID, id);
    }
    pub(crate) fn insert_multi<D>(
        &self,
        count: usize,
        transforms: &Transforms,
        sys: &Sys,
        syst: &System,
        t_: &[i32],
        t_offset: usize,
        perf: &Perf,
        c_offset: usize,
        c_ids: &[i32],
        // c_id: usize,
        f: &D,
    ) where
        D: Fn() -> T + Send + Sync,
    {
        (0..count).into_par_iter().for_each(|i| {
            let id = c_ids[c_offset + i];
            let t = t_[t_offset + i];
            self.write_t(id, t, f());
            if let Some(trans) = transforms.get(t) {
                self.init(&trans, id, sys);
                trans.entity().insert(T::ID, id);
                self.on_start(&trans, id, syst);
            }
        });
    }
    fn _reduce_last(&mut self) {
        self.avail.commit();
        let mut id = self.last;
        while id >= 0 && !**self.valid[id as usize].get_mut() {
            // not thread safe!
            id -= 1;
        }
        self.last = id;
    }
    pub fn _erase(&self, id: i32) {
        // self.avail = self.avail.min(id);
        self.avail.push(id);
        unsafe {
            **self.valid[id as usize].get() = false;
        }
    }
    pub fn new(has_update: bool, has_late_update: bool, has_render: bool) -> Storage<T> {
        Storage::<T> {
            data: SegVec::new(),
            valid: SegVec::new(),
            avail: Avail::new(),
            new_ids_cache: VecCache::new(),
            new_offsets_cache: VecCache::new(),
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
            + ID_trait
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
    fn update(&self, transforms: &Transforms, sys: &System, world: &World) {
        if !self.has_update {
            return;
        }
        self.par_for_each(|t_id, d| {
            if let Some(trans) = transforms.get(t_id) {
                d.update(&trans, &sys, world);
            } else {
                panic!("transform {} is invalid", t_id);
            }
        });
        // (0..last).into_par_iter().for_each(|i| {
        //     if unsafe { *self.valid[i].get() } {
        //         let mut d = self.data[i].1.lock();
        //         if let Some(trans) = transforms.get(unsafe { *self.data[i].0.get() }) {
        //             d.update(&trans, &sys, world);
        //         } else {
        //             panic!("transform {} is invalid", unsafe { *self.data[i].0.get() });
        //         }
        //     }
        // });
    }
    fn late_update(&self, transforms: &Transforms, sys: &System) {
        if !self.has_late_update {
            return;
        }
        self.par_for_each(|t_id, d| {
            let trans = transforms.get(t_id).unwrap();
            d.late_update(&trans, &sys);
        })
    }
    fn remove(&self, i: i32) {
        self._erase(i);
    }
    fn deinit(&self, transform: &Transform, i: i32, sys: &Sys) {
        self.data[i as usize].1.lock().deinit(transform, i, sys);
    }
    fn init(&self, transform: &Transform, i: i32, sys: &Sys) {
        self.data[i as usize].1.lock().init(transform, i, sys);
    }
    fn on_start(&self, transform: &Transform, i: i32, sys: &System) {
        self.data[i as usize].1.lock().on_start(transform, sys);
    }
    fn on_destroy(&self, transform: &Transform, i: i32, sys: &System) {
        self.data[i as usize].1.lock().on_destroy(transform, sys);
    }
    fn inspect(&self, transform: &Transform, i: i32, ui: &mut egui::Ui, sys: &Sys) {
        self.data[i as usize]
            .1
            .lock()
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
        let p = self.data[i as usize].1.lock().clone();

        self.insert(t, p)
    }

    fn serialize(&self, i: i32) -> serde_yaml::Value {
        serde_yaml::to_value(&*self.data[i as usize].1.lock()).unwrap()
    }
    fn clear(&mut self, transforms: &Transforms, sys: &Sys) {
        self.par_for_each(|id, d| {
            let trans = transforms.get(id).unwrap();
            d.deinit(&trans, id, sys);
        });
    }

    fn deserialize(&mut self, transform: i32, d: serde_yaml::Value) -> i32 {
        let d: T = serde_yaml::from_value(d).unwrap();
        self.insert(transform, d)
    }

    fn on_render(&mut self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData) + Send + Sync>>) {
        if !self.has_render {
            return;
        }
        self.for_each(|t_id, d| {
            render_jobs.push(d.on_render(t_id));
        });
    }

    fn editor_update(&mut self, transforms: &Transforms, sys: &System, input: &Input) {
        if !self.has_update {
            return;
        }
        self.par_for_each(|t_id, d| {
            let trans = transforms.get(t_id).unwrap();
            d.editor_update(&trans, &sys);
        });
    }
    fn get_id(&self) -> u64 {
        T::ID
    }
    fn reduce_last(&mut self) {
        self._reduce_last();
    }
    fn allocate(&mut self, count: usize) -> CacheVec<i32> {
        self._allocate(count)
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}
