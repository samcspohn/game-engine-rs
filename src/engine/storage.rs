use std::{
    any::{Any, TypeId},
    cell::SyncUnsafeCell,
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap},
    marker::PhantomData,
    mem::transmute,
    ops::Div,
    ptr::slice_from_raw_parts,
    sync::{
        atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering},
        Arc,
    },
};

use bitvec::vec::BitVec;
use crossbeam::{epoch::Atomic, queue::SegQueue};
use force_send_sync::SendSync;
use id::ID_trait;
use ncollide3d::na::storage;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;
use segvec::SegVec;
use serde::{Deserialize, Serialize};
use thincollections::thin_vec::ThinVec;
use vulkano::padded::Padded;

use crate::engine::world::component::Update;

use super::{
    atomic_vec::AtomicVec,
    input::Input,
    perf::Perf,
    transform_compute::cs::s,
    world::{
        component::{Component, EditorUpdate, LateUpdate, OnRender, System},
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
    // fn late_update(&self, transforms: &Transforms, sys: &System);
    // fn editor_update(&mut self, transforms: &Transforms, sys: &System, input: &Input);
    // fn on_render(&mut self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData) + Send + Sync>>);
    fn copy(&mut self, t: i32, i: i32) -> i32;
    fn remove(&self, i: i32);
    #[inline]
    fn deinit(&self, transform: &Transform, i: i32, sys: &Sys);
    #[inline]
    fn init(&self, transform: &Transform, i: i32, sys: &Sys);
    #[inline]
    fn on_start(&self, transform: &Transform, i: i32, sys: &System);
    #[inline]
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
    pub valid: SegVec<SyncUnsafeCell<Padded<bool, 3>>>,
    new_ids_cache: VecCache<i32>,
    new_offsets_cache: VecCache<i32>,
    avail: Avail,
    last: i32,
    extent: i32,
    has_update: bool,
    has_render: bool,
    has_late_update: bool,
}
impl<T: ComponentTraits> Storage<T> {
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
                f(t_id, &mut d);
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
            self.valid
                .resize_with(c, || SyncUnsafeCell::new(Padded(false)));
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
    fn update(&self, transforms: &Transforms, sys: &System, world: &World) {

        // if let Some(s) = self.as_any().downcast_ref::<&dyn StorageBase>().unwrap().as_any().downcast_ref::<Storage<UpdateTrait>>() {
        //     s.par_for_each(|t_id, d| {
        //         if let Some(trans) = transforms.get(t_id) {
        //             d.update(&trans, &sys, world);
        //         } else {
        //             panic!("transform {} is invalid", t_id);
        //         }
        //     });
        // }

        // if !self.has_update {
        //     return;
        // }
        // if storage_has_update::<T>() {
        //     self.par_for_each(|t_id, d| {
        //         if let Some(trans) = transforms.get(t_id) {
        //             (d as &mut dyn Update).update(&trans, &sys, world);
        //         } else {
        //             panic!("transform {} is invalid", t_id);
        //         }
        //     });
        // }
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
            // + AsAny
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
        self.update(transforms, sys, world);
        // if !self.has_update {
        //     return;
        // }
        // if storage_has_update::<T>() {
        //     self.par_for_each(|t_id, d| {
        //         if let Some(trans) = transforms.get(t_id) {
        //             (d as &mut dyn Update).update(&trans, &sys, world);
        //         } else {
        //             panic!("transform {} is invalid", t_id);
        //         }
        //     });
        // }
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
    // fn late_update(&self, transforms: &Transforms, sys: &System) {
    //     if !self.has_late_update {
    //         return;
    //     }
    //     if storage_has_late_update::<T>() {
    //         // if let Some(_) = T::as_any(&T::default()).downcast_ref::<dyn LateUpdate>() {
    //         self.par_for_each(|t_id, d| {
    //             let trans = transforms.get(t_id).unwrap();
    //             (d as &mut dyn LateUpdate).late_update(&trans, &sys);
    //         })
    //     }
    // }
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

    // fn on_render(&mut self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData) + Send + Sync>>) {
    //     if !self.has_render {
    //         return;
    //     }
    //     if let Some(_) = T::as_any(&T::default()).downcast_ref::<OnRender>() {
    //         self.for_each(|t_id, d| {
    //             render_jobs.push(d.on_render(t_id));
    //         });
    //     }
    //     // self.for_each(|t_id, d| {
    //     //     render_jobs.push(d.on_render(t_id));
    //     // });
    // }

    // fn editor_update(&mut self, transforms: &Transforms, sys: &System, input: &Input) {
    //     if !self.has_update {
    //         return;
    //     }
    //     if let Some(a) = T::as_any(&T::default()).downcast_ref::<Update>() {
    //         self.par_for_each(|t_id, d| {
    //             let trans = transforms.get(t_id).unwrap();
    //             d.editor_update(&trans, &sys);
    //         });
    //     }
    // }
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

pub trait StorageTrait: StorageBase + Send + Sync {}
impl<T: StorageBase + Send + Sync> StorageTrait for T {}

// pub trait AsAny {
//     fn as_any(&self) -> &dyn Any;
//     fn as_any_mut(&mut self) -> &mut dyn Any;
// }

// impl<T: 'static + Sized> AsAny for Storage<T> {
//     fn as_any(&self) -> &dyn Any {
//         self as &dyn Any
//     }
//     fn as_any_mut(&mut self) -> &mut dyn Any {
//         self as &mut dyn Any
//     }
// }

pub fn storage_has_update<T: ?Sized>() -> bool {
    use std::cell::Cell;
    use std::marker::PhantomData;

    struct IsQuxTest<'a, T: ?Sized> {
        is_qux: &'a Cell<bool>,
        _marker: PhantomData<T>,
    }
    impl<T: ?Sized> Clone for IsQuxTest<'_, T> {
        fn clone(&self) -> Self {
            self.is_qux.set(false);
            IsQuxTest {
                is_qux: self.is_qux,
                _marker: PhantomData,
            }
        }
    }
    impl<T: ?Sized + Update> Copy for IsQuxTest<'_, T> {}

    let is_qux = Cell::new(true);
    _ = [IsQuxTest::<T> {
        is_qux: &is_qux,
        _marker: PhantomData,
    }]
    .clone();

    is_qux.get()
}
pub fn storage_has_late_update<T: ?Sized>() -> bool {
    use std::cell::Cell;
    use std::marker::PhantomData;

    struct IsQuxTest<'a, T: ?Sized> {
        is_qux: &'a Cell<bool>,
        _marker: PhantomData<T>,
    }
    impl<T: ?Sized> Clone for IsQuxTest<'_, T> {
        fn clone(&self) -> Self {
            self.is_qux.set(false);
            IsQuxTest {
                is_qux: self.is_qux,
                _marker: PhantomData,
            }
        }
    }
    impl<T: ?Sized + LateUpdate> Copy for IsQuxTest<'_, T> {}

    let is_qux = Cell::new(true);
    _ = [IsQuxTest::<T> {
        is_qux: &is_qux,
        _marker: PhantomData,
    }]
    .clone();

    is_qux.get()
}
pub fn storage_has_on_render<T: ?Sized>() -> bool {
    use std::cell::Cell;
    use std::marker::PhantomData;

    struct IsQuxTest<'a, T: ?Sized> {
        is_qux: &'a Cell<bool>,
        _marker: PhantomData<T>,
    }
    impl<T: ?Sized> Clone for IsQuxTest<'_, T> {
        fn clone(&self) -> Self {
            self.is_qux.set(false);
            IsQuxTest {
                is_qux: self.is_qux,
                _marker: PhantomData,
            }
        }
    }
    impl<T: ?Sized + OnRender> Copy for IsQuxTest<'_, T> {}

    let is_qux = Cell::new(true);
    _ = [IsQuxTest::<T> {
        is_qux: &is_qux,
        _marker: PhantomData,
    }]
    .clone();

    is_qux.get()
}
pub fn storage_has_editor_update<T: ?Sized>() -> bool {
    use std::cell::Cell;
    use std::marker::PhantomData;

    struct IsQuxTest<'a, T: ?Sized> {
        is_qux: &'a Cell<bool>,
        _marker: PhantomData<T>,
    }
    impl<T: ?Sized> Clone for IsQuxTest<'_, T> {
        fn clone(&self) -> Self {
            self.is_qux.set(false);
            IsQuxTest {
                is_qux: self.is_qux,
                _marker: PhantomData,
            }
        }
    }
    impl<T: ?Sized + EditorUpdate> Copy for IsQuxTest<'_, T> {}

    let is_qux = Cell::new(true);
    _ = [IsQuxTest::<T> {
        is_qux: &is_qux,
        _marker: PhantomData,
    }]
    .clone();

    is_qux.get()
}


pub trait StorageUpdaterBase {
    fn update(&self, transforms: &Transforms, sys: &System, world: &World);
}
pub fn new_storage_updater<T: ComponentTraits + Update>(
    storage: Arc<RwLock<Box<dyn StorageBase + 'static + Send + Sync>>>,
) -> Arc<dyn StorageUpdaterBase + 'static + Send + Sync> {
    Arc::new(StorageUpdater::<T>::new(storage))
}


pub struct StorageUpdater<T> {
    storage: Arc<RwLock<Box<dyn StorageBase + 'static + Send + Sync>>>,
    p: PhantomData<T>,
}

impl<T: ComponentTraits + Update> StorageUpdaterBase for StorageUpdater<T> {
    fn update(&self, transforms: &Transforms, sys: &System, world: &World) {
        let storage = self.storage.read();
        let stor = unsafe { storage.as_any().downcast_ref_unchecked::<Storage<T>>() };
        stor.par_for_each(|t_id, d| {
            if let Some(trans) = transforms.get(t_id) {
                d.update(&trans, &sys, world);
            } else {
                panic!("transform {} is invalid", t_id);
            }
        });
    }
}
impl <T: ComponentTraits + Update> StorageUpdater<T> {
    pub fn new(storage: Arc<RwLock<Box<dyn StorageBase + 'static + Send + Sync>>>) -> Self {
        Self {
            storage,
            p: PhantomData,
        }
    }
}

pub trait StorageLateUpdaterBase {
    fn late_update(&self, transforms: &Transforms, sys: &System);
}

pub fn new_storage_late_updater<T: ComponentTraits + LateUpdate>(
    storage: Arc<RwLock<Box<dyn StorageBase + 'static + Send + Sync>>>,
) -> Arc<dyn StorageLateUpdaterBase + 'static + Send + Sync> {
    Arc::new(StorageLateUpdater::<T>::new(storage))
}
pub struct StorageLateUpdater<T> {
    storage: Arc<RwLock<Box<dyn StorageBase + 'static + Send + Sync>>>,
    p: PhantomData<T>,
}
impl<T: ComponentTraits + LateUpdate> StorageLateUpdaterBase for StorageLateUpdater<T> {
    fn late_update(&self, transforms: &Transforms, sys: &System) {
        let storage = self.storage.read();
        let stor = unsafe { storage.as_any().downcast_ref_unchecked::<Storage<T>>() };
        stor.par_for_each(|t_id, d| {
            if let Some(trans) = transforms.get(t_id) {
                d.late_update(&trans, &sys);
            } else {
                panic!("transform {} is invalid", t_id);
            }
        });
    }
}
impl<T: ComponentTraits + LateUpdate> StorageLateUpdater<T> {
    pub fn new(storage: Arc<RwLock<Box<dyn StorageBase + 'static + Send + Sync>>>) -> Self {
        Self {
            storage,
            p: PhantomData,
        }
    }
}

pub trait StorageEditorUpdaterBase {
    fn editor_update(&self, transforms: &Transforms, sys: &System, input: &Input);
}
pub fn new_storage_editor_updater<T: ComponentTraits + EditorUpdate>(
    storage: Arc<RwLock<Box<dyn StorageBase + 'static + Send + Sync>>>,
) -> Arc<dyn StorageEditorUpdaterBase + 'static + Send + Sync> {
    Arc::new(StorageEditorUpdater::<T>::new(storage))
}
pub struct StorageEditorUpdater<T> {
    storage: Arc<RwLock<Box<dyn StorageBase + 'static + Send + Sync>>>,
    p: PhantomData<T>,
}

impl<T: ComponentTraits + EditorUpdate> StorageEditorUpdaterBase for StorageEditorUpdater<T> {
    fn editor_update(&self, transforms: &Transforms, sys: &System, input: &Input) {
        let storage = self.storage.read();
        let stor = unsafe { storage.as_any().downcast_ref_unchecked::<Storage<T>>() };
        stor.par_for_each(|t_id, d| {
            if let Some(trans) = transforms.get(t_id) {
                d.editor_update(&trans, &sys);
            } else {
                panic!("transform {} is invalid", t_id);
            }
        });
    }
}
impl<T: ComponentTraits + EditorUpdate> StorageEditorUpdater<T> {
    pub fn new(storage: Arc<RwLock<Box<dyn StorageBase + 'static + Send + Sync>>>) -> Self {
        Self {
            storage,
            p: PhantomData,
        }
    }
}

pub trait StorageOnRenderBase {
    fn on_render(&self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData) + Send + Sync>>);
}
pub fn new_storage_on_render<T: ComponentTraits + OnRender>(
    storage: Arc<RwLock<Box<dyn StorageBase + 'static + Send + Sync>>>,
) -> Arc<dyn StorageOnRenderBase + 'static + Send + Sync> {
    Arc::new(StorageOnRender::<T>::new(storage))
}
pub struct StorageOnRender<T> {
    storage: Arc<RwLock<Box<dyn StorageBase + 'static + Send + Sync>>>,
    p: PhantomData<T>,
}
impl<T: ComponentTraits + OnRender> StorageOnRenderBase for StorageOnRender<T> {
    fn on_render(&self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData) + Send + Sync>>) {
        let storage = self.storage.read();
        let stor = unsafe { storage.as_any().downcast_ref_unchecked::<Storage<T>>() };
        stor.for_each(|t_id, d| {
            render_jobs.push(d.on_render(t_id));
        });
    }
}
impl<T: ComponentTraits + OnRender> StorageOnRender<T> {
    pub fn new(storage: Arc<RwLock<Box<dyn StorageBase + 'static + Send + Sync>>>) -> Self {
        Self {
            storage,
            p: PhantomData,
        }
    }
}
// pub trait StorageLateUpdate {
//     fn late_update(&self, transforms: &Transforms, sys: &System);
// }
// pub trait StorageUpdate {
//     fn update(&self, transforms: &Transforms, sys: &System, world: &World);
// }

// pub trait StorageEditorUpdate {
//     fn editor_update(&mut self, transforms: &Transforms, sys: &System, input: &Input);
// }

// pub trait StorageOnRender {
//     fn on_render(&mut self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData) + Send + Sync>>);
// }

// impl<
//         T: 'static
//             + Component
//             + ID_trait
//             + Send
//             + Sync
//             + Default
//             + Clone
//             + Serialize
//             // + AsAny
//             + for<'a> Deserialize<'a>
//             + LateUpdate,
//     > StorageLateUpdate for Storage<T>
// {
//     fn late_update(&self, transforms: &Transforms, sys: &System) {
//         self.par_for_each(|t_id, d| {
//             let trans = transforms.get(t_id).unwrap();
//             d.late_update(&trans, &sys);
//         })
//     }
// }
// pub trait StorageLateUpdateTrait: StorageBase + StorageLateUpdate + Send + Sync {}
// impl<T: StorageBase + StorageLateUpdate + Send + Sync> StorageLateUpdateTrait for T {}

// impl<
//         T: 'static
//             + Component
//             + ID_trait
//             + Send
//             + Sync
//             + Default
//             + Clone
//             + Serialize
//             // + AsAny
//             + for<'a> Deserialize<'a>
//             + Update,
//     > StorageUpdate for Storage<T>
// {
//     fn update(&self, transforms: &Transforms, sys: &System, world: &World) {
//         self.par_for_each(|t_id, d| {
//             if let Some(trans) = transforms.get(t_id) {
//                 d.update(&trans, &sys, world);
//             } else {
//                 panic!("transform {} is invalid", t_id);
//             }
//         });
//     }
// }

// pub trait StorageUpdateTrait: StorageBase + StorageUpdate + Send + Sync {}
// impl<T: StorageBase + StorageUpdate + Send + Sync> StorageUpdateTrait for T {}

// impl<
//         T: 'static
//             + Component
//             + ID_trait
//             + Send
//             + Sync
//             + Default
//             + Clone
//             + Serialize
//             // + AsAny
//             + for<'a> Deserialize<'a>
//             + EditorUpdate,
//     > StorageEditorUpdate for Storage<T>
// {
//     fn editor_update(&mut self, transforms: &Transforms, sys: &System, input: &Input) {
//         self.par_for_each(|t_id, d| {
//             let trans = transforms.get(t_id).unwrap();
//             d.editor_update(&trans, &sys);
//         });
//     }
// }
// pub trait StorageEditorUpdateTrait: StorageBase + StorageEditorUpdate + Send + Sync {}
// impl<T: StorageBase + StorageEditorUpdate + Send + Sync> StorageEditorUpdateTrait for T {}

// impl<
//         T: 'static
//             + Component
//             + ID_trait
//             + Send
//             + Sync
//             + Default
//             + Clone
//             + Serialize
//             // + AsAny
//             + for<'a> Deserialize<'a>
//             + OnRender,
//     > StorageOnRender for Storage<T>
// {
//     fn on_render(&mut self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData) + Send + Sync>>) {
//         self.for_each(|t_id, d| {
//             render_jobs.push(d.on_render(t_id));
//         });
//     }
// }
// pub trait StorageOnRenderTrait: StorageBase + StorageOnRender + Send + Sync {}
// impl<T: StorageBase + StorageOnRender + Send + Sync> StorageOnRenderTrait for T {}

pub trait ComponentTraits:
    'static + Component + ID_trait + Send + Sync + Default + Clone + Serialize + for<'a> Deserialize<'a>
{
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
    > ComponentTraits for T
{
}
pub trait UpdateTrait: ComponentTraits + Update {}
impl<T: ComponentTraits + Update> UpdateTrait for T {}

pub trait LateUpdateTrait: ComponentTraits + LateUpdate {}
impl<T: ComponentTraits + LateUpdate> LateUpdateTrait for T {}

pub trait OnRenderTrait: ComponentTraits + OnRender {}
impl<T: ComponentTraits + OnRender> OnRenderTrait for T {}

pub trait EditorUpdateTrait: ComponentTraits + EditorUpdate {}
impl<T: ComponentTraits + EditorUpdate> EditorUpdateTrait for T {}

// pub fn update_storage<T: ComponentTraits>(
//     s: &dyn StorageBase,
//     transforms: &Transforms,
//     sys: &System,
//     world: &World,
// ) {
//     if let Some(s) = s.as_any().downcast_ref::<Storage<T>>() {
//         s.par_for_each(|t_id, d| {
//             if let Some(trans) = transforms.get(t_id) {
//                 d.as_any().update(&trans, sys, world);
//                 // d.update(&trans, &sys, world);
//             } else {
//                 panic!("transform {} is invalid", t_id);
//             }
//         });
//     }
// }

// pub fn late_update_storage<
//     T: 'static
//         + Component
//         + ID_trait
//         + Send
//         + Sync
//         + Default
//         + Clone
//         + Serialize
//         + LateUpdate
//         // + AsAny
//         + for<'a> Deserialize<'a>,
// >(
//     s: &dyn StorageBase,
//     transforms: &Transforms,
//     sys: &System,
// ) {
//     if let Some(s) = s.as_any().downcast_ref::<Storage<T>>() {
//         s.par_for_each(|t_id, d| {
//             if let Some(trans) = transforms.get(t_id) {
//                 d.late_update(&trans, &sys);
//             } else {
//                 panic!("transform {} is invalid", t_id);
//             }
//         });
//     }
// }

// pub fn on_render_storage<
//     T: 'static
//         + Component
//         + ID_trait
//         + Send
//         + Sync
//         + Default
//         + Clone
//         + Serialize
//         + OnRender
//         // + AsAny
//         + for<'a> Deserialize<'a>,
// >(
//     s: &mut dyn StorageBase,
//     render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData) + Send + Sync>>,
// ) {
//     if let Some(s) = s.as_any_mut().downcast_mut::<Storage<T>>() {
//         s.on_render(render_jobs);
//     }
// }

// pub fn editor_update_storage<
//     T: 'static
//         + Component
//         + ID_trait
//         + Send
//         + Sync
//         + Default
//         + Clone
//         + Serialize
//         + EditorUpdate
//         // + AsAny
//         + for<'a> Deserialize<'a>,
// >(
//     s: &mut dyn StorageBase,
//     transforms: &Transforms,
//     sys: &System,
//     input: &Input,
// ) {
//     if let Some(s) = s.as_any_mut().downcast_mut::<Storage<T>>() {
//         s.editor_update(transforms, sys, input);
//     }
// }
