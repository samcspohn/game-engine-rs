use std::{
    any::{Any, TypeId},
    cmp::Reverse,
    collections::{BTreeSet, BinaryHeap},
    sync::atomic::{AtomicBool, AtomicI32, Ordering},
};

use parking_lot::Mutex;
use rayon::prelude::*;
use segvec::SegVec;
use serde::{Deserialize, Serialize};
use sync_unsafe_cell::SyncUnsafeCell;

use crate::editor::inspectable::Inspectable;

use super::{
    input::Input,
    world::{
        component::{Component, System, _ComponentID},
        transform::{Transform, Transforms},
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
    fn erase(&mut self, i: i32);
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
    // pub data: Vec<BinaryHeap<Reverse<i32>>>,
    pub data: DaryHeap<Reverse<i32>, 4>,
    // pub new_elem: SegVec<i32>,
    // last: usize
}
impl Avail {
    pub fn new() -> Self {
        // let mut a = sharded_slab::Slab::<Mutex<i32>>::new();
        // let key = a.insert(Mutex::new(0));
        // // a.remove(key);
        // rayon::scope(|s|{
        //     a.unique_iter().for_each(|b|{
        //         s.spawn(|s| {
        //           *b.lock() += 1;  
        //         })
        //     })

        // });
        // let mut data = radix_heap::RadixHeapMap::new();
        // data.push(Reverse(0), ());
        Self {
            data: DaryHeap::new(), // data: (0..8).into_iter().map(|_| BinaryHeap::new()).collect(),
                                      // new_elem: SegVec::new(),
        }
    }
    pub fn commit(&mut self) {
        // self.data.constrain();
        // self.data.sort_unstable()
        // let mut last = self.data.len();
        // self.data.resize(self.data.len() + self.new_elem.len(), 0);
        // while let Some(a) = self.new_elem.pop() {
        //     let mut i = last;
        //     last += 1;
        //     while i > 0 && a > self.data[i - 1] {
        //         self.data[i] = self.data[i - 1];
        //         i -= 1;
        //     }
        //     self.data[i] = a;
        // }
        // // unsafe { self.data}
    }
    pub fn push(&mut self, i: i32) {
        self.data.push(Reverse(i));
        // self.data
        //     .iter_mut()
        //     .min_by(|a, b| a.len().cmp(&b.len()))
        //     .unwrap()
        //     .push(Reverse(i));
    }
    pub fn pop(&mut self) -> Option<i32> {
        // match self
        //     .data
        //     .iter_mut()
        //     .max_by(|a, b| a.peek().cmp(&b.peek()))
        //     .unwrap()
        //     .pop()
        // {
        //     Some(Reverse(a)) => Some(a),
        //     None => None,
        // }
        match self.data.pop() {
            Some(Reverse(a)) => Some(a),
            None => None,
        }
    }
}
#[repr(C)]
pub struct Storage<T> {
    pub data: SegVec<Mutex<(i32, T)>>,
    pub valid: SegVec<SyncUnsafeCell<bool>>,
    avail: Avail,
    last: i32,
    extent: i32,
    has_update: bool,
    has_render: bool,
    has_late_update: bool,
}
impl<T: 'static> Storage<T> {
    pub fn get_next_id(&mut self) -> (bool, i32) {
        // let mut i = self.avail;
        // self.avail += 1;
        // while i < self.extent && unsafe { *self.valid[i as usize].get() } {
        //     i = self.avail;
        //     self.avail += 1;
        // }
        if let Some(i) = self.avail.pop() {
            self.last = self.last.max(i);
            (true, i)
        } else {
            let i = self.extent;
            self.extent += 1;
            self.last = i;
            (false, i)
        }
        // if i == self.extent {
        //     // push back
        //     self.extent += 1;
        //     self.last = i;
        //     (false, i)
        // } else {
        //     // insert
        //     self.last = self.last.max(i);
        //     (true, i)
        // }
    }
    pub fn emplace(&mut self, transform: i32, d: T) -> i32 {
        let (b, id) = self.get_next_id();
        if b {
            unsafe {
                *self.valid[id as usize].get() = true;
            }
            *self.data[id as usize].lock() = (transform, d);
        } else {
            self.data.push(Mutex::new((transform, d)));
            self.valid.push(SyncUnsafeCell::new(true));
        }
        id
    }
    pub fn insert_multi<D: Fn() -> T>(&mut self, transforms: &[i32], d: D) {}
    fn _reduce_last(&mut self) {
        self.avail.commit();
        let mut id = self.last;
        while id >= 0 && !unsafe { *self.valid[id as usize].get() } {
            // not thread safe!
            id -= 1;
        }
        self.last = id;
        // let mut id = id;
        // if id == self.last.load(Ordering::Relaxed) {
        //     while id >= 0 && !unsafe { *self.valid[id as usize].get() } {
        //         // not thread safe!
        //         id -= 1;
        //     }
        //     self.last.store(id, Ordering::Relaxed); // multi thread safe?
        // }
    }
    pub fn _erase(&mut self, id: i32) {
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
        // let data = &self.data[0..last];
        // let valid = &self.valid[0..last];
        // data.par_iter().zip_eq(valid.par_iter()).for_each(|(d, v)| {
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
    fn erase(&mut self, i: i32) {
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
        self.emplace(t, def)
    }

    fn get_type(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn copy(&mut self, t: i32, i: i32) -> i32 {
        let p = self.data[i as usize].lock().1.clone();

        self.emplace(t, p)
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
        self.emplace(transform, d)
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
