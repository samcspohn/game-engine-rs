use std::{sync::atomic::{AtomicBool, Ordering, AtomicI32}, cmp::Reverse, any::{TypeId, Any}};

use parking_lot::Mutex;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use sync_unsafe_cell::SyncUnsafeCell;

use crate::{editor::inspectable::Inspectable};

use super::{input::Input, RenderJobData, component::{System, Component, _ComponentID}, world::{World, Sys, transform::{Transforms, Transform}}};


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
    fn erase(&self, i: i32);
    fn deinit(&self, transform: &Transform, i: i32, sys: &Sys);
    fn init(&self, transform: &Transform, i: i32, sys: &Sys);
    fn inspect(&self, transform: &Transform, i: i32, ui: &mut egui::Ui, sys: &Sys);
    fn get_name(&self) -> &'static str;
    fn get_id(&self) -> u64;
    fn get_type(&self) -> TypeId;
    fn new_default(&mut self, t: i32) -> i32;
    // fn new(
    //     &mut self,
    //     t: i32,
    //     f: &Box<dyn Fn() -> (impl Component) + 'static + Sync + Send>,
    // ) -> i32;
    fn serialize(&self, i: i32) -> serde_yaml::Value;
    fn deserialize(&mut self, transform: i32, d: serde_yaml::Value) -> i32;
    fn clear(&mut self);
}

// use pqueue::Queue;
#[repr(C)]
pub struct Storage<T> {
    pub data: Vec<Mutex<(i32, T)>>,
    pub valid: Vec<SyncUnsafeCell<bool>>,
    avail: AtomicI32,
    last: AtomicI32,
    extent: AtomicI32,
    has_update: bool,
    has_render: bool,
    has_late_update: bool,
}
impl<T: 'static> Storage<T> {
    pub fn get_next_id(&self) -> (bool, i32) {
        let i = self.avail.load(Ordering::Relaxed);
        // self.count.fetch_add(1, Ordering::Relaxed);
        let extent = self.extent.load(Ordering::Relaxed);
        if i < extent {
            unsafe {
                *self.valid[i as usize].get() = true;
            }
            let mut _i = i;
            while _i < extent && unsafe { *self.valid[_i as usize].get() } {
                // find next open slot
                _i += 1;
            }
            self.avail.store(_i, Ordering::Relaxed);
            self.last.fetch_max(_i, Ordering::Relaxed);
            return (true, i);
        } else {
            let extent = extent + 1;
            self.extent.store(extent, Ordering::Relaxed);
            self.avail.store(extent, Ordering::Relaxed);
            self.last.store(extent - 1, Ordering::Relaxed);
            return (false, extent - 1);
        }
    }
    pub fn emplace(&mut self, transform: i32, d: T) -> i32 {
        let (b, id) = self.get_next_id();
        if b {
            *self.data[id as usize].lock() = (transform, d);
        } else {
            self.data.push(Mutex::new((transform, d)));
            self.valid.push(SyncUnsafeCell::new(true));
        }
        id
    }
    pub fn insert_multi<D: Fn() -> T>(&mut self, transforms: &[i32], d: D) {}
    fn reduce_last(&self, id: i32) {
        let mut id = id;
        if id == self.last.load(Ordering::Relaxed) {
            while id >= 0 && !unsafe { *self.valid[id as usize].get() } {
                // not thread safe!
                id -= 1;
            }
            self.last.store(id, Ordering::Relaxed); // multi thread safe?
        }
    }
    pub fn erase(&self, id: i32) {
        // self.data[id as usize] = None;
        self.avail.fetch_min(id, Ordering::Relaxed);
        // drop(&self.data[id as usize].lock().1);
        unsafe {
            *self.valid[id as usize].get() = false;
        }
        self.reduce_last(id);
    }
    // pub fn get(&self, i: &i32) -> &Mutex<T> {
    //     &self.data[*i as usize]
    // }
    pub fn new(has_update: bool, has_late_update: bool, has_render: bool) -> Storage<T> {
        Storage::<T> {
            data: Vec::new(),
            valid: Vec::new(),
            avail: AtomicI32::new(0),
            last: AtomicI32::new(-1),
            extent: AtomicI32::new(0),
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
        let last = (self.last.load(Ordering::Relaxed) + 1) as usize;
        let data = &self.data[0..last];
        let valid = &self.valid[0..last];
        data.par_iter().zip_eq(valid.par_iter()).for_each(|(d, v)| {
            if unsafe { *v.get() } {
                let mut d = d.lock();
                let trans = transforms.get(d.0);
                d.1.update(&trans, &sys, world);
            }
        });
    }
    fn late_update(&mut self, transforms: &Transforms, sys: &System) {
        if !self.has_late_update {
            return;
        }
        let last = (self.last.load(Ordering::Relaxed) + 1) as usize;
        let data = &self.data[0..last];
        let valid = &self.valid[0..last];
        data.par_iter().zip_eq(valid.par_iter()).for_each(|(d, v)| {
            if unsafe { *v.get() } {
                let mut d = d.lock();
                let trans = transforms.get(d.0);
                d.1.late_update(&trans, &sys);
            }
        });
    }
    fn erase(&self, i: i32) {
        self.erase(i);
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
    fn clear(&mut self) {
        *self = Self::new(self.has_update, self.has_late_update, self.has_render);
    }

    fn deserialize(&mut self, transform: i32, d: serde_yaml::Value) -> i32 {
        let d: T = serde_yaml::from_value(d).unwrap();
        self.emplace(transform, d)
    }

    fn on_render(&mut self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData)>>) {
        if !self.has_render {
            return;
        }
        let last = (self.last.load(Ordering::Relaxed) + 1) as usize;
        let data = &self.data[0..last];
        let valid = &self.valid[0..last];
        data.iter().zip(valid.iter()).for_each(|(d, v)| {
            if unsafe { *v.get() } {
                let mut d = d.lock();
                let t_id = d.0;
                render_jobs.push(d.1.on_render(t_id));
            }
        });
    }

    fn editor_update(&mut self, transforms: &Transforms, sys: &System, input: &Input) {
        if !self.has_update {
            return;
        }

        let last = (self.last.load(Ordering::Relaxed) + 1) as usize;
        let data = &self.data[0..last];
        let valid = &self.valid[0..last];
        data.par_iter().zip_eq(valid.par_iter()).for_each(|(d, v)| {
            if unsafe { *v.get() } {
                let mut d = d.lock();
                let trans = transforms.get(d.0);
                d.1.editor_update(&trans, &sys);
            }
            // }
        });
    }
    fn get_id(&self) -> u64 {
        T::ID
    }
}