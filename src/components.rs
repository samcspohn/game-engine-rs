// #![feature(once_cell)]

use once_cell::sync::Lazy;
use rayon::prelude::*;
use std::{
    any::{Any, TypeId},
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap},
    fmt::Debug, cell::RefCell,
};
use std::{collections::HashMap};
// use parking_lot::{RwLock, Mutex};
use spin::{RwLock, Mutex};

// pub trait AToAny: 'static {
//     fn as_any(&self) -> &dyn Any;
//     fn as_any_mut(&mut self) -> &mut dyn Any;
// }

// impl<T: 'static> AToAny for T {
//     fn as_any(&self) -> &dyn Any {
//         self
//     }
//     fn as_any_mut(&mut self) -> &mut dyn Any {
//         self as &mut dyn Any
//     }
// }

pub trait DefferedInitBase {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    // fn push(&mut self) -> i32;
    fn erase(&mut self, i: i32);
}

pub struct DefferedInitVec<T: 'static + Debug + Send + Sync + Component> {
    pub data: BTreeMap<i32, Mutex<T>>,
    avail: BinaryHeap<Reverse<i32>>,
    deffered_extent: i32,
}

impl<T: 'static + Debug + Send + Sync + Component> DefferedInitBase for DefferedInitVec<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self as &mut dyn Any
    }
    fn erase(&mut self, i: i32) {
        self.avail.push(Reverse(i));
    }
}
impl<T: 'static + Debug + Send + Sync + Component> DefferedInitVec<T> {
    fn push(&mut self, a: T) -> i32 {
        match self.avail.pop() {
            Some(Reverse(i)) => {
                self.data.insert(i, Mutex::new(a));
                i
            }
            None => {
                self.data.insert(self.deffered_extent, Mutex::new(a));
                self.deffered_extent += 1;
                self.deffered_extent as i32 - 1
            }
        }
    }
}
trait StorageBase: Debug + Send {
    fn update(&mut self);
    fn late_update(&mut self);

    fn init_deffered(&mut self, dvec: &mut Box<dyn DefferedInitBase>);
    fn erase(&mut self, i: i32);
    fn print(&self);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

#[derive(Debug)]
pub struct Storage<T> {
    data: Vec<Mutex<T>>,
    valid: Vec<bool>,
    extent: i32,
}

pub trait Component {
    fn update(&mut self) {
        println!("update!")
    }
    fn late_update(&mut self) {
        println!("late_update!")
    }
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: std::fmt::Debug + std::marker::Send + std::marker::Sync + Component> Storage<T> {
    fn new() -> Storage<T> {
        Storage::<T> {
            data: vec![],
            valid: vec![],
            extent: 0,
        }
    }
    fn emplace(&mut self, i: i32, a: Mutex<T>) {
        if i as usize >= self.data.len() {
            self.data.push(a);
            self.valid.push(true);
            self.extent += 1;
        } else {
            self.valid[i as usize] = true;
            self.data[i as usize] = a;
        }
    }

    fn erase(&mut self, i: i32) {
        self.valid[i as usize] = false;
    }
    fn _get(&mut self, i: usize) -> &mut Mutex<T> {
        &mut self.data[i]
    }
    fn get(&mut self, i: i32) -> &mut Mutex<T> {
        self._get(i as usize)
    }
    fn get_v(&self, i: i32) -> bool {
        self.valid[i as usize]
    }
}

impl<T: 'static + std::fmt::Debug + Send + Sync + Component> StorageBase for Storage<T>
where
    T: Component,
{
    fn update(&mut self) {
        (0..self.extent as usize).into_par_iter().for_each(|i| {
            if self.valid[i] {
                self.data[i].lock().unwrap().update();
            }
        });
    }
    fn late_update(&mut self) {
        (0..self.extent as usize).into_par_iter().for_each(|i| {
            if self.valid[i] {
                self.data[i].lock().unwrap().late_update();
            }
        });
    }
    fn init_deffered(&mut self, dvec: &mut Box<dyn DefferedInitBase>) {
        if let Some(v) = dvec.as_any_mut().downcast_mut::<DefferedInitVec<T>>() {
            // v.data.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            // for (i, a) in v.data {
            //     // let (i, a) = v.data.pop().unwrap();
            //     self.emplace(i, a);
            // }
            // v.data.
            while !v.data.is_empty() {
                let (i, a) = v.data.pop_first().unwrap();
                self.emplace(i, a)
            }
            // v.data.clear();
        }
    }
    fn erase(&mut self, i: i32) {
        self.valid[i as usize] = false;
    }
    fn print(&self) {
        (0..self.extent as usize).into_iter().for_each(|i| {
            if self.valid[i] {
                print!("{:?}, ", self.data[i].lock().unwrap());
            }
        });
    }
    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self as &mut dyn Any
    }
}

pub struct Registry {
    pub components: RwLock<HashMap<TypeId, Box<dyn StorageBase + Sync>>>,
    pub deffered_init: RwLock<HashMap<TypeId, Box<dyn DefferedInitBase>>>,
}

impl<'a> Registry {
    pub fn register<T: 'static + Debug + Send + Sync + Component>(&mut self) {
        let key: TypeId = TypeId::of::<T>();
        let data: Box<dyn StorageBase + Sync + Send> = Box::new(Storage::<T>::new());
        let dv: Box<dyn DefferedInitBase + Sync + Send> = Box::new(DefferedInitVec::<T> {
            data: BTreeMap::new(),
            avail: BinaryHeap::new(),
            deffered_extent: 0,
        });
        self.components.write().unwrap().insert(key.clone(), data);
        self.deffered_init.write().unwrap().insert(key.clone(), dv);
    }

    pub fn new() -> Registry {
        Registry {
            components: RwLock::new(HashMap::new()),
            deffered_init: RwLock::new(HashMap::new()),
        }
    }

    pub fn do_game_loop(&mut self) {
        for (_, y) in self.components.write().unwrap().iter_mut() {
            y.update();
        }
        for (_, y) in self.components.write().unwrap().iter_mut() {
            y.late_update();
        }

        self.init_deffered();
    }

    pub fn print(&self) {
        for (_, y) in self.components.write().unwrap().iter_mut() {
            y.print();
        }
    }
    pub fn emplace<
        T: 'static + std::fmt::Debug + std::marker::Send + std::marker::Sync + Component,
    >(
        &mut self,
        a: T,
    ) -> i32 {
        let key: TypeId = TypeId::of::<T>();
        if let Some(def) = self
            .deffered_init
            .write()
            .unwrap()
            .get_mut(&key)
            .unwrap()
            .as_any_mut()
            .downcast_mut::<DefferedInitVec<T>>()
        {
            def.push(a)
        } else {
            -1
        }
    }
    fn init_deffered(&mut self) {
        for (x, y) in self.deffered_init.write().unwrap().iter_mut() {
            self.components
                .write()
                .unwrap()
                .get_mut(&x)
                .unwrap()
                .init_deffered(y);
        }
    }
    pub fn erase<
        T: 'static + std::fmt::Debug + std::marker::Send + std::marker::Sync + Component,
    >(
        &mut self,
        i: i32,
    ) {
        let key: TypeId = TypeId::of::<T>();
        self.components
            .write()
            .unwrap()
            .get_mut(&key)
            .unwrap()
            .erase(i);
        self.deffered_init
            .write()
            .unwrap()
            .get_mut(&key)
            .unwrap()
            .erase(i);
    }

    // pub fn get_def<
    //     T: 'static + std::fmt::Debug + std::marker::Send + std::marker::Sync + Component,
    // >(
    //     &mut self,
    //     i: i32,
    // ) -> RefCell<&Mutex<T>> {
    //     let key: TypeId = TypeId::of::<T>();

    //     let x = self.deffered_init
    //     .read()
    //     .unwrap()
    //     .get(&key)
    //     .unwrap();
        
    //         let y = x.as_any()
    //         .downcast_ref::<DefferedInitVec<T>>()
    //         .unwrap()
    //         .data
    //         .get(&i).unwrap();
        
    //         RefCell::new(y)
    // }
}

pub static mut COMPONENT_REGISTRY: Lazy<Registry> = Lazy::new(|| Registry::new());

pub fn new_component<T: 'static + Debug + Send + Sync + Component>(a: T) -> i32 {
    unsafe { COMPONENT_REGISTRY.emplace::<T>(a) }
}
