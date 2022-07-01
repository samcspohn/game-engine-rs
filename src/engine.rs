
#[path ="transform.rs"] pub mod transform;
#[path="physics.rs"] pub mod physics;
// use component_derive::component;


use transform::{Transforms, Transform};
use std::{
    any::{Any, TypeId},

    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};

// use rand::prelude::*;
use rapier3d::prelude::*;
use rayon::prelude::*;

use crossbeam::queue::SegQueue;
use parking_lot::{Mutex, RwLock};


pub struct LazyMaker {
    work: SegQueue<Box<dyn FnOnce(&mut World) + Send + Sync>>,
}

impl LazyMaker {
    pub fn append<T: 'static>(&self, f: T)
    where
        T: FnOnce(&mut World) + Send + Sync,
    {
        self.work.push(Box::new(f));
    }
    pub fn init(&self, wrld: &mut World) {
        while let Some(w) = self.work.pop() {
            w(wrld);
        }
    }
    pub fn new() -> LazyMaker {
        LazyMaker {
            work: SegQueue::new(),
        }
    }
}

pub trait Component {
    fn init(&mut self, t: Transform);
    fn update(&mut self, trans: &Transforms, sys: ( &physics::Physics, &LazyMaker));
}


pub trait StorageBase {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn update(
        &self,
        transforms: &Transforms,
        phys: &physics::Physics,
        lazy_maker: &LazyMaker,
    );
    fn erase(&mut self, i: i32);
}

pub struct Storage<T> {
    data: Vec<Option<Mutex<T>>>,
    avail: BinaryHeap<Reverse<i32>>,
    extent: i32,
}
impl<T: 'static + Component> Storage<T> {
    pub fn emplace(&mut self, d: T) -> i32 {
        match self.avail.pop() {
            Some(Reverse(i)) => {
                self.data[i as usize] = Some(Mutex::new(d));
                i
            }
            None => {
                self.data.push(Some(Mutex::new(d)));
                self.extent += 1;
                self.extent as i32 - 1
            }
        }
    }
    pub fn erase(&mut self, id: i32) {
        self.data[id as usize] = None;
        self.avail.push(Reverse(id));
    }
    pub fn get(&self, i: &i32) -> Option<&Mutex<T>> {
        if let Some(d) = &self.data[*i as usize] {
            return Some(d);
        }else{
            None
        }
    }
}

impl<T: 'static + Component + Send + Sync> StorageBase for Storage<T> {
    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self as &mut dyn Any
    }
    fn update(
        &self,
        transforms: &Transforms,
        physics: &physics::Physics,
        lazy_maker: &LazyMaker,
    ) {
        (0..self.data.len())
            .into_par_iter()
            .chunks(64 * 64)
            .for_each(|slice| {
                for i in slice {
                    if let Some(d) = &self.data[i] {
                        d.lock()
                            .update(&transforms, (&physics, &lazy_maker));
                    }
                }
            });
        // (0..self.data.len())
        // .into_par_iter().for_each(|i| {
        //     // for i in slice {
        //         if let Some(d) = &self.data[i] {
        //             d.lock().update((&poss, &collider_set, &query_pipeline, &lazy_maker));
        //         }
        //     // }
        // });
        // self.data.par_chunks(64*64).for_each(|x| {
        //     for i in x {
        //         if let Some(d) = i {
        //             d.lock().update((&poss, &collider_set, &query_pipeline, &lazy_maker));
        //         }
        //     }
        // });
    }
    fn erase(&mut self, i: i32) {
        self.erase(i);
    }
}

#[derive(Clone, Copy)]
pub struct GameObject {
    pub t: Transform,
}

// #[derive(Default)]
pub struct World {
    entities: RwLock<Vec<Option<RwLock<HashMap<TypeId, i32>>>>>,
    pub transforms: RwLock<Transforms>,
    components: HashMap<TypeId, RwLock<Box<dyn StorageBase + 'static + Sync>>>,
    root: Transform,
    // bombs: Vec<Option<Mutex<Bomb>>>,
    // makers: Vec<Option<Mutex<Maker>>>,
}

impl World {
    pub fn new() -> World {
        let trans = RwLock::new(Transforms::new());
        let root = trans.write().new_root();
        World {
            entities: RwLock::new(vec![None]),
            transforms: trans,
            components: HashMap::new(),
            root: root,
        }
    }
    pub fn instantiate(&self) -> GameObject {
        let mut trans = self.transforms.write();
        let ret = GameObject {
            t: trans.new_transform(self.root),
        };
        self.entities
            .write()
            .push(Some(RwLock::new(HashMap::new())));
        ret
    }
    pub fn add_component<T: 'static + Component>(&mut self, g: GameObject, mut d: T) {
        d.init(g.t);
        let key: TypeId = TypeId::of::<T>();
        if let Some(stor) = self
            .components
            .get_mut(&key)
            .unwrap()
            .write()
            .as_any_mut()
            .downcast_mut::<Storage<T>>()
        {
            stor.emplace(d);
        }
    }
    pub fn register<T: 'static + Send + Sync + Component>(&mut self) {
        let key: TypeId = TypeId::of::<T>();
        let data = Storage::<T> {
            data: Vec::new(),
            avail: BinaryHeap::new(),
            extent: 0,
        };
        self.components
            .insert(key.clone(), RwLock::new(Box::new(data)));
    }
    pub fn delete(&self, g: GameObject) {
        // remove/deinit components
        if let Some(g_components) = &self.entities.read()[g.t.0 as usize] {
            for (t, id) in &*g_components.write() {
                self.components.get(&t).unwrap().write().erase(*id);
            }
        }
        // remove entity
        self.entities.write()[g.t.0 as usize] = None;

        // remove transform
        self.transforms.write().remove(g.t);
    }
    pub fn get_component<T: 'static + Send + Sync + Component, F>(
        &self,
        g: GameObject,
        f: F

    ) where  F:FnOnce(Option<&Mutex<T>>) {
        let key: TypeId = TypeId::of::<T>();

        if let Some(components) = &self.entities.read()[g.t.0 as usize] {
            if let Some(id) = &components.read().get(&key) {
                if let Some(stor_base) = &self.components.get(&key) {
                    if let Some(stor) = stor_base.write().as_any().downcast_ref::<Storage<T>>() {
                        f(stor.get(id));

                    }
                }
            }
        }
    }

    pub fn update(&self,phys: &physics::Physics, lazy_maker: &LazyMaker){
        let transforms = self.transforms.read();

        for (_, stor) in &self.components {
            stor.read()
                .update(&transforms, &phys, &lazy_maker);
        }
    }
}