#[path = "physics.rs"]
pub mod physics;
#[path = "transform.rs"]
pub mod transform;

use std::{
    any::{Any, TypeId},
    cmp::Reverse,
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use transform::{Transform, Transforms};

// use rand::prelude::*;
// use rapier3d::prelude::*;
use rayon::prelude::*;

use crossbeam::queue::SegQueue;

use parking_lot::RwLock;
use spin::Mutex;

use crate::{
    input::Input, inspectable::Inspectable, model::ModelManager, particles::ParticleCompute,
    renderer_component2::RendererManager,
};

use self::physics::Physics;

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
    pub fn do_defered(&self, wrld: &mut World) {
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

pub struct System<'a> {
    pub trans: &'a Transforms,
    pub physics: &'a crate::engine::physics::Physics,
    pub defer: &'a crate::engine::LazyMaker,
    pub input: &'a crate::input::Input,
    pub modeling: &'a parking_lot::Mutex<crate::ModelManager>,
    pub rendering: &'a RwLock<crate::RendererManager>,
}

pub trait Component {
    // fn assign_transform(&mut self, t: Transform);
    fn init(&mut self, transform: Transform, id: i32, sys: &mut Sys) {}
    fn deinit(&mut self, transform: Transform, id: i32, _sys: &mut Sys) {}
    fn update(&mut self, transform: Transform, sys: &System) {}
}

pub struct _Storage<T> {
    pub data: Vec<T>,
    pub valid: Vec<AtomicBool>,
    avail: pqueue::Queue<Reverse<i32>>,
    extent: i32,
}
impl<T: 'static> _Storage<T> {
    pub fn emplace(&mut self, d: T) -> i32 {
        match self.avail.pop() {
            Some(Reverse(i)) => {
                self.data[i as usize] = d;
                i
            }
            None => {
                self.data.push(d);
                self.extent += 1;
                self.extent as i32 - 1
            }
        }
    }
    pub fn erase(&mut self, id: i32) {
        // self.data[id as usize] = None;
        self.avail.push(Reverse(id));
    }
    pub fn get(&self, i: &i32) -> &T {
        &self.data[*i as usize]
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
    fn update(
        &mut self,
        transforms: &Transforms,
        phys: &physics::Physics,
        lazy_maker: &LazyMaker,
        input: &Input,
        modeling: &parking_lot::Mutex<crate::ModelManager>,
        rendering: &RwLock<crate::RendererManager>,
    );
    fn erase(&mut self, i: i32);
    fn deinit(&self, transform: Transform, i: i32, sys: &mut Sys);
    fn init(&self, transform: Transform, i: i32, sys: &mut Sys);
    fn inspect(&self, transform:Transform, i: i32, ui: &mut egui::Ui, sys: &mut Sys);
    fn get_name(&self) -> &'static str;
    fn get_hash(&self) -> TypeId;
    fn new_default(&mut self, t: i32) -> i32;
}

// use pqueue::Queue;
pub struct Storage<T> {
    pub data: Vec<Mutex<(i32, T)>>,
    pub valid: Vec<AtomicBool>,
    avail: pqueue::Queue<Reverse<i32>>,
    extent: i32,
    has_update: bool,
}
impl<T: 'static> Storage<T> {
    pub fn emplace(&mut self, transform: i32, d: T) -> i32 {
        match self.avail.pop() {
            Some(Reverse(i)) => {
                *self.data[i as usize].lock() = (transform, d);
                self.valid[i as usize].store(true, Ordering::Relaxed);
                i
            }
            None => {
                self.data.push(Mutex::new((transform, d)));
                self.valid.push(AtomicBool::new(true));
                self.extent += 1;
                self.extent as i32 - 1
            }
        }
    }
    pub fn erase(&mut self, id: i32) {
        // self.data[id as usize] = None;
        self.valid[id as usize].store(false, Ordering::Relaxed);
        self.avail.push(Reverse(id));
    }
    // pub fn get(&self, i: &i32) -> &Mutex<T> {
    //     &self.data[*i as usize]
    // }
    pub fn new(has_update: bool) -> Storage<T> {
        Storage::<T> {
            data: Vec::new(),
            valid: Vec::new(),
            avail: pqueue::Queue::new(),
            extent: 0,
            has_update,
        }
    }
}

impl<T: 'static + Component + Inspectable + Send + Sync + Default> StorageBase for Storage<T> {
    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self as &mut dyn Any
    }
    fn update(
        &mut self,
        transforms: &Transforms,
        physics: &physics::Physics,
        lazy_maker: &LazyMaker,
        input: &Input,
        modeling: &parking_lot::Mutex<crate::ModelManager>,
        rendering: &RwLock<crate::RendererManager>,
    ) {
        if !self.has_update {
            return;
        }
        let chunk_size = (self.data.len() / (64 * 64)).max(1);

        let sys = System {
            trans: &transforms,
            physics,
            defer: &lazy_maker,
            input,
            modeling,
            rendering,
        };
        // (0..self.data.len())
        self.data
            .par_iter_mut()
            .zip_eq(self.valid.par_iter())
            .enumerate()
            .chunks(chunk_size)
            .for_each(|slice| {
                for (_i, (d, v)) in slice {
                    if v.load(Ordering::Relaxed) {
                        let mut d = d.lock();
                        let trans = Transform {
                            id: d.0,
                            transforms: &transforms,
                        };
                        d.1.update(trans, &sys);
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
    fn deinit(&self, transform: Transform, i: i32, sys: &mut Sys) {
        self.data[i as usize].lock().1.deinit(transform, i, sys);
    }
    fn init(&self, transform: Transform, i: i32, sys: &mut Sys) {
        self.data[i as usize].lock().1.init(transform, i, sys);
    }
    fn inspect(&self, transform: Transform, i: i32, ui: &mut egui::Ui, sys: &mut Sys) {
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

    fn get_hash(&self) -> TypeId {
        TypeId::of::<T>()
    }
}

#[derive(Clone, Copy)]
pub struct GameObject {
    pub t: i32,
}

pub struct Sys {
    pub model_manager: Arc<parking_lot::Mutex<ModelManager>>,
    pub renderer_manager: Arc<RwLock<RendererManager>>,
    pub physics: Physics, // bombs: Vec<Option<Mutex<Bomb>>>,
    pub particles: Arc<ParticleCompute>,
}
// #[derive(Default)]
pub struct World {
    pub entities: RwLock<Vec<Option<RwLock<HashMap<TypeId, i32>>>>>,
    pub transforms: RwLock<Transforms>,
    pub components: HashMap<TypeId, RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>,
    root: i32,
    pub sys: Mutex<Sys>, // makers: Vec<Option<Mutex<Maker>>>,
}

#[allow(dead_code)]
impl World {
    pub fn new(
        modeling: Arc<parking_lot::Mutex<ModelManager>>,
        renderer_manager: Arc<RwLock<RendererManager>>,
        physics: Physics,
        particles: Arc<ParticleCompute>,
    ) -> World {
        let trans = RwLock::new(Transforms::new());
        let root = trans.write().new_root();
        World {
            entities: RwLock::new(vec![None]),
            transforms: trans,
            components: HashMap::new(),
            root,
            sys: Mutex::new(Sys {
                model_manager: modeling,
                renderer_manager,
                physics,
                particles,
            }),
        }
    }
    pub fn instantiate(&self) -> GameObject {
        let mut trans = self.transforms.write();
        let ret = GameObject {
            t: trans.new_transform(self.root),
        };
        {
            let entities = &mut self.entities.write();
            if ret.t as usize >= entities.len() {
                entities.push(Some(RwLock::new(HashMap::new())));
            } else {
                entities[ret.t as usize] = Some(RwLock::new(HashMap::new()));
            }
        }
        ret
    }
    pub fn instantiate_with_transform(&self, transform: transform::_Transform) -> GameObject {
        let mut trans = self.transforms.write();
        let ret = GameObject {
            t: trans.new_transform_with(self.root, transform),
        };
        {
            let entities = &mut self.entities.write();
            if ret.t as usize >= entities.len() {
                entities.push(Some(RwLock::new(HashMap::new())));
            } else {
                entities[ret.t as usize] = Some(RwLock::new(HashMap::new()));
            }
        }
        ret
    }
    pub fn instantiate_with_transform_with_parent(
        &self,
        parent: i32,
        transform: transform::_Transform,
    ) -> GameObject {
        let mut trans = self.transforms.write();
        let ret = GameObject {
            t: trans.new_transform_with(parent, transform),
        };
        {
            let entities = &mut self.entities.write();
            if ret.t as usize >= entities.len() {
                entities.push(Some(RwLock::new(HashMap::new())));
            } else {
                entities[ret.t as usize] = Some(RwLock::new(HashMap::new()));
            }
        }
        ret
    }
    pub fn add_component<T: 'static + Send + Sync + Component + Inspectable + Default>(
        &mut self,
        g: GameObject,
        mut d: T,
    ) {
        // d.assign_transform(g.t);
        let key: TypeId = TypeId::of::<T>();
        if let Some(stor) = self
            .components
            .get(&key)
            .unwrap()
            .write()
            .as_any_mut()
            .downcast_mut::<Storage<T>>()
        {
            let c_id = stor.emplace(g.t, d);
            let trans = Transform {
                id: g.t,
                transforms: &self.transforms.read(),
            };
            stor.init(trans, c_id, &mut self.sys.lock());
            if let Some(ent_components) = &self.entities.read()[g.t as usize] {
                ent_components.write().insert(key, c_id);
            }
        } else {
            panic!("no type key?")
        }
    }
    pub fn add_component_id(&mut self, g: GameObject, key: TypeId, c_id: i32) {
        // d.assign_transform(g.t);

        if let Some(ent_components) = &self.entities.read()[g.t as usize] {
            ent_components.write().insert(key, c_id);
            if let Some(stor) = self.components.get(&key) {
                let trans = Transform {
                    id: g.t,
                    transforms: &self.transforms.read(),
                };
                stor.write().init(trans, c_id, &mut self.sys.lock());
            }
        }
    }
    pub fn remove_component(&mut self, g: GameObject, key: TypeId, c_id: i32) {
        if let Some(ent_components) = &self.entities.read()[g.t as usize] {
            if let Some(stor) = self.components.get(&key) {
                let trans = Transform {
                    id: g.t,
                    transforms: &self.transforms.read(),
                };
                stor.write().deinit(trans, c_id, &mut self.sys.lock());
                stor.write().erase(c_id);
            }
            ent_components.write().remove(&key);
        }
    }
    pub fn register<T: 'static + Send + Sync + Component + Inspectable + Default>(
        &mut self,
        has_update: bool,
    ) {
        let key: TypeId = TypeId::of::<T>();
        let data = Storage::<T>::new(has_update);
        self.components
            .insert(key.clone(), RwLock::new(Box::new(data)));
    }
    pub fn delete(&mut self, g: GameObject) {
        // remove/deinit components
        let ent = &mut self.entities.write();
        if let Some(g_components) = &ent[g.t as usize] {
            for (t, id) in &*g_components.write() {
                let stor = &mut self.components.get(&t).unwrap().write();
                let trans = Transform {
                    id: g.t,
                    transforms: &self.transforms.read(),
                };
                stor.deinit(trans, *id, &mut self.sys.lock());
                stor.erase(*id);
            }
        }
        // remove entity
        ent[g.t as usize] = None; // todo make read()

        // remove transform
        self.transforms.write().remove(g.t);
    }
    // pub fn get_component<T: 'static + Send + Sync + Component, F>(&self, g: GameObject, f: F)
    // where
    //     F: FnOnce(&Mutex<T>),
    // {
    //     let key: TypeId = TypeId::of::<T>();

    //     if let Some(components) = &self.entities.read()[g.t.0 as usize] {
    //         if let Some(id) = &components.read().get(&key) {
    //             if let Some(stor_base) = &self.components.get(&key) {
    //                 if let Some(stor) = stor_base.write().as_any().downcast_ref::<Storage<T>>() {
    //                     f(stor.get(id));
    //                 }
    //             }
    //         }
    //     }
    // }
    pub fn get_components<T: 'static + Send + Sync + Component>(
        &self,
    ) -> Option<&RwLock<Box<dyn StorageBase + Send + Sync>>> {
        let key: TypeId = TypeId::of::<T>();
        self.components.get(&key)
    }

    pub fn update(
        &mut self,
        // phys: &physics::Physics,
        lazy_maker: &LazyMaker,
        input: &Input,
        // modeling: &Mutex<crate::ModelManager>,
        // rendering: &Mutex<crate::RendererManager>,
    ) {
        let transforms = self.transforms.read();
        let sys = self.sys.lock();
        for (_, stor) in &self.components {
            stor.write().update(
                &transforms,
                &sys.physics,
                &lazy_maker,
                &input,
                &sys.model_manager,
                &sys.renderer_manager,
            );
        }
    }
}
