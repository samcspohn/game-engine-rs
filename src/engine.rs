#[path = "physics.rs"]
pub mod physics;
#[path = "transform.rs"]
pub mod transform;
// use component_derive::component;

use std::{
    any::{Any, TypeId},
    cmp::Reverse,
    collections::{BinaryHeap, HashMap}, sync::Arc,
};
use tobj::Model;
use transform::{Transform, Transforms};

// use rand::prelude::*;
// use rapier3d::prelude::*;
use rayon::prelude::*;

use crossbeam::queue::SegQueue;
use parking_lot::{Mutex, RwLock};
// use spin::{Mutex, RwLock};

use crate::{input::Input, model::ModelManager, renderer_component::RendererManager};

use self::physics::Physics;

pub struct LazyMaker {
    work: SegQueue<
        Box<
            dyn FnOnce(&mut World)
                + Send
                + Sync,
        >,
    >,
}

impl LazyMaker {
    pub fn append<T: 'static>(&self, f: T)
    where
        T: FnOnce(&mut World)
            + Send
            + Sync,
    {
        self.work.push(Box::new(f));
    }
    pub fn init(
        &self,
        wrld: &mut World,
    ) {
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
    pub modeling: &'a Mutex<crate::ModelManager>,
    pub rendering: &'a Mutex<crate::RendererManager>,
}

pub trait Component {
    fn init(&mut self, t: Transform, sys: &mut Sys);
    fn deinit(&mut self, t: Transform, sys: &mut Sys) {}
    fn update(&mut self, sys: &System);
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
        modeling: &Mutex<crate::ModelManager>,
        rendering: &Mutex<crate::RendererManager>,
    );
    fn erase(&mut self, i: i32);
    fn deinit(&mut self, i: i32, t: Transform, sys: &mut Sys);
}

pub struct Storage<T> {
    pub data: Vec<Option<T>>,
    avail: BinaryHeap<Reverse<i32>>,
    extent: i32,
    has_update: bool,
}
impl<T: 'static> Storage<T> {
    pub fn emplace(&mut self, d: T) -> i32 {
        match self.avail.pop() {
            Some(Reverse(i)) => {
                self.data[i as usize] = Some(d);
                i
            }
            None => {
                self.data.push(Some(d));
                self.extent += 1;
                self.extent as i32 - 1
            }
        }
    }
    pub fn erase(&mut self, id: i32) {
        self.data[id as usize] = None;
        self.avail.push(Reverse(id));
    }
    pub fn get(&self, i: &i32) -> Option<&T> {
        if let Some(d) = &self.data[*i as usize] {
            return Some(d);
        } else {
            None
        }
    }
    pub fn new(has_update: bool) -> Storage<T> {
        Storage::<T> {
            data: Vec::new(),
            avail: BinaryHeap::new(),
            extent: 0,
            has_update,
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
        &mut self,
        transforms: &Transforms,
        physics: &physics::Physics,
        lazy_maker: &LazyMaker,
        input: &Input,
        modeling: &Mutex<crate::ModelManager>,
        rendering: &Mutex<crate::RendererManager>,
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
            .enumerate()
            .chunks(chunk_size)
            .for_each(|slice| {
                for (i, o) in slice {
                    if let Some(d) = o {
                        d.update(&sys);
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
    fn deinit(&mut self, i: i32, t: Transform, sys: &mut Sys) {
        self.data[i as usize].as_mut().unwrap().deinit(t, sys);
    }
}

#[derive(Clone, Copy)]
pub struct GameObject {
    pub t: Transform,
}

pub struct Sys {
    pub model_manager: Arc<Mutex<ModelManager>>,
    pub renderer_manager: Mutex<RendererManager>,
    pub physics: Physics, // bombs: Vec<Option<Mutex<Bomb>>>,
}
// #[derive(Default)]
pub struct World {
    entities: RwLock<Vec<Option<RwLock<HashMap<TypeId, i32>>>>>,
    pub transforms: RwLock<Transforms>,
    components: HashMap<TypeId, RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>,
    root: Transform,
    pub sys: Sys
    
                      // makers: Vec<Option<Mutex<Maker>>>,
}

#[allow(dead_code)]
impl World {
    pub fn new(
        modeling: Arc<Mutex<ModelManager>>,
        renderer_manager: Mutex<RendererManager>,
        physics: Physics,
    ) -> World {
        let trans = RwLock::new(Transforms::new());
        let root = trans.write().new_root();
        World {
            entities: RwLock::new(vec![None]),
            transforms: trans,
            components: HashMap::new(),
            root,
            sys: Sys {
                model_manager: modeling,
                renderer_manager,
                physics,
            }
        }
    }
    pub fn instantiate(&self) -> GameObject {
        let mut trans = self.transforms.write();
        let ret = GameObject {
            t: trans.new_transform(self.root),
        };
        {
            let entities = &mut self.entities.write();
            if ret.t.0 as usize >= entities.len() {
                entities.push(Some(RwLock::new(HashMap::new())));
            } else {
                entities[ret.t.0 as usize] = Some(RwLock::new(HashMap::new()));
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
            if ret.t.0 as usize >= entities.len() {
                entities.push(Some(RwLock::new(HashMap::new())));
            } else {
                entities[ret.t.0 as usize] = Some(RwLock::new(HashMap::new()));
            }
        }
        ret
    }
    pub fn add_component<T: 'static + Component>(&mut self, g: GameObject, mut d: T) {
        d.init(g.t, &mut self.sys);
        let key: TypeId = TypeId::of::<T>();
        if let Some(stor) = self
            .components
            .get(&key)
            .unwrap()
            .write()
            .as_any_mut()
            .downcast_mut::<Storage<T>>()
        {
            let c_id = stor.emplace(d);
            if let Some(ent_components) = &self.entities.read()[g.t.0 as usize] {
                ent_components.write().insert(key, c_id);
            }
        } else {
            panic!("no type key?")
        }
    }
    pub fn register<T: 'static + Send + Sync + Component>(&mut self, has_update: bool) {
        let key: TypeId = TypeId::of::<T>();
        let data = Storage::<T> {
            data: Vec::new(),
            avail: BinaryHeap::new(),
            extent: 0,
            has_update,
        };
        self.components
            .insert(key.clone(), RwLock::new(Box::new(data)));
    }
    pub fn delete(&mut self, g: GameObject) {
        // remove/deinit components
        let ent = &mut self.entities.write();
        if let Some(g_components) = &ent[g.t.0 as usize] {
            for (t, id) in &*g_components.write() {
                let stor = &mut self.components.get(&t).unwrap().write();
                stor.deinit(*id, g.t, &mut self.sys);
                stor.erase(*id);
            }
        }
        // remove entity
        ent[g.t.0 as usize] = None; // todo make read()

        // remove transform
        self.transforms.write().remove(g.t);
    }
    pub fn get_component<T: 'static + Send + Sync + Component, F>(&self, g: GameObject, f: F)
    where
        F: FnOnce(Option<&T>),
    {
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

    pub fn update(
        &mut self,
        // phys: &physics::Physics,
        lazy_maker: &LazyMaker,
        input: &Input,
        // modeling: &Mutex<crate::ModelManager>,
        // rendering: &Mutex<crate::RendererManager>,
    ) {
        let transforms = self.transforms.read();

        for (_, stor) in &self.components {
            stor.write()
                .update(&transforms, &self.sys.physics, &lazy_maker, &input, &self.sys.model_manager, &self.sys.renderer_manager);
        }
    }
}
