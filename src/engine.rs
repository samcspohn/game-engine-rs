#[path = "physics.rs"]
pub mod physics;
#[path = "transform.rs"]
pub mod transform;

use std::{
    any::{Any, TypeId},
    cmp::Reverse,
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicI32, Ordering},
        Arc,
    },
};

use serde::{Deserialize, Serialize};
use sync_unsafe_cell::SyncUnsafeCell;
use transform::{Transform, Transforms};

// use rand::prelude::*;
// use rapier3d::prelude::*;
use rayon::prelude::*;

use crossbeam::queue::SegQueue;

use parking_lot::Mutex;
use parking_lot::RwLock;
// use spin::Mutex;
use vulkano::{
    buffer::DeviceLocalBuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        PrimaryAutoCommandBuffer,
    },
    pipeline::graphics::viewport::Viewport,
};

use crate::{
    camera::Camera, input::Input, inspectable::Inspectable, model::ModelManager,
    particles::ParticleCompute, renderer::RenderPipeline, renderer_component2::RendererManager,
    vulkan_manager::VulkanManager,
};

use self::{physics::Physics, transform::_Transform};

// macro_rules! component {
//     () => {
//         #[derive(Default,Clone,Deserialize, Serialize)]
//     };
// }

// RenderJobData {
//     builder: &mut builder,
//     transforms: transform_compute.transform.clone(),
//     mvp: transform_compute.mvp.clone(),
//     view: &view,
//     proj: &proj,
//     pipeline: &rend,
//     viewport: &viewport,
//     texture_manager: &texture_manager,
//     vk: vk.clone(),
// };

pub struct RenderJobData<'a> {
    pub builder: &'a mut AutoCommandBufferBuilder<
        PrimaryAutoCommandBuffer,
        Arc<StandardCommandBufferAllocator>,
    >,
    pub transforms: Arc<DeviceLocalBuffer<[crate::transform_compute::cs::ty::transform]>>,
    pub mvp: Arc<DeviceLocalBuffer<[crate::transform_compute::cs::ty::MVP]>>,
    pub view: &'a nalgebra_glm::Mat4,
    pub proj: &'a nalgebra_glm::Mat4,
    pub pipeline: &'a RenderPipeline,
    pub viewport: &'a Viewport,
    pub texture_manager: &'a parking_lot::Mutex<crate::TextureManager>,
    pub vk: Arc<crate::vulkan_manager::VulkanManager>,
}
pub struct Defer {
    work: SegQueue<Box<dyn FnOnce(&mut World) + Send + Sync>>,
}

impl Defer {
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
    pub fn new() -> Defer {
        Defer {
            work: SegQueue::new(),
        }
    }
}

pub struct System<'a> {
    pub physics: &'a crate::engine::physics::Physics,
    pub defer: &'a crate::engine::Defer,
    pub input: &'a crate::input::Input,
    pub model_manager: &'a parking_lot::Mutex<crate::ModelManager>,
    pub rendering: &'a RwLock<crate::RendererManager>,
    pub vk: Arc<VulkanManager>,
}

pub trait Component {
    // fn assign_transform(&mut self, t: Transform);
    fn init(&mut self, _transform: &Transform, _id: i32, _sys: &Sys) {}
    fn deinit(&mut self, _transform: &Transform, _id: i32, _sys: &Sys) {}
    fn on_start(&mut self, _transform: &Transform, _sys: &System) {} // TODO implement call
    fn on_destroy(&mut self, _transform: &Transform, _sys: &System) {} // TODO implement call
    fn update(&mut self, _transform: &Transform, _sys: &System, world: &World) {}
    fn late_update(&mut self, _transform: &Transform, _sys: &System) {}
    fn editor_update(&mut self, _transform: &Transform, _sys: &System) {}
    fn on_render(&mut self, _t_id: i32) -> Box<dyn Fn(&mut RenderJobData)> {
        Box::new(|_rd: &mut RenderJobData| {})
    }
    // fn as_any(&self) -> &dyn Any;
}

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
        // self.data[id as usize] = None;
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
    fn update(&mut self, transforms: &Transforms, sys: &Sys, input: &Input, world: &World);
    fn late_update(&mut self, transforms: &Transforms, sys: &Sys, input: &Input);
    fn editor_update(&mut self, transforms: &Transforms, sys: &Sys, input: &Input);
    fn on_render(&mut self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData)>>);
    fn copy(&mut self, t: i32, i: i32) -> i32;
    fn erase(&self, i: i32);
    fn deinit(&self, transform: &Transform, i: i32, sys: &Sys);
    fn init(&self, transform: &Transform, i: i32, sys: &Sys);
    fn inspect(&self, transform: &Transform, i: i32, ui: &mut egui::Ui, sys: &Sys);
    fn get_name(&self) -> &'static str;
    fn get_type(&self) -> TypeId;
    fn new_default(&mut self, t: i32) -> i32;
    // fn new(
    //     &mut self,
    //     t: i32,
    //     f: &Box<dyn Fn() -> (impl Component) + 'static + Sync + Send>,
    // ) -> i32;
    fn serialize(&self, i: i32) -> Result<String, ron::Error>;
    fn deserialize(&mut self, transform: i32, d: String) -> i32;
    fn clear(&mut self);
}

// use pqueue::Queue;
pub struct Storage<T> {
    pub data: Vec<Mutex<(i32, T)>>,
    pub valid: Vec<SyncUnsafeCell<bool>>,
    avail: AtomicI32,
    // avail: rudac::heap::FibonacciHeap<i32>,
    // avail: pqueue::Queue<Reverse<i32>>,
    extent: i32,
    has_update: bool,
    has_render: bool,
    has_late_update: bool,
}
impl<T: 'static> Storage<T> {
    pub fn emplace(&mut self, transform: i32, d: T) -> i32 {
        let i = self.avail.load(Ordering::Relaxed);
        if i < self.extent {
            *self.data[i as usize].lock() = (transform, d);
            unsafe {
                *self.valid[i as usize].get() = true;
            }
            let mut _i = i;
            // self.avail += 1;
            while _i < self.extent && unsafe { *self.valid[_i as usize].get() } {
                // find next open slot
                _i += 1;
            }
            self.avail.store(_i, Ordering::Relaxed);
            return i;
        } else {
            self.data.push(Mutex::new((transform, d)));
            self.valid.push(SyncUnsafeCell::new(true));
            self.extent += 1;
            self.avail.store(self.extent, Ordering::Relaxed);
            return self.extent - 1;
        }
    }
    pub fn erase(&self, id: i32) {
        // self.data[id as usize] = None;
        self.avail.fetch_min(id, Ordering::Relaxed);
        drop(&*self.data[id as usize].lock());
        unsafe {
            *self.valid[id as usize].get() = false;
        }
    }
    // pub fn get(&self, i: &i32) -> &Mutex<T> {
    //     &self.data[*i as usize]
    // }
    pub fn new(has_update: bool, has_late_update: bool, has_render: bool) -> Storage<T> {
        Storage::<T> {
            data: Vec::new(),
            valid: Vec::new(),
            avail: AtomicI32::new(0),
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
            + Inspectable
            + Send
            + Sync
            + Default
            + Clone
            + Serialize
            + for<'a> Deserialize<'a>,
    > StorageBase for Storage<T>
{
    // fn new(
    //     &mut self,
    //     t: i32,
    //     f: &Box<dyn Fn(&mut dyn StorageBase) + 'static + Sync + Send>,
    // ) -> i32 {
    //     let a = f();
    //     self.emplace(t, a)
    // }
    fn as_any(&self) -> &dyn Any {
        self as &dyn Any
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self as &mut dyn Any
    }
    fn update(&mut self, transforms: &Transforms, sys: &Sys, input: &Input, world: &World) {
        if !self.has_update {
            return;
        }
        let chunk_size = (self.data.len() / (64 * 64)).max(1);

        let sys = System {
            // trans: transforms,
            physics: &sys.physics.lock(),
            defer: &sys.defer,
            input,
            model_manager: &sys.model_manager,
            rendering: &sys.renderer_manager,
            vk: sys.vk.clone(),
        };

        self.data
            .par_iter()
            .zip_eq(self.valid.par_iter())
            // .enumerate()
            .chunks(chunk_size)
            .for_each(|slice| {
                for (d, v) in slice {
                    if unsafe { *v.get() } {
                        let mut d = d.lock();
                        let trans = transforms.get(d.0);
                        d.1.update(&trans, &sys, world);
                    }
                }
            });
    }
    fn late_update(&mut self, transforms: &Transforms, sys: &Sys, input: &Input) {
        if !self.has_late_update {
            return;
        }
        let chunk_size = (self.data.len() / (64 * 64)).max(1);

        let sys = System {
            // trans: transforms,
            physics: &sys.physics.lock(),
            defer: &sys.defer,
            input,
            model_manager: &sys.model_manager,
            rendering: &sys.renderer_manager,
            vk: sys.vk.clone(),
        };
        self.data
            .par_iter()
            .zip_eq(self.valid.par_iter())
            .enumerate()
            .chunks(chunk_size)
            .for_each(|slice| {
                for (_i, (d, v)) in slice {
                    if unsafe { *v.get() } {
                        let mut d = d.lock();
                        let trans = transforms.get(d.0);
                        d.1.late_update(&trans, &sys);
                    }
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

    fn serialize(&self, i: i32) -> Result<String, ron::Error> {
        ron::to_string(&self.data[i as usize].lock().1)
    }
    fn clear(&mut self) {
        self.data.clear();
        self.avail = AtomicI32::new(0);
        self.extent = 0;
        self.valid.clear();
    }

    fn deserialize(&mut self, transform: i32, d: String) -> i32 {
        let d: T = ron::from_str(&d).unwrap();
        self.emplace(transform, d)
    }

    fn on_render(&mut self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData)>>) {
        if !self.has_render {
            return;
        }
        self.data
            .iter_mut()
            .zip(self.valid.iter())
            .enumerate()
            .for_each(|(_i, (d, v))| {
                if unsafe { *v.get() } {
                    let mut d = d.lock();
                    let t_id = d.0;
                    render_jobs.push(d.1.on_render(t_id));
                }
            });
    }

    fn editor_update(&mut self, transforms: &Transforms, sys: &Sys, input: &Input) {
        if !self.has_update {
            return;
        }
        let chunk_size = (self.data.len() / (64 * 64)).max(1);

        let sys = System {
            // trans: transforms,
            physics: &sys.physics.lock(),
            defer: &sys.defer,
            input,
            model_manager: &sys.model_manager,
            rendering: &sys.renderer_manager,
            vk: sys.vk.clone(),
        };
        self.data
            .par_iter()
            .zip_eq(self.valid.par_iter())
            // .enumerate()
            // .chunks(chunk_size)
            .for_each(|(d, v)| {
                // for  (d, v) in slice {
                if unsafe { *v.get() } {
                    let mut d = d.lock();
                    let trans = transforms.get(d.0);
                    d.1.editor_update(&trans, &sys);
                }
                // }
            });
    }
}

#[derive(Clone, Copy)]
pub struct GameObject {
    pub t: i32,
}

pub struct Sys {
    pub model_manager: Arc<parking_lot::Mutex<ModelManager>>,
    pub renderer_manager: Arc<RwLock<RendererManager>>,
    pub physics: Arc<Mutex<Physics>>,
    pub particles: Arc<ParticleCompute>,
    pub vk: Arc<VulkanManager>,
    pub defer: Defer,
}
// #[derive(Default)]
pub struct World {
    // pub(crate) device: Arc<Device>,
    pub(crate) entities: RwLock<Vec<Mutex<Option<HashMap<TypeId, i32>>>>>,
    pub(crate) transforms: Transforms,
    pub(crate) components:
        HashMap<TypeId, Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>>,
    pub(crate) components_names:
        HashMap<String, Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>>,
    pub(crate) root: i32,
    pub(crate) sys: Sys, // makers: Vec<Option<Mutex<Maker>>>,
    to_destroy: Arc<Mutex<Vec<i32>>>,
    to_instantiate: Arc<Mutex<Vec<_GameObjectParBuilder>>>,
    // defer: SegQueue<Box<dyn FnOnce(&mut World) + Send + Sync>>,
}

#[allow(dead_code)]
impl World {
    pub fn new(
        modeling: Arc<parking_lot::Mutex<ModelManager>>,
        renderer_manager: Arc<RwLock<RendererManager>>,
        physics: Arc<Mutex<Physics>>,
        particles: Arc<ParticleCompute>,
        vk: Arc<VulkanManager>,
        defer: Defer,
    ) -> World {
        let mut trans = Transforms::new();
        let root = trans.new_root();
        World {
            entities: RwLock::new(vec![Mutex::new(None)]),
            transforms: trans,
            components: HashMap::new(),
            components_names: HashMap::new(),
            root,
            sys: Sys {
                model_manager: modeling,
                renderer_manager,
                physics,
                particles,
                vk: vk,
                defer,
            },
            to_destroy: Arc::new(Mutex::new(Vec::new())),
            to_instantiate: Arc::new(Mutex::new(Vec::new())),
            // defer: SegQueue::new(),
        }
    }
    pub fn defer_instantiate(&mut self) {
        let mut v = Vec::new();
        let mut y = self.to_instantiate.lock();
        std::mem::swap(&mut v, &mut y);
        drop(y);
        for a in v.into_iter() {
            if let Some(t_func) = a.t_func {
                let t = self
                    .transforms
                    .multi_transform_with(a.parent, a.count, t_func);

                let mut ent = self.entities.write();

                for i in &t {
                    if (*i as usize) < ent.len() {
                        *ent[*i as usize].lock() = Some(HashMap::new());
                    } else {
                        break;
                    }
                }
                if let Some(last) = t.last() {
                    while ent.len() <= *last as usize {
                        ent.push(Mutex::new(Some(HashMap::new())));
                    }
                }
                drop(ent);

                for b in a.comp_funcs.iter() {
                    b(self, &t);
                    // let key = b.0;
                    // if let Some(c) = self.components.get(&key) {
                    //     let mut stor = c.write();
                    //     for t in &t {
                    //         let t = *t;
                    //         let c_id = b.1(stor.as_mut(), t);
                    //         // let c_id = stor.new(t, &b.1);
                    //         let trans = self.transforms.get(t);
                    //         stor.init(&trans, c_id, &mut self.sys.lock());
                    //         if let Some(ent_components) = ent[t as usize].lock().as_mut() {
                    //             ent_components.insert(key, c_id);
                    //         }
                    //     }
                    // } else {
                    //     panic!("no type key?")
                    // }
                }
            }
        }
    }
    pub fn instantiate_many(
        &self,
        count: i32,
        chunk: i32,
        parent: &Transform,
    ) -> GameObjectParBuilder {
        GameObjectParBuilder::new(parent, count, chunk, &self)
    }
    pub fn instantiate(&mut self) -> GameObject {
        let ret = GameObject {
            t: self.transforms.new_transform(self.root),
        };
        {
            let entities = &mut self.entities.write();
            if ret.t as usize >= entities.len() {
                entities.push(Mutex::new(Some(HashMap::new())));
            } else {
                entities[ret.t as usize] = Mutex::new(Some(HashMap::new()));
            }
        }
        ret
    }
    pub fn instantiate_with_transform(&mut self, transform: transform::_Transform) -> GameObject {
        let ret = GameObject {
            t: self.transforms.new_transform_with(self.root, transform),
        };
        {
            let entities = &mut self.entities.write();
            if ret.t as usize >= entities.len() {
                entities.push(Mutex::new(Some(HashMap::new())));
            } else {
                entities[ret.t as usize] = Mutex::new(Some(HashMap::new()));
            }
        }
        ret
    }
    // pub fn get_transform(&self, t: i32) -> _Transform {
    //     let transforms = &self.transforms;
    //     _Transform {
    //         position: transforms.get_position(t),
    //         rotation: transforms.get_rotation(t),
    //         scale: transforms.get_scale(t),
    //     }
    // }
    fn copy_game_object_child(&mut self, t: i32, new_parent: i32) {
        let tr = self.transforms.get(t).get_transform();
        let g = self.instantiate_with_transform_with_parent(new_parent, tr);
        let entities = self.entities.read();
        let src = entities[t as usize].lock();
        let mut dest = entities[g.t as usize].lock();
        if let (Some(src_obj), Some(dest_obj)) = (&mut src.as_ref(), &mut dest.as_mut()) {
            let children: Vec<i32> = {
                // let mut dest_obj = dest_obj;
                for c in src_obj.iter() {
                    dest_obj.insert(
                        *c.0,
                        self.copy_component_id(&self.transforms.get(g.t), *c.0, *c.1),
                    );
                }
                let x = self
                    .transforms
                    .get(t)
                    .get_meta()
                    .children
                    .iter()
                    .copied()
                    .collect();
                x
            };
            drop(src);
            drop(dest);
            drop(entities);
            for c in children {
                self.copy_game_object_child(c, g.t);
            }
        } else {
            panic!("copy object not valid");
        }
    }
    pub fn copy_game_object(&mut self, t: i32) -> GameObject {
        let trans = self.transforms.get(t);
        let parent = { trans.get_parent().id };
        let tr = trans.get_transform();
        drop(trans);
        let g = self.instantiate_with_transform_with_parent(parent, tr);
        let entities = self.entities.read();
        let src = entities[t as usize].lock();
        let mut dest = entities[g.t as usize].lock();
        if let (Some(src_obj), Some(dest_obj)) = (&mut src.as_ref(), &mut dest.as_mut()) {
            let children: Vec<i32> = {
                // let mut dest_obj = dest_obj.write();
                for c in src_obj.iter() {
                    dest_obj.insert(
                        *c.0,
                        self.copy_component_id(&self.transforms.get(g.t), *c.0, *c.1),
                    );
                }
                let x = self
                    .transforms
                    .get(t)
                    .get_meta()
                    .children
                    .iter()
                    .copied()
                    .collect();
                x
            };
            drop(src);
            drop(dest);
            drop(entities);
            for c in children {
                self.copy_game_object_child(c, g.t);
            }
        } else {
            panic!("copy object not valid");
        }
        g
    }
    fn copy_component_id(&self, t: &Transform, key: TypeId, c_id: i32) -> i32 {
        if let Some(stor) = self.components.get(&key) {
            let mut stor = stor.write();
            let c = stor.copy(t.id, c_id);
            stor.init(&t, c, &self.sys);
            c
        } else {
            panic!("no component storage for key");
        }
    }
    pub fn instantiate_with_transform_with_parent(
        &mut self,
        parent: i32,
        transform: transform::_Transform,
    ) -> GameObject {
        // let mut trans = self.transforms.;
        let ret = GameObject {
            t: self.transforms.new_transform_with(parent, transform),
        };
        {
            let entities = &mut self.entities.write();
            if ret.t as usize >= entities.len() {
                entities.push(Mutex::new(Some(HashMap::new())));
            } else {
                entities[ret.t as usize] = Mutex::new(Some(HashMap::new()));
            }
        }
        ret
    }
    pub fn add_component<
        T: 'static
            + Send
            + Sync
            + Component
            + Inspectable
            + Default
            + Clone
            + Serialize
            + for<'a> Deserialize<'a>,
    >(
        &mut self,
        g: GameObject,
        d: T,
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
            let trans = self.transforms.get(g.t);
            stor.init(&trans, c_id, &self.sys);
            if let Some(ent_components) = self.entities.read()[g.t as usize].lock().as_mut() {
                ent_components.insert(key, c_id);
            }
        } else {
            panic!("no type key?")
        }
    }
    pub fn add_component_id(&mut self, g: GameObject, key: TypeId, c_id: i32) {
        // d.assign_transform(g.t);

        if let Some(ent_components) = self.entities.read()[g.t as usize].lock().as_mut() {
            ent_components.insert(key, c_id);
            if let Some(stor) = self.components.get(&key) {
                let trans = self.transforms.get(g.t);
                stor.write().init(&trans, c_id, &self.sys);
            }
        }
    }
    pub fn deserialize(&mut self, g: GameObject, key: String, s: String) {
        // d.assign_transform(g.t);

        if let Some(stor) = self.components_names.get(&key) {
            let mut stor = stor.write();
            let c_id = stor.deserialize(g.t, s);
            let trans = self.transforms.get(g.t);
            stor.init(&trans, c_id, &self.sys);
            if let Some(ent_components) = self.entities.read()[g.t as usize].lock().as_mut() {
                ent_components.insert(stor.get_type(), c_id);
            }
        } else {
            panic!("no type key?")
        }
    }
    pub fn remove_component(&mut self, g: GameObject, key: TypeId, c_id: i32) {
        if let Some(ent_components) = self.entities.read()[g.t as usize].lock().as_mut() {
            if let Some(stor) = self.components.get(&key) {
                let trans = self.transforms.get(g.t);
                stor.write().deinit(&trans, c_id, &self.sys);
                stor.write().erase(c_id);
            }
            ent_components.remove(&key);
        }
    }
    pub fn register<
        T: 'static
            + Send
            + Sync
            + Component
            + Inspectable
            + Default
            + Clone
            + Serialize
            + for<'a> Deserialize<'a>,
    >(
        &mut self,
        has_update: bool,
        has_late_update: bool,
        has_render: bool,
    ) {
        let key: TypeId = TypeId::of::<T>();
        // let c = T::default();
        // let has_render = T::on_render != &Component::on_render;
        let data = Storage::<T>::new(has_update, has_late_update, has_render);
        let component_storage: Arc<RwLock<Box<dyn StorageBase + Send + Sync + 'static>>> =
            Arc::new(RwLock::new(Box::new(data)));
        self.components.insert(key, component_storage.clone());
        self.components_names.insert(
            component_storage.read().get_name().to_string(),
            component_storage.clone(),
        );

        // let component_storage = Arc::new(RwLock::new(Box::new(data)));
        // self.components
        //     .insert(key.clone(), component_storage.clone());
    }
    pub fn destroy(&self, g: i32) {
        self.to_destroy.lock().push(g);
    }
    pub(crate) fn _destroy(&mut self) {
        let ent = self.entities.write();
        // let sys = self.sys.lock();

        self.to_destroy.lock().par_iter().for_each(|t| {
            let g = *t;
            let mut ent = ent[g as usize].lock();
            if let Some(g_components) = ent.as_mut() {
                let trans = self.transforms.get(g);
                for (t, id) in g_components.iter() {
                    let stor = &mut self.components.get(t).unwrap().write();

                    stor.deinit(&trans, *id, &self.sys);
                    stor.erase(*id);
                }
                // remove entity
                *ent = None; // todo make read()
                
                // remove transform
                self.transforms.remove(trans);
            }
        });
        self.to_destroy.lock().clear();
        // remove/deinit components
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
    ) -> Option<&Arc<RwLock<Box<dyn StorageBase + Send + Sync>>>> {
        let key: TypeId = TypeId::of::<T>();
        self.components.get(&key)
    }

    pub fn do_defered(&mut self) {
        while let Some(w) = self.sys.defer.work.pop() {
            w(self);
        }
        // self.sys.defer.do_defered(self);
    }
    pub fn update(&self, input: &Input) {
        // let sys = self.sys.lock();
        for (_, stor) in &self.components {
            stor.write().update(
                &self.transforms,
                &self.sys,
                input,
                &self,
            );
        }
    }

    pub(crate) fn late_update(&self, input: &Input) {
        // let sys = self.sys.lock();
        for (_, stor) in &self.components {
            stor.write().late_update(&self.transforms, &self.sys, input);
        }
    }
    pub(crate) fn update_cameras(&mut self) {
        let camera_components = self.get_components::<Camera>().unwrap().read();
        let camera_storage = camera_components
            .as_any()
            .downcast_ref::<Storage<Camera>>()
            .unwrap();
        camera_storage
            .valid
            .iter()
            .zip(camera_storage.data.iter())
            .for_each(|(v, d)| {
                if unsafe { *v.get() } {
                    let mut d = d.lock();
                    let id: i32 = d.0;
                    d.1._update(&self.transforms.get(id));
                }
            });
    }
    pub(crate) fn editor_update(&mut self, input: &Input) {
        // let sys = self.sys.lock();
        for (_, stor) in &self.components {
            stor.write().editor_update(
                &self.transforms,
                &self.sys,
                input,
            );
        }
    }
    pub fn render(&self) -> Vec<Box<dyn Fn(&mut RenderJobData)>> {
        // let transforms = self.transforms.read();
        // let sys = self.sys.lock();
        let mut render_jobs = vec![];
        for (_, stor) in &self.components {
            stor.write().on_render(&mut render_jobs);
        }
        render_jobs
    }

    pub fn clear(&mut self) {
        // let mut sys = self.sys.lock();
        self.sys.renderer_manager.write().clear();
        self.sys.physics.lock().clear();
        self.entities.write().clear();
        for a in &self.components {
            a.1.write().clear();
        }
        self.transforms.clear();
        self.root = self.transforms.new_root();
        self.entities.write().push(Mutex::new(None));
    }
}

pub struct GameObjectBuilder {}
impl GameObjectBuilder {
    pub fn with_com<T: Component>(&mut self, d: T) {}
}

struct _GameObjectParBuilder {
    count: i32,
    chunk: i32,
    parent: i32,
    t_func: Option<Box<dyn Fn() -> _Transform + Send + Sync>>,
    comp_funcs: Vec<Box<dyn Fn(&mut World, &Vec<i32>) + Send + Sync>>,
}

impl _GameObjectParBuilder {
    fn from(g: GameObjectParBuilder) -> Self {
        Self {
            count: g.count,
            chunk: g.chunk,
            parent: g.parent,
            t_func: g.transform_func,
            comp_funcs: g.comp_funcs,
        }
    }
}
pub struct GameObjectParBuilder<'a> {
    world: &'a World,
    count: i32,
    chunk: i32,
    parent: i32,
    transform_func: Option<Box<dyn Fn() -> _Transform + Send + Sync>>,
    comp_funcs: Vec<Box<dyn Fn(&mut World, &Vec<i32>) + Send + Sync>>,
}
impl<'a> GameObjectParBuilder<'a> {
    pub fn new(parent: &Transform, count: i32, chunk: i32, world: &'a World) -> Self {
        GameObjectParBuilder {
            world,
            count,
            chunk,
            parent: parent.id,
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
        T: 'static
            + Send
            + Sync
            + Component
            + Inspectable
            + Default
            + Clone
            + Serialize
            + for<'b> Deserialize<'b>,
    >(
        mut self,
        f: &'static (dyn Fn() -> T + Send + Sync),
    ) -> Self {
        self.comp_funcs
            .push(Box::new(|world: &mut World, t_: &Vec<i32>| {
                let key: TypeId = TypeId::of::<T>();
                if let Some(stor) = world
                    .components
                    .get(&key)
                    .unwrap()
                    .write()
                    .as_any_mut()
                    .downcast_mut::<Storage<T>>()
                {
                    for g in t_ {
                        let c_id = stor.emplace(*g, f());
                        let trans = world.transforms.get(*g);
                        stor.init(&trans, c_id, &mut world.sys);
                        if let Some(ent_components) =
                            world.entities.read()[*g as usize].lock().as_mut()
                        {
                            ent_components.insert(key, c_id);
                        }
                    }
                } else {
                    panic!("no type key?")
                }
            }));
        self
    }
    pub fn build(mut self) {
        // if let Some(t_func) = self.transform_func {
        self.world
            .to_instantiate
            .lock()
            .push(_GameObjectParBuilder::from(self));
        // }
    }
}
