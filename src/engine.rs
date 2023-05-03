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

use serde::{Deserialize, Serialize};
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
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{Device, Queue},
    memory::allocator::StandardMemoryAllocator,
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
    pub trans: &'a Transforms,
    pub physics: &'a crate::engine::physics::Physics,
    pub defer: &'a crate::engine::Defer,
    pub input: &'a crate::input::Input,
    pub model_manager: &'a parking_lot::Mutex<crate::ModelManager>,
    pub rendering: &'a RwLock<crate::RendererManager>,
    pub vk: Arc<VulkanManager>,
}

pub trait Component {
    // fn assign_transform(&mut self, t: Transform);
    fn init(&mut self, transform: Transform, id: i32, sys: &mut Sys) {}
    fn deinit(&mut self, transform: Transform, id: i32, _sys: &mut Sys) {}
    fn on_start(&mut self, transform: Transform, sys: &System) {} // TODO implement call
    fn on_destroy(&mut self, transform: Transform, sys: &System) {} // TODO implement call
    fn update(&mut self, transform: Transform, sys: &System) {}
    fn late_update(&mut self, transform: Transform, sys: &System) {}
    fn editor_update(&mut self, transform: Transform, sys: &System) {}
    fn on_render(&mut self, t_id: i32) -> Box<dyn Fn(&mut RenderJobData) -> ()> {
        Box::new(|rd: &mut RenderJobData| {})
    }
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
    fn update(
        &mut self,
        transforms: &Transforms,
        phys: &physics::Physics,
        lazy_maker: &Defer,
        input: &Input,
        modeling: &parking_lot::Mutex<crate::ModelManager>,
        rendering: &RwLock<crate::RendererManager>,
        vk: Arc<VulkanManager>,
    );
    fn late_update(
        &mut self,
        transforms: &Transforms,
        phys: &physics::Physics,
        lazy_maker: &Defer,
        input: &Input,
        modeling: &parking_lot::Mutex<crate::ModelManager>,
        rendering: &RwLock<crate::RendererManager>,
        vk: Arc<VulkanManager>,
    );
    fn editor_update(
        &mut self,
        transforms: &Transforms,
        phys: &physics::Physics,
        lazy_maker: &Defer,
        input: &Input,
        modeling: &parking_lot::Mutex<crate::ModelManager>,
        rendering: &RwLock<crate::RendererManager>,
        vk: Arc<VulkanManager>,
    );
    fn on_render(&mut self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData) -> ()>>);
    fn copy(&mut self, t: i32, i: i32) -> i32;
    fn erase(&mut self, i: i32);
    fn deinit(&self, transform: Transform, i: i32, sys: &mut Sys);
    fn init(&self, transform: Transform, i: i32, sys: &mut Sys);
    fn inspect(&self, transform: Transform, i: i32, ui: &mut egui::Ui, sys: &mut Sys);
    fn get_name(&self) -> &'static str;
    fn get_type(&self) -> TypeId;
    fn new_default(&mut self, t: i32) -> i32;
    fn serialize(&self, i: i32) -> Result<String, ron::Error>;
    fn deserialize(&mut self, transform: i32, d: String) -> i32;
    fn clear(&mut self);
}

// use pqueue::Queue;
pub struct Storage<T> {
    pub data: Vec<Mutex<(i32, T)>>,
    pub valid: Vec<AtomicBool>,
    avail: pqueue::Queue<Reverse<i32>>,
    extent: i32,
    has_update: bool,
    has_render: bool,
    has_late_update: bool,
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
        drop(self.data[id as usize].lock());
        self.valid[id as usize].store(false, Ordering::Relaxed);
        self.avail.push(Reverse(id));
    }
    // pub fn get(&self, i: &i32) -> &Mutex<T> {
    //     &self.data[*i as usize]
    // }
    pub fn new(has_update: bool, has_late_update: bool, has_render: bool) -> Storage<T> {
        Storage::<T> {
            data: Vec::new(),
            valid: Vec::new(),
            avail: pqueue::Queue::new(),
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
        lazy_maker: &Defer,
        input: &Input,
        modeling: &parking_lot::Mutex<crate::ModelManager>,
        rendering: &RwLock<crate::RendererManager>,
        vk: Arc<VulkanManager>,
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
            model_manager: modeling,
            rendering,
            vk,
        };
        // (0..self.data.len())

        // {
        //     let range = 0..self.valid.len();
        //     let v_iter = self.valid.iter();
        //     rayon::scope(|s| {
        //         for slice in range
        //             .into_iter()
        //             .zip(v_iter)
        //             .collect::<Vec<(usize, &AtomicBool)>>()
        //             .chunks(chunk_size)
        //         {
        //             s.spawn(move |_| {
        //                 for (_i, v) in slice.clone() {
        //                     if v.load(Ordering::Relaxed) {
        //                         let d = &self.data[*_i];
        //                         let mut d = d.lock();
        //                         let trans = Transform {
        //                             id: d.0,
        //                             transforms: &transforms,
        //                         };
        //                         d.1.update(trans, &sys);
        //                     }
        //                 }
        //             });
        //         }
        //     });
        // }

        self.data
            .par_iter()
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
    fn late_update(
        &mut self,
        transforms: &Transforms,
        physics: &physics::Physics,
        lazy_maker: &Defer,
        input: &Input,
        modeling: &parking_lot::Mutex<crate::ModelManager>,
        rendering: &RwLock<crate::RendererManager>,
        vk: Arc<VulkanManager>,
    ) {
        if !self.has_late_update {
            return;
        }
        let chunk_size = (self.data.len() / (64 * 64)).max(1);

        let sys = System {
            trans: &transforms,
            physics,
            defer: &lazy_maker,
            input,
            model_manager: modeling,
            rendering,
            vk,
        };

        self.data
            .par_iter()
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
                        d.1.late_update(trans, &sys);
                    }
                }
            });
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
        self.avail = pqueue::Queue::new();
        self.extent = 0;
        self.valid.clear();
    }

    fn deserialize(&mut self, transform: i32, d: String) -> i32 {
        let d: T = ron::from_str(&d).unwrap();
        self.emplace(transform, d)
    }

    fn on_render(&mut self, render_jobs: &mut Vec<Box<dyn Fn(&mut RenderJobData) -> ()>>) {
        if !self.has_render {
            return;
        }
        self.data
            .iter_mut()
            .zip(self.valid.iter())
            .enumerate()
            .for_each(|(_i, (d, v))| {
                if v.load(Ordering::Relaxed) {
                    let mut d = d.lock();
                    let t_id = d.0;
                    render_jobs.push(d.1.on_render(t_id));
                }
            });
    }

    fn editor_update(
        &mut self,
        transforms: &Transforms,
        phys: &physics::Physics,
        lazy_maker: &Defer,
        input: &Input,
        modeling: &parking_lot::Mutex<crate::ModelManager>,
        rendering: &RwLock<crate::RendererManager>,
        vk: Arc<VulkanManager>,
    ) {
        if !self.has_update {
            return;
        }
        let chunk_size = (self.data.len() / (64 * 64)).max(1);

        let sys = System {
            trans: &transforms,
            physics: phys,
            defer: &lazy_maker,
            input,
            model_manager: modeling,
            rendering,
            vk,
        };
        self.data
            .par_iter()
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
                        d.1.editor_update(trans, &sys);
                    }
                }
            });
    }
}

#[derive(Clone, Copy)]
pub struct GameObject {
    pub t: i32,
}

pub struct Sys {
    // pub device: Arc<Device>,
    // pub queue: Arc<Queue>,
    pub model_manager: Arc<parking_lot::Mutex<ModelManager>>,
    pub renderer_manager: Arc<RwLock<RendererManager>>,
    pub physics: Physics, // bombs: Vec<Option<Mutex<Bomb>>>,
    pub particles: Arc<ParticleCompute>,
    // pub memory_allocator: Arc<StandardMemoryAllocator>,
    // pub descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    // pub command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    pub vk: Arc<VulkanManager>,
}
// #[derive(Default)]
pub struct World {
    // pub(crate) device: Arc<Device>,
    pub(crate) entities: RwLock<Vec<Option<RwLock<HashMap<TypeId, i32>>>>>,
    pub(crate) transforms: RwLock<Transforms>,
    pub(crate) components:
        HashMap<TypeId, Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>>,
    pub(crate) components_names:
        HashMap<String, Arc<RwLock<Box<dyn StorageBase + 'static + Sync + Send>>>>,
    pub(crate) root: i32,
    pub(crate) sys: Arc<Mutex<Sys>>, // makers: Vec<Option<Mutex<Maker>>>,
}

#[allow(dead_code)]
impl World {
    pub fn new(
        modeling: Arc<parking_lot::Mutex<ModelManager>>,
        renderer_manager: Arc<RwLock<RendererManager>>,
        physics: Physics,
        particles: Arc<ParticleCompute>,
        vk: Arc<VulkanManager>,
    ) -> World {
        let trans = RwLock::new(Transforms::new());
        let root = trans.write().new_root();
        World {
            entities: RwLock::new(vec![None]),
            transforms: trans,
            components: HashMap::new(),
            components_names: HashMap::new(),
            root,
            sys: Arc::new(Mutex::new(Sys {
                model_manager: modeling,
                renderer_manager,
                physics,
                particles,
                vk: vk.clone(),
            })),
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
    pub fn get_transform(&self, t: i32) -> _Transform {
        let transforms = self.transforms.read();
        _Transform {
            position: transforms.get_position(t),
            rotation: transforms.get_rotation(t),
            scale: transforms.get_scale(t),
        }
    }
    fn copy_game_object_child(&self, t: i32, new_parent: i32) {
        let g = self.instantiate_with_transform_with_parent(new_parent, self.get_transform(t));
        let entities = self.entities.read();
        if let (Some(src_obj), Some(dest_obj)) = (&entities[t as usize], &entities[g.t as usize]) {
            let children: Vec<i32> = {
                let mut dest_obj = dest_obj.write();
                let transforms = self.transforms.read();
                for c in src_obj.read().iter() {
                    dest_obj.insert(
                        *c.0,
                        self.copy_component_id(transforms.get_transform(g.t), *c.0, *c.1),
                    );
                }
                let x = transforms.meta[t as usize]
                    .lock()
                    .children
                    .iter()
                    .map(|e| *e)
                    .collect();
                x
            };
            drop(entities);
            for c in children {
                self.copy_game_object_child(c, g.t);
            }
        } else {
            panic!("copy object not valid");
        }
    }
    pub fn copy_game_object(&self, t: i32) -> GameObject {
        let parent = { self.transforms.read().get_parent(t) };
        let g = self.instantiate_with_transform_with_parent(parent, self.get_transform(t));
        let entities = self.entities.read();
        if let (Some(src_obj), Some(dest_obj)) = (&entities[t as usize], &entities[g.t as usize]) {
            let children: Vec<i32> = {
                let mut dest_obj = dest_obj.write();
                let transforms = self.transforms.read();
                for c in src_obj.read().iter() {
                    dest_obj.insert(
                        *c.0,
                        self.copy_component_id(transforms.get_transform(g.t), *c.0, *c.1),
                    );
                }
                let x = transforms.meta[t as usize]
                    .lock()
                    .children
                    .iter()
                    .map(|e| *e)
                    .collect();
                x
            };
            drop(entities);
            for c in children {
                self.copy_game_object_child(c, g.t);
            }
        } else {
            panic!("copy object not valid");
        }
        g
    }
    fn copy_component_id(&self, t: Transform, key: TypeId, c_id: i32) -> i32 {
        if let Some(stor) = self.components.get(&key) {
            let mut stor = stor.write();
            let c = stor.copy(t.id, c_id);
            stor.init(t, c, &mut self.sys.lock());
            return c;
        } else {
            panic!("no component storage for key");
        }
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
    pub fn deserialize(&mut self, g: GameObject, key: String, s: String) {
        // d.assign_transform(g.t);

        if let Some(stor) = self.components_names.get(&key) {
            let mut stor = stor.write();
            let c_id = stor.deserialize(g.t, s);
            let trans = Transform {
                id: g.t,
                transforms: &self.transforms.read(),
            };
            stor.init(trans, c_id, &mut self.sys.lock());
            if let Some(ent_components) = &self.entities.read()[g.t as usize] {
                ent_components.write().insert(stor.get_type(), c_id);
            }
        } else {
            panic!("no type key?")
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
        self.components
            .insert(key.clone(), component_storage.clone());
        self.components_names.insert(
            component_storage.read().get_name().to_string(),
            component_storage.clone(),
        );

        // let component_storage = Arc::new(RwLock::new(Box::new(data)));
        // self.components
        //     .insert(key.clone(), component_storage.clone());
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
    ) -> Option<&Arc<RwLock<Box<dyn StorageBase + Send + Sync>>>> {
        let key: TypeId = TypeId::of::<T>();
        self.components.get(&key)
    }

    pub fn update(
        &mut self,
        // phys: &physics::Physics,
        lazy_maker: &Defer,
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
                sys.vk.clone(),
            );
        }
    }

    pub(crate) fn late_update(
        &mut self,
        // phys: &physics::Physics,
        lazy_maker: &Defer,
        input: &Input,
        // modeling: &Mutex<crate::ModelManager>,
        // rendering: &Mutex<crate::RendererManager>,
    ) {
        let transforms = self.transforms.read();
        let sys = self.sys.lock();
        for (_, stor) in &self.components {
            stor.write().late_update(
                &transforms,
                &sys.physics,
                &lazy_maker,
                &input,
                &sys.model_manager,
                &sys.renderer_manager,
                sys.vk.clone(),
            );
        }
    }
    pub(crate) fn update_cameras(&mut self) {
        let transforms = self.transforms.read();
        // let sys = self.sys.lock();
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
                if v.load(Ordering::Relaxed) {
                    let mut d = d.lock();
                    let id: i32 = d.0;
                    d.1._update(transforms.get_transform(id));
                }
            });
    }
    pub(crate) fn editor_update(
        &mut self,
        lazy_maker: &Defer,
        input: &Input,
    ) {
        let transforms = self.transforms.read();
        let sys = self.sys.lock();
        for (_, stor) in &self.components {
            stor.write().editor_update(
                &transforms,
                &sys.physics,
                &lazy_maker,
                &input,
                &sys.model_manager,
                &sys.renderer_manager,
                sys.vk.clone(),
            );
        }
    }
    pub fn render(&self) -> Vec<Box<dyn Fn(&mut RenderJobData) -> ()>> {
        // let transforms = self.transforms.read();
        // let sys = self.sys.lock();
        let mut render_jobs = vec![];
        for (_, stor) in &self.components {
            stor.write().on_render(&mut render_jobs);
        }
        render_jobs
    }

    pub fn clear(&mut self) {
        let mut sys = self.sys.lock();
        sys.renderer_manager.write().clear();
        sys.physics.clear();
        self.entities.write().clear();
        for a in &self.components {
            a.1.write().clear();
        }
        self.transforms.write().clear();
        self.root = self.transforms.write().new_root();
        self.entities.write().push(None);
    }
}
