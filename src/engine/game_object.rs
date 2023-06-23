use serde::{Deserialize, Serialize};

use crate::{editor::inspectable::Inspectable};

use super::{
    component::{Component, _ComponentID},
    world::{World, transform::_Transform}, storage::{Storage, StorageBase},
};

pub struct GameObjectBuilder {}
impl GameObjectBuilder {
    pub fn with_com<T: Component>(&mut self, d: T) {}
}

pub(crate) struct _GameObjectParBuilder {
    pub(crate) count: i32,
    pub(crate) chunk: i32,
    pub(crate) parent: i32,
    pub(crate) t_func: Option<Box<dyn Fn() -> _Transform + Send + Sync>>,
    pub(crate) comp_funcs: Vec<Box<dyn Fn(&mut World, &Vec<i32>) + Send + Sync>>,
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
    pub fn new(parent: i32, count: i32, chunk: i32, world: &'a World) -> Self {
        GameObjectParBuilder {
            world,
            count,
            chunk,
            parent: parent,
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
            + _ComponentID
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
                // let key: TypeId = TypeId::of::<T>();
                let key = T::ID;
                if let Some(stor) = world.components.get(&key) {
                    let mut stor_lock = stor.write();
                    let mut src = stor_lock.as_mut();
                    // let src = stor_lock.as_mut();
                    let stor: &mut &mut Storage<T> = unsafe { std::mem::transmute(&mut src) };
                    // if key == TypeId::of::<ParticleEmitter>() {}
                    for g in t_ {
                        let c_id = (*stor).emplace(*g, f());
                        let trans = world.transforms.get(*g);
                        (*stor).init(&trans, c_id, &mut world.sys);
                        if let Some(ent_components) =
                            world.entities.read()[*g as usize].lock().as_mut()
                        {
                            ent_components.insert(key.clone(), c_id);
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