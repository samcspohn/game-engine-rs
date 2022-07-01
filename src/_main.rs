use std::collections::HashMap;
// use std::sync::Arc;
// use std::mem;

trait Component {
    fn update(&mut self) {
        println!("update!")
    }
    fn late_update(&mut self) {
        println!("late_update!")
    }
}

struct Renderer {}

impl Component for Renderer {
    fn update(&mut self) {
        println!("render update!");
    }
}

struct Collider {}

impl Component for Collider {
    fn update(&mut self) {
        println!("collider update!");
    }
}

trait StorageBase {
    fn update(&mut self);
}

struct Storage<T> {
    data: Vec<T>,
}
impl<T:'static> Storage<T> {
    fn new() -> Storage<T> {
        Storage::<T> { data: Vec::new() }
    }

}

impl<T: 'static> StorageBase for Storage<T>
where
    T: Component,
{
    fn update(&mut self) {
        for x in self.data.iter_mut() {
            x.update();
            x.late_update();
        }
    }
}

struct Registry {
    components: HashMap<std::any::TypeId, Box<dyn StorageBase>>,
}

impl<'a> Registry {
    pub fn get_storage<T: 'static>(&'a mut self) -> &mut Storage<T> {
        let key: std::any::TypeId = std::any::TypeId::of::<T>();
        let b = self.components.get_mut(&key).unwrap();
        unsafe{
            std::mem::transmute::<&mut std::boxed::Box<dyn StorageBase>,&mut std::boxed::Box<Storage<T>>>(b)
        }
    }
    fn register<T: 'static + Component>(&mut self) {
        let key: std::any::TypeId = std::any::TypeId::of::<T>();
        let data: Box<dyn StorageBase> = Box::new(Storage::<T>::new());
        self.components.insert(key, data);
    }

    fn new() -> Registry {
        Registry {
            components: HashMap::new(),
        }
    }
    fn update(&mut self) {
        for (_, y) in self.components.iter_mut() {
            y.update();
        }
    }
}

fn main() {

    let mut registry = Registry::new();
    registry.register::<Renderer>();
    registry.register::<Collider>();

    {
        // registry.get_storage::<Renderer>().data.push(Renderer{});
        let renderers = registry.get_storage::<Renderer>();
        renderers.data.push(Renderer {});
        renderers.data.push(Renderer {});
        renderers.data.push(Renderer {});
        renderers.data.push(Renderer {});
        renderers.data.push(Renderer {});
        let colliders = registry.get_storage::<Collider>();
        colliders.data.push(Collider {});
        colliders.data.push(Collider {});
        colliders.data.push(Collider {});
        colliders.data.push(Collider {});
    }

    registry.update();
}
