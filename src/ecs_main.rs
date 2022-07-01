
#![feature(map_first_last)]

use components::*;
// #![feature(once_cell)]
use once_cell::sync::Lazy;

use crossbeam::atomic::AtomicCell;
use rayon::prelude::*;
use std::{
    any::{Any, TypeId},
    cell::RefCell,
    cmp::Reverse,
    collections::BinaryHeap,
    fmt::Debug,
    ops::{Deref, DerefMut},
    rc::Rc,
    sync::{Arc, Mutex},
};
use std::{collections::HashMap, sync::RwLock};

use hecs::World;
// use crate::components::new_component;
mod components;

#[derive(Debug)]
struct Dummy {
    pub create: bool,
}

impl Component for Dummy {
    fn update(&mut self) {
        if self.create {
            let x = new_component(Dummy { create: false });
            let key: TypeId = TypeId::of::<Dummy>();
            unsafe {

                COMPONENT_REGISTRY
                .deffered_init
                .read()
                .unwrap()
                .get(&key)
                .unwrap()
                .as_any()
                .downcast_ref::<DefferedInitVec<Dummy>>()
                .unwrap()
                .data
                .get(&x)
                .unwrap()
                .lock()
                .unwrap()
                .create = true;
            }
        }
        println!("dummy ")
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self as &mut dyn std::any::Any
    }
}

fn main() {
    // let x = RwLock::new(vec![Mutex::new(0)]);

    // for i in 1..1000 {
    //     x.write().unwrap().push(Mutex::new(i));
    // }

    // (0..1000).into_par_iter().for_each(|i| {
    //     let d = x.read().unwrap();
    //     let mut y = d[i].lock().unwrap();
    //     println!("{}", *y);
    //     *y += 1;
    //     if *y % 10 == 0 {
    //         x.write().unwrap().push(Mutex::new(*y));
    //     }
    // });


    let mut world = World::new();
    // Nearly any type can be used as a component with zero boilerplate
    let a = world.spawn((123, true, "abc"));
    let b = world.spawn((42, false));
    // Systems can be simple for loops
    for (id, (number, &flag)) in world.query_mut::<(&mut i32, &bool)>() {
    if flag { *number *= 2; }
    }
    // Random access is simple and safe
    assert_eq!(*world.get::<i32>(a).unwrap(), 246);
    assert_eq!(*world.get::<i32>(b).unwrap(), 42);
    

    // unsafe {
    //     COMPONENT_REGISTRY.register::<Dummy>();
    // }
    // for _ in 0..5 {
    //     new_component(Dummy { create: false });
    // }
    // unsafe {
    //     COMPONENT_REGISTRY.do_game_loop();
    //     println!("1");
    //     COMPONENT_REGISTRY.do_game_loop();
    //     println!("2");
    //     COMPONENT_REGISTRY.emplace::<Dummy>(Dummy { create: true });

    //     COMPONENT_REGISTRY.do_game_loop();
    //     println!("3");

    //     COMPONENT_REGISTRY.do_game_loop();
    //     println!("4");

    //     // COMPONENT_REGISTRY.erase::<Dummy>(3);

    //     COMPONENT_REGISTRY.do_game_loop();
    //     println!("5");
    // }
    // println!("");
}
