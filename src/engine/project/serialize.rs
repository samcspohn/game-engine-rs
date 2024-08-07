use std::collections::HashMap;

use segvec::SegVec;
use serde::{Deserialize, Serialize};

use crate::engine::world::{
    entity,
    transform::{Transforms, _Transform, TRANSFORM_MAP},
    World,
};
struct SerTransform {
    id: i32,
    t: _Transform,
}
#[derive(Serialize, Deserialize)]
struct SerGameObject {
    t: (i32, _Transform),
    c: Vec<(String, serde_yaml::Value)>,
    t_c: Vec<SerGameObject>,
}
#[derive(Serialize, Deserialize)]
struct SerGameObject2 {
    t: _Transform,
    c: Vec<(String, serde_yaml::Value)>,
    t_c: Vec<SerGameObject2>,
}

fn serialize_c(t: i32, world: &World, transforms: &Transforms) -> SerGameObject {
    let trans = world.transforms.get(t).unwrap();
    let mut g_o = SerGameObject {
        t: (trans.id, trans.get_transform()), // set t
        c: vec![],
        t_c: vec![],
    };
    let ent = trans.entity();
    for c in ent.components.iter() {
        if let Some(stor) = &world.components.get(c.0) {
            let t_id: String = stor.1.read().get_name().to_string();
            // let t_id: String = format!("{:?}",c.0);

            match c.1 {
                entity::Components::Id(id) => {
                    g_o.c.push((t_id, stor.1.read().serialize(*id)));
                }
                entity::Components::V(v) => {
                    for id in v {
                        g_o.c.push((t_id.clone(), stor.1.read().serialize(*id)));
                    }
                }
            }
        }
    }

    // set children
    // let t = transforms.get(t);
    for c in trans.get_meta().children.iter() {
        g_o.t_c.push(serialize_c(*c, world, transforms));
    }
    g_o
}

pub fn serialize(world: &World, scene_file: &str) {
    let root = world.root;
    let transforms = &world.transforms;
    // let t_r = Transform {
    //     id: root,
    //     transforms: &transforms,
    // };
    let t_r = transforms.get(root).unwrap();
    let mut root = SerGameObject {
        t: (0, _Transform::default()),
        c: vec![],
        t_c: vec![],
    };

    for c in t_r.get_meta().children.iter() {
        root.t_c.push(serialize_c(*c, world, &transforms));
    }
    std::fs::write(scene_file, serde_yaml::to_string(&root).unwrap()).unwrap();
}

pub fn serialize_new(scene_file: &str) {
    let mut root = SerGameObject {
        t: (0, _Transform::default()),
        c: vec![],
        t_c: vec![],
    };
    std::fs::write(scene_file, serde_yaml::to_string(&root).unwrap()).unwrap();
}

fn deserialize_c<'a>(
    parent: i32,
    sgo: SerGameObject,
    world: &'a mut World,
    defer: &mut SegVec<Box<dyn Fn(&mut World)>>,
) {
    let g = world.create_with_transform_with_parent(parent, sgo.t.1);
    unsafe { &mut *TRANSFORM_MAP }.insert(sgo.t.0, g);
    // let c = sgo.c.clone();
    defer.push(Box::new(move |world: &mut World| {
        for (typ, val) in &sgo.c {
            // let id: TypeId = unsafe {std::mem::transmute(typ)};
            world.deserialize(g, typ, val);
        }
    }));
    for c in sgo.t_c {
        deserialize_c(g, c, world, defer);
    }
}

pub fn deserialize(world: &mut World, scene_file: &str) {
    if let Ok(s) = std::fs::read_to_string(scene_file) {
        unsafe {
            (*TRANSFORM_MAP).clear();
            (*TRANSFORM_MAP).insert(-1, -1);
            (*TRANSFORM_MAP).insert(0, 0);
        }
        let sgo: SerGameObject = serde_yaml::from_str(s.as_str()).unwrap();
        world.clear();
        let mut defer: SegVec<Box<dyn Fn(&mut World)>> = SegVec::new();
        for c in sgo.t_c {
            deserialize_c(world.root, c, world, &mut defer);
        }
        for a in defer {
            a(world);
        }
    }
}
