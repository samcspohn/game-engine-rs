

use serde::{Deserialize, Serialize};

use crate::engine::world::{transform::{_Transform, Transforms}, World};


#[derive(Serialize, Deserialize)]
struct SerGameObject {
    t: _Transform,
    c: Vec<(String, serde_yaml::Value)>,
    t_c: Vec<SerGameObject>,
}

fn serialize_c(t: i32, world: &World, transforms: &Transforms) -> SerGameObject {
    let mut g_o = SerGameObject {
        t: world.transforms.get(t).get_transform(), // set t
        c: vec![],
        t_c: vec![],
    };
    // set components
    let entities = world.entities.read();
    if let Some(ent) = entities[t as usize].lock().as_ref() {
        for c in ent.iter() {
            if let Some(stor) = &world.components.get(c.0) {
                let t_id: String = stor.read().get_name().to_string();
                // let t_id: String = format!("{:?}",c.0);
                g_o.c
                    .push((t_id, stor.read().serialize(*c.1)));
            }
        }
    }

    // set children
    let t = transforms.get(t);
    for c in t.get_meta().children.iter() {
        g_o.t_c.push(serialize_c(*c, world, transforms));
    }
    g_o
}

pub fn serialize(world: &World) {
    let root = world.root;
    let transforms = &world.transforms;
    // let t_r = Transform {
    //     id: root,
    //     transforms: &transforms,
    // };
    let t_r = transforms.get(root);
    let mut root = SerGameObject {
        t: _Transform::default(),
        c: vec![],
        t_c: vec![],
    };

    for c in t_r.get_meta().children.iter() {
        root.t_c.push(serialize_c(*c, world, &transforms));
    }
    std::fs::write("test.yaml", serde_yaml::to_string(&root).unwrap()).unwrap();
}

fn deserialize_c(parent: i32, sgo: SerGameObject, world: &mut World) {
    let g = world.instantiate_with_transform_with_parent(parent, sgo.t);
    for (typ, val) in sgo.c {
        // let id: TypeId = unsafe {std::mem::transmute(typ)};
        world.deserialize(g, typ, val);
    }
    for c in sgo.t_c {
        deserialize_c(g.t, c, world);
    }
}

pub fn deserialize(world: &mut World) {
    if let Ok(s) = std::fs::read_to_string("test.yaml") {
        let sgo: SerGameObject = serde_yaml::from_str(s.as_str()).unwrap();

        // std::fs::write("test.yaml", serde_yaml::to_string(&sgo).unwrap()).unwrap();

        world.clear();
        for c in sgo.t_c {
            deserialize_c(world.root, c, world);
        }
    }
}