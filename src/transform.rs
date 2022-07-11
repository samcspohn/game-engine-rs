use deepmesa::lists::{linkedlist::Node, LinkedList};
use force_send_sync::SendSync;
use glm::Vec3;
use nalgebra_glm as glm;
// use parking_lot::{Mutex, RwLock};
use spin::{Mutex,RwLock};
use std::{cmp::Reverse, collections::BinaryHeap};

struct TransformMeta {
    children: RwLock<SendSync<LinkedList<Transform>>>,
    parent: Transform,
    child_id: SendSync<Node<Transform>>,
}

impl TransformMeta {
    fn new() -> TransformMeta {
        unsafe {
            TransformMeta {
                children: RwLock::new(SendSync::new(LinkedList::new())),
                parent: Transform(-1),
                child_id: SendSync::new(Node::default()),
            }
        }
    }
}
pub struct Transforms {
    pub positions: Vec<Mutex<glm::Vec3>>,
    rotations: Vec<Mutex<glm::Quat>>,
    scales: Vec<Mutex<glm::Vec3>>,
    meta: Vec<Mutex<TransformMeta>>,
    avail: BinaryHeap<Reverse<i32>>,
    extent: i32,
}

#[derive(Clone, Copy)]
pub struct Transform(pub i32);

impl Transforms {
    pub fn new() -> Transforms {
        Transforms {
            positions: Vec::new(),
            rotations: Vec::new(),
            scales: Vec::new(),
            meta: Vec::new(),
            avail: BinaryHeap::new(),
            extent: 0,
        }
    }
    pub fn new_root(&mut self) -> Transform {
        match self.avail.pop() {
            Some(Reverse(i)) => {
                self.positions[i as usize] = Mutex::new(glm::vec3(0.0, 0.0, 0.0));
                self.rotations[i as usize] = Mutex::new(glm::quat(1.0, 0.0, 0.0, 0.0));
                self.scales[i as usize] = Mutex::new(glm::vec3(1.0, 1.0, 1.0));
                self.meta[i as usize] = Mutex::new(TransformMeta::new());
                Transform(i)
            }
            None => {
                self.positions.push(Mutex::new(glm::vec3(0.0, 0.0, 0.0)));
                self.rotations
                    .push(Mutex::new(glm::quat(1.0, 0.0, 0.0, 0.0)));
                self.scales.push(Mutex::new(glm::vec3(1.0, 1.0, 1.0)));
                self.meta.push(Mutex::new(TransformMeta::new()));
                self.extent += 1;
                Transform(self.extent as i32 - 1)
            }
        }
    }
    pub fn new_transform(&mut self, parent: Transform) -> Transform {
        let ret = match self.avail.pop() {
            Some(Reverse(i)) => {
                self.positions[i as usize] = Mutex::new(glm::vec3(0.0, 0.0, 0.0));
                self.rotations[i as usize] = Mutex::new(glm::quat(1.0, 0.0, 0.0, 0.0));
                self.scales[i as usize] = Mutex::new(glm::vec3(1.0, 1.0, 1.0));
                self.meta[i as usize] = Mutex::new(TransformMeta::new());
                Transform(i)
            }
            None => {
                self.positions.push(Mutex::new(glm::vec3(0.0, 0.0, 0.0)));
                self.rotations
                    .push(Mutex::new(glm::quat(1.0, 0.0, 0.0, 0.0)));
                self.scales.push(Mutex::new(glm::vec3(1.0, 1.0, 1.0)));
                self.meta.push(Mutex::new(TransformMeta::new()));
                self.extent += 1;
                Transform(self.extent as i32 - 1)
            }
        };
        let mut meta = self.meta[ret.0 as usize].lock();
        meta.parent = parent;
        unsafe {
            meta.child_id = SendSync::new(
                self.meta[parent.0 as usize]
                    .lock()
                    .children
                    .write()
                    .push_tail(ret),
            );
        }
        ret
    }
    pub fn remove(&mut self, t: Transform) {
        {
            let meta = self.meta[t.0 as usize].lock();
            self.meta[meta.parent.0 as usize]
                .lock()
                .children
                .write()
                .pop_node(&meta.child_id);
        }

        self.meta[t.0 as usize] = Mutex::new(TransformMeta::new());
        self.avail.push(Reverse(t.0));
    }

    pub fn _move(&self, t: Transform, v: Vec3) {
        *self.positions[t.0 as usize].lock() += v;
    }
    pub fn translate(&self, t: Transform, mut v: Vec3) {
        
        v = glm::quat_to_mat3(&self.rotations[t.0 as usize].lock()) * v;
        *self.positions[t.0 as usize].lock() += v;
    }
    pub fn get_position(&self, t: Transform) -> Vec3 {
        *self.positions[t.0 as usize].lock()
    }
    pub fn set_position(&self, t: Transform, v: Vec3) {
        *self.positions[t.0 as usize].lock() = v;
    }

    pub fn rotate() {}
}
