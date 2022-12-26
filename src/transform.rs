use core::panic;
use deepmesa::lists::{linkedlist::Node, LinkedList};
use force_send_sync::SendSync;
use glm::{Mat3, Quat, Vec3};
use nalgebra_glm as glm;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
// use spin::{Mutex,RwLock};
use puffin_egui::puffin;
use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

pub struct TransformMeta {
    pub children: RwLock<SendSync<LinkedList<i32>>>,
    parent: i32,
    child_id: SendSync<Node<i32>>,
}

impl TransformMeta {
    fn new() -> TransformMeta {
        unsafe {
            TransformMeta {
                children: RwLock::new(SendSync::new(LinkedList::new())),
                parent: -1,
                child_id: SendSync::new(Node::default()),
            }
        }
    }
}
pub struct Transforms {
    pub positions: Vec<Mutex<glm::Vec3>>,
    pub rotations: Vec<Mutex<glm::Quat>>,
    pub scales: Vec<Mutex<glm::Vec3>>,
    meta: Vec<Mutex<TransformMeta>>,
    avail: BinaryHeap<Reverse<i32>>,
    updates: Vec<[AtomicBool; 3]>,
    extent: i32,
}

#[derive(Clone, Copy)]
pub struct _Transform {
    pub position: glm::Vec3,
    pub rotation: glm::Quat,
    pub scale: glm::Vec3,
}
impl Default for _Transform {
    fn default() -> Self {
        _Transform {
            position: glm::vec3(0., 0., 0.),
            rotation: glm::quat(1.0, 0.0, 0.0, 0.0),
            scale: glm::vec3(1.0, 1.0, 1.0),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Transform<'a> {
    pub id: i32,
    pub transforms: &'a Transforms,
}

#[allow(dead_code)]
impl<'a> Transform<'a> {
    pub fn forward(&self) -> glm::Vec3 {
        glm::quat_to_mat3(&*self.transforms.rotations[self.id as usize].lock()) * glm::Vec3::z()
    }
    pub fn right(&self) -> glm::Vec3 {
        glm::quat_to_mat3(&*self.transforms.rotations[self.id as usize].lock()) * glm::Vec3::x()
    }
    pub fn up(&self) -> glm::Vec3 {
        glm::quat_to_mat3(&*self.transforms.rotations[self.id as usize].lock()) * glm::Vec3::y()
    }

    pub fn _move(&self, v: Vec3) {
        self.transforms._move(self.id, v);
    }
    // fn move_child(&self,  v: Vec3) {
    //     self._move(t, v);
    //     for child in self.meta[t as usize].lock().children.read().iter() {
    //         self.move_child(*child, v);
    //     }
    // }
    pub fn translate(&self, mut v: Vec3) {
        self.transforms.translate(self.id, v);
    }
    pub fn get_position(&self) -> Vec3 {
        self.transforms.get_position(self.id)
    }
    pub fn set_position(&self, v: Vec3) {
        self.transforms.set_position(self.id, v);
    }
    pub fn get_rotation(&self) -> Quat {
        self.transforms.get_rotation(self.id)
    }
    pub fn set_rotation(&self, r: Quat) {
        self.transforms.set_rotation(self.id, r);
    }
    // fn set_rotation_child(&self, tc: i32, rot: &Mat3, pos: &Vec3) {
    //     let mut rotat = self.rotations[tc as usize].lock();
    //     let mut posi = self.positions[tc as usize].lock();

    //     *posi = pos + rot * (*posi - pos);
    //     self.u_pos(tc);
    //     *rotat = glm::mat3_to_quat(&rot) * *rotat;
    //     self.u_rot(tc);
    //     for child in self.meta[tc as usize].lock().children.read().iter() {
    //         self.set_rotation_child(*child, &rot, &pos)
    //     }
    // }
    pub fn get_scale(&self) -> Vec3 {
        self.transforms.get_scale(self.id)
    }
    pub fn set_scale(&self, s: Vec3) {
        self.transforms.set_scale(self.id, s);
    }
    pub fn scale(&self, s: Vec3) {
        self.transforms.scale(self.id, s);
    }
    // fn scale_child(&self,  p: &Vec3, s: &Vec3) {
    //     let mut scl = self.scales[t as usize].lock();
    //     let mut posi = self.positions[t as usize].lock();

    //     *posi = mul_vec3(&(*posi - p), s) + p;
    //     self.u_pos(t);
    //     *scl = mul_vec3(s, &*scl);
    //     self.u_scl(t);
    //     for child in self.meta[t as usize].lock().children.read().iter() {
    //         self.scale_child(*child, p, s);
    //     }
    // }

    pub fn rotate(&self, axis: &Vec3, radians: f32) {
        self.transforms.rotate(self.id, axis, radians);
    }

    pub fn get_meta(&self) -> &Mutex<TransformMeta> {
        &self.transforms.meta[self.id as usize]
    }
    // fn rotate_child(&self,  axis: &Vec3, p: &Vec3, r: &Quat, radians: f32) {
    //     let ax = glm::quat_to_mat3(&r) * axis;
    //     let mut rot = self.rotations[t as usize].lock();
    //     let mut pos = self.positions[t as usize].lock();

    //     *pos = *pos + glm::rotate_vec3(&(*pos - p), radians, &ax);
    //     *rot = glm::quat_rotate(
    //         &*rot,
    //         radians,
    //         &(glm::quat_to_mat3(&glm::quat_inverse(&*rot)) * ax),
    //     );
    //     for child in self.meta[t as usize].lock().children.read().iter() {
    //         self.rotate_child(*child, axis, p, r, radians);
    //     }
    // }
}

pub const POS_U: usize = 0;
pub const ROT_U: usize = 1;
pub const SCL_U: usize = 2;
fn div_vec3(a: &Vec3, b: &Vec3) -> Vec3 {
    glm::vec3(a.x / b.x, a.y / b.y, a.z / b.z)
}
fn mul_vec3(a: &Vec3, b: &Vec3) -> Vec3 {
    glm::vec3(a.x * b.x, a.y * b.y, a.z * b.z)
}

#[allow(dead_code)]
impl Transforms {
    pub fn new() -> Transforms {
        Transforms {
            positions: Vec::new(),
            rotations: Vec::new(),
            scales: Vec::new(),
            meta: Vec::new(),
            updates: Vec::new(),
            avail: BinaryHeap::new(),
            extent: 0,
        }
    }
    pub fn new_root(&mut self) -> i32 {
        match self.avail.pop() {
            Some(Reverse(i)) => {
                self.positions[i as usize] = Mutex::new(glm::vec3(0.0, 0.0, 0.0));
                self.rotations[i as usize] = Mutex::new(glm::quat(1.0, 0.0, 0.0, 0.0));
                self.scales[i as usize] = Mutex::new(glm::vec3(1.0, 1.0, 1.0));
                self.meta[i as usize] = Mutex::new(TransformMeta::new());
                self.updates[i as usize][POS_U].store(true, Ordering::Relaxed);
                self.updates[i as usize][ROT_U].store(true, Ordering::Relaxed);
                self.updates[i as usize][SCL_U].store(true, Ordering::Relaxed);
                i
            }
            None => {
                self.positions.push(Mutex::new(glm::vec3(0.0, 0.0, 0.0)));
                self.rotations
                    .push(Mutex::new(glm::quat(1.0, 0.0, 0.0, 0.0)));
                self.scales.push(Mutex::new(glm::vec3(1.0, 1.0, 1.0)));
                self.meta.push(Mutex::new(TransformMeta::new()));
                self.updates.push([
                    AtomicBool::new(true),
                    AtomicBool::new(true),
                    AtomicBool::new(true),
                ]);
                self.extent += 1;
                self.extent as i32 - 1
            }
        }
    }
    pub fn new_transform(&mut self, parent: i32) -> i32 {
        self.new_transform_with(parent, Default::default())
    }
    pub fn new_transform_with(&mut self, parent: i32, transform: _Transform) -> i32 {
        if parent == -1 {
            panic!("no")
        }
        let ret = match self.avail.pop() {
            Some(Reverse(i)) => {
                self.positions[i as usize] = Mutex::new(transform.position);
                self.rotations[i as usize] = Mutex::new(transform.rotation);
                self.scales[i as usize] = Mutex::new(transform.scale);
                self.meta[i as usize] = Mutex::new(TransformMeta::new());
                self.updates[i as usize][POS_U].store(true, Ordering::Relaxed);
                self.updates[i as usize][ROT_U].store(true, Ordering::Relaxed);
                self.updates[i as usize][SCL_U].store(true, Ordering::Relaxed);
                i
            }
            None => {
                self.positions.push(Mutex::new(transform.position));
                self.rotations.push(Mutex::new(transform.rotation));
                self.scales.push(Mutex::new(transform.scale));
                self.meta.push(Mutex::new(TransformMeta::new()));
                self.updates.push([
                    AtomicBool::new(true),
                    AtomicBool::new(true),
                    AtomicBool::new(true),
                ]);
                self.extent += 1;
                self.extent as i32 - 1
            }
        };
        let mut meta = self.meta[ret as usize].lock();
        meta.parent = parent;
        unsafe {
            meta.child_id = SendSync::new(
                self.meta[parent as usize]
                    .lock()
                    .children
                    .write()
                    .push_tail(ret),
            );
        }
        ret
    }
    pub fn remove(&mut self, t: i32) {
        {
            let meta = self.meta[t as usize].lock();
            if meta.parent < 0 {
                panic!("wat?")
            }
            self.meta[meta.parent as usize]
                .lock()
                .children
                .write()
                .pop_node(&meta.child_id);
        }

        self.meta[t as usize] = Mutex::new(TransformMeta::new());
        self.avail.push(Reverse(t));
    }
    pub fn adopt(&mut self, p: i32, t: i32) {
        let p_meta = self.meta[p as usize].lock();
        let mut t_meta = self.meta[t as usize].lock();

        if t == p {
            return;
        }
        if t_meta.parent == p {
            return;
        }

        self.meta[t_meta.parent as usize]
        .lock()
        .children
        .write()
        .pop_node(&t_meta.child_id);

        t_meta.parent = p;
        unsafe {
            t_meta.child_id = SendSync::new(
                p_meta
                    .children
                    .write()
                    .push_tail(t),
            );
        }


    }

    fn u_pos(&self, t: i32) {
        self.updates[t as usize][POS_U].store(true, Ordering::Relaxed);
    }
    fn u_rot(&self, t: i32) {
        self.updates[t as usize][ROT_U].store(true, Ordering::Relaxed);
    }
    fn u_scl(&self, t: i32) {
        self.updates[t as usize][SCL_U].store(true, Ordering::Relaxed);
    }

    pub fn forward(&self, t: i32) -> glm::Vec3 {
        glm::quat_to_mat3(&*self.rotations[t as usize].lock()) * glm::Vec3::z()
    }
    pub fn right(&self, t: i32) -> glm::Vec3 {
        glm::quat_to_mat3(&*self.rotations[t as usize].lock()) * glm::Vec3::x()
    }
    pub fn up(&self, t: i32) -> glm::Vec3 {
        glm::quat_to_mat3(&*self.rotations[t as usize].lock()) * glm::Vec3::y()
    }

    pub fn _move(&self, t: i32, v: Vec3) {
        *self.positions[t as usize].lock() += v;
        self.u_pos(t);
    }
    pub fn move_child(&self, t: i32, v: Vec3) {
        self._move(t, v);
        for child in self.meta[t as usize].lock().children.read().iter() {
            self.move_child(*child, v);
        }
    }
    pub fn translate(&self, t: i32, mut v: Vec3) {
        v = glm::quat_to_mat3(&self.rotations[t as usize].lock()) * v;
        *self.positions[t as usize].lock() += v;
        self.u_pos(t);
        for child in self.meta[t as usize].lock().children.read().iter() {
            self.move_child(*child, v);
        }
    }
    pub fn get_position(&self, t: i32) -> Vec3 {
        *self.positions[t as usize].lock()
    }
    pub fn set_position(&self, t: i32, v: Vec3) {
        *self.positions[t as usize].lock() = v;
        self.u_pos(t);
    }
    pub fn get_rotation(&self, t: i32) -> Quat {
        *self.rotations[t as usize].lock()
    }
    pub fn set_rotation(&self, t: i32, r: Quat) {
        let mut r_l = self.rotations[t as usize].lock();
        *r_l = r;
        self.u_rot(t);
        let inv_rot = glm::inverse(&glm::quat_to_mat3(&*r_l));
        drop(r_l);
        let pos = *self.positions[t as usize].lock();
        for child in self.meta[t as usize].lock().children.read().iter() {
            self.set_rotation_child(*child, &inv_rot, &pos)
        }
    }
    fn set_rotation_child(&self, tc: i32, rot: &Mat3, pos: &Vec3) {
        let mut rotat = self.rotations[tc as usize].lock();
        let mut posi = self.positions[tc as usize].lock();

        *posi = pos + rot * (*posi - pos);
        self.u_pos(tc);
        *rotat = glm::mat3_to_quat(&rot) * *rotat;
        self.u_rot(tc);
        for child in self.meta[tc as usize].lock().children.read().iter() {
            self.set_rotation_child(*child, &rot, &pos)
        }
    }
    pub fn get_scale(&self, t: i32) -> Vec3 {
        *self.scales[t as usize].lock()
    }
    pub fn set_scale(&self, t: i32, s: Vec3) {
        // *self.positions[t as usize].lock() = v;
        // self.pos_u(t);
        let scl = *self.scales[t as usize].lock();
        self.scale(t, glm::vec3(s.x / scl.x, s.y / scl.y, s.z / scl.z));
    }
    pub fn scale(&self, t: i32, s: Vec3) {
        let mut scl = self.positions[t as usize].lock();
        *scl = mul_vec3(&s, &*scl);
        self.u_scl(t);
        let pos = self.get_position(t);
        for child in self.meta[t as usize].lock().children.read().iter() {
            self.scale_child(*child, &pos, &s);
        }
    }
    fn scale_child(&self, t: i32, p: &Vec3, s: &Vec3) {
        let mut scl = self.scales[t as usize].lock();
        let mut posi = self.positions[t as usize].lock();

        *posi = mul_vec3(&(*posi - p), s) + p;
        self.u_pos(t);
        *scl = mul_vec3(s, &*scl);
        self.u_scl(t);
        for child in self.meta[t as usize].lock().children.read().iter() {
            self.scale_child(*child, p, s);
        }
    }

    pub fn rotate(&self, t: i32, axis: &Vec3, radians: f32) {
        let mut rot = self.rotations[t as usize].lock();
        *rot = glm::quat_rotate(&*rot, radians, axis);
        self.u_rot(t);
        let pos = self.get_position(t);
        let rot = *rot;
        for child in self.meta[t as usize].lock().children.read().iter() {
            self.rotate_child(*child, axis, &pos, &rot, radians);
        }
    }

    fn rotate_child(&self, t: i32, axis: &Vec3, p: &Vec3, r: &Quat, radians: f32) {
        let ax = glm::quat_to_mat3(&r) * axis;
        let mut rot = self.rotations[t as usize].lock();
        let mut pos = self.positions[t as usize].lock();

        *pos = *pos + glm::rotate_vec3(&(*pos - p), radians, &ax);
        *rot = glm::quat_rotate(
            &*rot,
            radians,
            &(glm::quat_to_mat3(&glm::quat_inverse(&*rot)) * ax),
        );
        for child in self.meta[t as usize].lock().children.read().iter() {
            self.rotate_child(*child, axis, p, r, radians);
        }
    }

    pub fn get_transform_data_updates(
        &mut self,
    ) -> Arc<(
        usize,
        Vec<Arc<(Vec<Vec<i32>>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>)>>,
    )> {
        // let pos_iter = self.positions.par_iter_mut();
        // let rot_iter = self.rotations.par_iter_mut();
        // let scl_iter = self.scales.par_iter_mut();
        let len = self.updates.len();
        // let u_iter = self.updates.par_iter_mut();

        let transform_data = Mutex::new(Vec::<
            Arc<(
                std::vec::Vec<std::vec::Vec<i32>>,
                std::vec::Vec<[f32; 3]>,
                std::vec::Vec<[f32; 4]>,
                std::vec::Vec<[f32; 3]>,
            )>,
        >::new());
        (0..num_cpus::get()).into_par_iter().for_each(|id| {
            let mut transform_ids = vec![
                Vec::<i32>::with_capacity(len),
                Vec::<i32>::with_capacity(len),
                Vec::<i32>::with_capacity(len),
            ];
            let mut pos = Vec::<[f32; 3]>::with_capacity(len);
            let mut rot = Vec::<[f32; 4]>::with_capacity(len);
            let mut scl = Vec::<[f32; 3]>::with_capacity(len);
            {
                let start = len / num_cpus::get() * id;
                let mut end = start + len / num_cpus::get();
                if id == num_cpus::get() - 1 {
                    end = len;
                }

                let u = &self.updates;
                let p = &self.positions;
                let r = &self.rotations;
                let s = &self.scales;

                puffin::profile_scope!("collect data");
                for i in start..end {
                    if u[i][POS_U].load(Ordering::Relaxed) {
                        transform_ids[POS_U].push(i as i32);
                        let p = p[i].lock();
                        // p.data
                        pos.push([p.x, p.y, p.z]);
                        u[i][POS_U].store(false, Ordering::Relaxed);
                    }
                    if u[i][ROT_U].load(Ordering::Relaxed) {
                        transform_ids[ROT_U].push(i as i32);
                        let r = r[i].lock();
                        // let a: [f32;4] = r.coords.into();
                        rot.push([r.w, r.k, r.j, r.i]);
                        u[i][ROT_U].store(false, Ordering::Relaxed);
                    }
                    if u[i][SCL_U].load(Ordering::Relaxed) {
                        transform_ids[SCL_U].push(i as i32);
                        let s = s[i].lock();
                        scl.push([s.x, s.y, s.z]);
                        u[i][SCL_U].store(false, Ordering::Relaxed);
                    }
                }
            }
            {
                puffin::profile_scope!("wrap in Arc");
                let ret = Arc::new((transform_ids, pos, rot, scl));
                transform_data.lock().push(ret);
            }
        });
        Arc::new((len, transform_data.into_inner()))
    }
}
