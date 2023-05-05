use core::panic;
use deepmesa::lists::{linkedlist::Node, LinkedList};
use force_send_sync::SendSync;
use glm::{Quat, Vec3};
use nalgebra_glm as glm;
use parking_lot::{Mutex, MutexGuard};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
// use spin::{Mutex,RwLock};
use num_integer::Roots;

use serde::{Deserialize, Serialize};
use sync_unsafe_cell::SyncUnsafeCell;

use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

pub struct TransformMeta {
    pub children: SendSync<LinkedList<i32>>,
    pub(crate) parent: i32,
    pub(crate) child_id: SendSync<Node<i32>>,
}

impl TransformMeta {
    fn new() -> TransformMeta {
        unsafe {
            TransformMeta {
                children: SendSync::new(LinkedList::new()),
                parent: -1,
                child_id: SendSync::new(Node::default()),
            }
        }
    }
}
pub struct Transforms {
    mutex: Vec<Mutex<()>>,
    positions: Vec<SyncUnsafeCell<glm::Vec3>>,
    rotations: Vec<SyncUnsafeCell<glm::Quat>>,
    scales: Vec<SyncUnsafeCell<glm::Vec3>>,
    meta: Vec<SyncUnsafeCell<TransformMeta>>,
    avail: BinaryHeap<Reverse<i32>>,
    updates: Vec<SyncUnsafeCell<[bool; 3]>>,
    extent: i32,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
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

pub struct Transform<'a> {
    _lock: MutexGuard<'a, ()>,
    pub id: i32,
    pub transforms: &'a Transforms,
}

#[allow(dead_code)]
impl<'a> Transform<'a> {
    pub fn forward(&self) -> glm::Vec3 {
        let q = unsafe { *self.transforms.rotations[self.id as usize].get() };
        glm::quat_to_mat3(&q) * glm::Vec3::z()
    }
    pub fn right(&self) -> glm::Vec3 {
        glm::quat_to_mat3(unsafe { &*self.transforms.rotations[self.id as usize].get() })
            * glm::Vec3::x()
    }
    pub fn up(&self) -> glm::Vec3 {
        glm::quat_to_mat3(unsafe { &*self.transforms.rotations[self.id as usize].get() })
            * glm::Vec3::y()
    }
    pub fn _move(&self, v: Vec3) {
        self.transforms._move(self.id, v);
    }
    pub fn translate(&self, v: Vec3) {
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
    pub fn get_scale(&self) -> Vec3 {
        self.transforms.get_scale(self.id)
    }
    pub fn set_scale(&self, s: Vec3) {
        self.transforms.set_scale(self.id, s);
    }
    pub fn scale(&self, s: Vec3) {
        self.transforms.scale(self.id, s);
    }
    pub fn rotate(&self, axis: &Vec3, radians: f32) {
        self.transforms.rotate(self.id, axis, radians);
    }

    pub fn get_meta(&self) -> &TransformMeta {
        unsafe { &*self.transforms.meta[self.id as usize].get() }
    }
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
// fn quat_x_vec(q: &Quat,v: &Vec3) -> Vec3 {
//     let quat_vec = glm::vec3(q.coords.w, q.coords.x, q.coords.y);
//     let uv = glm::cross(&quat_vec, v);
//     let uuv = glm::cross(&quat_vec, &uv);

//     v + ((uv * q.w) + uuv) * 2.
// }

#[allow(dead_code)]
impl Transforms {
    pub fn active(&self) -> usize {
        self.extent as usize - self.avail.len()
    }
    pub fn get_transform<'a>(&self, t: i32) -> Transform {
        Transform {
            _lock: self.mutex[t as usize].lock(),
            id: t,
            transforms: self,
        }
    }
    pub fn clear(&mut self) {
        self.positions.clear();
        self.rotations.clear();
        self.scales.clear();
        self.meta.clear();
        self.updates.clear();
        self.avail.clear();
        self.extent = 0;
    }
    pub fn new() -> Transforms {
        Transforms {
            mutex: Vec::new(),
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
                unsafe {
                    self.mutex[i as usize] = Mutex::new(());
                    *self.positions[i as usize].get() = glm::vec3(0.0, 0.0, 0.0);
                    *self.rotations[i as usize].get() = glm::quat(1.0, 0.0, 0.0, 0.0);
                    *self.scales[i as usize].get() = glm::vec3(1.0, 1.0, 1.0);
                    *self.meta[i as usize].get() = TransformMeta::new();
                    let u = &mut *self.updates[i as usize].get();
                    u[POS_U] = true;
                    u[ROT_U] = true;
                    u[SCL_U] = true;
                }
                i
            }
            None => {
                self.mutex.push(Mutex::new(()));
                self.positions
                    .push(SyncUnsafeCell::new(glm::vec3(0.0, 0.0, 0.0)));
                self.rotations
                    .push(SyncUnsafeCell::new(glm::quat(1.0, 0.0, 0.0, 0.0)));
                self.scales
                    .push(SyncUnsafeCell::new(glm::vec3(1.0, 1.0, 1.0)));
                self.meta.push(SyncUnsafeCell::new(TransformMeta::new()));
                self.updates.push(SyncUnsafeCell::new([true, true, true]));
                self.extent += 1;
                self.extent - 1
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
                unsafe {
                    self.mutex[i as usize] = Mutex::new(());
                    *self.positions[i as usize].get() = transform.position;
                    *self.rotations[i as usize].get() = transform.rotation;
                    *self.scales[i as usize].get() = transform.scale;
                    *self.meta[i as usize].get() = TransformMeta::new();
                    let u = &mut *self.updates[i as usize].get();
                    u[POS_U] = true;
                    u[ROT_U] = true;
                    u[SCL_U] = true;
                }
                i
            }
            None => {
                self.mutex.push(Mutex::new(()));
                self.positions.push(SyncUnsafeCell::new(transform.position));
                self.rotations.push(SyncUnsafeCell::new(transform.rotation));
                self.scales.push(SyncUnsafeCell::new(transform.scale));
                self.meta.push(SyncUnsafeCell::new(TransformMeta::new()));
                self.updates.push(SyncUnsafeCell::new([true, true, true]));
                self.extent += 1;
                self.extent - 1
            }
        };
        unsafe {
            let mut meta = &mut *self.meta[ret as usize].get();
            meta.parent = parent;
            meta.child_id =
                SendSync::new((*self.meta[parent as usize].get()).children.push_tail(ret));
        }
        ret
    }
    pub fn remove(&mut self, t: i32) {
        {
            unsafe {
                let meta = &*self.meta[t as usize].get();
                if meta.parent < 0 {
                    panic!("wat?")
                }
                (*self.meta[meta.parent as usize].get())
                    .children
                    .pop_node(&meta.child_id);

                *self.meta[t as usize].get() = TransformMeta::new();
                self.avail.push(Reverse(t));
            }
        }
    }
    pub fn adopt(&mut self, p: i32, t: i32) {
        let p_meta = unsafe { &mut *self.meta[p as usize].get() };
        let t_meta = unsafe { &mut *self.meta[t as usize].get() };

        if t == p {
            return;
        }
        if t_meta.parent == p {
            return;
        }

        unsafe {
            (*self.meta[t_meta.parent as usize].get())
                .children
                .pop_node(&t_meta.child_id);

            t_meta.parent = p;

            t_meta.child_id = SendSync::new(p_meta.children.push_tail(t));
        }
    }
    pub fn change_place_in_hier(&mut self, c: i32, t: i32, insert_under: bool) {
        let c_meta = unsafe { &mut *self.meta[c as usize].get() };
        let t_meta = unsafe { &mut *self.meta[t as usize].get() };
        if insert_under {
            if t == c {
                return;
            }
            if t_meta.parent == c {
                c_meta.children.pop_node(&t_meta.child_id);

                t_meta.parent = c;
                unsafe {
                    t_meta.child_id = SendSync::new(c_meta.children.push_head(t));
                }
                return;
            }

            self.meta[t_meta.parent as usize]
                .get_mut()
                .children
                .pop_node(&t_meta.child_id);

            t_meta.parent = c;
            unsafe {
                t_meta.child_id = SendSync::new(c_meta.children.push_head(t));
            }
            return;
        }

        if t == c {
            return;
        }

        self.meta[t_meta.parent as usize]
            .get_mut()
            .children
            .pop_node(&t_meta.child_id);

        t_meta.parent = c_meta.parent;
        unsafe {
            t_meta.child_id = SendSync::new(
                self.meta[c_meta.parent as usize]
                    .get_mut()
                    .children
                    .push_next(&c_meta.child_id, t)
                    .unwrap(),
            );
        }
    }
    fn u_pos(&self, t: i32) {
        unsafe {
            (*self.updates[t as usize].get())[POS_U] = true;
        }
    }
    fn u_rot(&self, t: i32) {
        unsafe {
            (*self.updates[t as usize].get())[ROT_U] = true;
        }
    }
    fn u_scl(&self, t: i32) {
        unsafe {
            (*self.updates[t as usize].get())[SCL_U] = true;
        }
    }

    pub fn forward(&self, t: i32) -> glm::Vec3 {
        glm::quat_to_mat3(unsafe { &*self.rotations[t as usize].get() }) * glm::Vec3::z()
    }
    pub fn right(&self, t: i32) -> glm::Vec3 {
        glm::quat_to_mat3(unsafe { &*self.rotations[t as usize].get() }) * glm::Vec3::x()
    }
    pub fn up(&self, t: i32) -> glm::Vec3 {
        glm::quat_to_mat3(unsafe { &*self.rotations[t as usize].get() }) * glm::Vec3::y()
    }

    pub fn _move(&self, t: i32, v: Vec3) {
        unsafe {
            *self.positions[t as usize].get() += v;
        }
        self.u_pos(t);
    }
    pub fn move_child(&self, t: i32, v: Vec3) {
        self._move(t, v);

        for child in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            self.move_child(*child, v);
        }
    }
    pub fn translate(&self, t: i32, mut v: Vec3) {
        v = glm::quat_to_mat3(unsafe { &*self.rotations[t as usize].get() }) * v;
        unsafe {
            *self.positions[t as usize].get() += v;
        }
        self.u_pos(t);
        for child in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            self.move_child(*child, v);
        }
    }
    pub fn get_position(&self, t: i32) -> Vec3 {
        unsafe { *self.positions[t as usize].get() }
    }
    pub fn set_position(&self, t: i32, v: Vec3) {
        unsafe {
            *self.positions[t as usize].get() = v;
        }
        self.u_pos(t);
    }
    pub fn get_rotation(&self, t: i32) -> Quat {
        unsafe { *self.rotations[t as usize].get() }
    }
    pub fn set_rotation(&self, t: i32, r: Quat) {
        self.u_rot(t);
        let r_l = unsafe { &mut *self.rotations[t as usize].get() };
        let rot = r * (glm::quat_conjugate(&*r_l) / glm::quat_dot(&*r_l, &*r_l)); //glm::inverse(&glm::quat_to_mat3(&*r_l));
        *r_l = r;
        let pos = unsafe { *self.positions[t as usize].get() };
        for child in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            self.set_rotation_child(*child, &rot, &pos)
        }
    }
    fn set_rotation_child(&self, tc: i32, rot: &Quat, pos: &Vec3) {
        let rotat = unsafe { &mut *self.rotations[tc as usize].get() };
        let posi = unsafe { &mut *self.positions[tc as usize].get() };

        *posi = pos + glm::quat_to_mat3(rot) * (*posi - pos);
        *rotat = rot * *rotat;
        self.u_pos(tc);
        self.u_rot(tc);
        for child in unsafe { (*self.meta[tc as usize].get()).children.iter() } {
            self.set_rotation_child(*child, rot, pos)
        }
    }
    pub fn get_scale(&self, t: i32) -> Vec3 {
        unsafe { *self.scales[t as usize].get() }
    }
    pub fn set_scale(&self, t: i32, s: Vec3) {
        // *self.positions[t as usize].lock() = v;
        // self.pos_u(t);
        let scl = unsafe { *self.scales[t as usize].get() };
        self.scale(t, glm::vec3(s.x / scl.x, s.y / scl.y, s.z / scl.z));
    }
    pub fn scale(&self, t: i32, s: Vec3) {
        let scl = unsafe { &mut *self.scales[t as usize].get() };
        *scl = mul_vec3(&s, &scl);
        self.u_scl(t);
        let pos = self.get_position(t);
        for child in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            self.scale_child(*child, &pos, &s);
        }
    }
    fn scale_child(&self, t: i32, p: &Vec3, s: &Vec3) {
        let scl = unsafe { &mut *self.scales[t as usize].get() };
        let posi = unsafe { &mut *self.positions[t as usize].get() };

        *posi = mul_vec3(&(*posi - p), s) + p;
        self.u_pos(t);
        *scl = mul_vec3(s, &scl);
        self.u_scl(t);
        for child in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            self.scale_child(*child, p, s);
        }
    }

    pub fn rotate(&self, t: i32, axis: &Vec3, radians: f32) {
        let rot = unsafe { &mut *self.rotations[t as usize].get() };
        *rot = glm::quat_rotate(rot, radians, axis);
        let rot = *rot;
        self.u_rot(t);
        let pos = self.get_position(t);
        let mut ax = glm::quat_to_mat3(&rot) * axis;
        ax.x = -ax.x;
        ax.y = -ax.y;
        for child in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            self.rotate_child(*child, &ax, &pos, &rot, radians);
        }
    }

    fn rotate_child(&self, t: i32, axis: &Vec3, pos: &Vec3, r: &Quat, radians: f32) {
        // let ax = glm::quat_to_mat3(&r) * axis;
        // let ax = quat_x_vec(&r, axis);
        // let _ax = glm::inverse(&glm::quat_to_mat3(&r)) * axis;
        let mut ax = *axis;
        ax.x = -ax.x;
        ax.y = -ax.y;
        self.u_rot(t);
        self.u_pos(t);
        let rot = unsafe { &mut *self.rotations[t as usize].get() };
        let p = unsafe { &mut *self.positions[t as usize].get() };

        *p = pos + glm::rotate_vec3(&(*p - pos), radians, axis);
        *rot = glm::quat_rotate(
            &*rot,
            radians,
            &(glm::quat_to_mat3(&glm::quat_inverse(&*rot)) * ax),
        );
        for child in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            self.rotate_child(*child, axis, pos, r, radians);
        }
    }

    pub fn get_parent(&self, t: i32) -> i32 {
        unsafe { (*self.meta[t as usize].get()).parent }
    }

    pub fn get_transform_data_updates(
        &mut self,
    ) -> Arc<(
        usize,
        Vec<Arc<(Vec<Vec<i32>>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>)>>,
    )> {
        let len = self.updates.len();
        let transform_data = Arc::new(Mutex::new(Vec::<
            Arc<(Vec<Vec<i32>>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>)>,
        >::new()));
        let self_ = Arc::new(&self);
        rayon::scope(|s| {
            let num_jobs = num_cpus::get().min(len / 64).max(1); // TODO: find best number dependent on cpu
            for id in 0..num_jobs {
                let start = len / num_jobs * id;
                let mut end = start + len / num_jobs;
                if id == num_jobs - 1 {
                    end = len;
                }
                let end = end;
                let transform_data = transform_data.clone();
                let self_ = self_.clone();
                s.spawn(move |_| {
                    let len = end - start;
                    let mut transform_ids = vec![
                        Vec::<i32>::with_capacity(len),
                        Vec::<i32>::with_capacity(len),
                        Vec::<i32>::with_capacity(len),
                    ];
                    let mut pos = Vec::<[f32; 3]>::with_capacity(len);
                    let mut rot = Vec::<[f32; 4]>::with_capacity(len);
                    let mut scl = Vec::<[f32; 3]>::with_capacity(len);

                    let p = &self_.positions;
                    let r = &self_.rotations;
                    let s = &self_.scales;

                    for i in start..end {
                        unsafe {
                            let u = &mut *self_.updates[i].get();
                            if u[POS_U] {
                                transform_ids[POS_U].push(i as i32);
                                let p = &*p[i].get();
                                pos.push([p.x, p.y, p.z]);
                                u[POS_U] = false;
                            }
                            if u[ROT_U] {
                                transform_ids[ROT_U].push(i as i32);
                                let r = &*r[i].get();
                                rot.push([r.w, r.k, r.j, r.i]);
                                u[ROT_U] = false;
                            }
                            if u[SCL_U] {
                                transform_ids[SCL_U].push(i as i32);
                                let s = &*s[i].get();
                                scl.push([s.x, s.y, s.z]);
                                u[SCL_U] = false;
                            }
                        }
                    }
                    let ret = Arc::new((transform_ids, pos, rot, scl));
                    transform_data.lock().push(ret);
                });
            }
        });
        // self.updates
        //     .par_iter_mut()
        //     .enumerate()
        //     .chunks(
        //         num_cpus::get()
        //             .sqrt()
        //             .max((len as f64 / num_cpus::get() as f64).ceil() as usize),
        //     )
        //     .for_each(|slice| {
        //         let len = slice.len();
        //         let mut transform_ids = vec![
        //             Vec::<i32>::with_capacity(len),
        //             Vec::<i32>::with_capacity(len),
        //             Vec::<i32>::with_capacity(len),
        //         ];
        //         let mut pos = Vec::<[f32; 3]>::with_capacity(len);
        //         let mut rot = Vec::<[f32; 4]>::with_capacity(len);
        //         let mut scl = Vec::<[f32; 3]>::with_capacity(len);

        //         let p = &self.positions;
        //         let r = &self.rotations;
        //         let s = &self.scales;

        //         for (i, u) in slice {
        //             if u[POS_U].load(Ordering::Relaxed) {
        //                 transform_ids[POS_U].push(i as i32);
        //                 let p = p[i].lock();
        //                 // p.data
        //                 pos.push([p.x, p.y, p.z]);
        //                 u[POS_U].store(false, Ordering::Relaxed);
        //             }
        //             if u[ROT_U].load(Ordering::Relaxed) {
        //                 transform_ids[ROT_U].push(i as i32);
        //                 let r = r[i].lock();
        //                 // let a: [f32;4] = r.coords.into();
        //                 rot.push([r.w, r.k, r.j, r.i]);
        //                 u[ROT_U].store(false, Ordering::Relaxed);
        //             }
        //             if u[SCL_U].load(Ordering::Relaxed) {
        //                 transform_ids[SCL_U].push(i as i32);
        //                 let s = s[i].lock();
        //                 scl.push([s.x, s.y, s.z]);
        //                 u[SCL_U].store(false, Ordering::Relaxed);
        //             }
        //         }
        //         let ret = Arc::new((transform_ids, pos, rot, scl));
        //         transform_data.lock().push(ret);
        //     });
        /////////////////////////////////////////////////////////////////////////////////////////////////
        // (0..num_cpus::get()).into_par_iter().for_each(|id| {
        //     let mut transform_ids = vec![
        //         Vec::<i32>::with_capacity(len),
        //         Vec::<i32>::with_capacity(len),
        //         Vec::<i32>::with_capacity(len),
        //     ];
        //     let mut pos = Vec::<[f32; 3]>::with_capacity(len);
        //     let mut rot = Vec::<[f32; 4]>::with_capacity(len);
        //     let mut scl = Vec::<[f32; 3]>::with_capacity(len);
        //     {
        //         let start = len / num_cpus::get() * id;
        //         let mut end = start + len / num_cpus::get();
        //         if id == num_cpus::get() - 1 {
        //             end = len;
        //         }

        //         let u = &self.updates;
        //         let p = &self.positions;
        //         let r = &self.rotations;
        //         let s = &self.scales;

        //         puffin::profile_scope!("collect data");
        //         for i in start..end {
        //             if u[i][POS_U].load(Ordering::Relaxed) {
        //                 transform_ids[POS_U].push(i as i32);
        //                 let p = p[i].lock();
        //                 // p.data
        //                 pos.push([p.x, p.y, p.z]);
        //                 u[i][POS_U].store(false, Ordering::Relaxed);
        //             }
        //             if u[i][ROT_U].load(Ordering::Relaxed) {
        //                 transform_ids[ROT_U].push(i as i32);
        //                 let r = r[i].lock();
        //                 // let a: [f32;4] = r.coords.into();
        //                 rot.push([r.w, r.k, r.j, r.i]);
        //                 u[i][ROT_U].store(false, Ordering::Relaxed);
        //             }
        //             if u[i][SCL_U].load(Ordering::Relaxed) {
        //                 transform_ids[SCL_U].push(i as i32);
        //                 let s = s[i].lock();
        //                 scl.push([s.x, s.y, s.z]);
        //                 u[i][SCL_U].store(false, Ordering::Relaxed);
        //             }
        //         }
        //     }
        //     {
        //         puffin::profile_scope!("wrap in Arc");
        //         let ret = Arc::new((transform_ids, pos, rot, scl));
        //         transform_data.lock().push(ret);
        //     }
        // });
        Arc::new((
            self.extent as usize,
            Arc::try_unwrap(transform_data).unwrap().into_inner(),
        ))
    }
}

// let f = |id| {
//     let mut transform_ids = vec![
//         Vec::<i32>::with_capacity(len),
//         Vec::<i32>::with_capacity(len),
//         Vec::<i32>::with_capacity(len),
//     ];
//     let mut pos = Vec::<[f32; 3]>::with_capacity(len);
//     let mut rot = Vec::<[f32; 4]>::with_capacity(len);
//     let mut scl = Vec::<[f32; 3]>::with_capacity(len);
//     {
//         let start = len / num_cpus::get() * id;
//         let mut end = start + len / num_cpus::get();
//         if id == num_cpus::get() - 1 {
//             end = len;
//         }

//         let u = &self.updates;
//         let p = &self.positions;
//         let r = &self.rotations;
//         let s = &self.scales;

//         puffin::profile_scope!("collect data");
//         for i in start..end {
//             if u[i][POS_U].load(Ordering::Relaxed) {
//                 transform_ids[POS_U].push(i as i32);
//                 let p = p[i].lock();
//                 // p.data
//                 pos.push([p.x, p.y, p.z]);
//                 u[i][POS_U].store(false, Ordering::Relaxed);
//             }
//             if u[i][ROT_U].load(Ordering::Relaxed) {
//                 transform_ids[ROT_U].push(i as i32);
//                 let r = r[i].lock();
//                 // let a: [f32;4] = r.coords.into();
//                 rot.push([r.w, r.k, r.j, r.i]);
//                 u[i][ROT_U].store(false, Ordering::Relaxed);
//             }
//             if u[i][SCL_U].load(Ordering::Relaxed) {
//                 transform_ids[SCL_U].push(i as i32);
//                 let s = s[i].lock();
//                 scl.push([s.x, s.y, s.z]);
//                 u[i][SCL_U].store(false, Ordering::Relaxed);
//             }
//         }
//     }
//     {
//         puffin::profile_scope!("wrap in Arc");
//         let ret = Arc::new((transform_ids, pos, rot, scl));
//         transform_data.lock().push(ret);
//     }
// };
// if self.positions.len() < num_cpus::get() {
//     (0..num_cpus::get()).into_iter().for_each(|i| f(i));
// } else {
//     (0..num_cpus::get()).into_par_iter().for_each(|i| f(i));
// }
