use core::panic;
use crossbeam::{atomic::AtomicConsume, queue::SegQueue};
use deepmesa::lists::{
    linkedlist::{Iter, Node},
    LinkedList,
};
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
        atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering},
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
        self.transforms.forward(self.id)
    }
    pub fn right(&self) -> glm::Vec3 {
        self.transforms.right(self.id)
    }
    pub fn up(&self) -> glm::Vec3 {
        self.transforms.up(self.id)
    }
    pub fn _move(&self, v: Vec3) {
        self.transforms._move(self.id, v);
    }
    pub fn move_child(&self, v: Vec3) {
        self.transforms.move_child(&self, v);
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
    pub fn get_children(&self) -> TransformIter {
        let meta = unsafe { &*self.transforms.meta[self.id as usize].get() };
        TransformIter {
            iter: meta.children.iter(),
            transforms: self.transforms,
        }
    }
    pub fn adopt(&self, child: &Transform) {
        self.transforms.adopt(self.id, child.id);
    }
    pub fn get_transform(&self) -> _Transform {
        // let transforms = &self.transforms;
        _Transform {
            position: self.get_position(),
            rotation: self.get_rotation(),
            scale: self.get_scale(),
        }
    }
    pub fn get_parent(&self) -> Transform {
        self.transforms.get(self.transforms.get_parent(self.id))
    }

    pub(crate) fn get_meta(&self) -> &mut TransformMeta {
        unsafe { &mut *self.transforms.meta[self.id as usize].get() }
    }
}

pub struct TransformIter<'a> {
    iter: Iter<'a, i32>,
    transforms: &'a Transforms,
}

impl<'a> Iterator for TransformIter<'a> {
    type Item = Transform<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(it) = self.iter.next() {
            Some(self.transforms.get(*it))
        } else {
            None
        }
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

pub struct CacheVec<T: Send + Sync> {
    r: Arc<SegQueue<CacheVec<T>>>,
    pub v: Arc<SyncUnsafeCell<Vec<T>>>,
}
impl<T: Send + Sync> CacheVec<T> {
    pub fn len(&self) -> usize {
        unsafe { &*self.v.get() }.len()
    }
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        let a = unsafe { &*self.v.get() }.iter();
        a
    }
    pub fn get(&self) -> &mut Vec<T> {
        unsafe { &mut *self.v.get() }
    }
}
struct VecCache<T: Send + Sync> {
    count: AtomicUsize,
    avail: Arc<SegQueue<CacheVec<T>>>,
    // store: Arc<Mutex<Vec<Arc<Mutex<Vec<T>>>>>>,
}
impl<T: Send + Sync> Drop for CacheVec<T> {
    fn drop(&mut self) {
        self.r.push(CacheVec {
            r: self.r.clone(),
            v: self.v.clone(),
        });
    }
}
impl<'a, T: Send + Sync> VecCache<T> {
    pub fn new() -> Self {
        Self {
            avail: Arc::new(SegQueue::new()),
            count: AtomicUsize::new(0),
            // store: Arc::new(Mutex::new(Vec::new())),
        }
    }
    pub fn get_vec(&self, capacity: usize) -> CacheVec<T> {
        if let Some(mut a) = self.avail.pop() {
            let b = unsafe { &mut *a.v.get() };
            let len = b.len();
            b.clear();
            b.reserve(capacity);
            a
        } else {
            // let mut a = self.store.lock();
            let c = Arc::new(SyncUnsafeCell::new(Vec::with_capacity(capacity)));
            self.count.fetch_add(1, Ordering::Relaxed);
            let b = CacheVec {
                r: self.avail.clone(),
                v: c,
            };
            // a.push(c);
            b
        }
    }
}

pub struct TransformData {
    pub pos_id: Vec<CacheVec<i32>>,
    pub rot_id: Vec<CacheVec<i32>>,
    pub scl_id: Vec<CacheVec<i32>>,
    pub pos_data: Vec<CacheVec<[f32; 3]>>,
    pub rot_data: Vec<CacheVec<[f32; 4]>>,
    pub scl_data: Vec<CacheVec<[f32; 3]>>,
    pub extent: usize,
}
use segvec::SegVec;
use dary_heap::DaryHeap;

use crate::engine::storage::Avail;
pub struct Transforms {
    self_lock: Mutex<()>,
    mutex: SegVec<SyncUnsafeCell<Mutex<()>>>,
    positions: SegVec<SyncUnsafeCell<glm::Vec3>>,
    rotations: SegVec<SyncUnsafeCell<glm::Quat>>,
    scales: SegVec<SyncUnsafeCell<glm::Vec3>>,
    valid: SegVec<SyncUnsafeCell<bool>>,
    meta: SegVec<SyncUnsafeCell<TransformMeta>>,
    last: i32,
    avail: i32,
    count: i32,
    updates: SegVec<SyncUnsafeCell<[bool; 3]>>,
    extent: i32,
    ids_cache: VecCache<i32>,
    pos_cache: VecCache<[f32; 3]>,
    rot_cache: VecCache<[f32; 4]>,
    scl_cache: VecCache<[f32; 3]>,
}

#[allow(dead_code)]
impl Transforms {
    pub fn active(&self) -> usize {
        self.count as usize
    }
    pub fn get<'a>(&self, t: i32) -> Transform {
        // TODO: make option
        Transform {
            _lock: unsafe { (*self.mutex[t as usize].get()).lock() },
            id: t,
            transforms: self,
        }
    }
    pub fn clear(&mut self) {
        *self = Self::new();
        // self.positions.clear();
        // self.rotations.clear();
        // self.scales.clear();
        // self.meta.clear();
        // self.updates.clear();
        // self.avail = AtomicI32::new(0);
        // self.count = AtomicI32::new(0);
        // self.extent = 0;
    }
    pub fn new() -> Transforms {
        Transforms {
            self_lock: Mutex::new(()),
            mutex: SegVec::new(),
            positions: SegVec::new(),
            rotations: SegVec::new(),
            scales: SegVec::new(),
            meta: SegVec::new(),
            updates: SegVec::new(),
            valid: SegVec::new(),
            last: -1,
            avail: 0,
            count: 0,
            extent: 0,
            ids_cache: VecCache::new(),
            pos_cache: VecCache::new(),
            rot_cache: VecCache::new(),
            scl_cache: VecCache::new(),
        }
    }

    fn write_transform(&self, i: i32, t: _Transform) {
        unsafe {
            *self.mutex[i as usize].get() = Mutex::new(());
            *self.positions[i as usize].get() = t.position;
            *self.rotations[i as usize].get() = t.rotation;
            *self.scales[i as usize].get() = t.scale;
            *self.meta[i as usize].get() = TransformMeta::new();
            *self.valid[i as usize].get() = true;
            let u = &mut *self.updates[i as usize].get();
            u[POS_U] = true;
            u[ROT_U] = true;
            u[SCL_U] = true;
        }
    }
    fn push_transform(&mut self, t: _Transform) {
        self.mutex.push(SyncUnsafeCell::new(Mutex::new(())));
        self.positions.push(SyncUnsafeCell::new(t.position));
        self.rotations.push(SyncUnsafeCell::new(t.rotation));
        self.scales.push(SyncUnsafeCell::new(t.scale));
        self.meta.push(SyncUnsafeCell::new(TransformMeta::new()));
        self.valid.push(SyncUnsafeCell::new(true));
        self.updates.push(SyncUnsafeCell::new([true, true, true]));
    }
    fn get_next_id(&mut self) -> (bool, i32) {
        self.count += 1;
        let mut i = self.avail;
        self.avail += 1;
        while i < self.extent && unsafe { *self.valid[i as usize].get() } {
            i = self.avail;
            self.avail += 1;
        }
        if i == self.extent {
            // push back
            self.extent += 1;
            self.last = i;
            (false, i)
        } else {
            // insert
            self.last = self.last.max(i);
            (true, i)
        }
        // if let Some(i) = self.avail.pop() {
        //     self.last = self.last.max(i);
        //     (true, i)
        // } else {
        //     let i = self.extent;
        //     self.extent += 1;
        //     self.last = i;
        //     (false, i)
        // }
        // self.count.fetch_add(1, Ordering::Acquire);
        // let mut i = self.avail.fetch_add(1, Ordering::Relaxed);
        // while i < self.extent.load(Ordering::Relaxed) && unsafe { *self.valid[i as usize].get() } {
        //     i = self.avail.fetch_add(1, Ordering::Relaxed);
        // }
        // if i == self.extent.load(Ordering::Relaxed) {
        //     self.extent.fetch_add(1, Ordering::Relaxed);
        //     self.last.store(i, Ordering::Relaxed);
        //     (false, i)
        // } else {
        //     self.last.fetch_max(i, Ordering::Relaxed);
        //     (true, i)
        // }

        // let i = self.avail.load(Ordering::Relaxed);
        // self.count.fetch_add(1, Ordering::Relaxed);
        // let extent = self.extent.load(Ordering::Relaxed);
        // if i < extent {
        //     unsafe {
        //         *self.valid[i as usize].get() = true;
        //     }
        //     let mut _i = i;
        //     while _i < extent && unsafe { *self.valid[_i as usize].get() } {
        //         // find next open slot
        //         _i += 1;
        //     }
        //     self.avail.store(_i, Ordering::Relaxed);
        //     self.last.fetch_max(_i, Ordering::Relaxed);
        //     return (true, i);
        // } else {
        //     let extent = i + 1;
        //     self.extent.store(extent, Ordering::Relaxed);
        //     self.avail.store(extent, Ordering::Relaxed);
        //     self.last.store(i, Ordering::Relaxed);
        //     return (false, i);
        // }
    }
    pub(crate) fn reduce_last(&mut self) {
        // let i = self.last.fetch_add(-1, Ordering::Relaxed);
        // println!("data: {}, to commit: {}", self.avail.data.len(), self.avail.new_elem.len());
        // self.avail.commit();
        let mut id = self.last;
        while id >= 0 && !unsafe { *self.valid[id as usize].get() } {
            // not thread safe!
            id -= 1;
        }
        self.last = id;
        // let mut id = id;
        // if id == self.last.load(Ordering::Relaxed) {
        //     while id >= 0 && !unsafe { *self.valid[id as usize].get() } {
        //         // not thread safe!
        //         id -= 1;
        //     }
        //     self.last.store(id, Ordering::Relaxed); // multi thread safe?
        // }
    }
    pub fn new_root(&mut self) -> i32 {
        let (write, id) = self.get_next_id();
        if write {
            self.write_transform(id, _Transform::default());
        } else {
            self.push_transform(_Transform::default());
        }
        id
    }

    pub fn new_transform(&mut self, parent: i32) -> i32 {
        self.new_transform_with(parent, Default::default())
    }
    pub fn new_transform_with(&mut self, parent: i32, transform: _Transform) -> i32 {
        if parent == -1 {
            panic!("no")
        }
        let (write, id) = self.get_next_id();
        if write {
            self.write_transform(id, transform);
        } else {
            self.push_transform(transform);
        }
        if id as usize >= self.positions.len() {
            panic!("next id invalid/out of range: {:?}", (write, id));
        }
        unsafe {
            let meta = &mut *self.meta[id as usize].get();
            meta.parent = parent;
            meta.child_id =
                SendSync::new((*self.meta[parent as usize].get()).children.push_tail(id));
        }
        id
    }

    // pub fn multi_transform_with_avail<T>(&mut self, count: i32, t_func: T) ->
    //     where
    //         T: Fn() -> _Transform + Send + Sync, {

    //         }
    pub fn multi_transform_with<T>(&mut self, parent: i32, count: i32, t_func: T) -> Vec<i32>
    where
        T: Fn() -> _Transform + Send + Sync,
    {
        let mut c = 1;
        let mut r = Vec::new();
        let t_func = Arc::new(&t_func);
        let _self = Arc::new(&self);

        for _ in 0..count {
            let i = self.new_transform_with(parent, t_func());
            r.push(i);
        }
        r
        // rayon::scope(|s| {
        //     let _self = _self.clone();
        // let mut avail = _self.avail.lock();
        //     while let Some(i) = avail.pop() {
        //         let t_func = t_func.clone();
        //         let _self = _self.clone();
        //         r.push(i);
        //         // let i = i.clone();
        //         // s.spawn(move |_| {
        //         let i = i.clone();
        //         let t_func = t_func.clone();
        //         let _self = _self.clone();

        //         let transform = t_func();
        //         unsafe {
        //             *_self.mutex[i as usize].get() = Mutex::new(());
        //             *_self.positions[i as usize].get() = transform.position;
        //             *_self.rotations[i as usize].get() = transform.rotation;
        //             *_self.scales[i as usize].get() = transform.scale;
        //             *_self.meta[i as usize].get() = TransformMeta::new();
        //             let u = &mut *_self.updates[i as usize].get();
        //             u[POS_U] = true;
        //             u[ROT_U] = true;
        //             u[SCL_U] = true;
        //             let parent = _self.get(parent);
        //             // parent.adopt(&self.get(i));
        //             let meta = &mut *_self.meta[i as usize].get();
        //             meta.parent = parent.id;
        //             (*_self.meta[i as usize].get()).child_id =
        //                 SendSync::new(parent.get_meta().children.push_tail(i));
        //         }
        //         // });

        //         c += 1;
        //         if c == count {
        //             break;
        //         }
        //     }
        //     // }
        // });
        // // drop(_self);
        // for _ in c..count {
        //     r.push(self.extent);
        //     let transform = t_func();
        //     self.mutex.push(SyncUnsafeCell::new(Mutex::new(())));
        //     self.positions.push(SyncUnsafeCell::new(transform.position));
        //     self.rotations.push(SyncUnsafeCell::new(transform.rotation));
        //     self.scales.push(SyncUnsafeCell::new(transform.scale));
        //     self.meta.push(SyncUnsafeCell::new(TransformMeta::new()));
        //     self.updates.push(SyncUnsafeCell::new([true, true, true]));
        //     unsafe {
        //         let meta = &mut *self.meta[self.extent as usize].get();
        //         meta.parent = parent;
        //         meta.child_id = SendSync::new(
        //             (*self.meta[parent as usize].get())
        //                 .children
        //                 .push_tail(self.extent),
        //         );
        //     }
        //     self.extent += 1;
        // }
        // r
    }
    pub fn remove(&mut self, t: i32) {
        // self.avail.push(t);
        self.avail = self.avail.min(t);
        self.count -= 1;
        // self.reduce_last(id);

        unsafe {
            let meta = &*self.meta[t as usize].get();
            if meta.parent >= 0 {
                // panic!("wat? - child:{} parent: {}", t.id, meta.parent);
                (&mut *self.meta[t as usize].get()).children.pop_node(&meta.child_id);
            }
            *self.meta[t as usize].get() = TransformMeta::new();
            *self.valid[t as usize].get() = false;
        }
    }
    pub fn adopt(&self, p: i32, t: i32) {
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
    pub fn change_place_in_hier(&self, c: i32, t: i32, insert_under: bool) {
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

            unsafe { &mut *self.meta[t_meta.parent as usize].get() }
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

        unsafe {
            (*self.meta[t_meta.parent as usize].get())
                .children
                .pop_node(&t_meta.child_id);

            t_meta.parent = c_meta.parent;
            t_meta.child_id = SendSync::new(
                (*self.meta[c_meta.parent as usize].get())
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

    fn forward(&self, t: i32) -> glm::Vec3 {
        glm::quat_to_mat3(unsafe { &*self.rotations[t as usize].get() }) * glm::Vec3::z()
    }
    fn right(&self, t: i32) -> glm::Vec3 {
        glm::quat_to_mat3(unsafe { &*self.rotations[t as usize].get() }) * glm::Vec3::x()
    }
    fn up(&self, t: i32) -> glm::Vec3 {
        glm::quat_to_mat3(unsafe { &*self.rotations[t as usize].get() }) * glm::Vec3::y()
    }

    fn _move(&self, t: i32, v: Vec3) {
        unsafe {
            *self.positions[t as usize].get() += v;
        }
        self.u_pos(t);
    }
    fn move_child(&self, t: &Transform, v: Vec3) {
        self._move(t.id, v);

        for child in t.get_children() {
            self.move_child(&child, v);
        }
    }
    fn translate(&self, t: i32, mut v: Vec3) {
        v = glm::quat_to_mat3(unsafe { &*self.rotations[t as usize].get() }) * v;
        unsafe {
            *self.positions[t as usize].get() += v;
        }
        self.u_pos(t);
        for child_id in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            let child = self.get(*child_id);
            self.move_child(&child, v);
        }
    }
    fn get_position(&self, t: i32) -> Vec3 {
        unsafe { *self.positions[t as usize].get() }
    }
    fn set_position(&self, t: i32, v: Vec3) {
        unsafe {
            *self.positions[t as usize].get() = v;
        }
        self.u_pos(t);
    }
    fn get_rotation(&self, t: i32) -> Quat {
        unsafe { *self.rotations[t as usize].get() }
    }
    fn set_rotation(&self, t: i32, r: Quat) {
        self.u_rot(t);
        let r_l = unsafe { &mut *self.rotations[t as usize].get() };
        let rot = r * (glm::quat_conjugate(&*r_l) / glm::quat_dot(&*r_l, &*r_l)); //glm::inverse(&glm::quat_to_mat3(&*r_l));
        *r_l = r;
        let pos = unsafe { *self.positions[t as usize].get() };
        for child_id in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            let child = self.get(*child_id);
            self.set_rotation_child(&child, &rot, &pos)
        }
    }
    fn set_rotation_child(&self, t: &Transform, rot: &Quat, pos: &Vec3) {
        let rotat = unsafe { &mut *self.rotations[t.id as usize].get() };
        let posi = unsafe { &mut *self.positions[t.id as usize].get() };

        *posi = pos + glm::quat_to_mat3(rot) * (*posi - pos);
        *rotat = rot * *rotat;
        self.u_pos(t.id);
        self.u_rot(t.id);
        for child in t.get_children() {
            self.set_rotation_child(&child, rot, pos)
        }
    }
    fn get_scale(&self, t: i32) -> Vec3 {
        unsafe { *self.scales[t as usize].get() }
    }
    fn set_scale(&self, t: i32, s: Vec3) {
        // *self.positions[t as usize].lock() = v;
        // self.pos_u(t);
        let scl = unsafe { *self.scales[t as usize].get() };
        self.scale(t, glm::vec3(s.x / scl.x, s.y / scl.y, s.z / scl.z));
    }
    fn scale(&self, t: i32, s: Vec3) {
        let scl = unsafe { &mut *self.scales[t as usize].get() };
        *scl = mul_vec3(&s, &scl);
        self.u_scl(t);
        let pos = self.get_position(t);
        for child_id in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            let child = self.get(*child_id);
            self.scale_child(&child, &pos, &s);
        }
    }
    fn scale_child(&self, t: &Transform, p: &Vec3, s: &Vec3) {
        let scl = unsafe { &mut *self.scales[t.id as usize].get() };
        let posi = unsafe { &mut *self.positions[t.id as usize].get() };

        *posi = mul_vec3(&(*posi - p), s) + p;
        self.u_pos(t.id);
        *scl = mul_vec3(s, &scl);
        self.u_scl(t.id);
        for child in t.get_children() {
            self.scale_child(&child, p, s);
        }
    }

    fn rotate(&self, t: i32, axis: &Vec3, radians: f32) {
        let rot = unsafe { &mut *self.rotations[t as usize].get() };
        *rot = glm::quat_rotate(rot, radians, axis);
        let rot = *rot;
        self.u_rot(t);
        let pos = self.get_position(t);
        let mut ax = glm::quat_to_mat3(&rot) * axis;
        ax.x = -ax.x;
        ax.y = -ax.y;
        for child_id in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            let child = self.get(*child_id);
            self.rotate_child(&child, &ax, &pos, &rot, radians);
        }
    }

    fn rotate_child(&self, t: &Transform, axis: &Vec3, pos: &Vec3, r: &Quat, radians: f32) {
        // let ax = glm::quat_to_mat3(&r) * axis;
        // let ax = quat_x_vec(&r, axis);
        // let _ax = glm::inverse(&glm::quat_to_mat3(&r)) * axis;
        let mut ax = *axis;
        ax.x = -ax.x;
        ax.y = -ax.y;
        self.u_rot(t.id);
        self.u_pos(t.id);
        let rot = unsafe { &mut *self.rotations[t.id as usize].get() };
        let p = unsafe { &mut *self.positions[t.id as usize].get() };

        *p = pos + glm::rotate_vec3(&(*p - pos), radians, axis);
        *rot = glm::quat_rotate(
            &*rot,
            radians,
            &(glm::quat_to_mat3(&glm::quat_inverse(&*rot)) * ax),
        );
        for child in t.get_children() {
            self.rotate_child(&child, axis, pos, r, radians);
        }
    }

    fn get_parent(&self, t: i32) -> i32 {
        unsafe { (*self.meta[t as usize].get()).parent }
    }

    pub fn get_transform_data_updates(&mut self) -> TransformData {
        let last = self.last as usize + 1;
        // let _positions = &self.positions[0..last];
        // let _rotations = &self.rotations[0..last];
        // let _scales = &self.scales[0..last];
        // let _updates = &self.updates[0..last];
        let len = last;
        let transform_data = Arc::new(Mutex::new(TransformData {
            pos_id: Vec::new(),
            rot_id: Vec::new(),
            scl_id: Vec::new(),
            pos_data: Vec::new(),
            rot_data: Vec::new(),
            scl_data: Vec::new(),
            extent: self.extent as usize,
        }));
        // let transform_data = Arc::new(Mutex::new(
        //    (Vec::<VecCache::<i32>::new()>),
        // ));
        let self_ = Arc::new(&self);
        // println!("vec cache size: {}", self.pos_cache.count.load(Ordering::Relaxed));

        rayon::scope(|s| {
            // let num_jobs = (len / 4096).max(1); // TODO: find best number dependent on cpu
            let num_jobs = num_cpus::get().min(last / 2048).max(1); // TODO: find best number dependent on cpu
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
                    let _pos_ids = self_.ids_cache.get_vec(len);
                    let _rot_ids = self_.ids_cache.get_vec(len);
                    let _scl_ids = self_.ids_cache.get_vec(len);

                    let _pos = self_.pos_cache.get_vec(len);
                    let _rot = self_.rot_cache.get_vec(len);
                    let _scl = self_.scl_cache.get_vec(len);

                    {
                        let mut p_ids = _pos_ids.get();
                        let mut r_ids = _rot_ids.get();
                        let mut s_ids = _scl_ids.get();

                        let mut pos = _pos.get();
                        let mut rot = _rot.get();
                        let mut scl = _scl.get();

                        let p = &self_.positions;
                        let r = &self_.rotations;
                        let s = &self_.scales;
                        let _u = &self_.updates;

                        for i in start..end {
                            unsafe {
                                if !*self_.valid[i].get() {
                                    continue;
                                }
                                let u = &mut *_u[i].get();
                                if u[POS_U] {
                                    p_ids.push(i as i32);
                                    let p = &*p[i].get();
                                    pos.push([p.x, p.y, p.z]);
                                    u[POS_U] = false;
                                }
                                if u[ROT_U] {
                                    r_ids.push(i as i32);
                                    let r = &*r[i].get();
                                    rot.push([r.w, r.k, r.j, r.i]);
                                    u[ROT_U] = false;
                                }
                                if u[SCL_U] {
                                    s_ids.push(i as i32);
                                    let s = &*s[i].get();
                                    scl.push([s.x, s.y, s.z]);
                                    u[SCL_U] = false;
                                }
                            }
                        }
                    }

                    // let ret = Arc::new((transform_ids, pos, rot, scl));
                    // transform_data.lock().push(ret);

                    let mut td = transform_data.lock();
                    td.pos_data.push(_pos);
                    td.pos_id.push(_pos_ids);
                    td.rot_data.push(_rot);
                    td.rot_id.push(_rot_ids);
                    td.scl_data.push(_scl);
                    td.scl_id.push(_scl_ids);
                });
            }
        });
        Arc::into_inner(transform_data).unwrap().into_inner()
    }
}
