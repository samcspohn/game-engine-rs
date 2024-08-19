use bitvec::vec::BitVec;
use core::panic;
use crossbeam::{atomic::AtomicConsume, queue::SegQueue};
// use deepmesa::lists::{
//     linkedlist::{Iter, Node},
//     LinkedList,
// };
use deepmesa::collections::{
    linkedlist::{Iter, Node},
    LinkedList,
};
use force_send_sync::SendSync;
use glm::{quat_cross_vec, quat_rotate_vec3, vec3, Quat, Vec3};
use nalgebra_glm as glm;
use once_cell::sync::Lazy;
use parking_lot::{Mutex, MutexGuard};
use rayon::prelude::*;
use vulkano::buffer::Subbuffer;
// use spin::{Mutex,RwLock};
use num_integer::Roots;

use crate::{
    editor::editor_ui::DRAGGED_TRANSFORM,
    engine::{
        prelude::Inspectable_,
        prelude::{Inpsect, Ins},
        project::asset_manager::drop_target,
        world::nalgebra::Isometry3,
    },
};
use serde::{
    de::{self, Visitor},
    ser::SerializeStruct,
    Deserialize, Deserializer, Serialize,
};
use std::{
    cell::SyncUnsafeCell,
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    mem::MaybeUninit,
    sync::{
        atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering},
        Arc,
    },
    thread::ThreadId,
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
            rotation: glm::quat_look_at_lh(&Vec3::z(), &Vec3::y()),
            scale: glm::vec3(1.0, 1.0, 1.0),
        }
    }
}

pub struct Transform<'a> {
    // thr_id: ThreadId,
    _lock: MutexGuard<'a, ()>,
    pub id: i32,
    // pub transforms: &'a Transforms,
}

#[allow(dead_code)]
impl<'a> Transform<'a> {
    pub fn forward(&self) -> glm::Vec3 {
        unsafe { &*TRANSFORMS }.forward(self.id)
    }
    pub fn right(&self) -> glm::Vec3 {
        unsafe { &*TRANSFORMS }.right(self.id)
    }
    pub fn up(&self) -> glm::Vec3 {
        unsafe { &*TRANSFORMS }.up(self.id)
    }
    pub fn _move(&self, v: Vec3) {
        unsafe { &*TRANSFORMS }._move(self.id, v);
    }
    pub fn move_child(&self, v: Vec3) {
        unsafe { &*TRANSFORMS }.move_child(self.id, v);
    }
    pub fn translate(&self, v: Vec3) {
        unsafe { &*TRANSFORMS }.translate(self.id, v);
    }
    pub fn get_position(&self) -> Vec3 {
        unsafe { &*TRANSFORMS }.get_position(self.id)
    }
    pub fn set_position(&self, v: &Vec3) {
        unsafe { &*TRANSFORMS }.set_position(self.id, v);
    }
    pub(crate) fn set_position_(&self, v: &Vec3) {
        unsafe { &*TRANSFORMS }.set_position_(self.id, v);
    }
    pub fn get_rotation(&self) -> Quat {
        unsafe { &*TRANSFORMS }.get_rotation(self.id)
    }
    pub fn set_rotation(&self, r: &Quat) {
        unsafe { &*TRANSFORMS }.set_rotation(self.id, r);
    }
    pub(crate) fn set_rotation_(&self, r: &Quat) {
        unsafe { &*TRANSFORMS }.set_rotation_(self.id, r);
    }
    pub fn look_at(&self, direction: &Vec3, up: &Vec3) {
        unsafe { &*TRANSFORMS }.look_at(self.id, direction, up);
    }
    pub fn get_scale(&self) -> Vec3 {
        unsafe { &*TRANSFORMS }.get_scale(self.id)
    }
    pub fn set_scale(&self, s: Vec3) {
        unsafe { &*TRANSFORMS }.set_scale(self.id, s);
    }
    pub fn scale(&self, s: Vec3) {
        unsafe { &*TRANSFORMS }.scale(self.id, s);
    }
    pub fn rotate(&self, axis: &Vec3, radians: f32) {
        unsafe { &*TRANSFORMS }.rotate(self.id, axis, radians);
    }
    pub fn get_children(&self) -> TransformIter {
        let meta = unsafe { &*unsafe { &*TRANSFORMS }.meta[self.id as usize].get() };
        TransformIter {
            iter: meta.children.iter(),
            transforms: unsafe { &*TRANSFORMS },
        }
    }
    pub fn adopt(&self, child: &Transform) {
        unsafe { &*TRANSFORMS }.adopt(self.id, child.id);
    }
    pub fn get_transform(&self) -> _Transform {
        // let transforms = &unsafe { **TRANSFORMS };
        _Transform {
            position: self.get_position(),
            rotation: self.get_rotation(),
            scale: self.get_scale(),
        }
    }
    pub fn get_parent(&self) -> Transform {
        unsafe { &*TRANSFORMS }
            .get(unsafe { &*TRANSFORMS }.get_parent(self.id))
            .unwrap()
    }

    pub(crate) fn get_meta(&self) -> &mut TransformMeta {
        unsafe { &mut *unsafe { &*TRANSFORMS }.meta[self.id as usize].get() }
    }
    pub fn entity(&self) -> &mut Entity {
        unsafe { &mut *unsafe { &*TRANSFORMS }.entity[self.id as usize].get() }
    }
    pub fn get_matrix(&self) -> glm::Mat4 {
        unsafe { &*TRANSFORMS }.get_matrix(self.id)
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
            Some(unsafe { &*TRANSFORMS }.get(*it).unwrap())
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
    pub fn push(&self, d: T) {
        self.get().push(d);
    }
}
pub struct VecCache<T: Send + Sync> {
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
            // store: Arc::new(Mutex::new(Vec::new())),
        }
    }
    pub fn get_vec(&self, capacity: usize) -> CacheVec<T> {
        if let Some(mut a) = self.avail.pop() {
            let b = unsafe { &mut *a.v.get() };
            b.clear();
            b.reserve(capacity);
            a
        } else {
            // let mut a = self.store.lock();
            let c = Arc::new(SyncUnsafeCell::new(Vec::with_capacity(capacity)));
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
    pub pos_data: Vec<CacheVec<[f32; 4]>>,
    pub rot_data: Vec<CacheVec<[f32; 4]>>,
    pub scl_data: Vec<CacheVec<[f32; 4]>>,
    pub extent: usize,
}
use dary_heap::DaryHeap;
use segvec::SegVec;

use crate::engine::{storage::Avail, transform_compute::cs::transform};

use super::{entity::Entity, Sys};
pub type TransformBuf = (
    Subbuffer<[u32]>,      // pos_i
    Subbuffer<[[f32; 3]]>, // pos
    Subbuffer<[u32]>,      // rot_i
    Subbuffer<[[f32; 4]]>, //rot
    Subbuffer<[u32]>,      //scl_i
    Subbuffer<[[f32; 3]]>, // scl
);

// lazy_static::lazy_static! {
// }
// pub static TRANSFORM_MAP: *const SendSync<Mutex<HashMap<i32, i32>>> = std::ptr::null_mut(); //Lazy<Mutex<HashMap<i32, i32>>> = Lazy::new(|| Mutex::new(HashMap::new()));
pub static mut TRANSFORM_MAP: *mut HashMap<i32, i32> = std::ptr::null_mut();

pub static mut TRANSFORMS: *mut Transforms = std::ptr::null_mut();

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct TransformRef {
    #[serde(deserialize_with = "transform_map")]
    id: i32,
}
impl Default for TransformRef {
    fn default() -> Self {
        Self { id: -1 }
    }
}
impl TransformRef {
    pub fn get(&self) -> Option<Transform> {
        unsafe { &*TRANSFORMS }.get(self.id)
    }
}
fn transform_map<'de, D>(deserializer: D) -> Result<i32, D::Error>
where
    D: Deserializer<'de>,
{
    let s: i32 = Deserialize::deserialize(deserializer)?;
    println!("transform ref: {}", s);
    Ok(*unsafe { &*TRANSFORM_MAP }.get(&s).unwrap())
}
impl<'a> Inpsect for Ins<'a, TransformRef> {
    fn inspect(&mut self, name: &str, ui: &mut egui::Ui, _sys: &Sys) -> bool {
        let label = format!("{}", self.0.id);
        // let drop_data = _sys.assets_manager.drag_drop_data.lock().clone();
        let drop_data = _sys.dragged_transform;
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(name));
            drop_target(ui, true, |ui| {
                let response = ui.add(egui::Label::new(label.as_str()));
                if response.hovered() && ui.input(|i| i.pointer.any_released()) {
                    // if let id = drop_data {
                    self.0.id = drop_data;
                    // }
                }
            });
        })
        .response
        .changed()
    }
}
pub struct Transforms {
    self_lock: Mutex<()>,
    mutex: SegVec<SyncUnsafeCell<Mutex<()>>>,
    positions: SegVec<SyncUnsafeCell<glm::Vec3>>,
    pub(crate) rotations: SegVec<SyncUnsafeCell<glm::Quat>>,
    scales: SegVec<SyncUnsafeCell<glm::Vec3>>,
    pub(super) valid: SegVec<SyncUnsafeCell<bool>>,
    meta: SegVec<SyncUnsafeCell<TransformMeta>>,
    pub(super) entity: SegVec<SyncUnsafeCell<Entity>>,
    last: i32,
    avail: Avail,
    // count: i32,
    updates: SegVec<SyncUnsafeCell<[bool; 3]>>,
    extent: i32,
    ids_cache: VecCache<i32>,
    pos_cache: VecCache<[f32; 4]>,
    rot_cache: VecCache<[f32; 4]>,
    scl_cache: VecCache<[f32; 4]>,
    new_trans_cache: VecCache<i32>,
}

#[allow(dead_code)]
impl Transforms {
    pub fn get_matrix(&self, t: i32) -> glm::Mat4 {
        let pos = self.get_position(t);
        let rot = self.get_rotation(t);
        let scl = self.get_scale(t);
        glm::scaling(&scl) * glm::quat_to_mat4(&rot) * glm::translation(&pos)
    }
    pub fn active(&self) -> usize {
        self.meta.len() - self.avail.len()
        // self.count as usize
    }
    pub fn len(&self) -> usize {
        self.meta.len()
    }
    pub fn last_active(&self) -> i32 {
        self.last + 1
    }
    pub fn get<'a>(&self, t: i32) -> Option<Transform> {
        // TODO: make option
        if unsafe { *self.valid[t as usize].get() } {
            Some(Transform {
                // thr_id: std::thread::current().id(),
                _lock: unsafe { (*self.mutex[t as usize].get()).lock() },
                id: t,
            })
        } else {
            None
        }
    }
    // pub fn clear(&mut self) {
    //     *self = Self::new();
    //     // self.positions.clear();
    //     // self.rotations.clear();
    //     // self.scales.clear();
    //     // self.meta.clear();
    //     // self.updates.clear();
    //     // self.avail = AtomicI32::new(0);
    //     // self.count = AtomicI32::new(0);
    //     // self.extent = 0;
    // }
    pub fn new() -> Transforms {
        Transforms {
            self_lock: Mutex::new(()),
            mutex: SegVec::new(),
            positions: SegVec::new(),
            rotations: SegVec::new(),
            scales: SegVec::new(),
            meta: SegVec::new(),
            entity: SegVec::new(),
            updates: SegVec::new(),
            valid: SegVec::new(),
            last: -1,
            avail: Avail::new(),
            // count: 0,
            extent: 0,
            ids_cache: VecCache::new(),
            pos_cache: VecCache::new(),
            rot_cache: VecCache::new(),
            scl_cache: VecCache::new(),
            new_trans_cache: VecCache::new(),
        }
    }

    pub(super) fn write_transform(&self, i: i32, t: _Transform) -> Transform<'_> {
        unsafe {
            *self.mutex[i as usize].get() = Mutex::new(());
            *self.positions[i as usize].get() = t.position;
            *self.rotations[i as usize].get() = t.rotation;
            *self.scales[i as usize].get() = t.scale;
            let meta = &mut (*self.meta[i as usize].get());
            *meta.child_id = Node::default();
            meta.children.clear();
            meta.parent = -1;
            // *self.meta[i as usize].get() = TransformMeta::new();
            *self.entity[i as usize].get() = Entity::new();
            *self.valid[i as usize].get() = true;
            // self.valid.set(i as usize, true);
            let u = &mut *self.updates[i as usize].get();
            u[POS_U] = true;
            u[ROT_U] = true;
            u[SCL_U] = true;
        }
        self.get(i).unwrap()
    }
    fn push_transform(&mut self, t: _Transform) {
        self.mutex.push(SyncUnsafeCell::new(Mutex::new(())));
        self.positions.push(SyncUnsafeCell::new(t.position));
        self.rotations.push(SyncUnsafeCell::new(t.rotation));
        self.scales.push(SyncUnsafeCell::new(t.scale));
        self.meta.push(SyncUnsafeCell::new(TransformMeta::new()));
        self.entity.push(SyncUnsafeCell::new(Entity::new()));
        self.valid.push(SyncUnsafeCell::new(true));
        self.updates.push(SyncUnsafeCell::new([true, true, true]));
    }
    fn get_next_id(&mut self) -> (bool, i32) {
        if let Some(i) = self.avail.pop() {
            self.last = self.last.max(i);
            (true, i)
        } else {
            let i = self.extent;
            self.extent += 1;
            self.last = i;
            (false, i)
        }
    }
    pub(crate) fn reduce_last(&mut self) {
        self.avail.commit();
        let mut id = self.last;
        while id >= 0 && !unsafe { *self.valid[id as usize].get() } {
            // not thread safe!
            id -= 1;
        }
        self.last = id;
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
        let (write, id) = self.get_next_id();
        if write {
            self.write_transform(id, transform);
        } else {
            self.push_transform(transform);
        }
        if id as usize >= self.positions.len() {
            panic!("next id invalid/out of range: {:?}", (write, id));
        }
        if parent > -1 {
            unsafe {
                let meta = &mut *self.meta[id as usize].get();
                meta.parent = parent;
                meta.child_id =
                    SendSync::new((*self.meta[parent as usize].get()).children.push_tail(id));
            }
        }
        id
    }
    fn reserve(&mut self, count: usize) {
        let c = self.avail.len().min(count);
        let c = count - c;
        if c > 0 {
            let c = c + self.len();
            self.mutex
                .resize_with(c, || SyncUnsafeCell::new(Mutex::new(())));
            self.positions
                .resize_with(c, || SyncUnsafeCell::new([0., 0., 0.].into()));
            self.rotations.resize_with(c, || {
                SyncUnsafeCell::new(glm::quat_look_at_lh(&Vec3::z(), &Vec3::y()))
            });
            self.scales
                .resize_with(c, || SyncUnsafeCell::new([1., 1., 1.].into()));
            self.meta
                .resize_with(c, || SyncUnsafeCell::new(TransformMeta::new()));
            self.entity
                .resize_with(c, || SyncUnsafeCell::new(Entity::new()));
            self.valid.resize_with(c, || SyncUnsafeCell::new(true));
            self.updates
                .resize_with(c, || SyncUnsafeCell::new([true, true, true]));
        }
    }

    // pub fn multi_transform_with_avail<T>(&mut self, count: i32, t_func: T) ->
    //     where
    //         T: Fn() -> _Transform + Send + Sync, {

    //         }
    // pub fn multi_transform(&mut self, parent:i32, count: i32) -> CacheVec<i32> {
    //     self.multi_transform_with(parent, count, || _Transform::default())
    // }
    pub(crate) fn _allocate(&mut self, count: usize) -> CacheVec<i32> {
        let mut r = self.new_trans_cache.get_vec(count);
        let c = self.avail.len().min(count as usize);
        self.reserve(count);
        let mut max = -1;
        for _ in 0..c {
            if let Some(i) = self.avail.pop() {
                max = i;
                r.push(i);
            }
        }
        // self.count += c as i32;
        self.last = self.last.max(max);
        // r.get().resize(count, -1);
        for i in (c..count) {
            self.last += 1;
            r.get().push(self.last);
        }
        self.extent = self.extent.max(self.last + 1);
        r
    }
    pub fn multi_transform_with(
        &mut self,
        count: usize,
        t_funcs: Vec<(i32, i32, Option<&Box<dyn Fn() -> _Transform + Send + Sync>>)>,
        offsets: &Vec<i32>,
    ) -> CacheVec<i32> {
        let mut r = self.new_trans_cache.get_vec(count);
        let c = self.avail.len().min(count as usize);
        self.reserve(count);
        let mut max = -1;
        for _ in 0..c {
            if let Some(i) = self.avail.pop() {
                max = i;
                r.push(i);
            }
        }
        // self.count += c as i32;
        self.last = self.last.max(max);

        // r.get().extend((c..count).enumerate().map(|i| i as i32 + self.last));
        r.get().resize(count, -1);
        for i in (c..count) {
            self.last += 1;
            r.get()[i] = self.last;
        }
        let default: Box<dyn Fn() -> _Transform + Send + Sync> = Box::new(|| _Transform::default());
        t_funcs.into_par_iter().zip_eq(offsets.par_iter()).for_each(
            |((parent, _count, t_func), o_id)| {
                (0.._count).into_par_iter().for_each(|i| {
                    let t_func = match t_func {
                        Some(f) => f,
                        None => &default,
                    };
                    self.write_transform(r.get()[(*o_id + i) as usize], t_func());
                });
            },
        );
        self.extent = self.extent.max(self.last + 1);
        r
    }
    pub fn remove(&self, t: i32) {
        self.avail.push(t);
        unsafe {
            let meta = &*self.meta[t as usize].get();
            if meta.parent >= 0 {
                let _ = (*self.mutex[meta.parent as usize].get()).lock();
                (&mut *self.meta[meta.parent as usize].get())
                    .children
                    .pop_node(&meta.child_id);
            }
            // *self.meta[t as usize].get() = TransformMeta::new();
            let meta = &mut (*self.meta[t as usize].get());
            *meta.child_id = Node::default();
            meta.children.clear();
            meta.parent = -1;
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
    pub(crate) fn clean(&mut self) {
        self.valid.iter().enumerate().for_each(|(i, mut a)| {
            if unsafe { *a.get() } {
                self.avail.push(i as i32);
            }
            unsafe {
                *a.get() = false;
            }
            // a.set(false)
        });
        // for
        // unsafe {
        //     self.positions.iter().for_each(|a| *a.get() = )
        // }
        // self.avail = ;
        // self.count = 0;
        self.last = -1;
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
        unsafe { quat_rotate_vec3(&*self.rotations[t as usize].get(), &glm::Vec3::z()) }
    }
    fn right(&self, t: i32) -> glm::Vec3 {
        unsafe { quat_rotate_vec3(&*self.rotations[t as usize].get(), &glm::Vec3::x()) }
    }
    fn up(&self, t: i32) -> glm::Vec3 {
        unsafe { quat_rotate_vec3(&*self.rotations[t as usize].get(), &glm::Vec3::y()) }
    }

    fn _move(&self, t: i32, v: Vec3) {
        unsafe {
            *self.positions[t as usize].get() += v;
        }
        self.u_pos(t);
    }
    fn move_child(&self, t: i32, v: Vec3) {
        self._move(t, v);

        for child_id in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            let child = self.get(*child_id).unwrap();
            self.move_child(child.id, v);
        }
    }
    fn rot_mul(&self, t: i32, v: &Vec3) -> Vec3 {
        unsafe { glm::quat_rotate_vec3(&*self.rotations[t as usize].get(), v) }
    }
    fn translate(&self, t: i32, mut v: Vec3) {
        v = self.rot_mul(t, &v);
        unsafe {
            *self.positions[t as usize].get() += v;
        }
        self.u_pos(t);
        for child_id in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            let child = self.get(*child_id).unwrap();
            self.move_child(child.id, v);
        }
    }
    fn get_position(&self, t: i32) -> Vec3 {
        unsafe { *self.positions[t as usize].get() }
    }
    fn set_position_(&self, t: i32, v: &Vec3) {
        self.u_pos(t);
        // let offset = v - self.get_position(t);
        // self.translate(t, offset);
        unsafe {
            *self.positions[t as usize].get() = *v;
        }
        //         player_transform.move_child(glm::quat_rotate_vec3(&rotation, &(vel * speed)));
        //     }
        // });
        // for child_id in unsafe { (*self.meta[t as usize].get()).children.iter() } {
        //     let child = self.get(*child_id).unwrap();
        //     self.move_child(&child, offset);
        // }
    }
    fn set_position(&self, t: i32, v: &Vec3) {
        // self.u_pos(t);
        let offset = v - self.get_position(t);
        self.move_child(t, offset);
        // unsafe {
        //     *self.positions[t as usize].get() = v;
        // }
        //         player_transform.move_child(glm::quat_rotate_vec3(&rotation, &(vel * speed)));
        //     }
        // });
        // for child_id in unsafe { (*self.meta[t as usize].get()).children.iter() } {
        //     let child = self.get(*child_id).unwrap();
        //     self.move_child(&child, offset);
        // }
    }
    fn get_rotation(&self, t: i32) -> Quat {
        unsafe { *self.rotations[t as usize].get() }
    }
    fn look_at(&self, t: i32, dir: &Vec3, up: &Vec3) {
        let rot = Isometry3::face_towards(&[0., 0., 0.].into(), &[dir.x, dir.y, dir.z].into(), up);
        self.set_rotation(t, &rot.rotation.coords.into());
    }
    fn set_rotation(&self, t: i32, r: &Quat) {
        self.u_rot(t);
        let r_l = unsafe { &mut *self.rotations[t as usize].get() };
        // let rot = r * (glm::quat_conjugate(&*r_l) / glm::quat_dot(&*r_l, &*r_l));

        let mut rot: Option<Quat> = if unsafe { (*self.meta[t as usize].get()).children.len() } > 0
        {
            Some(r * r_l.simd_try_inverse().simd_unwrap())
        } else {
            None
        };
        *r_l = *r;
        let pos = unsafe { *self.positions[t as usize].get() };
        if let Some(rot) = rot {
            for child_id in unsafe { (*self.meta[t as usize].get()).children.iter() } {
                let child = self.get(*child_id).unwrap();
                self.set_rotation_child(&child, &rot, &pos)
            }
        }
    }
    fn set_rotation_(&self, t: i32, r: &Quat) {
        self.u_rot(t);
        let r_l = unsafe { &mut *self.rotations[t as usize].get() };
        // let rot = r * (glm::quat_conjugate(&*r_l) / glm::quat_dot(&*r_l, &*r_l));
        // let rot = r * r_l.simd_try_inverse().simd_unwrap();
        *r_l = *r;
        // let pos = unsafe { *self.positions[t as usize].get() };
        // for child_id in unsafe { (*self.meta[t as usize].get()).children.iter() } {
        //     let child = self.get(*child_id).unwrap();
        //     self.set_rotation_child(&child, &rot, &pos)
        // }
    }
    fn set_rotation_child(&self, t: &Transform, rot: &Quat, pos: &Vec3) {
        let rotat = unsafe { &mut *self.rotations[t.id as usize].get() };
        let posi = unsafe { &mut *self.positions[t.id as usize].get() };

        *posi = pos + glm::quat_rotate_vec3(rot, &(*posi - pos));
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
            let child = self.get(*child_id).unwrap();
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
        let mut ax = glm::quat_rotate_vec3(&rot, &axis);
        for child_id in unsafe { (*self.meta[t as usize].get()).children.iter() } {
            let child = self.get(*child_id).unwrap();
            self.rotate_child(&child, &ax, &pos, radians);
        }
    }

    fn rotate_child(&self, t: &Transform, ax: &Vec3, pos: &Vec3, radians: f32) {
        self.u_rot(t.id);
        self.u_pos(t.id);
        let rot = unsafe { &mut *self.rotations[t.id as usize].get() };
        let p = unsafe { &mut *self.positions[t.id as usize].get() };

        *p = pos + glm::rotate_vec3(&(*p - pos), radians, ax);
        *rot = glm::quat_rotate(
            rot,
            radians,
            &(glm::quat_rotate_vec3(&glm::quat_inverse(&*rot), &ax)),
        );
        for child in t.get_children() {
            self.rotate_child(&child, ax, pos, radians);
        }
    }

    fn get_parent(&self, t: i32) -> i32 {
        unsafe { (*self.meta[t as usize].get()).parent }
    }

    pub fn get_transform_data_updates(&mut self, transforms_buf: TransformBuf) {
        let last = self.last as usize + 1;
        // let _positions = &self.positions[0..last];
        // let _rotations = &self.rotations[0..last];
        // let _scales = &self.scales[0..last];
        // let _updates = &self.updates[0..last];
        let len = last;
        // let transform_data = Arc::new(Mutex::new(TransformData {
        //     pos_id: Vec::new(),
        //     rot_id: Vec::new(),
        //     scl_id: Vec::new(),
        //     pos_data: Vec::new(),
        //     rot_data: Vec::new(),
        //     scl_data: Vec::new(),
        //     extent: self.extent as usize,
        // }));
        // let transform_data = Arc::new(Mutex::new(
        //    (Vec::<VecCache::<i32>::new()>),
        // ));
        let self_ = Arc::new(&self);
        // println!("vec cache size: {}", self.pos_cache.count.load(Ordering::Relaxed));
        let pos_i = SyncUnsafeCell::new(transforms_buf.0.write().unwrap());
        let pos = SyncUnsafeCell::new(transforms_buf.1.write().unwrap());
        let rot_i = SyncUnsafeCell::new(transforms_buf.2.write().unwrap());
        let rot = SyncUnsafeCell::new(transforms_buf.3.write().unwrap());
        let scl_i = SyncUnsafeCell::new(transforms_buf.4.write().unwrap());
        let scl = SyncUnsafeCell::new(transforms_buf.5.write().unwrap());
        // unsafe {
        //     *(*pos_i.get()).last_mut().unwrap() = 0;
        //     *(*rot_i.get()).last_mut().unwrap() = 0;
        //     *(*scl_i.get()).last_mut().unwrap() = 0;
        // }
        // rayon::scope(|s| {
        // let num_jobs = (len / 4096).max(1); // TODO: find best number dependent on cpu
        // let num_jobs = num_cpus::get().min(last / 2048).max(1); // TODO: find best number dependent on cpu
        // for id in 0..num_jobs {
        // let start = len / num_jobs * id;
        // let mut end = start + len / num_jobs;
        // if id == num_jobs - 1 {
        //     end = len;
        // }
        // let end = end;
        // let transform_data = transform_data.clone();
        let self_ = self_.clone();
        // let transforms_buf = &transforms_buf;
        let pos_i = &pos_i;
        let pos = &pos;
        let rot_i = &rot_i;
        let rot = &rot;
        let scl_i = &scl_i;
        let scl = &scl;

        let p = &self_.positions;
        let r = &self_.rotations;
        let s = &self_.scales;
        let _u = &self_.updates;
        // let len = scl.len();
        (0..len)
            .into_par_iter()
            .chunks(32)
            .enumerate()
            .for_each(|(id, _i)| {
                let mut pos_mask = 0u32;
                let mut rot_mask = 0u32;
                let mut scl_mask = 0u32;
                let mut bit = 0;
                unsafe {
                    // let len = _i.len();
                    // let the_rest = 32 - len;
                    for i in _i {
                        if !*self_.valid[i].get() {
                            bit += 1;
                            // pos_mask = pos_mask << 1;
                            // rot_mask = rot_mask << 1;
                            // scl_mask = scl_mask << 1;
                            continue;
                        }
                        let u = &mut *_u[i].get();
                        if u[POS_U] {
                            let p = &*p[i].get();
                            (*pos.get())[i] = [p.x, p.y, p.z];
                            pos_mask |= 1 << bit;
                            // (*pos_i.get())[i] = 1;
                            // p_ids.push(i as i32);
                            // pos.push([p.x, p.y, p.z, 0.]);
                            u[POS_U] = false;
                        }
                        if u[ROT_U] {
                            let r = &*r[i].get();
                            (*rot.get())[i] = r.coords.into();
                            rot_mask |= 1 << bit;
                            // (*rot_i.get())[i] = 1;
                            // r_ids.push(i as i32);
                            // rot.push([r.w, r.k, r.j, r.i]);
                            u[ROT_U] = false;
                        }
                        if u[SCL_U] {
                            let s = &*s[i].get();
                            (*scl.get())[i] = [s.x, s.y, s.z];
                            scl_mask |= 1 << bit;
                            // (*scl_i.get())[i] = 1;
                            // s_ids.push(i as i32);
                            // scl.push([s.x, s.y, s.z, 0.]);
                            u[SCL_U] = false;
                        }
                        bit += 1;
                        // pos_mask <<= 1;
                        // rot_mask <<= 1;
                        // scl_mask <<= 1;
                    }
                    // pos_mask <<= the_rest;
                    // rot_mask <<= the_rest;
                    // scl_mask <<= the_rest;
                    (*pos_i.get())[id] = pos_mask;
                    (*rot_i.get())[id] = rot_mask;
                    (*scl_i.get())[id] = scl_mask;
                }
            });
        // s.spawn(move |_| {
        //     let len = end - start;
        // let _pos_ids = self_.ids_cache.get_vec(len);
        // let _rot_ids = self_.ids_cache.get_vec(len);
        // let _scl_ids = self_.ids_cache.get_vec(len);

        // let _pos = self_.pos_cache.get_vec(len);
        // let _rot = self_.rot_cache.get_vec(len);
        // let _scl = self_.scl_cache.get_vec(len);

        // {
        // let mut p_ids = _pos_ids.get();
        // let mut r_ids = _rot_ids.get();
        // let mut s_ids = _scl_ids.get();

        // let mut pos = _pos.get();
        // let mut rot = _rot.get();
        // let mut scl = _scl.get();

        // let p = &self_.positions;
        // let r = &self_.rotations;
        // let s = &self_.scales;
        // let _u = &self_.updates;

        // for i in start..end {
        //     unsafe {
        //         if !*self_.valid[i].get() {
        //             continue;
        //         }
        //         let u = &mut *_u[i].get();
        //         if u[POS_U] {
        //             let p = &*p[i].get();
        //             (*pos.get())[i] = [p.x, p.y, p.z];
        //             (*pos_i.get())[i] = 1;
        //             // p_ids.push(i as i32);
        //             // pos.push([p.x, p.y, p.z, 0.]);
        //             u[POS_U] = false;
        //         } else {
        //             (*pos_i.get())[i] = 0;
        //         }
        //         if u[ROT_U] {
        //             let r = &*r[i].get();
        //             (*rot.get())[i] = [r.w, r.k, r.j, r.i];
        //             (*rot_i.get())[i] = 1;
        //             // r_ids.push(i as i32);
        //             // rot.push([r.w, r.k, r.j, r.i]);
        //             u[ROT_U] = false;
        //         } else {
        //             (*rot_i.get())[i] = 0;
        //         }
        //         if u[SCL_U] {
        //             let s = &*s[i].get();
        //             (*scl.get())[i] = [s.x, s.y, s.z];
        //             (*scl_i.get())[i] = 1;
        //             // s_ids.push(i as i32);
        //             // scl.push([s.x, s.y, s.z, 0.]);
        //             u[SCL_U] = false;
        //         } else {
        //             (*scl_i.get())[i] = 0;
        //         }
        //     }
        // }
        // }

        // let ret = Arc::new((transform_ids, pos, rot, scl));
        // transform_data.lock().push(ret);

        // let mut td = transform_data.lock();
        // td.pos_data.push(_pos);
        // td.pos_id.push(_pos_ids);
        // td.rot_data.push(_rot);
        // td.rot_id.push(_rot_ids);
        // td.scl_data.push(_scl);
        // td.scl_id.push(_scl_ids);
        //         });
        //     }
        // });
        // Arc::into_inner(transform_data).unwrap().into_inner()
    }
}
