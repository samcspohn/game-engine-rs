use parking_lot::Mutex;
use segvec::{SegVec, Slice, SliceMut};
use std::{
    cell::SyncUnsafeCell,
    sync::atomic::{AtomicUsize, Ordering},
};

pub struct AtomicVec<T> {
    lock: Mutex<()>,
    data: SyncUnsafeCell<SegVec<std::mem::MaybeUninit<T>>>,
    index: AtomicUsize,
}

impl<T> AtomicVec<T> {
    pub fn new() -> Self {
        let mut data = SegVec::with_capacity(4);
        // unsafe {
        //     data.set_len(4);
        // }
        Self {
            lock: Mutex::new(()),
            data: SyncUnsafeCell::new(data),
            index: AtomicUsize::new(0),
        }
    }
    pub fn len(&self) -> usize {
        self.index.load(Ordering::Relaxed)
    }
    // pub fn push_multi<'a>(&mut self, count: usize) -> (MutexGuard<()>, &'a [T]) {
    //     let _l = self.lock.lock();
    //     unsafe {
    //         let index = self.index.load(Ordering::Relaxed);
    //         (*self.data.get()).reserve(count);
    //         (*self.data.get()).set_len(index + count);
    //         self.index.fetch_add(count, Ordering::Relaxed);
    //         (_l, &(*self.data.get())[index..(index + count)])
    //     }
    // }
    pub fn push(&self, d: T) {
        let i = self.index.fetch_add(1, Ordering::SeqCst);
        if i < unsafe { (*self.data.get()).len() } {
            unsafe { (*self.data.get())[i].write(d) };
        } else {
            unsafe {
                let _l = self.lock.lock();
                while i >= (*self.data.get()).len() {
                    let data = &mut (*self.data.get());
                    let len = data.len() + 1;
                    let new_len = len.next_power_of_two();
                    data.resize_with(new_len, || std::mem::MaybeUninit::<T>::uninit())
                }
                (*self.data.get())[i].write(d);
            };
        }
    }
    pub fn get<'a>(&self) -> Slice<'a, std::mem::MaybeUninit<T>> {
        let len = self.index.load(Ordering::Relaxed);
        unsafe { (*self.data.get()).slice(0..len) }
    }
    pub fn get_mut<'a>(&mut self) -> SliceMut<'a, std::mem::MaybeUninit<T>> {
        let len = self.index.load(Ordering::Relaxed);
        unsafe { (*self.data.get()).slice_mut(0..len) }
    }
    pub fn clear(&mut self) {
        let len = self.index.load(Ordering::Relaxed);
        let a = unsafe { (*self.data.get()).slice_mut(0..len) };
        for b in a {
            unsafe {
                b.assume_init_drop();
            }
        }
        self.index.store(0, Ordering::Relaxed);
    }
}
impl<T: Copy> AtomicVec<T> {
    pub fn get_vec(&self) -> Vec<T> {
        // let _l = self.lock.lock();
        // let mut v = Vec::new();
        let len = self.index.load(Ordering::Relaxed);
        // unsafe {
        //     let cap = (*self.data.get()).capacity();
        //     std::mem::swap(&mut v, &mut *self.data.get());
        //     v.set_len(len);
        //     *self.data.get() = Vec::with_capacity(cap);
        //     (*self.data.get()).set_len(cap);
        // }
        // v
        let mut v = Vec::with_capacity(len);
        // std::mem::swap(&mut v, unsafe { &mut *self.data.get() });
        // unsafe {
        //     unsafe { &mut *self.data.get() }.set_len(len.next_power_of_two());
        // }
        for i in (0..len).into_iter() {
            unsafe { v.push((*self.data.get())[i].assume_init()) };
        }
        // self.clear();
        self.index.store(0, Ordering::Relaxed);
        v
    }
}
