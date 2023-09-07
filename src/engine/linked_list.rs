use std::cmp::Reverse;

use dary_heap::DaryHeap;

pub struct LinkedList<T> {
    data: Vec<T>,
    next: Vec<i32>,
    prev: Vec<i32>,
    front: Option<i32>,
    back: Option<i32>,
    avail: DaryHeap<Reverse<i32>, 2>,
}

impl<T> LinkedList<T> {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            next: Vec::new(),
            prev: Vec::new(),
            avail: DaryHeap::new(),
            front: None,
            back: None,
        }
    }
    pub fn push_back(&mut self, d: T) -> i32 {
        if let Some(last) = &mut self.back {
            if let Some(Reverse(id)) = self.avail.pop() {
                self.data[id as usize] = d;
                self.next[*last as usize] = id;
                self.prev[id as usize] = *last;
                *last = id;
                id
            } else {
                let id = self.data.len() as i32;
                self.data.push(d);
                self.next.push(-1);
                self.prev.push(*last);
                self.next[*last as usize] = id;
                *last = id;
                id
            }
        } else {
            self.front = Some(0);
            self.back = Some(0);
            self.data.push(d);
            self.next.push(-1);
            self.prev.push(-1);
            0
        }
    }
    pub fn len(&self) -> usize {
        self.data.len() - self.avail.len()
    }
    pub fn remove(&mut self, id: i32) {
        self.avail.push(Reverse(id));
        let next = self.next[id as usize];
        let prev = self.prev[id as usize];
        if self.len() > 0 {
            if prev >= 0 { // not front
                self.next[prev as usize] = next;
            } else {
                self.front = Some(next)
            }
            if next >= 0 {
                self.prev[next as usize] = prev;
            }
        } else {
            self.front = None;
            self.back = None;
        }
    }
    pub fn push_next(&mut self, before: i32, d: T) {
        if let Some(Reverse(id)) = self.avail.pop() {
            self.data[id as usize] = d;
            let b_next = self.next[before as usize];
            self.next[id as usize] = b_next;
            self.next[before as usize] = id;
            self.prev[id as usize] = before;
            self.prev[b_next as usize] = id;

        }
    }
}

