use std::{collections::BTreeMap, time::{Duration, Instant}};

use crossbeam::queue::SegQueue;
use segvec::SegVec;

struct SubPerf {
    data: SegVec<Duration>,
    outliers: Vec<f32>,
    avg: f32,
}
impl SubPerf {
    fn new() -> Self {
        SubPerf {
            data: SegVec::new(),
            outliers: Vec::new(),
            avg: 0f32,
        }
    }
    fn push(&mut self, d: Duration) {
        let dr = (d.as_nanos() as u128) as f32 / 1e6;
        self.data.push(d);
        if self.data.len() > 50 {
            if dr > self.avg * 3. {
                self.outliers.push(dr);
            }
            self.data.remove(0);
        }
        self.avg = (self.avg * (self.data.len() as f32 - 1.) + dr) / self.data.len() as f32;
    }
    fn print(&mut self, name: &str) {
        let len = self.data.len();
        println!(
            "{}: {:?} ms",
            name,
            (self.data.iter().sum::<Duration>().as_nanos() / len as u128) as f32 / 1e6
        );
        self.outliers
            .sort_by(|a, b| (-a).partial_cmp(&(-b)).unwrap());
        if self.outliers.len() > 0 {
            println!(
                "              outliers: {:?}",
                self.outliers[0..4.min(self.outliers.len())].to_vec()
            );
        }
    }
}

pub struct Perf {
    data: BTreeMap<String, SubPerf>,
}

pub struct PerfNode<'a> {
    id: String,
    perf: &'a mut Perf,
    inst: Instant,
}
impl Drop for PerfNode<'_> {
    fn drop(&mut self) {
        self.perf.update(self.id.clone(), Instant::now() - self.inst);
    }
}
impl Perf {
    pub fn new() -> Self {
        Self {
            data: BTreeMap::new(),
        }
    }
    pub fn node(&mut self, k: &str) -> PerfNode {
        PerfNode { id: k.into(), perf: self, inst: Instant::now() }
    }
    pub fn update(&mut self, k: String, d: Duration) {
        let perf = &mut self.data;
        if let Some(q) = perf.get_mut(&k) {
            q.push(d);
        } else {
            perf.insert(k.clone(), SubPerf::new());
            if let Some(q) = perf.get_mut(&k) {
                q.push(d);
            }
        }
    }

    pub fn print(&mut self) {
        let mut p = {
            let mut a = BTreeMap::new();
            std::mem::swap(&mut a, &mut self.data);
            // self.data = BtreeMap::new();
            a.into_iter()
        };
        // let p = self.data.into_iter();
        for (k, mut x) in p {
            x.print(&k);
            // let len = x.len();
            // println!(
            //     "{}: {:?} ms",
            //     k,
            //     (x.into_iter().sum::<Duration>().as_nanos() / len as u128) as f32 / 1e6
            // );
        }
    }
}
