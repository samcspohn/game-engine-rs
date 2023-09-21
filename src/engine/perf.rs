use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use crossbeam::queue::SegQueue;
use parking_lot::RwLock;
use segvec::SegVec;

struct SubPerf {
    data: RwLock<SegVec<Duration>>,
    outliers: RwLock<Vec<(Instant,f32)>>,
    avg: RwLock<f32>,
}
impl SubPerf {
    fn new() -> Self {
        SubPerf {
            data: RwLock::new(SegVec::new()),
            outliers: RwLock::new(Vec::new()),
            avg: RwLock::new(0f32),
        }
    }
    fn push(&self, d: Duration) {
        let dr = (d.as_nanos() as u128) as f32 / 1e6;
        let mut data = self.data.write();
        data.push(d);
        if data.len() > 50 {
            if dr > *self.avg.read() * 3. {
                self.outliers.write().push((Instant::now(), dr));
            }
            data.remove(0);
        }
        let mut avg = self.avg.write();
        *avg = (*avg * (data.len() as f32 - 1.) + dr) / data.len() as f32;
    }
    fn print(&mut self, name: &str, start_time: Instant) {
        let mut data = self.data.write();
        let mut outliers = self.outliers.write();
        let len = data.len();
        println!("{}:", name);
        println!(
            "              {:?} ms",
            (data.iter().sum::<Duration>().as_nanos() / len as u128) as f32 / 1e6
        );
        outliers.sort_by(|a, b| (-a.1).partial_cmp(&(-b.1)).unwrap());
        if outliers.len() > 0 {
            println!(
                "              outliers: {:?}",
                outliers[0..4.min(outliers.len())].iter().map(|(inst, dur)| {
                    ((start_time - *inst).as_millis() / 1000,dur)
                }).collect::<Vec<_>>()
            );
        }
    }
}

pub struct Perf {
    start_time: Instant,
    data: Arc<RwLock<BTreeMap<String, SubPerf>>>,
}

pub struct PerfNode {
    id: String,
    perf: Arc<RwLock<BTreeMap<String, SubPerf>>>,
    inst: Instant,
}
impl Drop for PerfNode {
    fn drop(&mut self) {
        Perf::update(
            self.perf.clone(),
            self.id.clone(),
            Instant::now() - self.inst,
        );
    }
}
impl Perf {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            data: Arc::new(RwLock::new(BTreeMap::new())),
        }
    }
    pub fn node(&self, k: &str) -> PerfNode {
        PerfNode {
            id: k.into(),
            perf: self.data.clone(),
            inst: Instant::now(),
        }
    }
    fn update(data: Arc<RwLock<BTreeMap<String, SubPerf>>>, k: String, d: Duration) {
        // let perf = &mut self.data;
        let b = {
            let a = data.read();
            if let Some(q) = a.get(&k) {
                q.push(d);
                false
            } else {
                true
            }
        };
        if b {
            let mut a = data.write();
            a.insert(k.clone(), SubPerf::new());
            if let Some(q) = a.get(&k) {
                q.push(d)
            }
        }
        // if let Some(q) = data.read().get(&k) {
        //     q.push(d);
        // } else {
        //     data.write().insert(k.clone(), SubPerf::new());
        //     if let Some(q) = data.read().get(&k) {
        //         q.push(d);
        //     }
        // }
    }

    pub fn print(&self) {
        let mut p = {
            let mut a = BTreeMap::new();
            std::mem::swap(&mut a, &mut self.data.write());
            // self.data = BtreeMap::new();
            a.into_iter()
        };

        for (k, mut x) in p {
            x.print(&k, Instant::now());
        }
    }
}
