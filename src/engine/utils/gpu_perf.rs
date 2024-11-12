use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use crossbeam::queue::SegQueue;
use parking_lot::RwLock;
use segvec::SegVec;
use vulkano::command_buffer::PrimaryAutoCommandBuffer;
use crate::engine::rendering::vulkan_manager::VulkanManager;

use super::{perf::SubPerf, PrimaryCommandBuffer};
// struct SubPerf {
//     data: RwLock<SegVec<Duration>>,
//     outliers: RwLock<Vec<(Instant, f32)>>,
//     avg: RwLock<f32>,
// }
// impl SubPerf {
//     fn new() -> Self {
//         SubPerf {
//             data: RwLock::new(SegVec::new()),
//             outliers: RwLock::new(Vec::new()),
//             avg: RwLock::new(0f32),
//         }
//     }
//     fn push(&self, d: Duration) {
//         let dr = (d.as_nanos() as u128) as f32 / 1e6;
//         let mut data = self.data.write();
//         data.push(d);
//         if data.len() > 50 {
//             if dr > *self.avg.read() * 3. {
//                 self.outliers.write().push((Instant::now(), dr));
//             }
//             data.remove(0);
//         }
//         let mut avg = self.avg.write();
//         *avg = (*avg * (data.len() as f32 - 1.) + dr) / data.len() as f32;
//     }
//     fn print(&mut self, name: &str, start_time: Instant) {
//         let mut data = self.data.write();
//         let mut outliers = self.outliers.write();
//         let len = data.len();
//         println!(
//             "{}: {:?} ms",
//             name,
//             (data.iter().sum::<Duration>().as_nanos() / len as u128) as f32 / 1e6
//         );
//         outliers.sort_by(|a, b| (-a.1).partial_cmp(&(-b.1)).unwrap());
//         if outliers.len() > 0 {
//             println!(
//                 "              outliers: {:?}",
//                 outliers[0..4.min(outliers.len())]
//                     .iter()
//                     .map(|(inst, dur)| { ((start_time - *inst).as_millis() / 1000, dur) })
//                     .collect::<Vec<_>>()
//             );
//         }
//     }
// }

type GPUPerfData = RwLock<BTreeMap<String, (i32, SubPerf)>>;
pub struct GpuPerf {
    start_time: Instant,
    pub(crate) data: Arc<GPUPerfData>,
    vk: Arc<VulkanManager>,
}

pub struct PerfNode {
    id: String,
    perf: Arc<GPUPerfData>,
    query: i32,
    // builder: &'a mut super::PrimaryCommandBuffer,
    vk: Arc<VulkanManager>,
    // inst: Instant,
}
// impl Drop for PerfNode {
//     fn drop(&mut self) {
//         GPUPerf::update(
//             self.perf.clone(),
//             self.id.clone(),
//             Instant::now() - self.inst,
//         );
//     }
// }
// impl PerfNode {
//     pub fn get_elapsed(&self) -> Duration {
//         Instant::now() - self.inst
//     }
// }
impl PerfNode {
    pub fn end(&self, builder: &mut PrimaryCommandBuffer) {
        self.vk.end_query(&self.query, builder);
    }
}
// impl<'a> Drop for PerfNode<'a> {
//     fn drop(&mut self) {
//         self.vk.end_query(&self.query, &mut self.builder);
//         // self.builder.write().end_query(self.query);
//         // self.builder.write().reset_query_pool(self.vk.query_pool(), self.query, 1);
//         // self.builder.write().write_timestamp(self.vk.query_pool(), self.query);
//         // GPUPerf::update(
//         //     self.perf.clone(),
//         //     self.id.clone(),
//         //     self.vk.get_query(self.query),
//         // );
//     }
// }
impl GpuPerf {
    pub fn new(vk: Arc<VulkanManager>) -> Self {
        Self {
            start_time: Instant::now(),
            data: Arc::new(RwLock::new(BTreeMap::new())),
            vk,
        }
    }
    pub fn node(&self, k: &str, builder: &mut PrimaryCommandBuffer) -> PerfNode {
        let mut q_id = -1;
        let b = {
            let a = self.data.read();
            if let Some(q) = a.get(k) {
                q_id = q.0;
                // q.push(d);
                false
            } else {
                true
            }
        };
        if b {
            let mut a = self.data.write();
            q_id = self.vk.new_query();
            a.insert(k.into(), (q_id, SubPerf::new()));
            // if let Some(q) = a.get(&k) {
            //     // q.push(d)
            // }
        }
        self.vk.begin_query(&q_id, builder);
        PerfNode {
            id: k.into(),
            perf: self.data.clone(),
            query: q_id,
            // builder,
            vk: self.vk.clone(),
        }
    }
    fn update(data: Arc<GPUPerfData>, k: String, d: Duration) {
        // let b = {
        //     let a = data.read();
        //     if let Some(q) = a.get(&k) {
        //         q.1.push(d);
        //         false
        //     } else {
        //         true
        //     }
        // };
        // if b {
        //     let mut a = data.write();
        //     a.insert(k.clone(), SubPerf::new());
        //     if let Some(q) = a.get(&k) {
        //         q.1.push(d)
        //     }
        // }
        let a = data.read();
        if let Some(q) = a.get(&k) {
            q.1.push(d);
        }
    }

    pub fn print(&self) {
        let mut p = {
            let mut a = BTreeMap::new();
            std::mem::swap(&mut a, &mut self.data.write());
            // self.data = BtreeMap::new();
            a.into_iter()
        };

        for (k, mut x) in p {
            x.1.print(&k, Instant::now());
        }
    }
}
