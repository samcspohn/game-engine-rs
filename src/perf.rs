use std::{time::Duration, collections::HashMap};

use crossbeam::queue::SegQueue;

pub struct Perf {
    pub data: HashMap<String, SegQueue<Duration>>,

}

impl Perf {
    pub fn update(&mut self, k: String,d: Duration) {
        let perf = &mut self.data;
        if let Some(q) = perf.get_mut(&k) {
            q.push(d);
            if q.len() > 200 {
                q.pop();
            }
        } else {
            perf.insert(k.clone(), SegQueue::new());
            if let Some(q) = perf.get_mut(&k) {
                q.push(d);
                if q.len() > 200 {
                    q.pop();
                }
            }

        }

    }

    pub fn print(&mut self) {

        let p = {
            
            let mut a = HashMap::new();
            std::mem::swap(&mut a, &mut self.data);
            // self.data = HashMap::new();
            a.into_iter()
        };
        // let p = self.data.into_iter();
        for (k, x) in p {
            let len = x.len();
            println!("{}: {:?}", k, (x.into_iter().map(|a| a).sum::<Duration>() / len as u32));
        }
    }
}