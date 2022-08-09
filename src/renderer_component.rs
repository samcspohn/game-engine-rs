use std::{collections::HashMap, sync::Arc};

use crate::{
    engine::{transform::Transform, Component, Storage},
    fast_buffer,
    renderer::Id,
};
use component_derive::component;
use parking_lot::Mutex;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator, IntoParallelIterator, IndexedParallelIterator};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::Device,
};

#[component]
#[derive(Default, Clone, Copy)]
pub struct Renderer {
    model_id: i32,
    id: i32,
}

// #[derive(Default)]
pub struct RendererInstances {
    pub model_id: i32,
    // pub transforms: Vec<Id>
    pub transforms: Storage<Id>,
}

#[derive(Default)]
pub struct RendererManager {
    pub renderers: HashMap<i32, RendererInstances>,
}

pub struct Offset {
    pub offset: u32,
    pub count: u32,
    pub model_id: i32
}
impl RendererManager {
    pub fn get_instances(&self) -> Arc<(Vec<Offset>, Vec<Id>)> {
        let mut offsets = Vec::<Offset>::new();
        let mut all_instances = Vec::<Id>::new();
        for (id, rm) in &self.renderers {
            // if let Some(mr) = mm.models_ids.get(&id) {
            let chunk_size = (rm.transforms.data.len() / num_cpus::get()).max(1);
            let instances = rm.transforms.data.par_iter().chunks(chunk_size).map(|slice| {
                let mut v = Vec::<Id>::with_capacity(chunk_size);

                for i in slice {
                    if let Some(i) = i {
                        v.push(*i);
                    }
                }
                v
            }).collect::<Vec<Vec<Id>>>();
            // let instances = rm
            //     .transforms
            //     .data
            //     .iter()
            //     .filter(|x| x.is_some())
            //     .map(|x| x.unwrap())
            //     .collect::<Vec<Id>>();
            let count = instances.iter().map(|x| x.len()).sum::<usize>() as u32;
            if let Some(last_offset) = offsets.last() {
                let last_offset = last_offset.clone();
                offsets.push(Offset { offset: last_offset.offset + last_offset.count, count, model_id: *id});
            } else {
                offsets.push(Offset { offset: 0, count, model_id: *id});
            }
            for insts in instances {
                all_instances.extend(insts);
            }
            // rend.bind_mesh(&mut builder, instances, count, curr_mvp_buffer.clone(), &mr.mesh);

            // }
        }
        // let instances = fast_buffer(device.clone(), &all_instances);
        Arc::new((offsets, all_instances))
    }
}

impl Renderer {
    pub fn from(t: Transform, r: &Renderer, rm: &mut RendererManager) -> Renderer {
        let ri_id = if let Some(ri) = rm.renderers.get_mut(&&r.model_id) {
            ri.transforms.emplace(Id { id: t.0 })
        } else {
            let mut ri = RendererInstances {
                model_id: r.model_id,
                transforms: Storage::new(),
            };
            let ri_id = ri.transforms.emplace(Id { id: t.0 });
            rm.renderers.insert(r.model_id, ri);
            ri_id
        };
        Renderer {
            model_id: r.model_id,
            t,
            id: ri_id,
        }
    }
    pub fn new(t: Transform, model_id: i32, rm: &mut RendererManager) -> Renderer {
        let ri_id = if let Some(ri) = rm.renderers.get_mut(&model_id) {
            ri.transforms.emplace(Id { id: t.0 })
        } else {
            let mut ri = RendererInstances {
                model_id: model_id,
                transforms: Storage::new(),
            };
            let ri_id = ri.transforms.emplace(Id { id: t.0 });
            rm.renderers.insert(model_id, ri);
            ri_id
        };
        Renderer {
            model_id: model_id,
            t,
            id: ri_id,
        }
    }
}

impl Component for Renderer {
    fn init(&mut self, t: Transform) {}
    fn update(&mut self, sys: &crate::engine::System) {}
}
