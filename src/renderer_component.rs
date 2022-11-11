use std::{collections::HashMap, rc::Rc, sync::Arc};

use crate::{
    engine::{transform::Transform, Component, Storage, Sys, World},
    fast_buffer,
    renderer::Id,
};
use parking_lot::{Mutex, RwLock};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool},
    device::Device,
    pipeline::ComputePipeline,
    shader::ShaderModule, command_buffer::DrawIndexedIndirectCommand,
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
    pub transform_ids_gpu: Arc<CpuAccessibleBuffer<[i32]>>,
    pub renderers_gpu: Arc<CpuAccessibleBuffer<[i32]>>,
    pub updates_gpu: Arc<CpuAccessibleBuffer<[i32]>>,
    pub indirect: Option<Arc<CpuAccessibleBuffer<[DrawIndexedIndirectCommand]>>>,
    pub transforms_gpu_len: i32,
    pub transform_updates: HashMap<i32, i32>,
}

pub mod ur {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/update_renderers.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub struct RendererManager {
    pub renderers: RwLock<HashMap<i32, RendererInstances>>,
    pub device: Arc<Device>,
    pub shader: Arc<ShaderModule>,
    pub pipeline: Arc<ComputePipeline>,
    pub uniform: CpuBufferPool<ur::ty::Data>,
}

pub struct Offset {
    pub offset: u32,
    pub count: u32,
    pub model_id: i32,
}

impl RendererManager {
    pub fn new(device: Arc<Device>) -> RendererManager {
        let shader = ur::load(device.clone()).unwrap();

        // Create compute-pipeline for applying compute shader to vertices.
        let pipeline = vulkano::pipeline::ComputePipeline::new(
            device.clone(),
            shader.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create compute shader");

        RendererManager {
            renderers: RwLock::new(HashMap::new()),
            device: device.clone(),
            shader,
            pipeline,
            uniform: CpuBufferPool::<ur::ty::Data>::new(device.clone(), BufferUsage::all()),
        }
    }

    pub fn get_instances(&self) -> Arc<(Vec<Offset>, Vec<Id>)> {
        // let mut offsets = Vec::<Offset>::new();

        // let cap = {
        //     self.renderers
        //         .read()
        //         .iter()
        //         .map(|(_, rm)| rm.transforms.data.len())
        //         .sum()
        // };
        // let mut all_instances = Vec::<Id>::with_capacity(cap);
        // for (id, rm) in self.renderers.read().iter() {
        //     // if let Some(mr) = mm.models_ids.get(&id) {
        //     let chunk_size = (rm.transforms.data.len() / num_cpus::get()).max(1);
        //     let instances = rm
        //         .transforms
        //         .data
        //         .par_iter()
        //         .chunks(chunk_size)
        //         .map(|slice| {
        //             let mut v = Vec::<Id>::with_capacity(chunk_size);

        //             for i in slice {
        //                 if let Some(i) = i {
        //                     v.push(*i);
        //                 }
        //             }
        //             v
        //         })
        //         .collect::<Vec<Vec<Id>>>();
        //     let count: u32 = instances.iter().map(|v| v.len() as u32).sum();
        //     // let instances = rm
        //     //     .transforms
        //     //     .data
        //     //     .iter()
        //     //     .filter(|x| x.is_some())
        //     //     .map(|x| x.unwrap())
        //     //     .collect::<Vec<Id>>();
        //     // let count = instances.iter().map(|x| x.len()).sum::<usize>() as u32;
        //     if let Some(last_offset) = offsets.last() {
        //         let last_offset = last_offset.clone();
        //         offsets.push(Offset {
        //             offset: last_offset.offset + last_offset.count,
        //             count,
        //             model_id: *id,
        //         });
        //     } else {
        //         offsets.push(Offset {
        //             offset: 0,
        //             count,
        //             model_id: *id,
        //         });
        //     }
        //     for insts in instances {
        //         all_instances.extend(insts);
        //     }
        //     // rend.bind_mesh(&mut builder, instances, count, curr_mvp_buffer.clone(), &mr.mesh);

        //     // }
        // }
        // // let instances = fast_buffer(device.clone(), &all_instances);
        Arc::new((vec![], vec![]))
    }
}

impl Renderer {
    // pub fn from(t: Transform, r: &Renderer, rm: &mut RendererManager) -> Renderer {
    //     let ri_id = if let Some(ri) = rm.renderers.get_mut(&&r.model_id) {
    //         ri.transforms.emplace(Id { id: t.0 })
    //     } else {
    //         let mut ri = RendererInstances {
    //             model_id: r.model_id,
    //             transforms: Storage::new(false),
    //             transforms_gpu: CpuAccessibleBuffer::from_iter( vec![0])
    //         };
    //         let ri_id = ri.transforms.emplace(Id { id: t.0 });
    //         rm.renderers.insert(r.model_id, ri);
    //         ri_id
    //     };
    //     Renderer {
    //         model_id: r.model_id,
    //         t,
    //         id: ri_id,
    //     }
    // }
    pub fn new(t: Transform, model_id: i32) -> Renderer {
        Renderer {
            model_id: model_id,
            t,
            id: 0,
        }
    }
}

impl Component for Renderer {
    fn init(&mut self, t: Transform, sys: &mut Sys) {
        self.t = t;
        let rm = &mut sys.renderer_manager.write();
        self.id = if let Some(ri) = rm.renderers.write().get_mut(&self.model_id) {
            let id = ri.transforms.emplace(Id { id: t.0 });
            ri.transform_updates.insert(id, t.0);
            id
        } else {
            -1
        };
        if self.id == -1 {
            self.id = {
                let mut ri = RendererInstances {
                    model_id: self.model_id,
                    transforms: Storage::new(false),
                    transform_ids_gpu: CpuAccessibleBuffer::from_iter(
                        rm.device.clone(),
                        BufferUsage::all(),
                        true,
                        vec![t.0],
                    )
                    .unwrap(),
                    renderers_gpu: CpuAccessibleBuffer::from_iter(
                        rm.device.clone(),
                        BufferUsage::all(),
                        true,
                        vec![0],
                    )
                    .unwrap(),
                    updates_gpu: CpuAccessibleBuffer::from_iter(
                        rm.device.clone(),
                        BufferUsage::all(),
                        true,
                        vec![0],
                    )
                    .unwrap(),
                    indirect: None,
                    // indirect: CpuAccessibleBuffer::from_iter(
                    //     rm.device.clone(),
                    //     BufferUsage::all(),
                    //     true,
                    //     vec![DrawIndexedIndirectCommand {index_count: 0, instance_count: 0, first_index: 0, vertex_offset: 0,first_instance: 0}],
                    // )
                    // .unwrap(),
                    transforms_gpu_len: 1,
                    transform_updates: HashMap::new(),
                };
                let ri_id = ri.transforms.emplace(Id { id: t.0 });
                ri.transform_updates.insert(ri_id, t.0);
                rm.renderers.write().insert(self.model_id, ri);
                ri_id
            }
        }
    }
    fn deinit(&mut self, t: Transform, sys: &mut Sys) {
        // let rm = ;
        if let Some(ri) = sys.renderer_manager.write().renderers.write().get_mut(&self.model_id) {
            ri.transforms.erase(self.id);
            ri.transform_updates.insert(self.id, -1);
        }
    }
    fn update(&mut self, sys: &crate::engine::System) {}
}
