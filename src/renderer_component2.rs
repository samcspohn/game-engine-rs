use std::{
    collections::{BTreeMap, HashMap},
    rc::Rc,
    sync::Arc,
};

use crate::{
    engine::{transform::Transform, Component, Storage, Sys, World},
    fast_buffer,
    renderer::Id,
};
use bytemuck::{Pod, Zeroable};
use component_derive::component;
use parking_lot::{Mutex, RwLock};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool},
    command_buffer::DrawIndexedIndirectCommand,
    device::Device,
    pipeline::ComputePipeline,
    shader::ShaderModule,
};

#[component]
#[derive(Default, Clone, Copy)]
pub struct Renderer {
    model_id: i32,
    id: i32,
}

// #[derive(Default)]
// pub struct RendererInstances {
//     pub model_id: i32,
//     // pub transforms: Vec<Id>
//     // pub transforms: Storage<Id>,
//     // pub transform_ids_gpu: Arc<CpuAccessibleBuffer<[i32]>>,
//     // pub renderers_gpu: Arc<CpuAccessibleBuffer<[i32]>>,
//     // pub updates_gpu: Arc<CpuAccessibleBuffer<[i32]>>,
//     // pub indirect: Option<Arc<CpuAccessibleBuffer<[DrawIndexedIndirectCommand]>>>,
//     pub transforms_gpu_len: i32,
//     pub transform_updates: HashMap<i32, i32>,
// }

pub mod ur {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/update_renderers2.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

#[derive(Clone, Copy)]
pub struct Indirect {
    pub id: i32,
    pub count: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct TransformId {
    pub indirect_id: i32,
    pub transform_id: i32,
}

pub struct RendererData {
    pub model_indirect: BTreeMap<i32, Indirect>,
    pub indirect_model: BTreeMap<i32, i32>,
    pub transforms_len: i32,

    pub updates: Vec<i32>,
}

pub struct SharedRendererData {
    pub transform_ids_gpu: Arc<CpuAccessibleBuffer<[TransformId]>>,
    pub renderers_gpu: Arc<CpuAccessibleBuffer<[i32]>>,
    pub updates_gpu: Arc<CpuAccessibleBuffer<[i32]>>,
    pub indirect: Storage<DrawIndexedIndirectCommand>,
    pub indirect_buffer: Arc<CpuAccessibleBuffer<[DrawIndexedIndirectCommand]>>,

    pub device: Arc<Device>,
    pub shader: Arc<ShaderModule>,
    pub pipeline: Arc<ComputePipeline>,
    pub uniform: Arc<CpuBufferPool<ur::ty::Data>>,
}

pub struct RendererManager {
    pub model_indirect: RwLock<BTreeMap<i32, Indirect>>,
    pub indirect_model: RwLock<BTreeMap<i32, i32>>,

    pub transforms: Storage<TransformId>,
    pub updates: HashMap<i32, TransformId>,
    pub shr_data: RwLock<SharedRendererData>,
    // pub transform_ids_gpu: Arc<CpuAccessibleBuffer<[TransformId]>>,
    // pub transforms_gpu_len: i32,
    // pub renderers_gpu: Arc<CpuAccessibleBuffer<[i32]>>,
    // pub updates_gpu: Arc<CpuAccessibleBuffer<[i32]>>,
    // pub indirect: Storage<DrawIndexedIndirectCommand>,
    // pub indirect_buffer: Arc<CpuAccessibleBuffer<[DrawIndexedIndirectCommand]>>,

    // pub device: Arc<Device>,
    // pub shader: Arc<ShaderModule>,
    // pub pipeline: Arc<ComputePipeline>,
    // pub uniform: Arc<CpuBufferPool<ur::ty::Data>>,
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
            model_indirect: RwLock::new(BTreeMap::new()),
            indirect_model: RwLock::new(BTreeMap::new()),
            updates: HashMap::new(),
            transforms: Storage::new(false),
            shr_data: RwLock::new(SharedRendererData {
                transform_ids_gpu: CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::all(),
                    true,
                    vec![TransformId {
                        indirect_id: -1,
                        transform_id: -1,
                    }],
                )
                .unwrap(),
                renderers_gpu: CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::all(),
                    true,
                    vec![0],
                )
                .unwrap(),
                updates_gpu: CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::all(),
                    true,
                    vec![0],
                )
                .unwrap(),
                indirect: Storage::new(false),
                indirect_buffer: CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::all(),
                    true,
                    vec![DrawIndexedIndirectCommand {
                        index_count: 0,
                        instance_count: 0,
                        first_index: 0,
                        vertex_offset: 0,
                        first_instance: 0,
                    }],
                )
                .unwrap(),
                device: device.clone(),
                shader,
                pipeline,
                uniform: Arc::new(CpuBufferPool::<ur::ty::Data>::new(
                    device.clone(),
                    BufferUsage::all(),
                )),
            }),
        }
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
        let mut ind_id = if let Some(ind) = rm.model_indirect.write().get_mut(&self.model_id) {
            ind.count += 1;
            ind.id.clone()
        } else {
            -1
        };

        if ind_id == -1 {
            ind_id = rm
                .shr_data
                .write()
                .indirect
                .emplace(DrawIndexedIndirectCommand {
                    index_count: sys
                        .model_manager
                        .lock()
                        .models_ids
                        .get(&self.model_id)
                        .unwrap()
                        .mesh
                        .indeces
                        .len() as u32,
                    instance_count: 0,
                    first_index: 0,
                    vertex_offset: 0,
                    first_instance: 0,
                });
            rm.model_indirect.write().insert(
                self.model_id,
                Indirect {
                    id: ind_id,
                    count: 1,
                },
            );
            rm.indirect_model.write().insert(ind_id, self.model_id);
        }
        self.id = rm.transforms.emplace(TransformId {
            indirect_id: ind_id,
            transform_id: self.t.0,
        });
        rm.updates.insert(
            self.id,
            TransformId {
                indirect_id: ind_id,
                transform_id: t.0,
            },
        );
    }
    fn deinit(&mut self, _t: Transform, sys: &mut Sys) {
        sys.renderer_manager
            .write()
            .model_indirect
            .write()
            .get_mut(&self.model_id)
            .unwrap()
            .count -= 1;
        sys.renderer_manager.write().updates.insert(
            self.id,
            TransformId {
                indirect_id: -1,
                transform_id: -1,
            },
        );
        sys.renderer_manager.write().transforms.erase(self.id);
    }
    fn update(&mut self, _sys: &crate::engine::System) {}
}
