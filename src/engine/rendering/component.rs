use component_derive::ComponentID;
use parking_lot::{Mutex, RwLock};
use puffin_egui::puffin;
use std::{
    any::TypeId,
    array,
    collections::{BTreeMap, HashMap},
    sync::Arc,
};
use thincollections::thin_vec::ThinVec;

use crate::{
    editor::inspectable::{Inpsect, Ins, Inspectable},
    engine::{
        project::asset_manager::AssetInstance,
        storage::_Storage,
        transform_compute::TransformCompute,
        world::{
            component::{Component, _ComponentID},
            transform::Transform,
            Sys,
        },
    },
};
// use bytemuck::{Pod, Zeroable};
// use parking_lot::RwLock;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        BufferContents, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CopyBufferInfo,
        DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::DeviceOwned,
    memory::allocator::{MemoryAllocator, MemoryUsage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    shader::ShaderModule,
};

use super::{
    model::{ModelManager, ModelRenderer},
    vulkan_manager::VulkanManager,
};

#[derive(ComponentID, Default, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct Renderer {
    model_id: AssetInstance<ModelRenderer>,
    #[serde(skip_serializing, skip_deserializing)]
    id: Vec<i32>,
}

impl Component for Renderer {
    fn init(&mut self, transform: &Transform, _id: i32, sys: &Sys) {
        let mut rm = sys.renderer_manager.write();
        let mut model_indirect = rm.model_indirect.write();
        let mut ind_id = if let Some(ind) = model_indirect.get_mut(&self.model_id.id) {
            ind.iter_mut()
                .map(|ind| {
                    ind.count += 1;
                    ind.id
                })
                .collect::<Vec<i32>>()
        } else {
            let mut vec = Vec::new();

            let ind_id = unsafe {
                sys.get_model_manager()
                    .lock()
                    .as_any()
                    .downcast_ref_unchecked::<ModelManager>()
            }
            .assets_id
            .get(&self.model_id.id)
            .unwrap()
            .lock()
            .meshes
            .iter()
            .map(|mesh| {
                let id = rm
                    .shr_data
                    .write()
                    .indirect
                    .emplace(DrawIndexedIndirectCommand {
                        index_count: mesh.indeces.len() as u32,
                        instance_count: 0,
                        first_index: 0,
                        vertex_offset: 0,
                        first_instance: 0,
                    });
                vec.push(Indirect { id, count: 1 });
                rm.indirect_model.write().insert(id, self.model_id.id);
                id
            })
            .collect();
            model_indirect.insert(self.model_id.id, vec);
            ind_id
        };
        drop(model_indirect);

        self.id = ind_id
            .into_iter()
            .map(|id| {
                let _id = rm.transforms.emplace(ur::transform_id {
                    indirect_id: id,
                    id: transform.id,
                });
                rm.updates.insert(
                    _id,
                    ur::transform_id {
                        indirect_id: id,
                        id: transform.id,
                    },
                );
                _id
            })
            .collect();
    }
    fn deinit(&mut self, _transform: &Transform, _id: i32, sys: &Sys) {
        let mut rm = sys.renderer_manager.write();
        // reduce count in indirect
        if let Some(model_ind) = rm.model_indirect.write().get_mut(&self.model_id.id) {
            for ind in model_ind {
                ind.count -= 1;
            }
        }
        for id in &self.id {
            rm.updates.insert(
                *id,
                ur::transform_id {
                    indirect_id: -1,
                    id: -1,
                },
            );
            rm.transforms.erase(*id);
        }
    }
}

impl Inspectable for Renderer {
    fn inspect(&mut self, transform: &Transform, id: i32, ui: &mut egui::Ui, sys: &Sys) {
        let mut m_id = self.model_id;
        if Ins(&mut m_id).inspect("model_id", ui, sys) {
            self.deinit(transform, id, sys);
            self.model_id = m_id;
            self.init(transform, id, sys);
        }
    }
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
        path: "src/shaders/update_renderers2.comp",
        // types_meta: {
        //     use bytemuck::{Pod, Zeroable};

        //     #[derive(Clone, Copy, Zeroable, Pod)]
        // },
    }
}

#[derive(Clone, Copy)]
pub struct Indirect {
    pub id: i32,
    pub count: i32,
}

// #[repr(C)]
// #[derive(Clone, Copy, Debug, Default, BufferContents)]
// pub struct TransformId {
//     pub indirect_id: i32,
//     pub transform_id: i32,
// }

pub struct RendererData {
    pub model_indirect: BTreeMap<i32, Vec<Indirect>>,
    pub indirect_model: BTreeMap<i32, i32>,
    pub transforms_len: i32,

    pub updates: Vec<i32>,
}

pub struct SharedRendererData {
    pub transform_ids_gpu: Subbuffer<[ur::transform_id]>,
    pub renderers_gpu: Subbuffer<[i32]>,
    pub updates_gpu: Subbuffer<[i32]>,
    pub indirect: _Storage<DrawIndexedIndirectCommand>,
    pub indirect_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,

    pub vk: Arc<VulkanManager>,
    pub shader: Arc<ShaderModule>,
    pub pipeline: Arc<ComputePipeline>,
    pub uniform: Mutex<SubbufferAllocator>,
}

impl SharedRendererData {
    pub fn update(
        &mut self,
        // rm: &mut parking_lot::RwLockWriteGuard<SharedRendererData>,
        rd: &mut RendererData,
        vk: Arc<VulkanManager>,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        renderer_pipeline: Arc<ComputePipeline>,
        transform_compute: &TransformCompute,
    ) -> Vec<i32> {
        // let rm = self;
        if self.transform_ids_gpu.len() < rd.transforms_len as u64 {
            let len = rd.transforms_len;
            let max_len = (len as f32 + 1.).log2().ceil();
            let max_len = 2_u32.pow(max_len as u32);

            let copy_buffer = self.transform_ids_gpu.clone();
            unsafe {
                self.transform_ids_gpu = vk.buffer_array(max_len as u64, MemoryUsage::DeviceOnly);
                self.renderers_gpu = vk.buffer_array(max_len as u64, MemoryUsage::DeviceOnly);
            }

            // let copy = CopyBufferInfo::buffers(copy_buffer, rm.transform_ids_gpu.clone());
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    copy_buffer,
                    self.transform_ids_gpu.clone(),
                ))
                .unwrap();
        }
        if !self.indirect.data.is_empty() {
            self.indirect_buffer = vk.buffer_from_iter(self.indirect.data.clone());
        }

        let mut offset_vec = Vec::new();
        let mut offset = 0;
        for (ind_id, m_id) in rd.indirect_model.iter() {
            if let Some(model_ind) = rd.model_indirect.get(m_id) {
                for ind in model_ind.iter() {
                    if ind.id == *ind_id {
                        offset_vec.push(offset);
                        offset += ind.count;
                        break;
                    }
                }
            }
        }
        if !offset_vec.is_empty() {
            let offsets_buffer = vk.buffer_from_iter(offset_vec.clone());

            {
                puffin::profile_scope!("update renderers: stage 0");
                let update_num = rd.updates.len() / 3;
                let mut rd_updates = Vec::new();
                std::mem::swap(&mut rd_updates, &mut rd.updates);
                if update_num > 0 {
                    self.updates_gpu = vk.buffer_from_iter(rd_updates);
                }
                // stage 0
                let uniforms = self.uniform.lock().allocate_sized().unwrap();
                let data = ur::Data {
                    num_jobs: update_num as i32,
                    stage: 0.into(),
                    view: Default::default(),
                    // _dummy0: Default::default(),
                };
                *uniforms.write().unwrap() = data;

                let update_renderers_set = PersistentDescriptorSet::new(
                    &vk.desc_alloc,
                    renderer_pipeline
                        .layout()
                        .set_layouts()
                        .get(0) // 0 is the index of the descriptor set.
                        .unwrap()
                        .clone(),
                    [
                        // 0 is the binding of the data in this set. We bind the `DeviceLocalBuffer` of vertices here.
                        WriteDescriptorSet::buffer(0, self.updates_gpu.clone()),
                        WriteDescriptorSet::buffer(1, self.transform_ids_gpu.clone()),
                        WriteDescriptorSet::buffer(2, self.renderers_gpu.clone()),
                        WriteDescriptorSet::buffer(3, self.indirect_buffer.clone()),
                        WriteDescriptorSet::buffer(4, transform_compute.gpu_transforms.clone()),
                        WriteDescriptorSet::buffer(5, offsets_buffer),
                        WriteDescriptorSet::buffer(6, uniforms),
                    ],
                )
                .unwrap();

                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        renderer_pipeline.layout().clone(),
                        0, // Bind this descriptor set to index 0.
                        update_renderers_set,
                    )
                    .dispatch([update_num as u32 / 128 + 1, 1, 1])
                    .unwrap();
            }
            offset_vec
        } else {
            Vec::new()
        }
    }
}

pub struct RendererManager {
    pub model_indirect: RwLock<BTreeMap<i32, Vec<Indirect>>>,
    pub indirect_model: RwLock<BTreeMap<i32, i32>>,

    pub transforms: _Storage<ur::transform_id>,
    pub updates: HashMap<i32, ur::transform_id>,
    pub shr_data: Arc<RwLock<SharedRendererData>>,
}

pub fn buffer_usage_all() -> BufferUsage {
    BufferUsage::TRANSFER_SRC
        | BufferUsage::TRANSFER_DST
        | BufferUsage::UNIFORM_TEXEL_BUFFER
        | BufferUsage::STORAGE_TEXEL_BUFFER
        | BufferUsage::UNIFORM_BUFFER
        | BufferUsage::STORAGE_BUFFER
        | BufferUsage::INDEX_BUFFER
        | BufferUsage::VERTEX_BUFFER
        | BufferUsage::INDIRECT_BUFFER
        | BufferUsage::SHADER_DEVICE_ADDRESS
    // BufferUsage {
    //     transfer_src: true,
    //     transfer_dst: true,
    //     uniform_texel_buffer: true,
    //     storage_texel_buffer: true,
    //     uniform_buffer: true,
    //     storage_buffer: true,
    //     index_buffer: true,
    //     vertex_buffer: true,
    //     indirect_buffer: true,
    //     shader_device_address: true,
    //     ..Default::default()
    // }
}

impl RendererManager {
    pub fn new(vk: Arc<VulkanManager>) -> RendererManager {
        let shader = ur::load(vk.device.clone()).unwrap();

        // Create compute-pipeline for applying compute shader to vertices.
        let pipeline = vulkano::pipeline::ComputePipeline::new(
            vk.device.clone(),
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
            transforms: _Storage::new(),
            shr_data: Arc::new(RwLock::new(SharedRendererData {
                transform_ids_gpu: vk.buffer_from_iter(vec![ur::transform_id {
                    indirect_id: -1,
                    id: -1,
                }]),
                renderers_gpu: vk.buffer_from_iter(vec![0]),
                updates_gpu: vk.buffer_from_iter(vec![0]),
                indirect: _Storage::new(),
                indirect_buffer: vk.buffer_from_iter(vec![DrawIndexedIndirectCommand {
                    index_count: 0,
                    instance_count: 0,
                    first_index: 0,
                    vertex_offset: 0,
                    first_instance: 0,
                }]),
                shader,
                pipeline,
                vk: vk.clone(),
                uniform: Mutex::new(vk.sub_buffer_allocator()),
            })),
        }
    }
    pub(crate) fn get_renderer_data(&mut self) -> RendererData {
        let renderer_data = RendererData {
            model_indirect: self
                .model_indirect
                .read()
                .iter()
                .map(|(k, v)| (*k, v.iter().copied().collect()))
                .collect(),
            indirect_model: self
                .indirect_model
                .read()
                .iter()
                .map(|(k, v)| (*k, *v))
                .collect(),
            updates: self
                .updates
                .iter()
                .flat_map(|(id, t)| vec![*id, t.indirect_id, t.id].into_iter())
                .collect(),
            transforms_len: self.transforms.data.len() as i32,
        };
        self.updates.clear();
        renderer_data
    }
    pub(crate) fn clear(&mut self) {
        self.transforms.clear();
        let mut m = self.model_indirect.write();
        for (_, m) in m.iter_mut() {
            for i in m.iter_mut() {
                i.count = 0;
            }
        }
        // self.model_indirect.write().clear();
        // self.indirect_model.write().clear();
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
    pub fn new(model_id: i32) -> Renderer {
        Renderer {
            model_id: AssetInstance::<ModelRenderer>::new(model_id),
            id: [0].into_iter().collect(),
        }
    }
}
