use id::*;
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
    editor::inspectable::{Inpsect, Ins},
    engine::{
        project::asset_manager::AssetInstance,
        storage::_Storage,
        transform_compute::TransformCompute,
        utils,
        world::{component::Component, transform::Transform, Sys},
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
    descriptor_set::{DescriptorSet, PersistentDescriptorSet, WriteDescriptorSet},
    device::DeviceOwned,
    memory::allocator::{MemoryAllocator, MemoryTypeFilter},
    pipeline::{
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
        ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout,
        PipelineShaderStageCreateInfo,
    },
    shader::ShaderModule,
};

use super::{
    model::{ModelManager, ModelRenderer, Skeleton},
    vulkan_manager::VulkanManager,
};

#[derive(ID, Default, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct Renderer {
    model_id: AssetInstance<ModelRenderer>,
    #[serde(skip_serializing, skip_deserializing)]
    transformIds: Vec<i32>,
    #[serde(skip_serializing, skip_deserializing)]
    pub skeleton: Option<i32>,
}
impl Renderer {
    pub fn get_model(&self) -> AssetInstance<ModelRenderer> {
        self.model_id.clone()
    }
}
impl Component for Renderer {
    fn init(&mut self, transform: &Transform, _id: i32, sys: &Sys) {
        let ind_id = {
            let mut rm = sys.renderer_manager.write();
            let mut model_indirect = rm.model_indirect.write();
            let indirect_counts = &mut rm.shr_data.write().indirect_counts;

            if let Some(ind) = model_indirect.get_mut(&self.model_id.id) {
                ind.iter()
                    .map(|ind_id| {
                        indirect_counts[*ind_id as usize] += 1;
                        *ind_id
                    })
                    .collect::<Vec<i32>>()
            } else {
                let ind_id = sys.assets_manager.get_manager(|m: &ModelManager| {
                    m.assets_id
                        .get(&self.model_id.id)
                        .and_then(|l| {
                            let l = l.lock();
                            let ids: Vec<_> = l
                                .model
                                .meshes
                                .iter()
                                .map(|mesh| {
                                    let id = mesh.indirect_index as i32;
                                    indirect_counts[id as usize] += 1;
                                    id
                                })
                                .collect();
                            Some(ids)
                        })
                        .unwrap()
                });
                model_indirect.insert(self.model_id.id, ind_id.clone());
                ind_id
            }
        };
        let mut rm = sys.renderer_manager.write();

        let skeleton = sys.assets_manager.get_manager(|m: &ModelManager| {
            m.assets_id
                .get(&self.model_id.id)
                .unwrap()
                .lock()
                .model
                .has_skeleton
        });

        if skeleton {
            self.skeleton = Some({
                let mut skel_man = sys.skeletons_manager.write();
                if let Some(skel_stor) = skel_man.get_mut(&self.model_id.id) {
                    skel_stor.emplace(Mutex::new(Skeleton::new(rand::random::<usize>() % 25)))
                } else {
                    skel_man.insert(self.model_id.id, _Storage::new());
                    skel_man
                        .get_mut(&self.model_id.id)
                        .unwrap()
                        .emplace(Mutex::new(Skeleton::new(rand::random::<usize>() % 25)))
                }
                // sys.skeletons_manager
                // .write()
                // .entry(self.model_id.id)
                // .or_insert(_Storage::new())
                // .emplace(Skeleton::new(self.model_id, rand::random::<usize>() % 26)),
            });
        }

        self.transformIds = ind_id // reference to transform ids
            .into_iter()
            .map(|id| {
                let _id = rm.transforms.emplace(ur::transform_id {
                    indirect_id: id,
                    id: transform.id,
                    skeleton_id: self.skeleton.unwrap_or(-1),
                    padding: 0,
                });
                rm.updates.insert(
                    _id,
                    ur::transform_id {
                        indirect_id: id,
                        id: transform.id,
                        skeleton_id: self.skeleton.unwrap_or(-1),
                        padding: 0,
                    },
                );
                _id
            })
            .collect();
    }
    fn deinit(&mut self, _transform: &Transform, _id: i32, sys: &Sys) {
        let mut rm = sys.renderer_manager.write();
        {
            let indirect_counts = &mut rm.shr_data.write().indirect_counts;
            // reduce count in indirect
            if let Some(model_ind) = rm.model_indirect.write().get_mut(&self.model_id.id) {
                for ind in model_ind {
                    indirect_counts[*ind as usize] -= 1;
                }
            }
        }
        for tid in &self.transformIds {
            rm.updates.insert(
                *tid,
                ur::transform_id {
                    indirect_id: -1,
                    id: -1,
                    skeleton_id: -1,
                    padding: 0,
                },
            );
            rm.transforms.erase(*tid);
        }
        if let Some(skel) = self.skeleton {
            sys.skeletons_manager
                .write()
                .get_mut(&self.model_id.id)
                .unwrap()
                .erase(skel);
        }
    }
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
        path: "shaders/update_renderers2.comp",
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
    // pub index: usize,
}

// #[repr(C)]
// #[derive(Clone, Copy, Debug, Default, BufferContents)]
// pub struct TransformId {
//     pub indirect_id: i32,
//     pub transform_id: i32,
// }

pub struct RendererData {
    // pub model_indirect: BTreeMap<i32, Vec<i32>>,
    // pub indirect_model: BTreeMap<i32, i32>,
    pub transforms_len: u32,

    pub updates: Vec<i32>,
}

pub struct SharedRendererData {
    pub transform_ids_gpu: Subbuffer<[ur::transform_id]>,
    pub renderers_gpu: Subbuffer<[[i32; 2]]>,
    pub updates_gpu: Subbuffer<[i32]>,
    pub indirect: Vec<DrawIndexedIndirectCommand>,
    pub texture_ids: Vec<i32>,
    pub indirect_counts: Vec<i32>,
    pub indirect_buffer: Subbuffer<[DrawIndexedIndirectCommand]>,
    pub temp_sums: Subbuffer<[i32]>,
    pub vk: Arc<VulkanManager>,
    pub shader: Arc<ShaderModule>,
    pub pipeline: Arc<ComputePipeline>,
    // pub uniform: Mutex<SubbufferAllocator>,
}

impl SharedRendererData {
    pub fn update(
        &mut self,
        // rm: &mut parking_lot::RwLockWriteGuard<SharedRendererData>,
        rd: &mut RendererData,
        vk: Arc<VulkanManager>,
        builder: &mut utils::PrimaryCommandBuffer,
        renderer_pipeline: Arc<ComputePipeline>,
        transform_compute: &TransformCompute,
    ) -> Vec<i32> {
        // let rm = self;
        if self.transform_ids_gpu.len() < rd.transforms_len as u64 {
            let len = rd.transforms_len;
            let max_len = (rd.transforms_len as usize).next_power_of_two();

            let copy_buffer = self.transform_ids_gpu.clone();
            unsafe {
                self.transform_ids_gpu =
                    vk.buffer_array(max_len as u64, MemoryTypeFilter::PREFER_DEVICE);
                self.renderers_gpu =
                    vk.buffer_array(max_len as u64, MemoryTypeFilter::PREFER_DEVICE);
            }

            // let copy = CopyBufferInfo::buffers(copy_buffer, rm.transform_ids_gpu.clone());
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    copy_buffer,
                    self.transform_ids_gpu.clone(),
                ))
                .unwrap();
        }

        let mut offset_vec = Vec::new();
        let mut offset = 0;

        // for count in self.indirect_counts.iter() {
        //     offset_vec.push(offset);
        //     offset += *count;
        // }
        // if !offset_vec.is_empty() {
        // let offsets_buffer = vk.buffer_from_iter(offset_vec.clone()); // don't need here

        {
            puffin::profile_scope!("update renderers: stage 0");
            let update_num = rd.updates.len() / 4;
            let mut rd_updates = Vec::new();
            std::mem::swap(&mut rd_updates, &mut rd.updates);
            if update_num > 0 {
                self.updates_gpu = vk.buffer_from_iter(rd_updates);
            }

            let update_renderers_set = PersistentDescriptorSet::new(
                &vk.desc_alloc,
                renderer_pipeline
                    .layout()
                    .set_layouts()
                    .get(0)
                    .unwrap()
                    .clone(),
                [
                    WriteDescriptorSet::buffer(0, self.updates_gpu.clone()),
                    WriteDescriptorSet::buffer(1, self.transform_ids_gpu.clone()),
                    WriteDescriptorSet::buffer(2, self.renderers_gpu.clone()),
                    WriteDescriptorSet::buffer(3, self.indirect_buffer.clone()),
                    WriteDescriptorSet::buffer(4, transform_compute.gpu_transforms.clone()),
                    WriteDescriptorSet::buffer(5, self.indirect_buffer.clone()),
                ],
                [],
            )
            .unwrap();

            builder
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    renderer_pipeline.layout().clone(),
                    0,
                    update_renderers_set,
                )
                .unwrap()
                .push_constants(
                    renderer_pipeline.layout().clone(),
                    0,
                    ur::Data {
                        num_jobs: update_num as i32,
                        stage: 0.into(),
                        view: Default::default(),
                        pass: 0.into(),
                    },
                )
                .unwrap()
                .dispatch([update_num as u32 / 128 + 1, 1, 1])
                .unwrap();
        }
        offset_vec
        // } else {
        //     Vec::new()
        // }
    }
}

pub struct RendererManager {
    pub model_indirect: RwLock<BTreeMap<i32, Vec<i32>>>,
    // pub indirect_model: RwLock<BTreeMap<i32, i32>>,
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
}

impl RendererManager {
    pub fn new(vk: Arc<VulkanManager>) -> RendererManager {
        let shader = ur::load(vk.device.clone()).unwrap();
        let pipeline = utils::pipeline::compute_pipeline(vk.clone(), shader.clone());

        RendererManager {
            model_indirect: RwLock::new(BTreeMap::new()),
            // indirect_model: RwLock::new(BTreeMap::new()),
            updates: HashMap::new(),
            transforms: _Storage::new(),
            shr_data: Arc::new(RwLock::new(SharedRendererData {
                transform_ids_gpu: vk.buffer_from_iter(vec![ur::transform_id {
                    indirect_id: -1,
                    id: -1,
                    skeleton_id: -1,
                    padding: 0,
                }]),
                renderers_gpu: vk.buffer_from_iter(vec![[0, 0]]),
                updates_gpu: vk.buffer_from_iter(vec![0]),
                indirect: Vec::new(),
                indirect_counts: Vec::new(),
                indirect_buffer: vk.buffer_from_iter(vec![DrawIndexedIndirectCommand {
                    index_count: 0,
                    instance_count: 0,
                    first_index: 0,
                    vertex_offset: 0,
                    first_instance: 0,
                }]),
                temp_sums: vk.buffer_from_iter(vec![0]),
                // offsets_buffer: vk.buffer_from_iter(vec![0]),
                shader,
                pipeline,
                vk: vk.clone(),
                texture_ids: Vec::new(),
                // uniform: Mutex::new(vk.sub_buffer_allocator()),
            })),
        }
    }
    pub(crate) fn get_renderer_data(&mut self) -> RendererData {
        let renderer_data = RendererData {
            // model_indirect: self
            //     .model_indirect
            //     .read()
            //     .iter()
            //     .map(|(k, v)| (*k, v.clone()))
            //     .collect(),
            // indirect_model: self
            //     .indirect_model
            //     .read()
            //     .iter()
            //     .map(|(k, v)| (*k, *v))
            //     .collect(),
            updates: self
                .updates
                .iter()
                .flat_map(|(id, t)| vec![*id, t.indirect_id, t.id, t.skeleton_id].into_iter())
                .collect(),
            transforms_len: self.transforms.data.len() as u32,
        };
        self.updates.clear();
        renderer_data
    }
    pub(crate) fn clear(&mut self) {
        self.transforms.clear();
        // let mut m = self.model_indirect.write();
        // for (_, m) in m.iter_mut() {
        //     for ind in m.iter_mut() {
        //         *ind = 0;
        //     }
        // }
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
            transformIds: [0].into(),
            skeleton: None,
        }
    }
}
