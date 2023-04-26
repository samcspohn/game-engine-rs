use parking_lot::RwLock;
use puffin_egui::puffin;
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use crate::{
    drag_drop::{self, drop_target},
    engine::{transform::Transform, Component, Storage, Sys, World, _Storage},
    inspectable::{Inpsect, Ins, Inspectable},
    vulkan_manager::VulkanManager, transform_compute::TransformCompute,
};
use bytemuck::{Pod, Zeroable};
// use parking_lot::RwLock;
use rayon::prelude::{IndexedParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CopyBufferInfo,
        DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Device,
    memory::allocator::{FreeListAllocator, GenericMemoryAllocator, MemoryUsage},
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    shader::ShaderModule,
};

#[derive(Default, Clone, Copy, Serialize, Deserialize)]
struct ModelId {
    id: i32,
}
impl<'a> Inpsect for Ins<'a, ModelId> {
    fn inspect(&mut self, name: &str, ui: &mut egui::Ui, sys: &mut Sys) {
        let drop_data = drag_drop::DRAG_DROP_DATA.lock();

        let model: String = match sys.model_manager.lock().assets_id.get(&self.0.id) {
            Some(model) => model.lock().file.clone(),
            None => "".into(),
        };
        let can_accept_drop_data = match drop_data.rfind(".obj") {
            Some(_) => true,
            None => false,
        };
        // println!("can accept drop data:{}",can_accept_drop_data);
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(name));
            drop_target(ui, can_accept_drop_data, |ui| {
                // let model_name = sys.model_manager.lock().models.get(k)
                let response = ui.add(egui::Label::new(model.as_str()));
                if response.hovered() && ui.input().pointer.any_released() {
                    let model_file: String = drop_data.clone();

                    if let Some(id) = sys.model_manager.lock().assets.get(&model_file) {
                        self.0.id = *id;
                    }
                }
            });
        });
    }
}

// #[component]
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Renderer {
    model_id: ModelId,
    // #[serde(skip_serializing, skip_deserializing)]
    // TODO fix incrementing id when toggling play/stop
    id: i32,
}

impl Inspectable for Renderer {
    fn inspect(&mut self, transform: Transform, id: i32, ui: &mut egui::Ui, sys: &mut Sys) {
        // ui.add(egui::Label::new("Renderer"));
        // egui::CollapsingHeader::new(std::any::type_name::<Self>())
        //     .default_open(true)
        //     .show(ui, |ui| {
        let m_id = self.model_id;
        Ins(&mut self.model_id).inspect("model_id", ui, sys);

        if self.model_id.id != m_id.id {
            // self.deinit(transform, id, sys);
            let rm = &mut sys.renderer_manager.write();
            rm.model_indirect.write().get_mut(&m_id.id).unwrap().count -= 1;
            // sys.renderer_manager.write().updates.insert(
            //     self.id,
            //     TransformId {
            //         indirect_id: -1,
            //         transform_id: -1,
            //     },
            // );
            // sys.renderer_manager.write().transforms.erase(self.id);

            let mut ind_id = if let Some(ind) = rm.model_indirect.write().get_mut(&self.model_id.id)
            {
                ind.count += 1;
                ind.id
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
                            .assets_id
                            .get(&self.model_id.id)
                            .unwrap()
                            .lock()
                            .mesh
                            .indeces
                            .len() as u32,
                        instance_count: 0,
                        first_index: 0,
                        vertex_offset: 0,
                        first_instance: 0,
                    });
                rm.model_indirect.write().insert(
                    self.model_id.id,
                    Indirect {
                        id: ind_id,
                        count: 1,
                    },
                );
                rm.indirect_model.write().insert(ind_id, self.model_id.id);
            }
            rm.transforms.data[self.id as usize] = TransformId {
                indirect_id: ind_id,
                transform_id: transform.id,
            };
            rm.updates.insert(
                self.id,
                TransformId {
                    indirect_id: ind_id,
                    transform_id: transform.id,
                },
            );
        }
        // });
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
    pub indirect: _Storage<DrawIndexedIndirectCommand>,
    pub indirect_buffer: Arc<CpuAccessibleBuffer<[DrawIndexedIndirectCommand]>>,

    pub device: Arc<Device>,
    pub mem: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
    pub shader: Arc<ShaderModule>,
    pub pipeline: Arc<ComputePipeline>,
    pub uniform: Arc<CpuBufferPool<ur::ty::Data>>,
}

impl SharedRendererData {
    pub fn update(
        &mut self,
        // rm: &mut parking_lot::RwLockWriteGuard<SharedRendererData>,
        rd: &mut RendererData,
        vk: Arc<VulkanManager>,
        builder: &mut AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, Arc<StandardCommandBufferAllocator>>,
        renderer_pipeline: Arc<ComputePipeline>,
        transform_compute: &TransformCompute,
    ) -> Vec<i32> {
        let rm = self;
        if rm.transform_ids_gpu.len() < rd.transforms_len as u64 {
            let len = rd.transforms_len;
            let max_len = (len as f32 + 1.).log2().ceil();
            let max_len = (2 as u32).pow(max_len as u32);

            let copy_buffer = rm.transform_ids_gpu.clone();
            unsafe {
                rm.transform_ids_gpu = CpuAccessibleBuffer::uninitialized_array(
                    &vk.mem_alloc,
                    max_len as u64,
                    buffer_usage_all(),
                    false,
                )
                .unwrap();
                rm.renderers_gpu = CpuAccessibleBuffer::uninitialized_array(
                    &vk.mem_alloc,
                    max_len as u64,
                    buffer_usage_all(),
                    false,
                )
                .unwrap();
            }

            // let copy = CopyBufferInfo::buffers(copy_buffer, rm.transform_ids_gpu.clone());
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    copy_buffer,
                    rm.transform_ids_gpu.clone(),
                ))
                .unwrap();
        }
        if rm.indirect.data.len() > 0 {
            rm.indirect_buffer = CpuAccessibleBuffer::from_iter(
                &vk.mem_alloc,
                buffer_usage_all(),
                false,
                rm.indirect.data.clone(),
            )
            .unwrap();
        }

        let mut offset_vec = Vec::new();
        let mut offset = 0;
        for (_, m_id) in rd.indirect_model.iter() {
            offset_vec.push(offset);
            if let Some(ind) = rd.model_indirect.get(&m_id) {
                offset += ind.count;
            }
        }
        if offset_vec.len() > 0 {
            let offsets_buffer = CpuAccessibleBuffer::from_iter(
                &vk.mem_alloc,
                buffer_usage_all(),
                false,
                offset_vec.clone(),
            )
            .unwrap();

            {
                puffin::profile_scope!("update renderers: stage 0");
                let update_num = rd.updates.len() / 3;
                let mut rd_updates = Vec::new();
                std::mem::swap(&mut rd_updates, &mut rd.updates);
                if update_num > 0 {
                    rm.updates_gpu = CpuAccessibleBuffer::from_iter(
                        &vk.mem_alloc,
                        buffer_usage_all(),
                        false,
                        rd_updates,
                    )
                    .unwrap();
                }

                // stage 0
                let uniforms = rm
                    .uniform
                    .from_data(ur::ty::Data {
                        num_jobs: update_num as i32,
                        stage: 0,
                        view: Default::default(),
                        _dummy0: Default::default(),
                    })
                    .unwrap();

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
                        WriteDescriptorSet::buffer(0, rm.updates_gpu.clone()),
                        WriteDescriptorSet::buffer(1, rm.transform_ids_gpu.clone()),
                        WriteDescriptorSet::buffer(2, rm.renderers_gpu.clone()),
                        WriteDescriptorSet::buffer(3, rm.indirect_buffer.clone()),
                        WriteDescriptorSet::buffer(4, transform_compute.transform.clone()),
                        WriteDescriptorSet::buffer(5, offsets_buffer.clone()),
                        WriteDescriptorSet::buffer(6, uniforms.clone()),
                    ],
                )
                .unwrap();

                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        renderer_pipeline.layout().clone(),
                        0, // Bind this descriptor set to index 0.
                        update_renderers_set.clone(),
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
    pub model_indirect: RwLock<BTreeMap<i32, Indirect>>,
    pub indirect_model: RwLock<BTreeMap<i32, i32>>,

    pub transforms: _Storage<TransformId>,
    pub updates: HashMap<i32, TransformId>,
    pub shr_data: Arc<RwLock<SharedRendererData>>,
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

pub fn buffer_usage_all() -> BufferUsage {
    BufferUsage {
        transfer_src: true,
        transfer_dst: true,
        uniform_texel_buffer: true,
        storage_texel_buffer: true,
        uniform_buffer: true,
        storage_buffer: true,
        index_buffer: true,
        vertex_buffer: true,
        indirect_buffer: true,
        shader_device_address: true,
        ..Default::default()
    }
}

impl RendererManager {
    pub fn new(
        device: Arc<Device>,
        mem: Arc<GenericMemoryAllocator<Arc<FreeListAllocator>>>,
    ) -> RendererManager {
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
            transforms: _Storage::new(),
            shr_data: Arc::new(RwLock::new(SharedRendererData {
                transform_ids_gpu: CpuAccessibleBuffer::from_iter(
                    // device.clone(),
                    &mem,
                    buffer_usage_all(),
                    true,
                    vec![TransformId {
                        indirect_id: -1,
                        transform_id: -1,
                    }],
                )
                .unwrap(),
                renderers_gpu: CpuAccessibleBuffer::from_iter(
                    // device.clone(),
                    &mem,
                    buffer_usage_all(),
                    true,
                    vec![0],
                )
                .unwrap(),
                updates_gpu: CpuAccessibleBuffer::from_iter(
                    // device.clone(),
                    &mem,
                    buffer_usage_all(),
                    true,
                    vec![0],
                )
                .unwrap(),
                indirect: _Storage::new(),
                indirect_buffer: CpuAccessibleBuffer::from_iter(
                    &mem,
                    buffer_usage_all(),
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
                mem: mem.clone(),
                uniform: Arc::new(CpuBufferPool::<ur::ty::Data>::new(
                    mem.clone(),
                    // device.clone(),
                    buffer_usage_all(),
                    MemoryUsage::Upload,
                )),
            })),
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
    pub fn new(model_id: i32) -> Renderer {
        Renderer {
            model_id: ModelId { id: model_id },
            id: 0,
        }
    }
}

impl Component for Renderer {
    fn init(&mut self, transform: Transform, _id: i32, sys: &mut Sys) {
        let rm = &mut sys.renderer_manager.write();
        let mut ind_id = if let Some(ind) = rm.model_indirect.write().get_mut(&self.model_id.id) {
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
                        .assets_id
                        .get(&self.model_id.id)
                        .unwrap()
                        .lock()
                        .mesh
                        .indeces
                        .len() as u32,
                    instance_count: 0,
                    first_index: 0,
                    vertex_offset: 0,
                    first_instance: 0,
                });
            rm.model_indirect.write().insert(
                self.model_id.id,
                Indirect {
                    id: ind_id,
                    count: 1,
                },
            );
            rm.indirect_model.write().insert(ind_id, self.model_id.id);
        }
        self.id = rm.transforms.emplace(TransformId {
            indirect_id: ind_id,
            transform_id: transform.id,
        });
        rm.updates.insert(
            self.id,
            TransformId {
                indirect_id: ind_id,
                transform_id: transform.id,
            },
        );
    }
    fn deinit(&mut self, _transform: Transform, _id: i32, sys: &mut Sys) {
        sys.renderer_manager
            .write()
            .model_indirect
            .write()
            .get_mut(&self.model_id.id)
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
}
