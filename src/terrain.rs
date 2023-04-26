use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicI32, Ordering},
        Arc,
    },
};

use crossbeam::epoch::Atomic;
use egui::DragValue;
use noise::{NoiseFn, Perlin};

use nalgebra_glm as glm;
use parking_lot::Mutex;

use rapier3d::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
// use vulkano::buffer::{CpuAccessibleBuffer, DeviceLocalBuffer};
// use vulkano::descriptor_set::WriteDescriptorSet;
// use vulkano::device::Device;
use vulkano::{
    buffer::{BufferSlice, BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer, TypedBufferAccess},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, DrawIndexedIndirectCommand,
        PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, SecondaryAutoCommandBuffer,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
    impl_vertex,
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            rasterization::{CullMode, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    shader::ShaderModule,
    sync::{self, FlushError, GpuFuture},
};

use crate::{
    asset_manager::AssetManagerBase,
    engine::{RenderJobData, System},
    transform_compute,
};
use crate::{model, renderer::RenderPipeline, texture::Texture};
use crate::{renderer_component2::buffer_usage_all, terrain::transform::Transform};
// use crate::transform_compute::MVP;
use crate::transform_compute::cs::ty::MVP;
use crate::{
    engine::{transform, Component, Sys, World},
    inspectable::{Inpsect, Ins, Inspectable},
    model::{Mesh, ModelManager, Normal, Vertex, UV},
    renderer_component2::Renderer,
};

// struct Chunk {
//     verts: Vec<model::Vertex>,
//     normals: Vec<model::Normal>,
//     uvs: Vec<UV>,
//     indeces: Vec<u32>,
// }

struct TerrainChunkRenderData {
    pub vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pub normals_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
    pub uvs_buffer: Arc<CpuAccessibleBuffer<[UV]>>,
    pub index_buffer: Arc<CpuAccessibleBuffer<[u32]>>,
    pub texture: Option<i32>,
}
// #[component]
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Terrain {
    // pub device: Arc<Device>,
    // pub queue: Arc<Queue>,
    // pub texture_manager: Arc<TextureManager>,
    #[serde(skip_serializing, skip_deserializing)]
    pub chunks: Arc<Mutex<HashMap<i32, HashMap<i32, ColliderHandle>>>>,
    #[serde(skip_serializing, skip_deserializing)]
    cur_chunks: Arc<AtomicI32>,
    #[serde(skip_serializing, skip_deserializing)]
    prev_chunks: i32,
    #[serde(skip_serializing, skip_deserializing)]
    tcrd: Option<Arc<TerrainChunkRenderData>>,
    pub terrain_size: i32,
    pub chunk_range: i32,
}

impl Inspectable for Terrain {
    fn inspect(&mut self, transform: Transform, id: i32, ui: &mut egui::Ui, sys: &mut Sys) {
        // egui::CollapsingHeader::new("Terrain")
        //     .default_open(true)
        //     .show(ui, |ui| {
        Ins(&mut self.chunk_range).inspect("chunk_range", ui, sys);
        Ins(&mut self.terrain_size).inspect("terrain_size", ui, sys);
        // });
    }
}

impl Terrain {
    pub fn generate(&mut self, _transform: &Transform, sys: &System) {
        // let collider_set = &world.physics.collider_set;
        // let mm = &mut world.modeling.lock();
        let perlin = Perlin::new(0);

        let chunk_range = self.chunk_range;
        let terrain_size = self.terrain_size;
        {
            let mut chunks = self.chunks.lock();
            for x in -self.chunk_range..self.chunk_range {
                chunks.entry(x).or_default();
            }
        }
        if self.tcrd.is_none() {
            let num_chunks = (chunk_range * 2) * (chunk_range * 2);
            let num_verts_chunk = terrain_size * terrain_size;
            self.tcrd = Some(Arc::new(TerrainChunkRenderData {
                texture: Some(
                    sys.model_manager
                        .lock()
                        .const_params
                        .1
                        .lock()
                        .from_file("grass.png"),
                ),
                vertex_buffer: unsafe {
                    CpuAccessibleBuffer::uninitialized_array(
                        &sys.vk.mem_alloc,
                        (num_chunks * num_verts_chunk) as u64,
                        buffer_usage_all(),
                        false,
                    )
                    .unwrap()
                },
                normals_buffer: unsafe {
                    CpuAccessibleBuffer::uninitialized_array(
                        &sys.vk.mem_alloc,
                        (num_chunks * num_verts_chunk) as u64,
                        buffer_usage_all(),
                        false,
                    )
                    .unwrap()
                },
                uvs_buffer: unsafe {
                    CpuAccessibleBuffer::uninitialized_array(
                        &sys.vk.mem_alloc,
                        (num_chunks * num_verts_chunk) as u64,
                        buffer_usage_all(),
                        false,
                    )
                    .unwrap()
                },
                index_buffer: unsafe {
                    CpuAccessibleBuffer::uninitialized_array(
                        &sys.vk.mem_alloc,
                        (num_chunks * ((terrain_size - 1) * (terrain_size - 1)) * 6) as u64,
                        buffer_usage_all(),
                        false,
                    )
                    .unwrap()
                },
            }));
        }
        // let mut skip = 0;
        // let mut index_skip = 0;
        let command_buffers = Arc::new(Mutex::new(Vec::new()));
        // if self.cur_chunks.load(Ordering::Relaxed) == self.prev_chunks {
        //     return;
        // }

        rayon::scope(|s| {
            for x in -chunk_range..chunk_range {
                for z in -chunk_range..chunk_range {
                    if let Some(x) = self.chunks.lock().get(&x) {
                        if x.contains_key(&z) {
                            continue;
                        }
                    }
                    let x = x;
                    let z = z;
                    if let Some(tcrd) = &self.tcrd {
                        let tcrd = tcrd.clone();
                        let command_buffers = command_buffers.clone();
                        let chunks = self.chunks.clone();
                        let cur_chunks = self.cur_chunks.clone();

                        s.spawn(move |_| {
                            let m = Terrain::generate_chunk(
                                &perlin,
                                x,
                                z,
                                terrain_size,
                                sys.vk.device.clone(),
                            );

                            let ter_verts: Vec<Point<f32>> =
                                m.0.iter()
                                    .map(|v| point![v.position[0], v.position[1], v.position[2]])
                                    .collect();
                            let ter_indeces: Vec<[u32; 3]> = m
                                .3
                                .chunks(3)
                                .map(|slice| [slice[0] as u32, slice[1] as u32, slice[2] as u32])
                                .collect();

                            let collider = ColliderBuilder::trimesh(ter_verts, ter_indeces)
                                .collision_groups(InteractionGroups::none())
                                .solver_groups(InteractionGroups::none())
                                // .collision_groups(InteractionGroups::new(
                                //     0b10.into(),
                                //     (!0b10).into(),
                                // ))
                                .build();
                            // .solver_groups(InteractionGroups::new(0b0011.into(), 0b1011.into()));
                            chunks
                                .lock()
                                .entry(x)
                                .or_insert(HashMap::new())
                                .insert(z, ColliderHandle::invalid());
                            let x_ = x;
                            let z_ = z;

                            let mut builder = AutoCommandBufferBuilder::primary(
                                &sys.vk.comm_alloc,
                                sys.vk.queue.queue_family_index(),
                                CommandBufferUsage::OneTimeSubmit,
                            )
                            .unwrap();

                            let z_skip = terrain_size * terrain_size;
                            let x_skip = (chunk_range * 2) * z_skip;

                            let x = x + chunk_range;
                            let z = z + chunk_range;

                            let vertex_slice =
                                BufferSlice::from_typed_buffer_access(tcrd.vertex_buffer.clone())
                                    .slice(
                                        (x * x_skip + z * z_skip) as u64
                                            ..(x * x_skip + (z + 1) * z_skip) as u64,
                                    )
                                    .unwrap();
                            let vertecies = CpuAccessibleBuffer::from_iter(
                                &sys.vk.mem_alloc,
                                BufferUsage {
                                    transfer_src: true,
                                    ..Default::default()
                                },
                                false,
                                m.0.clone(),
                            )
                            .unwrap();
                            builder
                                .copy_buffer(CopyBufferInfo::buffers(vertecies, vertex_slice))
                                .unwrap();
                            let vertex_slice =
                                BufferSlice::from_typed_buffer_access(tcrd.normals_buffer.clone())
                                    .slice(
                                        (x * x_skip + z * z_skip) as u64
                                            ..(x * x_skip + (z + 1) * z_skip) as u64,
                                    )
                                    .unwrap();
                            let normals = CpuAccessibleBuffer::from_iter(
                                &sys.vk.mem_alloc,
                                BufferUsage {
                                    transfer_src: true,
                                    ..Default::default()
                                },
                                false,
                                m.1.clone(),
                            )
                            .unwrap();
                            builder
                                .copy_buffer(CopyBufferInfo::buffers(normals, vertex_slice))
                                .unwrap();

                            let vertex_slice =
                                BufferSlice::from_typed_buffer_access(tcrd.uvs_buffer.clone())
                                    .slice(
                                        (x * x_skip + z * z_skip) as u64
                                            ..(x * x_skip + (z + 1) * z_skip) as u64,
                                    )
                                    .unwrap();
                            let uvs = CpuAccessibleBuffer::from_iter(
                                &sys.vk.mem_alloc,
                                BufferUsage {
                                    transfer_src: true,
                                    ..Default::default()
                                },
                                false,
                                m.2.clone(),
                            )
                            .unwrap();
                            builder
                                .copy_buffer(CopyBufferInfo::buffers(uvs, vertex_slice))
                                .unwrap();

                            let start_index = (x * x_skip + z * z_skip) as u32;
                            let z_skip = (terrain_size - 1) * (terrain_size - 1) * 6;
                            let x_skip = (chunk_range * 2) * z_skip;
                            let index_slice =
                                BufferSlice::from_typed_buffer_access(tcrd.index_buffer.clone())
                                    .slice(
                                        (x * x_skip + z * z_skip) as u64
                                            ..(x * x_skip + (z + 1) * z_skip) as u64,
                                    )
                                    .unwrap();
                            let indexs = CpuAccessibleBuffer::from_iter(
                                &sys.vk.mem_alloc,
                                BufferUsage {
                                    transfer_src: true,
                                    ..Default::default()
                                },
                                false,
                                m.3.iter()
                                    .map(|i| i + start_index)
                                    .collect::<Vec<u32>>()
                                    .clone(),
                            )
                            .unwrap();
                            builder
                                .copy_buffer(CopyBufferInfo::buffers(indexs, index_slice))
                                .unwrap();
                            let command_buffer = builder.build().unwrap();
                            command_buffers.lock().push(Arc::new(command_buffer));

                            let _chunks = chunks.clone();
                            sys.defer.append(move |world| {
                                let handle = world.sys.lock().physics.collider_set.insert(collider);
                                _chunks.lock().get_mut(&x_).unwrap().insert(z_, handle);
                            });
                            cur_chunks.fetch_add(1, Ordering::Relaxed);
                        });
                    }
                }
            }
        });
        {
            // let command_buffers_g = command_buffers.lock();
            for command_buffer in (*command_buffers.lock()).clone() {
                let _ = command_buffer.execute(sys.vk.queue.clone()).unwrap();
            }
        }
    }

    fn generate_chunk(
        noise: &Perlin,
        _x: i32,
        _z: i32,
        terrain_size: i32,
        device: Arc<Device>,
    ) -> (Vec<model::Vertex>, Vec<model::Normal>, Vec<UV>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut uvs = Vec::new();

        let make_vert = |x: i32, z: i32| {
            let __x = x as f32 + (_x * (terrain_size - 1)) as f32;
            let __z = z as f32 + (_z * (terrain_size - 1)) as f32;
            Vertex {
                position: [
                    __x * 10.0,
                    noise.get([__x as f64 / 50., __z as f64 / 50.]) as f32 * 100.
                        + noise.get([__x as f64 / 10., __z as f64 / 10.]) as f32 * 7.,
                    __z * 10.0,
                ],
            }
        };

        for i in 0..terrain_size {
            for j in 0..terrain_size {
                // let _x = i as f32 - (terrain_size as f32 / 2.);
                // let _z = j as f32 - (terrain_size as f32 / 2.);
                // let _y = noise.get([i as f64 / 50.0, j as f64 / 50.0]) * 10.;

                // let _v = Vertex {
                //     position: [_x, _y as f32, _z],
                // };
                vertices.push(make_vert(i, j));
                let x = i as f32 + (_x * (terrain_size - 1)) as f32;
                let z = j as f32 + (_z * (terrain_size - 1)) as f32;
                uvs.push(UV { uv: [x, z] })
            }
        }

        let xz = |x, z| (x * terrain_size + z) as usize;

        let mut normals = Vec::with_capacity((terrain_size * terrain_size) as usize);
        // normals.resize(
        //     (terrain_size * terrain_size) as usize,
        //     Normal {
        //         normal: [0., 0., 0.],
        //     },
        // );
        unsafe {
            normals.set_len((terrain_size * terrain_size) as usize);
        }
        for x in 0..terrain_size as i32 {
            for z in 0..terrain_size as i32 {
                if x == 0 || x == terrain_size - 1 || z == 0 || z == terrain_size - 1 {
                    let p = vertices[xz(x, z) as usize];
                    let a1: glm::Vec3 = glm::cross(
                        &(p - make_vert(x, z - 1)).to_vec3(),
                        &(p - make_vert(x - 1, z - 1)).to_vec3(),
                    );
                    let a2 = glm::cross(
                        &(p - make_vert(x - 1, z - 1)).to_vec3(),
                        &(p - make_vert(x - 1, z)).to_vec3(),
                    );
                    let a3 = glm::cross(
                        &(p - make_vert(x + 1, z)).to_vec3(),
                        &(p - make_vert(x, z - 1)).to_vec3(),
                    );
                    let a4 = glm::cross(
                        &(p - make_vert(x - 1, z)).to_vec3(),
                        &(p - make_vert(x, z + 1)).to_vec3(),
                    );
                    let a5 = glm::cross(
                        &(p - make_vert(x + 1, z + 1)).to_vec3(),
                        &(p - make_vert(x + 1, z)).to_vec3(),
                    );
                    let a6 = glm::cross(
                        &(p - make_vert(x + 1, z + 1)).to_vec3(),
                        &(p - make_vert(x + 1, z + 1)).to_vec3(),
                    );
                    let n = glm::normalize(&(a1 + a2 + a3 + a4 + a5 + a6));
                    normals[xz(x, z) as usize] = Normal {
                        normal: [n.x, n.y, n.z],
                    };
                } else {
                    let p = vertices[xz(x, z)];
                    let a1 = glm::cross(
                        &(p - vertices[xz(x, z - 1)]).to_vec3(),
                        &(p - vertices[xz(x - 1, z - 1)]).to_vec3(),
                    );
                    let a2 = glm::cross(
                        &(p - vertices[xz(x - 1, z - 1)]).to_vec3(),
                        &(p - vertices[xz(x - 1, z)]).to_vec3(),
                    );
                    let a3 = glm::cross(
                        &(p - vertices[xz(x + 1, z)]).to_vec3(),
                        &(p - vertices[xz(x, z - 1)]).to_vec3(),
                    );
                    let a4 = glm::cross(
                        &(p - vertices[xz(x - 1, z)]).to_vec3(),
                        &(p - vertices[xz(x, z + 1)]).to_vec3(),
                    );
                    let a5 = glm::cross(
                        &(p - vertices[xz(x + 1, z + 1)]).to_vec3(),
                        &(p - vertices[xz(x + 1, z)]).to_vec3(),
                    );
                    let a6 = glm::cross(
                        &(p - vertices[xz(x + 1, z + 1)]).to_vec3(),
                        &(p - vertices[xz(x + 1, z + 1)]).to_vec3(),
                    );
                    let n = glm::normalize(&(a1 + a2 + a3 + a4 + a5 + a6));
                    normals[xz(x, z) as usize] = Normal {
                        normal: [n.x, n.y, n.z],
                    };
                }
            }
        }

        let mut indeces = Vec::new();
        for i in 0..(terrain_size - 1) {
            for j in 0..(terrain_size - 1) {
                indeces.push(xz(i, j) as u32);
                indeces.push(xz(i + 1, j + 1) as u32);
                indeces.push(xz(i, j + 1) as u32);
                indeces.push(xz(i, j) as u32);
                indeces.push(xz(i + 1, j) as u32);
                indeces.push(xz(i + 1, j + 1) as u32);
            }
        }

        // let mesh = Mesh::new_procedural(vertices, normals, indeces, uvs, device.clone());
        // mesh.texture = Some(mm.texture_manager.texture("grass.png"));
        // mesh
        (vertices, normals, uvs, indeces)
    }
}

impl Component for Terrain {
    fn on_render(&mut self, t_id: i32) -> Box<dyn Fn(&mut RenderJobData) -> ()> {
        let chunks = self.chunks.clone();
        // static mut COMMAND_BUFFER: Option<SecondaryAutoCommandBuffer> = None;
        let cur_chunks = self.cur_chunks.load(Ordering::Relaxed);
        let prev_chunks = self.prev_chunks;

        // Box::new(move |_rd: &mut RenderJobData| {})

        if let Some(tcrd) = &self.tcrd {
            let vertex_buffer = tcrd.vertex_buffer.clone();
            let normals_buffer = tcrd.normals_buffer.clone();
            let uvs_buffer = tcrd.uvs_buffer.clone();
            let index_buffer = tcrd.index_buffer.clone();
            let texture = tcrd.texture.clone();
            Box::new(move |rd: &mut RenderJobData| {
                let RenderJobData {
                    builder,
                    transforms,
                    mvp,
                    view,
                    proj,
                    pipeline,
                    viewport,
                    texture_manager,
                    vk,
                } = rd;
                let instance_data = vec![t_id];
                // let mvp_data = vec![MVP {
                //     mvp: (**proj * **view * glm::Mat4::identity()).into(),
                // }];

                static mut INSTANCE_BUFFER: Option<Arc<CpuAccessibleBuffer<[i32]>>> = None;
                if unsafe { INSTANCE_BUFFER.is_none() } {
                    unsafe {
                        INSTANCE_BUFFER = Some(
                            CpuAccessibleBuffer::<[i32]>::from_iter(
                                &vk.mem_alloc,
                                BufferUsage {
                                    transfer_src: true,
                                    storage_buffer: true,
                                    ..Default::default()
                                },
                                false,
                                instance_data,
                            )
                            .unwrap(),
                        );
                    }
                }

                let layout = pipeline.pipeline.layout().set_layouts().get(0).unwrap();

                let mut descriptors = Vec::new();

                // if let Some(mvp) = unsafe { &mvp_buffer } {
                descriptors.push(WriteDescriptorSet::buffer(0, mvp.clone()));
                // }

                if let Some(texture) = texture.as_ref() {
                    if let Some(texture) = texture_manager.lock().get_id(texture) {
                        let texture = texture.lock();
                        descriptors.push(WriteDescriptorSet::image_view_sampler(
                            1,
                            texture.image.clone(),
                            texture.sampler.clone(),
                        ));
                    }
                } else {
                    panic!("no terrain texture");
                }
                if let Some(i) = unsafe { &INSTANCE_BUFFER } {
                    descriptors.push(WriteDescriptorSet::buffer(2, i.clone()));
                }
                let set = PersistentDescriptorSet::new(
                    &vk.desc_alloc,
                    layout.clone(),
                    descriptors,
                )
                .unwrap();
                builder
                    // .set_viewport(0, [viewport.clone()])
                    // .bind_pipeline_graphics(pipeline.pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.pipeline.layout().clone(),
                        0,
                        set.clone(),
                    )
                    .bind_vertex_buffers(
                        0,
                        (
                            vertex_buffer.clone(),
                            normals_buffer.clone(),
                            uvs_buffer.clone(),
                            // instance_buffer.clone(),
                        ),
                    )
                    .bind_index_buffer(index_buffer.clone())
                    .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
                    .unwrap();
            })
        } else {
            Box::new(move |_rd: &mut RenderJobData| {})
        }
    }
    fn update(&mut self, transform: Transform, sys: &crate::engine::System) {
        self.prev_chunks = self.cur_chunks.load(Ordering::Relaxed);
        self.generate(&transform, sys);
    }
    fn editor_update(&mut self, transform: Transform, sys: &System) {
        self.prev_chunks = self.cur_chunks.load(Ordering::Relaxed);
        self.generate(&transform, sys);
    }
    fn deinit(&mut self, transform: Transform, id: i32, sys: &mut Sys) {
        let chunks = self.chunks.lock();
        for x in chunks.iter() {
            for (z, col) in x.1 {
                sys.physics.remove_collider(*col);
            }
        }
    }
}
