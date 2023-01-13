use std::{collections::HashMap, sync::Arc};

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
    buffer::{BufferSlice, BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer},
    command_buffer::{
        AutoCommandBufferBuilder, DrawIndexedIndirectCommand, PrimaryAutoCommandBuffer,
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
};

use crate::engine::{RenderJobData, System};
use crate::renderer::RenderPipeline;
use crate::terrain::transform::Transform;
// use crate::transform_compute::MVP;
use crate::transform_compute::cs::ty::MVP;
use crate::{
    engine::{transform, Component, Sys, World},
    inspectable::{Inpsect, Ins, Inspectable},
    model::{Mesh, ModelManager, Normal, Vertex, UV},
    renderer_component2::Renderer,
};

struct CustRendData {
    instance_buffer: Arc<CpuAccessibleBuffer<[i32]>>,
    mvp_buffer: Arc<CpuAccessibleBuffer<[MVP]>>,
}

// #[component]
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Terrain {
    // pub device: Arc<Device>,
    // pub queue: Arc<Queue>,
    // pub texture_manager: Arc<TextureManager>,
    #[serde(skip_serializing, skip_deserializing)]
    pub chunks: Arc<Mutex<HashMap<i32, HashMap<i32, Mesh>>>>,
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
    pub fn generate(&mut self, transform: &Transform, sys: &System) {
        // let collider_set = &world.physics.collider_set;
        // let mm = &mut world.modeling.lock();
        let perlin = Perlin::new();

        let chunk_range = self.chunk_range;
        let terrain_size = self.terrain_size;
        {
            let mut chunks = self.chunks.lock();
            for x in -self.chunk_range..self.chunk_range {
                chunks.entry(x).or_default();
            }
        }
        (-chunk_range..chunk_range).into_par_iter().for_each(|x| {
            (-chunk_range..chunk_range).into_par_iter().for_each(|z| {
                {
                    if let Some(x) = self.chunks.lock().get(&x) {
                        if x.contains_key(&z) {
                            return;
                        }
                    }
                }
                let mut m =
                    Terrain::generate_chunk(&perlin, x, z, terrain_size, sys.device.clone());
                m.texture = Some(
                    sys.model_manager
                        .lock()
                        .texture_manager
                        .texture("grass.png"),
                );

                let ter_verts: Vec<Point<f32>> = m
                    .vertices
                    .iter()
                    .map(|v| point![v.position[0], v.position[1], v.position[2]])
                    .collect();
                let ter_indeces: Vec<[u32; 3]> = m
                    .indeces
                    .chunks(3)
                    .map(|slice| [slice[0] as u32, slice[1] as u32, slice[2] as u32])
                    .collect();

                let collider = ColliderBuilder::trimesh(ter_verts, ter_indeces);
                self.chunks.lock().get_mut(&x).unwrap().insert(z, m);
                {
                    // let t = transform.id;
                    // let m_id = sys.model_manager.lock().procedural(m);
                    {
                        // let chunks = self.chunks.clone();
                        sys.defer.append(move |world| {
                            world.sys.lock().physics.collider_set.insert(collider);
                        });
                    };
                }
            });
        });

        // let mut chunks = chunks.lock();
        // for x in -chunk_range..chunk_range {
        //     chunks.insert(x, HashMap::new());
        //     for z in -chunk_range..chunk_range {
        //         let m =
        //             Terrain::generate_chunk(&perlin, x, z, terrain_size, &mut world.sys.model_manager.lock());
        //         let ter_verts: Vec<Point<f32>> = m
        //             .vertices
        //             .iter()
        //             .map(|v| point![v.position[0] + (x * (terrain_size - 1)) as f32, v.position[1], v.position[2] + (z * (terrain_size - 1)) as f32])
        //             .collect();
        //         let ter_indeces: Vec<[u32; 3]> = m
        //             .indeces
        //             .chunks(3)
        //             .map(|slice| [slice[0] as u32, slice[1] as u32, slice[2] as u32])
        //             .collect();

        //         let collider = ColliderBuilder::trimesh(ter_verts, ter_indeces);
        //         world.sys.physics.collider_set.insert(collider);

        //         let m_id = world.sys.model_manager.lock().procedural(m);

        //         let g = world.instantiate_with_transform_with_parent(t, transform::_Transform { position: glm::vec3((x * (terrain_size - 1)) as f32, 0.0, (z * (terrain_size - 1)) as f32), ..Default::default() });
        //         world.add_component(g, Renderer::new(m_id));
        //         chunks.get_mut(&x).unwrap().insert(z, g.t);

        //         // world.add_component(GameObject {t}, Renderer::new(t,m_id));

        //         // rm.renderers.insert(m_id, RendererInstances {model_id: m_id, transforms: vec![]});
        //     }
        // }
    }

    fn generate_chunk(
        noise: &Perlin,
        _x: i32,
        _z: i32,
        terrain_size: i32,
        device: Arc<Device>,
    ) -> Mesh {
        let mut vertices = Vec::new();
        let mut uvs = Vec::new();

        let make_vert = |x: i32, z: i32| {
            let __x = x as f32 + (_x * (terrain_size - 1)) as f32 ;
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
                indeces.push(xz(i, j) as u16);
                indeces.push(xz(i + 1, j + 1) as u16);
                indeces.push(xz(i, j + 1) as u16);
                indeces.push(xz(i, j) as u16);
                indeces.push(xz(i + 1, j) as u16);
                indeces.push(xz(i + 1, j + 1) as u16);
            }
        }

        let mesh = Mesh::new_procedural(vertices, normals, indeces, uvs, device.clone());
        // mesh.texture = Some(mm.texture_manager.texture("grass.png"));
        mesh
    }
}

impl Component for Terrain {
    fn on_render(&mut self) -> Box<dyn FnOnce(&mut RenderJobData) -> ()> {
        let chunks = self.chunks.clone();
        Box::new(move |rd: &mut RenderJobData| {
            let RenderJobData {
                builder,
                transform: _,
                view,
                proj,
                pipeline,
                device,
            } = rd;
            let instance_buffer = vec![0];
            let mvp_buffer = vec![MVP {
                mvp: (**proj * **view * glm::Mat4::identity()).into(),
            }];
            let instance_buffer = CpuAccessibleBuffer::<[i32]>::from_iter(
                device.clone(),
                BufferUsage::storage_buffer(),
                false,
                instance_buffer,
            )
            .unwrap();
            let mvp_buffer = CpuAccessibleBuffer::<[MVP]>::from_iter(
                device.clone(),
                BufferUsage::storage_buffer(),
                false,
                mvp_buffer,
            )
            .unwrap();

            // let mut inst = 0;
            let chunks = chunks.lock();
            for (_, x) in chunks.iter() {
                for (_, z) in x {
                    let layout = pipeline.pipeline.layout().set_layouts().get(0).unwrap();

                    let mut descriptors = Vec::new();

                    descriptors.push(WriteDescriptorSet::buffer(0, mvp_buffer.clone()));

                    if let Some(texture) = z.texture.as_ref() {
                        descriptors.push(WriteDescriptorSet::image_view_sampler(
                            1,
                            texture.image.clone(),
                            texture.sampler.clone(),
                        ));
                    } else {
                        panic!("no terrain texture");
                    }
                    descriptors.push(WriteDescriptorSet::buffer(2, instance_buffer.clone()));
                    if let Ok(set) = PersistentDescriptorSet::new(layout.clone(), descriptors) {
                        builder
                            .bind_descriptor_sets(
                                PipelineBindPoint::Graphics,
                                pipeline.pipeline.layout().clone(),
                                0,
                                set.clone(),
                            )
                            .bind_vertex_buffers(
                                0,
                                (
                                    z.vertex_buffer.clone(),
                                    z.normals_buffer.clone(),
                                    z.uvs_buffer.clone(),
                                    // instance_buffer.clone(),
                                ),
                            )
                            // .bind_vertex_buffers(1, transforms_buffer.data.clone())
                            .bind_index_buffer(z.index_buffer.clone())
                            .draw_indexed(z.indeces.len() as u32, 1, 0, 0, 0)
                            .unwrap();
                    }
                }
            }
        })

        // let render_data = CustRendData {instance_buffer}
    }
    fn update(&mut self, transform: Transform, sys: &crate::engine::System) {
        self.generate(&transform, sys);
    }
}
