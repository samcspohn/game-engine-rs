use std::{collections::HashMap, sync::Arc};

use egui::DragValue;
use noise::{NoiseFn, Perlin};

use nalgebra_glm as glm;
use parking_lot::Mutex;

use rapier3d::prelude::*;
use rayon::prelude::*;
use vulkano::device::Device;

use crate::terrain::transform::Transform;
use crate::{
    engine::{transform, Component, Sys, World},
    inspectable::{Inpsect, Ins, Inspectable},
    model::{Mesh, ModelManager, Normal, Vertex, UV},
    renderer_component2::Renderer,
};

// #[component]
#[derive(Default)]
pub struct Terrain {
    // pub device: Arc<Device>,
    // pub queue: Arc<Queue>,
    // pub texture_manager: Arc<TextureManager>,
    pub chunks: Arc<Mutex<HashMap<i32, HashMap<i32, i32>>>>,
    pub terrain_size: i32,
    pub chunk_range: i32,
}

impl Inspectable for Terrain {
    fn inspect(&mut self, ui: &mut egui::Ui) {
        // egui::CollapsingHeader::new("Terrain")
        //     .default_open(true)
        //     .show(ui, |ui| {
                Ins(&mut self.chunk_range).inspect("chunk_range", ui);
            // });
    }
}

impl Terrain {
    pub fn generate(
        world: &mut World,
        chunks: Arc<Mutex<HashMap<i32, HashMap<i32, i32>>>>,
        terrain_size: i32,
        chunk_range: i32,
        t: i32,
    ) {
        // let collider_set = &world.physics.collider_set;
        // let mm = &mut world.modeling.lock();
        let perlin = Perlin::new();

        {
            let mut chunks = chunks.lock();
            for x in -chunk_range..chunk_range {
                chunks.insert(x, HashMap::new());
            }
        }
        let world = Mutex::new(world);
        (-chunk_range..chunk_range).into_par_iter().for_each(|x| {
            (-chunk_range..chunk_range).into_par_iter().for_each(|z| {
                let mut m = Terrain::generate_chunk(
                    &perlin,
                    x,
                    z,
                    terrain_size,
                    world.lock().sys.model_manager.lock().device.clone(),
                );
                m.texture = Some(
                    world
                        .lock()
                        .sys
                        .model_manager
                        .lock()
                        .texture_manager
                        .texture("grass.png"),
                );

                let ter_verts: Vec<Point<f32>> = m
                    .vertices
                    .iter()
                    .map(|v| {
                        point![
                            v.position[0] + (x * (terrain_size - 1)) as f32,
                            v.position[1],
                            v.position[2] + (z * (terrain_size - 1)) as f32
                        ]
                    })
                    .collect();
                let ter_indeces: Vec<[u32; 3]> = m
                    .indeces
                    .chunks(3)
                    .map(|slice| [slice[0] as u32, slice[1] as u32, slice[2] as u32])
                    .collect();

                let collider = ColliderBuilder::trimesh(ter_verts, ter_indeces);

                let g = {
                    let mut world = world.lock();
                    world.sys.physics.collider_set.insert(collider);

                    let m_id = world.sys.model_manager.lock().procedural(m);

                    let g = world.instantiate_with_transform_with_parent(
                        t,
                        transform::_Transform {
                            position: glm::vec3(
                                (x * (terrain_size - 1)) as f32,
                                0.0,
                                (z * (terrain_size - 1)) as f32,
                            ),
                            ..Default::default()
                        },
                    );
                    world.add_component(g, Renderer::new(m_id));
                    g
                };
                {
                    let mut chunks = chunks.lock();
                    chunks.get_mut(&x).unwrap().insert(z, g.t);
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
            let __x = x as f32 + (_x * (terrain_size - 1)) as f32;
            let __z = z as f32 + (_z * (terrain_size - 1)) as f32;
            Vertex {
                position: [
                    x as f32,
                    noise.get([__x as f64 / 50., __z as f64 / 50.]) as f32 * 10.,
                    z as f32,
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

        let mut normals = Vec::new();
        normals.resize(
            (terrain_size * terrain_size) as usize,
            Normal {
                normal: [0., 0., 0.],
            },
        );
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

        let mut mesh = Mesh::new_procedural(vertices, normals, indeces, uvs, device.clone());
        // mesh.texture = Some(mm.texture_manager.texture("grass.png"));
        mesh
    }
}

impl Component for Terrain {
    // fn assign_transform(&mut self, t: Transform) {
    //     self.t = t;
    // }
    // fn init(&mut self, _t: Transform, _sys: &mut Sys) {

    // }
    fn update(&mut self, transform: Transform, sys: &crate::engine::System) {
        if self.chunks.lock().len() > 0 {
            return;
        }

        let chunks = self.chunks.clone();
        let ts = self.terrain_size;
        let t = transform.id;
        let chunk_range = self.chunk_range;
        sys.defer.append(move |world| {
            Terrain::generate(world, chunks, ts, chunk_range, t);
        });
    }
}
