use std::{collections::HashMap, sync::Arc};

use noise::{NoiseFn, Perlin};

use nalgebra_glm as glm;
use rapier3d::prelude::ColliderSet;
use rapier3d::prelude::*;
use vulkano::device::{Device};

use crate::{model::{Mesh, Normal, Vertex, UV}, texture::{TextureManager}};

#[derive(Clone)]
pub struct Terrain {
    pub device: Arc<Device>,
    // pub queue: Arc<Queue>,
    pub texture_manager: Arc<TextureManager>,
    pub chunks: HashMap<i32, HashMap<i32, Mesh>>,
    pub terrain_size: i32,
}

impl Terrain {
    pub fn generate(&mut self, collider_set: &mut ColliderSet) {
        let perlin = Perlin::new();

        for x in -8..8 {
            self.chunks.insert(x, HashMap::new());
            for z in -8..8 {
                let m =
                    Terrain::generate_chunk(&perlin, x, z, self.terrain_size, self.device.clone(), self.texture_manager.clone());
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
                collider_set.insert(collider);
                
                self.chunks.get_mut(&x).unwrap().insert(z, m);
            }
        }
    }

    fn generate_chunk(
        noise: &Perlin,
        _x: i32,
        _z: i32,
        terrain_size: i32,
        device: Arc<Device>,
        // queue: Arc<Queue>
        texture_manager: Arc<TextureManager>
    ) -> Mesh {
        let mut vertices = Vec::new();
        let mut uvs = Vec::new();

        let make_vert = |x: i32, z: i32| {
            let x = x as f32 + (_x * (terrain_size - 1)) as f32;
            let z = z as f32 + (_z * (terrain_size - 1)) as f32;
            Vertex {
                position: [x, noise.get([x as f64 / 50., z as f64 / 50.]) as f32 * 10., z],
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
                uvs.push(UV { uv: [x, z]})
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
        mesh.texture = Some(texture_manager.texture("grass.png"));
        mesh
    }
    
    pub fn update(&mut self) {}
}
