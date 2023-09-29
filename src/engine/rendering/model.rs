// extern crate assimp;

use std::{ops::Sub, sync::Arc};

// use ai::import::Importer;
// use assimp as ai;
// use bytemuck::{Pod, Zeroable};
use component_derive::AssetID;
use parking_lot::{Mutex, RwLock};

// use std::mem::size_of;
use nalgebra_glm as glm;
// use rapier3d::na::Norm;
use crate::{
    editor::inspectable::Inspectable_,
    engine::{
        project::asset_manager::{self, Asset, AssetManagerBase},
        world::World,
        VulkanManager,
    },
};
use vulkano::{
    buffer::{Buffer, BufferContents, Subbuffer},
    memory::allocator::{MemoryAllocator, MemoryUsage, StandardMemoryAllocator},
    pipeline::graphics::vertex_input::Vertex,
};
use vulkano::{device::Device, impl_vertex};

mod mesh;
// impl_vertex!(glm::Vec3, position);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, BufferContents, Vertex)]
pub struct _Vertex {
    #[format(R32G32B32_SFLOAT)]
    pub position: [f32; 3],
}
// impl_vertex!(Vertex, position);

impl Sub for _Vertex {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            position: [
                self.position[0] - other.position[0],
                self.position[1] - other.position[1],
                self.position[2] - other.position[2],
            ],
        }
    }
}

impl _Vertex {
    pub fn to_vec3(&self) -> glm::Vec3 {
        glm::vec3(self.position[0], self.position[1], self.position[2])
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, BufferContents, Vertex)]
pub struct UV {
    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
}
// impl_vertex!(UV, uv);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, BufferContents, Vertex)]
pub struct Normal {
    #[format(R32G32B32_SFLOAT)]
    pub normal: [f32; 3],
}

// impl_vertex!(Normal, normal);

#[derive(Clone)]
pub struct Mesh {
    pub vertices: Vec<_Vertex>,
    pub normals: Vec<Normal>,
    pub uvs: Vec<UV>,
    pub indeces: Vec<u16>,

    pub vertex_buffer: Subbuffer<[_Vertex]>,
    pub uvs_buffer: Subbuffer<[UV]>,
    pub index_buffer: Subbuffer<[u16]>,
    pub normals_buffer: Subbuffer<[Normal]>,
    pub texture: Option<i32>,
}

impl Mesh {
    pub fn new_procedural(
        vertices: Vec<_Vertex>,
        normals: Vec<Normal>,
        indeces: Vec<u16>,
        uvs: Vec<UV>,
        vk: &VulkanManager,
    ) -> Mesh {
        let vertex_buffer = vk.buffer_from_iter(vertices.clone());
        // Buffer::from_iter(allocator, buffer_usage_all(), false, vertices.clone())
        //     .unwrap();
        let uvs_buffer = vk.buffer_from_iter(uvs.clone());
        let normals_buffer = vk.buffer_from_iter(normals.clone());
        let index_buffer = vk.buffer_from_iter(indeces.clone());

        Mesh {
            vertices,
            uvs,
            indeces,
            normals,
            vertex_buffer,
            uvs_buffer,
            normals_buffer,
            index_buffer,
            texture: None,
        }
    }

    pub fn load_model(
        path: &str,
        texture_manager: Arc<Mutex<TextureManager>>,
        vk: &VulkanManager,
    ) -> Vec<Mesh> {
        let _path = std::path::Path::new(path);
        let model = tobj::load_obj(path, &(tobj::GPU_LOAD_OPTIONS));
        let (meshes, materials) =
            model.expect(format!("Failed to load OBJ file: {}", path).as_str());
        let mut _meshes = Vec::new();
        for (_i, m) in meshes.iter().enumerate() {
            let mut vertices = Vec::new();
            let mut indices = Vec::new();
            let mut normals = Vec::new();
            let mut uvs = Vec::new();

            let mesh = &m.mesh;

            for face in mesh.indices.chunks(3) {
                indices.push(face[0] as u16);
                indices.push(face[2] as u16);
                indices.push(face[1] as u16);
            }
            for v in mesh.positions.chunks(3) {
                vertices.push(_Vertex {
                    position: [v[0], v[1], v[2]],
                });
            }

            for n in mesh.normals.chunks(3) {
                normals.push(Normal {
                    normal: [n[0], n[1], n[2]],
                });
            }

            for uv in mesh.texcoords.chunks(2) {
                uvs.push(UV { uv: [uv[0], uv[1]] });
            }

            let vertex_buffer = vk.buffer_from_iter(vertices.clone());
            let uvs_buffer = vk.buffer_from_iter(uvs.clone());
            let normals_buffer = vk.buffer_from_iter(normals.clone());
            let index_buffer = vk.buffer_from_iter(indices.clone());

            let mut texture = None;
            // for mat in materials.iter() {
            match &materials {
                Ok(mat) => {
                    if let Some(mat_id) = m.mesh.material_id {
                        let m = &mat[mat_id];
                        if let Some(diffuse) = &m.diffuse_texture {
                            let diff_path: &str =
                                &(_path.parent().unwrap().to_str().unwrap().to_string()
                                    + "/"
                                    + m.diffuse_texture.as_ref().unwrap());
                            texture = Some(texture_manager.lock().from_file(diff_path));
                            println!("{}, {:?}", diff_path, texture);
                        }
                    }
                }
                Err(_) => {}
            }
            // }

            _meshes.push(Mesh {
                vertices,
                uvs,
                indeces: indices,
                normals,
                vertex_buffer,
                uvs_buffer,
                normals_buffer,
                index_buffer,
                texture,
            })
        }

        _meshes
    }
}

use crate::engine::project::asset_manager::_AssetID;

use super::{component::buffer_usage_all, texture::TextureManager};
#[derive(AssetID)]
pub struct ModelRenderer {
    pub file: String,
    pub meshes: Vec<Mesh>,
    pub count: u32,
}

impl Asset<ModelRenderer, (Arc<Mutex<TextureManager>>, Arc<VulkanManager>)> for ModelRenderer {
    fn from_file(
        file: &str,
        params: &(Arc<Mutex<TextureManager>>, Arc<VulkanManager>),
    ) -> ModelRenderer {
        let meshes = Mesh::load_model(file, params.0.clone(), &params.1);
        ModelRenderer {
            file: file.into(),
            meshes,
            count: 1,
        }
    }

    fn reload(&mut self, file: &str, params: &(Arc<Mutex<TextureManager>>, Arc<VulkanManager>)) {
        let _mesh = Mesh::load_model(file, params.0.clone(), &params.1);
    }
}

impl Inspectable_ for ModelRenderer {
    fn inspect(&mut self, ui: &mut egui::Ui, _world: &mut World) {
        ui.add(egui::Label::new(self.file.as_str()));
    }
}

pub type ModelManager =
    asset_manager::AssetManager<(Arc<Mutex<TextureManager>>, Arc<VulkanManager>), ModelRenderer>;
