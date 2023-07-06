// extern crate assimp;

use std::{
    ops::Sub,
    sync::Arc,
};

// use ai::import::Importer;
// use assimp as ai;
use bytemuck::{Pod, Zeroable};
use component_derive::AssetID;
use parking_lot::Mutex;

// use std::mem::size_of;
use nalgebra_glm as glm;
// use rapier3d::na::Norm;
use crate::{
    editor::inspectable::Inspectable_,engine::{world::World, project::asset_manager::{self, Asset, AssetManagerBase}},
};
use vulkano::memory::allocator::{MemoryAllocator, StandardMemoryAllocator};
use vulkano::{
    buffer::{CpuAccessibleBuffer},
    device::Device,
    impl_vertex,
};

// impl_vertex!(glm::Vec3, position);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    pub position: [f32; 3],
}
impl_vertex!(Vertex, position);

impl Sub for Vertex {
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

impl Vertex {
    pub fn to_vec3(&self) -> glm::Vec3 {
        glm::vec3(self.position[0], self.position[1], self.position[2])
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct UV {
    pub uv: [f32; 2],
}
impl_vertex!(UV, uv);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Normal {
    pub normal: [f32; 3],
}

impl_vertex!(Normal, normal);

#[derive(Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub normals: Vec<Normal>,
    pub uvs: Vec<UV>,
    pub indeces: Vec<u16>,

    pub vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pub uvs_buffer: Arc<CpuAccessibleBuffer<[UV]>>,
    pub index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
    pub normals_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
    pub texture: Option<i32>, // pub texture: Option<Arc<ImageView<ImmutableImage>>>,
                                       // pub sampler: Option<Arc<Sampler>>,
}

impl Mesh {
    pub fn new_procedural(
        vertices: Vec<Vertex>,
        normals: Vec<Normal>,
        indeces: Vec<u16>,
        uvs: Vec<UV>,
        _device: Arc<Device>,
        allocator: &(impl MemoryAllocator + ?Sized),
    ) -> Mesh {
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            // device.clone(),
            allocator,
            buffer_usage_all(),
            false,
            vertices.clone(),
        )
        .unwrap();
        let uvs_buffer = CpuAccessibleBuffer::from_iter(
            // device.clone(),
            allocator,
            buffer_usage_all(),
            false,
            uvs.clone(),
        )
        .unwrap();
        let normals_buffer = CpuAccessibleBuffer::from_iter(
            // device.clone(),
            allocator,
            buffer_usage_all(),
            false,
            normals.clone(),
        )
        .unwrap();
        let index_buffer = CpuAccessibleBuffer::from_iter(
            // device.clone(),
            allocator,
            buffer_usage_all(),
            false,
            indeces.clone(),
        )
        .unwrap();

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
            // sampler: None,
        }
    }

    pub fn load_model(
        path: &str,
        _device: Arc<Device>,
        texture_manager: Arc<Mutex<TextureManager>>,
        allocator: &(impl MemoryAllocator + ?Sized),
    ) -> Mesh {
        // let sub_path = path.split("/");
        let _path = std::path::Path::new(path);
        let model = tobj::load_obj(path, &(tobj::GPU_LOAD_OPTIONS));
        let (models, materials) = model.expect(format!("Failed to load OBJ file: {}",path).as_str());
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();

        for (_i, m) in models.iter().enumerate() {
            let mesh = &m.mesh;

            // println!("model[{}].name = \'{}\'", i, m.name);
            // println!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);

            // println!(
            //     "Size of model[{}].face_arities: {}",
            //     i,
            //     mesh.face_arities.len()
            // );

            // let mut next_face = 0;

            for face in mesh.indices.chunks(3) {
                indices.push(face[0] as u16);
                indices.push(face[2] as u16);
                indices.push(face[1] as u16);
            }

            // Normals and texture coordinates are also loaded, but not printed in this example
            // println!("model[{}].vertices: {}", i, mesh.positions.len() / 3);

            // assert!(mesh.positions.len() % 3 == 0);
            for v in mesh.positions.chunks(3) {
                vertices.push(Vertex {
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
        }
        // println!("num indices {}", indices.len());
        // println!("num vertices {}", vertices.len());
        // println!("num normals {}", normals.len());

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            // device.clone(),
            allocator,
            buffer_usage_all(),
            false,
            vertices.clone(),
        )
        .unwrap();
        let uvs_buffer = CpuAccessibleBuffer::from_iter(
            // device.clone(),
            allocator,
            buffer_usage_all(),
            false,
            uvs.clone(),
        )
        .unwrap();
        let normals_buffer = CpuAccessibleBuffer::from_iter(
            // device.clone(),
            allocator,
            buffer_usage_all(),
            false,
            normals.clone(),
        )
        .unwrap();
        let index_buffer = CpuAccessibleBuffer::from_iter(
            // device.clone(),
            allocator,
            buffer_usage_all(),
            false,
            indices.clone(),
        )
        .unwrap();

        let mut texture = None;
        // let mut sampler = None;
        for mat in materials.iter() {
            for m in mat {
                // m.diffuse_texture;
                if m.diffuse_texture.is_none() {
                    continue;
                }
                let diff_path: &str = &(_path.parent().unwrap().to_str().unwrap().to_string()
                    + "/"
                    + m.diffuse_texture.as_ref().unwrap());
                // println!("tex {}", diff_path);
                texture = Some(texture_manager.lock().from_file(diff_path));
                // texture = Some(Arc::new(Texture::from_file(diff_path, device.clone(), queue.clone())));
            }
        }

        Mesh {
            vertices,
            uvs,
            indeces: indices,
            normals,
            vertex_buffer,
            uvs_buffer,
            normals_buffer,
            index_buffer,
            texture,
            // sampler,
        }
    }
    // pub fn draw(&self, amount: i32) {}
}

use crate::engine::project::asset_manager::_AssetID;

use super::{renderer_component::buffer_usage_all, texture::TextureManager};
#[derive(AssetID)]
pub struct ModelRenderer {
    pub file: String,
    pub mesh: Mesh,
    pub count: u32,
}

impl
    Asset<
        ModelRenderer,
        (
            Arc<Device>,
            Arc<Mutex<TextureManager>>,
            Arc<StandardMemoryAllocator>,
        ),
    > for ModelRenderer
{
    fn from_file(
        file: &str,
        params: &(
            Arc<Device>,
            Arc<Mutex<TextureManager>>,
            Arc<StandardMemoryAllocator>,
        ),
    ) -> ModelRenderer {
        let mesh = Mesh::load_model(file, params.0.clone(), params.1.clone(), &params.2);
        ModelRenderer {
            file: file.into(),
            mesh,
            count: 1,
        }
    }

    fn reload(
        &mut self,
        file: &str,
        params: &(
            Arc<Device>,
            Arc<Mutex<TextureManager>>,
            Arc<StandardMemoryAllocator>,
        ),
    ) {
        let _mesh = Mesh::load_model(file, params.0.clone(), params.1.clone(), &params.2);
    }
}

impl Inspectable_ for ModelRenderer {
    fn inspect(&mut self, ui: &mut egui::Ui, _world: &parking_lot::Mutex<World>) {
        ui.add(egui::Label::new(self.file.as_str()));
    }
}


pub type ModelManager = asset_manager::AssetManager<
    (
        Arc<Device>,
        Arc<Mutex<TextureManager>>,
        Arc<StandardMemoryAllocator>,
    ),
    ModelRenderer,
>;