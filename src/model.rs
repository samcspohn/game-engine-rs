// extern crate assimp;

use std::{collections::HashMap, ops::Sub, sync::Arc};

// use ai::import::Importer;
// use assimp as ai;
use bytemuck::{Pod, Zeroable};
use tobj;
// use std::mem::size_of;
use nalgebra_glm as glm;
// use rapier3d::na::Norm;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
    device::Device,
    impl_vertex,
};

use crate::texture::{Texture, TextureManager};

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
    pub texture: Option<Arc<Texture>>, // pub texture: Option<Arc<ImageView<ImmutableImage>>>,
                                       // pub sampler: Option<Arc<Sampler>>,
}

impl Mesh {
    pub fn new_procedural(
        vertices: Vec<Vertex>,
        normals: Vec<Normal>,
        indeces: Vec<u16>,
        uvs: Vec<UV>,
        device: Arc<Device>,
    ) -> Mesh {
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            vertices.clone(),
        )
        .unwrap();
        let uvs_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, uvs.clone())
                .unwrap();
        let normals_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            normals.clone(),
        )
        .unwrap();
        let index_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
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
        device: Arc<Device>,
        texture_manager: Arc<TextureManager>,
    ) -> Mesh {
        // let sub_path = path.split("/");
        let _path = std::path::Path::new(path.into());
        let model = tobj::load_obj(path, &(tobj::GPU_LOAD_OPTIONS));
        let (models, materials) = model.expect("Failed to load OBJ file");
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();

        for (i, m) in models.iter().enumerate() {
            let mesh = &m.mesh;

            println!("model[{}].name = \'{}\'", i, m.name);
            println!("model[{}].mesh.material_id = {:?}", i, mesh.material_id);

            println!(
                "Size of model[{}].face_arities: {}",
                i,
                mesh.face_arities.len()
            );

            // let mut next_face = 0;

            for face in mesh.indices.chunks(3) {
                indices.push(face[0] as u16);
                indices.push(face[2] as u16);
                indices.push(face[1] as u16);
            }

            // Normals and texture coordinates are also loaded, but not printed in this example
            println!("model[{}].vertices: {}", i, mesh.positions.len() / 3);

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
        println!("num indices {}", indices.len());
        println!("num vertices {}", vertices.len());
        println!("num normals {}", normals.len());

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            vertices.clone(),
        )
        .unwrap();
        let uvs_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, uvs.clone())
                .unwrap();
        let normals_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            normals.clone(),
        )
        .unwrap();
        let index_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            indices.clone(),
        )
        .unwrap();

        let mut texture = None;
        // let mut sampler = None;
        for mat in materials.iter() {
            for m in mat {
                // m.diffuse_texture;
                if m.diffuse_texture == "" {
                    continue;
                }
                let diff_path: &str = &(_path.parent().unwrap().to_str().unwrap().to_string()
                    + "/"
                    + &m.diffuse_texture);
                println!("tex {}", diff_path);
                texture = Some(texture_manager.texture(diff_path));
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

pub struct ModelRenderer {
    pub file: String,
    pub mesh: Mesh,
    pub count: u32,
}

pub struct ModelManager {
    pub device: Arc<Device>,
    pub texture_manager: Arc<TextureManager>,
    pub models: HashMap<String, i32>,
    pub models_ids: HashMap<i32, ModelRenderer>,
    pub model_id_gen: i32,
}

impl ModelManager {
    pub fn from_file(&mut self, path: &str) -> i32 {
        let id = self.model_id_gen;
        self.model_id_gen += 1;

        let mesh = Mesh::load_model(path, self.device.clone(), self.texture_manager.clone());
        let m = ModelRenderer { file: path.into() ,mesh, count: 1 };

        self.models_ids.insert(id, m);
        self.models.insert(path.into(), id);
        id
    }

    pub fn procedural(&mut self, mesh: Mesh) -> i32 {
        let id = self.model_id_gen;
        self.model_id_gen += 1;
        let m = ModelRenderer { file: "".into(), mesh, count: 1 };

        // let mesh = Mesh::load_model(path, self.device.clone(), self.texture_manager.clone());

        self.models_ids.insert(id, m);
        id
        // self.models.insert(path.into(), id);
    }
}
