extern crate assimp;

use std::{ops::Sub, sync::Arc};

use ai::import::Importer;
use assimp as ai;
use bytemuck::{Pod, Zeroable};
// use std::mem::size_of;
use nalgebra_glm as glm;
use rapier3d::na::Norm;
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer},
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
pub struct Normal {
    pub normal: [f32; 3],
}

impl_vertex!(Normal, normal);

#[derive(Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub normals: Vec<Normal>,
    pub indeces: Vec<u16>,

    pub vertex_buffer: Arc<CpuAccessibleBuffer<[Vertex]>>,
    pub index_buffer: Arc<CpuAccessibleBuffer<[u16]>>,
    pub normals_buffer: Arc<CpuAccessibleBuffer<[Normal]>>,
}

impl Mesh {
    pub fn new_procedural(
        vertices: Vec<Vertex>,
        normals: Vec<Normal>,
        indeces: Vec<u16>,
        device: Arc<Device>,
    ) -> Mesh {
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            vertices.clone(),
        )
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
            indeces,
            normals,
            vertex_buffer,
            normals_buffer,
            index_buffer,
        }
    }

    pub fn load_model(path: &str, device: Arc<Device>) -> Mesh {
        let importer = Importer::new();
        // let scene = importer.read_file("examples/box.obj");
        // Log to stdout and a file `log.txt`
        // ai::log::add_log_stream(ai::log::Stdout);
        // ai::log::add_log_stream(ai::log::File("log.txt"));
        // ai::log::enable_verbose_logging(true);

        // let importer = ai::Importer::new();

        // The file to import
        let scene = importer.read_file(path).unwrap();

        // let mut m = Mesh{vertices: Vec::new(), indexs : Vec::new()};
        let mut vertices = Vec::new();
        let mut indeces = Vec::new();
        let mut normals = Vec::new();
        // let colors = [[1f32,0f32,0f32],[0f32,1f32,0f32],[0f32,0f32,1f32]];
        // let mut color_index = 0;
        // Print all the vertices in all the meshes
        for mesh in scene.mesh_iter() {
            for vert in mesh.vertex_iter() {
                // println!("vector3: {},{},{}", vert.x,vert.y,vert.z);
                vertices.push(Vertex {
                    position: [vert.x, vert.y, vert.z],
                });
                // color_index = (color_index + 1) % 3;
            }
            for face in mesh.face_iter() {
                // println!("face : {},{},{},{}", face[0],face[1],face[2],face[3]);
                // for i in 0..4 {
                // }
                indeces.push(face[0] as u16);
                indeces.push(face[1] as u16);
                indeces.push(face[2] as u16);

                indeces.push(face[0] as u16);
                indeces.push(face[2] as u16);
                indeces.push(face[3] as u16);
            }
            for norm in mesh.normal_iter() {
                normals.push(Normal {
                    normal: [norm.x, norm.y, norm.z],
                });
            }
        }
        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            false,
            vertices.clone(),
        )
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
            indeces,
            normals,
            vertex_buffer,
            normals_buffer,
            index_buffer,
        }
    }
    pub fn draw(&self, amount: i32) {}
}
