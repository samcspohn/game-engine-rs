extern crate assimp;

use assimp as ai;
use ai::import::Importer;
use bytemuck::{Zeroable, Pod};
// use std::mem::size_of;
use nalgebra_glm as glm;
use rapier3d::na::Norm;
use vulkano::impl_vertex;

// impl_vertex!(glm::Vec3, position);


#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Vertex {
    position: [f32; 3],
}
impl_vertex!(Vertex,position);

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
pub struct Normal {
    normal: [f32; 3],
}

impl_vertex!(Normal, normal);

pub struct Mesh {
    pub vertices : Vec<Vertex>,
    pub normals : Vec<Normal>,
    pub indeces : Vec<u16>,
}

pub fn load_model() -> Mesh{
    let importer = Importer::new();
    // let scene = importer.read_file("examples/box.obj");
    // Log to stdout and a file `log.txt`
    // ai::log::add_log_stream(ai::log::Stdout);
    // ai::log::add_log_stream(ai::log::File("log.txt"));
    // ai::log::enable_verbose_logging(true);

    // let importer = ai::Importer::new();

    // The file to import
    let scene = importer.read_file("src/cube/cube.obj").unwrap();

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
            vertices.push(Vertex {position:[vert.x,vert.y,vert.z]});
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
            normals.push(Normal {normal:[norm.x,norm.y,norm.z]});
        }
    }

    Mesh{vertices, indeces, normals }
}

impl Mesh {
    pub fn draw(&self, amount : i32) {

    }
}