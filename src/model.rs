// extern crate assimp;

use std::{io::Cursor, ops::Sub, sync::Arc};

use std::fs::File;
use std::io;
use std::io::prelude::*;
// use ai::import::Importer;
// use assimp as ai;
use bytemuck::{Pod, Zeroable};
use tobj;
// use std::mem::size_of;
use nalgebra_glm as glm;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{ImageDimensions, ImmutableImage, MipmapsCount};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo};
// use rapier3d::na::Norm;
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
    pub texture: Option<Arc<ImageView<ImmutableImage>>>,
    pub sampler: Option<Arc<Sampler>>,
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
            sampler: None,
        }
    }

    pub fn load_model(path: &str, device: Arc<Device>, queue: Arc<Queue>) -> Mesh {
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

            let mut next_face = 0;

            for face in &mesh.indices {
                indices.push(*face as u16);
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
        let mut sampler = None;
        for mat in materials.iter() {
            for m in mat {
                // m.diffuse_texture;
                let diff_path: String = _path.parent().unwrap().to_str().unwrap().to_string() + "/" + &m.diffuse_texture;
                println!("tex {}", diff_path);
                texture = {
                    match File::open(diff_path) {
                        Ok(mut f) => {
                            let mut png_bytes = Vec::new();
                            let _ = f.read_to_end(&mut png_bytes);
                            // let png_bytes = include_bytes!("rust_mascot.png").to_vec();
                            let cursor = Cursor::new(png_bytes);
                            let decoder = png::Decoder::new(cursor);
                            let mut reader = decoder.read_info().unwrap().1;
                            let info = reader.info();
                            let dimensions = ImageDimensions::Dim2d {
                                width: info.width,
                                height: info.height,
                                array_layers: 1,
                            };
                            let mut image_data = Vec::new();
                            image_data.resize((info.width * info.height * 4) as usize, 0);
                            reader.next_frame(&mut image_data).unwrap();

                            let image = ImmutableImage::from_iter(
                                image_data,
                                dimensions,
                                MipmapsCount::One,
                                Format::R8G8B8A8_SRGB,
                                queue.clone(),
                            )
                            .unwrap()
                            .0;

                            Some(ImageView::new_default(image).unwrap())
                        },
                        Err(_) => None
                    }
                };

                sampler = Some(
                    Sampler::new(
                        device.clone(),
                        SamplerCreateInfo {
                            mag_filter: Filter::Linear,
                            min_filter: Filter::Linear,
                            address_mode: [SamplerAddressMode::Repeat; 3],
                            ..Default::default()
                        },
                    )
                    .unwrap(),
                );
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
            sampler,
        }
    }
    pub fn draw(&self, amount: i32) {}
}
