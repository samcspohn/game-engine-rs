// extern crate assimp;

use core::panic;
use std::{
    borrow::Borrow,
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    mem::transmute,
    ops::{Deref, Sub},
    rc::Rc,
    sync::Arc,
    u16::MAX,
};

// use ai::import::Importer;
// use assimp as ai;
// use bytemuck::{Pod, Zeroable};
use crate::engine::{
    particles::shaders::cs::al,
    prelude::{Component, Inpsect, Ins},
    project::asset_manager::AssetInstance,
    rendering::texture::Texture,
    utils,
};
use force_send_sync::SendSync;
use glium::{buffer::Content, vertex};
use glm::{float_bits_to_int, IVec2};
use id::*;
use parking_lot::{Mutex, RwLock};

// use std::mem::size_of;
use nalgebra_glm::{self as glm, quat, vec3, Mat4, Quat, Vec3};
use rapier3d::na::{self, Matrix4x3};
use russimp::{
    animation::{Animation, QuatKey, VectorKey},
    bone::{self, Bone, VertexWeight},
    material::TextureType,
    node::Node,
    scene::{PostProcess, Scene},
    sys::aiQuaternionInterpolate,
    Matrix4x4,
};
use serde::{Deserialize, Serialize};
// use rapier3d::na::Norm;
use crate::{
    editor::inspectable::Inspectable_,
    engine::{
        project::asset_manager::{self, Asset, AssetManagerBase},
        world::World,
        VulkanManager,
    },
};
use rayon::prelude::*;
use vulkano::{
    buffer::{subbuffer::BufferWriteGuard, Buffer, BufferContents, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo, DrawIndexedIndirectCommand,
        PrimaryCommandBufferAbstract,
    },
    memory::allocator::{MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::graphics::vertex_input::Vertex,
    sync::GpuFuture,
    DeviceSize,
};
use vulkano::{device::Device, impl_vertex};
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
    // pub vertices: Vec<_Vertex>, // TODO: change to [f32;3]
    // pub normals: Vec<Normal>,
    // pub uvs: Vec<UV>,
    // pub indices: Vec<u32>,
    // pub bone_ids: Vec<smallvec::SmallVec<[u16; 4]>>,
    pub vertex_bones: Vec<IVec2>,
    pub bone_weight_offsets: Vec<u32>,
    pub aabb: (Vec3, Vec3),

    // pub vertex_buffer: Subbuffer<[_Vertex]>,
    // pub uvs_buffer: Subbuffer<[UV]>,
    // pub index_buffer: Subbuffer<[u32]>,
    // pub normals_buffer: Subbuffer<[Normal]>,
    pub vertex_offset: u32,
    pub vertex_count: u32, // number of vertices in this mesh
    // pub normals_offset: u32,
    // pub uvs_offset: u32,
    pub index_offset: u32,
    pub index_count: u32, // number of indices in this mesh
    // pub indirect_command: DrawIndexedIndirectCommand,
    pub indirect_index: usize,
    pub bone_weights_offsets_counts_buf: Subbuffer<[[u32; 2]]>,
    // pub bone_weights_counts_buf: Subbuffer<[u32]>,
    pub bone_weights_buffer: Option<Subbuffer<[[i32; 2]]>>,
    pub texture: Option<i32>,
}

pub static mut ALL_VERTICES: Vec<_Vertex> = Vec::new();
pub static mut ALL_NORMALS: Vec<Normal> = Vec::new();
pub static mut ALL_UVS: Vec<UV> = Vec::new();
pub static mut ALL_INDICES: Vec<u32> = Vec::new();

pub static mut ALL_VERTEX_BUFFER: Option<Subbuffer<[_Vertex]>> = None;
pub static mut ALL_NORMALS_BUFFER: Option<Subbuffer<[Normal]>> = None;
pub static mut ALL_UVS_BUFFER: Option<Subbuffer<[UV]>> = None;
pub static mut ALL_INDICES_BUFFER: Option<Subbuffer<[u32]>> = None;

pub fn force_update_mesh_buffers(vk: &VulkanManager) {
    unsafe { ALL_VERTEX_BUFFER = Some(vk.buffer_from_iter(unsafe { ALL_VERTICES.clone() })) };
    unsafe { ALL_NORMALS_BUFFER = Some(vk.buffer_from_iter(unsafe { ALL_NORMALS.clone() })) };
    unsafe { ALL_UVS_BUFFER = Some(vk.buffer_from_iter(unsafe { ALL_UVS.clone() })) };
    unsafe { ALL_INDICES_BUFFER = Some(vk.buffer_from_iter(unsafe { ALL_INDICES.clone() })) };
}
#[derive(Clone, Serialize, Deserialize)]
pub struct BoneInfo {
    pub offset: Mat4,
    pub final_transformation: Mat4,
}

#[derive(Default, Clone)]
struct _VertexWeight {
    pub weight: f32,
    pub vertex_id: u32,
}
#[derive(Default, Clone)]
pub struct _Bone {
    pub weights: Vec<_VertexWeight>,
    pub name: String,
    pub offset_matrix: Mat4,
}

struct BoneNode {
    name: String,
    transformation: Mat4,
    children: Vec<BoneNode>,
}

struct Anim {
    position_keys: Vec<VectorKey>,
    rotation_keys: Vec<QuatKey>,
    scale_keys: Vec<VectorKey>,
}
pub struct Model {
    pub meshes: Vec<Mesh>,
    // pub animations: Vec<Animation>,
    pub scene: force_send_sync::SendSync<Arc<Scene>>,
    pub bone_hierarchy: BoneNode,
    pub has_skeleton: bool,
    pub bone_names_index: HashMap<String, (u32, _Bone)>,
    pub bone_info: Vec<Mat4>,
}
fn create_hierarchy(bn: &mut BoneNode, node: &Node) {
    // let root = node.;
    // let mut children = Vec::new();
    for child in node.children.borrow().iter() {
        let mut child_node = BoneNode {
            name: child.name.clone(),
            transformation: glm::transpose(unsafe { &transmute(child.transformation) }),
            children: Vec::new(),
        };
        create_hierarchy(&mut child_node, &child);
        bn.children.push(child_node);
    }
    // let mut bone_hierarchy = BoneNode { name: node.name.clone(),  transformation: glm::transpose(unsafe { &transmute(node.transformation) }), children: Vec::new() };
}
impl Model {
    pub fn load_model(
        path: &str,
        texture_manager: Arc<Mutex<TextureManager>>,
        vk: &VulkanManager,
        renderers: &mut SharedRendererData,
    ) -> Model {
        let _path = std::path::Path::new(path);
        let model = russimp::scene::Scene::from_file(
            path,
            vec![
                PostProcess::CalculateTangentSpace,
                PostProcess::Triangulate,
                PostProcess::JoinIdenticalVertices,
                PostProcess::SortByPrimitiveType,
            ],
        );
        // println!("model stats: {:?}", model);
        // let model = tobj::load_obj(path, &(tobj::GPU_LOAD_OPTIONS));
        let (scene) = model.expect(format!("Failed to load OBJ file: {}", path).as_str());

        let bone_names_index = scene
            .meshes
            .iter()
            .map(|m| {
                m.bones
                    .iter()
                    .map(|(bone)| {
                        (
                            bone.name.clone(),
                            _Bone {
                                weights: bone
                                    .weights
                                    .iter()
                                    .map(|vw| _VertexWeight {
                                        weight: vw.weight,
                                        vertex_id: vw.vertex_id,
                                    })
                                    .collect(),
                                name: bone.name.clone(),
                                offset_matrix: glm::transpose(unsafe {
                                    &transmute(bone.offset_matrix.clone())
                                }),
                            },
                        )
                    })
                    .into_iter()
            })
            .flatten()
            .collect::<HashMap<String, _Bone>>()
            .into_iter()
            .enumerate()
            .map(|(id, (b_name, b_))| (b_name, (id as u32, b_)))
            .collect::<HashMap<String, (u32, _Bone)>>();

        let bone_info: Vec<Mat4> = bone_names_index
            .iter()
            .map(|(b_name, (id, b_))| b_.offset_matrix)
            .collect();

        let root = scene.root.borrow().as_ref().unwrap();
        let mut bone_hierarchy = BoneNode {
            name: root.name.clone(),
            transformation: glm::transpose(unsafe { &transmute(root.transformation) }),
            children: Vec::new(),
        };
        create_hierarchy(&mut bone_hierarchy, &root);

        let mut _meshes = Vec::new();
        println!("here");

        let mut tex_map = HashMap::new();
        for mat in scene.materials.iter() {
            // println!("texture: {:?}", mat.textures);
            // println!("properties: {:?}", mat.properties);
            for tex in mat.textures.iter() {
                if *tex.0 != TextureType::Diffuse {
                    continue; // Only process diffuse textures
                }
                let tex_val: &russimp::material::Texture = &tex.1.deref().borrow();
                tex_map.entry(tex_val.filename.clone()).or_insert(unsafe {
                    force_send_sync::SendSync::new(std::ptr::addr_of!(*tex_val))
                });
            }
        }
        {
            rayon::scope(|s| {
                for tex in tex_map.into_iter() {
                    // Do something with the texture map
                    let tex_man = &texture_manager;
                    s.spawn(move |_| {
                        let tex = Texture::from_embedded_texture(
                            unsafe { &**tex.1 },
                            _path.parent().unwrap().to_str().unwrap(),
                            vk,
                        );
                        tex_man.lock().from_texture(&tex);
                    });
                    // tex_man.from_embedded_texture(&tex.1.1.deref().borrow(), _path.parent().unwrap().to_str().unwrap());
                }
            });
        }

        for mesh in scene.meshes.iter() {
            if let Some(mesh) = Mesh::load_mesh(
                &mesh,
                &scene,
                &bone_names_index,
                texture_manager.clone(),
                &_path,
                vk,
                renderers,
            ) {
                _meshes.push(mesh);
            }
        }
        println!("number of animations: {}", scene.animations.len());
        // scene.root;

        // let mut _anims = Vec::new();
        // for anim in scene.animations.iter() {

        // }

        Model {
            meshes: _meshes,
            // animations: scene.animations.clone(),
            has_skeleton: bone_info.len() > 0,
            bone_info,
            bone_names_index,
            scene: unsafe { SendSync::new(Arc::new(scene)) },
            bone_hierarchy,
        }
    }
}
impl Mesh {
    // pub fn new_procedural(
    //     vertices: Vec<_Vertex>,
    //     normals: Vec<Normal>,
    //     indeces: Vec<u32>,
    //     uvs: Vec<UV>,
    //     vk: &VulkanManager,
    // ) -> SeletalMesh {
    //     let vertex_buffer = vk.buffer_from_iter(vertices.clone());
    //     let uvs_buffer = vk.buffer_from_iter(uvs.clone());
    //     let normals_buffer = vk.buffer_from_iter(normals.clone());
    //     let index_buffer = vk.buffer_from_iter(indeces.clone());

    //     SeletalMesh {
    //         vertices,
    //         uvs,
    //         indeces,
    //         normals,
    //         vertex_buffer,
    //         uvs_buffer,
    //         normals_buffer,
    //         index_buffer,
    //         texture: None,
    //     }
    // }

    pub fn load_mesh(
        mesh: &russimp::mesh::Mesh,
        scene: &russimp::scene::Scene,
        bone_name_index: &HashMap<String, (u32, _Bone)>,
        texture_manager: Arc<Mutex<TextureManager>>,
        _path: &std::path::Path,
        vk: &VulkanManager,
        renderers: &mut SharedRendererData,
    ) -> Option<Mesh> {
        // let mut vertices = Vec::new();
        // let mut indices = Vec::new();
        // let mut normals = Vec::new();
        // let mut uvs = Vec::new();
        let mut vertex_bones = Vec::new();

        let index_offset = unsafe { ALL_INDICES.len() as u32 };
        let vertex_offset = unsafe { ALL_VERTICES.len() as u32 };
        if mesh.vertices.len() == 0 {
            println!("Mesh has no vertices, skipping");
            return None;
        }

        // let mesh = &m.mesh;
        let mut skip = false;
        for face in mesh.faces.iter() {
            if face.0.len() < 3 {
                skip = true;
                break;
            }
            unsafe {
                ALL_INDICES.push(face.0[0]);
                ALL_INDICES.push(face.0[2]);
                ALL_INDICES.push(face.0[1]);
            }
        }
        if skip {
            return None;
        }
        let mut max = Vec3::new(f32::MIN, f32::MIN, f32::MIN);
        let mut min = Vec3::new(f32::MAX, f32::MAX, f32::MAX);
        for v in mesh.vertices.iter() {
            max.x = max.x.max(v.x);
            max.y = max.y.max(v.y);
            max.z = max.z.max(v.z);
            min.x = min.x.min(v.x);
            min.y = min.y.min(v.y);
            min.z = min.z.min(v.z);

            unsafe {
                ALL_VERTICES.push(_Vertex {
                    position: [v.x, v.y, v.z],
                })
            };
        }

        if mesh.normals.len() == 0 {
            unsafe {
                ALL_NORMALS.extend(vec![
                    Normal {
                        normal: [0.0, 1.0, 0.0],
                    };
                    mesh.vertices.len()
                ]);
            }
        } else {
            for n in mesh.normals.iter() {
                unsafe {
                    ALL_NORMALS.push(Normal {
                        normal: [n.x, n.y, n.z],
                    })
                };
            }
        }

        if mesh.texture_coords.len() == 0 {
            unsafe {
                ALL_UVS.extend(vec![UV { uv: [0.0, 0.0] }; mesh.vertices.len()]);
            }
        } else {
            for tex_coords in mesh.texture_coords.iter() {
                if let Some(tex_coords) = tex_coords {
                    for uv in tex_coords {
                        unsafe {
                            ALL_UVS.push(UV { uv: [uv.x, uv.y] });
                        } // TODO: support multiple uvs
                    }
                    break;
                }
            }
        }
        if unsafe { ALL_UVS.len() } < unsafe { ALL_VERTICES.len() } {
            // Fill remaining uvs with [0.0, 0.0]
            unsafe {
                ALL_UVS.extend(vec![
                    UV { uv: [0.0, 0.0] };
                    unsafe { ALL_VERTICES.len() } - ALL_UVS.len()
                ]);
            }
        }

        let mut vertex_weights = BTreeMap::<u32, Vec<(u32, f32)>>::new(); // vertexid -> [(bone,weight)]

        for bone in mesh.bones.iter() {
            if let Some((id, _bone)) = bone_name_index.get(&bone.name) {
                for weight in &bone.weights {
                    vertex_weights
                        .entry(weight.vertex_id as u32)
                        .or_default()
                        .push((*id, weight.weight));
                }
            }
        }
        let mut offset: u32 = 0;
        // let mut vert_weight = vertex_weights.iter();
        // let mut vert = vert_weight.next();
        let mut bone_weight_offsets = Vec::new();
        for i in 0..(unsafe { ALL_VERTICES.len() } as u32 - vertex_offset) {
            bone_weight_offsets.push(offset);
            if let Some(weights) = vertex_weights.get(&i) {
                // if *vw.0 == i {
                offset += weights.len() as u32;
                // }
            }
        }
        for (vertex_id, weights) in vertex_weights.iter() {
            for (bone_id, weight) in weights {
                vertex_bones.push(IVec2::new(*bone_id as i32, float_bits_to_int(*weight)));
            }
        }

        // let vertex_buffer = vk.buffer_from_iter(vertices.clone());
        // if uvs.len() == 0 {
        //     uvs = vec![UV { uv: [0.0, 0.0] }; vertices.len()];
        // }
        // let uvs_buffer = vk.buffer_from_iter(uvs.clone());
        // let normals_buffer = vk.buffer_from_iter(normals.clone());
        // let index_buffer = vk.buffer_from_iter(indices.clone());
        if unsafe { ALL_INDICES_BUFFER.is_none() } {
            unsafe {
                ALL_VERTEX_BUFFER = Some(vk.buffer_from_iter(unsafe { ALL_VERTICES.clone() }))
            };
            unsafe {
                ALL_NORMALS_BUFFER = Some(vk.buffer_from_iter(unsafe { ALL_NORMALS.clone() }))
            };
            unsafe { ALL_UVS_BUFFER = Some(vk.buffer_from_iter(unsafe { ALL_UVS.clone() })) };
            unsafe {
                ALL_INDICES_BUFFER = Some(vk.buffer_from_iter(unsafe { ALL_INDICES.clone() }))
            };
        }

        if let (Some(all_vertex_buffer), Some(all_indices_buffer)) =
            unsafe { (ALL_VERTEX_BUFFER.as_ref(), ALL_INDICES_BUFFER.as_ref()) }
        {
            let mut builder = AutoCommandBufferBuilder::primary(
                &vk.comm_alloc,
                vk.queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            if all_vertex_buffer.len() < unsafe { ALL_VERTICES.len() as DeviceSize }
                || all_indices_buffer.len() < unsafe { ALL_INDICES.len() as DeviceSize }
            {
                // resize vertices
                let new_len = unsafe { ALL_VERTICES.len().next_power_of_two() as u64 };
                let new_buffer = vk.buffer_array(new_len, MemoryTypeFilter::PREFER_DEVICE);
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        unsafe { ALL_VERTEX_BUFFER.as_ref().unwrap().clone() },
                        new_buffer.clone(),
                    ))
                    .unwrap();
                unsafe { ALL_VERTEX_BUFFER = Some(new_buffer) };

                // resize normals
                let new_normals_buffer = vk.buffer_array(new_len, MemoryTypeFilter::PREFER_DEVICE);
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        unsafe { ALL_NORMALS_BUFFER.as_ref().unwrap().clone() },
                        new_normals_buffer.clone(),
                    ))
                    .unwrap();
                unsafe { ALL_NORMALS_BUFFER = Some(new_normals_buffer) };

                // resize uvs
                let new_uvs_buffer = vk.buffer_array(new_len, MemoryTypeFilter::PREFER_DEVICE);
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        unsafe { ALL_UVS_BUFFER.as_ref().unwrap().clone() },
                        new_uvs_buffer.clone(),
                    ))
                    .unwrap();
                unsafe { ALL_UVS_BUFFER = Some(new_uvs_buffer) };

                // resize indices
                let new_len = unsafe { ALL_INDICES.len().next_power_of_two() as u64 };
                let new_indices_buffer = vk.buffer_array(new_len, MemoryTypeFilter::PREFER_DEVICE);
                builder
                    .copy_buffer(CopyBufferInfo::buffers(
                        unsafe { ALL_INDICES_BUFFER.as_ref().unwrap().clone() },
                        new_indices_buffer.clone(),
                    ))
                    .unwrap();
                unsafe { ALL_INDICES_BUFFER = Some(new_indices_buffer) };
            }

            let buf = vk.buffer_from_iter(unsafe {
                ALL_VERTICES[vertex_offset as usize..].iter().cloned()
            });
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    buf,
                    all_vertex_buffer.clone().slice(vertex_offset as u64..),
                ))
                .unwrap();

            // Copy normals
            let normals_buf = vk
                .buffer_from_iter(unsafe { ALL_NORMALS[vertex_offset as usize..].iter().cloned() });
            let all_normals_buffer = unsafe { ALL_NORMALS_BUFFER.as_ref().unwrap().clone() };
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    normals_buf,
                    all_normals_buffer.slice(vertex_offset as u64..).clone(),
                ))
                .unwrap();

            // Copy uvs
            let uvs_buf =
                vk.buffer_from_iter(unsafe { ALL_UVS[vertex_offset as usize..].iter().cloned() });
            let all_uvs_buffer = unsafe { ALL_UVS_BUFFER.as_ref().unwrap().clone() };
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    uvs_buf,
                    all_uvs_buffer.slice(vertex_offset as u64..).clone(),
                ))
                .unwrap();

            // Copy indices
            let indices_buf = vk
                .buffer_from_iter(unsafe { ALL_INDICES[index_offset as usize..].iter().cloned() });
            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    indices_buf,
                    all_indices_buffer.clone().slice(index_offset as u64..),
                ))
                .unwrap();

            let command_buffer = builder.build().unwrap();
            command_buffer
                .execute(vk.queue.clone())
                .unwrap()
                .then_signal_fence_and_flush()
                .unwrap();
        }

        let bone_weights_buffer: Option<Subbuffer<[[i32; 2]]>> = if vertex_bones.len() == 0 {
            None
        } else {
            Some(vk.buffer_from_iter(vertex_bones.iter().map(|i| [i.x, i.y])))
        };
        let bone_weights_offsets_counts_buffer =
            if bone_weight_offsets.len() == 0 || vertex_weights.len() == 0 {
                vk.buffer_array(1, MemoryTypeFilter::PREFER_DEVICE)
            } else {
                vk.buffer_from_iter(
                    bone_weight_offsets
                        .iter()
                        .zip(
                            vertex_weights
                                .iter()
                                .map(|(vert, weights)| weights.len() as u32),
                        )
                        .map(|(of, co)| [*of, co]),
                )
            };
        // let bone_weights_counts_buf = if vertex_weights.len() == 0 {
        //     vk.buffer_array(1, MemoryTypeFilter::PREFER_DEVICE)
        // } else {
        //     vk.buffer_from_iter(
        //         vertex_weights
        //             .iter()
        //             .map(|(vert, weights)| weights.len() as u32),
        //     )
        // };

        let mut texture = None;
        // for mat in materials.iter() {
        // println!(
        //     "materials: {}, {:?}",
        //     mesh.material_index,
        //     scene.materials.get(mesh.material_index as usize)
        // );
        if let Some(mat) = scene.materials.get(mesh.material_index as usize) {
            if let Some(tex) = mat.textures.get(&TextureType::Diffuse) {
                let tex_val: &russimp::material::Texture = &tex.deref().borrow();
                // if tex_val.filename.starts_with("*") {

                //     // Embedded texture
                //     let embedded_texture_ref = tex_val.deref().borrow();
                //     texture = Some(texture_manager.lock().from_embedded_texture(
                //         &*embedded_texture_ref,
                //         _path.parent().unwrap().to_str().unwrap(),
                //     ));
                // } else {
                if !tex_val.filename.starts_with("*") {
                    // External texture file
                    let diff_path: &str = &(_path.parent().unwrap().to_str().unwrap().to_string()
                        + "/"
                        + &tex_val.filename);
                    println!("Loading external diffuse texture: {}", diff_path);
                    texture = Some(texture_manager.lock().from_file(diff_path));
                }
            }
            if texture.is_none() {
                for prop in &mat.properties {
                    // println!("prop.key: {}, prop.data: {:?}", prop.key, prop.data);
                    match prop.semantic {
                        TextureType::Diffuse => {
                            match &prop.data {
                                russimp::material::PropertyTypeInfo::String(s) => {
                                    // Check if it's an embedded texture reference (starts with "*")
                                    if s.starts_with("*") {
                                        continue; // Skip embedded textures for now
                                                  // Extract embedded texture index
                                                  // if let Ok(texture_index) = s[1..].parse::<usize>() {
                                                  //     // For embedded textures, we need to access them through material.textures
                                                  //     if let Some(embedded_texture_rc) = mat.textures.get(&TextureType::Diffuse) {
                                                  //         println!("Loading embedded diffuse texture at index: {}", texture_index);
                                                  //         let embedded_texture_ref = (**embedded_texture_rc).borrow();
                                                  //         texture = Some(texture_manager.lock().from_embedded_texture(&*embedded_texture_ref, _path.parent().unwrap().to_str().unwrap()));
                                                  //         println!("Loaded embedded texture: {:?}", texture);
                                                  //     } else {
                                                  //         println!("Embedded texture not found in material.textures");
                                                  //     }
                                                  // } else {
                                                  //     println!("Failed to parse embedded texture index from: {}", s);
                                                  // }
                                    } else {
                                        // External texture file
                                        let diff_path: &str = &(_path
                                            .parent()
                                            .unwrap()
                                            .to_str()
                                            .unwrap()
                                            .to_string()
                                            + "/"
                                            + &s);
                                        println!("Loading external diffuse texture: {}", diff_path);
                                        texture = Some(texture_manager.lock().from_file(diff_path));
                                        println!("Loaded external texture: {:?}", texture);
                                    }
                                }
                                russimp::material::PropertyTypeInfo::Buffer(data) => {
                                    // Direct embedded texture data in buffer
                                    println!("Loading diffuse texture from direct buffer data");
                                    // panic!("Direct buffer data loading not implemented");
                                    // texture = Some(texture_manager.lock().from_buffer_data(data, &prop.key,prop.));
                                    println!("Loaded texture from buffer: {:?}", texture);
                                }
                                _ => {
                                    println!(
                                        "Unsupported diffuse texture data type: {:?}",
                                        prop.data
                                    );
                                }
                            }
                        }
                        TextureType::Specular => {
                            println!("specular texture: {:?}", prop.data);
                        }
                        TextureType::Normals => {
                            println!("normal texture: {:?}", prop.data);
                        }
                        _ => {}
                    }
                }
            }
            // }
            // for (_type, tex) in &mat.textures {
            //     println!("diffuse path: {:?}, {}", _type, tex.borrow().filename);
            //     if *_type == TextureType::Diffuse {
            //         let diff_path: &str =
            //             &(_path.parent().unwrap().to_str().unwrap().to_string()
            //                 + "/"
            //                 + &tex.borrow().filename);
            //         println!("diffuse path: {}", diff_path);
            //         texture = Some(texture_manager.lock().from_file(diff_path));
            //         println!("{}, {:?}", diff_path, texture);
            //     }
            // }

            // if let Some(diffuse) = mat.textures {
            //     let diff_path: &str =
            //         &(_path.parent().unwrap().to_str().unwrap().to_string()
            //             + "/"
            //             + m.diffuse_texture.as_ref().unwrap());
            //     texture = Some(texture_manager.lock().from_file(diff_path));
            //     println!("{}, {:?}", diff_path, texture);
            // }
        }
        let aabb = (min, max);
        println!("aabb: {:?}", aabb);
        // }
        let indirect_index = renderers.indirect.len();
        renderers.indirect.push(DrawIndexedIndirectCommand {
            index_count: unsafe { ALL_INDICES.len() as u32 - index_offset },
            instance_count: 0,
            first_index: index_offset,
            vertex_offset,
            first_instance: 0,
        });
        let tex_man_l = texture_manager.lock();
        let tex_man: &TextureManager = &*tex_man_l;
        let tex = tex_man.assets_id.get(texture.as_ref().unwrap_or(&0)).map(|tex| {
            tex.lock().index
        }).unwrap();
        renderers.texture_ids.push(tex as i32);
        renderers.indirect_counts.push(0);

        return Some(Mesh {
            // vertices,
            // uvs,
            // indices,
            // normals,
            vertex_bones,
            bone_weight_offsets,
            // vertex_buffer,
            // uvs_buffer,
            // normals_buffer,
            // index_buffer,
            vertex_offset: vertex_offset.clone(),
            vertex_count: unsafe { ALL_VERTICES.len() as u32 - vertex_offset },
            index_offset,
            indirect_index,
            index_count: unsafe { ALL_INDICES.len() as u32 - index_offset },
            texture,
            bone_weights_buffer,
            bone_weights_offsets_counts_buf: bone_weights_offsets_counts_buffer,
            aabb,
        });
        // }

        // _meshes
    }
}

use super::{
    component::{buffer_usage_all, SharedRendererData},
    texture::TextureManager,
};
#[derive(ID)]
pub struct ModelRenderer {
    pub file: String,
    pub model: Model,
    pub count: u32,
}

impl
    Asset<
        ModelRenderer,
        (
            Arc<Mutex<TextureManager>>,
            Arc<VulkanManager>,
            Arc<RwLock<SharedRendererData>>,
        ),
    > for ModelRenderer
{
    fn from_file(
        file: &str,
        params: &(
            Arc<Mutex<TextureManager>>,
            Arc<VulkanManager>,
            Arc<RwLock<SharedRendererData>>,
        ),
    ) -> ModelRenderer {
        let model = Model::load_model(file, params.0.clone(), &params.1, &mut params.2.write());
        ModelRenderer {
            file: file.into(),
            model,
            count: 1,
        }
    }

    fn reload(
        &mut self,
        file: &str,
        params: &(
            Arc<Mutex<TextureManager>>,
            Arc<VulkanManager>,
            Arc<RwLock<SharedRendererData>>,
        ),
    ) {
        let _mesh = Model::load_model(file, params.0.clone(), &params.1, &mut params.2.write());
    }
}

impl Inspectable_ for ModelRenderer {
    fn inspect(&mut self, ui: &mut egui::Ui, _world: &mut World) -> bool {
        ui.add(egui::Label::new(self.file.as_str()));
        ui.separator();
        self.model
            .scene
            .animations
            .iter()
            .enumerate()
            .for_each(|(i, x)| {
                ui.add(egui::Label::new(format!("{}: {}", x.name, i)));
            });
        true
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub type ModelManager = asset_manager::AssetManager<
    (
        Arc<Mutex<TextureManager>>,
        Arc<VulkanManager>,
        Arc<RwLock<SharedRendererData>>,
    ),
    ModelRenderer,
>;

#[derive(Default, Clone)]
pub struct Skeleton {
    // pub model: AssetInstance<ModelRenderer>,
    pub anim_id: usize,
    // pub bones: Vec<Mat4>,
    // pub bone_info: Vec<BoneInfo>,
}

// impl Component for Skeleton {
//     fn inspect(
//         &mut self,
//         transform: &crate::engine::prelude::Transform,
//         id: i32,
//         ui: &mut egui::Ui,
//         sys: &crate::engine::prelude::Sys,
//     ) {
//         Ins(&mut self.model).inspect("model", ui, sys);
//     }
// }

fn calc_interpolated_vector(t: &Vec<VectorKey>, time: f64) -> Vec3 {
    // let mut scaling = Vec3::new(0,0,0);
    if t.len() == 1 {
        let v = t[0].value;
        vec3(v.x, v.y, v.z)
    } else {
        t.iter()
            .as_slice()
            .windows(2)
            .filter(|x| time >= x[0].time && time < x[1].time)
            .map(|x| {
                let t1 = x[0].time;
                let t2 = x[1].time;
                let delta_time = t2 - t1;
                let factor = (time - t1) / delta_time;
                let start = x[0].value;
                let start = vec3(start.x, start.y, start.z);
                let end = x[1].value;
                let end = vec3(end.x, end.y, end.z);
                let delta = end - start;
                start + factor as f32 * delta
                // glm::scale(&scaling, out)
            })
            .collect::<Vec<Vec3>>()[0]
    }
}
fn interpolate(p_start: &Quat, p_end: &Quat, factor: f32) -> Quat {
    // calc cosine theta
    let mut cosom = p_start.coords.x * p_end.coords.x
        + p_start.coords.y * p_end.coords.y
        + p_start.coords.z * p_end.coords.z
        + p_start.coords.w * p_end.coords.w;

    // adjust signs (if necessary)
    let mut end: Quat = p_end.clone();
    if (cosom < 0.) {
        cosom = -cosom;
        end.coords.x = -end.coords.x; // Reverse all signs
        end.coords.y = -end.coords.y;
        end.coords.z = -end.coords.z;
        end.w = -end.w;
    }

    // Calculate coefficients
    let mut sclp = 0.;
    let mut sclq = 0.;

    //  if 1.0 - cosom > 0.01 // 0.0001 -> some epsillon
    //  {
    // Standard case (slerp)
    let mut omega = 0.;
    let mut sinom = 0.;
    omega = cosom.cos(); // extract theta from dot product's cos theta
    sinom = omega.sin();
    sclp = ((1.0 - factor) * omega).sin() / sinom;
    sclq = (factor * omega).sin() / sinom;
    //  } else
    //  {
    //      // Very close, do linear interp (because it's faster)
    //      sclp = 1.0 - factor;
    //      sclq = factor;
    //  }

    let x = sclp * p_start.coords.x + sclq * end.coords.x;
    let y = sclp * p_start.coords.y + sclq * end.coords.y;
    let z = sclp * p_start.coords.z + sclq * end.coords.z;
    let w = sclp * p_start.w + sclq * end.w;
    quat(x, y, z, w)
}
fn calc_interpolated_quat(t: &Vec<QuatKey>, time: f64) -> Quat {
    if t.len() == 1 {
        let v = t[0].value;
        quat(v.x, v.y, v.z, v.w)
    } else {
        t.iter()
            .as_slice()
            .windows(2)
            .filter(|x| time >= x[0].time && time < x[1].time)
            .map(|x| {
                let t1 = x[0].time;
                let t2 = x[1].time;
                let delta_time = t2 - t1;
                let factor = (time - t1) / delta_time;
                let start = x[0].value;
                let start = quat(start.x, start.y, start.z, start.w);
                let end = x[1].value;
                let end = quat(end.x, end.y, end.z, end.w);

                interpolate(&start, &end, factor as f32).normalize()
                // let a = start.lerp(&end, factor as f32);
                // let v = vec3(a.i, a.j, a.k).normalize();
                // let a = quat(v.x, v.y, v.z, a.w).normalize();
                // a
                // start
                // let delta = end - start;
                // start + factor as f32 * delta
                // glm::scale(&scaling, out)
            })
            .collect::<Vec<Quat>>()
            .pop()
            .unwrap_or(quat(0., 0., 0., 1.))
    }
}
impl Skeleton {
    pub fn new(anim_id: usize) -> Skeleton {
        Skeleton {
            // model: m_id,
            anim_id,
        }
    }
    fn read_node_hierarchy(
        &mut self,
        inverse_transform: &Mat4,
        time: f64,
        node: &BoneNode,
        parent_transform: &Mat4,
        anim: &Animation,
        bone_names_index: &HashMap<String, (u32, _Bone)>,
        bones: &mut [[[f32; 4]; 3]],
        // anim_id: usize,
    ) {
        let name = node.name.clone();
        // let anim = &scene.animations[anim_id];
        // let mut node_transform = node.transformation;

        let mut node_anim = None;
        // find node anim
        for channel in &anim.channels {
            if channel.name == name {
                node_anim = Some(channel);
            }
        }
        let node_transform = if let Some(node_anim) = node_anim {
            let scl = calc_interpolated_vector(&node_anim.scaling_keys, time);
            let scaling = glm::scaling(&scl);

            let pos = calc_interpolated_vector(&node_anim.position_keys, time);
            let translation = glm::translation(&pos);

            let rot = calc_interpolated_quat(&node_anim.rotation_keys, time);
            let rotation = glm::quat_to_mat4(&rot);

            translation * rotation * scaling
        } else {
            node.transformation
        };

        let global_transform = parent_transform * node_transform;

        if let Some((bone_index, bone_)) = bone_names_index.get(&name) {
            let a = inverse_transform * global_transform * bone_.offset_matrix;
            let a = (a.row(0), a.row(1), a.row(2));
            bones[*bone_index as usize] = [
                [a.0[0], a.0[1], a.0[2], a.0[3]],
                [a.1[0], a.1[1], a.1[2], a.1[3]],
                [a.2[0], a.2[1], a.2[2], a.2[3]],
            ];
        }
        for child in node.children.iter() {
            self.read_node_hierarchy(
                &inverse_transform,
                time,
                &child,
                &global_transform,
                anim,
                bone_names_index,
                bones,
                // anim_id,
            );
        }
    }
    pub fn get_skeleton(&mut self, model: &Model, time: f64, bones: &mut [[[f32; 4]; 3]]) {
        // model_manager
        //     .assets_id
        //     .get(&self.model.id)
        //     .and_then(|x| Some(x.lock()))
        //     .and_then(|x| {
        // let anim_id = x
        //     .model
        //     .scene
        //     .animations
        //     .iter()
        //     .enumerate()
        //     .filter(|(i, a)| a.name.contains("attack2"))
        //     .map(|(i, a)| i)
        //     .next()
        //     .unwrap_or(0);

        // self.anim_id = anim_id;

        // let time = time % x.model.scene.animations[0].duration;

        let time_in_ticks = time * model.scene.animations[self.anim_id].ticks_per_second;
        let animation_time = time_in_ticks % model.scene.animations[self.anim_id].duration;

        // let mut bones = Vec::with_capacity(model.bone_info.len());
        // unsafe { bones.set_len(model.bone_info.len()) }

        let inverse_transformation = glm::inverse(&model.bone_hierarchy.transformation);
        // let inverse_transformation = unsafe {
        //     glm::inverse(&glm::transpose(&std::mem::transmute::<Matrix4x4, Mat4>(
        //         model.scene.root.as_ref().unwrap().transformation,
        //     )))
        // };
        let anim = &model.scene.animations[self.anim_id];

        self.read_node_hierarchy(
            &inverse_transformation,
            animation_time,
            &model.bone_hierarchy,
            &Mat4::identity(),
            &anim,
            &model.bone_names_index,
            bones,
        );

        // let animations = &x.model.animations;
        // let anim = &animations[0];
        // for bone_keys in &anim.channels { // channel per bone
        // }
        // bones.iter().map(|x| { *x }.into()).collect()
        // })
        // .unwrap()
    }
}
