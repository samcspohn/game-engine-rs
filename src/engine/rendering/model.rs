// extern crate assimp;

use std::{
    borrow::Borrow,
    collections::{BTreeMap, HashMap},
    mem::transmute,
    ops::Sub,
    rc::Rc,
    sync::Arc, u16::MAX,
};

// use ai::import::Importer;
// use assimp as ai;
// use bytemuck::{Pod, Zeroable};
use crate::engine::{
    prelude::{Component, Inpsect, Ins},
    project::asset_manager::AssetInstance,
};
use force_send_sync::SendSync;
use glium::buffer::Content;
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
use vulkano::{
    buffer::{subbuffer::BufferWriteGuard, Buffer, BufferContents, Subbuffer},
    memory::allocator::{MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::graphics::vertex_input::Vertex,
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
    pub vertices: Vec<_Vertex>, // TODO: change to [f32;3]
    pub normals: Vec<Normal>,
    pub uvs: Vec<UV>,
    pub indices: Vec<u32>,
    // pub bone_ids: Vec<smallvec::SmallVec<[u16; 4]>>,
    pub vertex_bones: Vec<IVec2>,
    pub bone_weight_offsets: Vec<u32>,
    pub aabb: (Vec3, Vec3),

    pub vertex_buffer: Subbuffer<[_Vertex]>,
    pub uvs_buffer: Subbuffer<[UV]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub normals_buffer: Subbuffer<[Normal]>,
    pub bone_weights_offsets_counts_buf: Subbuffer<[[u32; 2]]>,
    // pub bone_weights_counts_buf: Subbuffer<[u32]>,
    pub bone_weights_buffer: Option<Subbuffer<[[i32; 2]]>>,
    pub texture: Option<i32>,
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
        for mesh in scene.meshes.iter() {
            if let Some(mesh) = Mesh::load_mesh(
                &mesh,
                &scene,
                &bone_names_index,
                texture_manager.clone(),
                &_path,
                vk,
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
    ) -> Option<Mesh> {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut normals = Vec::new();
        let mut uvs = Vec::new();
        let mut vertex_bones = Vec::new();

        // let mesh = &m.mesh;
        let mut skip = false;
        for face in mesh.faces.iter() {
            if face.0.len() < 3 {
                skip = true;
                break;
            }
            indices.push(face.0[0]);
            indices.push(face.0[2]);
            indices.push(face.0[1]);
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
            
            vertices.push(_Vertex {
                position: [v.x, v.y, v.z],
            });
        }

        for n in mesh.normals.iter() {
            normals.push(Normal {
                normal: [n.x, n.y, n.z],
            });
        }

        for tex_coords in mesh.texture_coords.iter() {
            if let Some(tex_coords) = tex_coords {
                for uv in tex_coords {
                    uvs.push(UV { uv: [uv.x, uv.y] }); // TODO: support multiple uvs
                }
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
        for i in 0..vertices.len() as u32 {
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

        let vertex_buffer = vk.buffer_from_iter(vertices.clone());
        let uvs_buffer = vk.buffer_from_iter(uvs.clone());
        let normals_buffer = vk.buffer_from_iter(normals.clone());
        let index_buffer = vk.buffer_from_iter(indices.clone());
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
            for prop in &mat.properties {
                // println!("prop.key: {}, prop.data: {:?}", prop.key, prop.data);
                if prop.semantic == TextureType::Diffuse {
                    let a = || {
                        println!("prop: {:?}", prop);
                    };
                    match &prop.data {
                        russimp::material::PropertyTypeInfo::Buffer(_) => a(),
                        russimp::material::PropertyTypeInfo::IntegerArray(_) => a(),
                        russimp::material::PropertyTypeInfo::FloatArray(_) => a(),
                        russimp::material::PropertyTypeInfo::String(s) => {
                            let diff_path: &str =
                                &(_path.parent().unwrap().to_str().unwrap().to_string() + "/" + &s);
                            println!("diffuse path: {}", diff_path);
                            texture = Some(texture_manager.lock().from_file(diff_path));
                            println!("{}, {:?}", diff_path, texture);
                        }
                    }
                    // prop.semantic == "Diffuse"
                }
            }
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
        let aabb = (
            min,
            max,
        );
        println!("aabb: {:?}", aabb);
        // }

        return Some(Mesh {
            vertices,
            uvs,
            indices,
            normals,
            vertex_bones,
            bone_weight_offsets,
            vertex_buffer,
            uvs_buffer,
            normals_buffer,
            index_buffer,
            texture,
            bone_weights_buffer,
            bone_weights_offsets_counts_buf: bone_weights_offsets_counts_buffer,
            aabb,
        });
        // }

        // _meshes
    }
}

use super::{component::buffer_usage_all, texture::TextureManager};
#[derive(ID)]
pub struct ModelRenderer {
    pub file: String,
    pub model: Model,
    pub count: u32,
}

impl Asset<ModelRenderer, (Arc<Mutex<TextureManager>>, Arc<VulkanManager>)> for ModelRenderer {
    fn from_file(
        file: &str,
        params: &(Arc<Mutex<TextureManager>>, Arc<VulkanManager>),
    ) -> ModelRenderer {
        let model = Model::load_model(file, params.0.clone(), &params.1);
        ModelRenderer {
            file: file.into(),
            model,
            count: 1,
        }
    }

    fn reload(&mut self, file: &str, params: &(Arc<Mutex<TextureManager>>, Arc<VulkanManager>)) {
        let _mesh = Model::load_model(file, params.0.clone(), &params.1);
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

pub type ModelManager =
    asset_manager::AssetManager<(Arc<Mutex<TextureManager>>, Arc<VulkanManager>), ModelRenderer>;

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
            .collect::<Vec<Quat>>().pop().unwrap_or(quat(0., 0., 0., 1.))
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
