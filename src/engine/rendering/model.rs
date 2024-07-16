// extern crate assimp;

use std::{
    borrow::Borrow,
    collections::{BTreeMap, HashMap},
    ops::Sub,
    rc::Rc,
    sync::Arc,
};

// use ai::import::Importer;
// use assimp as ai;
// use bytemuck::{Pod, Zeroable};
use crate::engine::{
    particles::shaders::cs::b,
    prelude::{Component, Inpsect, Ins, _ComponentID},
    project::asset_manager::AssetInstance,
};
use component_derive::{AssetID, ComponentID};
use glium::buffer::Content;
use glm::{float_bits_to_int, IVec2};
use parking_lot::{Mutex, RwLock};

// use std::mem::size_of;
use nalgebra_glm::{self as glm, quat, vec3, Mat4, Quat, Vec3};
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
    buffer::{Buffer, BufferContents, Subbuffer},
    memory::allocator::{MemoryAllocator, MemoryUsage, StandardMemoryAllocator},
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
    pub indeces: Vec<u32>,
    // pub bone_ids: Vec<smallvec::SmallVec<[u16; 4]>>,
    pub vertex_bones: Vec<IVec2>,
    pub bone_weight_offsets: Vec<u32>,

    pub vertex_buffer: Subbuffer<[_Vertex]>,
    pub uvs_buffer: Subbuffer<[UV]>,
    pub index_buffer: Subbuffer<[u32]>,
    pub normals_buffer: Subbuffer<[Normal]>,
    pub bone_weights_offsets_buf: Option<Subbuffer<[u32]>>,
    pub bone_weights_counts_buf: Option<Subbuffer<[u32]>>,
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
    pub offset_matrix: Matrix4x4,
}
pub struct Model {
    pub meshes: Vec<Mesh>,
    // pub animations: Vec<Animation>,
    pub scene: Arc<Scene>,
    pub bone_names_index: HashMap<String, (u32, _Bone)>,
    pub bone_info: Vec<Mat4>,
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
                                offset_matrix: bone.offset_matrix.clone(),
                            },
                        )
                    })
                    .into_iter()
            })
            .flatten()
            .enumerate()
            .map(|(id, (b_name, b_))| (b_name, (id as u32, b_)))
            .collect::<HashMap<String, (u32, _Bone)>>();

        let bone_info: Vec<Mat4> = bone_names_index
            .iter()
            .map(|(b_name, (id, b_))| unsafe { std::mem::transmute(b_.offset_matrix) })
            .collect();

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
        // scene.root;

        // let mut _anims = Vec::new();
        // for anim in scene.animations.iter() {

        // }

        Model {
            meshes: _meshes,
            // animations: scene.animations.clone(),
            bone_info,
            bone_names_index,
            scene: Arc::new(scene),
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
        // let _path = std::path::Path::new(path);
        // let model = russimp::scene::Scene::from_file(
        //     path,
        //     vec![
        //         PostProcess::CalculateTangentSpace,
        //         PostProcess::Triangulate,
        //         PostProcess::JoinIdenticalVertices,
        //         PostProcess::SortByPrimitiveType,
        //     ],
        // );
        // // println!("model stats: {:?}", model);
        // // let model = tobj::load_obj(path, &(tobj::GPU_LOAD_OPTIONS));
        // let (scene) = model.expect(format!("Failed to load OBJ file: {}", path).as_str());
        // let mut _meshes = Vec::new();
        // println!("here");

        // for mesh in scene.meshes.iter() {
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
        for v in mesh.vertices.iter() {
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

        for (id, bone) in mesh.bones.iter().enumerate() {
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
        let mut vert_weight = vertex_weights.iter();
        let mut vert = vert_weight.next();
        let mut bone_weight_offsets = Vec::new();
        for i in 0..vertices.len() as u32 {
            bone_weight_offsets.push(offset);
            if let Some(vw) = vert {
                if *vw.0 == i {
                    offset += vw.1.len() as u32;
                    vert = vert_weight.next();
                }
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
        let bone_weights_offsets_buffer = if bone_weight_offsets.len() == 0 {
            None
        } else {
            Some(vk.buffer_from_iter(bone_weight_offsets.clone()))
        };
        let bone_weights_counts_buf = if vertex_weights.len() == 0 {
            None
        } else {
            Some(vk.buffer_from_iter(vertex_weights.iter().map(|(vert,weights)| weights.len() as u32)))
        };

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

        // }

        return Some(Mesh {
            vertices,
            uvs,
            indeces: indices,
            normals,
            vertex_bones,
            bone_weight_offsets,
            vertex_buffer,
            uvs_buffer,
            normals_buffer,
            index_buffer,
            texture,
            bone_weights_buffer,
            bone_weights_counts_buf,
            bone_weights_offsets_buf: bone_weights_offsets_buffer,
        });
        // }

        // _meshes
    }
}

use crate::engine::project::asset_manager::_AssetID;

use super::{component::buffer_usage_all, texture::TextureManager};
#[derive(AssetID)]
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
    fn inspect(&mut self, ui: &mut egui::Ui, _world: &mut World) {
        ui.add(egui::Label::new(self.file.as_str()));
    }
}

pub type ModelManager =
    asset_manager::AssetManager<(Arc<Mutex<TextureManager>>, Arc<VulkanManager>), ModelRenderer>;

#[derive(ComponentID, Default, Clone, Serialize, Deserialize)]
pub struct Skeleton {
    pub model: AssetInstance<ModelRenderer>,
    // pub bones: Vec<Mat4>,
    // pub bone_info: Vec<BoneInfo>,
}

impl Component for Skeleton {
    fn inspect(
        &mut self,
        transform: &crate::engine::prelude::Transform,
        id: i32,
        ui: &mut egui::Ui,
        sys: &crate::engine::prelude::Sys,
    ) {
        Ins(&mut self.model).inspect("model", ui, sys);
    }
}

fn calc_interpolated_vector(t: &Vec<VectorKey>, time: f64) -> Vec3 {
    // let mut scaling = Vec3::new(0,0,0);
    if t.len() == 1 {
        let v = t[0].value;
        vec3(v.x, v.y, v.z)
    } else {
        t.iter()
            .as_slice()
            .windows(2)
            .filter(|x| time > x[0].time && time < x[1].time)
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
fn calc_interpolated_quat(t: &Vec<QuatKey>, time: f64) -> Quat {
    if t.len() == 1 {
        let v = t[0].value;
        quat(v.x, v.y, v.z, v.w)
    } else {
        t.iter()
            .as_slice()
            .windows(2)
            .filter(|x| time > x[0].time && time < x[1].time)
            .map(|x| {
                let t1 = x[0].time;
                let t2 = x[1].time;
                let delta_time = t2 - t1;
                let factor = (time - t1) / delta_time;
                let start = x[0].value;
                let start = quat(start.x, start.y, start.z, start.w);
                let end = x[1].value;
                let end = quat(end.x, end.y, end.z, end.w);
                start.lerp(&end, factor as f32)
                // let delta = end - start;
                // start + factor as f32 * delta
                // glm::scale(&scaling, out)
            })
            .collect::<Vec<Quat>>()[0]
    }
}
impl Skeleton {
    fn read_node_hierarchy(
        &mut self,
        time: f64,
        node: &Node,
        parent_transform: &Mat4,
        scene: &Scene,
        bone_names_index: &HashMap<String, (u32, _Bone)>,
        bone_info: &mut Vec<Mat4>,
    ) {
        let name = node.name.clone();
        let anim = &scene.animations[0];
        let mut node_transform: nalgebra_glm::Mat4 =
            unsafe { std::mem::transmute(node.transformation) };

        let mut node_anim = None;
        // find node anim
        for channel in &anim.channels {
            if channel.name == name {
                node_anim = Some(channel);
            }
        }
        if let Some(node_anim) = node_anim {
            let scl = calc_interpolated_vector(&node_anim.scaling_keys, time);
            let scaling = glm::scaling(&scl);

            let pos = calc_interpolated_vector(&node_anim.position_keys, time);
            let translation = glm::translation(&pos);

            let rot = calc_interpolated_quat(&node_anim.rotation_keys, time);
            let rotation = glm::quat_to_mat4(&rot);

            node_transform = translation * rotation * scaling;
        }

        let global_transform = parent_transform * node_transform;

        if let Some((bone_index, bone_)) = bone_names_index.get(&name) {
            bone_info[*bone_index as usize] = unsafe {
                std::mem::transmute::<Matrix4x4, Mat4>(scene.root.as_ref().unwrap().transformation)
            } * global_transform
                * unsafe { std::mem::transmute::<Matrix4x4, Mat4>(bone_.offset_matrix) };
        }
        for child in node.children.borrow().iter() {
            self.read_node_hierarchy(
                time,
                &child,
                &global_transform,
                scene,
                bone_names_index,
                bone_info,
            );
        }
    }
    pub fn get_skeleton(&mut self, model_manager: &ModelManager, time: f64) -> Vec<[[f32; 4]; 4]> {
        model_manager
            .assets_id
            .get(&self.model.id)
            .and_then(|x| Some(x.lock()))
            .and_then(|x| {
                let root = x.model.scene.root.as_ref().unwrap().clone();

                let mut bones = Vec::with_capacity(x.model.bone_info.len());
                unsafe { bones.set_len(x.model.bone_info.len()) }

                self.read_node_hierarchy(
                    time,
                    &root,
                    &Mat4::identity(),
                    &x.model.scene,
                    &x.model.bone_names_index,
                    &mut bones,
                );

                // let animations = &x.model.animations;
                // let anim = &animations[0];
                // for bone_keys in &anim.channels { // channel per bone
                // }
                Some(bones.iter().map(|x| { *x }.into()).collect())
            })
            .unwrap()
    }
}
