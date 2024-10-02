use egui::TextureId;
use egui_winit_vulkano::Gui;
use glium::buffer::Content;
use glm::{cross, dot, normalize, radians, vec1, vec2, vec4, Mat4, Quat, Vec2, Vec3, Vec4};
use id::*;
use nalgebra_glm as glm;
use parking_lot::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use puffin_egui::puffin;
use rapier3d::{na::ComplexField, parry::simba::scalar::SupersetOf};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    default,
    sync::{atomic::AtomicI32, Arc},
};
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, DrawIndirectCommand,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    format::{ClearValue, Format},
    image::{
        view::ImageView, AttachmentImage, ImageAccess, ImageDimensions, ImageUsage,
        ImageViewAbstract, SampleCount, StorageImage,
    },
    memory::allocator::MemoryUsage,
    padded::Padded,
    pipeline::{
        graphics::viewport::Viewport, ComputePipeline, GraphicsPipeline, Pipeline,
        PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sampler::{SamplerAddressMode, SamplerCreateInfo, LOD_CLAMP_NONE},
    sync::GpuFuture,
};
use winit::event::VirtualKeyCode;

use crate::{
    editor::inspectable::{Inpsect, Ins},
    engine::{
        input::Input,
        particles::{
            particles::{ParticleDebugPipeline, ParticleRenderPipeline, ParticlesSystem},
            shaders::scs,
        },
        perf::Perf,
        project::asset_manager::{AssetInstance, AssetsManager},
        rendering::{component::ur, debug, model::Skeleton},
        time::Time,
        transform_compute::{cs::Data, TransformCompute},
        world::World,
        world::{
            component::Component,
            transform::{Transform, TransformBuf, TransformData},
            Sys,
        },
        RenderData,
    },
};

use super::{
    component::{RendererData, SharedRendererData},
    debug::DebugSystem,
    lighting::{
        // light_bounding::LightBounding,
        lighting::LightingSystem,
        lighting_compute::{
            lt::{self, tile},
            LightingCompute, NUM_TILES,
        },
    },
    model::{ModelManager, ModelRenderer},
    pipeline::{fs, RenderPipeline},
    texture::{Texture, TextureManager},
    vulkan_manager::VulkanManager,
};

pub struct CameraList {
    pub cameras: Mutex<Vec<(i32, CameraData)>>,
    pub camera_id_gen: AtomicI32,
}

pub static CAMERA_LIST: CameraList = CameraList {
    cameras: Mutex::new(Vec::new()),
    camera_id_gen: AtomicI32::new(0),
};

// #[derive(Clone)]
pub struct CameraDataId {
    id: i32,
}

impl Clone for CameraDataId {
    fn clone(&self) -> Self {
        let cid = CameraDataId {
            id: CAMERA_LIST
                .camera_id_gen
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        };
        let mut list = CAMERA_LIST.cameras.lock();

        let cd = list
            .binary_search_by(|(id, cam)| id.cmp(&self.id))
            .map(|index| {
                let cd = &list[index].1;
                let _cd = CameraData::new(cd.vk.clone(), cd.num_samples);
                _cd
            })
            .unwrap();
        list.push((cid.id, cd));
        list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cid
        // *let cid = CameraDataId::new(vk: , num_samples)
    }
}

impl CameraDataId {
    pub fn new(vk: Arc<VulkanManager>, num_samples: u32) -> CameraDataId {
        let cid = CameraDataId {
            id: CAMERA_LIST
                .camera_id_gen
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        };
        let mut list = CAMERA_LIST.cameras.lock();
        list.push((cid.id, CameraData::new(vk, num_samples)));
        list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        cid
    }
    pub fn get<D>(&self, f: D)
    where
        D: FnOnce(&mut CameraData),
    {
        let mut list = CAMERA_LIST.cameras.lock();
        list.binary_search_by(|(id, cam)| id.cmp(&self.id))
            .map(|index| f(&mut list[index].1));
    }
}

impl Drop for CameraDataId {
    fn drop(&mut self) {
        let mut list = CAMERA_LIST.cameras.lock();
        let index = list.binary_search_by(|(id, cam)| id.cmp(&self.id)).unwrap();
        list.remove(index);
        list.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    }
}

// #[derive(Clone)]
pub struct CameraData {
    pub is_active: bool,
    pub is_visible: bool,
    num_samples: u32,
    rend: RenderPipeline,
    particle_render_pipeline: ParticleRenderPipeline,
    particle_debug_pipeline: ParticleDebugPipeline,
    pub debug: DebugSystem,
    light_debug: Arc<GraphicsPipeline>,
    _render_pass: Arc<RenderPass>,
    viewport: Viewport,
    framebuffer: Arc<Framebuffer>,
    pub texture_id: Option<egui::TextureId>,
    pub image: Arc<dyn ImageAccess>,
    pub view: Arc<dyn ImageViewAbstract>,

    samples: SampleCount,
    vk: Arc<VulkanManager>,
    pub camera_view_data: std::collections::VecDeque<CameraViewData>,
}
#[derive(Clone, Default)]
struct Plane {
    normal: Vec3,
    dist: f32,
}
#[derive(Clone, Default)]
pub struct _Frustum {
    points: [Vec3; 8],
    planes: [Vec4; 6],
}
impl _Frustum {
    fn create(cam: &CameraViewData) -> Self {
        let target = cam.cam_pos + glm::quat_rotate_vec3(&cam.cam_rot, &-Vec3::z());
        let up = glm::quat_rotate_vec3(&cam.cam_rot, &Vec3::y());
        // cvd.inv_rot = glm::inverse(&glm::quat_to_mat4(&rot));
        // cvd.view = glm::quat_to_mat4(&glm::quat_conjugate(&rot)) * glm::translate(&glm::identity(), &-pos);
        let view = glm::look_at_rh(&cam.cam_pos, &target, &up);

        let mut frustum = _Frustum::default();
        let mut ndc_pts: [Vec2; 4] = Default::default(); // corners of tile in ndc
        ndc_pts[0] = vec2(-1., -1.); // lower left
        ndc_pts[1] = vec2(1., -1.); // lower right
        ndc_pts[2] = vec2(1., 1.); // upper right
        ndc_pts[3] = vec2(-1., 1.); // upper left

        let min_depth = 0.0;
        let max_depth = 1.0;
        let mut temp: Vec4;
        let inv_projview = glm::inverse(&(cam.proj * cam.view));
        for i in 0..4 {
            temp = inv_projview * vec4(ndc_pts[i].x, ndc_pts[i].y, min_depth, 1.0);
            frustum.points[i] = temp.xyz() / temp.w;
            temp = inv_projview * vec4(ndc_pts[i].x, ndc_pts[i].y, max_depth, 1.0);
            frustum.points[i + 4] = temp.xyz() / temp.w;
        }

        let mut temp_normal: Vec3;
        for i in 0..4 {
            // bottom...
            // left, top, right, bottom
            // Cax+Cby+Ccz+Cd = 0, planes[i] = (Ca, Cb, Cc, Cd)
            //  temp_normal: normal without normalization
            temp_normal = cross(
                &(frustum.points[i] - &cam.cam_pos),
                &(frustum.points[i + 1] - cam.cam_pos),
            );
            temp_normal = normalize(&temp_normal);
            frustum.planes[i] = vec4(
                temp_normal.x,
                temp_normal.y,
                temp_normal.z,
                dot(&temp_normal, &frustum.points[i]),
            );
        }
        // near plane
        {
            temp_normal = cross(
                &(frustum.points[1] - frustum.points[0]),
                &(frustum.points[3] - frustum.points[0]),
            );
            temp_normal = normalize(&temp_normal);
            frustum.planes[4] = vec4(
                temp_normal.x,
                temp_normal.y,
                temp_normal.z,
                dot(&temp_normal, &frustum.points[0]),
            );
        }
        // far plane
        {
            temp_normal = cross(
                &(frustum.points[7] - frustum.points[4]),
                &(frustum.points[5] - frustum.points[4]),
            );
            temp_normal = normalize(&temp_normal);
            frustum.planes[5] = vec4(
                temp_normal.x,
                temp_normal.y,
                temp_normal.z,
                dot(&temp_normal, &frustum.points[4]),
            );
        }
        frustum
    }
}

impl Into<crate::engine::particles::shaders::scs::Frustum> for _Frustum {
    fn into(self) -> crate::engine::particles::shaders::scs::Frustum {
        // let mut planes: [[f32; 4]; 6] = {

        // }
        scs::Frustum {
            planes: self
                .planes
                .into_iter()
                .map(|p| p.into())
                .collect::<Vec<[f32; 4]>>()
                .try_into()
                .unwrap(),
            points: self
                .points
                .into_iter()
                .map(|p| Padded(p.into()))
                .collect::<Vec<Padded<[f32; 3], 4>>>()
                .try_into()
                .unwrap(),
        }
    }
}
impl Into<debug::gs1::Frustum> for _Frustum {
    fn into(self) -> debug::gs1::Frustum {
        let f: scs::Frustum = self.into();
        unsafe { *(f.to_void_ptr() as *const debug::gs1::Frustum) }
    }
}

#[derive(Clone, Default)]
#[repr(C)]
pub struct CameraViewData {
    pub cam_pos: Vec3,
    pub cam_up: Vec3,
    pub cam_forw: Vec3,
    pub cam_right: Vec3,
    pub cam_rot: Quat,
    pub(crate) view: Mat4,
    pub(crate) proj: Mat4,
    pub inv_rot: Mat4,
    pub frustum: _Frustum,
    pub(crate) dimensions: [f32; 2],
    pub near: f32,
    pub far: f32,
}

pub(crate) static mut MAIN_CAMERA: Option<i32> = None;

pub fn set_main_cam(cam: &Camera) {
    unsafe {
        MAIN_CAMERA = Some(cam.data.as_ref().unwrap().id);
    }
}
#[derive(ID, Serialize, Deserialize)]
#[serde(default)]
pub struct Camera {
    #[serde(skip_serializing, skip_deserializing)]
    pub(crate) data: Option<CameraDataId>,
    pub main_cam: bool,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    use_msaa: bool,
    samples: u32,
    #[serde(skip_serializing, skip_deserializing)]
    rebuild_renderpass: bool,
}

impl Clone for Camera {
    fn clone(&self) -> Self {
        Self {
            data: None,
            main_cam: false,
            fov: self.fov.clone(),
            near: self.near.clone(),
            far: self.far.clone(),
            use_msaa: self.use_msaa.clone(),
            samples: self.samples.clone(),
            rebuild_renderpass: true,
        }
    }
}
impl Default for Camera {
    fn default() -> Self {
        Self {
            data: None,
            main_cam: false,
            fov: 60f32,
            near: 0.1f32,
            far: 100f32,
            use_msaa: false,
            samples: 1,
            rebuild_renderpass: true,
        }
    }
}

impl Component for Camera {
    fn init(&mut self, _transform: &Transform, _id: i32, sys: &Sys) {
        self.data = Some(CameraDataId::new(sys.vk.clone(), self.samples));

        unsafe {
            if self.main_cam {
                MAIN_CAMERA = Some(self.data.as_ref().unwrap().id);
            }
        }
        // self.data = Some(Arc::new(Mutex::new(CameraData::new(
        //     sys.vk.clone(),
        //     self.samples,
        // ))));
    }
    fn inspect(&mut self, _transform: &Transform, _id: i32, ui: &mut egui::Ui, sys: &Sys) {
        Ins(&mut self.fov).inspect("fov", ui, sys);
        Ins(&mut self.near).inspect("near", ui, sys);
        Ins(&mut self.far).inspect("far", ui, sys);
        // Ins(&mut self.use_msaa).inspect("use msaa", ui, sys);
        // if self.use_msaa {
        ui.horizontal(|ui| {
            ui.add(egui::Label::new("samples: "));
            // ui.add(egui::DragValue::new(&mut self.samples).);
            // if ui.add(egui::Button::new(format!("{}", self.samples))).clicked() {
            ui.menu_button(format!("{}", self.samples), |ui| {
                if ui.button("1").clicked() {
                    self.samples = 1;
                    self.rebuild_renderpass = true;
                    ui.close_menu();
                }
                if ui.button("2").clicked() {
                    self.samples = 2;
                    self.rebuild_renderpass = true;
                    ui.close_menu();
                }
                if ui.button("4").clicked() {
                    self.samples = 4;
                    self.rebuild_renderpass = true;
                    ui.close_menu();
                }
                if ui.button("8").clicked() {
                    self.samples = 8;
                    self.rebuild_renderpass = true;
                    ui.close_menu();
                }
                if ui.button("16").clicked() {
                    self.samples = 16;
                    self.rebuild_renderpass = true;
                    ui.close_menu();
                }
            });
            // }
        });

        if ui.checkbox(&mut self.main_cam, "Is Main Camera").changed() {
            unsafe {
                if !self.main_cam {
                    MAIN_CAMERA = None;
                } else {
                    MAIN_CAMERA = Some(self.data.as_ref().unwrap().id);
                }
            }
        }
        // if ui.button("Set as main camera").clicked() {
        //     set_main_cam(&self);
        // }

        // Ins(&mut self.samples).inspect("samples", ui, sys);
    }
}
impl Camera {
    // pub fn get_data(&self) -> Option<Arc<Mutex<CameraData>>> {
    //     self.data.as_ref().map(|data| data.get(f).clone())
    // }
    pub fn _update(&mut self, transform: &Transform) {
        if unsafe {
            MAIN_CAMERA
                .as_ref()
                .and_then(|id| self.data.as_ref().and_then(|my_id| Some(my_id.id != *id)))
                .unwrap_or(false)
        } {
            self.main_cam = false;
        }

        if let Some(cam_data) = &self.data {
            // let mut cvd = CameraViewData::default();
            cam_data.get(|cam_data: &mut CameraData| {
                cam_data.update(
                    transform.get_position(),
                    transform.get_rotation(),
                    self.near,
                    self.far,
                    self.fov,
                    self.rebuild_renderpass,
                    self.samples,
                );
            });
            self.rebuild_renderpass = false;
            // Some(cvd)
        } else {
            // None
        }
    }
}
impl CameraData {
    pub fn update(
        &mut self,
        pos: Vec3,
        rot: Quat,
        near: f32,
        far: f32,
        fov: f32,
        rebuild_pipeline: bool,
        num_samples: u32,
    ) {
        if rebuild_pipeline {
            *self = Self::new(self.vk.clone(), num_samples);
        }
        if !self.is_visible || !self.is_active {
            return;
        }

        let mut cvd = CameraViewData::default();
        cvd.cam_pos = pos;
        cvd.cam_rot = rot;
        // let rot = glm::quat_to_mat3(&cvd.cam_rot);
        let target = cvd.cam_pos + glm::quat_rotate_vec3(&rot, &Vec3::z());
        let up = glm::quat_rotate_vec3(&rot, &Vec3::y());
        // cvd.inv_rot = glm::inverse(&glm::quat_to_mat4(&rot));
        cvd.inv_rot = glm::look_at_rh(
            &glm::Vec3::zeros(),
            &(glm::quat_rotate_vec3(&rot, &Vec3::z())),
            &up,
        );
        // cvd.view = glm::quat_to_mat4(&glm::quat_conjugate(&rot)) * glm::translate(&glm::identity(), &-pos);
        cvd.view = glm::look_at_rh(&cvd.cam_pos, &target, &up);
        // cvd.view = set_view_direction(pos, rot * -Vec3::z(), Vec3::y());
        let aspect_ratio = self.viewport.dimensions[0] / self.viewport.dimensions[1];
        // let x = glm::pers
        cvd.proj = glm::perspective_zo(aspect_ratio, fov.to_radians(), near, far);
        cvd.cam_up = up;
        cvd.cam_forw = glm::quat_rotate_vec3(&cvd.cam_rot, &Vec3::z());
        cvd.cam_right = glm::quat_rotate_vec3(&cvd.cam_rot, &Vec3::x());
        cvd.dimensions = self.viewport.dimensions;
        cvd.near = near;
        cvd.far = far;
        cvd.frustum = _Frustum::create(&cvd);
        // cvd.proj.column_mut(1)[1] *= -1.;
        // cvd
        self.camera_view_data.push_back(cvd);
    }
    pub fn new(vk: Arc<VulkanManager>, num_samples: u32) -> Self {
        let samples = match num_samples {
            2 => SampleCount::Sample2,
            4 => SampleCount::Sample4,
            8 => SampleCount::Sample8,
            16 => SampleCount::Sample16,
            _ => SampleCount::Sample1,
        };
        let use_msaa = num_samples != 1;
        let render_pass = if use_msaa {
            vulkano::single_pass_renderpass!(
                vk.device.clone(),
                attachments: {
                    intermediary: {
                        load: Clear,
                        store: DontCare,
                        format: Format::R8G8B8A8_UNORM,
                        samples: num_samples,
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: Format::D32_SFLOAT,
                        samples: num_samples,
                    },
                    color: {
                        load: DontCare,
                        store: Store,
                        format: Format::R8G8B8A8_UNORM,
                        // Same here, this has to match.
                        samples: 1,
                    },
                },
                pass:
                    { color: [intermediary], depth_stencil: {depth}, resolve: [color] }
            )
            .unwrap()
        } else {
            // vulkano::single_pass_renderpass!(
            //     vk.device.clone(),
            //     attachments: {
            //         color: {
            //             load: DontCare,
            //             store: Store,
            //             format: Format::R8G8B8A8_UNORM,
            //             // Same here, this has to match.
            //             samples: 1,
            //         },
            //         depth: {
            //             load: Clear,
            //             store: DontCare,
            //             format: Format::D32_SFLOAT,
            //             samples: 1,
            //         },
            //         // garbage: {
            //         //     load: DontCare,
            //         //     store: DontCare,
            //         //     format: Format::R8_UNORM,
            //         //     // Same here, this has to match.
            //         //     samples: 1,
            //         // }
            //     },
            //     pass:
            //         { color: [color], depth_stencil: {depth}}
            // )
            // .unwrap()

            vulkano::ordered_passes_renderpass!(
                vk.device.clone(),
                attachments: {
                    color: {
                        load: Clear,
                        store: Store,
                        format: Format::R8G8B8A8_UNORM,
                        samples: 1,
                    },
                    depth: {
                        load: Clear,
                        store: DontCare,
                        format: Format::D32_SFLOAT,
                        samples: 1,
                    }
                },
                passes: [
                    { color: [color], depth_stencil: {depth}, input: [] }
                ]
            )
            .unwrap()
        };
        // let render_pass = ;
        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [1.0, -1.0],
            depth_range: 0.0..1.0,
        };

        let (framebuffer, image, view): (
            Arc<Framebuffer>,
            Arc<dyn ImageAccess>,
            Arc<dyn ImageViewAbstract>,
        ) = if use_msaa {
            window_size_dependent_setup_msaa(
                [1920, 1080],
                render_pass.clone(),
                &mut viewport,
                vk.clone(),
                samples,
            )
        } else {
            window_size_dependent_setup(
                [1024, 1024],
                render_pass.clone(),
                &mut viewport,
                vk.clone(),
            )
        };
        // let subpass = Subpass::from(render_pass, 0).unwrap();
        let rend = RenderPipeline::new(render_pass.clone(), [1920, 1080], 0, vk.clone());

        Self {
            rend,
            particle_render_pipeline: ParticleRenderPipeline::new(vk.clone(), render_pass.clone()),
            particle_debug_pipeline: ParticleDebugPipeline::new(vk.clone(), render_pass.clone()),
            light_debug: LightingCompute::new_pipeline(vk.clone(), render_pass.clone()),
            debug: DebugSystem::new(vk.clone(), render_pass.clone()),
            _render_pass: render_pass,
            viewport,
            framebuffer,
            texture_id: None,
            image,
            view,
            camera_view_data: VecDeque::new(), // swapchain,
            samples,
            // tiles: Mutex::new(vk.buffer_array(NUM_TILES, MemoryUsage::DeviceOnly)),
            vk,
            is_active: true,
            is_visible: false,
            num_samples,
        }
    }
    pub fn resize(&mut self, dimensions: [u32; 2], vk: Arc<VulkanManager>, gui: &mut Gui) {
        if self.framebuffer.extent() != dimensions {
            if self.samples == SampleCount::Sample1 {
                (self.framebuffer, self.image, self.view) = window_size_dependent_setup(
                    dimensions,
                    self._render_pass.clone(),
                    &mut self.viewport,
                    vk,
                )
            } else {
                (self.framebuffer, self.image, self.view) = window_size_dependent_setup_msaa(
                    dimensions,
                    self._render_pass.clone(),
                    &mut self.viewport,
                    vk,
                    self.samples,
                )
            }
            if let Some(tex) = self.texture_id.as_ref() {
                gui.unregister_user_image(*tex);
            }
            self.texture_id = Some(gui.register_user_image_view(
                self.view.clone(),
                SamplerCreateInfo {
                    lod: 0.0..=LOD_CLAMP_NONE,
                    mip_lod_bias: -0.2,
                    address_mode: [SamplerAddressMode::Repeat; 3],
                    ..Default::default()
                },
            ))
        }
    }
    pub fn render(
        &mut self,
        vk: Arc<VulkanManager>,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform_compute: &TransformCompute,
        // lights
        light_len: u32,
        lights: Subbuffer<[lt::light]>,
        visible_lights: Subbuffer<[u32]>,
        visible_lights_count: Subbuffer<u32>,
        light_templates: Subbuffer<[fs::lightTemplate]>,
        // end lights
        particles: Arc<ParticlesSystem>,
        transform_buf: TransformBuf,
        renderer_pipeline: Arc<ComputePipeline>,
        offset_vec: Vec<i32>,
        rm: &mut RwLockWriteGuard<SharedRendererData>,
        rrd: &mut RendererData,
        assets: Arc<AssetsManager>,
        world: &mut World,
        // render_jobs: &Vec<Box<dyn Fn(&mut RenderData) + Send + Sync>>,
        cvd: CameraViewData,
        perf: &Perf,
        light_list: Subbuffer<[u32]>,
        tiles: Subbuffer<[tile]>,
        light_debug: bool,
        particle_debug: bool,
        input: &Input,
        time: &Time,
        skeletal_data: &HashMap<i32, Subbuffer<[[[f32; 4]; 3]]>>,
        // debug: &mut DebugSystem,
    ) -> Option<Arc<dyn ImageAccess>> {
        assets.get_manager2(|model_manager: &ModelManager| {
            assets.get_manager2(|texture_manager: &TextureManager| {
                transform_compute.update_mvp(builder, cvd.view, cvd.proj, transform_buf);

                if !offset_vec.is_empty() {
                    let offsets_buffer = vk.buffer_from_iter(offset_vec);
                    // Buffer::from_iter(&vk.mem_alloc, buffer_usage_all(), false, offset_vec).unwrap();
                    {
                        // per camera
                        puffin::profile_scope!("update renderers: stage 1");
                        // stage 1
                        let uniforms = self.vk.allocate(ur::Data {
                            num_jobs: rrd.transforms_len,
                            stage: 1.into(),
                            view: cvd.view.into(),
                            // _dummy0: Default::default(),
                        });
                        // {
                        //     puffin::profile_scope!("update renderers: stage 1: uniform data");
                        //     let data = ur::Data {
                        //         num_jobs: rd.transforms_len,
                        //         stage: 1.into(),
                        //         view: cvd.view.into(),
                        //         // _dummy0: Default::default(),
                        //     };
                        //     let u = rm.uniform.lock().allocate_sized().unwrap();
                        //     *u.write().unwrap() = data;
                        //     u
                        // };
                        if !rm.indirect.data.is_empty() {
                            rm.indirect_buffer = vk.buffer_from_iter(rm.indirect.data.clone());
                        }
                        let update_renderers_set = {
                            puffin::profile_scope!("update renderers: stage 1: descriptor set");
                            PersistentDescriptorSet::new(
                                &vk.desc_alloc,
                                renderer_pipeline
                                    .layout()
                                    .set_layouts()
                                    .get(0) // 0 is the index of the descriptor set.
                                    .unwrap()
                                    .clone(),
                                [
                                    WriteDescriptorSet::buffer(0, rm.updates_gpu.clone()),
                                    WriteDescriptorSet::buffer(1, rm.transform_ids_gpu.clone()),
                                    WriteDescriptorSet::buffer(2, rm.renderers_gpu.clone()),
                                    WriteDescriptorSet::buffer(3, rm.indirect_buffer.clone()),
                                    WriteDescriptorSet::buffer(
                                        4,
                                        transform_compute.gpu_transforms.clone(),
                                    ),
                                    WriteDescriptorSet::buffer(5, offsets_buffer),
                                    WriteDescriptorSet::buffer(6, uniforms),
                                ],
                            )
                            .unwrap()
                        };
                        {
                            puffin::profile_scope!(
                                "update renderers: stage 1: bind pipeline/dispatch"
                            );
                            builder
                                .bind_pipeline_compute(renderer_pipeline.clone())
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Compute,
                                    renderer_pipeline.layout().clone(),
                                    0, // Bind this descriptor set to index 0.
                                    update_renderers_set,
                                )
                                .dispatch([rrd.transforms_len as u32 / 128 + 1, 1, 1])
                                .unwrap();
                        }
                    }
                }
                let particle_sort = perf.node("particle sort");
                particles.sort.sort(
                    // per camera
                    &cvd,
                    transform_compute.gpu_transforms.clone(),
                    &particles.particle_buffers,
                    vk.device.clone(),
                    vk.queue.clone(),
                    builder,
                    &vk.desc_alloc,
                    &perf,
                    &input,
                );
                drop(particle_sort);
                // light_bounding.render(
                //     builder,
                //     light_ids.clone(),
                //     lights.clone(),
                //     tiles.clone(),
                //     light_draw_indirect.clone(),
                //     cvd.proj * cvd.view,
                // );

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![
                                Some([0.2, 0.25, 1., 1.].into()),
                                Some(1f32.into()),
                                None,
                            ],
                            ..RenderPassBeginInfo::framebuffer(self.framebuffer.clone())
                        },
                        SubpassContents::Inline,
                    )
                    .unwrap();
                builder.set_viewport(0, [self.viewport.clone()]);

                self.rend.bind_pipeline(builder);

                // let mut skeleton = Skeleton::default();
                // skeleton.model.id = model_manager
                //     .assets_id
                //     .iter()
                //     .filter(|x| x.1.lock().model.has_skeleton)
                //     .map(|x| *x.0)
                //     .next()
                //     .unwrap();

                let empty = vk.buffer_from_iter([0]);
                // {
                // let mm = model_manager.lock();
                let mm = model_manager;
                let render_models = perf.node("render models");
                let mut offset = 0;
                let max = rm.renderers_gpu.len();
                for (_ind_id, m_id) in rrd.indirect_model.iter() {
                    if let Some(model_indr) = rrd.model_indirect.get(m_id) {
                        for (i, indr) in model_indr.iter().enumerate() {
                            if indr.id == *_ind_id {
                                if let Some(mr) = mm.assets_id.get(m_id) {
                                    let mr = mr.lock();
                                    if indr.count == 0 {
                                        continue;
                                    }
                                    let indirect_buffer = rm
                                        .indirect_buffer
                                        .clone()
                                        .slice(indr.id as u64..(indr.id + 1) as u64);
                                    let renderer_buffer = rm.renderers_gpu.clone().slice(
                                        offset
                                            ..(offset + indr.count as u64)
                                                .min(rm.renderers_gpu.len()),
                                    );
                                    self.rend.bind_mesh(
                                        &texture_manager,
                                        builder,
                                        vk.desc_alloc.clone(),
                                        renderer_buffer.clone(),
                                        transform_compute.mvp.clone(),
                                        // lights
                                        light_len,
                                        lights.clone(),
                                        light_templates.clone(),
                                        tiles.clone(),
                                        cvd.dimensions,
                                        // end lights
                                        transform_compute.gpu_transforms.clone(),
                                        &mr.model.meshes[i],
                                        indirect_buffer.clone(),
                                        cvd.cam_pos.clone(),
                                        light_list.clone(),
                                        visible_lights.clone(),
                                        visible_lights_count.clone(),
                                        skeletal_data.get(&m_id),
                                        mr.model.has_skeleton,
                                        empty.clone(),
                                        mr.model.bone_info.len() as i32,
                                    );
                                }
                                offset += indr.count as u64;
                                break;
                            }
                        }
                    }
                }
                drop(render_models);
                // }
                let render_jobs_perf = perf.node("render jobs");
                let mut rd = RenderData {
                    builder,
                    // uniforms: Arc::new(Mutex::new(vk.sub_buffer_allocator())),
                    gpu_transforms: transform_compute.gpu_transforms.clone(),
                    light_len,
                    lights: lights.clone(),
                    light_templates: light_templates.clone(),
                    light_list: light_list.clone(),
                    visible_lights: visible_lights.clone(),
                    visible_lights_count: visible_lights_count.clone(),
                    tiles: tiles.clone(),
                    screen_dims: cvd.dimensions,
                    mvp: transform_compute.mvp.clone(),
                    view: &cvd.view,
                    proj: &cvd.proj,
                    pipeline: &self.rend,
                    viewport: &self.viewport,
                    texture_manager: texture_manager,
                    vk: vk.clone(),
                    cam_pos: cvd.cam_pos,
                };
                world.render(&mut rd);
                // for job in render_jobs {
                //     job(&mut rjd);
                // }
                drop(render_jobs_perf);

                static mut CAM_POS: Vec3 = Vec3::new(0.0, 0.0, 0.0);
                static mut CAM_FORW: Vec3 = Vec3::new(0.0, 0.0, 0.0);
                static mut FRUSTUM: debug::gs1::Frustum = debug::gs1::Frustum {
                    planes: [[0., 0., 0., 0.]; 6],
                    points: [Padded([0., 0., 0.]); 8],
                };
                static mut LOCK_FRUSTUM: bool = false;

                if input.get_key_up(&VirtualKeyCode::C) {
                    unsafe {
                        LOCK_FRUSTUM = !LOCK_FRUSTUM;
                    }
                }
                if unsafe { !LOCK_FRUSTUM } {
                    unsafe {
                        FRUSTUM = cvd.frustum.clone().into();
                        CAM_FORW = cvd.cam_forw;
                        CAM_POS = cvd.cam_pos;
                    }
                }
                // self.debug.append_arrow(
                //     glm::vec3(0.0, 1.0, 0.0),
                //     glm::vec3(0.0, 0.0, 3.0),
                //     8.0,
                //     vec4(1.0, 0.0, 0.0, 1.0),
                // );
                // self.debug.append_arrow(
                //     glm::vec3(0.0, 1.0, 0.0),
                //     glm::vec3(5.0, 0.0, 8.0),
                //     12.0,
                //     vec4(1.0, 1.0, 0.0, 1.0),
                // );
                // self.debug
                //     .append_frustum(unsafe { FRUSTUM.clone() }, vec4(0., 1.0, 1.0, 1.0));
                // let point = unsafe { CAM_POS + CAM_FORW * 30.0 };
                // for p in unsafe { &FRUSTUM.planes } {
                //     let dir = normalize(&glm::vec3(p[0], p[1], p[2]));
                //     let orig = dir * -p[3];
                //     let v = point - orig;
                //     let dist = v.x * dir.x + v.y * dir.y + v.z * dir.z;
                //     let point = point - dist * dir;
                //     let point = point - (glm::dot(&dir, &point) - p[3]) * dir;
                //     self.debug
                //         .append_arrow(dir, point, 8.0, vec4(1.0, 1.0, 0.0, 1.0));
                // }
                self.debug.draw(builder, &cvd);

                if (particle_debug) {
                    particles.debug_particles(
                        &self.particle_debug_pipeline,
                        builder,
                        &cvd,
                        transform_compute.gpu_transforms.clone(),
                        //
                        lights.clone(),
                        light_templates.clone(),
                        tiles.clone(),
                    );
                }

                particles.render_particles(
                    &self.particle_render_pipeline,
                    builder,
                    cvd.view,
                    cvd.proj,
                    cvd.inv_rot,
                    cvd.cam_rot.coords.into(),
                    cvd.cam_pos.into(),
                    transform_compute.gpu_transforms.clone(),
                    //
                    lights.clone(),
                    light_templates.clone(),
                    tiles.clone(),
                );

                if (light_debug) {
                    let set = PersistentDescriptorSet::new(
                        &self.vk.desc_alloc,
                        self.light_debug
                            .layout()
                            .set_layouts()
                            .get(0)
                            .unwrap()
                            .clone(),
                        [WriteDescriptorSet::buffer(0, tiles.clone())],
                    )
                    .unwrap();
                    builder
                        .bind_pipeline_graphics(self.light_debug.clone())
                        .bind_descriptor_sets(
                            PipelineBindPoint::Graphics,
                            self.light_debug.layout().clone(),
                            0,
                            set,
                        )
                        .draw(NUM_TILES as u32, 1, 0, 0)
                        .unwrap();
                }
                builder.end_render_pass().unwrap();
                // self.camera_view_data.pop_front();
                Some(self.image.clone())
            })
        })
    }
}
/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup_msaa(
    dimensions: [u32; 2],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
    vk: Arc<VulkanManager>,
    samples: SampleCount,
) -> (
    Arc<Framebuffer>,
    Arc<dyn ImageAccess>,
    Arc<dyn ImageViewAbstract>,
) {
    viewport.dimensions = [dimensions[0] as f32, -(dimensions[1] as f32)];
    viewport.origin[1] = -viewport.dimensions[1];
    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient_multisampled(
            &vk.mem_alloc,
            dimensions,
            samples,
            Format::D32_SFLOAT,
        )
        .unwrap(),
    )
    .unwrap();

    let intermediary = AttachmentImage::transient_multisampled(
        &vk.mem_alloc,
        dimensions,
        samples,
        Format::R8G8B8A8_UNORM,
    )
    .unwrap();
    let intermediary = ImageView::new_default(intermediary.clone()).unwrap();

    let image = StorageImage::new(
        &vk.mem_alloc,
        ImageDimensions::Dim2d {
            width: dimensions[0],
            height: dimensions[1],
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        Some(vk.queue.queue_family_index()),
    )
    .unwrap();
    let view = ImageView::new_default(image.clone()).unwrap();

    let frame_buf = if samples == SampleCount::Sample1 {
        Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![view.clone(), depth_buffer],
                ..Default::default()
            },
        )
        .unwrap()
    } else {
        Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![intermediary, depth_buffer, view.clone()],
                ..Default::default()
            },
        )
        .unwrap()
    };

    (frame_buf, image, view)
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    dimensions: [u32; 2],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
    vk: Arc<VulkanManager>,
) -> (
    Arc<Framebuffer>,
    Arc<dyn ImageAccess>,
    Arc<dyn ImageViewAbstract>,
) {
    viewport.dimensions = [dimensions[0] as f32, -(dimensions[1] as f32)];
    viewport.origin[1] = -viewport.dimensions[1];
    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(&vk.mem_alloc, dimensions, Format::D32_SFLOAT).unwrap(),
    )
    .unwrap();
    // let garbage = ImageView::new_default(
    //     AttachmentImage::transient(&vk.mem_alloc, dimensions, Format::R8_UNORM).unwrap(),
    // )
    // .unwrap();

    let image = AttachmentImage::with_usage(
        &vk.mem_alloc,
        dimensions,
        Format::R8G8B8A8_UNORM,
        ImageUsage::SAMPLED
            | ImageUsage::STORAGE
            | ImageUsage::COLOR_ATTACHMENT
            | ImageUsage::TRANSFER_SRC, // | ImageUsage::INPUT_ATTACHMENT,
    )
    .unwrap();
    let view = ImageView::new_default(image.clone()).unwrap();
    let frame_buf = Framebuffer::new(
        render_pass.clone(),
        FramebufferCreateInfo {
            attachments: vec![view.clone(), depth_buffer.clone()],
            ..Default::default()
        },
    )
    .unwrap();

    (frame_buf, image, view)
    // (fb, images)
}
