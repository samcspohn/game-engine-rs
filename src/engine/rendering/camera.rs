use std::{collections::VecDeque, sync::Arc};

use component_derive::ComponentID;
use egui::TextureId;
use glm::{radians, vec1, Mat4, Quat, Vec3};
use nalgebra_glm as glm;
use parking_lot::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use puffin_egui::puffin;
use rapier3d::na::ComplexField;
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, DrawIndirectCommand,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    format::{ClearValue, Format},
    image::{
        view::ImageView, AttachmentImage, ImageAccess, ImageDimensions, ImageUsage, SampleCount,
        StorageImage,
    },
    memory::allocator::MemoryUsage,
    pipeline::{
        graphics::viewport::Viewport, ComputePipeline, GraphicsPipeline, Pipeline,
        PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sync::GpuFuture,
};

use crate::{
    editor::inspectable::{Inpsect, Ins},
    engine::{
        particles::particles::{ParticleRenderPipeline, ParticlesSystem},
        perf::Perf,
        project::asset_manager::AssetsManager,
        rendering::component::ur,
        transform_compute::{cs::Data, TransformCompute},
        world::{
            component::{Component, _ComponentID},
            transform::{Transform, TransformBuf, TransformData},
            Sys,
        },
        RenderJobData,
    },
};

use super::{
    component::{RendererData, SharedRendererData},
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

// #[derive(Clone)]
pub struct CameraData {
    rend: RenderPipeline,
    particle_render_pipeline: ParticleRenderPipeline,
    light_debug: Arc<GraphicsPipeline>,
    _render_pass: Arc<RenderPass>,
    viewport: Viewport,
    framebuffer: Arc<Framebuffer>,
    pub render_textures_ids: Option<Vec<TextureId>>,
    pub image: Arc<dyn ImageAccess>,

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
struct Frustum {
    top: Plane,
    bottom: Plane,
    right: Plane,
    left: Plane,
    far: Plane,
    near: Plane,
}
impl Frustum {
    fn create(
        cam: &mut CameraViewData,
        far_dist: f32,
        fovY: f32,
        aspect: f32,
        near_dist: f32,
    ) -> Self {
        let frustum = Frustum::default();
        let rot = glm::quat_to_mat3(&cam.cam_rot);
        let h_far = far_dist * (fovY).tan();
        let w_far = h_far * aspect;
        let h_near = near_dist * (fovY).tan();
        let w_near = h_near * aspect;
        let p = cam.cam_pos;
        let d = rot * glm::Vec3::z();
        let right = rot * glm::Vec3::x();
        let up = rot * glm::Vec3::y();
        let fc = p + d * far_dist;
        let ftl = fc + (up * h_far / 2.) - (right * w_far / 2.);
        let ftr = fc + (up * h_far / 2.) + (right * w_far / 2.);
        let fbl = fc - (up * h_far / 2.) - (right * w_far / 2.);
        let fbr = fc - (up * h_far / 2.) + (right * w_far / 2.);

        let nc = p + d * near_dist;

        let ntl = nc + (up * h_near / 2.) - (right * w_near / 2.);
        let ntr = nc + (up * h_near / 2.) + (right * w_near / 2.);
        let nbl = nc - (up * h_near / 2.) - (right * w_near / 2.);
        let nbr = nc - (up * h_near / 2.) + (right * w_near / 2.);
        frustum
        //     let halfVSide = zFar * (fovY * 0.5).tan();
        //     let halfHSide = halfVSide * aspect;
        //     let rot = glm::quat_to_mat3(&cam.cam_rot);
        //     let front = rot * glm::Vec3::z();
        //     let right = rot * glm::Vec3::x();
        //     let up = rot * glm::Vec3::y();

        //     let p = cam.cam_pos;
        //     let d = front;
        // let frontMultFar = zFar * front;
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
    frustum: Frustum,
    pub(crate) dimensions: [f32; 2],
    pub near: f32,
    pub far: f32,
}

#[derive(ComponentID, Clone, Serialize, Deserialize)]
#[repr(C)]
#[serde(default)]
pub struct Camera {
    #[serde(skip_serializing, skip_deserializing)]
    data: Option<Arc<Mutex<CameraData>>>,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    use_msaa: bool,
    samples: u32,
    #[serde(skip_serializing, skip_deserializing)]
    rebuild_renderpass: bool,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            data: None,
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
        self.data = Some(Arc::new(Mutex::new(CameraData::new(
            sys.vk.clone(),
            self.samples,
        ))));
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

        // Ins(&mut self.samples).inspect("samples", ui, sys);
    }
}
impl Camera {
    pub fn get_data(&self) -> Option<Arc<Mutex<CameraData>>> {
        self.data.as_ref().map(|data| data.clone())
    }
    pub fn _update(&mut self, transform: &Transform) -> Option<CameraViewData> {
        if let Some(cam_data) = &self.data {
            let r = Some(cam_data.lock().update(
                transform.get_position(),
                transform.get_rotation(),
                self.near,
                self.far,
                self.fov,
                self.rebuild_renderpass,
                self.samples,
            ));
            self.rebuild_renderpass = false;
            r
        } else {
            None
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
    ) -> CameraViewData {
        if rebuild_pipeline {
            *self = Self::new(self.vk.clone(), num_samples);
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
        // cvd.proj.column_mut(1)[1] *= -1.;
        cvd
        // self.camera_view_data.push_back(cvd);
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
                        format: Format::E5B9G9R9_UFLOAT_PACK32,
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
                        format: Format::E5B9G9R9_UFLOAT_PACK32,
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
                        format: Format::E5B9G9R9_UFLOAT_PACK32,
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

        let (framebuffer, image): (Arc<Framebuffer>, Arc<dyn ImageAccess>) = if use_msaa {
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
            particle_render_pipeline: ParticleRenderPipeline::new(
                vk.clone(),
                render_pass.clone(),
                // use_msaa,
            ),
            light_debug: LightingCompute::new_pipeline(vk.clone(), render_pass.clone()),
            _render_pass: render_pass,
            viewport,
            framebuffer,
            render_textures_ids: None,
            image,
            camera_view_data: VecDeque::new(), // swapchain,
            samples,
            // tiles: Mutex::new(vk.buffer_array(NUM_TILES, MemoryUsage::DeviceOnly)),
            vk,
        }
    }
    pub fn resize(&mut self, dimensions: [u32; 2], vk: Arc<VulkanManager>) {
        if self.framebuffer.extent() != dimensions {
            if self.samples == SampleCount::Sample1 {
                (self.framebuffer, self.image) = window_size_dependent_setup(
                    dimensions,
                    self._render_pass.clone(),
                    &mut self.viewport,
                    vk,
                )
            } else {
                (self.framebuffer, self.image) = window_size_dependent_setup_msaa(
                    dimensions,
                    self._render_pass.clone(),
                    &mut self.viewport,
                    vk,
                    self.samples,
                )
            }
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
        rd: &mut RendererData,
        assets: Arc<AssetsManager>,
        render_jobs: &Vec<Box<dyn Fn(&mut RenderJobData) + Send + Sync>>,
        cvd: CameraViewData,
        perf: &Perf,
        light_list: Subbuffer<[u32]>,
        tiles: Subbuffer<[tile]>,
    ) -> Option<Arc<dyn ImageAccess>> {
        let _model_manager = assets.get_manager::<ModelRenderer>();
        let __model_manager = _model_manager.lock();
        let model_manager: &ModelManager =
            unsafe { __model_manager.as_any().downcast_ref_unchecked() };
        let _texture_manager = assets.get_manager::<Texture>();
        let __texture_manager = _texture_manager.lock();
        let texture_manager: &TextureManager =
            unsafe { __texture_manager.as_any().downcast_ref_unchecked() };
        transform_compute.update_mvp(builder, cvd.view, cvd.proj, transform_buf);

        if !offset_vec.is_empty() {
            let offsets_buffer = vk.buffer_from_iter(offset_vec);
            // Buffer::from_iter(&vk.mem_alloc, buffer_usage_all(), false, offset_vec).unwrap();
            {
                // per camera
                puffin::profile_scope!("update renderers: stage 1");
                // stage 1
                let uniforms = self.vk.allocate(ur::Data {
                    num_jobs: rd.transforms_len,
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
                            WriteDescriptorSet::buffer(4, transform_compute.gpu_transforms.clone()),
                            WriteDescriptorSet::buffer(5, offsets_buffer),
                            WriteDescriptorSet::buffer(6, uniforms),
                        ],
                    )
                    .unwrap()
                };
                {
                    puffin::profile_scope!("update renderers: stage 1: bind pipeline/dispatch");
                    builder
                        .bind_pipeline_compute(renderer_pipeline.clone())
                        .bind_descriptor_sets(
                            PipelineBindPoint::Compute,
                            renderer_pipeline.layout().clone(),
                            0, // Bind this descriptor set to index 0.
                            update_renderers_set,
                        )
                        .dispatch([rd.transforms_len as u32 / 128 + 1, 1, 1])
                        .unwrap();
                }
            }
        }
        let particle_sort = perf.node("particle sort");
        particles.sort.sort(
            // per camera
            cvd.view.into(),
            cvd.proj.into(),
            cvd.cam_pos.into(),
            transform_compute.gpu_transforms.clone(),
            &particles.particle_buffers,
            vk.device.clone(),
            vk.queue.clone(),
            builder,
            &vk.desc_alloc,
            &perf,
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
                    clear_values: vec![Some([0.2, 0.25, 1., 1.].into()), Some(1f32.into()), None],
                    ..RenderPassBeginInfo::framebuffer(self.framebuffer.clone())
                },
                SubpassContents::Inline,
            )
            .unwrap();
        builder.set_viewport(0, [self.viewport.clone()]);

        self.rend.bind_pipeline(builder);

        // {
        // let mm = model_manager.lock();
        let mm = model_manager;
        let render_models = perf.node("render models");
        let mut offset = 0;
        let max = rm.renderers_gpu.len();
        for (_ind_id, m_id) in rd.indirect_model.iter() {
            if let Some(model_indr) = rd.model_indirect.get(m_id) {
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
                                offset..(offset + indr.count as u64).min(rm.renderers_gpu.len()),
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
                                &mr.meshes[i],
                                indirect_buffer.clone(),
                                cvd.cam_pos.clone(),
                                light_list.clone(),
                                visible_lights.clone(),
                                visible_lights_count.clone(),
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
        let mut rjd = RenderJobData {
            builder,
            uniforms: Arc::new(Mutex::new(vk.sub_buffer_allocator())),
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
        for job in render_jobs {
            job(&mut rjd);
        }
        drop(render_jobs_perf);
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

        builder.end_render_pass().unwrap();
        // self.camera_view_data.pop_front();
        Some(self.image.clone())
    }
}
/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup_msaa(
    dimensions: [u32; 2],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
    vk: Arc<VulkanManager>,
    samples: SampleCount,
) -> (Arc<Framebuffer>, Arc<dyn ImageAccess>) {
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
        Format::E5B9G9R9_UFLOAT_PACK32,
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
        Format::E5B9G9R9_UFLOAT_PACK32,
        Some(vk.queue.queue_family_index()),
    )
    .unwrap();
    let view = ImageView::new_default(image.clone()).unwrap();

    let frame_buf = if samples == SampleCount::Sample1 {
        Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![view, depth_buffer],
                ..Default::default()
            },
        )
        .unwrap()
    } else {
        Framebuffer::new(
            render_pass.clone(),
            FramebufferCreateInfo {
                attachments: vec![intermediary, depth_buffer, view],
                ..Default::default()
            },
        )
        .unwrap()
    };

    (frame_buf, image)
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    dimensions: [u32; 2],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
    vk: Arc<VulkanManager>,
) -> (Arc<Framebuffer>, Arc<dyn ImageAccess>) {
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
        Format::E5B9G9R9_UFLOAT_PACK32,
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
            attachments: vec![view, depth_buffer.clone()],
            ..Default::default()
        },
    )
    .unwrap();

    (frame_buf, image)
    // (fb, images)
}
