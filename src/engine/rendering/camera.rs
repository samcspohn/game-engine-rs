use std::{collections::VecDeque, sync::Arc};

use component_derive::ComponentID;
use glm::{radians, vec1, Mat4, Quat, Vec3};
use nalgebra_glm as glm;
use parking_lot::{Mutex, MutexGuard, RwLockWriteGuard, RwLock, RwLockReadGuard};
use puffin_egui::puffin;
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{BufferSlice, CpuAccessibleBuffer, CpuBufferPool},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    format::{ClearValue, Format},
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage},
    pipeline::{graphics::viewport::Viewport, ComputePipeline, Pipeline, PipelineBindPoint},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    sync::GpuFuture,
};

use crate::{
    editor::inspectable::{Inpsect, Ins, Inspectable},
    engine::{
        particles::particles::{ParticleCompute, ParticleRenderPipeline},
        rendering::renderer_component::ur,
        transform_compute::{cs::ty::Data, TransformCompute},
        world::{
            component::{Component, _ComponentID},
            transform::{Transform, TransformData},
            Sys,
        },
        RenderJobData, project::asset_manager::AssetsManager,
    },
};

use super::{
    model::{ModelManager, ModelRenderer},
    pipeline::RenderPipeline,
    renderer_component::{buffer_usage_all, RendererData, SharedRendererData},
    texture::{TextureManager, Texture},
    vulkan_manager::VulkanManager,
};

// #[derive(Clone)]
pub struct CameraData {
    rend: RenderPipeline,
    particle_render_pipeline: ParticleRenderPipeline,
    // surface: Arc<Surface>,
    _render_pass: Arc<RenderPass>,
    viewport: Viewport,
    // swapchain: Arc<Swapchain>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pub output: Vec<Arc<AttachmentImage>>,
    pub camera_view_data: std::collections::VecDeque<CameraViewData>,
    // recreate_swapchain: bool,
    // previous_frame_end: Option<Box<dyn GpuFuture>>,
}
#[derive(Clone, Default)]
pub struct CameraViewData {
    cam_pos: Vec3,
    cam_rot: Quat,
    view: Mat4,
    proj: Mat4,
}

#[derive(ComponentID, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct Camera {
    #[serde(skip_serializing, skip_deserializing)]
    data: Option<Arc<Mutex<CameraData>>>,
    fov: f32,
    near: f32,
    far: f32,
}

// impl ComponentID for Camera {
//     // const ID: u64 = std::any::type_name::<Self>().hash();
//         const ID: u64 = const_fnv1a_hash::fnv1a_hash_64("Camera".as_bytes(), None);
//     // fn hello_world() {
//     //     println!("Hello, World! My name is {}", stringify!(#name));
//     // }
// }

impl Default for Camera {
    fn default() -> Self {
        Self {
            data: None,
            fov: 60f32,
            near: 0.1f32,
            far: 100f32,
        }
    }
}
impl Inspectable for Camera {
    fn inspect(&mut self, _transform: &Transform, _id: i32, ui: &mut egui::Ui, sys: &Sys) {
        Ins(&mut self.fov).inspect("fov", ui, sys);
        Ins(&mut self.near).inspect("near", ui, sys);
        Ins(&mut self.far).inspect("far", ui, sys);
    }
}
impl Component for Camera {
    fn init(&mut self, _transform: &Transform, _id: i32, sys: &Sys) {
        self.data = Some(Arc::new(Mutex::new(CameraData::new(sys.vk.clone()))));
    }
}
impl Camera {
    pub fn get_data(&self) -> Option<Arc<Mutex<CameraData>>> {
        self.data.as_ref().map(|data| data.clone())
    }
    pub fn _update(&mut self, transform: &Transform) {
        if let Some(cam_data) = &self.data {
            cam_data.lock().update(
                transform.get_position(),
                transform.get_rotation(),
                self.near,
                self.far,
                self.fov,
            )
        }
    }
}
impl CameraData {
    pub fn update(&mut self, pos: Vec3, rot: Quat, near: f32, far: f32, fov: f32) {
        // let cvd = &mut self.camera_view_data;
        let mut cvd = CameraViewData::default();
        cvd.cam_pos = pos;
        cvd.cam_rot = rot;
        let rot = glm::quat_to_mat3(&cvd.cam_rot);
        let target = cvd.cam_pos + rot * Vec3::z();
        let up = rot * Vec3::y();
        cvd.view = glm::look_at_lh(&cvd.cam_pos, &target, &up);
        let aspect_ratio = self.viewport.dimensions[0] / self.viewport.dimensions[1];
        cvd.proj = glm::perspective(aspect_ratio, radians(&vec1(fov)).x, near, far);
        self.camera_view_data.push_back(cvd);
    }
    pub fn new(vk: Arc<VulkanManager>) -> Self {
        let render_pass = vulkano::ordered_passes_renderpass!(
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
        .unwrap();
        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.0..1.0,
        };

        let (framebuffers, output) = window_size_dependent_setup(
            [1920, 1080],
            render_pass.clone(),
            &mut viewport,
            vk.clone(),
        );

        let rend = RenderPipeline::new(
            vk.device.clone(),
            render_pass.clone(),
            [1920, 1080],
            vk.queue.clone(),
            0,
            vk.mem_alloc.clone(),
            &vk.comm_alloc,
        );
        Self {
            // surface,
            rend,
            particle_render_pipeline: ParticleRenderPipeline::new(vk, render_pass.clone()),
            _render_pass: render_pass,
            viewport,
            framebuffers,
            output,
            camera_view_data: VecDeque::new(), // swapchain,
        }
    }
    pub fn resize(&mut self, dimensions: [u32; 2], vk: Arc<VulkanManager>) {
        (self.framebuffers, self.output) = window_size_dependent_setup(
            dimensions,
            self._render_pass.clone(),
            &mut self.viewport,
            vk,
        )
    }
    pub fn render(
        &mut self,
        vk: Arc<VulkanManager>,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform_compute: &mut TransformCompute,
        particles: Arc<ParticleCompute>,
        transform_data: &TransformData,
        renderer_pipeline: Arc<ComputePipeline>,
        offset_vec: Vec<i32>,
        rm: &mut RwLockWriteGuard<SharedRendererData>,
        rd: &mut RendererData,
        image_num: u32,
        // model_manager: &MutexGuard<ModelManager>,
        // texture_manager: &Mutex<TextureManager>,
        assets: Arc<AssetsManager>,
        render_jobs: &Vec<Box<dyn Fn(&mut RenderJobData)>>,
    ) {
        let cvd = self.camera_view_data.front();
        if cvd.is_none() {
            return;
        }
        let cvd = cvd.unwrap();
        let _model_manager = assets.get_manager::<ModelRenderer>();
        let __model_manager = _model_manager.lock();
        let model_manager: &ModelManager = __model_manager.as_any().downcast_ref().unwrap();
        let _texture_manager = assets.get_manager::<Texture>();
        let __texture_manager = _texture_manager.lock();
        let texture_manager: &TextureManager = __texture_manager.as_any().downcast_ref().unwrap();
        transform_compute.update_mvp(builder, cvd.view, cvd.proj, transform_data.extent as i32);

        if !offset_vec.is_empty() {
            let offsets_buffer = CpuAccessibleBuffer::from_iter(
                &vk.mem_alloc,
                buffer_usage_all(),
                false,
                offset_vec,
            )
            .unwrap();
            {
                // per camera
                puffin::profile_scope!("update renderers: stage 1");
                // stage 1
                let uniforms = {
                    puffin::profile_scope!("update renderers: stage 1: uniform data");
                    rm.uniform
                        .from_data(ur::ty::Data {
                            num_jobs: rd.transforms_len,
                            stage: 1,
                            view: cvd.view.into(),
                            _dummy0: Default::default(),
                        })
                        .unwrap()
                };
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
        particles.sort.sort(
            // per camera
            cvd.view.into(),
            cvd.proj.into(),
            transform_compute.gpu_transforms.clone(),
            &particles.particle_buffers,
            vk.device.clone(),
            vk.queue.clone(),
            builder,
            &vk.desc_alloc,
        );
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.2, 0.25, 1., 1.].into()), Some(1f32.into())],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[image_num as usize].clone(),
                    )
                },
                SubpassContents::Inline,
            )
            .unwrap()
            .set_viewport(0, [self.viewport.clone()]);

        self.rend.bind_pipeline(builder);

        {
            // let mm = model_manager.lock();
            let mm = model_manager;

            let mut offset = 0;

            for (_ind_id, m_id) in rd.indirect_model.iter() {
                if let Some(indr) = rd.model_indirect.get(m_id) {
                    if let Some(mr) = mm.assets_id.get(m_id) {
                        let mr = mr.lock();
                        if indr.count == 0 {
                            continue;
                        }
                        if let Some(indirect_buffer) =
                            BufferSlice::from_typed_buffer_access(rm.indirect_buffer.clone())
                                .slice(indr.id as u64..(indr.id + 1) as u64)
                        {
                            // println!("{}",indirect_buffer.len());
                            if let Some(renderer_buffer) =
                                BufferSlice::from_typed_buffer_access(rm.renderers_gpu.clone())
                                    .slice(offset..(offset + indr.count as u64))
                            {
                                // println!("{}",renderer_buffer.len());
                                self.rend.bind_mesh(
                                    &texture_manager,
                                    builder,
                                    vk.desc_alloc.clone(),
                                    renderer_buffer.clone(),
                                    transform_compute.mvp.clone(),
                                    &mr.mesh,
                                    indirect_buffer.clone(),
                                );
                            }
                        }
                    }
                    offset += indr.count as u64;
                }
            }
        }
        let mut rjd = RenderJobData {
            builder,
            gpu_transforms: transform_compute.gpu_transforms.clone(),
            mvp: transform_compute.mvp.clone(),
            view: &cvd.view,
            proj: &cvd.proj,
            pipeline: &self.rend,
            viewport: &self.viewport,
            texture_manager: texture_manager,
            vk: vk.clone(),
        };
        for job in render_jobs {
            job(&mut rjd);
        }
        particles.render_particles(
            &self.particle_render_pipeline,
            builder,
            cvd.view,
            cvd.proj,
            cvd.cam_rot.coords.into(),
            cvd.cam_pos.into(),
            transform_compute.gpu_transforms.clone(),
            vk.mem_alloc.clone(),
            &vk.comm_alloc,
            vk.desc_alloc.clone(),
        );
        builder.end_render_pass().unwrap();
        self.camera_view_data.pop_front();
        // self.output[image_num as usize].clone()
    }
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    dimensions: [u32; 2],
    // images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
    vk: Arc<VulkanManager>,
    // color: &mut FrameImage,
) -> (Vec<Arc<Framebuffer>>, Vec<Arc<AttachmentImage>>) {
    // let dimensions = [1920, 1080]; //images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(&vk.mem_alloc, dimensions, Format::D32_SFLOAT).unwrap(),
    )
    .unwrap();

    // color.arc = AttachmentImage::with_usage(
    //     &mem,
    //     dimensions,
    //     Format::R8G8B8A8_UNORM,
    //     ImageUsage {
    //         transfer_src: true,
    //         transfer_dst: false,
    //         sampled: false,
    //         storage: true,
    //         color_attachment: true,
    //         depth_stencil_attachment: false,
    //         transient_attachment: false,
    //         input_attachment: true,
    //         ..ImageUsage::empty()
    //     },
    // )
    // .unwrap();
    let mut images = Vec::new();
    let fb = vk
        .images
        .iter()
        .map(|_| {
            let image = AttachmentImage::with_usage(
                &vk.mem_alloc,
                dimensions,
                Format::R8G8B8A8_UNORM,
                ImageUsage {
                    transfer_src: false,
                    transfer_dst: false,
                    sampled: true,
                    storage: true,
                    color_attachment: true,
                    depth_stencil_attachment: false,
                    transient_attachment: false,
                    input_attachment: true,
                    ..ImageUsage::empty()
                },
            )
            .unwrap();
            images.push(image.clone());
            let view = ImageView::new_default(image).unwrap();
            // let view = ImageView::new_default(image.clone()).unwrap();
            // let color_view = ImageView::new_default(color.arc.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    (fb, images)
}
