use std::sync::Arc;

use glm::{Mat4, Quat, Vec3, radians, Vec1, vec1};
use nalgebra_glm as glm;
use parking_lot::{Mutex, MutexGuard, RwLockWriteGuard};
use puffin_egui::puffin;
use vulkano::{
    buffer::{BufferSlice, CpuAccessibleBuffer, CpuBufferPool},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassContents,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::Device,
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, SwapchainImage},
    memory::allocator::StandardMemoryAllocator,
    pipeline::{graphics::viewport::Viewport, ComputePipeline, Pipeline, PipelineBindPoint},
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{
        acquire_next_image, AcquireError, Surface, Swapchain, SwapchainCreateInfo,
        SwapchainCreationError,
    },
    sync::GpuFuture,
};
use winit::window::Window;

use crate::{
    engine::{transform::Transform, Component, RenderJobData, System},
    model::ModelManager,
    particles::{ParticleCompute, ParticleRenderPipeline},
    renderer::RenderPipeline,
    renderer_component2::{buffer_usage_all, ur, RendererData, SharedRendererData},
    texture::TextureManager,
    transform_compute::{cs::ty::Data, TransformCompute},
    vulkan_manager::VulkanManager,
    FrameImage,
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
    // recreate_swapchain: bool,
    // previous_frame_end: Option<Box<dyn GpuFuture>>,
    cam_pos: Vec3,
    cam_rot: Quat,
    view: Mat4,
    proj: Mat4,
}
#[derive(Clone)]
pub struct Camera {
    data: Arc<Mutex<CameraData>>,
    fov: f32,
    near: f32,
    far: f32,
}
impl Component for Camera {
    fn update(&mut self, transform: Transform, sys: &System) {
        self.data.lock().update(
            transform.get_position(),
            transform.get_rotation(),
            self.near,
            self.far,
            self.fov,
        )
        // let mut data = self.data.lock();
        // data.cam_pos = transform.get_position();
        // data.cam_rot = transform.get_rotation();
        // let rot = glm::quat_to_mat3(&data.cam_rot);
        // let target = data.cam_pos + rot * Vec3::z();
        // let up = rot * Vec3::y();
        // let view: Mat4 = glm::look_at_lh(&data.cam_pos, &target, &up);
        // data.view = view;
    }
}
impl Camera {
    // pub fn get_data(&self) -> Arc<CameraData> {
    //     self.data.clone()
    // }
}
impl CameraData {
    pub fn update(&mut self, pos: Vec3, rot: Quat, near: f32, far: f32, fov: f32) {
        self.cam_pos = pos;
        self.cam_rot = rot;
        let rot = glm::quat_to_mat3(&self.cam_rot);
        let target = self.cam_pos + rot * Vec3::z();
        let up = rot * Vec3::y();
        self.view = glm::look_at_lh(&self.cam_pos, &target, &up);
        let aspect_ratio = self.viewport.dimensions[0] / self.viewport.dimensions[1];
        self.proj = glm::perspective(
            aspect_ratio,
            radians(&vec1(fov)).x,
            near,
            far,
        );
    }
    pub fn new(vk: Arc<VulkanManager>) -> Self {
        let render_pass = vulkano::ordered_passes_renderpass!(
            vk.device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: DontCare,
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
            particle_render_pipeline: ParticleRenderPipeline::new(&vk, render_pass.clone()),
            _render_pass: render_pass,
            viewport,
            framebuffers,
            output,
            // previous_frame_end: Some(()),
            cam_pos: Vec3::new(0., 0., 0.),
            cam_rot: Quat::new(1., 0., 0., 0.),
            view: Mat4::identity(),
            proj: Mat4::identity(),
            // swapchain,
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
        transform_uniforms: &CpuBufferPool<Data>,
        transform_data: Arc<(
            usize,
            Vec<Arc<(Vec<Vec<i32>>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>)>>,
        )>,
        compute_pipeline: Arc<ComputePipeline>,
        renderer_pipeline: Arc<ComputePipeline>,
        offset_vec: Vec<i32>,
        rm: &mut RwLockWriteGuard<SharedRendererData>,
        rd: &mut RendererData,
        image_num: u32,
        model_manager: &MutexGuard<ModelManager>,
        texture_manager: &Mutex<TextureManager>,
        render_jobs: &Vec<Box<dyn Fn(&mut RenderJobData)>>,
    ) -> Arc<AttachmentImage> {
        // let a = Mutex::new(());
        // let b = a.lock();
        // let aspect_ratio = self.viewport.dimensions[0] / self.viewport.dimensions[1]; // *editor_ui::EDITOR_ASPECT_RATIO.lock();
        //                                                                               // vk.swapchain.lock().image_extent()[0] as f32 / vk.swapchain.lock().image_extent()[1] as f32;
        // self.proj = glm::perspective(
        //     aspect_ratio,
        //     std::f32::consts::FRAC_PI_2 as f32,
        //     0.01,
        //     10000.0,
        // );

        // let rot = glm::quat_to_mat3(&self.cam_rot);
        // let target = self.cam_pos + rot * Vec3::z();
        // let up = rot * Vec3::y();
        // self.view = glm::look_at_lh(&self.cam_pos, &target, &up);

        transform_compute.update_mvp(
            builder,
            vk.device.clone(),
            self.view.clone(),
            self.proj.clone(),
            &transform_uniforms,
            compute_pipeline.clone(),
            transform_data.0 as i32,
            vk.mem_alloc.clone(),
            &vk.comm_alloc,
            vk.desc_alloc.clone(),
        );
        if offset_vec.len() > 0 {
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
                            num_jobs: rd.transforms_len as i32,
                            stage: 1,
                            view: self.view.into(),
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
                            WriteDescriptorSet::buffer(4, transform_compute.transform.clone()),
                            WriteDescriptorSet::buffer(5, offsets_buffer.clone()),
                            WriteDescriptorSet::buffer(6, uniforms.clone()),
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
                            update_renderers_set.clone(),
                        )
                        .dispatch([rd.transforms_len as u32 / 128 + 1, 1, 1])
                        .unwrap();
                }
            }
        }
        particles.sort.sort(
            // per camera
            self.view.into(),
            self.proj.into(),
            transform_compute.transform.clone(),
            &particles.particle_buffers,
            vk.device.clone(),
            vk.queue.clone(),
            builder,
            &vk.desc_alloc,
        );
        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![
                        Some([0.2, 0.25, 1., 1.].into()),
                        // Some([0., 0., 0., 1.].into()),
                        Some(1f32.into()),
                    ],
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
                if let Some(ind) = rd.model_indirect.get(&m_id) {
                    if let Some(mr) = mm.assets_id.get(&m_id) {
                        let mr = mr.lock();
                        if ind.count == 0 {
                            continue;
                        }
                        if let Some(indirect_buffer) =
                            BufferSlice::from_typed_buffer_access(rm.indirect_buffer.clone())
                                .slice(ind.id as u64..(ind.id + 1) as u64)
                        {
                            // println!("{}",indirect_buffer.len());
                            if let Some(renderer_buffer) =
                                BufferSlice::from_typed_buffer_access(rm.renderers_gpu.clone())
                                    .slice(offset..(offset + ind.count as u64) as u64)
                            {
                                // println!("{}",renderer_buffer.len());
                                self.rend.bind_mesh(
                                    &texture_manager.lock(),
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
                    offset += ind.count as u64;
                }
            }
        }
        let mut rjd = RenderJobData {
            builder,
            transforms: transform_compute.transform.clone(),
            mvp: transform_compute.mvp.clone(),
            view: &self.view,
            proj: &self.proj,
            pipeline: &self.rend,
            viewport: &self.viewport,
            texture_manager: &texture_manager,
            vk: vk.clone(),
        };
        for job in render_jobs {
            job(&mut rjd);
        }
        particles.render_particles(
            &self.particle_render_pipeline,
            builder,
            self.view.clone(),
            self.proj.clone(),
            self.cam_rot.coords.into(),
            self.cam_pos.into(),
            transform_compute.transform.clone(),
            vk.mem_alloc.clone(),
            &vk.comm_alloc,
            vk.desc_alloc.clone(),
        );
        builder.end_render_pass().unwrap();
        self.output[image_num as usize].clone()
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
