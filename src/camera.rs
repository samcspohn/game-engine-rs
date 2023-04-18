use std::sync::Arc;

use glm::{Vec3, Quat, Mat4};
use nalgebra_glm as glm;
use vulkano::{image::{SwapchainImage, view::ImageView, ImageAccess, AttachmentImage, ImageUsage}, render_pass::{RenderPass, Framebuffer, FramebufferCreateInfo}, pipeline::graphics::viewport::Viewport, memory::allocator::StandardMemoryAllocator, format::Format, device::Device, swapchain::{Surface, Swapchain, SwapchainCreateInfo}};
use winit::window::Window;

use crate::{engine::{Component, transform::Transform, System}, FrameImage};


#[derive(Clone)]
pub struct CameraData {
    surface: Arc<Surface>,
    render_pass: Arc<RenderPass>,
    view_port: Arc<Viewport>,
    swap_chain: Arc<Swapchain>,
    cam_pos: Vec3,
    cam_rot: Quat,
    view: Mat4,
    proj: Mat4,
}
pub struct Camera {
    data: CameraData,
    fov: f32,
    near: f32,
    far: f32,
}
impl Component for Camera {
    fn update(&mut self, transform: Transform, sys: &System) {
        self.data.cam_pos = transform.get_position();
        self.data.cam_rot = transform.get_rotation();
        let rot = glm::quat_to_mat3(&self.data.cam_rot);
        let target = self.data.cam_pos + rot * Vec3::z();
        let up = rot * Vec3::y();
        let view: Mat4 = glm::look_at_lh(&self.data.cam_pos, &target, &up);
        self.data.view = view;
    }
}
impl Camera {
    pub fn get_mat(&self) -> CameraData {
        self.data.clone()
    }
}
impl CameraData {
    pub fn resize(&mut self) {

    }
    pub fn init(&mut self, device: Arc<Device>, surface: Arc<Surface>) {
        let (mut swapchain, images) = {
            // Querying the capabilities of the surface. When we create the swapchain we can only
            // pass values that are allowed by the capabilities.
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
    
            // Choosing the internal format that the images will have.
            let image_format = Some(
                device
                    .physical_device()
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            );
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
    
            // Please take a look at the docs for the meaning of the parameters we didn't mention.
            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count,
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage {
                        color_attachment: true,
                        ..ImageUsage::empty()
                    },
                    composite_alpha: surface_capabilities
                        .supported_composite_alpha
                        .iter()
                        .next()
                        .unwrap(),
                    present_mode: vulkano::swapchain::PresentMode::Immediate,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        self.swap_chain = swapchain.clone();
        self.render_pass = vulkano::ordered_passes_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R8G8B8A8_UNORM,
                    samples: 1,
                },
                final_color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.image_format(),
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
                { color: [color], depth_stencil: {depth}, input: [] },
                // { color: [color], depth_stencil: {depth}, input: [] }, // for secondary cmmand buffers
                { color: [final_color], depth_stencil: {}, input: [] } // Create a second renderpass to draw egui
            ]
        )
        .unwrap();
    }
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
    mem: Arc<StandardMemoryAllocator>,
    color: &mut FrameImage,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(&mem, dimensions, Format::D32_SFLOAT).unwrap(),
    )
    .unwrap();

    color.arc = AttachmentImage::with_usage(
        &mem,
        dimensions,
        Format::R8G8B8A8_UNORM,
        ImageUsage {
            transfer_src: true,
            transfer_dst: false,
            sampled: false,
            storage: true,
            color_attachment: true,
            depth_stencil_attachment: false,
            transient_attachment: false,
            input_attachment: true,
            ..ImageUsage::empty()
        },
    )
    .unwrap();
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            let color_view = ImageView::new_default(color.arc.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![color_view.clone(), view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}