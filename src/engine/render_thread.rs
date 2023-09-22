use std::{
    collections::HashMap,
    mem::size_of,
    sync::Arc,
    time::{Duration, Instant},
};

use super::{
    particles::shaders::cs::{burst, emitter_init},
    rendering::{
        camera::{CameraData, CameraViewData},
        component::RendererData,
    },
    utils::GPUWork,
    world::transform::TransformData,
    Engine, EnginePtr, RenderJobData, IMAGE_VIEW,
};
use crate::{
    editor::{self, editor_ui::EDITOR_WINDOW_DIM},
    engine::rendering::vulkan_manager::VulkanManager,
};
use crossbeam::channel::{Receiver, Sender};
use egui::TextureId;
use force_send_sync::SendSync;
use glm::Vec3;
use nalgebra_glm as glm;
use parking_lot::Mutex;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo,
        SecondaryAutoCommandBuffer, SubpassContents,
    },
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageDimensions, ImmutableImage, MipmapsCount, StorageImage,
        SwapchainImage,
    },
    memory::allocator::MemoryUsage,
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    sampler::{SamplerAddressMode, SamplerCreateInfo, LOD_CLAMP_NONE},
    swapchain::{
        acquire_next_image, AcquireError, SwapchainAcquireFuture, SwapchainCreateInfo,
        SwapchainCreationError, SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
};

#[repr(C)]
pub struct RenderingData {
    pub transform_data: TransformData,
    pub cam_datas: Vec<(Option<Arc<Mutex<CameraData>>>, Option<CameraViewData>)>,
    pub main_cam_id: i32,
    pub renderer_data: RendererData,
    pub emitter_inits: (usize, Vec<emitter_init>, Vec<emitter_init>, Vec<burst>),
    pub gpu_work: GPUWork,
    pub render_jobs: Vec<Box<dyn Fn(&mut RenderJobData<'_>) + Send + Sync>>,
    pub _image_num: u32,
    pub gui_commands: SendSync<SecondaryAutoCommandBuffer>,
    pub engine_ptr: EnginePtr,
    pub recreate_swapchain: bool,
    pub editor_size: [u32; 2], // pub image: Arc<ImageView<StorageImage>>,
}

/// This method is called once during initialization, then again whenever the window is resized
pub(super) fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            // let color_view = ImageView::new_default(color.arc.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

// fn swapchain_thread(
//     vk: Arc<VulkanManager>,
//     commands: Receiver<(u32, SwapchainAcquireFuture, PrimaryAutoCommandBuffer)>,
// ) {
//     let mut previous_frame_end = Some(sync::now(vk.device.clone()).boxed());

//     loop {
//         let (image_num, acquire_future, command_buffer) = commands.recv().unwrap();

//         previous_frame_end.as_mut().unwrap().cleanup_finished();
//         let future = previous_frame_end
//             .take()
//             .unwrap()
//             .join(acquire_future)
//             .then_execute(vk.queue.clone(), command_buffer)
//             .unwrap()
//             .then_swapchain_present(
//                 vk.queue.clone(),
//                 SwapchainPresentInfo::swapchain_image_index(vk.swapchain().clone(), image_num),
//             )
//             .then_signal_fence_and_flush();

//         match future {
//             Ok(future) => {
//                 previous_frame_end = Some(future.boxed());
//             }
//             Err(FlushError::OutOfDate) => {
//                 // recreate_swapchain = true;
//                 previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
//             }
//             Err(e) => {
//                 println!("failed to flush future: {e}");
//                 previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
//             }
//         }
//     }
// }
pub(super) fn render_fn(rd: RendererData) {}
pub(super) fn render_thread(
    vk: Arc<VulkanManager>,
    // render_pass: Arc<RenderPass>,
    rendering_data: Receiver<(u32, SwapchainAcquireFuture, PrimaryAutoCommandBuffer)>,
    rendering_complete: Sender<()>,
) {
    // let mut viewport = Viewport {
    //     origin: [0.0, 0.0],
    //     dimensions: [0.0, 0.0],
    //     depth_range: 0.0..1.0,
    // };

    // let mut framebuffers =
    //     window_size_dependent_setup(&vk.images, render_pass.clone(), &mut viewport);
    // let mut recreate_swapchain = true;
    // let mut fc_map: HashMap<i32, HashMap<u32, TextureId>> = HashMap::new();
    // let mut editor_window_image: Option<Arc<dyn ImageAccess>> = None;
    // let (comms_snd, coms_rcv) = crossbeam::channel::bounded(1);
    // let img_view = None;
    // let swapchain_thread = Arc::new({
    //     let vk = vk.clone();
    //     std::thread::spawn(move || swapchain_thread(vk, coms_rcv))
    // });

    // let mut previous_frame_end = Some(sync::now(vk.device.clone()).boxed());

    rendering_complete.send(()).unwrap();
    // previous_frame_end: Some(sync::now(vk.device.clone()).boxed()),
    loop {
        let (image_num, acquire_future, command_buffer) = rendering_data.recv().unwrap();
        let future = acquire_future
            .then_execute(vk.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(
                vk.queue.clone(),
                SwapchainPresentInfo::swapchain_image_index(vk.swapchain().clone(), image_num),
            ).then_signal_fence();
            future.flush();
    rendering_complete.send(()).unwrap();

        //     .unwrap();
        // previous_frame_end.as_mut().unwrap().cleanup_finished();
        // let future = previous_frame_end
        //     .take()
        //     .unwrap()
        //     .join(acquire_future)
        //     .then_execute(vk.queue.clone(), command_buffer)
        //     .unwrap()
        //     .then_swapchain_present(
        //         vk.queue.clone(),
        //         SwapchainPresentInfo::swapchain_image_index(vk.swapchain().clone(), image_num),
        //     )
        //     .then_signal_fence();
        // let future = future.flush();
        // previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
        // let future = acquire_future
        //     .then_execute(vk.queue.clone(), command_buffer)
        //     .unwrap()
        //     // .then_execute(vk.queue.clone(), command_buffer)
        //     // .unwrap()
        //     .then_swapchain_present(
        //         vk.queue.clone(),
        //         SwapchainPresentInfo::swapchain_image_index(vk.swapchain().clone(), image_num),
        //     ).flush();
        //     // .then_signal_fence_and_flush();

        // match future {
        //     Ok(future) => {
        //         // *previous_frame_end = Some(future.boxed());
        //     }
        //     Err(FlushError::OutOfDate) => {
        //         // *recreate_swapchain = true;
        //         // *previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
        //     }
        //     Err(e) => {
        //         println!("failed to flush future: {e}");
        //         // *recreate_swapchain = true;
        //         // *previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
        //     }
        // }
    }
}
