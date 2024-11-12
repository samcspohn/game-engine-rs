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
    utils::{gpu_perf::{self, GpuPerf}, perf::Perf, GPUWork},
    world::transform::TransformData,
    Engine, EnginePtr, RenderData,
};
use crate::{
    editor::{self, editor_ui::EDITOR_WINDOW_DIM},
    engine::rendering::{lighting::lighting_compute::LIGHTING_COMPUTE_TIMESTAMP, vulkan_manager::VulkanManager},
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
    image::view::ImageView,
    memory::allocator::MemoryTypeFilter,
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass},
    swapchain::{
        acquire_next_image, SwapchainAcquireFuture, SwapchainCreateInfo, SwapchainPresentInfo,
    },
    sync::{self, GpuFuture},
};

#[repr(C)]
pub struct RenderingData {
    pub transform_data: TransformData,
    pub cam_datas: Vec<(Option<Arc<Mutex<CameraData>>>, Option<CameraViewData>)>,
    pub main_cam_id: i32,
    pub renderer_data: RendererData,
    pub emitter_inits: (usize, Vec<emitter_init>, Vec<emitter_init>, Vec<burst>),
    pub gpu_work: GPUWork,
    pub render_jobs: Vec<Box<dyn Fn(&mut RenderData<'_>) + Send + Sync>>,
    pub _image_num: u32,
    pub gui_commands: SendSync<SecondaryAutoCommandBuffer>,
    pub engine_ptr: EnginePtr,
    pub recreate_swapchain: bool,
    pub editor_size: [u32; 2], // pub image: Arc<ImageView<StorageImage>>,
}

// /// This method is called once during initialization, then again whenever the window is resized
// pub(super) fn window_size_dependent_setup(
//     images: &[Arc<Image>],
//     render_pass: Arc<RenderPass>,
//     viewport: &mut Viewport,
// ) -> Vec<Arc<Framebuffer>> {
//     let dimensions = images[0].dimensions().width_height();
//     viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
//     images
//         .iter()
//         .map(|image| {
//             let view = ImageView::new_default(image.clone()).unwrap();
//             // let color_view = ImageView::new_default(color.arc.clone()).unwrap();
//             Framebuffer::new(
//                 render_pass.clone(),
//                 FramebufferCreateInfo {
//                     attachments: vec![view],
//                     ..Default::default()
//                 },
//             )
//             .unwrap()
//         })
//         .collect::<Vec<_>>()
// }

pub(super) fn render_fn(rd: RendererData) {}
pub(super) fn render_thread(
    vk: Arc<VulkanManager>,
    gpu_perf: Arc<GpuPerf>,
    perf: Arc<Perf>,
    // render_pass: Arc<RenderPass>,
    rendering_data: Receiver<(
        bool,
        Option<(u32, SwapchainAcquireFuture, Arc<PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>>)>,
    )>,
    rendering_complete: Sender<bool>,
) {
    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
    rendering_complete.send(false).unwrap();
    loop {
        let (should_exit, rd) = rendering_data.recv().unwrap();
        if let Some((image_num, acquire_future, command_buffer)) = rd {
            previous_frame_end.take().unwrap().flush().unwrap();
            // let future = previous_frame_end
            //     .take()
            //     .unwrap()
            //     .join(acquire_future)
            let future = acquire_future
                .then_execute(vk.queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(
                    vk.queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(vk.swapchain().clone(), image_num),
                )
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(future.boxed());
                }
                Err(e) => {
                    println!("failed to flush future: {e}");
                    recreate_swapchain = true;
                    previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
                }
            }
            previous_frame_end.as_mut().unwrap().cleanup_finished();

            let get_gpu_metrics = perf.node("get_gpu_metrics");
            for (name, perf) in gpu_perf.data.write().iter() {
                let nanos = vk.get_query(&perf.0);
                let dur = Duration::from_nanos(nanos);
                let mut perf = perf.1.push(dur);
                // let mut perf = perf.write();
                // perf.print(name, gpu_perf.start_time);
            }
            drop(get_gpu_metrics);

            // let lighting_compute_nanos = vk.get_query(unsafe { &LIGHTING_COMPUTE_TIMESTAMP });
            // println!(
            //     "lighting compute: {} ms",
            //     (Duration::from_nanos(lighting_compute_nanos as u64).as_secs_f64() * 1000.0) as f32
            // );


            // let future = acquire_future
            //     .then_execute(vk.queue.clone(), command_buffer)
            //     .unwrap()
            //     // .then_execute(vk.queue.clone(), command_buffer)
            //     // .unwrap()
            //     .then_swapchain_present(
            //         vk.queue.clone(),
            //         SwapchainPresentInfo::swapchain_image_index(vk.swapchain().clone(), image_num),
            //     )
            //     .then_signal_fence()
            //     .flush(); // FREEZE HERE

            // match future {
            //     Ok(future) => {
            //         // *previous_frame_end = Some(future.boxed());
            //     }
            //     Err(FlushError::OutOfDate) => {
            //         recreate_swapchain = true;
            //         // *previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
            //     }
            //     Err(e) => {
            //         println!("failed to flush future: {e}");
            //         recreate_swapchain = true;
            //         // *previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
            //     }
            // }

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
            //     .then_signal_fence_and_flush();

            // match future {
            //     Ok(future) => {
            //         previous_frame_end = Some(future.boxed());
            //     }
            //     Err(FlushError::OutOfDate) => {
            //         recreate_swapchain = true;
            //         previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
            //     }
            //     Err(e) => {
            //         println!("failed to flush future: {e}");
            //         previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
            //     }
            // }
            // previous_frame_end.as_mut().unwrap().cleanup_finished();

            // if should_exit {
            //     return;
            // }
        }
        if should_exit {
            return;
        }
        rendering_complete.send(recreate_swapchain).unwrap();
        recreate_swapchain = false;
    }
}
