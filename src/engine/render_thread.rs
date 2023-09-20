use std::{
    collections::HashMap,
    mem::size_of,
    sync::Arc,
    time::{Duration, Instant},
};

use super::{
    particles::shaders::cs::{burst, emitter_init},
    rendering::{camera::CameraData, component::RendererData},
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
    pub cam_datas: Vec<Arc<Mutex<CameraData>>>,
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
fn window_size_dependent_setup(
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
pub(super) fn render_thread(
    vk: Arc<VulkanManager>,
    render_pass: Arc<RenderPass>,
    rendering_data: Receiver<(bool, RenderingData)>,
    rendering_complete: Sender<()>,
) {
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers =
        window_size_dependent_setup(&vk.images, render_pass.clone(), &mut viewport);
    let mut recreate_swapchain = true;
    let mut fc_map: HashMap<i32, HashMap<u32, TextureId>> = HashMap::new();
    let mut editor_window_image: Option<Arc<dyn ImageAccess>> = None;
    // let (comms_snd, coms_rcv) = crossbeam::channel::bounded(1);
    // let img_view = None;
    // let swapchain_thread = Arc::new({
    //     let vk = vk.clone();
    //     std::thread::spawn(move || swapchain_thread(vk, coms_rcv))
    // });

    let mut previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
    // rendering_complete.send(()).unwrap();
    loop {
        let (quit, rd) = rendering_data.recv().unwrap();
        if quit {
            return;
        }
        let RenderingData {
            transform_data,
            mut cam_datas,
            main_cam_id,
            renderer_data: mut rd,
            emitter_inits: particle_init_data,
            gpu_work,
            render_jobs,
            _image_num,
            // image,
            gui_commands,
            engine_ptr,
            recreate_swapchain: _recreate_swapchain,
            editor_size,
        } = rd;
        let engine = unsafe { engine_ptr.ptr.as_ref().unwrap() };
        let full_render_time = engine.perf.node("full render time");
        // let render_jobs = engine.world.lock().render();

        let dimensions = vk.window().inner_size();
        if dimensions.width == 0 || dimensions.height == 0 {
            return;
        }
        // println!("render particles: {}", vk.get_query(&0));
        recreate_swapchain |= _recreate_swapchain;
        if recreate_swapchain {
            let dimensions: [u32; 2] = vk.window().inner_size().into();

            let mut swapchain = vk.swapchain();
            let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                image_extent: dimensions,
                ..swapchain.create_info()
            }) {
                Ok(r) => r,
                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
            };

            vk.update_swapchain(new_swapchain);

            framebuffers =
                window_size_dependent_setup(&new_images, render_pass.clone(), &mut viewport);
            viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
            recreate_swapchain = false;
        }

        let (image_num, suboptimal, acquire_future) =
            match acquire_next_image(vk.swapchain(), Some(Duration::from_secs(10))) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    println!("falied to aquire next image");
                    return;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };
        if suboptimal {
            recreate_swapchain = true;
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            &vk.comm_alloc,
            vk.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let mut rm = engine.shared_render_data.write();

        engine.transform_compute.write()._get_update_data(&transform_data, image_num);
        rendering_complete.send(()).unwrap();
        engine.transform_compute.write().update_data(
            &mut builder,
            image_num,
            &transform_data,
            &engine.perf,
        );



        let particle_update = engine.perf.node("particle update");
        engine.particles_system.update(
            &mut builder,
            particle_init_data,
            &engine.transform_compute.read(),
            &engine.time,
        );
        drop(particle_update);

        let update_renderers = engine.perf.node("update renderers");
        // compute shader renderers
        let offset_vec = {
            // puffin::profile_scope!("process renderers");
            let renderer_pipeline = rm.pipeline.clone();

            builder.bind_pipeline_compute(renderer_pipeline.clone());

            // if !lock_cull {
            //     cull_view = view.clone();
            // }

            rm.update(
                &mut rd,
                vk.clone(),
                &mut builder,
                renderer_pipeline.clone(),
                &engine.transform_compute.read(),
            )
        };
        drop(update_renderers);
        
        let render_cameras = engine.perf.node("render camera(s)");

        while let Some(job) = gpu_work.pop() {
            job.unwrap()(&mut builder, vk.clone());
        }

        let mut game_image = None;
        for cam in cam_datas {
            game_image = cam.lock().render(
                vk.clone(),
                &mut builder,
                &engine.transform_compute.read(),
                engine.particles_system.clone(),
                &transform_data,
                rm.pipeline.clone(),
                offset_vec.clone(),
                &mut rm,
                &mut rd,
                image_num,
                engine.assets_manager.clone(),
                &render_jobs,
            );
        }

        drop(render_cameras);
        if editor_window_image.is_none()
            || editor_size
                != [
                    editor_window_image.as_ref().unwrap().dimensions().width(),
                    editor_window_image.as_ref().unwrap().dimensions().height(),
                ]
        {
            let image = ImmutableImage::from_iter(
                &vk.mem_alloc,
                (0..(editor_size[0] * editor_size[1])).map(|_| [0u8; 4]),
                ImageDimensions::Dim2d {
                    width: editor_size[0],
                    height: editor_size[1],
                    array_layers: 1,
                },
                MipmapsCount::Log2,
                Format::R8G8B8A8_UNORM,
                &mut builder,
            )
            .unwrap();
            let img_view = ImageView::new_default(image.clone()).unwrap();
            editor_window_image = Some(image.clone());
            *IMAGE_VIEW.lock() = Some((img_view, image));
            // img_view = Some(img_view);
        }

        if let Some(image) = &editor_window_image {
            if let Some(game_image) = game_image {
                builder
                    .copy_image(CopyImageInfo::images(game_image, image.clone()))
                    .unwrap();
            }
        }

        /////////////////////////////////////////////////////////

        let _render = engine.perf.node("_ render");

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0., 0., 0., 1.].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffers[image_num as usize].clone())
                },
                SubpassContents::SecondaryCommandBuffers,
            )
            .unwrap()
            .set_viewport(0, [viewport.clone()]);
        // engine.perf.update("_ begin render pass".into(), Instant::now() - _inst);
        let gui_commands = gui_commands.unwrap();
        builder.execute_commands(gui_commands).unwrap();

        builder.end_render_pass().unwrap();

        let _build_command_buffer = engine.perf.node("_ build command buffer");
        let command_buffer = builder.build().unwrap();
        let _execute = engine.perf.node("_ execute");
        // comms_snd.send((image_num, acquire_future, command_buffer));
        previous_frame_end.as_mut().unwrap().cleanup_finished();

        let future = previous_frame_end
            .take()
            .unwrap()
            .join(acquire_future)
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
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
            }
            Err(e) => {
                println!("failed to flush future: {e}");
                previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
            }
        }
        drop(_build_command_buffer);

        // let _wait_for_previous_frame = engine.perf.node("_ wait for previous frame");

        ///////////////////////////////////////////////////////////////////////////////
        /// execute commands now. move into seperate thread?
        ///////////////////////////////////////////////////////////////////////////////
        drop(_execute);
    }
}
