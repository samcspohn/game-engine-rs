#![allow(warnings)]
#![feature(downcast_unchecked)]
#![feature(sync_unsafe_cell)]

mod editor;
mod engine;
mod terrain_eng;

use crate::editor::editor_ui::EDITOR_ASPECT_RATIO;
use crate::engine::input::Input;
use crate::engine::main_loop::{main_loop, RenderingData};
use crate::engine::particles::{component::ParticleEmitter, particles::ParticleCompute};
use crate::engine::perf::Perf;
use crate::engine::project::asset_manager::{AssetManagerBase, AssetsManager};
use crate::engine::project::{file_watcher, Project};
use crate::engine::project::{load_project, save_project};
use crate::engine::rendering::camera::{Camera, CameraData};
use crate::engine::rendering::model::{ModelManager, ModelRenderer};
use crate::engine::rendering::renderer_component::{buffer_usage_all, Renderer};
use crate::engine::rendering::texture::TextureManager;
use crate::engine::rendering::vulkan_manager::VulkanManager;
use crate::engine::transform_compute::{cs, TransformCompute};
use crate::engine::world::World;
use crate::engine::{particles, runtime_compilation, transform_compute};

use crossbeam::channel::{Receiver, Sender};
use crossbeam::queue::SegQueue;
// use egui::plot::{HLine, Line, Plot, Value, Values};
use egui::TextureId;

use glm::{vec4, Vec3};
use nalgebra_glm as glm;
use notify::{RecursiveMode, Watcher};
use parking_lot::{Mutex, RwLock};
use puffin_egui::*;
use std::{
    collections::{BTreeMap, HashMap},
    env, fs,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self},
    time::{Duration, Instant},
};
use vulkano::{
    buffer::CpuBufferPool,
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
    },
    image::{view::ImageView, AttachmentImage, ImageAccess, SwapchainImage},
    memory::allocator::MemoryUsage,
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        acquire_next_image, AcquireError, SwapchainCreateInfo, SwapchainCreationError,
        SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
};
use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, ModifiersState, MouseButton,
        VirtualKeyCode, WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

#[cfg(target_os = "windows")]
mod win_alloc {
    use mimalloc::MiMalloc;

    #[global_allocator]
    static GLOBAL: MiMalloc = MiMalloc;
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    crate::engine::utils::SETTINGS
        .read()
        .get::<i32>("MAX_PARTICLES")
        .unwrap();
    let args: Vec<String> = env::args().collect();
    // dbg!(args);
    let engine_dir = env::current_dir().ok().unwrap();
    assert!(env::set_current_dir(&Path::new(&args[1])).is_ok()); // TODO move to top. procedurally generate cube/move cube to built in assets
    match thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Max) {
        Ok(_) => println!("set main thread priority"),
        Err(_) => println!("failed to set main thread priority"),
    }
    println!("main thread id: {:?}", thread::current().id());
    println!(
        "main thread priority: {:?}",
        thread_priority::get_current_thread_priority().ok().unwrap()
    );

    let path = "runtime";
    if let Ok(_) = fs::remove_dir_all(path) {}
    fs::create_dir(path).unwrap();

    let event_loop = EventLoop::new();
    let mut engine = engine::Engine::new(&event_loop, &engine_dir, &args[1]);
    let vk = engine.vk.clone();
    let render_pass = vulkano::single_pass_renderpass!(
        engine.vk.device.clone(),
        attachments: {
            final_color: {
                load: Clear,
                store: Store,
                format: vk.swapchain().image_format(),
                samples: 1,
            }
        },
        pass: { color: [final_color], depth_stencil: {}} // Create a second renderpass to draw egui
    )
    .unwrap();
    engine.init();
    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers =
        window_size_dependent_setup(&vk.images, render_pass.clone(), &mut viewport);
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(vk.device.clone()).boxed());

    let mut modifiers = ModifiersState::default();

    let mut gui = egui_winit_vulkano::Gui::new_with_subpass(
        &event_loop,
        vk.surface.clone(),
        Some(vk.swapchain().image_format()),
        vk.queue.clone(),
        Subpass::from(render_pass.clone(), 0).unwrap(),
    );

    let mut fc_map: HashMap<i32, HashMap<u32, TextureId>> = HashMap::new();

    let mut frame_time = Instant::now();

    let mut focused = true;

    let mut cam_data = Arc::new(Mutex::new(CameraData::new(vk.clone())));
    let mut playing_game = false;

    let mut editor_cam = editor::editor_cam::EditorCam {
        rot: glm::quat(-1., 0., 0., 0.),
        pos: Vec3::default(),
        speed: 30f32,
    };
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::DeviceEvent { event, .. } => engine.input.process_device(event, focused),
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::Focused(foc) => {
                        focused = foc;
                        println!("main event_loop id: {:?}", thread::current().id());
                    }
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                        engine.end();
                    }
                    WindowEvent::MouseInput {
                        device_id,
                        state,
                        button,
                        modifiers,
                    } => engine
                        .input
                        .process_mouse_input(device_id, state, button, modifiers),
                    WindowEvent::MouseWheel {
                        device_id,
                        delta,
                        phase,
                        modifiers,
                    } => engine
                        .input
                        .process_mouse_wheel(device_id, delta, phase, modifiers),

                    WindowEvent::KeyboardInput {
                        input: ky_input,
                        device_id,
                        is_synthetic,
                    } => engine
                        .input
                        .process_keyboard(device_id, ky_input, is_synthetic),
                    WindowEvent::ModifiersChanged(m) => modifiers = m,
                    WindowEvent::Resized(_size) => {
                        recreate_swapchain = true;
                    }
                    _ => (),
                }

                // if !input.get_key(&VirtualKeyCode::Space) {
                gui.update(&event);
                // }
            }
            Event::RedrawEventsCleared => {
                puffin::GlobalProfiler::lock().new_frame();

                puffin::profile_scope!("full");

                ////////////////////////////////////

                let _full = engine.perf.node("_ full");
                let mut rendering_data = match engine.update_sim() {
                    Some(a) => a,
                    None => return,
                };

                if !playing_game {
                    editor_cam.update(&engine.input);
                }

                engine
                    .file_watcher
                    .get_updates(engine.assets_manager.clone());

                let dimensions = vk.window().inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    println!("recreate swapchain");
                    println!("dimensions {}: {}", dimensions.width, dimensions.height);
                    let dimensions: [u32; 2] = vk.window().inner_size().into();

                    let mut swapchain = vk.swapchain();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate(SwapchainCreateInfo {
                            image_extent: dimensions,
                            ..swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    vk.update_swapchain(new_swapchain);

                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );
                    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
                    recreate_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) =
                    match acquire_next_image(vk.swapchain(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            println!("falied to aquire next image");
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                let cam_num = if playing_game {
                    rendering_data.main_cam_id
                } else {
                    -1
                };
                let fc = fc_map
                    .entry(cam_num)
                    .or_insert(HashMap::<u32, TextureId>::new())
                    .entry(image_num)
                    .or_insert_with(|| {
                        let frame_image_view = ImageView::new_default(if playing_game {
                            rendering_data.cam_datas[0].lock().output[image_num as usize].clone()
                        } else {
                            cam_data.lock().output[image_num as usize].clone()
                        })
                        .unwrap();

                        gui.register_user_image_view(frame_image_view.clone())
                    });

                if suboptimal {
                    recreate_swapchain = true;
                }

                let _gui = engine.perf.node("_ gui");
                let dimensions = *EDITOR_ASPECT_RATIO.lock();
                let mut _playing_game = false;
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    _playing_game = editor::editor_ui::editor_ui(
                        &engine.world,
                        &mut engine.fps_queue,
                        &ctx,
                        *fc,
                        engine.assets_manager.clone(),
                    );
                });
                {
                    let ear = EDITOR_ASPECT_RATIO.lock();
                    if dimensions != *ear {
                        cam_data.lock().resize(*ear, vk.clone());
                        fc_map.clear();
                    }
                }
                drop(_gui);

                let mut builder = AutoCommandBufferBuilder::primary(
                    &vk.comm_alloc,
                    vk.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();
                if !playing_game {
                    cam_data.lock().update(
                        editor_cam.pos,
                        editor_cam.rot,
                        0.01f32,
                        10_000f32,
                        70f32,
                    );
                    rendering_data.cam_datas = vec![cam_data.clone()];
                }
                engine.render(&mut builder, rendering_data, _playing_game, image_num);

                let _render = engine.perf.node("_ render");

                builder
                    .begin_render_pass(
                        RenderPassBeginInfo {
                            clear_values: vec![Some([0., 0., 0., 1.].into())],
                            ..RenderPassBeginInfo::framebuffer(
                                framebuffers[image_num as usize].clone(),
                            )
                        },
                        SubpassContents::SecondaryCommandBuffers,
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()]);
                // engine.perf.update("_ begin render pass".into(), Instant::now() - _inst);

                let _get_gui_commands = engine.perf.node("_ get gui commands");
                let size = vk.window().inner_size();
                let gui_commands = gui.draw_on_subpass_image([size.width, size.height]);
                drop(_get_gui_commands);

                builder.execute_commands(gui_commands).unwrap();

                frame_time = Instant::now();

                builder.end_render_pass().unwrap();

                let _build_command_buffer = engine.perf.node("_ build command buffer");
                let command_buffer = builder.build().unwrap();
                drop(_build_command_buffer);

                let _wait_for_previous_frame = engine.perf.node("_ wait for previous frame");
                let execute = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(vk.queue.clone(), command_buffer);
                drop(_wait_for_previous_frame);

                let _execute = engine.perf.node("_ execute");
                match execute {
                    Ok(execute) => {
                        let future = execute
                            .then_swapchain_present(
                                vk.queue.clone(),
                                SwapchainPresentInfo::swapchain_image_index(
                                    vk.swapchain(),
                                    image_num,
                                ),
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
                                println!("Failed to flush future: {:?}", e);
                                previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
                            }
                        }
                        // let c = vk.get_query(&particles.performance.init_emitters);
                        // println!("init emitters: {}",c);
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
                    }
                };
                drop(_execute);

                // was playing != next frame playing?
                if playing_game != _playing_game {
                    if playing_game {
                        for (k, v) in &fc_map {
                            if *k != -1 {
                                for (_k, v) in v.iter() {
                                    gui.unregister_user_image(*v);
                                }
                            }
                        }
                        let a = fc_map.remove(&-1).unwrap(); // get
                        fc_map.clear();
                        fc_map.insert(-1, a); // replace
                    }
                    recreate_swapchain = true;
                }
                playing_game = _playing_game;

                // if first_frame {
                //     puffin::set_scopes_on(true);
                //     first_frame = false;
                // }
            }
            _ => (),
        }
    });
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
