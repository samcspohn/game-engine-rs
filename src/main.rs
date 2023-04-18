use crossbeam::queue::SegQueue;
// use egui::plot::{HLine, Line, Plot, Value, Values};
use egui::{Color32, Rounding, Ui, WidgetText};
use glium::memory_object;
use rapier3d::na::dimension;
use std::{env, fs};
use vulkan_manager::VulkanManager;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    BlitImageInfo, CopyBufferInfo, CopyImageInfo, ImageBlit, PrimaryCommandBufferAbstract,
    RenderPassBeginInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::image::sys::RawImage;
use vulkano::image::view::ImageViewCreateInfo;
use vulkano::image::{ImageDimensions, ImageLayout, StorageImage};
use vulkano::memory::allocator::{MemoryUsage, StandardMemoryAllocator};
use vulkano::swapchain::SwapchainPresentInfo;
use vulkano::VulkanLibrary;
use winit::window::CursorGrabMode;
// use egui_dock::Tree;
use puffin_egui::*;

use nalgebra_glm as glm;
use parking_lot::{Mutex, RwLock};
use vulkano::buffer::{BufferContents, BufferSlice, TypedBufferAccess};

use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::{Features, QueueFlags};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
use winit::dpi::LogicalSize;

use winit::event::MouseButton;

use std::any::TypeId;
use std::collections::VecDeque;

use std::path::{Path, PathBuf};
use std::{
    collections::HashMap,
    ptr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self},
    time::{Duration, Instant},
};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        acquire_next_image, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
    DeviceSize,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, ModifiersState, VirtualKeyCode,
        WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use glm::{Mat4, Vec3};

use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
// use rust_test::{INDICES, NORMALS, VERTICES};
use notify::{RecommendedWatcher, RecursiveMode, Result, Watcher};
use walkdir::WalkDir;

mod engine;
mod input;
mod model;
mod perf;
mod renderer;
// mod renderer_component;
mod asset_manager;
mod camera;
mod color_gradient;
mod drag_drop;
mod editor_ui;
mod file_watcher;
mod game;
mod inspectable;
mod particle_sort;
mod particles;
mod project;
mod renderer_component2;
mod serialize;
mod terrain;
mod texture;
mod time;
mod transform_compute;
mod vulkan_manager;
// use rand::prelude::*;
// use rapier3d::prelude::*;

use crate::asset_manager::{AssetManagerBase, AssetsManager};
use crate::engine::physics::Physics;
use crate::engine::transform::{Transform, Transforms};
use crate::engine::{GameObject, RenderJobData, Sys, World};
use crate::game::{game_thread_fn, Bomb, Player};
use crate::inspectable::{Inpsect, Ins};
use crate::model::{ModelManager, ModelRenderer};
use crate::particles::cs::ty::t;
use crate::particles::ParticleEmitter;
use crate::perf::Perf;

use crate::project::{load_project, save_project};
use crate::renderer_component2::{buffer_usage_all, ur, Renderer, RendererData, RendererManager};
use crate::terrain::Terrain;
use crate::texture::TextureManager;
use crate::transform_compute::cs;
use crate::transform_compute::cs::ty::transform;
use crate::{drag_drop::drag_source, drag_drop::drop_target};
use crate::{input::Input, renderer::RenderPipeline};

struct FrameImage {
    arc: Arc<AttachmentImage>,
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    if let Ok(mut watcher) = notify::recommended_watcher(|res| match res {
        Ok(event) => println!("event: {:?}", event),
        Err(e) => println!("watch error: {:?}", e),
    }) {
        // Add a path to be watched. All files and directories at that path and
        // below will be monitored for changes.
        if let Ok(_) = watcher.watch(Path::new("./test_project"), RecursiveMode::Recursive) {}
    }
    let event_loop = EventLoop::new();
    let vk = VulkanManager::new(&event_loop);
    let device = vk.device.clone();
    let queue = vk.queue.clone();
    let surface = vk.surface.clone();

    // Before we can draw on the surface, we have to create what is called a swapchain. Creating
    // a swapchain allocates the color buffers that will contain the image that will ultimately
    // be visible on the screen. These images are returned alongside the swapchain.
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
    // // let img_count = swapchain.image_count();
    // let vk.mem_alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    // let vk.desc_alloc = Arc::new(StandardDescriptorSetAllocator::new(device.clone()));
    // let vk.comm_alloc = Arc::new(StandardCommandBufferAllocator::new(
    //     device.clone(),
    //     Default::default(),
    // ));
    let render_pass = vulkano::ordered_passes_renderpass!(
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

    // let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

    let texture_manager = Arc::new(Mutex::new(TextureManager::new((
        device.clone(),
        queue.clone(),
        vk.mem_alloc.clone(),
    ))));

    let model_manager = ModelManager::new((
        device.clone(),
        texture_manager.clone(),
        vk.mem_alloc.clone(),
    ));

    let renderer_manager = Arc::new(RwLock::new(RendererManager::new(
        device.clone(),
        vk.mem_alloc.clone(),
    )));
    // let cube_mesh = Mesh::load_model("src/cube/cube.obj", device.clone(), texture_manager.clone());

    let model_manager = Arc::new(Mutex::new(model_manager));
    {
        model_manager.lock().from_file("src/cube/cube.obj");
    }

    let particles = Arc::new(particles::ParticleCompute::new(
        device.clone(),
        render_pass.clone(),
        // swapchain.clone(),
        queue.clone(),
        vk.mem_alloc.clone(),
        &vk.comm_alloc,
        vk.desc_alloc.clone(),
    ));

    let assets_manager = Arc::new(Mutex::new(AssetsManager::new()));
    {
        let mut assets_manager = assets_manager.lock();
        assets_manager.add_asset_manager("model", &["obj"], model_manager.clone());
        assets_manager.add_asset_manager("texture", &["png", "jpeg"], texture_manager.clone());
        assets_manager.add_asset_manager(
            "particle_template",
            &["ptem"],
            particles.particle_template_manager.clone(),
        );
    }
    // let uniform_buffer =
    //     CpuBufferPool::<renderer::vs::ty::Data>::new(device.clone(), BufferUsage::all());

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };
    // let viewport2 = Viewport {
    //     origin: [0.0, 0.0],
    //     dimensions: [1024.0, 720.0],
    //     depth_range: 0.0..1.0,
    // };
    let dimensions = images[0].dimensions().width_height();
    let mut frame_color = FrameImage {
        arc: AttachmentImage::with_usage(
            &vk.mem_alloc,
            dimensions,
            Format::R8G8B8A8_UNORM,
            ImageUsage {
                // sampled: true,
                // storage: true,
                transfer_src: true,
                transient_attachment: false,
                input_attachment: true,
                ..ImageUsage::empty()
            },
        )
        .unwrap(),
    };

    let frame_img = StorageImage::new(
        &vk.mem_alloc,
        ImageDimensions::Dim2d {
            width: dimensions[0],
            height: dimensions[1],
            array_layers: 1,
        },
        Format::R8G8B8A8_UNORM,
        device.active_queue_family_indices().iter().copied(),
    )
    .unwrap();

    let mut framebuffers = window_size_dependent_setup(
        &images,
        render_pass.clone(),
        &mut viewport,
        vk.mem_alloc.clone(),
        &mut frame_color,
    );

    let rend = RenderPipeline::new(
        device.clone(),
        render_pass.clone(),
        images[0].dimensions().width_height(),
        queue.clone(),
        0,
        vk.mem_alloc.clone(),
        &vk.comm_alloc,
    );
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    let mut modifiers = ModifiersState::default();

    //////////////////////////////////////////////////

    let mut input = Input {
        ..Default::default()
    };

    let mut perf = Perf {
        data: HashMap::<String, SegQueue<Duration>>::new(),
    };

    let _loops = 0;

    let mut gui = egui_winit_vulkano::Gui::new_with_subpass(
        &event_loop,
        surface.clone(),
        Some(swapchain.image_format()),
        queue.clone(),
        Subpass::from(render_pass.clone(), 1).unwrap(),
    );
    let frame_image_view = ImageView::new_default(frame_img.clone()).unwrap();
    let fc = gui.register_user_image_view(frame_image_view.clone());

    let mut frame_time = Instant::now();
    // let mut fps_time: f32 = 0.0;

    let mut transform_compute = transform_compute::transform_buffer_init(
        device.clone(),
        // queue.clone(),
        vec![
            transform {
                position: Default::default(),
                _dummy0: Default::default(),
                rotation: Default::default(),
                scale: Default::default(),
                _dummy1: Default::default()
            };
            1
        ],
        vk.mem_alloc.clone(),
        &vk.comm_alloc,
        vk.desc_alloc.clone(),
    );

    let cs = cs::load(device.clone()).unwrap();

    // Create compute-pipeline for applying compute shader to vertices.
    let compute_pipeline = vulkano::pipeline::ComputePipeline::new(
        device.clone(),
        cs.entry_point("main").unwrap(),
        &(),
        None,
        |_| {},
    )
    .expect("Failed to create compute shader");

    let transform_uniforms = CpuBufferPool::<cs::ty::Data>::new(
        vk.mem_alloc.clone(),
        buffer_usage_all(),
        MemoryUsage::Download,
    );

    let mut fps_queue = std::collections::VecDeque::new();
    // let render_uniforms = CpuBufferPool::<vs::ty::UniformBufferObject>::new(device.clone(), BufferUsage::all());

    let mut focused = true;

    // puffin::set_scopes_on(true);

    let mut cull_view = glm::Mat4::identity();
    let mut lock_cull = false;
    let mut first_frame = true;

    /////////////////////////////////////////////////////////////////////////////////////////
    let (tx, rx): (Sender<Input>, Receiver<Input>) = mpsc::channel();
    let (rtx, rrx): (Sender<_>, Receiver<_>) = mpsc::channel();
    // let (ttx, trx): (Sender<Terrain>, Receiver<Terrain>) = mpsc::channel();
    let running = Arc::new(AtomicBool::new(true));

    let coms = (rrx, tx);

    // let _device = device.clone();
    // let _queue = queue.clone();
    // let _model_manager = model_manager.clone();
    // let _renderer_manager = renderer_manager.clone();
    // let _particles = particles.clone();

    let physics = Physics::new();

    let world = Arc::new(Mutex::new(World::new(
        model_manager.clone(),
        renderer_manager,
        physics,
        particles.clone(),
        device.clone(),
        queue.clone(),
        vk.mem_alloc.clone(),
        vk.desc_alloc.clone(),
        vk.comm_alloc.clone(),
    )));
    {
        let mut world = world.lock();
        world.register::<Renderer>(false, false);
        world.register::<ParticleEmitter>(false, false);
        // world.register::<Maker>(true);
        world.register::<Terrain>(true, true);
        world.register::<Bomb>(true, false);
        world.register::<Player>(true, false);
    }
    let rm = {
        let w = world.lock();
        let s = w.sys.lock();
        let rm = s.renderer_manager.read();
        let rm = rm.shr_data.clone();
        rm
    };

    let game_thread = {
        // let _perf = perf.clone();
        let _running = running.clone();
        let world = world.clone();
        thread::spawn(move || game_thread_fn(world.clone(), (rtx, rx), _running))
    };
    let mut game_thread = vec![game_thread];

    // let ter = trx.recv().unwrap();
    // println!("sending input");
    let _res = coms.1.send(input.clone());
    let mut file_watcher = file_watcher::FileWatcher::new("./test_project_rs");
    {
        let mut world = world.lock();
        load_project(&mut file_watcher, &mut world, assets_manager.clone());
        file_watcher.init(assets_manager.clone());
        // save_project(&file_watcher, &mut world, assets_manager.clone());
    }
    // let mut selected_transforms: HashMap<i32, bool> = HashMap::<i32, bool>::new();

    // let mut selected = None;
    // let _surface = surface.clone();
    event_loop.run(move |event, _, control_flow| {
        // let game_thread = game_thread.clone();
        *control_flow = ControlFlow::Poll;
        match event {
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta } => {
                    if focused {
                        input.mouse_x += delta.0;
                        input.mouse_y += delta.1;
                    }
                }
                _ => (),
            },
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::Focused(foc) => {
                        focused = foc;
                    }
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                        running.store(false, Ordering::SeqCst);
                        let game_thread = game_thread.remove(0);
                        let _res = coms.1.send(input.clone());

                        game_thread.join().unwrap();

                        save_project(&file_watcher, &world.lock(), assets_manager.clone());

                        perf.print();

                        // world.lock().sys.lock().physics.
                    }
                    WindowEvent::MouseInput {
                        device_id: _,
                        state,
                        button,
                        ..
                    } => match state {
                        ElementState::Pressed => {
                            input.mouse_buttons.insert(
                                match button {
                                    MouseButton::Left => 0,
                                    MouseButton::Middle => 1,
                                    MouseButton::Right => 2,
                                    MouseButton::Other(x) => x as u32,
                                },
                                true,
                            );
                        }
                        ElementState::Released => {
                            input.mouse_buttons.insert(
                                match button {
                                    MouseButton::Left => 0,
                                    MouseButton::Middle => 1,
                                    MouseButton::Right => 2,
                                    MouseButton::Other(x) => x as u32,
                                },
                                false,
                            );
                        }
                    },
                    // WindowEvent::AxisMotion { device_id, axis, value } => {
                    //     println!("axis {:#?}: {}", axis, value);
                    // },
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Released,
                                virtual_keycode: Some(key),
                                ..
                            },
                        ..
                    } => {
                        let _ = match key {
                            _ => {
                                input.key_downs.insert(key.clone(), false);
                                input.key_ups.insert(key.clone(), true);
                            }
                        };
                    }
                    WindowEvent::ModifiersChanged(m) => modifiers = m,
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(key),
                                ..
                            },
                        ..
                    } => {
                        let _ = match key {
                            _ => {
                                input.key_presses.insert(key.clone(), true);
                                input.key_downs.insert(key.clone(), true);
                            }
                        };
                    }
                    WindowEvent::Resized(_size) => {
                        recreate_swapchain = true;
                        // if size.width == 0 || size.height == 0 {
                        //     minimized = true;
                        // } else {
                        //     minimized = false;
                        //     app.resized = true;
                        // }
                    }
                    _ => (),
                }

                if !input.get_key(&VirtualKeyCode::Space) {
                    // let egui_consumed_event = egui_winit.on_event(&egui_ctx, &event);
                    gui.update(&event);
                    // if !egui_consumed_event {
                    //     // do your own event handling here
                    // };
                }
            }
            Event::RedrawEventsCleared => {
                puffin::GlobalProfiler::lock().new_frame();

                puffin::profile_scope!("full");
                ////////////////////////////////////

                let full = Instant::now();

                // if input.get_key(&VirtualKeyCode::Escape) {
                //     *control_flow = ControlFlow::Exit;
                //     running.store(false, Ordering::SeqCst);
                //     let game_thread = game_thread.remove(0);
                //     let _res = coms.1.send(input.clone());

                //     game_thread.join().unwrap();

                //     perf.print();
                // }
                static mut GRAB_MODE: bool = true;
                if input.get_key_press(&VirtualKeyCode::G) {
                    unsafe {
                        let _er = surface
                            .object()
                            .unwrap()
                            .downcast_ref::<Window>()
                            .unwrap()
                            .set_cursor_grab(match GRAB_MODE {
                                true => CursorGrabMode::Confined,
                                false => CursorGrabMode::None,
                            });
                        GRAB_MODE = !GRAB_MODE;
                    }
                }
                if input.get_key_press(&VirtualKeyCode::J) {
                    lock_cull = !lock_cull;
                    // lock_cull.
                }

                if input.get_key(&VirtualKeyCode::H) {
                    surface
                        .object()
                        .unwrap()
                        .downcast_ref::<Window>()
                        .unwrap()
                        .set_cursor_visible(modifiers.shift());
                }

                let (transform_data, cam_pos, cam_rot, rd, emitter_inits) = {
                    puffin::profile_scope!("wait for game");
                    let inst = Instant::now();
                    let (transform_data, cam_pos, cam_rot, renderer_data, emitter_inits) =
                        coms.0.recv().unwrap();

                    perf.update("wait for game".into(), Instant::now() - inst);
                    (
                        transform_data,
                        cam_pos,
                        cam_rot,
                        renderer_data,
                        emitter_inits,
                    )
                };

                file_watcher.get_updates(assets_manager.clone());

                let inst = Instant::now();
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    editor_ui::editor_ui(
                        &world,
                        &mut fps_queue,
                        &ctx,
                        fc.clone(),
                        assets_manager.clone(),
                    );
                });
                perf.update("gui".into(), Instant::now() - inst);

                let render_jobs = world.lock().render();

                // let rm = renderer_manager.read();
                let mut rm = rm.write();

                //
                let res = coms.1.send(input.clone());
                if res.is_err() {
                    return;
                    // println!("ohno");
                }

                input.key_presses.clear();
                input.key_ups.clear();
                input.mouse_x = 0.;
                input.mouse_y = 0.;

                /////////////////////////////////////////////////////////////////
                let inst = Instant::now();

                let (position_update_data, rotation_update_data, scale_update_data) = {
                    puffin::profile_scope!("buffer transform data");
                    let position_update_data = transform_compute.get_position_update_data(
                        device.clone(),
                        transform_data.clone(),
                        vk.mem_alloc.clone(),
                        &vk.comm_alloc,
                    );

                    let rotation_update_data = transform_compute.get_rotation_update_data(
                        device.clone(),
                        transform_data.clone(),
                        vk.mem_alloc.clone(),
                        &vk.comm_alloc,
                    );

                    let scale_update_data = transform_compute.get_scale_update_data(
                        device.clone(),
                        transform_data.clone(),
                        vk.mem_alloc.clone(),
                        &vk.comm_alloc,
                    );
                    (
                        position_update_data,
                        rotation_update_data,
                        scale_update_data,
                    )
                };
                perf.update("write to buffer".into(), Instant::now() - inst);

                // render_thread = thread::spawn( || {
                let inst = Instant::now();

                let dimensions = surface
                    .object()
                    .unwrap()
                    .downcast_ref::<Window>()
                    .unwrap()
                    .inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                if recreate_swapchain {
                    println!("recreate swapchain");
                    println!("dimensions {}: {}", dimensions.width, dimensions.height);
                    let dimensions: [u32; 2] = surface
                        .object()
                        .unwrap()
                        .downcast_ref::<Window>()
                        .unwrap()
                        .inner_size()
                        .into();

                    let (new_swapchain, new_images) =
                        match swapchain.recreate(SwapchainCreateInfo {
                            image_extent: dimensions.into(),
                            ..swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    swapchain = new_swapchain;

                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        // device.clone(),
                        render_pass.clone(),
                        &mut viewport,
                        vk.mem_alloc.clone(),
                        &mut frame_color,
                    );
                    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
                    recreate_swapchain = false;
                }

                let aspect_ratio = // *editor_ui::EDITOR_ASPECT_RATIO.lock();
                    swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;
                let proj = glm::perspective(
                    aspect_ratio,
                    std::f32::consts::FRAC_PI_2 as f32,
                    0.01,
                    10000.0,
                );

                let rot = glm::quat_to_mat3(&cam_rot);
                let target = cam_pos + rot * Vec3::z();
                let up = rot * Vec3::y();
                let view: Mat4 = glm::look_at_lh(&cam_pos, &target, &up);

                let (image_num, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None) {
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
                    queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                builder
                    .copy_image(CopyImageInfo::images(
                        frame_color.arc.clone(),
                        frame_img.clone(),
                    ))
                    // .blit_image(BlitImageInfo::images(
                    //     frame_color.arc.clone(),
                    //     frame_img.clone(),
                    // ))
                    .unwrap();

                // Get the shapes from egui
                // let egui_output = egui_ctx.end_frame();
                // let platform_output = egui_output.platform_output;
                // egui_winit.handle_platform_output(window, &egui_ctx, platform_output);

                // gui
                // let _result = egui_painter
                //     .update_textures(egui_output.textures_delta, &mut builder)
                //     .expect("egui texture error");

                {
                    puffin::profile_scope!("transform update compute");

                    // compute shader transforms
                    transform_compute.update(
                        device.clone(),
                        &mut builder,
                        transform_data.0,
                        vk.mem_alloc.clone(),
                        &vk.comm_alloc,
                    );
                    // {
                    //     puffin::profile_scope!("update transforms");
                    // stage 0
                    transform_compute.update_positions(
                        &mut builder,
                        view.clone(),
                        proj.clone(),
                        &transform_uniforms,
                        compute_pipeline.clone(),
                        position_update_data,
                        vk.mem_alloc.clone(),
                        &vk.comm_alloc,
                        vk.desc_alloc.clone(),
                    );

                    // stage 1
                    transform_compute.update_rotations(
                        &mut builder,
                        view.clone(),
                        proj.clone(),
                        &transform_uniforms,
                        compute_pipeline.clone(),
                        rotation_update_data,
                        vk.mem_alloc.clone(),
                        &vk.comm_alloc,
                        vk.desc_alloc.clone(),
                    );

                    // stage 2
                    transform_compute.update_scales(
                        &mut builder,
                        view.clone(),
                        proj.clone(),
                        &transform_uniforms,
                        compute_pipeline.clone(),
                        scale_update_data,
                        vk.mem_alloc.clone(),
                        &vk.comm_alloc,
                        vk.desc_alloc.clone(),
                    );

                    // stage 3
                    transform_compute.update_mvp(
                        &mut builder,
                        device.clone(),
                        view.clone(),
                        proj.clone(),
                        &transform_uniforms,
                        compute_pipeline.clone(),
                        transform_data.0 as i32,
                        vk.mem_alloc.clone(),
                        &vk.comm_alloc,
                        vk.desc_alloc.clone(),
                    );
                }

                // compute shader renderers
                {
                    puffin::profile_scope!("process renderers");
                    let renderer_pipeline = rm.pipeline.clone();
                    // let renderers = &mut renderer_manager.renderers.write();

                    builder.bind_pipeline_compute(renderer_pipeline.clone());

                    if !lock_cull {
                        cull_view = view.clone();
                    }

                    // for (_, i) in renderers.iter_mut() {
                    if rm.transform_ids_gpu.len() < rd.transforms_len as u64 {
                        let len = rd.transforms_len;
                        let max_len = (len as f32 + 1.).log2().ceil();
                        let max_len = (2 as u32).pow(max_len as u32);

                        let copy_buffer = rm.transform_ids_gpu.clone();
                        unsafe {
                            // vulkano::buffer::CpuAccessibleBuffer::uninitialized_array(vk.mem_alloc, len, usage, host_cached)
                            rm.transform_ids_gpu = CpuAccessibleBuffer::uninitialized_array(
                                &vk.mem_alloc,
                                // device.clone(),
                                max_len as u64,
                                buffer_usage_all(),
                                false,
                            )
                            .unwrap();
                            rm.renderers_gpu = CpuAccessibleBuffer::uninitialized_array(
                                // vk.mem_alloc.clone(),
                                &vk.mem_alloc,
                                // device.clone(),
                                max_len as u64,
                                buffer_usage_all(),
                                false,
                            )
                            .unwrap();
                        }

                        // let copy = CopyBufferInfo::buffers(copy_buffer, rm.transform_ids_gpu.clone());
                        builder
                            .copy_buffer(CopyBufferInfo::buffers(
                                copy_buffer,
                                rm.transform_ids_gpu.clone(),
                            ))
                            .unwrap();
                    }
                    if rm.indirect.data.len() > 0 {
                        rm.indirect_buffer = CpuAccessibleBuffer::from_iter(
                            &vk.mem_alloc,
                            buffer_usage_all(),
                            false,
                            rm.indirect.data.clone(),
                        )
                        .unwrap();
                    }

                    let mut offset_vec = Vec::new();
                    let mut offset = 0;
                    for (_, m_id) in rd.indirect_model.iter() {
                        offset_vec.push(offset);
                        if let Some(ind) = rd.model_indirect.get(&m_id) {
                            offset += ind.count;
                        }
                    }
                    if offset_vec.len() > 0 {
                        let offsets_buffer = CpuAccessibleBuffer::from_iter(
                            &vk.mem_alloc,
                            buffer_usage_all(),
                            false,
                            offset_vec,
                        )
                        .unwrap();

                        {
                            puffin::profile_scope!("update renderers: stage 0");
                            let update_num = rd.updates.len() / 3;
                            if update_num > 0 {
                                rm.updates_gpu = CpuAccessibleBuffer::from_iter(
                                    &vk.mem_alloc,
                                    buffer_usage_all(),
                                    false,
                                    rd.updates,
                                )
                                .unwrap();
                            }

                            // stage 0
                            let uniforms = rm
                                .uniform
                                .from_data(ur::ty::Data {
                                    num_jobs: update_num as i32,
                                    stage: 0,
                                    view: cull_view.into(),
                                    _dummy0: Default::default(),
                                })
                                .unwrap();

                            let update_renderers_set = PersistentDescriptorSet::new(
                                &vk.desc_alloc,
                                renderer_pipeline
                                    .layout()
                                    .set_layouts()
                                    .get(0) // 0 is the index of the descriptor set.
                                    .unwrap()
                                    .clone(),
                                [
                                    // 0 is the binding of the data in this set. We bind the `DeviceLocalBuffer` of vertices here.
                                    WriteDescriptorSet::buffer(0, rm.updates_gpu.clone()),
                                    WriteDescriptorSet::buffer(1, rm.transform_ids_gpu.clone()),
                                    WriteDescriptorSet::buffer(2, rm.renderers_gpu.clone()),
                                    WriteDescriptorSet::buffer(3, rm.indirect_buffer.clone()),
                                    WriteDescriptorSet::buffer(
                                        4,
                                        transform_compute.transform.clone(),
                                    ),
                                    WriteDescriptorSet::buffer(5, offsets_buffer.clone()),
                                    WriteDescriptorSet::buffer(6, uniforms.clone()),
                                ],
                            )
                            .unwrap();

                            builder
                                .bind_descriptor_sets(
                                    PipelineBindPoint::Compute,
                                    renderer_pipeline.layout().clone(),
                                    0, // Bind this descriptor set to index 0.
                                    update_renderers_set.clone(),
                                )
                                .dispatch([update_num as u32 / 128 + 1, 1, 1])
                                .unwrap();
                        }
                        {
                            puffin::profile_scope!("update renderers: stage 1");
                            // stage 1
                            let uniforms = {
                                puffin::profile_scope!("update renderers: stage 1: uniform data");
                                rm.uniform
                                    .from_data(ur::ty::Data {
                                        num_jobs: rd.transforms_len as i32,
                                        stage: 1,
                                        view: cull_view.into(),
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
                                        WriteDescriptorSet::buffer(
                                            4,
                                            transform_compute.transform.clone(),
                                        ),
                                        WriteDescriptorSet::buffer(5, offsets_buffer.clone()),
                                        WriteDescriptorSet::buffer(6, uniforms.clone()),
                                    ],
                                )
                                .unwrap()
                            };
                            {
                                puffin::profile_scope!(
                                    "update renderers: stage 1: bind pipeline/dispatch"
                                );
                                builder
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
                }
                // let particles = particles.read();
                particles.emitter_init(
                    device.clone(),
                    &mut builder,
                    transform_compute.transform.clone(),
                    emitter_inits.1.clone(),
                    emitter_inits.0,
                    input.time.dt,
                    input.time.time,
                    cam_pos.into(),
                    cam_rot.coords.into(),
                    vk.mem_alloc.clone(),
                    &vk.comm_alloc,
                    vk.desc_alloc.clone(),
                );
                particles.emitter_update(
                    device.clone(),
                    &mut builder,
                    transform_compute.transform.clone(),
                    emitter_inits.0,
                    input.time.dt,
                    input.time.time,
                    cam_pos.into(),
                    cam_rot.coords.into(),
                    vk.mem_alloc.clone(),
                    &vk.comm_alloc,
                    vk.desc_alloc.clone(),
                );
                particles.particle_update(
                    device.clone(),
                    &mut builder,
                    transform_compute.transform.clone(),
                    input.time.dt,
                    input.time.time,
                    cam_pos.into(),
                    cam_rot.coords.into(),
                    vk.mem_alloc.clone(),
                    &vk.comm_alloc,
                    vk.desc_alloc.clone(),
                );
                particles.sort.sort(
                    view.into(),
                    proj.into(),
                    transform_compute.transform.clone(),
                    &particles.particle_buffers,
                    device.clone(),
                    queue.clone(),
                    &mut builder,
                    &vk.desc_alloc,
                );

                {
                    puffin::profile_scope!("render meshes");

                    builder
                        .begin_render_pass(
                            RenderPassBeginInfo {
                                clear_values: vec![
                                    Some([0., 0., 0., 1.].into()),
                                    Some([0., 0., 0., 1.].into()),
                                    Some(1f32.into()),
                                ],
                                ..RenderPassBeginInfo::framebuffer(
                                    framebuffers[image_num as usize].clone(),
                                )
                            },
                            SubpassContents::Inline,
                        )
                        .unwrap()
                        .set_viewport(0, [viewport.clone()]);

                    rend.bind_pipeline(&mut builder);

                    {
                        let mm = model_manager.lock();

                        let mut offset = 0;

                        for (_ind_id, m_id) in rd.indirect_model.iter() {
                            if let Some(ind) = rd.model_indirect.get(&m_id) {
                                if let Some(mr) = mm.assets_id.get(&m_id) {
                                    let mr = mr.lock();
                                    if ind.count == 0 {
                                        continue;
                                    }
                                    if let Some(indirect_buffer) =
                                        BufferSlice::from_typed_buffer_access(
                                            rm.indirect_buffer.clone(),
                                        )
                                        .slice(ind.id as u64..(ind.id + 1) as u64)
                                    {
                                        // println!("{}",indirect_buffer.len());
                                        if let Some(renderer_buffer) =
                                            BufferSlice::from_typed_buffer_access(
                                                rm.renderers_gpu.clone(),
                                            )
                                            .slice(offset..(offset + ind.count as u64) as u64)
                                        {
                                            // println!("{}",renderer_buffer.len());
                                            rend.bind_mesh(
                                                &texture_manager.lock(),
                                                &mut builder,
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
                        builder: &mut builder,
                        transforms: transform_compute.transform.clone(),
                        mvp: transform_compute.mvp.clone(),
                        view: &view,
                        proj: &proj,
                        pipeline: &rend,
                        viewport: &viewport,
                        texture_manager: &texture_manager,
                        vk: vk.clone(),
                    };
                    for job in render_jobs {
                        job(&mut rjd);
                    }

                    // builder
                    //     .next_subpass(SubpassContents::SecondaryCommandBuffers)
                    //     .unwrap();
                }
                // let particles = particles.read();
                particles.render_particles(
                    &mut builder,
                    view.clone(),
                    proj.clone(),
                    cam_rot.coords.into(),
                    cam_pos.into(),
                    transform_compute.transform.clone(),
                    vk.mem_alloc.clone(),
                    &vk.comm_alloc,
                    vk.desc_alloc.clone(),
                );

                // Automatically start the next render subpass and draw the gui
                let size = surface
                    .object()
                    .unwrap()
                    .downcast_ref::<Window>()
                    .unwrap()
                    .inner_size();

                builder.set_viewport(0, [viewport.clone()]);
                // gui.draw_on_image(previous_frame_end, images[image_num as usize].clone());
                builder
                    .next_subpass(SubpassContents::SecondaryCommandBuffers)
                    .unwrap();
                let gui_commands = gui.draw_on_subpass_image([size.width, size.height]);
                builder.execute_commands(gui_commands).unwrap();
                // egui_painter
                //     .draw(
                //         &mut builder,
                //         [(size.width as f32) / sf, (size.height as f32) / sf],
                //         &egui_ctx,
                //         egui_output.shapes,
                //     )
                //     .unwrap();

                input.time.time += input.time.dt;
                input.time.dt = frame_time.elapsed().as_secs_f64() as f32;

                fps_queue.push_back(input.time.dt);
                if fps_queue.len() > 100 {
                    fps_queue.pop_front();
                }

                // egui_bench.push(frame_time.elapsed().as_secs_f64(), positions.len());
                frame_time = Instant::now();

                builder.end_render_pass().unwrap();
                // builder
                // };

                // let command_buffer = {
                //     puffin::profile_scope!("builder build");
                let command_buffer = builder.build().unwrap();
                // };

                // {
                //     puffin::profile_scope!("execute command buffer");
                let execute = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer);

                match execute {
                    Ok(execute) => {
                        let future = execute
                            .then_swapchain_present(
                                queue.clone(),
                                SwapchainPresentInfo::swapchain_image_index(
                                    swapchain.clone(),
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
                                previous_frame_end = Some(sync::now(device.clone()).boxed());
                            }
                            Err(e) => {
                                println!("Failed to flush future: {:?}", e);
                                previous_frame_end = Some(sync::now(device.clone()).boxed());
                            }
                        }
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                };

                // }
                perf.update("render".into(), Instant::now() - inst);
                perf.update("full".into(), Instant::now() - full);

                if first_frame {
                    puffin::set_scopes_on(true);
                    first_frame = false;
                }

                // });
            }
            _ => (),
        }
    });
}

// /// This method is called once during initialization, then again whenever the window is resized
// fn _window_size_dependent_setup(
//     device: Arc<Device>,
//     images: &[Arc<SwapchainImage<Window>>],
//     render_pass: Arc<RenderPass>,
// ) -> Vec<Arc<Framebuffer>> {
//     let dimensions = images[0].dimensions().width_height();

//     let depth_buffer = ImageView::new_default(
//         AttachmentImage::transient(device.clone(), dimensions, Format::D32_SFLOAT).unwrap(),
//     )
//     .unwrap();

//     let framebuffers = images
//         .iter()
//         .map(|image| {
//             let view = ImageView::new_default(image.clone()).unwrap();
//             Framebuffer::new(
//                 render_pass.clone(),
//                 FramebufferCreateInfo {
//                     attachments: vec![view, depth_buffer.clone()],
//                     ..Default::default()
//                 },
//             )
//             .unwrap()
//         })
//         .collect::<Vec<_>>();

//     framebuffers
// }

// fn window_size_dependent_setup(
//     images: &[Arc<SwapchainImage>],
//     device: Arc<Device>,
//     render_pass: Arc<RenderPass>,
//     viewport: &mut Viewport,
// ) -> Vec<Arc<Framebuffer>> {
//     let dimensions = images[0].dimensions().width_height();
//     viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

//     let depth_buffer = ImageView::new_default(
//         AttachmentImage::transient(device.clone(), dimensions, Format::D32_SFLOAT).unwrap(),
//     )
//     .unwrap();

//     images
//         .iter()
//         .map(|image| {
//             let view = ImageView::new_default(image.clone()).unwrap();
//             Framebuffer::new(
//                 render_pass.clone(),
//                 FramebufferCreateInfo {
//                     attachments: vec![view, depth_buffer.clone()],
//                     ..Default::default()
//                 },
//             )
//             .unwrap()
//         })
//         .collect::<Vec<_>>()
// }

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
