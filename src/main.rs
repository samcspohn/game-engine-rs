#![allow(warnings)]

use camera::{Camera, CameraData};
use crossbeam::queue::SegQueue;
// use egui::plot::{HLine, Line, Plot, Value, Values};
use egui::TextureId;
use engine::input::Input;
use engine::project::asset_manager::{AssetManagerBase, AssetsManager};
use engine::project::{file_watcher, Project};
use engine::world::World;
use game::RenderingData;

use glm::{vec4, Vec3};
use nalgebra_glm as glm;
use notify::{RecursiveMode, Watcher};
use parking_lot::Mutex;
use puffin_egui::*;
use std::{
    collections::{BTreeMap, HashMap},
    env, fs,
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc,
        mpsc::{Receiver, Sender},
        Arc,
    },
    thread::{self},
    time::{Duration, Instant},
};
use vulkan_manager::VulkanManager;
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

mod camera;
mod color_gradient;
mod editor;
mod engine;
mod game;
mod model;
mod particle_sort;
mod particles;
mod perf;
mod physics;
mod render_pipeline;
mod renderer;
mod renderer_component;
mod runtime_compilation;
mod texture;
mod time;
mod transform_compute;
mod vulkan_manager;

use crate::editor::editor_ui::EDITOR_ASPECT_RATIO;
use crate::engine::project::{load_project, save_project};
use crate::game::game_thread_fn;
use crate::model::ModelManager;
use crate::particles::ParticleEmitter;
use crate::perf::Perf;
use crate::renderer_component::{buffer_usage_all, Renderer};
use crate::texture::TextureManager;
use crate::transform_compute::cs;

#[cfg(target_os = "windows")]
mod win_alloc {
    use mimalloc::MiMalloc;

    #[global_allocator]
    static GLOBAL: MiMalloc = MiMalloc;
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    if thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Max).is_ok() {
        println!("Set main thread priority");
    }
    println!("main thread id: {:?}", thread::current().id());
    println!(
        "main thread priority: {:?}",
        thread_priority::get_current_thread_priority().ok().unwrap()
    );

    let event_loop = EventLoop::new();
    let vk = VulkanManager::new(&event_loop);
    let render_pass = vulkano::single_pass_renderpass!(
        vk.device.clone(),
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

    let assets_manager = Arc::new(AssetsManager::new());

    let texture_manager = Arc::new(Mutex::new(TextureManager::new(
        (vk.device.clone(), vk.queue.clone(), vk.mem_alloc.clone()),
        &["png", "jpeg"],
    )));
    let model_manager = Arc::new(Mutex::new(ModelManager::new(
        (
            vk.device.clone(),
            texture_manager.clone(),
            vk.mem_alloc.clone(),
        ),
        &["obj"],
    )));
    let rs_manager = Arc::new(Mutex::new(runtime_compilation::RSManager::new((), &["rs"])));

    model_manager.lock().from_file("src/cube/cube.obj");

    let particles_system = Arc::new(particles::ParticleCompute::new(
        vk.device.clone(),
        vk.clone(),
    ));

    let world = Arc::new(Mutex::new(World::new(
        particles_system.clone(),
        vk.clone(),
        assets_manager.clone(),
    )));

    let path = "./test_project_rs/runtime";
    if let Ok(_) = fs::remove_dir_all(path) {}
    fs::create_dir(path).unwrap();

    #[cfg(target_os = "windows")]
    let dylib_ext = ["dll"];
    #[cfg(not(target_os = "windows"))]
    let dylib_ext = ["so"];

    let lib_manager = Arc::new(Mutex::new(runtime_compilation::LibManager::new(
        world.clone(),
        &dylib_ext,
    )));
    unsafe {
        assets_manager.add_asset_manager("rs", rs_manager.clone());
        assets_manager.add_asset_manager("lib", lib_manager.clone());
        assets_manager.add_asset_manager("model", model_manager.clone());
        assets_manager.add_asset_manager("texture", texture_manager.clone());
        assets_manager.add_asset_manager(
            "particle_template",
            particles_system.particle_template_manager.clone(),
        );
    }

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

    //////////////////////////////////////////////////

    let mut input = Input {
        ..Default::default()
    };

    let mut perf = Perf {
        data: BTreeMap::<String, SegQueue<Duration>>::new(),
    };

    let _loops = 0;

    let mut gui = egui_winit_vulkano::Gui::new_with_subpass(
        &event_loop,
        vk.surface.clone(),
        Some(vk.swapchain().image_format()),
        vk.queue.clone(),
        Subpass::from(render_pass.clone(), 0).unwrap(),
    );

    let mut fc_map: HashMap<i32, HashMap<u32, TextureId>> = HashMap::new();

    let mut frame_time = Instant::now();
    // let mut fps_time: f32 = 0.0;

    let mut transform_compute = transform_compute::transform_buffer_init(
        vk.device.clone(),
        // vk.queue.clone(),
        vec![
            crate::cs::ty::transform {
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
        vk.images.len() as u32,
    );

    let cs = cs::load(vk.device.clone()).unwrap();

    // Create compute-pipeline for applying compute shader to vertices.
    let compute_pipeline = vulkano::pipeline::ComputePipeline::new(
        vk.device.clone(),
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

    let mut focused = true;

    // puffin::set_scopes_on(true);

    let _cull_view = glm::Mat4::identity();
    let mut lock_cull = false;
    let mut first_frame = true;

    /////////////////////////////////////////////////////////////////////////////////////////
    let (tx, rx): (Sender<_>, Receiver<_>) = mpsc::channel();
    let (rtx, rrx): (Sender<_>, Receiver<_>) = mpsc::channel();
    let running = Arc::new(AtomicBool::new(true));

    let coms = (rrx, tx);

    {
        let mut world = world.lock();
        world.register::<Renderer>(false, false, false);
        world.register::<ParticleEmitter>(false, false, false);
        world.register::<Camera>(false, false, false);
        // world.register::<Terrain>(true, false, true);
    };

    let rm = {
        let w = world.lock();
        let rm = w.sys.renderer_manager.read();
        rm.shr_data.clone()
    };

    let game_thread = {
        let _running = running.clone();
        let world = world.clone();
        thread::spawn(move || game_thread_fn(world.clone(), (rtx, rx), _running))
    };
    let mut game_thread = vec![game_thread];

    let _res = coms.1.send((input.clone(), false));
    let mut file_watcher = file_watcher::FileWatcher::new("./test_project_rs");

    if let Ok(s) = std::fs::read_to_string("project.yaml") {
        {
            let project: Project = serde_yaml::from_str(s.as_str()).unwrap();
            file_watcher.files = project.files;
            assets_manager.deserialize(project.assets);
            file_watcher.init(assets_manager.clone());
        }
        // serialize::deserialize(&mut world.lock());
    }

    let mut cam_data = CameraData::new(vk.clone());
    let mut playing_game = false;

    let mut editor_cam = editor::editor_cam::EditorCam {
        rot: glm::quat(-1., 0., 0., 0.),
        pos: Vec3::default(),
        speed: 30f32,
    };
    event_loop.run(move |event, _, control_flow| {
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
                        println!("main event_loop id: {:?}", thread::current().id());
                    }
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
                        running.store(false, Ordering::SeqCst);
                        let game_thread = game_thread.remove(0);
                        let _res = coms.1.send((input.clone(), playing_game));
                        game_thread.join().unwrap();
                        save_project(&file_watcher, &world.lock(), assets_manager.clone());
                        perf.print();
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
                    WindowEvent::KeyboardInput { input: x, .. } => {
                        let _ = match x {
                            KeyboardInput {
                                state: ElementState::Released,
                                virtual_keycode: Some(key),
                                ..
                            } => {
                                input.key_downs.insert(key, false);
                                input.key_ups.insert(key, true);
                            }
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(key),
                                ..
                            } => {
                                input.key_presses.insert(key, true);
                                input.key_downs.insert(key, true);
                            }
                            _ => {}
                        };
                    }
                    WindowEvent::ModifiersChanged(m) => modifiers = m,
                    WindowEvent::Resized(_size) => {
                        recreate_swapchain = true;
                    }
                    _ => (),
                }

                if !input.get_key(&VirtualKeyCode::Space) {
                    gui.update(&event);
                }
            }
            Event::RedrawEventsCleared => {
                puffin::GlobalProfiler::lock().new_frame();

                puffin::profile_scope!("full");
                ////////////////////////////////////

                let full = Instant::now();
                let RenderingData {
                    transform_data,
                    cam_datas,
                    main_cam_id,
                    renderer_data: mut rd,
                    emitter_inits,
                    gpu_work,
                } = {
                    puffin::profile_scope!("wait for game");
                    let inst = Instant::now();
                    let rd = coms.0.recv().unwrap();

                    perf.update("wait for game".into(), Instant::now() - inst);
                    rd
                };

                if !playing_game {
                    editor_cam.update(&input);
                }

                file_watcher.get_updates(assets_manager.clone());

                let dimensions = vk
                    .surface
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
                    let dimensions: [u32; 2] = vk
                        .surface
                        .object()
                        .unwrap()
                        .downcast_ref::<Window>()
                        .unwrap()
                        .inner_size()
                        .into();

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
                let cam_num = if playing_game { main_cam_id } else { -1 };
                let fc = fc_map
                    .entry(cam_num)
                    .or_insert(HashMap::<u32, TextureId>::new())
                    .entry(image_num)
                    .or_insert_with(|| {
                        let frame_image_view = ImageView::new_default(if playing_game {
                            cam_datas[0].lock().output[image_num as usize].clone()
                        } else {
                            cam_data.output[image_num as usize].clone()
                        })
                        .unwrap();

                        gui.register_user_image_view(frame_image_view.clone())
                    });

                if suboptimal {
                    recreate_swapchain = true;
                }

                let inst = Instant::now();
                let dimensions = *EDITOR_ASPECT_RATIO.lock();
                let mut _playing_game = false;
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    _playing_game = editor::editor_ui::editor_ui(
                        &world,
                        &mut fps_queue,
                        &ctx,
                        *fc,
                        assets_manager.clone(),
                    );
                });
                {
                    let ear = EDITOR_ASPECT_RATIO.lock();
                    if dimensions != *ear {
                        cam_data.resize(*ear, vk.clone());
                        fc_map.clear();
                    }
                }

                perf.update("gui".into(), Instant::now() - inst);

                let render_jobs = world.lock().render();

                // let rm = renderer_manager.read();
                let mut rm = rm.write();

                // start new game frame
                let res = coms.1.send((input.clone(), _playing_game));
                if res.is_err() {
                    return;
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
                        &transform_data,
                        image_num,
                        vk.mem_alloc.clone(),
                    );

                    let rotation_update_data = transform_compute.get_rotation_update_data(
                        &transform_data,
                        image_num,
                        vk.mem_alloc.clone(),
                    );

                    let scale_update_data = transform_compute.get_scale_update_data(
                        &transform_data,
                        image_num,
                        vk.mem_alloc.clone(),
                    );
                    (
                        position_update_data,
                        rotation_update_data,
                        scale_update_data,
                    )
                };
                perf.update("write to buffer".into(), Instant::now() - inst);

                // render_thread = thread::spawn( || {

                let mut builder = AutoCommandBufferBuilder::primary(
                    &vk.comm_alloc,
                    vk.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                let inst = Instant::now();

                {
                    puffin::profile_scope!("transform update compute");

                    // compute shader transforms
                    transform_compute.update(
                        vk.device.clone(),
                        &mut builder,
                        transform_data.extent,
                        vk.mem_alloc.clone(),
                        &vk.comm_alloc,
                    );
                    // {
                    //     puffin::profile_scope!("update transforms");
                    // stage 0
                    transform_compute.update_positions(
                        &mut builder,
                        &transform_uniforms,
                        compute_pipeline.clone(),
                        position_update_data,
                        vk.mem_alloc.clone(),
                        &vk.comm_alloc,
                        vk.desc_alloc.clone(),
                        image_num,
                    );

                    // stage 1
                    transform_compute.update_rotations(
                        &mut builder,
                        &transform_uniforms,
                        compute_pipeline.clone(),
                        rotation_update_data,
                        vk.desc_alloc.clone(),
                        image_num,
                    );

                    // stage 2
                    transform_compute.update_scales(
                        &mut builder,
                        &transform_uniforms,
                        compute_pipeline.clone(),
                        scale_update_data,
                        vk.desc_alloc.clone(),
                        image_num,
                    );
                }
                perf.update("transform update".into(), inst.elapsed());
                let inst = Instant::now();
                {
                    particles_system.update(
                        &mut builder,
                        emitter_inits,
                        transform_compute.transform.clone(),
                        input.time.dt,
                        input.time.time,
                        editor_cam.pos.into(),
                        editor_cam.rot.coords.into(),
                    );
                }
                perf.update("particle update".into(), inst.elapsed());

                let inst = Instant::now();
                // compute shader renderers
                let offset_vec = {
                    puffin::profile_scope!("process renderers");
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
                        &transform_compute,
                    )
                };
                perf.update("update renderers".into(), inst.elapsed());
                let inst = Instant::now();

                while let Some(job) = gpu_work.pop() {
                    job.unwrap()(&mut builder);
                }
                // let a = Arc::try_unwrap(gpu_work).ok().unwrap();
                // let a = a.into_inner().unwrap();
                // let b = a.build().unwrap();
                // builder.execute_commands(b).unwrap();

                if !playing_game {
                    cam_data.update(editor_cam.pos, editor_cam.rot, 0.01f32, 10_000f32, 70f32);
                    cam_data.render(
                        vk.clone(),
                        &mut builder,
                        &mut transform_compute,
                        particles_system.clone(),
                        &transform_uniforms,
                        &transform_data,
                        compute_pipeline.clone(),
                        rm.pipeline.clone(),
                        offset_vec,
                        &mut rm,
                        &mut rd,
                        image_num,
                        &model_manager.lock(),
                        &texture_manager,
                        &render_jobs,
                    );
                } else {
                    for cam in cam_datas {
                        cam.lock().render(
                            vk.clone(),
                            &mut builder,
                            &mut transform_compute,
                            particles_system.clone(),
                            &transform_uniforms,
                            &transform_data,
                            compute_pipeline.clone(),
                            rm.pipeline.clone(),
                            offset_vec.clone(),
                            &mut rm,
                            &mut rd,
                            image_num,
                            &model_manager.lock(),
                            &texture_manager,
                            &render_jobs,
                        );
                    }
                }
                perf.update("render camera(s)".into(), inst.elapsed());
                let inst = Instant::now();

                {
                    puffin::profile_scope!("render meshes");

                    builder
                        .begin_render_pass(
                            RenderPassBeginInfo {
                                clear_values: vec![
                                    Some([0., 0., 0., 1.].into()),
                                    // Some([0., 0., 0., 1.].into()),
                                    // Some(1f32.into()),
                                ],
                                ..RenderPassBeginInfo::framebuffer(
                                    framebuffers[image_num as usize].clone(),
                                )
                            },
                            SubpassContents::SecondaryCommandBuffers,
                        )
                        .unwrap()
                        .set_viewport(0, [viewport.clone()]);
                }

                // Automatically start the next render subpass and draw the gui
                let size = vk
                    .surface
                    .object()
                    .unwrap()
                    .downcast_ref::<Window>()
                    .unwrap()
                    .inner_size();

                let gui_commands = gui.draw_on_subpass_image([size.width, size.height]);
                builder.execute_commands(gui_commands).unwrap();

                input.time.time += input.time.dt;
                input.time.dt = frame_time.elapsed().as_secs_f64() as f32;

                fps_queue.push_back(input.time.dt);
                if fps_queue.len() > 100 {
                    fps_queue.pop_front();
                }

                // egui_bench.push(frame_time.elapsed().as_secs_f64(), positions.len());
                frame_time = Instant::now();

                builder.end_render_pass().unwrap();

                let command_buffer = builder.build().unwrap();
                let execute = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(vk.queue.clone(), command_buffer);

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
                    // recreate_swapchain = true;
                }
                playing_game = _playing_game;

                perf.update("render".into(), Instant::now() - inst);
                perf.update("full".into(), Instant::now() - full);

                if first_frame {
                    puffin::set_scopes_on(true);
                    first_frame = false;
                }
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
