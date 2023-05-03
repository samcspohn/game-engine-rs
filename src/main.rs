use camera::{Camera, CameraData};
use crossbeam::queue::SegQueue;
// use egui::plot::{HLine, Line, Plot, Value, Values};
use egui::{TextureId};



use std::{env};
use vulkan_manager::VulkanManager;

use vulkano::command_buffer::{
    RenderPassBeginInfo,
};




use vulkano::memory::allocator::{MemoryUsage};
use vulkano::swapchain::SwapchainPresentInfo;

use winit::window::CursorGrabMode;
// use egui_dock::Tree;
use puffin_egui::*;

use nalgebra_glm as glm;
use parking_lot::{Mutex, RwLock};
use vulkano::buffer::{TypedBufferAccess};






use winit::event::MouseButton;


use std::collections::{BTreeMap};

use std::path::{Path};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self},
    time::{Duration, Instant},
};
use vulkano::{
    buffer::{CpuBufferPool},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    image::{view::ImageView, AttachmentImage, ImageAccess, SwapchainImage},
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    swapchain::{
        acquire_next_image, AcquireError, SwapchainCreateInfo, SwapchainCreationError,
    },
    sync::{self, FlushError, GpuFuture},
};

use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, ModifiersState, VirtualKeyCode,
        WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    window::{Window},
};

use glm::{vec4, Vec3};

use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
// use rust_test::{INDICES, NORMALS, VERTICES};
use notify::{RecursiveMode, Watcher};


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
mod render_pipeline;
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
use crate::editor_ui::{EDITOR_ASPECT_RATIO, PLAYING_GAME};
use crate::engine::physics::Physics;

use crate::engine::{World};
use crate::game::{game_thread_fn, Bomb, Player};

use crate::model::{ModelManager};
use crate::particles::ParticleEmitter;
use crate::perf::Perf;

use crate::project::{load_project, save_project};
use crate::renderer_component2::{buffer_usage_all, Renderer, RendererManager};
use crate::terrain::Terrain;
use crate::texture::TextureManager;
use crate::transform_compute::cs;
use crate::transform_compute::cs::ty::transform;

use crate::{input::Input};

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
    let render_pass = vulkano::single_pass_renderpass!(
        vk.device.clone(),
        attachments: {
            final_color: {
                load: Clear,
                store: Store,
                format: vk.swapchain.lock().image_format(),
                samples: 1,
            }
        },
        pass: { color: [final_color], depth_stencil: {}} // Create a second renderpass to draw egui
    )
    .unwrap();

    // let window = vk.surface.object().unwrap().downcast_ref::<Window>().unwrap();

    let texture_manager = Arc::new(Mutex::new(TextureManager::new((
        vk.device.clone(),
        vk.queue.clone(),
        vk.mem_alloc.clone(),
    ))));

    let model_manager = ModelManager::new((
        vk.device.clone(),
        texture_manager.clone(),
        vk.mem_alloc.clone(),
    ));

    let renderer_manager = Arc::new(RwLock::new(RendererManager::new(
        vk.device.clone(),
        vk.mem_alloc.clone(),
    )));
    // let cube_mesh = Mesh::load_model("src/cube/cube.obj", vk.device.clone(), texture_manager.clone());

    let model_manager = Arc::new(Mutex::new(model_manager));
    {
        model_manager.lock().from_file("src/cube/cube.obj");
    }

    let particles = Arc::new(particles::ParticleCompute::new(
        vk.device.clone(),
        vk.clone(),
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
    //     CpuBufferPool::<renderer::vs::ty::Data>::new(vk.device.clone(), BufferUsage::all());

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
        Some(vk.swapchain.lock().image_format()),
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
    let (tx, rx): (Sender<Input>, Receiver<Input>) = mpsc::channel();
    let (rtx, rrx): (Sender<_>, Receiver<_>) = mpsc::channel();
    // let (ttx, trx): (Sender<Terrain>, Receiver<Terrain>) = mpsc::channel();
    let running = Arc::new(AtomicBool::new(true));

    let coms = (rrx, tx);

    let physics = Physics::new();

    let world = Arc::new(Mutex::new(World::new(
        model_manager.clone(),
        renderer_manager,
        physics,
        particles.clone(),
        vk.clone(),
    )));
    {
        let mut world = world.lock();
        world.register::<Renderer>(false, false, false);
        world.register::<ParticleEmitter>(false, false, false);
        // world.register::<Maker>(true);
        world.register::<Terrain>(true, false, true);
        world.register::<Bomb>(true, false, false);
        world.register::<Player>(true, false, false);
        world.register::<Camera>(false, false, false);
    }
    let rm = {
        let w = world.lock();
        let s = w.sys.lock();
        let rm = s.renderer_manager.read();
        
        rm.shr_data.clone()
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

    let mut cam_data = CameraData::new(vk.clone());
    let mut playing_game = unsafe { PLAYING_GAME };

    let (mut cam_pos, mut cam_rot) = (Vec3::default(), glm::quat(-1., 0., 0., 0.));
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

                static mut GRAB_MODE: bool = true;
                if input.get_key_press(&VirtualKeyCode::G) {
                    unsafe {
                        let _er = vk
                            .surface
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
                    vk.surface
                        .object()
                        .unwrap()
                        .downcast_ref::<Window>()
                        .unwrap()
                        .set_cursor_visible(modifiers.shift());
                }

                let (transform_data, cam_datas, main_cam_id, mut rd, emitter_inits) = {
                    puffin::profile_scope!("wait for game");
                    let inst = Instant::now();
                    let (transform_data, cam_datas, main_cam_id, renderer_data, emitter_inits) =
                        coms.0.recv().unwrap();

                    perf.update("wait for game".into(), Instant::now() - inst);
                    (
                        transform_data,
                        cam_datas,
                        main_cam_id,
                        renderer_data,
                        emitter_inits,
                    )
                };

                let speed = 30f32 * input.time.dt;
                if !input.get_key(&VirtualKeyCode::LControl) && !unsafe { PLAYING_GAME } {
                    // forward/backward
                    if input.get_key(&VirtualKeyCode::W) {
                        cam_pos +=
                            (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * -speed;
                    }
                    if input.get_key(&VirtualKeyCode::S) {
                        cam_pos +=
                            (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * speed;
                    }
                    //left/right
                    if input.get_key(&VirtualKeyCode::A) {
                        cam_pos +=
                            (glm::quat_to_mat4(&cam_rot) * vec4(1.0, 0.0, 0.0, 1.0)).xyz() * -speed;
                    }
                    if input.get_key(&VirtualKeyCode::D) {
                        cam_pos +=
                            (glm::quat_to_mat4(&cam_rot) * vec4(1.0, 0.0, 0.0, 1.0)).xyz() * speed;
                    }
                    // up/down
                    if input.get_key(&VirtualKeyCode::Space) {
                        cam_pos +=
                            (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 1.0, 0.0, 1.0)).xyz() * -speed;
                    }
                    if input.get_key(&VirtualKeyCode::LShift) {
                        cam_pos +=
                            (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 1.0, 0.0, 1.0)).xyz() * speed;
                    }

                    if input.get_mouse_button(&2) {
                        cam_rot = glm::quat_rotate(
                            &cam_rot,
                            input.get_mouse_delta().0 as f32 * 0.01,
                            &(glm::inverse(&glm::quat_to_mat3(&cam_rot)) * Vec3::y()),
                        );
                        cam_rot = glm::quat_rotate(
                            &cam_rot,
                            input.get_mouse_delta().1 as f32 * 0.01,
                            &Vec3::x(),
                        );
                    }
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

                    let mut swapchain = vk.swapchain.lock();
                    let (new_swapchain, new_images) =
                        match swapchain.recreate(SwapchainCreateInfo {
                            image_extent: dimensions,
                            ..swapchain.create_info()
                        }) {
                            Ok(r) => r,
                            Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                            Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                        };

                    *swapchain = new_swapchain;

                    framebuffers = window_size_dependent_setup(
                        &new_images,
                        // vk.device.clone(),
                        render_pass.clone(),
                        &mut viewport,
                    );
                    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
                    recreate_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) =
                    match acquire_next_image(vk.swapchain.lock().clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            println!("falied to aquire next image");
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };
                let cam_num = if unsafe { PLAYING_GAME } {
                    main_cam_id
                } else {
                    -1
                };
                let fc = fc_map
                    .entry(cam_num)
                    .or_insert(HashMap::<u32, TextureId>::new())
                    .entry(image_num)
                    .or_insert_with(|| {
                        let frame_image_view = ImageView::new_default(if unsafe { PLAYING_GAME } {
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
                gui.immediate_ui(|gui| {
                    let ctx = gui.context();
                    editor_ui::editor_ui(
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

                if playing_game != unsafe { PLAYING_GAME } {
                    for (_k, v) in &fc_map {
                        for (_k, v) in v.iter() {
                            gui.unregister_user_image(*v);
                        }
                    }
                    fc_map.clear();
                }

                perf.update("gui".into(), Instant::now() - inst);

                let render_jobs = world.lock().render();

                // let rm = renderer_manager.read();
                let mut rm = rm.write();

                // start new game frame
                let res = coms.1.send(input.clone());
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
                        vk.device.clone(),
                        transform_data.clone(),
                        vk.mem_alloc.clone(),
                    );

                    let rotation_update_data = transform_compute.get_rotation_update_data(
                        vk.device.clone(),
                        transform_data.clone(),
                        vk.mem_alloc.clone(),
                    );

                    let scale_update_data = transform_compute.get_scale_update_data(
                        vk.device.clone(),
                        transform_data.clone(),
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
                let inst = Instant::now();

                let mut builder = AutoCommandBufferBuilder::primary(
                    &vk.comm_alloc,
                    vk.queue.queue_family_index(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                {
                    puffin::profile_scope!("transform update compute");

                    // compute shader transforms
                    transform_compute.update(
                        vk.device.clone(),
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
                        &transform_uniforms,
                        compute_pipeline.clone(),
                        scale_update_data,
                        vk.mem_alloc.clone(),
                        &vk.comm_alloc,
                        vk.desc_alloc.clone(),
                    );
                }
                {

                    particles.update(
                        &mut builder,
                        emitter_inits,
                        transform_compute.transform.clone(),
                        input.time.dt,
                        input.time.time,
                        cam_pos.into(),
                        cam_rot.coords.into(),
                    );
                }
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

                if !unsafe { PLAYING_GAME } {
                    cam_data.update(cam_pos, cam_rot, 0.01f32, 10_000f32, 70f32);
                    cam_data.render(
                        vk.clone(),
                        &mut builder,
                        &mut transform_compute,
                        particles.clone(),
                        &transform_uniforms,
                        transform_data,
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
                            particles.clone(),
                            &transform_uniforms,
                            transform_data.clone(),
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
                playing_game = unsafe { PLAYING_GAME };
                // builder
                // .copy_image(CopyImageInfo::images(
                //     image.clone(),
                //     frame_img.clone(),
                // ))
                // .blit_image(BlitImageInfo::images(
                //     frame_color.arc.clone(),
                //     frame_img.clone(),
                // ))
                // .unwrap();

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
                                    vk.swapchain.lock().clone(),
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
