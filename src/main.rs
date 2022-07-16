// use std::collections::{HashMap, HashSet};
// use std::ffi::CStr;
// use std::fs::File;
// use std::hash::{Hash, Hasher};
// use std::io::BufReader;
// use std::mem::size_of;
// use std::os::raw::c_void;
// use std::ptr::copy_nonoverlapping as memcpy;
// use std::{
//     sync::Arc,
//     time::{Duration, Instant},
// };

use egui::plot::{HLine, Line, Plot, Value, Values};
use egui::{Color32, Ui};
// use egui_vulkano::UpdateTexturesResult;
// use anyhow::{anyhow, Result};
// use log::*;
use nalgebra_glm as glm;
use vulkano::buffer::device_local;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::{Pipeline, PipelineBindPoint};
// use parking_lot::Mutex;
// use rayon::iter::{
//     IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
//     IntoParallelRefMutIterator, ParallelIterator,
// };
use winit::event::MouseButton;

use std::collections::VecDeque;
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

mod engine;
mod input;
mod model;
mod renderer;
mod texture;
mod time;
mod transform_compute;
// mod rendering;
// use transform::{Transform, Transforms};

use std::env;

// use rand::prelude::*;
// use rapier3d::prelude::*;

use crate::game::game_thread_fn;
use crate::texture::TextureManager;
use crate::{
    input::Input,
    model::Mesh,
    renderer::{ModelMat, RenderPipeline},
    // renderer::RenderPipeline,
    terrain::Terrain,
};

mod game;
mod terrain;

fn fast_buffer(device: Arc<Device>, data: &Vec<ModelMat>) -> Arc<CpuAccessibleBuffer<[ModelMat]>> {
    unsafe {
        // let inst = Instant::now();
        let uninitialized = CpuAccessibleBuffer::<[ModelMat]>::uninitialized_array(
            device.clone(),
            data.len() as DeviceSize,
            BufferUsage::all(),
            false,
        )
        .unwrap();
        {
            let mut mapping = uninitialized.write().unwrap();
            let mo_iter = data.iter();
            let m_iter = mapping.iter_mut();

            // let inst = Instant::now();
            mo_iter.zip(m_iter).for_each(|(i, o)| {
                // for (i, o) in slice {
                ptr::write(o, *i);
                // }
            });
            // update_perf("write to buffer".into(), Instant::now() - inst);
        }
        uninitialized
    }
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let event_loop = EventLoop::new();

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .unwrap();

    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_extensions))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),
            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let surface_capabilities = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format,
                image_extent: surface.window().inner_size().into(),
                image_usage: ImageUsage::color_attachment(),
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .iter()
                    .next()
                    .unwrap(),
                present_mode: vulkano::swapchain::PresentMode::Fifo,
                ..Default::default()
            },
        )
        .unwrap()
    };

    let texture_manager = Arc::new(TextureManager {
        device: device.clone(),
        queue: queue.clone(),
        textures: Default::default(),
    });

    let cube_mesh = Mesh::load_model("src/cube/cube.obj", device.clone(), texture_manager.clone());

    let uniform_buffer =
        CpuBufferPool::<renderer::vs::ty::Data>::new(device.clone(), BufferUsage::all());

    let render_pass = vulkano::ordered_passes_renderpass!(
        device.clone(),
        attachments: {
            color: {
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
            { color: [color], depth_stencil: {}, input: [] } // Create a second renderpass to draw egui
        ]
    )
    .unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers =
        window_size_dependent_setup(&images, device.clone(), render_pass.clone(), &mut viewport);

    // let mut framebuffers =
    //     window_size_dependent_setup(device.clone(), &images, render_pass.clone());

    let rend = RenderPipeline::new(
        device.clone(),
        render_pass.clone(),
        images[0].dimensions().width_height(),
        queue.clone(),
    );
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    // let rotation_start = Instant::now();

    let mut modifiers = ModifiersState::default();

    //////////////////////////////////////////////////

    let mut input = Input {
        ..Default::default()
    };

    let mut perf = HashMap::<String, Duration>::new();

    let (tx, rx): (Sender<Input>, Receiver<Input>) = mpsc::channel();
    let (rtx, rrx): (
        Sender<(
            Arc<Vec<ModelMat>>,
            Arc<Vec<ModelMat>>,
            glm::Vec3,
            glm::Quat,
            // Arc<&HashMap<i32, HashMap<i32, Mesh>>>,
        )>,
        Receiver<(
            Arc<Vec<ModelMat>>,
            Arc<Vec<ModelMat>>,
            glm::Vec3,
            glm::Quat,
            // Arc<&HashMap<i32, HashMap<i32, Mesh>>>,
        )>,
    ) = mpsc::channel();
    let (ttx, trx): (Sender<Terrain>, Receiver<Terrain>) = mpsc::channel();
    let running = Arc::new(AtomicBool::new(true));

    let coms = (rrx, tx);

    let _device = device.clone();
    let _queue = queue.clone();
    let game_thread = {
        // let _perf = perf.clone();
        let _running = running.clone();
        thread::spawn(move || {
            game_thread_fn(
                _device,
                _queue,
                texture_manager.clone(),
                (rtx, rx, ttx),
                _running,
            )
        })
    };
    let mut game_thread = vec![game_thread];

    let ter = trx.recv().unwrap();
    // println!("sending input");
    let _res = coms.1.send(input.clone());

    let mut loops = 0;

    let egui_ctx = egui::Context::default();
    let mut egui_winit = egui_winit::State::new(4096, surface.window());

    let mut egui_painter = egui_vulkano::Painter::new(
        device.clone(),
        queue.clone(),
        Subpass::from(render_pass.clone(), 1).unwrap(),
    )
    .unwrap();

    //Set up some window to look at for the test

    // let mut egui_test = egui_demo_lib::ColorTest::default();
    // let mut demo_windows = egui_demo_lib::DemoWindows::default();
    let mut egui_bench = Benchmark::new(1000);
    // let mut my_texture = egui_ctx.load_texture("my_texture", ColorImage::example());

    let mut frame_time = Instant::now();
    // let mut fps_time: f32 = 0.0;

    let mut transforms_buffer = transform_compute::transform_buffer_init(
        device.clone(),
        // queue.clone(),
        vec![
            ModelMat {
                ..Default::default()
            };
            1
        ],
    );

    mod cs {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/transform.comp",
            types_meta: {
                use bytemuck::{Pod, Zeroable};

                #[derive(Clone, Copy, Zeroable, Pod)]
            },
        }
    }
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

    let transform_uniforms =
    CpuBufferPool::<cs::ty::Data>::new(device.clone(), BufferUsage::all());

    let mut focused = true;
    event_loop.run(move |event, _, control_flow| {
        // let game_thread = game_thread.clone();
        *control_flow = ControlFlow::Poll;
        match event {
            // Event::MainEventsCleared if !destroying && !minimized => {
            //     // unsafe { app.render(&window) }.unwrap()
            // }
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta } => {
                    if focused {
                        input.mouse_x = delta.0;
                        input.mouse_y = delta.1;
                    }
                    // println!("mouse moved: {:?}", delta);
                }
                _ => (),
            },
            Event::WindowEvent { event, .. } => {
                let egui_consumed_event = egui_winit.on_event(&egui_ctx, &event);
                if !egui_consumed_event {
                    // do your own event handling here
                };
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
                        // game_thread.join();
                        // *running.lock() = false;
                        let p = perf.iter();
                        for (k, x) in p {
                            println!("{}: {:?}", k, (*x / loops));
                        }
                    }
                    WindowEvent::MouseInput {
                        device_id: _,
                        state,
                        button,
                        ..
                    } => match state {
                        ElementState::Pressed => {
                            println!("mouse button {:#?} pressed", button);
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
                            println!("mouse button {:#?} released", button);
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
            }
            Event::RedrawEventsCleared => {
                ////////////////////////////////////

                let full = Instant::now();

                if input.get_key(&VirtualKeyCode::Escape) {
                    *control_flow = ControlFlow::Exit;
                    running.store(false, Ordering::SeqCst);
                    let game_thread = game_thread.remove(0);
                    let _res = coms.1.send(input.clone());

                    // let (_, _, _, _) = coms.0.recv().unwrap();

                    game_thread.join().unwrap();
                    let p = perf.iter();
                    for (k, x) in p {
                        println!("{}: {:?}", k, (*x / loops));
                    }
                    // for (k, x) in perf.lock().iter() {
                    //     println!("{}: {:?}", k, (*x / loops));
                    // }
                }
                let mut update_perf = |k: String, v: Duration| {
                    if let Some(dur) = perf.get_mut(&k) {
                        *dur += v;
                    } else {
                        perf.insert(k, v);
                    }
                };
                static mut GRAB_MODE: bool = true;
                if input.get_key_press(&VirtualKeyCode::G) {
                    unsafe {
                        let _er = surface.window().set_cursor_grab(GRAB_MODE);
                        GRAB_MODE = !GRAB_MODE;
                    }
                }
                // if input.get_key(&VirtualKeyCode::L) { let _er = surface.window().set_cursor_grab(false); }
                // A => surface.window().set_cursor_grab(),
                if input.get_key(&VirtualKeyCode::H) {
                    surface.window().set_cursor_visible(modifiers.shift());
                }

                let inst = Instant::now();
                let (terrain_models, cube_models, cam_pos, cam_rot) = coms.0.recv().unwrap();

                update_perf("wait for game".into(), Instant::now() - inst);
                let res = coms.1.send(input.clone());
                // println!("input sent");
                if res.is_err() {
                    return;
                    // println!("ohno");
                }

                input.key_presses.clear();
                input.key_ups.clear();
                input.mouse_x = 0.;
                input.mouse_y = 0.;

                loops += 1;

                let inst = Instant::now();
                let _instance_buffer = fast_buffer(device.clone(), &terrain_models);
                let cube_instance_buffer = fast_buffer(device.clone(), &cube_models);

                update_perf("write to buffer".into(), Instant::now() - inst);
                //////////////////////////////////

                // render_thread = thread::spawn( || {
                let inst = Instant::now();

                let dimensions = surface.window().inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }

                let inst2 = Instant::now();
                previous_frame_end.as_mut().unwrap().cleanup_finished();

                update_perf("previous frame end".into(), Instant::now() - inst2);

                if recreate_swapchain {
                    println!("recreate swapchain");
                    println!("dimensions {}: {}", dimensions.width, dimensions.height);
                    let dimensions: [u32; 2] = surface.window().inner_size().into();

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
                        device.clone(),
                        render_pass.clone(),
                        &mut viewport,
                    );
                    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
                    recreate_swapchain = false;
                    // let new_framebuffers = window_size_dependent_setup(
                    //     device.clone(),
                    //     // &vs,
                    //     // &fs,
                    //     &new_images,
                    //     render_pass.clone(),
                    // );
                    // rend.regen(device.clone(), render_pass.clone(), dimensions.into());
                    // // pipeline = new_pipeline;
                    // framebuffers = new_framebuffers;
                    // recreate_swapchain = false;
                }

                let uniform_buffer_subbuffer = {
                    // note: this teapot was meant for OpenGL where the origin is at the lower left
                    //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
                    let aspect_ratio =
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

                    // let scale = glm::scale(&Mat4::identity(), &Vec3::new(0.1 as f32, 0.1, 0.1));

                    let uniform_data = renderer::vs::ty::Data {
                        view: view.into(),
                        proj: proj.into(),
                    };

                    uniform_buffer.next(uniform_data).unwrap()
                };



                let (image_num, suboptimal, acquire_future) =
                    match acquire_next_image(swapchain.clone(), None) {
                        Ok(r) => r,
                        Err(AcquireError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let mut builder = AutoCommandBufferBuilder::primary(
                    device.clone(),
                    queue.family(),
                    CommandBufferUsage::OneTimeSubmit,
                )
                .unwrap();

                egui_ctx.begin_frame(egui_winit.take_egui_input(surface.window()));
                // demo_windows.ui(&egui_ctx);

                // egui::Window::new("Color test")
                //     .vscroll(true)
                //     .show(&egui_ctx, |ui| {
                //         egui_test.ui(ui);
                //     });

                // egui::Window::new("Settings").show(&egui_ctx, |ui| {
                //     egui_ctx.settings_ui(ui);
                // });

                egui::Window::new("Benchmark")
                    .default_height(600.0)
                    .show(&egui_ctx, |ui| {
                        egui_bench.draw(ui);
                    });

                // egui::Window::new("Texture test").show(&egui_ctx, |ui| {
                //     ui.image(my_texture.id(), (200.0, 200.0));
                //     if ui.button("Reload texture").clicked() {
                //         // previous TextureHandle is dropped, causing egui to free the texture:
                //         my_texture = egui_ctx.load_texture("my_texture", ColorImage::example());
                //     }
                // });

                // Get the shapes from egui
                let egui_output = egui_ctx.end_frame();
                let platform_output = egui_output.platform_output;
                egui_winit.handle_platform_output(surface.window(), &egui_ctx, platform_output);

                let _result = egui_painter
                    .update_textures(egui_output.textures_delta, &mut builder)
                    .expect("egui texture error");

                // let wait_for_last_frame = result == UpdateTexturesResult::Changed;

                let new_transform_buffer = transform_compute::transform_buffer(
                    device.clone(),
                    // queue.clone(),
                    &mut transforms_buffer,
                    cube_models.to_vec(),
                );

                let mut curr_transform_buffer = transforms_buffer.data.clone();
                // let mut transforms_realloced = false;

                if let Some(device_local_buffer) = new_transform_buffer {
                    builder
                        .copy_buffer(transforms_buffer.data.clone(), device_local_buffer.clone())
                        .unwrap();
                        // transforms_realloced = true;
                        curr_transform_buffer = device_local_buffer.clone();
                }

                let transforms_sub_buffer = {

                    // let scale = glm::scale(&Mat4::identity(), &Vec3::new(0.1 as f32, 0.1, 0.1));

                    let uniform_data = cs::ty::Data {
                        num_jobs: cube_models.len() as i32,
                    };

                    transform_uniforms.next(uniform_data).unwrap()
                };

                let descriptor_set = PersistentDescriptorSet::new(
                    compute_pipeline
                        .layout()
                        .set_layouts()
                        .get(0) // 0 is the index of the descriptor set.
                        .unwrap()
                        .clone(),
                    [
                        // 0 is the binding of the data in this set. We bind the `DeviceLocalBuffer` of vertices here.
                        WriteDescriptorSet::buffer(0, curr_transform_buffer.clone()),
                        WriteDescriptorSet::buffer(1, cube_instance_buffer.clone()),
                        WriteDescriptorSet::buffer(2, transforms_sub_buffer.clone()),
                    ],
                )
                .unwrap();

                builder
                    .bind_pipeline_compute(compute_pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        compute_pipeline.layout().clone(),
                        0, // Bind this descriptor set to index 0.
                        descriptor_set.clone(),
                    )
                    .dispatch([cube_models.len() as u32 / 128 + 1, 1, 1])
                    .unwrap()
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()], // clear color
                    )
                    .unwrap()
                    .set_viewport(0, [viewport.clone()]);

                rend.bind_pipeline(&mut builder).bind_mesh(
                    &mut builder,
                    &uniform_buffer_subbuffer,
                    cube_instance_buffer,
                    curr_transform_buffer.clone(),
                    &cube_mesh,
                );

                for (_, z) in ter.chunks.iter() {
                    for (_, chunk) in z {
                        rend.bind_mesh(
                            &mut builder,
                            &uniform_buffer_subbuffer,
                            _instance_buffer.clone(),
                            curr_transform_buffer.clone(),
                            &chunk,
                        );
                    }
                }

                // Automatically start the next render subpass and draw the gui
                let size = surface.window().inner_size();
                let sf: f32 = surface.window().scale_factor() as f32;
                egui_painter
                    .draw(
                        &mut builder,
                        [(size.width as f32) / sf, (size.height as f32) / sf],
                        &egui_ctx,
                        egui_output.shapes,
                    )
                    .unwrap();

                input.time.dt = frame_time.elapsed().as_secs_f64() as f32;

                egui_bench.push(frame_time.elapsed().as_secs_f64());
                frame_time = Instant::now();

                builder.end_render_pass().unwrap();
                let command_buffer = builder.build().unwrap();

                let inst2 = Instant::now();
                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                update_perf("previous frame end future".into(), Instant::now() - inst2);

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

                transforms_buffer.data = curr_transform_buffer;
                update_perf("render".into(), Instant::now() - inst);
                update_perf("full".into(), Instant::now() - full);

                // });
            }
            _ => (),
        }
    });
}

/// This method is called once during initialization, then again whenever the window is resized
fn _window_size_dependent_setup(
    device: Arc<Device>,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();

    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(device.clone(), dimensions, Format::D32_SFLOAT).unwrap(),
    )
    .unwrap();

    let framebuffers = images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
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

    framebuffers
}

fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage<Window>>],
    device: Arc<Device>,
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];

    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(device.clone(), dimensions, Format::D32_SFLOAT).unwrap(),
    )
    .unwrap();

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

pub struct Benchmark {
    capacity: usize,
    data: VecDeque<f64>,
}

impl Benchmark {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            data: VecDeque::with_capacity(capacity),
        }
    }

    pub fn draw(&self, ui: &mut Ui) {
        let iter = self
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| Value::new(i as f64, *v * 1000.0));
        let curve = Line::new(Values::from_values_iter(iter)).color(Color32::BLUE);
        let target = HLine::new(1000.0 / 60.0).color(Color32::RED);

        if let Some(fps) = self.data.back() {
            let fps = 1.0 / fps;
            ui.label(format!("fps: {}", fps));
        }
        ui.label("Time in milliseconds that the gui took to draw:");
        Plot::new("plot")
            .view_aspect(2.0)
            .include_y(0)
            .show(ui, |plot_ui| {
                plot_ui.line(curve);
                plot_ui.hline(target)
            });
        ui.label("The red line marks the frametime target for drawing at 60 FPS.");
    }

    pub fn push(&mut self, v: f64) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(v);
    }
}
