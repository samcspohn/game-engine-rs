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

// use anyhow::{anyhow, Result};
// use log::*;
use nalgebra_glm as glm;
use parking_lot::Mutex;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
// use rendering::App;
// use thiserror::Error;
// use vulkanalia::loader::{LibloadingLoader, LIBRARY};
// use vulkanalia::prelude::v1_0::*;
// use vulkanalia::window as vk_window;
// use winit::dpi::LogicalSize;
// use winit::event::{Event, WindowEvent};
// use winit::event_loop::{ControlFlow, EventLoop};
// use winit::window::{Window, WindowBuilder};

// use vulkanalia::vk::ExtDebugUtilsExtension;
// use vulkanalia::vk::KhrSurfaceExtension;
// use vulkanalia::vk::KhrSwapchainExtension;

// /// Whether the validation layers should be enabled.
// const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
// /// The name of the validation layers.
// const VALIDATION_LAYER: vk::ExtensionName =
//     vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

// /// The required device extensions.
// const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

// /// The maximum number of frames that can be processed concurrently.
// const MAX_FRAMES_IN_FLIGHT: usize = 2;

// use winit::event::{DeviceEvent, ElementState, KeyboardInput, ModifiersState, VirtualKeyCode};

use std::{
    collections::HashMap,
    ptr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, Thread},
    time::{Duration, Instant},
};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            depth_stencil::DepthStencilState,
            input_assembly::InputAssemblyState,
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, Pipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
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

use bytemuck::{Pod, Zeroable};
use component_derive::component;
use engine::{
    physics,
    transform::{Transform, Transforms},
    Component, LazyMaker,
};
use glm::{cross, normalize, vec3, vec4, Mat4, Vec3, Vec4};

use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
// use rust_test::{INDICES, NORMALS, VERTICES};

mod engine;
mod input;
mod model;
mod renderer;
// mod rendering;
// use transform::{Transform, Transforms};

use std::env;

// use rand::prelude::*;
use rapier3d::prelude::*;

use crate::{
    engine::{physics::Physics, World},
    input::Input,
    model::Mesh,
    renderer::{ModelMat, RenderPipeline},
    // renderer::RenderPipeline,
    terrain::Terrain,
};
use model::{Normal, Vertex};

mod terrain;

#[component]
struct Bomb {
    vel: glm::Vec3,
}
impl Component for Bomb {
    fn init(&mut self, t: Transform) {
        self.t = t;
    }
    fn update(&mut self, trans: &Transforms, sys: (&physics::Physics, &LazyMaker)) {
        let pos = trans.get_position(self.t);
        let vel = self.vel;
        // let dir = vel * (1.0 / 100.0);
        let ray = rapier3d::prelude::Ray {
            origin: point![pos.x, pos.y, pos.z],
            dir: vel,
        };
        if let Some((_handle, _hit)) = sys.0.query_pipeline.cast_ray_and_get_normal(
            &&sys.0.collider_set,
            &ray,
            1.0 / 60.0,
            false,
            InteractionGroups::all(),
            None,
        ) {
            self.vel = glm::reflect_vec(&vel, &&_hit.normal);
        }
        trans._move(self.t, self.vel * (1.0 / 60.0));
        self.vel += glm::vec3(0.0, -9.81, 0.0) * 1.0 / 100.0;

        // *pos += vel * (1.0 / 60.0);
    }
}
#[component]
struct Maker {}
impl Component for Maker {
    fn init(&mut self, t: Transform) {
        self.t = t;
    }
    fn update(&mut self, trans: &Transforms, sys: (&physics::Physics, &LazyMaker)) {
        sys.1.append(|world| {
            let g = world.instantiate();
            world.add_component(
                g,
                Bomb {
                    t: Transform(-1),
                    vel: glm::vec3(rand::random(), rand::random(), rand::random()),
                },
            );
            world.get_component::<Bomb, _>(g, |b| {
                if let Some(b) = b {
                    b.lock().vel = glm::vec3(rand::random(), rand::random(), rand::random());
                }
            });
        });
    }
}

fn game_thread_fn(
    device: Arc<Device>,
    coms: (
        Sender<(
            Arc<Vec<ModelMat>>,
            Arc<Vec<ModelMat>>,
            glm::Vec3,
            glm::Quat,
            // Arc<&HashMap<i32, HashMap<i32, Mesh>>>,
        )>,
        Receiver<Input>,
        Sender<Terrain>,
    ),
    running: Arc<AtomicBool>,
) {
    let mut physics = Physics::new();

    /* Create the ground. */
    // let collider = ColliderBuilder::cuboid(100.0, 0.1, 100.0).build();
    // physics.collider_set.insert(collider);

    /* Create the bounding ball. */
    let rigid_body = RigidBodyBuilder::dynamic()
        .translation(vector![0.0, 10.0, 0.0])
        .build();
    let collider = ColliderBuilder::ball(0.5).restitution(0.7).build();
    let ball_body_handle = physics.rigid_body_set.insert(rigid_body);
    physics.collider_set.insert_with_parent(
        collider,
        ball_body_handle,
        &mut physics.rigid_body_set,
    );

    /* Create other structures necessary for the simulation. */
    let gravity = vector![0.0, -9.81, 0.0];

    let mut world = World::new();
    world.register::<Bomb>();
    world.register::<Maker>();

    let _root = world.instantiate();

    // use rand::Rng;
    for _ in 0..1_000 {
        // bombs
        let g = world.instantiate();
        world.add_component(
            g,
            Bomb {
                t: Transform(-1),
                vel: glm::vec3(
                    rand::random::<f32>() - 0.5,
                    rand::random::<f32>() - 0.5,
                    rand::random::<f32>() - 0.5,
                ) * 5.0,
            },
        );
        world.transforms.read()._move(
            g.t,
            glm::vec3(
                rand::random::<f32>() * 100. - 50.,
                100.0,
                rand::random::<f32>() * 100. - 50.,
            ),
        );
    }
    // {
    //     // maker
    //     let g = world.instantiate();
    //     world.add_component(g, Maker { t: Transform(-1) });
    // }
    let lazy_maker = LazyMaker::new();

    let mut ter = Terrain {
        chunks: HashMap::new(),
        device: device.clone(),
        terrain_size: 33,
    };
    ter.generate(&mut physics.collider_set);

    let res = coms.2.send(ter.clone());
    if res.is_err() {
        // println!("ohno");
    }

    ////////////////////////////////////////////////
    let mut cam_pos = glm::vec3(0.0 as f32, 0.0, -1.0);
    let mut cam_rot = glm::quat(1.0, 0.0, 0.0, 0.0);

    // let mut input = Input {
    //     ..Default::default()
    // };

    let mut perf = HashMap::<String, Duration>::new();

    let mut update_perf = |k: String, v: Duration| {
        if let Some(dur) = perf.get_mut(&k) {
            *dur += v;
        } else {
            perf.insert(k, v);
        }
    };
    let mut loops = 0;
    while running.load(Ordering::SeqCst) {
        loops += 1;
        // println!("waiting for input");
        let input = coms.1.recv().unwrap();
        // println!("input recvd");

        let speed = 0.3;
        // forward/backward
        if input.get_key(&VirtualKeyCode::W) {
            cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * -speed;
        }
        if input.get_key(&VirtualKeyCode::S) {
            cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * speed;
        }
        //left/right
        if input.get_key(&VirtualKeyCode::A) {
            cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(1.0, 0.0, 0.0, 1.0)).xyz() * -speed;
        }
        if input.get_key(&VirtualKeyCode::D) {
            cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(1.0, 0.0, 0.0, 1.0)).xyz() * speed;
        }
        // up/down
        if input.get_key(&VirtualKeyCode::Space) {
            cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 1.0, 0.0, 1.0)).xyz() * -speed;
        }
        if input.get_key(&VirtualKeyCode::LShift) {
            cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 1.0, 0.0, 1.0)).xyz() * speed;
        }

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

        let inst = Instant::now();
        physics.step(&gravity);
        world.update(&physics, &lazy_maker);

        const alot: f32 = 10_000_000. / 60.;

        if input.get_mouse_button(&1) {
            let _cam_rot = cam_rot.clone();
            let _cam_pos = cam_pos.clone();
            lazy_maker.append(move |world| {
                for _ in 0..(alot / 60.) as usize {
                    let g = world.instantiate();
                    world.add_component(
                        g,
                        Bomb {
                            t: Transform(-1),
                            vel: glm::quat_to_mat3(&_cam_rot) * -glm::Vec3::z() * 50.
                                + glm::vec3(rand::random(), rand::random(), rand::random()) * 18.,
                        },
                    );
                    world
                        .transforms
                        .read()
                        .set_position(g.t, _cam_pos - glm::Vec3::y() * 2.);
                }
            });
        }

        lazy_maker.init(&mut world);

        update_perf("world".into(), Instant::now() - inst);

        // dur += Instant::now() - inst;

        let inst = Instant::now();

        let positions = &world.transforms.read().positions;

        let mut cube_models: Vec<ModelMat> = Vec::with_capacity(positions.len());

        // cube_models.reserve(positions.len());
        unsafe {
            cube_models.set_len(positions.len());
        }
        let p_iter = positions.par_iter();
        let m_iter = cube_models.par_iter_mut();

        p_iter.zip_eq(m_iter).chunks(64 * 64).for_each(|slice| {
            for (x, y) in slice {
                let x = x.lock();
                *y = ModelMat {
                    pos: [x.x, x.y, x.z],
                };
            }
        });

        update_perf("get cube models".into(), Instant::now() - inst);
        let terrain_models: Vec<ModelMat> = vec![ModelMat {
            pos: glm::vec3(0., 0., 0.).into(),
        }];
        let terrain_models = Arc::new(terrain_models);
        let cube_models = Arc::new(cube_models);
        // let terr_chunks = Arc::new(&ter.chunks);
        // println!("sending models");
        let res = coms.0.send((
            terrain_models.clone(),
            cube_models.clone(),
            cam_pos.clone(),
            cam_rot.clone(),
            // terr_chunks,
        ));
        if res.is_err() {
            // println!("ohno");
        }
    }
    let p = perf.iter();
    for (k, x) in p {
        println!("{}: {:?}", k, (*x / loops));
    }
}

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
    // rayon::ThreadPoolBuilder::new().num_threads(63).build_global().unwrap();

    /////////////////////////////////////////////////

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
                min_image_count: 4u32,
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

    let cube_mesh = Mesh::load_model("src/cube/cube.obj", device.clone());

    let uniform_buffer =
        CpuBufferPool::<renderer::vs::ty::Data>::new(device.clone(), BufferUsage::all());

    let render_pass = vulkano::single_pass_renderpass!(
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
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
    .unwrap();

    let mut framebuffers =
        window_size_dependent_setup(device.clone(), &images, render_pass.clone());

    let mut rend = RenderPipeline::new(
        device.clone(),
        render_pass.clone(),
        images[0].dimensions().width_height(),
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
    let mut running = Arc::new(AtomicBool::new(true));

    let coms = (rrx, tx);

    let _device = device.clone();
    let game_thread = {
        // let _perf = perf.clone();
        let _running = running.clone();
        thread::spawn(move || game_thread_fn(_device, (rtx, rx, ttx), _running))
    };
    let mut game_thread = vec![game_thread];

    let ter = trx.recv().unwrap();
    // println!("sending input");
    let _res = coms.1.send(input.clone());

    let mut loops = 0;

    event_loop.run(move |event, _, control_flow| {
        // let game_thread = game_thread.clone();
        *control_flow = ControlFlow::Poll;
        match event {
            // Event::MainEventsCleared if !destroying && !minimized => {
            //     // unsafe { app.render(&window) }.unwrap()
            // }
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta } => {
                    input.mouse_x = delta.0;
                    input.mouse_y = delta.1;
                    // println!("mouse moved: {:?}", delta);
                }
                DeviceEvent::Button { button, state } => match state {
                    ElementState::Pressed => {
                        println!("mouse button {} pressed", button);
                        input.mouse_buttons.insert(button, true);
                    }
                    ElementState::Released => {
                        println!("mouse button {} released", button);
                        input.mouse_buttons.insert(button, false);
                    }
                },
                _ => (),
            },
            Event::WindowEvent { event, .. } => match event {
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
                WindowEvent::Resized(_size) => {
                    recreate_swapchain = true;
                    // if size.width == 0 || size.height == 0 {
                    //     minimized = true;
                    // } else {
                    //     minimized = false;
                    //     app.resized = true;
                    // }
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
                _ => (),
            },
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
                static mut grab_mode: bool = true;
                if input.get_key_press(&VirtualKeyCode::G) {
                    unsafe {
                        let _er = surface.window().set_cursor_grab(grab_mode);
                        grab_mode = !grab_mode;
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
                    let new_framebuffers = window_size_dependent_setup(
                        device.clone(),
                        // &vs,
                        // &fs,
                        &new_images,
                        render_pass.clone(),
                    );
                    rend.regen(
                        device.clone(),
                        render_pass.clone(),
                        images[0].dimensions().width_height(),
                    );
                    // pipeline = new_pipeline;
                    framebuffers = new_framebuffers;
                    recreate_swapchain = false;
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
                        // world: model.into(),
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
                builder
                    .begin_render_pass(
                        framebuffers[image_num].clone(),
                        SubpassContents::Inline,
                        vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()], // clear color
                    )
                    .unwrap();

                rend.bind_pipeline(&mut builder, &uniform_buffer_subbuffer)
                    .bind_mesh(&mut builder, cube_instance_buffer, &cube_mesh);

                for (_, z) in ter.chunks.iter() {
                    for (_, chunk) in z {
                        rend.bind_mesh(&mut builder, _instance_buffer.clone(), &chunk);
                    }
                }

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
                update_perf("render".into(), Instant::now() - inst);
                update_perf("full".into(), Instant::now() - full);

                // });
            }
            _ => (),
        }
    });
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    device: Arc<Device>,
    // vs: &ShaderModule,
    // fs: &ShaderModule,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
) -> (Vec<Arc<Framebuffer>>) {
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

// mod vs {
//     vulkano_shaders::shader! {
//         ty: "vertex",
//         path: "src/vert.glsl",
//         types_meta: {
//             use bytemuck::{Pod, Zeroable};

//             #[derive(Clone, Copy, Zeroable, Pod)]
//         },
//     }
// }

// mod fs {
//     vulkano_shaders::shader! {
//         ty: "fragment",
//         path: "src/frag.glsl"
//     }
// }
