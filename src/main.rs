use crossbeam::queue::SegQueue;
use egui::plot::{HLine, Line, Plot, Value, Values};
use egui::{Color32, Rounding, Ui, WidgetText};
use std::{env, fs};
// use egui_dock::Tree;
use puffin_egui::*;

use nalgebra_glm as glm;
use parking_lot::{Mutex, RwLock};
use vulkano::buffer::{BufferContents, BufferSlice, TypedBufferAccess};

use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Features;
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
// use rand::prelude::*;
// use rapier3d::prelude::*;

use crate::engine::physics::Physics;
use crate::engine::transform::{Transform, Transforms};
use crate::engine::{GameObject, RenderJobData, Sys, World};
use crate::game::{game_thread_fn, Bomb};
use crate::inspectable::{Inpsect, Ins};
use crate::model::ModelManager;
use crate::particles::cs::ty::t;
use crate::particles::ParticleEmitter;
use crate::perf::Perf;

use crate::project::{load_project, save_project};
use crate::renderer_component2::{ur, Renderer, RendererData, RendererManager};
use crate::terrain::Terrain;
use crate::texture::TextureManager;
use crate::transform_compute::cs;
use crate::transform_compute::cs::ty::transform;
use crate::{drag_drop::drag_source, drag_drop::drop_target};
use crate::{input::Input, renderer::RenderPipeline};

use rayon::prelude::*;

// enum TransformDrag {
//     DragToTransform(i32,i32),
//     DragBetweenTransform(i32,i32,bool)

// }
// lazy_static::lazy_static! {

//     static ref dragged_transform: Mutex<i32> = Mutex::new(0);
//     static ref transform_drag: Mutex<Option<TransformDrag>> = Mutex::new(None);

// }

// use egui::{
//     color_picker::{color_picker_color32, Alpha},
//     Id, LayerId, Slider,
// };

// use egui_dock::{DockArea, NodeIndex, Style, TabViewer};

// struct MyTab {
//     title: String,
//     world: Arc<Mutex<World>>,
//     f: fn(&mut World, &mut Ui)
// }

// struct MyTabs {
//     tree: Tree<MyTab>
// }

// struct MyContext {
//     pub title: String,
//     pub age: u32,
//     pub style: Option<Style>,
// }

// impl TabViewer for MyContext {
//     type Tab = String;

//     fn ui(&mut self, ui: &mut Ui, tab: &mut Self::Tab) {
//         // match tab.as_str() {
//         //     "Simple Demo" => self.simple_demo(ui),
//         //     "Style Editor" => self.style_editor(ui),
//         //     _ => {
//         //         ui.label(tab.as_str());
//         //     }
//         // }
//     }

//     // fn context_menu(&mut self, ui: &mut Ui, tab: &mut Self::Tab) {
//     //     match tab.as_str() {
//     //         "Simple Demo" => self.simple_demo_menu(ui),
//     //         _ => {
//     //             ui.label(tab.to_string());
//     //             ui.label("This is a context menu");
//     //         }
//     //     }
//     // }

//     fn title(&mut self, tab: &mut Self::Tab) -> WidgetText {
//         tab.as_str().into()
//     }
// }

//  use egui_dock::{NodeIndex};

//  struct MyTabs {
//      tree: Tree<String>
//  }

//  impl MyTabs {
//     //  pub fn new() -> Self {
//     //      let tab1 = "tab1".to_string();
//     //      let tab2 = "tab2".to_string();

//     //      let mut tree = Tree::new(vec![tab1]);
//     //      tree.split_left(NodeIndex::root(), 0.20, vec![tab2]);

//     //      Self { tree }
//     //  }

//      fn ui(&mut self, ui: &mut egui::Ui) {
//          let style = egui_dock::Style::from_egui(ui.style().as_ref());
//          egui_dock::DockArea::new(&mut self.tree)
//              .style(style)
//              .show_inside(ui, &mut TabViewer {});
//      }
//  }

//  struct TabViewer {}

//  impl egui_dock::TabViewer for TabViewer {
//      type Tab = String;

//      fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Self::Tab) {
//          ui.label(format("Content of {tab}"));
//      }

//      fn title(&mut self, tab: &mut Self::Tab) -> egui::WidgetText {
//          (&*tab).into()
//      }
//  }
//
//  # let mut my_tabs = MyTabs::new();
//  # egui::__run_test_ctx(|ctx| {
//  #     egui::CentralPanel::default().show(ctx, |ui| my_tabs.ui(ui));
//  # });

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

    // rayon::ThreadPoolBuilder::new().num_threads(63).build_global().unwrap();

    let event_loop = EventLoop::new();

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .unwrap();

    let surface = WindowBuilder::new()
        .with_inner_size(LogicalSize {
            width: 1920,
            height: 1080,
        })
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
            _ => panic!("no device"),
        })
        .unwrap();

    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let features = Features {
        geometry_shader: true,
        ..Default::default()
    };

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: physical_device
                .required_extensions()
                .union(&device_extensions),
            enabled_features: features,
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
                present_mode: vulkano::swapchain::PresentMode::Immediate,
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

    let model_manager = ModelManager {
        device: device.clone(),
        models: HashMap::new(),
        models_ids: HashMap::new(),
        texture_manager,
        model_id_gen: 0,
    };

    let renderer_manager = Arc::new(RwLock::new(RendererManager::new(device.clone())));

    // let cube_mesh = Mesh::load_model("src/cube/cube.obj", device.clone(), texture_manager.clone());

    let model_manager = Arc::new(Mutex::new(model_manager));
    {
        model_manager.lock().from_file("src/cube/cube.obj");
    }

    // let uniform_buffer =
    //     CpuBufferPool::<renderer::vs::ty::Data>::new(device.clone(), BufferUsage::all());

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
            { color: [color], depth_stencil: {depth}, input: [] }, // for secondary cmmand buffers
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
        0,
    );
    let rend2 = RenderPipeline::new(
        device.clone(),
        render_pass.clone(),
        images[0].dimensions().width_height(),
        queue.clone(),
        1,
    );
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    // let rotation_start = Instant::now();

    let mut modifiers = ModifiersState::default();

    //////////////////////////////////////////////////

    let mut particles = Arc::new(particles::ParticleCompute::new(
        device.clone(),
        render_pass.clone(),
        // swapchain.clone(),
        queue.clone(),
    ));

    let mut input = Input {
        ..Default::default()
    };

    // let mut perf = HashMap::<String, SegQueue<Duration>>::new();
    let mut perf = Perf {
        data: HashMap::<String, SegQueue<Duration>>::new(),
    };

    let _loops = 0;

    let egui_ctx = egui::Context::default();
    let mut egui_winit = egui_winit::State::new(4096, surface.window());

    let mut egui_painter = egui_vulkano::Painter::new(
        device.clone(),
        queue.clone(),
        Subpass::from(render_pass.clone(), 2).unwrap(),
    )
    .unwrap();

    //Set up some window to look at for the test

    // let mut egui_test = egui_demo_lib::ColorTest::default();
    // let mut demo_windows = egui_demo_lib::DemoWindows::default();
    // let mut egui_bench = Benchmark::new(1000);
    // let mut my_texture = egui_ctx.load_texture("my_texture", ColorImage::example());

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

    let transform_uniforms = CpuBufferPool::<cs::ty::Data>::new(device.clone(), BufferUsage::all());

    let mut fps_queue = std::collections::VecDeque::new();
    // let render_uniforms = CpuBufferPool::<vs::ty::UniformBufferObject>::new(device.clone(), BufferUsage::all());

    let mut focused = true;

    // puffin::set_scopes_on(true);

    let mut cull_view = glm::Mat4::identity();
    let mut lock_cull = false;
    let mut first_frame = true;

    /////////////////////////////////////////////////////////////////////////////////////////
    let (tx, rx): (Sender<Input>, Receiver<Input>) = mpsc::channel();
    let (rtx, rrx): (
        Sender<(
            Arc<(
                usize,
                Vec<Arc<(Vec<Vec<i32>>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>)>>,
            )>,
            glm::Vec3,
            glm::Quat,
            RendererData,
            (usize, Vec<crate::particles::cs::ty::emitter_init>),
            // Arc<(Vec<Offset>, Vec<Id>)>,
            // Arc<&HashMap<i32, HashMap<i32, Mesh>>>,
        )>,
        Receiver<(
            Arc<(
                usize,
                Vec<Arc<(Vec<Vec<i32>>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<[f32; 3]>)>>,
            )>,
            glm::Vec3,
            glm::Quat,
            RendererData,
            (usize, Vec<crate::particles::cs::ty::emitter_init>),
            // Arc<(Vec<Offset>, Vec<Id>)>,
            // Arc<&HashMap<i32, HashMap<i32, Mesh>>>,
        )>,
    ) = mpsc::channel();
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
        renderer_manager.clone(),
        physics,
        particles.clone(),
        device.clone(),
    )));
    {
        let mut world = world.lock();
        world.register::<Renderer>(false, false);
        world.register::<ParticleEmitter>(false, false);
        // world.register::<Maker>(true);
        world.register::<Terrain>(true, true);
        world.register::<Bomb>(true, false);
    }

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
        file_watcher.init(&mut world);
        load_project(&mut file_watcher, &mut world)
    }
    // let mut selected_transforms: HashMap<i32, bool> = HashMap::<i32, bool>::new();

    // let mut selected = None;

    event_loop.run(move |event, _, control_flow| {
        // let game_thread = game_thread.clone();
        *control_flow = ControlFlow::Poll;
        match event {
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta } => {
                    if focused {
                        input.mouse_x = delta.0;
                        input.mouse_y = delta.1;
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

                        save_project(&file_watcher, &world.lock());

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
                    let egui_consumed_event = egui_winit.on_event(&egui_ctx, &event);
                    if !egui_consumed_event {
                        // do your own event handling here
                    };
                }
            }
            Event::RedrawEventsCleared => {
                puffin::GlobalProfiler::lock().new_frame();

                puffin::profile_scope!("full");
                ////////////////////////////////////

                let full = Instant::now();

                if input.get_key(&VirtualKeyCode::Escape) {
                    *control_flow = ControlFlow::Exit;
                    running.store(false, Ordering::SeqCst);
                    let game_thread = game_thread.remove(0);
                    let _res = coms.1.send(input.clone());

                    game_thread.join().unwrap();

                    perf.print();
                }
                static mut GRAB_MODE: bool = true;
                if input.get_key_press(&VirtualKeyCode::G) {
                    unsafe {
                        let _er = surface.window().set_cursor_grab(GRAB_MODE);
                        GRAB_MODE = !GRAB_MODE;
                    }
                }
                if input.get_key_press(&VirtualKeyCode::J) {
                    lock_cull = !lock_cull;
                    // lock_cull.
                }

                if input.get_key(&VirtualKeyCode::H) {
                    surface.window().set_cursor_visible(modifiers.shift());
                }

                let (transform_data, cam_pos, cam_rot, rd, emitter_inits) = {
                    puffin::profile_scope!("wait for game");
                    let inst = Instant::now();
                    let (transform_data, cam_pos, cam_rot, renderer_data, emitter_inits) =
                        coms.0.recv().unwrap();

                    perf.update("wait for game".into(), Instant::now() - inst);
                    (transform_data, cam_pos, cam_rot, renderer_data, emitter_inits)
                };

                egui_ctx.begin_frame(egui_winit.take_egui_input(surface.window()));

                file_watcher.get_updates(&mut world.lock());

                editor_ui::editor_ui(&world, &mut fps_queue, &egui_ctx);

                let render_jobs = world.lock().render();

                let rm = renderer_manager.read();
                let mut rm = rm.shr_data.write();

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
                    let position_update_data = transform_compute
                        .get_position_update_data(device.clone(), transform_data.clone());

                    let rotation_update_data = transform_compute
                        .get_rotation_update_data(device.clone(), transform_data.clone());

                    let scale_update_data = transform_compute
                        .get_scale_update_data(device.clone(), transform_data.clone());
                    (
                        position_update_data,
                        rotation_update_data,
                        scale_update_data,
                    )
                };
                perf.update("write to buffer".into(), Instant::now() - inst);

                // render_thread = thread::spawn( || {
                let inst = Instant::now();

                let dimensions = surface.window().inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }
                previous_frame_end.as_mut().unwrap().cleanup_finished();

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
                }

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

                // Get the shapes from egui
                let egui_output = egui_ctx.end_frame();
                let platform_output = egui_output.platform_output;
                egui_winit.handle_platform_output(surface.window(), &egui_ctx, platform_output);

                let _result = egui_painter
                    .update_textures(egui_output.textures_delta, &mut builder)
                    .expect("egui texture error");

                {
                    puffin::profile_scope!("transform update compute");

                    // compute shader transforms
                    transform_compute.update(device.clone(), &mut builder, transform_data.0);
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
                    );

                    // stage 1
                    transform_compute.update_rotations(
                        &mut builder,
                        view.clone(),
                        proj.clone(),
                        &transform_uniforms,
                        compute_pipeline.clone(),
                        rotation_update_data,
                    );

                    // stage 2
                    transform_compute.update_scales(
                        &mut builder,
                        view.clone(),
                        proj.clone(),
                        &transform_uniforms,
                        compute_pipeline.clone(),
                        scale_update_data,
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
                            rm.transform_ids_gpu = CpuAccessibleBuffer::uninitialized_array(
                                device.clone(),
                                max_len as u64,
                                BufferUsage::all(),
                                false,
                            )
                            .unwrap();
                            rm.renderers_gpu = CpuAccessibleBuffer::uninitialized_array(
                                device.clone(),
                                max_len as u64,
                                BufferUsage::all(),
                                false,
                            )
                            .unwrap();
                        }

                        builder
                            .copy_buffer(copy_buffer, rm.transform_ids_gpu.clone())
                            .unwrap();
                    }
                    // let mut updates = rd.updates.lock().unwrap_unchecked();
                    // std::mem::swap(&mut updates, &mut rd.updates.as_mut());
                    // let updates = rd.updates.clone();
                    // let updates: Vec<i32> = _rm
                    //     .updates
                    //     .iter()
                    //     .flat_map(|(id, t)| {
                    //         vec![id.clone(), t.indirect_id.clone(), t.transform_id.clone()]
                    //             .into_iter()
                    //     })
                    //     .collect();

                    // let mut indirect_vec = Vec::new();
                    // for i in &rm.indirect.data {
                    //     indirect_vec.push(i.clone());
                    // }
                    rm.indirect_buffer = CpuAccessibleBuffer::from_iter(
                        device.clone(),
                        BufferUsage::all(),
                        false,
                        rm.indirect.data.clone(),
                    )
                    .unwrap();

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
                            device.clone(),
                            BufferUsage::all(),
                            false,
                            offset_vec,
                        )
                        .unwrap();

                        {
                            puffin::profile_scope!("update renderers: stage 0");
                            let update_num = rd.updates.len() / 3;
                            if update_num > 0 {
                                rm.updates_gpu = CpuAccessibleBuffer::from_iter(
                                    device.clone(),
                                    BufferUsage::all(),
                                    false,
                                    rd.updates,
                                )
                                .unwrap();
                            }

                            // stage 0
                            let uniforms = rm
                                .uniform
                                .next(ur::ty::Data {
                                    num_jobs: update_num as i32,
                                    stage: 0,
                                    view: cull_view.into(),
                                    _dummy0: Default::default(),
                                })
                                .unwrap();

                            let update_renderers_set = PersistentDescriptorSet::new(
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
                                    .next(ur::ty::Data {
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
                );
                particles.particle_update(
                    device.clone(),
                    &mut builder,
                    transform_compute.transform.clone(),
                    input.time.dt,
                    input.time.time,
                    cam_pos.into(),
                    cam_rot.coords.into(),
                );
                particles.sort.sort(
                    view.into(),
                    particles.particles.clone(),
                    particles.particle_positions_lifes.clone(),
                    device.clone(),
                    queue.clone(),
                    &mut builder,
                );

                {
                    puffin::profile_scope!("render meshes");

                    builder
                        .begin_render_pass(
                            framebuffers[image_num].clone(),
                            SubpassContents::Inline,
                            vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()], // clear color
                        )
                        .unwrap()
                        .set_viewport(0, [viewport.clone()]);

                    rend.bind_pipeline(&mut builder);

                    {
                        let mm = model_manager.lock();

                        let mut offset = 0;

                        for (_ind_id, m_id) in rd.indirect_model.iter() {
                            if let Some(ind) = rd.model_indirect.get(&m_id) {
                                if let Some(mr) = mm.models_ids.get(&m_id) {
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
                                                &mut builder,
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
                    // builder.end_render_pass().unwrap();
                    // builder
                    // .begin_render_pass(
                    //     framebuffers[image_num].clone(),
                    //     SubpassContents::SecondaryCommandBuffers,
                    //     vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()], // clear color
                    // )
                    // .unwrap()
                    // .set_viewport(0, [viewport.clone()]);

                    // let particles = particles.read();
                    particles.render_particles(
                        &mut builder,
                        view.clone(),
                        proj.clone(),
                        cam_rot.coords.into(),
                        cam_pos.into(),
                    );

                    builder
                        .next_subpass(SubpassContents::SecondaryCommandBuffers)
                        .unwrap();

                    // rend.bind_pipeline(&mut builder);
                    let mut rjd = RenderJobData {
                        builder: &mut builder,
                        transforms: transform_compute.transform.clone(),
                        mvp: transform_compute.mvp.clone(),
                        view: &view,
                        proj: &proj,
                        pipeline: &rend2,
                        device: device.clone(),
                        viewport: &viewport,
                    };
                    for job in render_jobs {
                        job(&mut rjd);
                    }
                    // &mut builder,
                    // transform_compute.transform.clone(),
                    // &view,
                    // &proj,
                    // &rend,
                    // device.clone(),
                }

                // Automatically start the next render subpass and draw the gui
                let size = surface.window().inner_size();
                let sf: f32 = surface.window().scale_factor() as f32;
                builder.set_viewport(0, [viewport.clone()]);
                egui_painter
                    .draw(
                        &mut builder,
                        [(size.width as f32) / sf, (size.height as f32) / sf],
                        &egui_ctx,
                        egui_output.shapes,
                    )
                    .unwrap();

                input.time.time += input.time.dt;
                input.time.dt = (frame_time.elapsed().as_secs_f64() as f32).min(0.1);

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
                            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
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
    entities: usize,
    data: VecDeque<f64>,
}

impl Benchmark {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entities: 0,
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
        ui.label(format!("entities: {}", self.entities));
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

    pub fn push(&mut self, v: f64, ent: usize) {
        self.entities = ent;
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(v);
    }
}
