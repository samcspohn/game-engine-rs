use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    env, fs,
    mem::size_of,
    ops::Add,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle, Thread},
    time::{Duration, Instant},
};

use crossbeam::{
    channel::{Receiver, Sender},
    queue::SegQueue,
};
use egui::TextureId;
use egui_winit_vulkano::{Gui, GuiConfig};
use force_send_sync::SendSync;
use glm::{quat_euler_angles, vec3, Vec3};
use lazy_static::lazy_static;
use num_integer::Roots;
use once_cell::sync::Lazy;
use puffin_egui::puffin;
use rapier3d::{
    na::{ComplexField, UnitQuaternion},
    prelude::*,
};
use serde::{Deserialize, Serialize};

use rayon::prelude::*;

use nalgebra_glm as glm;
use parking_lot::{Mutex, RwLock};
use thincollections::thin_map::ThinMap;
use vulkano::{
    buffer::{allocator::SubbufferAllocator, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, BlitImageInfo,
        CommandBufferUsage, CopyImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract,
        RenderPassBeginInfo, SubpassContents,
    },
    format::Format,
    image::{
        view::ImageView, AttachmentImage, ImageAccess, ImageDimensions, ImageUsage, ImmutableImage,
        MipmapsCount, StorageImage, SwapchainImage,
    },
    memory::allocator::MemoryUsage,
    pipeline::graphics::viewport::Viewport,
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    sampler::{SamplerAddressMode, SamplerCreateInfo, LOD_CLAMP_NONE},
    swapchain::{
        acquire_next_image, AcquireError, SwapchainAcquireFuture, SwapchainCreateInfo,
        SwapchainCreationError, SwapchainPresentInfo,
    },
    sync::{self, FlushError, GpuFuture},
    DeviceSize,
};
use winit::{
    dpi::{LogicalPosition, PhysicalSize},
    event::{Event, ModifiersState, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop, EventLoopBuilder, EventLoopProxy},
};
// use crate::{physics::Physics};

use crate::{
    editor::{self, editor_cam::EditorCam, editor_ui::EDITOR_WINDOW_DIM},
    engine::{
        audio::{
            asset::AudioManager,
            component::{AudioListener, AudioSource},
        },
        particles::{component::ParticleEmitter, shaders::scs::l},
        physics::{collider::_Collider, rigid_body::_RigidBody, Physics},
        // utils::look_at,
        prelude::{TransformRef, _Transform},
        project::{asset_manager::AssetManagerBase, serialize::serialize},
        rendering::{
            lighting::{
                lighting::{Light, LightingSystem},
                lighting_asset::{self, LightTemplateManager},
            },
            model::ModelManager,
        },
        storage::Storage,
        world::{
            transform::{Transforms, TRANSFORMS, TRANSFORM_MAP},
            NewCollider, NewRigidBody,
        },
    },
};

use self::{
    input::Input,
    particles::particles::ParticlesSystem,
    perf::Perf,
    physics::collider::_ColliderType,
    prelude::{Component, Inpsect, Ins, Sys, System, Transform},
    project::{
        asset_manager::AssetsManager,
        file_watcher::{self, FileWatcher},
        save_project, serialize, Project,
    },
    render_thread::RenderingData,
    rendering::{
        camera::{Camera, CameraData, CameraViewData},
        component::{Renderer, SharedRendererData},
        lighting::{
            // light_bounding::LightBounding,
            lighting_compute::{lt, LightingCompute},
        },
        model::{Mesh, ModelRenderer},
        pipeline::{
            fs::{light, lightTemplate},
            RenderPipeline,
        },
        texture::{Texture, TextureManager},
        vulkan_manager::VulkanManager,
    },
    time::Time,
    transform_compute::{
        cs::{transform, MVP},
        TransformCompute,
    },
    utils::GPUWork,
    world::World,
};

pub mod atomic_vec;
pub mod audio;
pub mod input;
pub mod linked_list;
pub mod perf;
pub mod project;
pub mod rendering;
pub mod runtime_compilation;
pub mod storage;
mod terrain_eng;
pub mod time;
pub mod transform_compute;
pub mod world;
// mod render_pipeline;
pub mod color_gradient;
mod input_thread;
pub mod main_loop;
pub mod particles;
pub mod physics;
pub(crate) mod prelude;
mod render_thread;
pub mod utils;
#[repr(C)]
pub struct RenderJobData<'a> {
    pub builder: &'a mut AutoCommandBufferBuilder<
        PrimaryAutoCommandBuffer,
        Arc<StandardCommandBufferAllocator>,
    >,
    // pub uniforms: Arc<Mutex<SubbufferAllocator>>,
    pub gpu_transforms: Subbuffer<[transform]>,
    pub light_len: u32,
    pub lights: Subbuffer<[lt::light]>,
    pub light_templates: Subbuffer<[lightTemplate]>,
    pub light_list: Subbuffer<[u32]>,
    pub visible_lights: Subbuffer<[u32]>,
    pub visible_lights_count: Subbuffer<u32>,
    // pub light_buckets: Subbuffer<[u32]>,
    // pub light_buckets_count: Subbuffer<[u32]>,
    // pub light_ids: Subbuffer<[u32]>,
    pub tiles: Subbuffer<[lt::tile]>,
    pub mvp: Subbuffer<[MVP]>,
    pub cam_pos: Vec3,
    pub screen_dims: [f32; 2],
    pub view: &'a nalgebra_glm::Mat4,
    pub proj: &'a nalgebra_glm::Mat4,
    pub pipeline: &'a RenderPipeline,
    pub viewport: &'a Viewport,
    pub texture_manager: &'a TextureManager,
    pub vk: Arc<VulkanManager>,
}

// pub struct ComponentRenderData {
//     pub vertex_buffer: Arc<Vec<Vertex>>,
//     pub normals_buffer: Arc<Vec<Normal>>,
//     pub uvs_buffer: Arc<Vec<UV>>,
//     pub index_buffer: Arc<Vec<u32>>,
//     pub texture: Option<i32>,
//     pub instance_buffer: Arc<Vec<i32>>
// }
pub struct Defer {
    work: SegQueue<Box<dyn FnOnce(&mut World) + Send + Sync>>,
}

impl Defer {
    pub fn append<T: 'static>(&self, f: T)
    where
        T: FnOnce(&mut World) + Send + Sync,
    {
        self.work.push(Box::new(f));
    }
    pub fn do_defered(&self, wrld: &mut World) {
        while let Some(w) = self.work.pop() {
            w(wrld);
        }
    }
    pub fn new() -> Defer {
        Defer {
            work: SegQueue::new(),
        }
    }
}

pub(crate) enum EngineEvent {
    Send,
    Quit,
}
struct EngineRenderer {
    viewport: Viewport,
    framebuffers: Vec<(Arc<SwapchainImage>, Arc<Framebuffer>)>,
    recreate_swapchain: bool,
    editor_window_image: Option<Arc<dyn ImageAccess>>,
    render_pass: Arc<RenderPass>,
    // previous_frame_end: Option<Box<dyn GpuFuture>>,
}
pub struct Engine {
    pub world: Arc<Mutex<World>>,
    pub(crate) assets_manager: Arc<AssetsManager>,
    pub(crate) recompile: Arc<AtomicBool>,
    pub(crate) project: Project,
    pub(crate) transform_compute: RwLock<TransformCompute>,
    pub(crate) lighting_compute: RwLock<LightingCompute>,
    // pub(crate) light_bounding: RwLock<LightBounding>,
    pub(crate) particles_system: Arc<ParticlesSystem>,
    pub(crate) lighting_system: Arc<LightingSystem>,
    pub(crate) playing_game: bool,
    // pub(crate) event_loop: EventLoop<()>,
    // channels
    pub(crate) input: Receiver<(
        Vec<WindowEvent<'static>>,
        Input,
        Option<PhysicalSize<u32>>,
        bool,
    )>,
    pub(crate) rendering_complete: Receiver<bool>,
    pub(crate) rendering_data:
        Sender<Option<(bool, u32, SwapchainAcquireFuture, PrimaryAutoCommandBuffer)>>,
    pub(crate) rendering_thread: Arc<JoinHandle<()>>,
    phys_upd_start: Sender<(bool, Arc<Mutex<Physics>>)>,
    phys_upd_compl: Receiver<()>,
    phys_step_compl: Receiver<()>,
    // end of channels
    pub(crate) cam_data: Arc<Mutex<CameraData>>,
    pub(crate) editor_cam: EditorCam,
    pub time: Time,
    pub perf: Arc<Perf>,
    pub(crate) vk: Arc<VulkanManager>,
    pub(crate) shared_render_data: Arc<RwLock<SharedRendererData>>,
    pub(crate) fps_queue: VecDeque<f32>,
    pub(crate) frame_time: Instant,
    pub(crate) running: Arc<AtomicBool>,
    pub(crate) input_thread: Arc<JoinHandle<()>>,
    pub(crate) physics_thread: Arc<JoinHandle<()>>,
    pub(crate) compiler_process: Option<std::process::Child>,
    // pub(crate) working_scene: String,
    pub(crate) file_watcher: FileWatcher,
    pub(crate) gui: SendSync<Gui>,
    pub(crate) tex_id: Option<TextureId>,
    pub(crate) image_view: Option<Arc<ImageView<ImmutableImage>>>,
    engine_dir: String,
    game_mode: bool,
    update_editor_window: bool,
    event_loop_proxy: EventLoopProxy<EngineEvent>,
    renderer: EngineRenderer,
}
pub struct EnginePtr {
    ptr: *const Engine,
}
unsafe impl Send for EnginePtr {}
unsafe impl Sync for EnginePtr {}

impl Engine {
    pub fn new(engine_dir: &PathBuf, project_dir: &str, game_mode: bool) -> Self {
        let event_loop: SendSync<EventLoop<EngineEvent>> =
            unsafe { SendSync::new(EventLoopBuilder::with_user_event().build()) };
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
            (texture_manager.clone(), vk.clone()),
            &["obj"],
        )));
        let recompiled = Arc::new(AtomicBool::new(false));
        let rs_manager = Arc::new(Mutex::new(runtime_compilation::RSManager::new(
            (recompiled.clone()),
            &["rs"],
        )));

        assert!(env::set_current_dir(&Path::new(engine_dir)).is_ok()); // procedurally generate cube/move cube to built in assets
        model_manager.lock().from_file("default/cube/cube.obj");

        texture_manager.lock().from_file("default/particle.png");
        assert!(env::set_current_dir(&Path::new(project_dir)).is_ok());

        let particles_system = Arc::new(ParticlesSystem::new(vk.clone(), texture_manager.clone()));
        let lighting_system = Arc::new(LightingSystem::new(vk.clone()));
        let light_manager = Arc::new(Mutex::new(LightTemplateManager::new(
            (lighting_system.light_templates.clone()),
            &["lgt"],
        )));
        let world = Arc::new(Mutex::new(World::new(
            particles_system.clone(),
            lighting_system.clone(),
            vk.clone(),
            assets_manager.clone(),
        )));
        unsafe {
            TRANSFORMS = &mut world.lock().transforms;
            TRANSFORM_MAP = &mut world.lock().transform_map;
        }

        #[cfg(target_os = "windows")]
        let dylib_ext = ["dll"];
        #[cfg(not(target_os = "windows"))]
        let dylib_ext = ["so"];

        let lib_manager = Arc::new(Mutex::new(runtime_compilation::LibManager::new(
            world.clone(),
            &dylib_ext,
        )));
        let audio_manager = Arc::new(Mutex::new(AudioManager::new(
            world.lock().sys.audio_manager.m.clone(),
            &["mp3", "wav", "ogg", "flac"],
        )));

        unsafe {
            if !game_mode {
                assets_manager.add_asset_manager("rs", rs_manager.clone());
                assets_manager.add_asset_manager("lib", lib_manager.clone());
            }
            assets_manager.add_asset_manager("model", model_manager.clone());
            assets_manager.add_asset_manager("texture", texture_manager.clone());
            assets_manager.add_asset_manager(
                "particle_template",
                particles_system.particle_template_manager.clone(),
            );
            assets_manager.add_asset_manager("light", light_manager.clone());
            assets_manager.add_asset_manager("audio", audio_manager.clone())
        }

        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.0..1.0,
        };

        //////////////////////////////////////////////////

        let mut input = Input {
            ..Default::default()
        };

        let mut perf = Perf::new();

        let mut frame_time = Instant::now();

        let mut transform_compute = TransformCompute::new(vk.clone());

        let mut fps_queue: VecDeque<f32> = std::collections::VecDeque::new();

        let running = Arc::new(AtomicBool::new(true));

        {
            let mut world = world.lock();
            world.register::<Renderer>(false, false, false);
            world.register::<ParticleEmitter>(false, false, false);
            world.register::<Camera>(false, false, false);
            world.register::<_Collider>(false, false, false);
            world.register::<_RigidBody>(false, false, false);
            world.register::<Light>(false, false, false);
            world.register::<AudioSource>(true, false, false);
            world.register::<AudioListener>(true, false, false);
            //
            world.register::<terrain_eng::TerrainEng>(true, false, true);
        };

        let rm = {
            let w = world.lock();
            let rm = w.sys.renderer_manager.read();
            rm.shr_data.clone()
        };
        let mut file_watcher = file_watcher::FileWatcher::new(".");

        // let mut cam_data = CameraData::new(vk.clone(), 2);
        let (input_snd, input_rcv) = crossbeam::channel::bounded(1);
        let mut gui = egui_winit_vulkano::Gui::new_with_subpass(
            &event_loop,
            vk.surface.clone(),
            vk.queue.clone(),
            Subpass::from(render_pass.clone(), 0).unwrap(),
            vk.swapchain().image_format(),
            GuiConfig {
                // preferred_format: Some(vk.swapchain().image_format()),
                allow_srgb_render_target: true,
                is_overlay: true,
                ..Default::default()
            },
        );
        let proxy = event_loop.create_proxy();
        let (rendering_snd, rendering_rcv) = crossbeam::channel::bounded(1);
        let (rendering_snd2, rendering_rcv2) = crossbeam::channel::bounded(1);
        let (phys_snd, phys_rcv) = crossbeam::channel::bounded(1);
        let (phys_snd2, phys_rcv2) = crossbeam::channel::bounded(1);
        let (phys_snd3, phys_rcv3) = crossbeam::channel::bounded(1);
        // let (rendering_snd, rendering_rcv) = crossbeam::channel::bounded(1);
        let input_thread = Arc::new({
            let vk = vk.clone();
            thread::spawn(move || input_thread::input_thread(event_loop, vk, input_snd))
        });
        let rendering_thread = Arc::new({
            let vk = vk.clone();
            thread::spawn(move || render_thread::render_thread(vk, rendering_rcv, rendering_snd2))
        });
        let perf = Arc::new(perf);
        // let mut phys = world.lock().sys.physics.clone();
        let physics_thread = Arc::new({
            // let vk = vk.clone();
            let perf = perf.clone();
            let world: SendSync<*const World> = unsafe { SendSync::new(&mut *world.lock()) };
            thread::spawn(move || {
                physics::physics_thread(world, perf, phys_rcv, phys_snd2, phys_snd3);
            })
        });
        proxy.send_event(EngineEvent::Send);

        let mut viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [0.0, 0.0],
            depth_range: 0.0..1.0,
        };

        let framebuffers =
            window_size_dependent_setup(&vk.images, render_pass.clone(), &mut viewport, &vk);
        let mut recreate_swapchain = true;
        let mut fc_map: HashMap<i32, HashMap<u32, TextureId>> = HashMap::new();
        let mut editor_window_image: Option<Arc<dyn ImageAccess>> = None;
        println!(
            "default quat: {}",
            glm::quat_look_at_lh(&Vec3::z(), &Vec3::y()).coords
        );

        Self {
            world,
            assets_manager,
            project: Project::default(),
            recompile: recompiled,
            transform_compute: RwLock::new(transform_compute),
            lighting_compute: RwLock::new(LightingCompute::new(vk.clone(), render_pass.clone())),
            // light_bounding: RwLock::new(LightBounding::new(vk.clone())),
            playing_game: game_mode,
            // coms,
            perf,
            shared_render_data: rm,
            input: input_rcv,
            rendering_data: rendering_snd,
            rendering_complete: rendering_rcv2,
            phys_upd_start: phys_snd,
            phys_upd_compl: phys_rcv2,
            phys_step_compl: phys_rcv3,
            cam_data: Arc::new(Mutex::new(CameraData::new(vk.clone(), 1))),
            renderer: EngineRenderer {
                viewport,
                framebuffers,
                recreate_swapchain,
                editor_window_image,
                render_pass,
                // previous_frame_end: Some(sync::now(vk.device.clone()).boxed()),
            },
            vk,
            editor_cam: editor::editor_cam::EditorCam {
                rot: glm::quat_look_at_lh(&Vec3::z(), &Vec3::y()),
                pos: Vec3::zeros(),
                speed: 30f32,
            },
            time: Time::default(),
            particles_system,
            lighting_system,
            fps_queue,
            frame_time,
            running,
            input_thread,
            rendering_thread,
            physics_thread,
            compiler_process: None,
            // working_scene: "test.scene".into(),
            file_watcher,
            tex_id: None,
            image_view: None,
            game_mode,
            gui: unsafe { SendSync::new(gui) },
            update_editor_window: true,
            event_loop_proxy: proxy,
            engine_dir: engine_dir.as_path().to_str().unwrap().to_string(),
        }
        //     event_loop,
        // )
    }
    pub fn init(&mut self) {
        self.project = if let Ok(s) = std::fs::read_to_string("project.yaml") {
            // {
            let project: Project = serde_yaml::from_str(s.as_str()).unwrap();
            self.file_watcher.files = project.files.clone();
            self.assets_manager.deserialize(&project.assets);
            self.file_watcher.init(self.assets_manager.clone());
            // }
            // serialize::deserialize(&mut world.lock());
            project
        } else {
            Project::default()
        };
        // TODO: remove in favor of placeholder component
        let mut args = vec!["build"];
        #[cfg(not(debug_assertions))]
        {
            println!("compiling for release");
            args.push("-r");
        }
        // args.push("-r");
        let com = Command::new("cargo")
            .args(args.as_slice())
            .env("RUSTFLAGS", "-Z threads=16")
            .status()
            .unwrap();
        self.file_watcher.get_updates(self.assets_manager.clone());
        serialize::deserialize(&mut self.world.lock(), &self.project.working_scene);
    }

    // pub fn run(mut self, event_loop: EventLoop<()>) {
    //     let mut focused = true;
    //     let mut input = Input::default();
    //     let mut modifiers = ModifiersState::default();
    //     // let mut recreate_swapchain = true;
    //     let mut size = None;
    //     let mut should_quit = false;
    //     // let mut event_loop = std::mem::take(&mut self.event_loop);
    //     event_loop.run(move |event, _, control_flow| {
    //         *control_flow = ControlFlow::Poll;
    //         match event {
    //             Event::DeviceEvent { event, .. } => input.process_device(event, focused),
    //             Event::WindowEvent { event, .. } => {
    //                 match event {
    //                     WindowEvent::Focused(foc) => {
    //                         focused = foc;
    //                         if !focused {
    //                             let _er = self
    //                                 .vk
    //                                 .window()
    //                                 .set_cursor_grab(winit::window::CursorGrabMode::None);
    //                             match _er {
    //                                 Ok(_) => {}
    //                                 Err(e) => {}
    //                             }
    //                         }
    //                     }
    //                     WindowEvent::CloseRequested => {
    //                         should_quit = true;
    //                         // *control_flow = ControlFlow::Exit;
    //                         // engine.end();
    //                     }
    //                     WindowEvent::MouseInput {
    //                         device_id,
    //                         state,
    //                         button,
    //                         modifiers,
    //                     } => input.process_mouse_input(device_id, state, button, modifiers),
    //                     WindowEvent::MouseWheel {
    //                         device_id,
    //                         delta,
    //                         phase,
    //                         modifiers,
    //                     } => input.process_mouse_wheel(device_id, delta, phase, modifiers),

    //                     WindowEvent::KeyboardInput {
    //                         input: ky_input,
    //                         device_id,
    //                         is_synthetic,
    //                     } => input.process_keyboard(device_id, ky_input, is_synthetic),
    //                     WindowEvent::ModifiersChanged(m) => modifiers = m,
    //                     WindowEvent::Resized(_size) => {
    //                         // recreate_swapchain = true;
    //                         size = Some(_size);
    //                     }
    //                     _ => (),
    //                 }
    //                 self.gui.update(&event);
    //                 // if let Some(event) = event.to_static() {
    //                 //     events.push(event);
    //                 // }
    //             }
    //             Event::RedrawEventsCleared => {
    //                 self.update_sim(input.clone(), size, should_quit);
    //                 size = None;
    //                 input.reset();
    //             }
    //             // Event::UserEvent(e) => {
    //             //     match e {
    //             //         EngineEvent::Send => {
    //             //             // let mut a = Vec::new();
    //             //             // swap(&mut a, &mut events);
    //             //             coms.send((events.clone(), input.clone(), size, should_quit));
    //             //             // recreate_swapchain = false;
    //             //             size = None;
    //             //             events.clear();
    //             //             input.reset();
    //             //         }
    //             //         EngineEvent::Quit => {
    //             //             // todo!()
    //             //             *control_flow = ControlFlow::Exit;
    //             //             return;
    //             //         }
    //             //     }
    //             // }
    //             _ => {} // Event::RedrawEventsCleared => {}
    //         }
    //     });
    // }

    pub fn update_sim(&mut self) -> bool {
        let full_frame_time = self.perf.node("full frame time");
        let (events, input, window_size, should_exit) = self.input.recv().unwrap();
        for event in events {
            self.gui.update(&event);
        }
        let EngineRenderer {
            viewport,
            framebuffers,
            recreate_swapchain,
            editor_window_image,
            render_pass,
            // previous_frame_end,
        } = &mut self.renderer;

        let mut world = self.world.lock();
        world.begin_frame(input.clone(), self.time.clone());
        // let world_sim = self.perf.node("world _update");
        let cvd = if (self.game_mode | self.playing_game) {
            puffin::profile_scope!("game loop");
            {
                puffin::profile_scope!("world update");
                if world.phys_time >= world.phys_step {
                    if let Ok(_) = self.phys_step_compl.try_recv() {
                        // skip if physics sim not complete

                        let phys_sim = self.perf.node("wait for phyiscs");
                        self.phys_upd_start
                            .send((true, world.sys.physics.clone()))
                            .unwrap(); // update physics transforms/ add/remove colliders/rigid bodies
                        self.phys_upd_compl.recv().unwrap();
                        world.phys_time = 0f32;
                    }
                }
                world.phys_time += self.time.dt;
                let world_update = self.perf.node("world _update");
                world._update(&self.perf);
                drop(world_update);
            }
            {
                puffin::profile_scope!("defered");
                {
                    let world_do_defered = self.perf.node("world do_deffered");
                    world.do_defered();
                }
                {
                    let world_destroy = self.perf.node("world _destroy");
                    world._destroy(&self.perf);
                }
                {
                    let world_update = self.perf.node("world instantiate");
                    world.defer_instantiate(&self.perf);
                    // world.init_colls_rbs();
                }
            }
            // let mut world = self.world.lock();
            world.update_cameras()
        } else {
            // let mut world = self.world.lock();

            world._destroy(&self.perf);
            // world.init_colls_rbs();
            world.editor_update(); // TODO: terrain update still breaking

            world.do_defered();
            self.phys_upd_start
                .send((false, world.sys.physics.clone()))
                .unwrap(); // update physics transforms/ add/remove colliders/rigid bodies
            self.phys_upd_compl.recv().unwrap();

            self.editor_cam.update(&input, &self.time);
            let cvd = self.cam_data.lock().update(
                self.editor_cam.pos,
                self.editor_cam.rot,
                0.01f32,
                10_000f32,
                70f32,
                false,
                1,
            );
            vec![(Some(self.cam_data.clone()), Some(cvd))]
        };
        // let mut world = self.world.lock();

        // drop(world_sim);

        let get_renderer_data = self.perf.node("get renderer data");
        let mut renderer_data = world.sys.renderer_manager.write().get_renderer_data();
        drop(get_renderer_data);
        let emitter_len = world.get_emitter_len();
        let emitter_inits = world.sys.particles_system.emitter_inits.get_vec();
        let emitter_deinits = world.sys.particles_system.emitter_deinits.get_vec();
        let particle_bursts = world.sys.particles_system.particle_burts.get_vec();
        let (main_cam_id, mut cam_datas) = world.get_cam_datas();
        let render_jobs = world.render();

        let cd = if (self.game_mode | self.playing_game) && cam_datas.len() > 0 {
            cam_datas[0].clone()
        } else {
            let cd = self.cam_data.clone();
            cd
        };
        if !self.game_mode {}
        let mut _cd = cd.lock();
        let _gui = self.perf.node("_ gui");
        let dimensions = *EDITOR_WINDOW_DIM.lock();
        let mut _playing_game = false;
        self.gui.immediate_ui(|gui| {
            let ctx = gui.context();
            _playing_game = editor::editor_ui::editor_ui(
                &mut world,
                &mut self.fps_queue,
                &ctx,
                self.tex_id.unwrap_or_default(),
                self.assets_manager.clone(),
                &self.file_watcher,
                self.game_mode | self.playing_game,
            );
        });
        if _playing_game && _playing_game != self.playing_game {
            // save current state of scene before play
            serialize(&world, "temp_scene");
        }
        if !_playing_game && _playing_game != self.playing_game {
            // just clicked stop
            while self.phys_step_compl.is_empty() {
                thread::sleep(Duration::from_millis(10));
            }
            world.clear();
            serialize::deserialize(&mut world, "temp_scene");
            fs::remove_file("temp_scene");
        }

        if !(self.game_mode | self.playing_game) {
            cam_datas = vec![cd.clone()];
            // let cd = self.cam_data.clone();
        };
        let transform_extent = world.transforms.last_active();

        drop(_cd);
        drop(world);
        drop(_gui);
        self.file_watcher.get_updates(self.assets_manager.clone());

        if self.recompile.load(Ordering::Relaxed) {
            if let Some(compiling) = &mut self.compiler_process {
                compiling.kill();
            }
            // self.compiler_thread
            let mut args = vec!["build"];
            #[cfg(not(debug_assertions))]
            {
                println!("compiling for release");
                args.push("-r");
            }
            // args.push("-r");
            let com = Command::new("cargo")
                .args(args.as_slice())
                .env("RUSTFLAGS", "-Z threads=16")
                .spawn()
                .unwrap();
            self.compiler_process = Some(com);
            self.recompile.store(false, Ordering::Relaxed);
        }

        let _get_gui_commands = self.perf.node("_ get gui commands");
        let _window_size = if let Some(size) = &window_size {
            *size
        } else {
            self.vk.window().inner_size()
        };
        let gui_commands = unsafe {
            SendSync::new(
                self.gui
                    .draw_on_subpass_image([_window_size.width, _window_size.height]),
            )
        };
        drop(_get_gui_commands);

        let cam_datas = cvd;

        let particle_init_data = (emitter_len, emitter_inits, emitter_deinits, particle_bursts);

        let _recreate_swapchain = window_size.is_some();
        let editor_size = *EDITOR_WINDOW_DIM.lock();
        self.time.time += self.time.dt;
        self.time.dt = self.frame_time.elapsed().as_secs_f64() as f32;

        self.fps_queue.push_back(self.time.dt);
        if self.fps_queue.len() > 100 {
            self.fps_queue.pop_front();
        }

        self.frame_time = Instant::now();

        let vk = self.vk.clone();
        let full_render_time = self.perf.node("full render time");

        static mut LIGHT_DEBUG: bool = false;

        if (input.get_key(&VirtualKeyCode::LControl)
            && input.get_key(&VirtualKeyCode::LAlt)
            && input.get_key_press(&VirtualKeyCode::L))
        {
            unsafe {
                LIGHT_DEBUG = !LIGHT_DEBUG;
            }
        }
        // begin rendering
        let clean_up = self.perf.node("previous frame end clean up finished");
        // previous_frame_end.as_mut().unwrap().cleanup_finished();
        *recreate_swapchain |= self.rendering_complete.recv().unwrap();
        drop(clean_up);

        if *recreate_swapchain {
            let dimensions: [u32; 2] = vk.window().inner_size().into();
            println!("dimensions: {dimensions:?}");

            let mut swapchain = vk.swapchain();
            let (new_swapchain, new_images): (_, Vec<Arc<SwapchainImage>>) = match swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: dimensions,
                    ..swapchain.create_info()
                }) {
                Ok(r) => r,
                Err(SwapchainCreationError::ImageExtentNotSupported {
                    provided,
                    min_supported,
                    max_supported,
                }) => {
                    println!("provided: {provided:?}, min_supported: {min_supported:?}, max_supported: {max_supported:?}");
                    self.rendering_data.send(None).unwrap();
                    self.event_loop_proxy.send_event(EngineEvent::Send);
                    return should_exit;
                }
                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
            };

            vk.update_swapchain(new_swapchain);

            *framebuffers =
                window_size_dependent_setup(&new_images, render_pass.clone(), viewport, &vk);
            viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
            *recreate_swapchain = false;
        }

        let (image_num, suboptimal, acquire_future) =
            match acquire_next_image(vk.swapchain(), Some(Duration::from_secs(30))) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    *recreate_swapchain = true;
                    println!("falied to aquire next image");
                    self.rendering_data.send(None).unwrap();
                    return true;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };
        if suboptimal {
            *recreate_swapchain = true;
        }

        // let render_jobs = engine.world.lock().render();

        let mut rm = self.shared_render_data.write();
        // let clean_up = self.perf.node("previous frame end clean up finished");
        // previous_frame_end.as_mut().unwrap().cleanup_finished();
        // drop(clean_up);
        // rayon::scope(|s| {
        //     let mut a = 0;
        //     s.spawn(|s| {
        //         a = 1;
        //     });
        // });
        let transforms_buf = {
            let allocate_transform_buf = self.perf.node("allocate transforms_buf");
            self.transform_compute
                .write()
                .alloc_buffers(transform_extent as DeviceSize)
        };
        {
            let get_transform_data = self.perf.node("get transform data");

            self.world
                .lock()
                .transforms
                .get_transform_data_updates(transforms_buf.clone());
        }

        let mut builder = AutoCommandBufferBuilder::primary(
            &vk.comm_alloc,
            vk.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        self.transform_compute
            .write()
            ._update_gpu_transforms(&mut builder, transforms_buf.1.len() as usize);

        // drop(get_transform_data);

        self.transform_compute.write().update_data(
            &mut builder,
            // image_num,
            transforms_buf.clone(),
            &self.perf,
        );
        // self.rendering_complete.recv().unwrap();

        let particle_update = self.perf.node("particle update");
        self.particles_system.update(
            &mut builder,
            particle_init_data,
            &self.transform_compute.read(),
            &self.time,
        );
        drop(particle_update);

        let update_renderers = self.perf.node("update renderers");
        // compute shader renderers
        let offset_vec = {
            // puffin::profile_scope!("process renderers");
            let renderer_pipeline = rm.pipeline.clone();

            builder.bind_pipeline_compute(renderer_pipeline.clone());

            // if !lock_cull {
            //     cull_view = view.clone();
            // }

            rm.update(
                &mut renderer_data,
                vk.clone(),
                &mut builder,
                renderer_pipeline.clone(),
                &self.transform_compute.read(),
            )
        };
        drop(update_renderers);
        let render_cameras = self.perf.node("render camera(s)");

        {
            let world = self.world.lock();
            while let Some(job) = world.gpu_work.pop() {
                job.unwrap()(&mut builder, vk.clone());
            }
        }
        let light_len = self
            .world
            .lock()
            .get_components::<Light>()
            .unwrap()
            .1
            .read()
            .len();

        let (light_templates, light_deinits, light_inits) = self
            .lighting_system
            .get_light_buffer(light_len, &mut builder);

        self.lighting_compute.write().update_lights_1(
            &mut builder,
            light_deinits,
            light_inits,
            self.lighting_system.lights.lock().clone(),
            self.transform_compute.read().gpu_transforms.clone(),
            light_templates.clone(),
        );
        let mut game_image = None;
        for cam in cam_datas {
            if let (Some(cam), Some(cvd)) = cam {
                let dims = if self.game_mode {
                    framebuffers[0].0.dimensions().width_height()
                } else {
                    *EDITOR_WINDOW_DIM.lock()
                };
                cam.lock().resize(dims, vk.clone());
                self.lighting_compute.write().update_lights_2(
                    &mut builder,
                    self.lighting_system.lights.lock().clone(),
                    &cvd,
                    self.transform_compute.read().gpu_transforms.clone(),
                    light_templates.clone(),
                    light_len as i32,
                );
                let lc = self.lighting_compute.read();
                game_image = cam.lock().render(
                    vk.clone(),
                    &mut builder,
                    &self.transform_compute.read(),
                    light_len as u32,
                    self.lighting_system.lights.lock().clone(),
                    lc.visible_lights.lock().clone(),
                    lc.visible_lights_c.clone(),
                    light_templates.clone(),
                    self.particles_system.clone(),
                    transforms_buf.clone(),
                    rm.pipeline.clone(),
                    offset_vec.clone(),
                    &mut rm,
                    &mut renderer_data,
                    self.assets_manager.clone(),
                    &render_jobs,
                    cvd,
                    &self.perf,
                    lc.light_list2.lock().clone(),
                    lc.tiles.lock().clone(),
                    unsafe { LIGHT_DEBUG },
                );
            }
        }
        drop(render_cameras);
        // resize editor window
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
            *editor_window_image = Some(image.clone());

            if let Some(tex_id) = self.tex_id {
                self.gui.unregister_user_image(tex_id);
            }

            let t = self.gui.register_user_image_view(
                img_view.clone(),
                SamplerCreateInfo {
                    lod: 0.0..=LOD_CLAMP_NONE,
                    mip_lod_bias: -0.2,
                    address_mode: [SamplerAddressMode::Repeat; 3],
                    ..Default::default()
                },
            );
            self.tex_id = Some(t);

            self.image_view = Some(img_view);
        }
        // copy editor image
        if !self.game_mode {
            if let Some(image) = &editor_window_image {
                if let Some(game_image) = &game_image {
                    builder
                        .copy_image(CopyImageInfo::images(game_image.clone(), image.clone()))
                        .unwrap();
                }
            }
        }

        /////////////////////////////////////////////////////////

        let _render = self.perf.node("_ render");

        // let dimensions = vk.window().inner_size();
        // if dimensions.width == 0 || dimensions.height == 0 {
        //     self.rendering_data.send(None).unwrap();
        //     self.event_loop_proxy.send_event(EngineEvent::Send);
        //     return should_exit;
        //     // return;
        // }
        // let clean_up = self.perf.node("previous frame end clean up finished");
        // previous_frame_end.as_mut().unwrap().cleanup_finished();
        // drop(clean_up);
        // // let wait_for_render = self.perf.node("wait for render");
        // // let out_of_date = self.rendering_complete.recv().unwrap();
        // // drop(wait_for_render);
        // // *recreate_swapchain |= _recreate_swapchain | out_of_date;
        // if *recreate_swapchain {
        //     let dimensions: [u32; 2] = vk.window().inner_size().into();
        //     println!("dimensions: {dimensions:?}");

        //     let mut swapchain = vk.swapchain();
        //     let (new_swapchain, new_images): (_, Vec<Arc<SwapchainImage>>) = match swapchain
        //         .recreate(SwapchainCreateInfo {
        //             image_extent: dimensions,
        //             ..swapchain.create_info()
        //         }) {
        //         Ok(r) => r,
        //         Err(SwapchainCreationError::ImageExtentNotSupported {
        //             provided,
        //             min_supported,
        //             max_supported,
        //         }) => {
        //             println!("provided: {provided:?}, min_supported: {min_supported:?}, max_supported: {max_supported:?}");
        //             // self.rendering_data.send(None).unwrap();
        //             self.event_loop_proxy.send_event(EngineEvent::Send);
        //             return should_exit;
        //         }
        //         Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
        //     };

        //     vk.update_swapchain(new_swapchain);

        //     *framebuffers =
        //         window_size_dependent_setup(&new_images, render_pass.clone(), viewport, &vk);
        //     viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
        //     *recreate_swapchain = false;
        // }

        // let (image_num, suboptimal, acquire_future) =
        //     match acquire_next_image(vk.swapchain(), Some(Duration::from_secs(30))) {
        //         Ok(r) => r,
        //         Err(AcquireError::OutOfDate) => {
        //             *recreate_swapchain = true;
        //             println!("falied to aquire next image");
        //             // self.rendering_data.send(None).unwrap();
        //             return true;
        //         }
        //         Err(e) => panic!("Failed to acquire next image: {:?}", e),
        //     };
        // if suboptimal {
        //     *recreate_swapchain = true;
        // }

        // engine.perf.update("_ begin render pass".into(), Instant::now() - _inst);
        let gui_commands = gui_commands.unwrap();

        builder
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0., 0., 0., 1.].into())],
                    ..RenderPassBeginInfo::framebuffer(framebuffers[image_num as usize].1.clone())
                },
                SubpassContents::SecondaryCommandBuffers,
            )
            .unwrap()
            .set_viewport(0, [viewport.clone()]);
        if !self.game_mode {
            builder.execute_commands(gui_commands).unwrap();
        }
        builder.end_render_pass().unwrap();

        if self.game_mode {
            if let Some(game_image) = &game_image {
                let image = framebuffers[image_num as usize].0.clone();
                builder
                    .blit_image(BlitImageInfo::images(game_image.clone(), image))
                    .unwrap();
            }
        }

        let _build_command_buffer = self.perf.node("_ build command buffer");
        let command_buffer = builder.build().unwrap();
        drop(_build_command_buffer);
        vk.finalize();
        // unsafe {
        //     *self.particles_system.cycle.get() = (*self.particles_system.cycle.get() + 1) % 2;
        // }
        let _execute = self.perf.node("_ execute");

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
        // previous_frame_end.take().unwrap().flush().unwrap();
        // // let future = previous_frame_end
        // //     .take()
        // //     .unwrap()
        // //     .join(acquire_future)
        // let future = acquire_future
        //     .then_execute(vk.queue.clone(), command_buffer)
        //     .unwrap()
        //     .then_swapchain_present(
        //         vk.queue.clone(),
        //         SwapchainPresentInfo::swapchain_image_index(vk.swapchain().clone(), image_num),
        //     )
        //     .then_signal_fence_and_flush();

        // match future {
        //     Ok(future) => {
        //         *previous_frame_end = Some(future.boxed());
        //     }
        //     Err(FlushError::OutOfDate) => {
        //         *recreate_swapchain = true;
        //         *previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
        //     }
        //     Err(e) => {
        //         println!("failed to flush future: {e}");
        //         *recreate_swapchain = true;
        //         *previous_frame_end = Some(sync::now(vk.device.clone()).boxed());
        //     }
        // }

        self.rendering_data.send(Some((
            should_exit,
            image_num,
            acquire_future,
            command_buffer,
            // previous_frame_end,
        )));
        drop(_execute);

        self.update_editor_window = window_size.is_some();
        self.playing_game = _playing_game;
        if !should_exit {
            self.event_loop_proxy.send_event(EngineEvent::Send);
        }
        should_exit
    }
    pub fn end(mut self) {
        println!("end");
        // self.world.lock().sys.physics.lock().get_counters();
        self.perf.print();
        // Arc::into_inner(self.rendering_thread).unwrap().join();
        self.event_loop_proxy.send_event(EngineEvent::Quit);
        if let Some(compiling) = &mut self.compiler_process {
            compiling.kill();
        }
        // Arc::into_inner(self.input_thread).unwrap().join();
    }
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
    vk: &VulkanManager,
) -> Vec<(Arc<SwapchainImage>, Arc<Framebuffer>)> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
    // let _image = AttachmentImage::with_usage(
    //     &vk.mem_alloc,
    //     dimensions,
    //     vk.swapchain().image_format(),
    //     // ImageUsage::SAMPLED
    //     ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST, // | ImageUsage::TRANSIENT_ATTACHMENT
    // | ImageUsage::STORAGE, // | ImageUsage::INPUT_ATTACHMENT,
    // )
    // .unwrap();
    images
        .iter()
        .map(|image| {
            // let view = ImageView::new_default(image.clone()).unwrap();
            let view = ImageView::new_default(image.clone()).unwrap();
            // let color_view = ImageView::new_default(color.arc.clone()).unwrap();
            (
                image.clone(),
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![view],
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
        })
        .collect::<Vec<_>>()
}
