use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::env;
use std::mem::size_of;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::thread;
use std::thread::JoinHandle;
use std::thread::Thread;
use std::time::Duration;
use std::time::Instant;

use crossbeam::channel::Receiver;
use crossbeam::channel::Sender;
use crossbeam::queue::SegQueue;
use egui::TextureId;
use egui_winit_vulkano::Gui;
use egui_winit_vulkano::GuiConfig;
use force_send_sync::SendSync;
use glm::Vec3;
use lazy_static::lazy_static;
use num_integer::Roots;
use once_cell::sync::Lazy;
use puffin_egui::puffin;
use serde::{Deserialize, Serialize};

use rayon::prelude::*;

use parking_lot::Mutex;
use parking_lot::RwLock;
use vulkano::buffer::Subbuffer;
use vulkano::command_buffer::CommandBufferUsage;
use vulkano::command_buffer::RenderPassBeginInfo;
use vulkano::command_buffer::SubpassContents;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::ImageAccess;
use vulkano::image::ImageDimensions;
use vulkano::image::ImageUsage;
use vulkano::image::ImmutableImage;
use vulkano::image::MipmapsCount;
use vulkano::image::SwapchainImage;
use vulkano::memory::allocator::MemoryUsage;
use vulkano::render_pass::Framebuffer;
use vulkano::render_pass::FramebufferCreateInfo;
use vulkano::render_pass::RenderPass;
use vulkano::render_pass::Subpass;
use vulkano::sampler::SamplerAddressMode;
use vulkano::sampler::SamplerCreateInfo;
use vulkano::sampler::LOD_CLAMP_NONE;
use vulkano::sync;
use vulkano::sync::GpuFuture;
// use spin::Mutex;
use nalgebra_glm as glm;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        PrimaryAutoCommandBuffer,
    },
    pipeline::graphics::viewport::Viewport,
};
use winit::dpi::PhysicalSize;
use winit::event::Event;
use winit::event::ModifiersState;
use winit::event::WindowEvent;
use winit::event_loop::EventLoop;
// use crate::{physics::Physics};

use crate::editor;
use crate::editor::editor_cam::EditorCam;
use crate::editor::editor_ui::EDITOR_WINDOW_DIM;
use crate::engine::particles::component::ParticleEmitter;
use crate::engine::particles::shaders::scs::l;
use crate::engine::project::asset_manager::AssetManagerBase;
use crate::engine::rendering::model::ModelManager;

use self::input::Input;
use self::particles::particles::ParticleCompute;
use self::perf::Perf;
use self::project::asset_manager::AssetsManager;
use self::project::file_watcher;
use self::project::file_watcher::FileWatcher;
use self::project::save_project;
use self::project::Project;
use self::render_thread::RenderingData;
use self::rendering::camera::Camera;
use self::rendering::camera::CameraData;
use self::rendering::component::Renderer;
use self::rendering::component::SharedRendererData;
use self::rendering::model::Mesh;
use self::rendering::model::ModelRenderer;
use self::rendering::pipeline::RenderPipeline;
use self::rendering::texture::Texture;
use self::rendering::texture::TextureManager;
use self::rendering::vulkan_manager::VulkanManager;
use self::time::Time;
use self::transform_compute::cs::transform;
use self::transform_compute::cs::MVP;
use self::transform_compute::TransformCompute;
use self::world::World;

pub mod atomic_vec;
pub mod input;
pub mod linked_list;
pub mod perf;
pub mod project;
pub mod rendering;
pub mod runtime_compilation;
pub mod storage;
pub mod time;
pub mod transform_compute;
pub mod world;
// mod render_pipeline;
pub mod color_gradient;
mod input_thread;
pub mod main_loop;
pub mod particles;
pub mod physics;
mod prelude;
mod render_thread;
pub mod utils;
#[repr(C)]
pub struct RenderJobData<'a> {
    pub builder: &'a mut AutoCommandBufferBuilder<
        PrimaryAutoCommandBuffer,
        Arc<StandardCommandBufferAllocator>,
    >,
    pub gpu_transforms: Subbuffer<[transform]>,
    pub mvp: Subbuffer<[MVP]>,
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

pub(crate) struct Engine {
    pub(crate) world: Arc<Mutex<World>>,
    pub(crate) assets_manager: Arc<AssetsManager>,
    pub(crate) project: Project,
    pub(crate) transform_compute: RwLock<TransformCompute>,
    pub(crate) particles_system: Arc<ParticleCompute>,
    pub(crate) playing_game: bool,
    pub(crate) coms: (Receiver<RenderingData>, Sender<(Input, bool)>),
    pub(crate) input: Receiver<(
        Vec<WindowEvent<'static>>,
        Input,
        Option<PhysicalSize<u32>>,
        bool,
    )>,
    pub(crate) rendering_data: Sender<(bool, RenderingData)>,
    pub(crate) cam_data: Arc<Mutex<CameraData>>,
    pub(crate) editor_cam: EditorCam,
    pub(crate) time: Time,
    pub(crate) perf: Arc<Perf>,
    pub(crate) vk: Arc<VulkanManager>,
    pub(crate) shared_render_data: Arc<RwLock<SharedRendererData>>,
    pub(crate) fps_queue: VecDeque<f32>,
    pub(crate) frame_time: Instant,
    pub(crate) running: Arc<AtomicBool>,
    pub(crate) input_thread: Arc<JoinHandle<()>>,
    pub(crate) rendering_thread: Arc<JoinHandle<()>>,
    pub(crate) file_watcher: FileWatcher,
    pub(crate) _image_num: u32,
    pub(crate) gui: SendSync<Gui>,
    pub(crate) tex_id: Option<TextureId>,
    update_editor_window: bool,
}
pub struct EnginePtr {
    ptr: *const Engine,
}
unsafe impl Send for EnginePtr {}
unsafe impl Sync for EnginePtr {}

lazy_static! {
    pub static ref IMAGE_VIEW: Lazy<Arc<Mutex<Option<(Arc<ImageView<ImmutableImage>>, Arc<dyn ImageAccess>)>>>> =
        Lazy::new(|| { Arc::new(Mutex::new(None)) });
}
impl Engine {
    pub(crate) fn new(engine_dir: &PathBuf, project_dir: &str) -> Self {
        let event_loop = unsafe { SendSync::new(EventLoop::new()) };
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
        let rs_manager = Arc::new(Mutex::new(runtime_compilation::RSManager::new((), &["rs"])));

        assert!(env::set_current_dir(&Path::new(engine_dir)).is_ok()); // procedurally generate cube/move cube to built in assets
        model_manager.lock().from_file("eng_res/cube/cube.obj");

        texture_manager.lock().from_file("eng_res/particle.png");
        assert!(env::set_current_dir(&Path::new(project_dir)).is_ok());

        let particles_system = Arc::new(ParticleCompute::new(vk.clone(), texture_manager.clone()));

        let world = Arc::new(Mutex::new(World::new(
            particles_system.clone(),
            vk.clone(),
            assets_manager.clone(),
        )));

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

        // let mut framebuffers =
        //     window_size_dependent_setup(&vk.images, render_pass.clone(), &mut viewport);
        // let mut recreate_swapchain = false;

        // let mut previous_frame_end = Some(sync::now(vk.device.clone()).boxed());

        // let mut modifiers = ModifiersState::default();

        //////////////////////////////////////////////////

        let mut input = Input {
            ..Default::default()
        };

        let mut perf = Perf::new();

        let mut frame_time = Instant::now();

        let mut transform_compute = TransformCompute::new(vk.clone());

        let mut fps_queue: VecDeque<f32> = std::collections::VecDeque::new();

        // let mut focused = true;

        // puffin::set_scopes_on(true);

        // let _cull_view = glm::Mat4::identity();
        // let mut lock_cull = false;
        // let mut first_frame = true;

        /////////////////////////////////////////////////////////////////////////////////////////
        let (tx, rx): (Sender<_>, Receiver<_>) = crossbeam::channel::bounded(1);
        let (rtx, rrx): (Sender<_>, Receiver<_>) = crossbeam::channel::bounded(1);
        let running = Arc::new(AtomicBool::new(true));

        let coms = (rrx, tx);

        {
            let mut world = world.lock();
            world.register::<Renderer>(false, false, false);
            world.register::<ParticleEmitter>(false, false, false);
            world.register::<Camera>(false, false, false);
            // world.register::<terrain_eng::TerrainEng>(true, false, true);
        };

        let rm = {
            let w = world.lock();
            let rm = w.sys.renderer_manager.read();
            rm.shr_data.clone()
        };

        // let game_thread = Arc::new({
        //     let _running = running.clone();
        //     let world = world.clone();
        //     thread::spawn(move || main_loop(world.clone(), (rtx, rx), _running))
        // });

        let _res = coms.1.send((input.clone(), false));
        let mut file_watcher = file_watcher::FileWatcher::new(".");

        let mut cam_data = CameraData::new(vk.clone());
        let (input_snd, input_rcv) = crossbeam::channel::bounded(1);
        // let final_render_pass = vulkano::single_pass_renderpass!(
        //     vk.device.clone(),
        //     attachments: {
        //         final_color: {
        //             load: Clear,
        //             store: Store,
        //             format: vk.swapchain().image_format(),
        //             samples: 1,
        //         }
        //     },
        //     pass: { color: [final_color], depth_stencil: {}} // Create a second renderpass to draw egui
        // )
        // .unwrap();
        let mut gui = egui_winit_vulkano::Gui::new_with_subpass(
            &event_loop,
            vk.surface.clone(),
            vk.queue.clone(),
            Subpass::from(render_pass.clone(), 0).unwrap(),
            GuiConfig {
                preferred_format: Some(vk.swapchain().image_format()),
                is_overlay: true,
                ..Default::default()
            },
        );
        let (rendering_snd, rendering_rcv) = crossbeam::channel::bounded(1);
        // let (rendering_snd, rendering_rcv) = crossbeam::channel::bounded(1);
        let input_thread = Arc::new(thread::spawn(move || {
            input_thread::input_thread(event_loop, input_snd)
        }));
        let rendering_thread = Arc::new({
            let vk = vk.clone();
            thread::spawn(move || render_thread::render_thread(vk, render_pass, rendering_rcv))
        });

        Self {
            world,
            assets_manager,
            project: Project::default(),
            transform_compute: RwLock::new(transform_compute),
            playing_game: false,
            coms,
            perf: Arc::new(perf),
            shared_render_data: rm,
            input: input_rcv,
            rendering_data: rendering_snd,
            cam_data: Arc::new(Mutex::new(CameraData::new(vk.clone()))),
            vk,
            editor_cam: editor::editor_cam::EditorCam {
                rot: glm::quat(-1., 0., 0., 0.),
                pos: Vec3::default(),
                speed: 30f32,
            },
            time: Time::default(),
            particles_system,
            fps_queue,
            frame_time,
            running,
            input_thread,
            rendering_thread,
            file_watcher,
            _image_num: 0,
            tex_id: None,
            gui: unsafe { SendSync::new(gui) },
            update_editor_window: true,
        }
    }
    pub(crate) fn init(&mut self) {
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
    }
    pub(crate) fn update_sim(&mut self) -> bool {
        let (events, input, window_size, should_exit) = self.input.recv().unwrap();
        for event in events {
            self.gui.update(&event);
        }
        let mut world = self.world.lock();
        let gpu_work = SegQueue::new();
        let world_sim = self.perf.node("world _update");
        if self.playing_game {
            puffin::profile_scope!("game loop");
            {
                puffin::profile_scope!("world update");
                if world.phys_time >= world.phys_step {
                    // let mut physics = world.sys.physics.lock();
                    let len = world.sys.physics.lock().rigid_body_set.len();
                    let num_threads = (len / (num_cpus::get().sqrt())).max(1).min(num_cpus::get());
                    world.sys.physics.lock().step(&world.gravity, &self.perf);

                    // let physics = Arc::new(&mut physics);
                    // rayon::ThreadPoolBuilder::new()
                    //     .num_threads(num_threads)
                    //     .build_scoped(
                    //         |thread| thread.run(),
                    //         |pool| {
                    //             pool.install(|| {
                    //                 world.sys.physics.lock().step(&world.gravity, &self.perf);
                    //             })
                    //         },
                    //     )
                    //     .unwrap();
                    // drop(physics);
                    world.phys_time -= world.phys_step;
                }
                world.phys_time += self.time.dt;
                let world_update = self.perf.node("world _update");
                world._update(&input, &self.time, &gpu_work, &self.perf);
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
                }
            }
        } else {
            world._destroy(&self.perf);
            world.editor_update(&input, &self.time, &gpu_work); // TODO: terrain update still breaking
            self.editor_cam.update(&input, &self.time);
            self.cam_data.lock().update(
                self.editor_cam.pos,
                self.editor_cam.rot,
                0.01f32,
                10_000f32,
                70f32,
            );
        }
        drop(world_sim);

        let get_transform_data = self.perf.node("get transform data");
        let transform_data = world.transforms.get_transform_data_updates();
        drop(get_transform_data);

        let get_renderer_data = self.perf.node("get renderer data");
        let renderer_data = world.sys.renderer_manager.write().get_renderer_data();
        drop(get_renderer_data);
        let emitter_len = world.get_emitter_len();
        let emitter_inits = world.sys.particles_system.emitter_inits.get_vec();
        let emitter_deinits = world.sys.particles_system.emitter_deinits.get_vec();
        let particle_bursts = world.sys.particles_system.particle_burts.get_vec();
        let (main_cam_id, mut cam_datas) = world.get_cam_datas();
        let render_jobs = world.render();

        self._image_num = (self._image_num + 1) % self.vk.swapchain().image_count();

        let cd = if self.playing_game {
            cam_datas[0].clone()
        } else {
            let cd = self.cam_data.clone();
            cam_datas = vec![cd.clone()];
            cd
        };

        let mut _cd = cd.lock();
        let _gui = self.perf.node("_ gui");
        let dimensions = *EDITOR_WINDOW_DIM.lock();
        let mut _playing_game = false;
        let tex_id = if let Some(tex_id) = self.tex_id {
            tex_id
        } else {
            let a = IMAGE_VIEW.lock();
            if let Some(img_view) = a.as_ref() {
                let t = self.gui.register_user_image_view(
                    img_view.0.clone(),
                    SamplerCreateInfo {
                        lod: 0.0..=LOD_CLAMP_NONE,
                        mip_lod_bias: -0.2,
                        address_mode: [SamplerAddressMode::Repeat; 3],
                        ..Default::default()
                    },
                );
                self.tex_id = Some(t);
                t
            } else {
                TextureId::default()
            }
        };

        self.gui.immediate_ui(|gui| {
            let ctx = gui.context();
            _playing_game = editor::editor_ui::editor_ui(
                &mut world,
                &mut self.fps_queue,
                &ctx,
                tex_id,
                self.assets_manager.clone(),
            );
        });
        {
            let ear = EDITOR_WINDOW_DIM.lock();
            if dimensions != *ear {
                _cd.resize(*ear, self.vk.clone());

                let tex_id = if let Some(tex_id) = self.tex_id {
                    tex_id
                } else {
                    let a = IMAGE_VIEW.lock();
                    if let Some(img_view) = a.as_ref() {
                        let t = self.gui.register_user_image_view(
                            img_view.0.clone(),
                            SamplerCreateInfo {
                                lod: 0.0..=LOD_CLAMP_NONE,
                                mip_lod_bias: -0.2,
                                address_mode: [SamplerAddressMode::Repeat; 3],
                                ..Default::default()
                            },
                        );
                        self.tex_id = Some(t);
                        t
                    } else {
                        TextureId::default()
                    }
                };

                // fc_map.clear();
            }
        }
        drop(_cd);
        drop(world);
        drop(_gui);
        self.file_watcher.get_updates(self.assets_manager.clone());


        let _get_gui_commands = self.perf.node("_ get gui commands");
        let size = if let Some(size) = &window_size {
            *size
        } else {
            self.vk.window().inner_size()
        };
        let gui_commands =
            unsafe { SendSync::new(self.gui.draw_on_subpass_image([size.width, size.height])) };
        drop(_get_gui_commands);

        let data = RenderingData {
            transform_data,
            cam_datas,
            main_cam_id,
            renderer_data,
            emitter_inits: (emitter_len, emitter_inits, emitter_deinits, particle_bursts),
            gpu_work,
            render_jobs,
            _image_num: self._image_num,
            gui_commands,
            recreate_swapchain: window_size.is_some(),
            editor_size: *EDITOR_WINDOW_DIM.lock(),
            engine_ptr: EnginePtr {
                ptr: std::ptr::from_ref(&self),
            },
        };

        self.time.time += self.time.dt;
        self.time.dt = self.frame_time.elapsed().as_secs_f64() as f32;

        self.fps_queue.push_back(self.time.dt);
        if self.fps_queue.len() > 100 {
            self.fps_queue.pop_front();
        }

        self.frame_time = Instant::now();

        let res = self.rendering_data.send((false, data));
        if res.is_err() {
            println!("ohno");
            panic!();
        }
        if self.update_editor_window {
            let a = IMAGE_VIEW.lock();
            if let Some(img_view) = a.as_ref() {
                if let Some(tex_id) = self.tex_id {
                    self.gui.unregister_user_image(tex_id);
                }

                let t = self.gui.register_user_image_view(
                    img_view.0.clone(),
                    SamplerCreateInfo {
                        lod: 0.0..=LOD_CLAMP_NONE,
                        mip_lod_bias: -0.2,
                        address_mode: [SamplerAddressMode::Repeat; 3],
                        ..Default::default()
                    },
                );
                self.tex_id = Some(t);
            }
        }
        self.update_editor_window = window_size.is_some();

        self.playing_game = _playing_game;
        should_exit
    }
    pub fn end(mut self) {
        Arc::into_inner(self.rendering_thread).unwrap().join();
        self.perf.print();
    }
    // fn get_rendering_data(&self) -> RenderingData {
    //     let get_transform_data = perf.node("get transform data");
    //     let transform_data = world.transforms.get_transform_data_updates();
    //     drop(get_transform_data);

    //     let get_renderer_data = perf.node("get renderer data");
    //     let renderer_data = world.sys.renderer_manager.write().get_renderer_data();
    //     drop(get_renderer_data);
    //     let emitter_len = world.get_emitter_len();
    //     let emitter_inits = world.sys.particles_system.emitter_inits.get_vec();
    //     let emitter_deinits = world.sys.particles_system.emitter_deinits.get_vec();
    //     let particle_bursts = world.sys.particles_system.particle_burts.get_vec();
    //     let (main_cam_id, cam_datas) = world.get_cam_datas();
    //     drop(world);
    //     let data = RenderingData {
    //         transform_data,
    //         cam_datas,
    //         main_cam_id,
    //         renderer_data,
    //         emitter_inits: (emitter_len, emitter_inits, emitter_deinits, particle_bursts),
    //         gpu_work,
    //     };
    //     // let res = coms.0.send(data);
    //     // if res.is_err() {
    //     //     println!("ohno");
    //     // }
    //     data
    // }
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
