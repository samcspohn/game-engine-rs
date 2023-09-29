use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    env,
    mem::size_of,
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
use glm::{vec3, Vec3};
use lazy_static::lazy_static;
use num_integer::Roots;
use once_cell::sync::Lazy;
use puffin_egui::puffin;
use serde::{Deserialize, Serialize};

use rayon::prelude::*;

use nalgebra_glm as glm;
use parking_lot::{Mutex, RwLock};
use vulkano::{
    buffer::Subbuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyImageInfo, PrimaryAutoCommandBuffer, PrimaryCommandBufferAbstract, RenderPassBeginInfo,
        SubpassContents,
    },
    format::Format,
    image::{
        view::ImageView, ImageAccess, ImageDimensions, ImageUsage, ImmutableImage, MipmapsCount,
        SwapchainImage,
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
    dpi::PhysicalSize,
    event::{Event, ModifiersState, WindowEvent},
    event_loop::{EventLoop, EventLoopBuilder, EventLoopProxy},
};
// use crate::{physics::Physics};

use crate::{
    editor::{self, editor_cam::EditorCam, editor_ui::EDITOR_WINDOW_DIM},
    engine::{
        particles::{component::ParticleEmitter, shaders::scs::l},
        // physics::collider::_Collider,
        project::asset_manager::AssetManagerBase,
        rendering::model::ModelManager,
        // utils::look_at,
    },
};

use self::{
    input::Input,
    particles::particles::ParticleCompute,
    perf::Perf,
    project::{
        asset_manager::AssetsManager,
        file_watcher::{self, FileWatcher},
        save_project, Project,
    },
    render_thread::RenderingData,
    rendering::{
        camera::{Camera, CameraData},
        component::{Renderer, SharedRendererData},
        model::{Mesh, ModelRenderer},
        pipeline::RenderPipeline,
        texture::{Texture, TextureManager},
        vulkan_manager::VulkanManager,
    },
    time::Time,
    transform_compute::{
        cs::{transform, MVP},
        TransformCompute,
    },
    world::World,
};

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
pub(crate) mod prelude;
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

pub(crate) enum EngineEvent {
    Send,
    Quit,
}
struct EngineRenderer {
    viewport: Viewport,
    framebuffers: Vec<Arc<Framebuffer>>,
    recreate_swapchain: bool,
    editor_window_image: Option<Arc<dyn ImageAccess>>,
    render_pass: Arc<RenderPass>,
}
pub(crate) struct Engine {
    pub(crate) world: Arc<Mutex<World>>,
    pub(crate) assets_manager: Arc<AssetsManager>,
    pub(crate) project: Project,
    pub(crate) transform_compute: RwLock<TransformCompute>,
    pub(crate) particles_system: Arc<ParticleCompute>,
    pub(crate) playing_game: bool,
    // pub(crate) coms: (Receiver<RenderingData>, Sender<(Input, bool)>),
    pub(crate) input: Receiver<(
        Vec<WindowEvent<'static>>,
        Input,
        Option<PhysicalSize<u32>>,
        bool,
    )>,
    pub(crate) rendering_data:
        Sender<Option<(bool, u32, SwapchainAcquireFuture, PrimaryAutoCommandBuffer)>>,
    pub(crate) rendering_complete: Receiver<bool>,
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
    pub(crate) image_view: Option<Arc<ImageView<ImmutableImage>>>,
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
    pub(crate) fn new(engine_dir: &PathBuf, project_dir: &str) -> Self {
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

        //////////////////////////////////////////////////

        let mut input = Input {
            ..Default::default()
        };

        let mut perf = Perf::new();

        let mut frame_time = Instant::now();

        let mut transform_compute = TransformCompute::new(vk.clone());

        let mut fps_queue: VecDeque<f32> = std::collections::VecDeque::new();

        let running = Arc::new(AtomicBool::new(true));

        // let coms = (rrx, tx);

        {
            let mut world = world.lock();
            world.register::<Renderer>(false, false, false);
            world.register::<ParticleEmitter>(false, false, false);
            world.register::<Camera>(false, false, false);
            // world.register::<_Collider>(false, true, false);
            // world.register::<terrain_eng::TerrainEng>(true, false, true);
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
        // let (rendering_snd, rendering_rcv) = crossbeam::channel::bounded(1);
        let input_thread = Arc::new({
            let vk = vk.clone();
            thread::spawn(move || input_thread::input_thread(event_loop, vk, input_snd))
        });
        let rendering_thread = Arc::new({
            let vk = vk.clone();
            thread::spawn(move || render_thread::render_thread(vk, rendering_rcv, rendering_snd2))
        });
        proxy.send_event(EngineEvent::Send);

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
        println!(
            "default quat: {}",
            glm::quat_look_at_lh(&Vec3::z(), &Vec3::y()).coords
        );
        Self {
            world,
            assets_manager,
            project: Project::default(),
            transform_compute: RwLock::new(transform_compute),
            playing_game: false,
            // coms,
            perf: Arc::new(perf),
            shared_render_data: rm,
            input: input_rcv,
            rendering_data: rendering_snd,
            rendering_complete: rendering_rcv2,
            cam_data: Arc::new(Mutex::new(CameraData::new(vk.clone(), 1))),
            renderer: EngineRenderer {
                viewport,
                framebuffers,
                recreate_swapchain,
                editor_window_image,
                render_pass,
            },
            vk,
            editor_cam: editor::editor_cam::EditorCam {
                rot: glm::quat_look_at_lh(&Vec3::z(), &Vec3::y()),
                pos: Vec3::zeros(),
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
            image_view: None,
            gui: unsafe { SendSync::new(gui) },
            update_editor_window: true,
            event_loop_proxy: proxy,
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
        let full_frame_time = self.perf.node("full frame time");
        let (events, input, window_size, should_exit) = self.input.recv().unwrap();
        for event in events {
            self.gui.update(&event);
        }
        let mut world = self.world.lock();
        let gpu_work = SegQueue::new();
        let world_sim = self.perf.node("world _update");
        let cvd = if self.playing_game {
            puffin::profile_scope!("game loop");
            {
                puffin::profile_scope!("world update");
                if world.phys_time >= world.phys_step {
                    // let mut physics = world.sys.physics.lock();
                    let len = world.sys.physics.lock().rigid_body_set.len();
                    let num_threads = (len / (num_cpus::get().sqrt())).max(1).min(num_cpus::get());
                    world.sys.physics.lock().step(&self.perf);
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
            world.update_cameras()
        } else {
            world._destroy(&self.perf);
            world.editor_update(&input, &self.time, &gpu_work); // TODO: terrain update still breaking
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
        drop(world_sim);

        let get_renderer_data = self.perf.node("get renderer data");
        let mut renderer_data = world.sys.renderer_manager.write().get_renderer_data();
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
            cd
        };

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
            );
        });
        let transform_extent = world.transforms.last_active();

        if self.playing_game {
            // cam_datas[0].clone()
        } else {
            cam_datas = vec![cd.clone()];
            // let cd = self.cam_data.clone();
        };

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

        let cam_datas = cvd;

        let particle_init_data = (emitter_len, emitter_inits, emitter_deinits, particle_bursts);

        let _recreate_swapchain = window_size.is_some() | self.playing_game != _playing_game;
        let editor_size = *EDITOR_WINDOW_DIM.lock();
        self.time.time += self.time.dt;
        self.time.dt = self.frame_time.elapsed().as_secs_f64() as f32;

        self.fps_queue.push_back(self.time.dt);
        if self.fps_queue.len() > 100 {
            self.fps_queue.pop_front();
        }

        self.frame_time = Instant::now();

        let EngineRenderer {
            viewport,
            framebuffers,
            recreate_swapchain,
            editor_window_image,
            render_pass,
        } = &mut self.renderer;

        let vk = self.vk.clone();
        let full_render_time = self.perf.node("full render time");
        // let render_jobs = engine.world.lock().render();

        let mut rm = self.shared_render_data.write();
        // previous_frame_end.as_mut().unwrap().cleanup_finished();
        // rayon::scope(|s| {
        //     let mut a = 0;
        //     s.spawn(|s| {
        //         a = 1;
        //     });
        // });
        let transforms_buf = {
            let allocate_transform_buf = self.perf.node("allocate transforms_buf");
            // let world = self.world.lock();
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
        // rendering_complete.send(()).unwrap();

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

        while let Some(job) = gpu_work.pop() {
            job.unwrap()(&mut builder, vk.clone());
        }

        let mut game_image = None;
        for cam in cam_datas {
            if let (Some(cam), Some(cvd)) = cam {
                cam.lock().resize(*EDITOR_WINDOW_DIM.lock(), vk.clone());
                game_image = cam.lock().render(
                    vk.clone(),
                    &mut builder,
                    &self.transform_compute.read(),
                    self.particles_system.clone(),
                    transforms_buf.clone(),
                    rm.pipeline.clone(),
                    offset_vec.clone(),
                    &mut rm,
                    &mut renderer_data,
                    // image_num,
                    self.assets_manager.clone(),
                    &render_jobs,
                    cvd,
                    &self.perf,
                );
            }
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

        if let Some(image) = &editor_window_image {
            if let Some(game_image) = game_image {
                builder
                    .copy_image(CopyImageInfo::images(game_image, image.clone()))
                    .unwrap();
            }
        }

        /////////////////////////////////////////////////////////

        let _render = self.perf.node("_ render");

        let dimensions = vk.window().inner_size();
        if dimensions.width == 0 || dimensions.height == 0 {
            // return;
        }
        // previous_frame_end.as_mut().unwrap().cleanup_finished();
        let wait_for_render = self.perf.node("wait for render");
        let out_of_date = self.rendering_complete.recv().unwrap();
        drop(wait_for_render);
        *recreate_swapchain |= _recreate_swapchain | out_of_date;
        if *recreate_swapchain {
            let dimensions: [u32; 2] = vk.window().inner_size().into();

            let mut swapchain = vk.swapchain();
            let (new_swapchain, new_images): (_, Vec<Arc<SwapchainImage>>) = match swapchain
                .recreate(SwapchainCreateInfo {
                    image_extent: dimensions,
                    ..swapchain.create_info()
                }) {
                Ok(r) => r,
                Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => {
                    self.rendering_data.send(None).unwrap();
                    return should_exit;
                }
                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
            };

            vk.update_swapchain(new_swapchain);

            *framebuffers = window_size_dependent_setup(&new_images, render_pass.clone(), viewport);
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
                    return should_exit;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };
        if suboptimal {
            *recreate_swapchain = true;
        }

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

        let _build_command_buffer = self.perf.node("_ build command buffer");
        let command_buffer = builder.build().unwrap();
        drop(_build_command_buffer);
        unsafe {
            *self.particles_system.cycle.get() = (*self.particles_system.cycle.get() + 1) % 3;
        }
        let _execute = self.perf.node("_ execute");
        self.rendering_data.send(Some((
            should_exit,
            image_num,
            acquire_future,
            command_buffer,
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
        self.perf.print();
        Arc::into_inner(self.rendering_thread).unwrap().join();
        self.event_loop_proxy.send_event(EngineEvent::Quit);
        // Arc::into_inner(self.input_thread).unwrap().join();
    }
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
