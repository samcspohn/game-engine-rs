use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::env;
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
use force_send_sync::SendSync;
use serde::{Deserialize, Serialize};
use sync_unsafe_cell::SyncUnsafeCell;

use rayon::prelude::*;

use parking_lot::Mutex;
use parking_lot::RwLock;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::CommandBufferUsage;
use vulkano::command_buffer::RenderPassBeginInfo;
use vulkano::command_buffer::SubpassContents;
use vulkano::image::view::ImageView;
use vulkano::image::ImageAccess;
use vulkano::image::SwapchainImage;
use vulkano::render_pass::Framebuffer;
use vulkano::render_pass::FramebufferCreateInfo;
use vulkano::render_pass::RenderPass;
use vulkano::render_pass::Subpass;
use vulkano::sync;
use vulkano::sync::GpuFuture;
// use spin::Mutex;
use nalgebra_glm as glm;
use vulkano::{
    buffer::DeviceLocalBuffer,
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        PrimaryAutoCommandBuffer,
    },
    pipeline::graphics::viewport::Viewport,
};
use winit::event::ModifiersState;
use winit::event_loop::EventLoop;
// use crate::{physics::Physics};

use crate::engine::particles::component::ParticleEmitter;
use crate::engine::project::asset_manager::AssetManagerBase;
use crate::engine::rendering::model::ModelManager;

use self::input::Input;
use self::main_loop::main_loop;
use self::main_loop::RenderingData;
use self::particles::particles::ParticleCompute;
// use self::particles::particles::ParticleEmitter;
use self::perf::Perf;
use self::project::asset_manager::AssetsManager;
use self::project::file_watcher;
use self::project::file_watcher::FileWatcher;
use self::project::save_project;
use self::project::Project;
use self::rendering::camera::Camera;
use self::rendering::camera::CameraData;
use self::rendering::model::Mesh;
use self::rendering::model::ModelRenderer;
use self::rendering::pipeline::RenderPipeline;
use self::rendering::renderer_component::Renderer;
use self::rendering::renderer_component::SharedRendererData;
use self::rendering::texture::Texture;
use self::rendering::texture::TextureManager;
use self::rendering::vulkan_manager::VulkanManager;
use self::transform_compute::cs::ty::transform;
use self::transform_compute::cs::ty::MVP;
use self::transform_compute::TransformCompute;
use self::world::World;

pub mod input;
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
pub mod main_loop;
pub mod particles;
pub mod physics;
pub mod utils;
mod prelude;

#[repr(C)]
pub struct RenderJobData<'a> {
    pub builder: &'a mut AutoCommandBufferBuilder<
        PrimaryAutoCommandBuffer,
        Arc<StandardCommandBufferAllocator>,
    >,
    pub gpu_transforms: Arc<DeviceLocalBuffer<[transform]>>,
    pub mvp: Arc<DeviceLocalBuffer<[MVP]>>,
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
    pub(crate) transform_compute: TransformCompute,
    pub(crate) particles_system: Arc<ParticleCompute>,
    pub(crate) playing_game: bool,
    pub(crate) coms: (Receiver<RenderingData>, Sender<(Input, bool)>),
    pub(crate) perf: Perf,
    pub(crate) vk: Arc<VulkanManager>,
    pub(crate) shared_render_data: Arc<RwLock<SharedRendererData>>,
    pub(crate) input: Input,
    pub(crate) fps_queue: VecDeque<f32>,
    pub(crate) frame_time: Instant,
    pub(crate) running: Arc<AtomicBool>,
    pub(crate) loop_thread: Option<Arc<JoinHandle<()>>>,
    pub(crate) file_watcher: FileWatcher,
}

impl Engine {
    pub(crate) fn new(event_loop: &EventLoop<()>, engine_dir: &PathBuf, project_dir: &str) -> Self {
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
                texture_manager.clone(),
                vk.mem_alloc.clone(),
            ),
            &["obj"],
        )));
        let rs_manager = Arc::new(Mutex::new(runtime_compilation::RSManager::new((), &["rs"])));

        assert!(env::set_current_dir(&Path::new(engine_dir)).is_ok()); // procedurally generate cube/move cube to built in assets
        model_manager
            .lock()
            .from_file("eng_res/cube/cube.obj");
        
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

        let mut perf = Perf {
            data: BTreeMap::<String, SegQueue<Duration>>::new(),
        };

        // let _loops = 0;

        // let mut gui = egui_winit_vulkano::Gui::new_with_subpass(
        //     &event_loop,
        //     vk.surface.clone(),
        //     Some(vk.swapchain().image_format()),
        //     vk.queue.clone(),
        //     Subpass::from(render_pass.clone(), 0).unwrap(),
        // );

        // let mut fc_map: HashMap<i32, HashMap<u32, TextureId>> = HashMap::new();

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

        let game_thread = Arc::new({
            let _running = running.clone();
            let world = world.clone();
            thread::spawn(move || main_loop(world.clone(), (rtx, rx), _running))
        });

        let _res = coms.1.send((input.clone(), false));
        let mut file_watcher = file_watcher::FileWatcher::new(".");

        let mut cam_data = CameraData::new(vk.clone());
        let mut playing_game = false;

        Self {
            world,
            assets_manager,
            project: Project::default(),
            transform_compute,
            playing_game,
            coms,
            perf,
            vk,
            shared_render_data: rm,
            input,
            particles_system,
            fps_queue,
            frame_time,
            running,
            loop_thread: Some(game_thread),
            file_watcher,
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
    pub(crate) fn update_sim(&mut self) -> Option<RenderingData> {
        // puffin::profile_scope!("wait for game");
        let inst = Instant::now();
        let rd = self.coms.0.recv().ok();

        self.perf
            .update("wait for game".into(), Instant::now() - inst);
        rd
    }
    pub(crate) fn render(
        &mut self,
        mut builder: &mut utils::PrimaryCommandBuffer,
        rd: RenderingData,
        _playing_game: bool,
        image_num: u32,
    ) {
        let RenderingData {
            transform_data,
            cam_datas,
            main_cam_id,
            renderer_data: mut rd,
            emitter_inits: particle_init_data,
            gpu_work,
        } = rd;

        let render_jobs = self.world.lock().render();

        let mut rm = self.shared_render_data.write();
        let vk = self.vk.clone();

        // start new game frame
        ///////////////////////////////////////////////////////////////
        let res = self.coms.1.send((self.input.clone(), _playing_game));
        if res.is_err() {
            return;
        }
        self.input.reset();
        /////////////////////////////////////////////////////////////////

        // let mut builder = AutoCommandBufferBuilder::primary(
        //     &vk.comm_alloc,
        //     vk.queue.queue_family_index(),
        //     CommandBufferUsage::OneTimeSubmit,
        // )
        // .unwrap();

        self.transform_compute.update_data(
            &mut builder,
            image_num,
            &transform_data,
            &mut self.perf,
        );

        let inst = Instant::now();
        self.particles_system.update(
            &mut builder,
            particle_init_data,
            &self.transform_compute,
            &self.input.time,
        );
        self.perf.update("particle update".into(), inst.elapsed());

        let inst = Instant::now();
        // compute shader renderers
        let offset_vec = {
            // puffin::profile_scope!("process renderers");
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
                &self.transform_compute,
            )
        };
        self.perf.update("update renderers".into(), inst.elapsed());
        let inst = Instant::now();

        while let Some(job) = gpu_work.pop() {
            job.unwrap()(&mut builder, vk.clone());
        }

        for cam in cam_datas {
            cam.lock().render(
                vk.clone(),
                builder,
                &mut self.transform_compute,
                self.particles_system.clone(),
                &transform_data,
                rm.pipeline.clone(),
                offset_vec.clone(),
                &mut rm,
                &mut rd,
                image_num,
                self.assets_manager.clone(),
                &render_jobs,
            );
        }
        // }

        self.perf.update("render camera(s)".into(), inst.elapsed());

        self.input.time.time += self.input.time.dt;
        self.input.time.dt = self.frame_time.elapsed().as_secs_f64() as f32;

        self.fps_queue.push_back(self.input.time.dt);
        if self.fps_queue.len() > 100 {
            self.fps_queue.pop_front();
        }

        self.frame_time = Instant::now();
    }
    pub(crate) fn end(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        let _ = self.coms.0.recv().unwrap();
        let _res = self.coms.1.send((self.input.clone(), self.playing_game));
        let loop_thread = self.loop_thread.as_ref().unwrap().clone();
        self.loop_thread = None;
        Arc::into_inner(loop_thread).unwrap().join().unwrap();
        save_project(
            &self.file_watcher,
            &self.world.lock(),
            self.assets_manager.clone(),
        );
        self.perf.print();
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
