use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    env,
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
use glm::{vec3, Vec3};
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
use vulkano::{
    buffer::Subbuffer,
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
    event_loop::{EventLoop, EventLoopBuilder, EventLoopProxy},
};
// use crate::{physics::Physics};

use crate::{
    editor::{self, editor_cam::EditorCam, editor_ui::EDITOR_WINDOW_DIM},
    engine::{
        particles::{component::ParticleEmitter, shaders::scs::l},
        physics::{
            collider::{_Collider, MESH_MAP, PROC_MESH_ID},
            rigid_body::_RigidBody,
        },
        // utils::look_at,
        prelude::{TransformRef, _Transform},
        project::asset_manager::AssetManagerBase,
        rendering::model::ModelManager,
        storage::Storage,
        world::transform::{Transforms, TRANSFORMS, TRANSFORM_MAP},
    },
};

use self::{
    input::Input,
    particles::particles::ParticleCompute,
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
    utils::GPUWork,
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
    framebuffers: Vec<(Arc<SwapchainImage>, Arc<Framebuffer>)>,
    recreate_swapchain: bool,
    editor_window_image: Option<Arc<dyn ImageAccess>>,
    render_pass: Arc<RenderPass>,
}
pub struct Engine {
    pub world: Arc<Mutex<World>>,
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
    pub time: Time,
    pub perf: Arc<Perf>,
    pub(crate) vk: Arc<VulkanManager>,
    pub(crate) shared_render_data: Arc<RwLock<SharedRendererData>>,
    pub(crate) fps_queue: VecDeque<f32>,
    pub(crate) frame_time: Instant,
    pub(crate) running: Arc<AtomicBool>,
    pub(crate) input_thread: Arc<JoinHandle<()>>,
    pub(crate) rendering_thread: Arc<JoinHandle<()>>,
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

use crate::engine::prelude::{ComponentID, _ComponentID};

#[derive(ComponentID, Default, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Player2 {
    rof: f32,
    speed: f32,
    grab_mode: bool,
    cursor_vis: bool,
    fov: f32,
    // explosion: AssetInstance<ParticleTemplate>,
    // shockwave: AssetInstance<ParticleTemplate>,
    // trail: AssetInstance<ParticleTemplate>,
    // flame: AssetInstance<ParticleTemplate>,
    init: bool,
    should_shoot: bool,
    player: TransformRef,
    // vel: Vec3,
}

impl Component for Player2 {
    fn update(&mut self, transform: &Transform, sys: &System, world: &World) {
        // unsafe {
        //     *EXPLOSION.get() = self.explosion;
        //     *SHOCKWAVE.get() = self.shockwave;
        // }
        let input = &sys.input;
        let speed = self.speed * sys.time.dt;

        if input.get_key_press(&VirtualKeyCode::G) {
            self.grab_mode = !self.grab_mode;
        }
        let _er = sys.vk.window().set_cursor_grab(match self.grab_mode {
            true => winit::window::CursorGrabMode::Locked,
            false => winit::window::CursorGrabMode::None,
        });
        // if self.grab_mode {
        match _er {
            Ok(_) => {}
            Err(e) => {
                if self.grab_mode {
                    match sys
                        .vk
                        .window()
                        .set_cursor_position(LogicalPosition::new(960, 540))
                    {
                        Ok(_) => {}
                        Err(e) => {
                            println!("{}", e);
                        }
                    }
                }
            }
        }
        // }
        if input.get_key_press(&VirtualKeyCode::H) {
            self.cursor_vis = !self.cursor_vis;
        }
        // if !self.cursor_vis {
        sys.vk.window().set_cursor_visible(self.cursor_vis);
        // }
        // if input.get_key_press(&VirtualKeyCode::J) {
        //     lock_cull = !lock_cull;
        //     // lock_cull.
        // }

        if input.get_key_press(&VirtualKeyCode::R) {
            self.speed *= 1.5;
        }
        if input.get_key_press(&VirtualKeyCode::F) {
            self.speed /= 1.5;
        }

        let mut vel = vec3(0., 0., 0.);
        // forward/backward
        if input.get_key(&VirtualKeyCode::W) {
            vel += vec3(0., 0., 1.);
            // transform.translate(vec3(0., 0., 1.) * speed);
        }
        if input.get_key(&VirtualKeyCode::S) {
            vel -= vec3(0., 0., 1.);
            // transform.translate(vec3(0., 0., 1.) * -speed);
        }
        // up/down
        if input.get_key(&VirtualKeyCode::Space) {
            vel += vec3(0., 1., 0.);
            // transform.translate(vec3(0., 1., 0.) * speed);
        }
        if input.get_key(&VirtualKeyCode::LShift) {
            vel -= vec3(0., 1., 0.);
            // transform.translate(vec3(0., 1., 0.) * -speed);
        }
        //left/right
        if input.get_key(&VirtualKeyCode::A) {
            vel -= vec3(1., 0., 0.);
            // transform.translate(vec3(1., 0., 0.) * -speed);
        }
        if input.get_key(&VirtualKeyCode::D) {
            vel += vec3(1., 0., 0.);
            // transform.translate(vec3(1., 0., 0.) * speed);
        }

        let y = input.get_mouse_scroll().1;
        self.fov = self.fov.add(y).clamp(5., 50.);
        // todo get_component Camera. adjust fov

        // if input.get_mouse_button(&2) {
        // let mut cam_rot = transform.get_rotation();
        // cam_rot = glm::quat_rotate(
        //     &cam_rot,
        //     input.get_mouse_delta().0 as f32 * 0.01,
        //     &(glm::inverse(&glm::quat_to_mat3(&cam_rot)) * Vec3::y()),
        // );
        // cam_rot = glm::quat_rotate(
        //     &cam_rot,
        //     input.get_mouse_delta().1 as f32 * 0.01,
        //     &Vec3::x(),
        // );
        // transform.set_rotation(cam_rot);
        transform.rotate(&Vec3::y(), input.get_mouse_delta().0 as f32 * 0.01);
        transform.rotate(&Vec3::x(), input.get_mouse_delta().1 as f32 * 0.01);

        transform.look_at(&transform.forward(), &Vec3::y());
        let player = self.player.clone();
        // let vel = self.vel;
        let rotation = transform.get_rotation();
        let speed = speed;
        sys.defer.append(move |world: &mut World| {
            if let Some(player_transform) = player.get() {
                player_transform.move_child(glm::quat_rotate_vec3(&rotation, &(vel * speed)));
            }
        });

        // if !self.init {
        //     // transform.set_position(vec3(0., 0., 0.));
        //     transform.set_rotation(look_at(&vec3(0.,0.,1.), &Vec3::y()));
        //     assert!(transform.forward() == vec3(0.,0.,1.));
        // //     let trail = self.trail.clone();
        // //     let flame = self.flame.clone();
        // //     for _ in 0..2000 {
        // //         world
        // //             .instantiate(0)
        // //             .with_transform(|| _Transform {
        // //                 position: glm::vec3(
        // //                     (rand::random::<f32>() - 0.5) * 1500f32,
        // //                     140f32,
        // //                     (rand::random::<f32>() - 0.5) * 1500f32,
        // //                 ),
        // //                 ..Default::default()
        // //             })
        // //             .with_com(move || Shooter {
        // //                 rpm: 0,
        // //                 trail,
        // //                 flame,
        // //                 accum: rand::random::<f32>() * 2.7,
        // //             }) // random max accum per frame
        // //             .build();
        // //     }
        //     self.init = true;
        // }
        // const ALOT: f32 = 10_000_000. / 60.;
        // // let id = self.trail.id;
        // // let id2 = self.flame.id;
        // let len = (ALOT * sys.time.dt.min(1.0 / 30.0)) as usize;
        // if input.get_key_press(&VirtualKeyCode::O) {
        //     self.should_shoot = !self.should_shoot;
        // }
        // if self.should_shoot {
        //     world
        //         .instantiate_many(len as i32, 0, -1)
        //         .with_transform(|| _Transform {
        //             position: glm::vec3(
        //                 (rand::random::<f32>() - 0.5) * 1500f32,
        //                 140f32,
        //                 (rand::random::<f32>() - 0.5) * 1500f32,
        //             ),
        //             ..Default::default()
        //         })
        //         .with_com(|| Bomb {
        //             vel: glm::Vec3::y() * 70. + utils::rand_sphere() * 50.,
        //         })
        //         .with_com(move || ParticleEmitter::new(id))
        //         // .with_com(move || ParticleEmitter::new(id2))
        //         // .with_com(move || Renderer::new(0))
        //         .build();
        // }

        if input.get_mouse_button(&2) {
            // let len = (ALOT / 1000. * sys.time.dt.min(1.0 / 30.0)) as usize;
            let pos = transform.get_position() - transform.up() * 10. + transform.forward() * 15.;
            let forw = transform.forward();
            world
                .instantiate(-1)
                .with_transform(move || _Transform {
                    position: pos,
                    ..Default::default()
                })
                // .with_com(move || Bomb {
                //     vel: forw * 100. + utils::rand_sphere() * 40. * rand::random::<f32>().sqrt(),
                // })
                .with_com(|| Renderer::new(0))
                // .with_com(move || ParticleEmitter::new(id))
                // .with_com(move || ParticleEmitter::new(id2))
                .with_com(move || _RigidBody::new(_ColliderType::Cuboid(vec3(1.0, 1.0, 1.0))))
                .build();
        }
    }
    fn inspect(&mut self, _transform: &Transform, _id: i32, ui: &mut egui::Ui, sys: &Sys) {
        Ins(&mut self.rof).inspect("rof", ui, sys);
        Ins(&mut self.speed).inspect("speed", ui, sys);
        // Ins(&mut self.explosion).inspect("explosion", ui, sys);
        // Ins(&mut self.shockwave).inspect("shockwave", ui, sys);
        // Ins(&mut self.trail).inspect("trail", ui, sys);
        // Ins(&mut self.flame).inspect("flame", ui, sys);
        Ins(&mut self.player).inspect("player container", ui, sys);
    }
}

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
        let rs_manager = Arc::new(Mutex::new(runtime_compilation::RSManager::new((), &["rs"])));

        assert!(env::set_current_dir(&Path::new(engine_dir)).is_ok()); // procedurally generate cube/move cube to built in assets
        model_manager.lock().from_file("default/cube/cube.obj");

        texture_manager.lock().from_file("default/particle.png");
        assert!(env::set_current_dir(&Path::new(project_dir)).is_ok());

        let particles_system = Arc::new(ParticleCompute::new(vk.clone(), texture_manager.clone()));

        let world = Arc::new(Mutex::new(World::new(
            particles_system.clone(),
            vk.clone(),
            assets_manager.clone(),
        )));
        unsafe {
            TRANSFORMS = std::ptr::from_mut(&mut world.lock().transforms);
            TRANSFORM_MAP = std::ptr::from_mut(&mut world.lock().transform_map);
            MESH_MAP = std::ptr::from_mut(&mut world.lock().mesh_map);
            PROC_MESH_ID = std::ptr::from_mut(&mut world.lock().proc_mesh_id);
        }

        #[cfg(target_os = "windows")]
        let dylib_ext = ["dll"];
        #[cfg(not(target_os = "windows"))]
        let dylib_ext = ["so"];

        let lib_manager = Arc::new(Mutex::new(runtime_compilation::LibManager::new(
            world.clone(),
            &dylib_ext,
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
            world.register::<_Collider>(false, false, false);
            world.register::<_RigidBody>(false, false, false);
            //
            world.register::<terrain_eng::TerrainEng>(true, false, true);
            world.register::<Player2>(true, false, false);
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
            transform_compute: RwLock::new(transform_compute),
            playing_game: game_mode,
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
            tex_id: None,
            image_view: None,
            game_mode,
            gui: unsafe { SendSync::new(gui) },
            update_editor_window: true,
            event_loop_proxy: proxy,
            engine_dir: engine_dir.as_path().to_str().unwrap().to_string(),
        }
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
        serialize::deserialize(&mut self.world.lock());
    }
    pub fn update_sim(&mut self) -> bool {
        let full_frame_time = self.perf.node("full frame time");
        let (events, input, window_size, should_exit) = self.input.recv().unwrap();
        for event in events {
            self.gui.update(&event);
        }
        let mut world = self.world.lock();
        let gpu_work = SegQueue::new();
        let world_sim = self.perf.node("world _update");
        let cvd = if (self.game_mode | self.playing_game) {
            puffin::profile_scope!("game loop");
            {
                // if self.playing_game != _playing_game {
                // world.sys.physics.lock().reset();
                // }
                puffin::profile_scope!("world update");
                if world.phys_time >= world.phys_step {
                    // let mut physics = world.sys.physics.lock();
                    // let len = world.sys.physics.lock().rigid_body_set.len();
                    // let num_threads = (len / (num_cpus::get().sqrt())).max(1).min(num_cpus::get());
                    {
                        let mut physics = world.sys.physics.lock();
                        let a = &world.get_components::<_Collider>().as_ref().unwrap().1;
                        let b = a.write();
                        let c = b.as_any().downcast_ref::<Storage<_Collider>>().unwrap();
                        (0..c.data.len()).into_par_iter().for_each(|i| {
                            if unsafe { *c.valid[i].get() } {
                                let d = c.data[i].lock();
                                let t = world.transforms.get(d.0).unwrap();
                                let handle = d.1.handle;
                                let collider =
                                    &mut (unsafe { &mut *physics.collider_set.get() })[handle];
                                collider.set_translation(t.get_position());
                                collider.set_rotation(UnitQuaternion::from_quaternion(
                                    t.get_rotation(),
                                ));
                            }
                        });
                        physics.step(&self.perf);
                        physics
                            .island_manager
                            .active_dynamic_bodies()
                            .par_iter()
                            .chain(physics.island_manager.active_kinematic_bodies().par_iter())
                            .for_each(|a| {
                                let rb = unsafe { physics.get_rigid_body(*a).unwrap() };
                                let t = world.transforms.get(rb.user_data as i32).unwrap();
                                t.set_position(rb.translation());
                                t.set_rotation(rb.rotation());
                            });
                    }
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
                    world.defer_instantiate(&input, &self.time, &gpu_work, &self.perf);
                    world.init_colls_rbs();
                }
            }
            // let mut world = self.world.lock();
            world.update_cameras()
        } else {
            // let mut world = self.world.lock();

            world._destroy(&self.perf);
            world.init_colls_rbs();
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
        // let mut world = self.world.lock();

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
                if self.game_mode {
                    true
                } else {
                    self.playing_game
                },
            );
        });
        // if _playing_game && self.playing_game != _playing_game {
        //     Command::new("cargo")
        //         .args(["run", "-r", "--bin", "game", "--", &self.engine_dir])
        //         .status()
        //         .unwrap();
        // }
        if !(self.game_mode | self.playing_game) {
            cam_datas = vec![cd.clone()];
            // let cd = self.cam_data.clone();
        };
        let transform_extent = world.transforms.last_active();

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

        let _recreate_swapchain = window_size.is_some();
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
                let dims = if self.game_mode {
                    framebuffers[0].0.dimensions().width_height()
                } else {
                    *EDITOR_WINDOW_DIM.lock()
                };
                cam.lock().resize(dims, vk.clone());
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
                Format::E5B9G9R9_UFLOAT_PACK32,
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
        // if self.game_mode {
        //     if let Some(game_image) = game_image {
        //         builder
        //             .copy_image(CopyImageInfo::images(game_image, .clone()))
        //             .unwrap();
        //     }
        // }
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
                    return should_exit;
                }
                Err(e) => panic!("Failed to acquire next image: {:?}", e),
            };
        if suboptimal {
            *recreate_swapchain = true;
        }

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
        // self.world.lock().sys.physics.lock().get_counters();
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
