use bytemuck::{Pod, Zeroable};
use component_derive::component;
use engine::{
    physics,
    transform::{Transform, Transforms},
    Component, LazyMaker,
};
use glm::{cross, normalize, vec3, vec4, Mat4, Vec3, Vec4};
use model::{Normal, Vertex};
use rust_test::{INDICES, NORMALS, VERTICES};

mod engine;
// use transform::{Transform, Transforms};

use std::{
    any::{Any, TypeId},
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    env,
};

// use rand::prelude::*;
use rapier3d::{
    na::{Matrix4, Point3, Quaternion, Vector3},
    prelude::*,
};
use rayon::prelude::*;

use crossbeam::queue::SegQueue;
use nalgebra_glm as glm;
use parking_lot::{Mutex, RwLock};

use crate::engine::{physics::Physics, World};
mod model;

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
            1.0 / 100.0,
            false,
            InteractionGroups::all(),
            None,
        ) {
            // let hit_point = ray.point_at(hit.toi); // Same as: `ray.origin + ray.dir * toi`
            // let hit_normal = hit.normal;
            // let v = nalgebra_glm::vec3(vel.x,vel.y,vel.z);
            self.vel = nalgebra_glm::reflect_vec(&vel, &&_hit.normal);
            // let v = v * x;
            // pos.x = hit_point.x;
            // pos.y = hit_point.y;
            // pos.z = hit_point.z;

            // self.vel.y = if vel.y > 0.0 {vel.y} else {-vel.y};
        }
        trans._move(self.t, self.vel * (1.0 / 100.0));
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

use std::{sync::Arc, time::Instant};
use vulkano::{
    buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess},
    command_buffer::{
        sys::RenderPassBeginInfo, AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents,
    },
    descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet},
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo,
    },
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageAccess, ImageUsage, SwapchainImage},
    impl_vertex,
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
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    event::{
        DeviceEvent, ElementState, Event, KeyboardInput, ModifiersState, VirtualKeyCode,
        WindowEvent,
    },
    event_loop::{ControlFlow, EventLoop},
    platform::unix::x11::{ffi::Cursor, util::NormalHints},
    window::{Window, WindowBuilder},
};
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct ModelMat {
    model: [[f32; 4]; 4],
}
impl_vertex!(ModelMat, model);

fn main() {
    env::set_var("RUST_BACKTRACE", "1");

    let mut physics = Physics::new();

    /* Create the ground. */
    let collider = ColliderBuilder::cuboid(100.0, 0.1, 100.0).build();
    physics.collider_set.insert(collider);

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
        world
            .transforms
            .read()
            ._move(g.t, glm::vec3(0.0, 10.0, 0.0));
    }
    // {
    //     // maker
    //     let g = world.instantiate();
    //     world.add_component(g, Maker { t: Transform(-1) });
    // }
    let lazy_maker = LazyMaker::new();

    /////////////////////////////////////////////////

    let required_extensions = vulkano_win::required_extensions();
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: required_extensions,
        ..Default::default()
    })
    .unwrap();

    let event_loop = EventLoop::new();
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
                ..Default::default()
            },
        )
        .unwrap()
    };

    let model = model::load_model();

    let vertex_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, model.vertices)
            .unwrap();
    let normals_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, model.normals)
            .unwrap();
    let index_buffer =
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, model.indeces)
            .unwrap();

    //     let vertex_buffer =
    //     CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, VERTICES)
    //         .unwrap();
    // let normals_buffer =
    //     CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, NORMALS).unwrap();
    // let index_buffer =
    //     CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, INDICES).unwrap();

    let uniform_buffer = CpuBufferPool::<vs::ty::Data>::new(device.clone(), BufferUsage::all());

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

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
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
    .unwrap();

    let (mut pipeline, mut framebuffers) =
        window_size_dependent_setup(device.clone(), &vs, &fs, &images, render_pass.clone());
    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());
    let rotation_start = Instant::now();

    let mut modifiers = ModifiersState::default();

    let mut cam_pos = glm::vec3(0.0 as f32, 0.0, 5.0);
    let mut cam_rot = glm::quat(1.0, 0.0, 0.0, 0.0);

    let mut key_downs = HashMap::<VirtualKeyCode, bool>::new();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta } => {
                    // println!("mouse moved: {:?}", delta);

                    let rot = glm::inverse(&glm::quat_to_mat3(&cam_rot));

                    cam_rot = glm::quat_rotate(&cam_rot, delta.0 as f32 * 0.01, &(rot * Vec3::y())); // left/right
                    cam_rot = glm::quat_rotate(&cam_rot, delta.1 as f32 * -0.01, &(rot * Vec3::x())); // up/down

                    // cam_rot = glm::to_quat(
                    //     &(glm::quat_to_mat4(&cam_rot)
                    //         * Mat4::from_scaled_axis(Vec3::y() * delta.0 as f32 * -0.01)),
                    // );
                    // cam_rot = glm::to_quat(
                    //     &(glm::quat_to_mat4(&cam_rot)
                    //         * Mat4::from_scaled_axis(Vec3::x() * delta.1 as f32 * 0.01)),
                    // );
                    let target = glm::quat_to_mat3(&cam_rot) * vec3(0.0, 0.0, 1.0);
                    
                    cam_rot = glm::quat_look_at_lh(
                        // &eye
                        &target,
                        &-vec3(0.0, 1.0, 0.0),
                    );

                    // let target = glm::quat_to_mat3(&cam_rot) * vec3(0.0, 0.0, 1.0);
                    // cam_rot = glm::quat_look_at_lh(
                    //     // &eye
                    //     &target,
                    //     &vec3(0.0, 1.0, 0.0),
                    // );
                    // let forward = target;
                    // let right = normalize(&cross(&Vec3::y(),&forward));
                    // let up = normalize(&cross(&forward,&right));

                    // let m00 = right.x;
                    // let m01 = up.x;
                    // let m02 = forward.x;
                    // let m10 = right.x;
                    // let m11 = up.x;
                    // let m12 = forward.x;
                    // let m20 = right.x;
                    // let m21 = up.x;
                    // let m22 = forward.x;

                    // // let ret = glm::quat(0.0 as f32,0.0,0.0,0.0);

                    // let w = (1.0f32 + m00 + m11 + m22).sqrt() * 0.5f32;
                    // let w4_recip = 1.0f32 / (4.0f32 * w);
                    // let x = (m21 - m12) * w4_recip;
                    // let y = (m02 - m20) * w4_recip;
                    // let z = (m10 - m01) * w4_recip;

                    // cam_rot = glm::quat(w,x,y,z);

                    // cam_rot = glm::to_quat(&Matrix4::look_at_rh(
                    //     &Point3::new(0.0 as f32, 0.0, 0.0),
                    //     &(glm::quat_to_mat4(&cam_rot) * glm::vec4(0.0 as f32, 0.0, 1.0, 1.0)).xyz()
                    //         .into(),
                    //     &Vector3::new(0.0, -1.0, 0.0),
                    // ));

                    // cam_rot = glm::quat_rotate(
                    //     &cam_rot,
                    //     delta.0 as f32 * 0.01,
                    //     &(glm::quat_to_mat3(&cam_rot) * vec3(0.0, 1.0, 0.0)),
                    // ); // left/right
                    // cam_rot = glm::quat_rotate(
                    //     &cam_rot,
                    //     delta.1 as f32 * 0.01,
                    //     &(glm::quat_to_mat3(&cam_rot) * vec3(1.0, 0.0, 0.0)),
                    // ); // up/down
                }
                DeviceEvent::Button { button, state } => match state {
                    ElementState::Pressed => println!("mouse button {} pressed", button),
                    ElementState::Released => println!("mouse button {} released", button),
                },
                _ => (),
            },
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            state: ElementState::Released,
                            virtual_keycode: Some(key),
                            ..
                        },
                    ..
                } => {
                    use winit::event::VirtualKeyCode::*;
                    let result = match key {
                        Escape => {
                            *control_flow = ControlFlow::Exit;
                            Ok(())
                        }
                        G => surface.window().set_cursor_grab(true),
                        L => surface.window().set_cursor_grab(false),
                        // A => surface.window().set_cursor_grab(),
                        H => {
                            surface.window().set_cursor_visible(modifiers.shift());
                            Ok(())
                        }
                        _ => {
                            key_downs.insert(key.clone(), false);
                            Ok(())
                        }
                    };

                    if let Err(err) = result {
                        println!("error: {}", err);
                    }
                }
                WindowEvent::Resized(_) => {
                    recreate_swapchain = true;
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
                            key_downs.insert(key.clone(), true);
                        }
                    };
                }
                _ => (),
            },

            // Event::WindowEvent {
            //     event: WindowEvent::Resized(_),
            //     ..
            // } => {
            //     recreate_swapchain = true;
            // }
            Event::RedrawEventsCleared => {
                ////////////////////////////////////
                
                let speed = 0.05;
                let rot = glm::inverse(&glm::quat_to_mat3(&cam_rot));
                // forward/backward
                if *key_downs.get(&VirtualKeyCode::W).unwrap_or(&false) {
                    cam_pos += rot * vec3(0.0, 0.0, 1.0) * speed;
                }
                if *key_downs.get(&VirtualKeyCode::S).unwrap_or(&false) {
                    cam_pos += rot * vec3(0.0, 0.0, -1.0) * speed;
                }
                //left/right
                if *key_downs.get(&VirtualKeyCode::A).unwrap_or(&false) {
                    cam_pos += rot * vec3(1.0, 0.0, 0.0) * speed;
                }
                if *key_downs.get(&VirtualKeyCode::D).unwrap_or(&false) {
                    cam_pos += rot * vec3(-1.0, 0.0, 0.0) * speed;
                }
                // up/down
                if *key_downs.get(&VirtualKeyCode::Space).unwrap_or(&false) {
                    cam_pos += rot * vec3(0.0, 1.0, 0.0) * speed;
                }
                if *key_downs.get(&VirtualKeyCode::LShift).unwrap_or(&false) {
                    cam_pos += rot * vec3(0.0, -1.0, 0.0) * speed;
                }
                // roll
                if *key_downs.get(&VirtualKeyCode::Q).unwrap_or(&false) {
                    cam_rot = glm::quat_rotate(&cam_rot, 0.03, &(rot * Vec3::z()));
                    // cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 1.0, 0.0, 1.0))
                    //     .xyz()
                    //     * 0.01
                }
                if *key_downs.get(&VirtualKeyCode::E).unwrap_or(&false) {
                    cam_rot = glm::quat_rotate(&cam_rot, -0.03, &(rot * Vec3::z()));
                    // cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, -1.0, 0.0, 1.0))
                    //     .xyz()
                    //     * 0.01
                }

                physics.step(&gravity);
                world.update(&physics, &lazy_maker);

                lazy_maker.init(&mut world);

                // let x = world.transforms.write().positions[10].lock().clone();
                // let mut models: Vec<ModelMat> = world
                //     .transforms
                //     .write()
                //     .positions
                //     .iter()
                //     // .filter(|p| {
                //     //     p.lock().z < -10.0
                //     // })
                //     .map(|p| ModelMat {
                //         model: Mat4::new_translation(&p.lock()).into(),
                //     })
                //     .collect();
                
                let target = rot * vec3(0.0, 0.0, 1.0);
                // glm::scale(&Mat4::identity(), &Vector3::new(0.05 as f32, 0.05, 0.1)) * glm::quat_to_mat4(&cam_rot) *
                let models: Vec<ModelMat> = vec![ModelMat {
                    model: ( glm::scale(&Mat4::identity(), &Vector3::new(0.5 as f32, 0.5, 0.5)) * glm::inverse(&glm::quat_to_mat4(&cam_rot)) * Mat4::new_translation(&(target))).into(),
                }];

                // let models = Mat4::new_translation(&x);
                let instance_buffer = CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::all(),
                    false,
                    models,
                )
                .unwrap();

                ////////////////////////////////////

                let dimensions = surface.window().inner_size();
                if dimensions.width == 0 || dimensions.height == 0 {
                    return;
                }

                previous_frame_end.as_mut().unwrap().cleanup_finished();

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
                    let (new_pipeline, new_framebuffers) = window_size_dependent_setup(
                        device.clone(),
                        &vs,
                        &fs,
                        &new_images,
                        render_pass.clone(),
                    );
                    pipeline = new_pipeline;
                    framebuffers = new_framebuffers;
                    recreate_swapchain = false;
                }

                let uniform_buffer_subbuffer = {
                    let elapsed = rotation_start.elapsed();
                    let rotation =
                        elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
                    // let rotation = cgmath::Matrix3::from_angle_y(cgmath::Rad(rotation as f32));
                    let axis = nalgebra::Vector3::y_axis();
                    let angle = rotation as f32;
                    let rotation = nalgebra::Rotation3::from_axis_angle(&axis, angle);

                    // note: this teapot was meant for OpenGL where the origin is at the lower left
                    //       instead the origin is at the upper left in Vulkan, so we reverse the Y axis
                    let aspect_ratio =
                        swapchain.image_extent()[0] as f32 / swapchain.image_extent()[1] as f32;
                    let proj = glm::perspective(
                        aspect_ratio,
                        std::f32::consts::FRAC_PI_2 as f32,
                        0.01,
                        100.0,
                    );
                    // let view = glm::quat_to_mat4(&cam_rot) * glm::Mat4::new_translation(&cam_pos);
                    // let view = Matrix4::look_at_rh(
                    //     &cam_pos.into(),
                    //     &(cam_pos
                    //         + (glm::quat_to_mat3(&cam_rot) * glm::vec3(0.0 as f32, 0.0, 1.0))
                    //             .xyz())
                    //     .into(),
                    //     &(glm::quat_to_mat3(&cam_rot) * Vector3::new(0.0, 1.0, 0.0)),
                    // );

                    // let eye = cam_pos; //Vec3::new(self.position.x,self.position.y,self.position.z);
                    let target = glm::quat_to_mat3(&cam_rot) * Vec3::z();
                    let up = glm::quat_to_mat3(&cam_rot) * Vec3::y();
                    let view = glm::inverse(&glm::look_at_lh(&vec3(0.0, 0.0, 0.0), &target, &up));
                    let view = view * glm::Mat4::new_translation(&cam_pos);

                    let scale = glm::scale(&Mat4::identity(), &Vector3::new(0.1 as f32, 0.1, 0.1));

                    let uniform_data = vs::ty::Data {
                        // world: model.into(),
                        view: (view * scale).into(),
                        proj: proj.into(),
                    };

                    uniform_buffer.next(uniform_data).unwrap()
                };

                let layout = pipeline.layout().set_layouts().get(0).unwrap();
                let set = PersistentDescriptorSet::new(
                    layout.clone(),
                    [WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer)],
                )
                .unwrap();

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
                    .unwrap()
                    .bind_pipeline_graphics(pipeline.clone())
                    .bind_descriptor_sets(
                        PipelineBindPoint::Graphics,
                        pipeline.layout().clone(),
                        0,
                        set.clone(),
                    )
                    .bind_vertex_buffers(
                        0,
                        (
                            vertex_buffer.clone(),
                            normals_buffer.clone(),
                            instance_buffer.clone(),
                        ),
                    )
                    .bind_index_buffer(index_buffer.clone())
                    .draw_indexed(
                        index_buffer.len() as u32,
                        instance_buffer.len() as u32,
                        0,
                        0,
                        0,
                    )
                    .unwrap()
                    .end_render_pass()
                    .unwrap();
                let command_buffer = builder.build().unwrap();

                let future = previous_frame_end
                    .take()
                    .unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer)
                    .unwrap()
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
            _ => (),
        }
    });
}

/// This method is called once during initialization, then again whenever the window is resized
fn window_size_dependent_setup(
    device: Arc<Device>,
    vs: &ShaderModule,
    fs: &ShaderModule,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
) -> (Arc<GraphicsPipeline>, Vec<Arc<Framebuffer>>) {
    let dimensions = images[0].dimensions().width_height();

    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(device.clone(), dimensions, Format::D16_UNORM).unwrap(),
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

    // In the triangle example we use a dynamic viewport, as its a simple example.
    // However in the teapot example, we recreate the pipelines with a hardcoded viewport instead.
    // This allows the driver to optimize things, at the cost of slower window resizes.
    // https://computergraphics.stackexchange.com/questions/5742/vulkan-best-way-of-updating-pipeline-viewport
    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(
            BuffersDefinition::new()
                .vertex::<Vertex>()
                .vertex::<Normal>()
                .instance::<ModelMat>(),
        )
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_fixed_scissor_irrelevant([
            Viewport {
                origin: [0.0, 0.0],
                dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                depth_range: 0.0..1.0,
            },
        ]))
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    (pipeline, framebuffers)
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/vert.glsl",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/frag.glsl"
    }
}
