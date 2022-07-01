use bytemuck::{Pod, Zeroable};
use component_derive::component;
use engine::{
    physics,
    transform::{Transform, Transforms},
    Component, LazyMaker,
};
use glm::{Mat4, Vec3};
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
    na::{Matrix4, Point3, Vector3},
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
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
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

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(_),
                ..
            } => {
                recreate_swapchain = true;
            }
            Event::RedrawEventsCleared => {
                ////////////////////////////////////

                physics.step(&gravity);
                world.update(&physics, &lazy_maker);

                lazy_maker.init(&mut world);

                // let x = world.transforms.write().positions[10].lock().clone();
                let models: Vec<ModelMat> = world
                    .transforms
                    .write()
                    .positions
                    .iter()
                    // .filter(|p| {
                    //     p.lock().z < -10.0
                    // })
                    .map(|p| ModelMat {
                        model: Mat4::new_translation(&p.lock()).into(),
                    })
                    .collect();

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
                    let view = Matrix4::look_at_rh(
                        &Point3::new(0.3 as f32, 0.3, 1.0),
                        &Point3::new(0.0, 0.0, 0.0),
                        &Vector3::new(0.0, -1.0, 0.0),
                    );
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

    // event_loop.run(move |event, _, control_flow| {
    //     match event {
    //         Event::WindowEvent {
    //             event: WindowEvent::CloseRequested,
    //             ..
    //         } => {
    //             *control_flow = ControlFlow::Exit;
    //         }
    //         Event::WindowEvent {
    //             event: WindowEvent::Resized(_),
    //             ..
    //         } => {
    //             recreate_swapchain = true;
    //         }
    //         Event::RedrawEventsCleared => {

    //             ////////////////////////////////////
    //             ///
    //             physics.step(&gravity);
    //             world.update(&physics, &lazy_maker);

    //             lazy_maker.init(&mut world);

    //             ////////////////////////////////////

    //             let dimensions = surface.window().inner_size();
    //             if dimensions.width == 0 || dimensions.height == 0 {
    //                 return;
    //             }
    //             previous_frame_end.as_mut().unwrap().cleanup_finished();
    //             if recreate_swapchain {
    //                 let (new_swapchain, new_images) =
    //                     match swapchain.recreate(SwapchainCreateInfo {
    //                         image_extent: dimensions.into(),
    //                         ..swapchain.create_info()
    //                     }) {
    //                         Ok(r) => r,
    //                         Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
    //                         Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
    //                     };

    //                 swapchain = new_swapchain;
    //                 framebuffers = window_size_dependent_setup(
    //                     &new_images,
    //                     render_pass.clone(),
    //                     &mut viewport,
    //                 );
    //                 recreate_swapchain = false;
    //             }

    //             // after which the function call will return an error.
    //             let (image_num, suboptimal, acquire_future) =
    //                 match acquire_next_image(swapchain.clone(), None) {
    //                     Ok(r) => r,
    //                     Err(AcquireError::OutOfDate) => {
    //                         recreate_swapchain = true;
    //                         return;
    //                     }
    //                     Err(e) => panic!("Failed to acquire next image: {:?}", e),
    //                 };

    //             if suboptimal {
    //                 recreate_swapchain = true;
    //             }
    //             let mut builder = AutoCommandBufferBuilder::primary(
    //                 device.clone(),
    //                 queue.family(),
    //                 CommandBufferUsage::OneTimeSubmit,
    //             )
    //             .unwrap();

    //             builder
    //                 // Before we can draw, we have to *enter a render pass*.
    //                 .begin_render_pass(
    //                     framebuffers[image_num].clone(),
    //                     SubpassContents::Inline,
    //                     vec![[0.0, 0.0, 0.0, 1.0].into()], // clear color
    //                 )
    //                 .unwrap()
    //                 .set_viewport(0, [viewport.clone()])
    //                 .bind_pipeline_graphics(pipeline.clone())
    //                 .bind_vertex_buffers(0, vertex_buffer.clone())
    //                 .draw(vertex_buffer.len() as u32, 1, 0, 0)
    //                 .unwrap()
    //                 .end_render_pass()
    //                 .unwrap();

    //             // Finish building the command buffer by calling `build`.
    //             let command_buffer = builder.build().unwrap();

    //             let future = previous_frame_end
    //                 .take()
    //                 .unwrap()
    //                 .join(acquire_future)
    //                 .then_execute(queue.clone(), command_buffer)
    //                 .unwrap()
    //                 .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
    //                 .then_signal_fence_and_flush();

    //             match future {
    //                 Ok(future) => {
    //                     previous_frame_end = Some(future.boxed());
    //                 }
    //                 Err(FlushError::OutOfDate) => {
    //                     recreate_swapchain = true;
    //                     previous_frame_end = Some(sync::now(device.clone()).boxed());
    //                 }
    //                 Err(e) => {
    //                     println!("Failed to flush future: {:?}", e);
    //                     previous_frame_end = Some(sync::now(device.clone()).boxed());
    //                 }
    //             }
    //         }
    //         _ => (),
    //     }
    // });

    // use std::time::Instant;
    // let now = Instant::now();
    // for _ in 0..4000 {

    //     physics.step(&gravity);
    //     world.update(&physics, &lazy_maker);

    //     lazy_maker.init(&mut world);
    // }
    // // println!("{:?}", vals);
    // let elapsed = now.elapsed() / 4000;
    // println!("Elapsed: {:.2?}", elapsed);
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
