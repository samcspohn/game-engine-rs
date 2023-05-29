use std::{
    collections::{HashMap},
    hash::BuildHasherDefault,
    sync::{
        atomic::{AtomicI32, Ordering},
        Arc,
    },
};

use nohash_hasher::NoHashHasher;
use parking_lot::Mutex;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder,
        PrimaryAutoCommandBuffer,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo,
    },
    image::{ImageUsage, SwapchainImage},
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    query::{QueryControlFlags, QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::LogicalSize,
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};

#[repr(C)]
pub struct VulkanManager {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub surface: Arc<Surface>,
    pub swapchain: Mutex<Arc<Swapchain>>,
    pub images: Vec<Arc<SwapchainImage>>,
    pub instance: Arc<Instance>,
    // pub event_loop: EventLoop<()>,
    pub mem_alloc: Arc<StandardMemoryAllocator>,
    pub desc_alloc: Arc<StandardDescriptorSetAllocator>,
    pub comm_alloc: Arc<StandardCommandBufferAllocator>,
    pub query_pool: Mutex<HashMap<i32, Arc<QueryPool>, nohash_hasher::BuildNoHashHasher<i32>>>,
    query_counter: AtomicI32,
}

impl VulkanManager {
    pub fn new(event_loop: &EventLoop<()>) -> Arc<Self> {
        // rayon::ThreadPoolBuilder::new().num_threads(63).build_global().unwrap();
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = vulkano_win::required_extensions(&library);
        // required_extensions.ext_headless_surface = true;
        // Now creating the instance.
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
                enumerate_portability: true,
                ..Default::default()
            },
        )
        .unwrap();

        // let event_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            // .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
            .with_inner_size(LogicalSize {
                width: 1920,
                height: 1080,
            })
            .build_vk_surface(event_loop, instance.clone())
            .unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            // nv_geometry_shader_passthrough: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.graphics
                            && p.surface_support(i as u32, &surface).unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 1,
                PhysicalDeviceType::IntegratedGpu => 0,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .expect("No suitable physical device found");

        // Some little debug infos.
        println!(
            "Using device: {} (type: {:?})",
            physical_device.properties().device_name,
            physical_device.properties().device_type,
        );

        // Now initializing the device. This is probably the most important object of Vulkan.
        //
        // The iterator of created queues is returned by the function alongside the device.

        let features = Features {
            geometry_shader: true,
            ..Default::default()
        };
        let (device, mut queues) = Device::new(
            // Which physical device to connect to.
            physical_device,
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: features,
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],

                ..Default::default()
            },
        )
        .unwrap();

        // Since we can request multiple queues, the `queues` variable is in fact an iterator. We
        // only use one queue in this example, so we just retrieve the first and only element of the
        // iterator.
        let queue = queues.next().unwrap();

        // let img_count = swapchain.image_count();
        let mem_alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let desc_alloc = Arc::new(StandardDescriptorSetAllocator::new(device.clone()));
        let comm_alloc = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let (swapchain, images) = {
            // Querying the capabilities of the surface. When we create the swapchain we can only
            // pass values that are allowed by the capabilities.
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();

            // Choosing the internal format that the images will have.
            let image_format = Some(
                device
                    .physical_device()
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            );
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();

            // Please take a look at the docs for the meaning of the parameters we didn't mention.
            Swapchain::new(
                device.clone(),
                surface.clone(),
                SwapchainCreateInfo {
                    min_image_count: surface_capabilities.min_image_count,
                    image_format,
                    image_extent: window.inner_size().into(),
                    image_usage: ImageUsage {
                        color_attachment: true,
                        ..ImageUsage::empty()
                    },
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
        // self.swapchain = swapchain.clone();
        // let query_pool = QueryPool::new(
        //     device.clone(),
        //     QueryPoolCreateInfo {
        //         query_count: 3,
        //         ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
        //     },
        // )
        // .unwrap();

        Arc::new(Self {
            device,
            queue,
            instance,
            surface,
            swapchain: Mutex::new(swapchain),
            images,
            mem_alloc,
            desc_alloc,
            comm_alloc,
            query_pool: Mutex::new(HashMap::<
                i32,
                Arc<QueryPool>,
                BuildHasherDefault<NoHashHasher<i32>>,
            >::default()),
            query_counter: AtomicI32::new(0),
        })
    }
    pub fn new_query(&self) -> i32 {
        let id = self.query_counter.fetch_add(1, Ordering::Relaxed);
        let query_pool = QueryPool::new(
            self.device.clone(),
            QueryPoolCreateInfo {
                query_count: 1,
                ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
            },
        )
        .unwrap();
        self.query_pool.lock().insert(id, query_pool);
        id
    }
    pub fn query(
        &self,
        id: &i32,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
    ) {
        let b = self.query_pool.lock();
        let a = b.get(id).unwrap();
        unsafe {
            builder
                .reset_query_pool(a.clone(), 0..1)
                .unwrap()
                .begin_query(
                    a.clone(),
                    0,
                    QueryControlFlags {
                        precise: false,
                        ..QueryControlFlags::empty()
                    },
                )
                .unwrap();
        }
        todo!();
    }
    pub fn end_query(
        &self,
        id: &i32,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
    ) {
        let b = self.query_pool.lock();
        let a = b.get(id).unwrap();
        builder.end_query(a.clone(), 0).unwrap();
        todo!();
    }
    pub fn get_query(&self, id: &i32) -> u64 {
        let mut query_results = [0u64];
        let b = self.query_pool.lock();
        let query_pool = b.get(id).unwrap();
        if let Some(res) = query_pool.queries_range(0..1) {
            res.get_results(
                &mut query_results,
                QueryResultFlags {
                    // Block the function call until the results are available.
                    // Note: if not all the queries have actually been executed, then this
                    // will wait forever for something that never happens!
                    wait: true,

                    // Blocking and waiting will never give partial results.
                    partial: false,

                    // Blocking and waiting will ensure the results are always available after
                    // the function returns.
                    //
                    // If you disable waiting, then this can be used to include the
                    // availability of each query's results. You need one extra element per
                    // query in your `query_results` buffer for this. This element will
                    // be filled with a zero/nonzero value indicating availability.
                    with_availability: false,
                    ..QueryResultFlags::empty()
                },
            )
            .unwrap();
        }
        
        todo!();
        query_results[0]
    }
}
