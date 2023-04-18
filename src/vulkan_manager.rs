use std::sync::Arc;

use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo,
    },
    instance::{Instance, InstanceCreateInfo},
    memory::allocator::StandardMemoryAllocator,
    swapchain::Surface,
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{dpi::LogicalSize, event_loop::EventLoop, window::WindowBuilder};

pub struct VulkanManager {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub surface: Arc<Surface>,
    // pub event_loop: EventLoop<()>,
    pub mem_alloc: Arc<StandardMemoryAllocator>,
    pub desc_alloc: Arc<StandardDescriptorSetAllocator>,
    pub comm_alloc: Arc<StandardCommandBufferAllocator>,
}

impl VulkanManager {
    pub fn new(event_loop: &EventLoop<()>) -> Arc<Self> {
        // rayon::ThreadPoolBuilder::new().num_threads(63).build_global().unwrap();
        let library = VulkanLibrary::new().unwrap();
        let required_extensions = vulkano_win::required_extensions(&library);

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
            .build_vk_surface(&event_loop, instance.clone())
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
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
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
        let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let descriptor_set_allocator =
            Arc::new(StandardDescriptorSetAllocator::new(device.clone()));
        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            Default::default(),
        ));

        Arc::new(Self {
            device,
            queue,
            surface,
            mem_alloc: memory_allocator,
            desc_alloc: descriptor_set_allocator,
            comm_alloc: command_buffer_allocator,
        })
    }
}
