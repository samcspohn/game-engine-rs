use std::{
    cell::SyncUnsafeCell,
    collections::HashMap,
    hash::BuildHasherDefault,
    sync::{
        atomic::{AtomicI32, Ordering},
        Arc,
    },
};

use force_send_sync::SendSync;
use nohash_hasher::NoHashHasher;
use parking_lot::Mutex;
use thincollections::thin_map::ThinMap;
use vulkano::{
    buffer::{
        allocator::{SubbufferAllocator, SubbufferAllocatorCreateInfo},
        Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer,
    },
    command_buffer::{
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
        AutoCommandBufferBuilder, PrimaryAutoCommandBuffer,
    },
    descriptor_set::allocator::StandardDescriptorSetAllocator,
    device::{
        physical::PhysicalDeviceType, Device, DeviceCreateInfo, DeviceExtensions, DeviceOwned,
        Features, Queue, QueueCreateInfo, QueueFlags,
    },
    format::Format,
    image::{Image, ImageUsage},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    library::DynamicLibraryLoader,
    memory::allocator::{
        AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter, StandardMemoryAllocator,
    },
    query::{QueryControlFlags, QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType},
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
    sync::{PipelineStage, Sharing},
    NonZeroDeviceSize, Version, VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::{LogicalPosition, LogicalSize, PhysicalSize},
    event_loop::EventLoop,
    window::{self, CursorGrabMode, Window, WindowBuilder},
};

use crate::engine::{utils, EngineEvent};

use super::component::buffer_usage_all;

const NUM_SUBALLOC: usize = 2;
#[repr(C)]
pub struct VulkanManager {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub surface: Arc<Surface>,
    pub swapchain: SyncUnsafeCell<Arc<Swapchain>>,
    pub images: Vec<Arc<Image>>,
    pub instance: Arc<Instance>,
    // pub event_loop: EventLoop<()>,
    pub mem_alloc: Arc<StandardMemoryAllocator>,
    pub desc_alloc: Arc<StandardDescriptorSetAllocator>,
    pub comm_alloc: Arc<StandardCommandBufferAllocator>,
    pub query_pool: Mutex<HashMap<i32, Arc<QueryPool>, nohash_hasher::BuildNoHashHasher<i32>>>,
    sub_alloc: Vec<Mutex<SubbufferAllocator>>,
    sub_alloc_unsized: Vec<Mutex<SubbufferAllocator>>,
    c: SyncUnsafeCell<usize>,
    query_counter: AtomicI32,
    pub(crate) show_cursor: Mutex<bool>,
    pub(crate) cursor_pos: Mutex<Option<LogicalPosition<f32>>>,
    pub(crate) grab_mode: Mutex<CursorGrabMode>,
    // a: ThinMap<std::ptr::>
}

impl VulkanManager {
    pub(crate) fn finalize(&self) {
        unsafe {
            let c = self.c.get();
            *c = (*c + 1) % NUM_SUBALLOC;
        }
    }
    pub fn allocate<T>(&self, d: T) -> Subbuffer<T>
    where
        T: BufferContents + Sized,
    {
        let ub = self.sub_alloc[unsafe { *self.c.get() }]
            .lock()
            .allocate_sized()
            .unwrap();
        *ub.write().unwrap() = d;
        ub
    }
    pub fn allocate_unsized<T>(&self, len: u64) -> Subbuffer<T>
    where
        T: BufferContents + ?Sized,
    {
        self.sub_alloc_unsized[unsafe { *self.c.get() }]
            .lock()
            .allocate_unsized(len)
            .unwrap()
        // let ub = self.sub_alloc[*self.c.lock() as usize]
        //     .allocate_sized()
        //     .unwrap();
        // *ub.write().unwrap() = d;
        // ub
    }
    pub fn buffer<T>(&self, memory_type_filter: MemoryTypeFilter) -> Subbuffer<T>
    where
        T: BufferContents + Sized,
    {
        let buf = Buffer::new_sized(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: buffer_usage_all(),
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter,
                ..Default::default()
            },
        )
        .unwrap();
        buf
    }
    pub fn buffer_array<T>(&self, size: u64, memory_type_filter: MemoryTypeFilter) -> Subbuffer<T>
    where
        T: BufferContents + ?Sized,
    {
        let buf = Buffer::new_unsized(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: buffer_usage_all(),
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter,
                ..Default::default()
            },
            size,
        )
        .unwrap();
        buf
    }
    pub fn buffer_from_data<T>(&self, d: T) -> Subbuffer<T>
    where
        T: BufferContents + Sized,
    {
        let buf = Buffer::from_data(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: buffer_usage_all(),
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            d,
        )
        .unwrap();
        buf
    }
    pub fn buffer_from_iter<T, I>(&self, iter: I) -> Subbuffer<[T]>
    where
        T: BufferContents,
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let buf = Buffer::from_iter(
            self.mem_alloc.clone(),
            BufferCreateInfo {
                usage: buffer_usage_all(),
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            iter,
        )
        .unwrap();
        buf
    }
    pub fn sub_buffer_allocator(&self) -> SubbufferAllocator {
        let sub_alloc = SubbufferAllocator::new(
            self.mem_alloc.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                ..Default::default()
            },
        );
        sub_alloc
    }
    pub fn sub_buffer_allocator_with_usage(&self, buffer_usage: BufferUsage) -> SubbufferAllocator {
        let sub_alloc = SubbufferAllocator::new(
            self.mem_alloc.clone(),
            SubbufferAllocatorCreateInfo {
                buffer_usage,
                ..Default::default()
            },
        );
        sub_alloc
    }
    pub fn set_cursor_pos(&self, pos: LogicalPosition<f32>) {
        *self.cursor_pos.lock() = Some(pos);
    }
    pub fn show_cursor(&self, show: bool) {
        *self.show_cursor.lock() = show;
    }
    pub fn grab_mode(&self, mode: CursorGrabMode) {
        *self.grab_mode.lock() = mode;
    }
    pub(crate) fn window(&self) -> &Window {
        unsafe {
            self.surface
                .object()
                .unwrap()
                .downcast_ref_unchecked::<Window>()
        }
    }
    pub fn swapchain(&self) -> Arc<Swapchain> {
        unsafe { &*self.swapchain.get() }.clone()
    }
    pub fn update_swapchain(&self, swapchain: Arc<Swapchain>) {
        unsafe {
            *self.swapchain.get() = swapchain;
        }
    }
    pub(crate) fn new(event_loop: &EventLoop<EngineEvent>) -> Arc<Self> {
        // rayon::ThreadPoolBuilder::new().num_threads(63).build_global().unwrap();
        let library = VulkanLibrary::new().unwrap();
        println!("{:?}", library.api_version());
        // let library = VulkanLibrary::with_loader()
        let required_extensions = vulkano_win::required_extensions(&library);
        // required_extensions.ext_headless_surface = true;
        // Now creating the instance.
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                // Enable enumerating devices that use non-conformant vulkan implementations. (ex. MoltenVK)
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                ..Default::default()
            },
        )
        .unwrap();

        // let event_loop = EventLoop::new();
        // let surface = WindowBuilder::new()
        //     // .with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
        //     .with_inner_size(PhysicalSize {
        //         width: 1920,
        //         height: 1080,
        //     })
        //     .build_vk_surface(event_loop, instance.clone())
        //     .unwrap();
        let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());
        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        let mut device_extensions = DeviceExtensions {
            khr_swapchain: true,
            // nv_geometry_shader_passthrough: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .filter(|p| {
                // For this example, we require at least Vulkan 1.3, or a device that has the
                p.api_version() >= Version::V1_3 || p.supported_extensions().khr_dynamic_rendering
            })
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
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

        if physical_device.api_version() < Version::V1_3 {
            device_extensions.khr_dynamic_rendering = true;
        }

        // Now initializing the device. This is probably the most important object of Vulkan.
        //
        // The iterator of created queues is returned by the function alongside the device.

        let features = Features {
            geometry_shader: true,
            runtime_descriptor_array: true,
            descriptor_binding_variable_descriptor_count: true,
            dynamic_rendering: true,
            ..Features::empty()
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
        std::fs::write(
            "device-properties.ron",
            format!("{:#?}", device.physical_device().properties()),
        );
        // println!("{:#?}", device.physical_device().properties());

        // Since we can request multiple queues, the `queues` variable is in fact an iterator. We
        // only use one queue in this example, so we just retrieve the first and only element of the
        // iterator.
        let queue = queues.next().unwrap();

        // let img_count = swapchain.image_count();
        let mem_alloc = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
        let desc_alloc = Arc::new(StandardDescriptorSetAllocator::new(
            device.clone(),
            Default::default(),
        ));
        let comm_alloc = Arc::new(StandardCommandBufferAllocator::new(
            device.clone(),
            StandardCommandBufferAllocatorCreateInfo {
                secondary_buffer_count: 32,
                ..Default::default()
            },
        ));
        let (swapchain, images) = {
            let surface_capabilities = device
                .physical_device()
                .surface_capabilities(&surface, Default::default())
                .unwrap();
            let image_format = Some(
                device
                    .physical_device()
                    .surface_formats(&surface, Default::default())
                    .unwrap()[0]
                    .0,
            );
            let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
            // let min_image_count = surface_capabilities
            //     .max_image_count
            //     .unwrap_or(3)
            //     .min(3)
            //     .max(surface_capabilities.min_image_count);
            let min_image_count = surface_capabilities.min_image_count;
            // let min_image_count = surface_capabilities.min_image_count;
            let mut swapchain_create_info = SwapchainCreateInfo {
                min_image_count,
                image_format: image_format.unwrap(),
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT | ImageUsage::TRANSFER_DST,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                present_mode: vulkano::swapchain::PresentMode::Mailbox,
                ..Default::default()
            };
            match Swapchain::new(
                device.clone(),
                surface.clone(),
                swapchain_create_info.clone(),
            ) {
                Ok(sc) => sc,
                Err(_) => {
                    swapchain_create_info.present_mode = vulkano::swapchain::PresentMode::Immediate;
                    Swapchain::new(device.clone(), surface.clone(), swapchain_create_info).unwrap()
                }
            }
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
            swapchain: SyncUnsafeCell::new(swapchain),
            images,
            mem_alloc: mem_alloc.clone(),
            desc_alloc,
            comm_alloc,
            query_pool: Mutex::new(HashMap::default()),
            query_counter: AtomicI32::new(0),
            sub_alloc: unsafe {
                (0..NUM_SUBALLOC)
                    .map(|_| {
                        Mutex::new(SubbufferAllocator::new(
                            mem_alloc.clone(),
                            SubbufferAllocatorCreateInfo {
                                buffer_usage: BufferUsage::UNIFORM_BUFFER,
                                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                                ..Default::default()
                            },
                        ))
                    })
                    .collect()
            },
            sub_alloc_unsized: unsafe {
                (0..NUM_SUBALLOC)
                    .map(|_| {
                        Mutex::new(SubbufferAllocator::new(
                            mem_alloc.clone(),
                            SubbufferAllocatorCreateInfo {
                                buffer_usage: BufferUsage::STORAGE_BUFFER,
                                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                                    | MemoryTypeFilter::HOST_RANDOM_ACCESS,
                                ..Default::default()
                            },
                        ))
                    })
                    .collect()
            },
            c: unsafe { SyncUnsafeCell::new(0) },
            show_cursor: Mutex::new(false),
            cursor_pos: Mutex::new(None),
            grab_mode: Mutex::new(CursorGrabMode::None),
        })
    }
    pub fn new_query(&self) -> i32 {
        let id = self.query_counter.fetch_add(1, Ordering::Relaxed);
        let query_pool = QueryPool::new(
            self.device.clone(),
            QueryPoolCreateInfo {
                query_count: 2,
                ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
            },
        )
        .unwrap();
        self.query_pool.lock().insert(id, query_pool);
        id
    }
    pub fn begin_query(&self, id: &i32, builder: &mut utils::PrimaryCommandBuffer) {
        let b = self.query_pool.lock();
        let a = b.get(id).unwrap();
        unsafe {
            builder
                .reset_query_pool(a.clone(), 0..2)
                .unwrap()
                .write_timestamp(a.clone(), 0, PipelineStage::TopOfPipe)
                // .begin_query(a.clone(), 0, QueryControlFlags::PRECISE)
                .unwrap();
        }
        // todo!();
    }
    pub fn end_query(&self, id: &i32, builder: &mut utils::PrimaryCommandBuffer) {
        let b = self.query_pool.lock();
        let a = b.get(id).unwrap();
        // builder.end_query(a.clone(), 0).unwrap();
        unsafe {
            builder
                .write_timestamp(a.clone(), 1, PipelineStage::BottomOfPipe)
                .unwrap();
        }
        // todo!();
    }
    pub fn get_query(&self, id: &i32) -> u64 {
        let mut query_results = [0u64; 2];
        let b = self.query_pool.lock();
        let query_pool = b.get(id).unwrap();
        loop {
            if let Ok(res) =
                query_pool.get_results(0..2, &mut query_results, QueryResultFlags::WAIT)
            {
                if res {
                    break;
                }
                // res.get_results(&mut query_results, QueryResultFlags::WAIT)
                //     .unwrap();
            }
        }

        // todo!();
        query_results[1] - query_results[0]
    }
}
