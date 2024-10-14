use std::sync::Arc;

use egui::TextureId;
use id::{ID_trait, ID};
use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, sys::CommandBufferBeginInfo,
        AutoCommandBufferBuilder, CommandBufferLevel, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryCommandBufferAbstract, RenderPassBeginInfo,
    },
    device::{Device, Queue},
    format::Format,
    image::{
        sampler::{Sampler, SamplerCreateInfo},
        view::ImageView,
        Image, ImageCreateInfo, ImageType, ImageUsage,
    },
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    DeviceSize,
};

use crate::engine::rendering::vulkan_manager::VulkanManager;
use crate::{
    editor::inspectable::Inspectable_,
    engine::{
        project::asset_manager::{Asset, AssetManager},
        world::World,
    },
};

pub type TextureManager = AssetManager<Arc<VulkanManager>, Texture>;

#[derive(ID)]
pub struct Texture {
    pub file: String,
    pub image: Arc<ImageView>,
    pub sampler: Arc<Sampler>,
    // ui_id: Option<TextureId>,
}
impl Inspectable_ for Texture {
    fn inspect(&mut self, _ui: &mut egui::Ui, _world: &mut World) -> bool {
        false
        // if let Some(id) = &self.ui_id {
        //     _ui.image(*id, egui::vec2(200., 200.));
        //     true
        // } else {
        //     false
        // }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub fn texture_from_bytes(
    vk: Arc<VulkanManager>,
    data: &[u8],
    width: u32,
    height: u32,
) -> (Arc<ImageView>, Arc<Sampler>) {
    let mut uploads = AutoCommandBufferBuilder::primary(
        &vk.comm_alloc,
        vk.queue.queue_family_index(),
        CommandBufferUsage::OneTimeSubmit,
    )
    .unwrap();

    let format = Format::R8G8B8A8_SRGB;
    let extent: [u32; 3] = [width, height, 1];
    let array_layers = 1u32;

    let buffer_size = format.block_size()
        * extent
            .into_iter()
            .map(|e| e as DeviceSize)
            .product::<DeviceSize>()
        * array_layers as DeviceSize;
    let upload_buffer = Buffer::new_slice(
        vk.mem_alloc.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        buffer_size,
    )
    .unwrap();

    {
        let mut image_data = &mut *upload_buffer.write().unwrap();

        image_data.copy_from_slice(data);
    }

    let image = Image::new(
        vk.mem_alloc.clone(),
        ImageCreateInfo {
            image_type: ImageType::Dim2d,
            format,
            extent,
            array_layers,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            ..Default::default()
        },
        AllocationCreateInfo::default(),
    )
    .unwrap();

    uploads
        .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            upload_buffer,
            image.clone(),
        ))
        .unwrap();

    let _ = uploads.build().unwrap().execute(vk.queue.clone()).unwrap();

    let sampler =
        Sampler::new(vk.device.clone(), SamplerCreateInfo::simple_repeat_linear()).unwrap();

    (ImageView::new_default(image).unwrap(), sampler)
}
impl Asset<Texture, (Arc<VulkanManager>)> for Texture {
    fn from_file(path: &str, params: &Arc<VulkanManager>) -> Texture {
        if let Ok(img) = image::open(path) {
            // let img_format = match img.color() {
            //     // image::ColorType::L8(_) => Format::R8_SNORM,
            //     image::ColorType::Rgb8 => Format::R8G8B8_SRGB,
            //     // image::ColorType::Palette(_) => Format::R8_SINT,
            //     // image::ColorType::GrayA(_) => Format::R8G8_SNORM,
            //     image::ColorType::Rgba8 => Format::R8G8B8A8_SRGB,
            //     _ => Format::R8_SNORM,
            // };

            // let pixels: Vec<u8> = if img_format == Format::R8G8B8_SRGB {
            //     img.as_bytes()
            //         .chunks(3)
            //         .flat_map(|p| [p[0], p[1], p[2], 1u8])
            //         .collect()
            // } else {
            //     img.as_bytes().iter().map(|u| *u).collect()
            // };
            let pixels: Vec<u8> = img.to_rgba8().iter().cloned().collect();

            // let vk = params;
            // let mut uploads = AutoCommandBufferBuilder::new(
            //     vk.comm_alloc.clone(),
            //     vk.queue.queue_family_index(),
            //     CommandBufferLevel::Primary,
            //     CommandBufferBeginInfo {
            //         usage: CommandBufferUsage::OneTimeSubmit,
            //         ..Default::default()
            //     },
            // )
            // .unwrap();

            // let format = Format::R8G8B8A8_SRGB;
            // let extent: [u32; 3] = [img.width(), img.height(), 1];
            // let array_layers = 3u32;

            // let buffer_size = format.block_size()
            //     * extent
            //         .into_iter()
            //         .map(|e| e as DeviceSize)
            //         .product::<DeviceSize>()
            //     * array_layers as DeviceSize;
            // let upload_buffer = Buffer::new_slice(
            //     vk.mem_alloc.clone(),
            //     BufferCreateInfo {
            //         usage: BufferUsage::TRANSFER_SRC,
            //         ..Default::default()
            //     },
            //     AllocationCreateInfo {
            //         memory_type_filter: MemoryTypeFilter::PREFER_HOST
            //             | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            //         ..Default::default()
            //     },
            //     buffer_size,
            // )
            // .unwrap();

            // {
            //     let mut image_data = &mut *upload_buffer.write().unwrap();

            //     //  let dimensions = ImageDimensions::Dim2d {
            //     //     //     width: img.width(),
            //     //     //     height: img.height(),
            //     //     //     array_layers: 1,
            //     //     // };

            //     image_data.copy_from_slice(&pixels);
            //     // for png_bytes in [
            //     //     include_bytes!("square.png").as_slice(),
            //     //     include_bytes!("star.png").as_slice(),
            //     //     include_bytes!("asterisk.png").as_slice(),
            //     // ] {
            //     //     let decoder = png::Decoder::new(png_bytes);
            //     //     let mut reader = decoder.read_info().unwrap();
            //     //     reader.next_frame(image_data).unwrap();
            //     //     let info = reader.info();
            //     //     image_data = &mut image_data[(info.width * info.height * 4) as usize..];
            //     // }
            // }

            // let image = Image::new(
            //     vk.mem_alloc.clone(),
            //     ImageCreateInfo {
            //         image_type: ImageType::Dim2d,
            //         format,
            //         extent,
            //         array_layers,
            //         usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
            //         ..Default::default()
            //     },
            //     AllocationCreateInfo::default(),
            // )
            // .unwrap();

            // uploads
            //     .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
            //         upload_buffer,
            //         image.clone(),
            //     ))
            //     .unwrap();

            // let view = ImageView::new_default(image).unwrap();

            // let sampler =
            //     Sampler::new(vk.device.clone(), SamplerCreateInfo::simple_repeat_linear()).unwrap();

            // let _ = uploads.end().unwrap().execute(vk.queue.clone()).unwrap();
            let (image, sampler) =
                texture_from_bytes(params.clone(), &pixels, img.width(), img.height());

            Texture {
                file: path.into(),
                image,
                sampler,
                // ui_id: None,
            }
        } else {
            panic!("file not found{}: ", path)
        }

        // let (device, queue, mem) = params;
        // let image = {
        //     match image::open(path) {
        //         Ok(img) => {
        //             // let dimensions = ImageDimensions::Dim2d {
        //             //     width: img.width(),
        //             //     height: img.height(),
        //             //     array_layers: 1,
        //             // };
        //             let img_format = match img.color() {
        //                 // image::ColorType::L8(_) => Format::R8_SNORM,
        //                 image::ColorType::Rgb8 => Format::R8G8B8_SRGB,
        //                 // image::ColorType::Palette(_) => Format::R8_SINT,
        //                 // image::ColorType::GrayA(_) => Format::R8G8_SNORM,
        //                 image::ColorType::Rgba8 => Format::R8G8B8A8_SRGB,
        //                 _ => Format::R8_SNORM,
        //             };

        //             let pixels: Vec<u8> = if img_format == Format::R8G8B8_SRGB {
        //                 img.as_bytes()
        //                     .chunks(3)
        //                     .flat_map(|p| [p[0], p[1], p[2], 1u8])
        //                     .collect()
        //             } else {
        //                 img.as_bytes().iter().map(|u| *u).collect()
        //             };
        //             let command_buffer_allocator =
        //                 StandardCommandBufferAllocator::new(device.clone(), Default::default());
        //             let mut uploads = AutoCommandBufferBuilder::primary(
        //                 &command_buffer_allocator,
        //                 queue.queue_family_index(),
        //                 CommandBufferUsage::OneTimeSubmit,
        //             )
        //             .unwrap();
        //             let image = ImmutableImage::from_iter(
        //                 mem,
        //                 pixels,
        //                 dimensions,
        //                 MipmapsCount::Log2,
        //                 Format::R8G8B8A8_SRGB,
        //                 &mut uploads,
        //             )
        //             .unwrap();

        //             let _ = uploads.build().unwrap().execute(queue.clone()).unwrap();

        //             ImageView::new_default(image).unwrap()
        //         }
        //         Err(_) => panic!("file not found{}: ", path),
        //     }
        // };

        // let sampler = Sampler::new(
        //     device.clone(),
        //     SamplerCreateInfo {
        //         lod: 0.0..=LOD_CLAMP_NONE,
        //         mip_lod_bias: -0.2,
        //         // mag_filter: Filter::Linear,
        //         // min_filter: Filter::Linear,
        //         address_mode: [SamplerAddressMode::Repeat; 3],
        //         // mip_lod_bias: 1.0,
        //         // anisotropy: Some(())
        //         ..Default::default()
        //     },
        // )
        // .unwrap();

        // Texture {
        //     file: path.into(),
        //     image,
        //     sampler,
        //     // ui_id: None,
        // }
    }

    fn reload(&mut self, file: &str, params: &(Arc<VulkanManager>)) {
        *self = Self::from_file(file, params)
    }
}
