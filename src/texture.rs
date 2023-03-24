use image::GenericImage;
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::Arc,
};

use parking_lot::RwLock;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
    memory::allocator::StandardMemoryAllocator,
    sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo, LOD_CLAMP_NONE},
};

use crate::{asset_manager::{self, Asset}, inspectable::Inspectable_};

// pub struct TextureManager {
//     pub device: Arc<Device>,
//     pub queue: Arc<Queue>,
//     pub textures: RwLock<HashMap<String, Arc<Texture>>>,
//     pub mem: Arc<StandardMemoryAllocator>,
// }

// impl TextureManager {
//     pub fn regen(&self, textures: BTreeSet<String>) {
//         for t in textures {
//             self.texture(&t);
//         }
//     }
//     pub fn texture(&self, path: &str) -> Arc<Texture> {
//         {
//             if let Some(tex) = self.textures.read().get(path.into()) {
//                 return tex.clone();
//             }
//         }
//         {
//             let tex = Arc::new(Texture::from_file(
//                 path,
//                 self.device.clone(),
//                 self.queue.clone(),
//                 &self.mem,
//             ));
//             self.textures.write().insert(path.into(), tex.clone());
//             tex
//         }
//     }
// }

pub type TextureManager = asset_manager::AssetManager<(Arc<Device>,Arc<Queue>,Arc<StandardMemoryAllocator>),Texture>;

pub struct Texture {
    pub file: String,
    pub image: Arc<ImageView<ImmutableImage>>,
    pub sampler: Arc<Sampler>,
}
impl Inspectable_ for Texture {
    fn inspect(&mut self, ui: &mut egui::Ui, world: &parking_lot::Mutex<crate::engine::World>) {
        
    }
}
impl Asset<Texture, (Arc<Device>,Arc<Queue>,Arc<StandardMemoryAllocator>)> for Texture {
    fn from_file(
        path: &str,
        params: &(Arc<Device>,Arc<Queue>,Arc<StandardMemoryAllocator>)
    ) -> Texture {

        let (device,
        queue,
        mem) = params;
        let image = {
            match image::open(path) {
                Ok(img) => {
                    let dimensions = ImageDimensions::Dim2d {
                        width: img.width(),
                        height: img.height(),
                        array_layers: 1,
                    };
                    let img_format = match img.color() {
                        // image::ColorType::L8(_) => Format::R8_SNORM,
                        image::ColorType::Rgb8 => Format::R8G8B8_SRGB,
                        // image::ColorType::Palette(_) => Format::R8_SINT,
                        // image::ColorType::GrayA(_) => Format::R8G8_SNORM,
                        image::ColorType::Rgba8 => Format::R8G8B8A8_SRGB,
                        _ => Format::R8_SNORM,
                    };

                    let pixels: Vec<u8> = if img_format == Format::R8G8B8_SRGB {
                        img.as_bytes()
                            .chunks(3)
                            .flat_map(|p| [p[0], p[1], p[2], 1u8])
                            .collect()
                    } else {
                        img.as_bytes().into_iter().map(|u| *u).collect()
                    };
                    let command_buffer_allocator =
                        StandardCommandBufferAllocator::new(device.clone(), Default::default());
                    let mut uploads = AutoCommandBufferBuilder::primary(
                        &command_buffer_allocator,
                        queue.queue_family_index(),
                        CommandBufferUsage::OneTimeSubmit,
                    )
                    .unwrap();
                    let image = ImmutableImage::from_iter(
                        mem,
                        pixels,
                        dimensions,
                        MipmapsCount::Log2,
                        Format::R8G8B8A8_SRGB,
                        &mut uploads,
                    )
                    .unwrap();

                    let _ = uploads.build().unwrap().execute(queue.clone()).unwrap();

                    ImageView::new_default(image).unwrap()
                }
                Err(_) => panic!("file not found{}", path),
            }
        };

        let sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                lod: 0.0..=LOD_CLAMP_NONE,
                mip_lod_bias: -0.2,
                // mag_filter: Filter::Linear,
                // min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                // mip_lod_bias: 1.0,
                // anisotropy: Some(())
                ..Default::default()
            },
        )
        .unwrap();

        Texture { file: path.into(), image, sampler }
    }

    fn reload(&mut self, file: &str, params: &(Arc<Device>,Arc<Queue>,Arc<StandardMemoryAllocator>)) {
        *self = Self::from_file(file, params)
    }
}
