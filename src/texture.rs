use image::GenericImage;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use parking_lot::RwLock;
use vulkano::{
    device::{Device, Queue},
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
    sampler::{Sampler, SamplerAddressMode, SamplerCreateInfo, LOD_CLAMP_NONE},
};

pub struct TextureManager {
    pub device: Arc<Device>,
    pub queue: Arc<Queue>,
    pub textures: RwLock<HashMap<String, Arc<Texture>>>,
}

impl TextureManager {
    pub fn regen(&self, textures: HashSet<String>) {
        for t in textures {
            self.texture(&t);
        }
    }
    pub fn texture(&self, path: &str) -> Arc<Texture> {
        {
            if let Some(tex) = self.textures.read().get(path.into()) {
                return tex.clone();
            }
        }
        {
            let tex = Arc::new(Texture::from_file(
                path,
                self.device.clone(),
                self.queue.clone(),
            ));
            self.textures.write().insert(path.into(), tex.clone());
            tex
        }
    }
}

pub struct Texture {
    pub image: Arc<ImageView<ImmutableImage>>,
    pub sampler: Arc<Sampler>,
}

impl Texture {
    pub fn from_file(path: &str, device: Arc<Device>, queue: Arc<Queue>) -> Texture {
        let image = {
            match image::open(path) {
                Ok(img) => {
                    let dimensions = ImageDimensions::Dim2d {
                        width: img.width(),
                        height: img.height(),
                        array_layers: 1,
                    };
                    let img_format = match img.color() {
                        image::ColorType::Gray(_) => Format::R8_SNORM,
                        image::ColorType::RGB(_) => Format::R8G8B8_SRGB,
                        image::ColorType::Palette(_) => Format::R8_SINT,
                        image::ColorType::GrayA(_) => Format::R8G8_SNORM,
                        image::ColorType::RGBA(_) => Format::R8G8B8A8_SRGB,
                    };

                    let pixels = if img_format == Format::R8G8B8_SRGB {
                        img.raw_pixels()
                            .chunks(3)
                            .flat_map(|p| [p[0], p[1], p[2], 1u8])
                            .collect()
                    } else {
                        img.raw_pixels()
                    };
                    let image = ImmutableImage::from_iter(
                        pixels,
                        dimensions,
                        MipmapsCount::Log2,
                        Format::R8G8B8A8_SRGB,
                        queue.clone(),
                    )
                    .unwrap()
                    .0;

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

        Texture { image, sampler }
    }
}
