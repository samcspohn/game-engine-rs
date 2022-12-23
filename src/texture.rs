use std::{
    collections::HashMap,
    fs::File,
    io::{Cursor, Read},
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
            match File::open(path) {
                Ok(mut f) => {
                    let mut png_bytes = Vec::new();
                    let _ = f.read_to_end(&mut png_bytes);
                    // let png_bytes = include_bytes!("rust_mascot.png").to_vec();
                    let cursor = Cursor::new(png_bytes);
                    let decoder = png::Decoder::new(cursor);
                    let mut reader = decoder.read_info().unwrap().1;
                    let info = reader.info();
                    let dimensions = ImageDimensions::Dim2d {
                        width: info.width,
                        height: info.height,
                        array_layers: 1,
                    };
                    let mut image_data = Vec::new();
                    image_data.resize((info.width * info.height * 4) as usize, 0);
                    reader.next_frame(&mut image_data).unwrap();

                    let image = ImmutableImage::from_iter(
                        image_data,
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
