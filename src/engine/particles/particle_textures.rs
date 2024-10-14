use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

use crate::engine::{rendering::{texture, vulkan_manager::VulkanManager}, utils};
use parking_lot::Mutex;
use vulkano::{
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        PrimaryCommandBufferAbstract,
    },
    device::Device,
    format::Format,
    image::{sampler::Sampler, view::ImageView, Image},
};

use crate::engine::{
    project::asset_manager::AssetInstance,
    rendering::texture::{Texture, TextureManager},
};

pub struct ParticleTextures {
    pub color_tex: (Arc<ImageView>, Arc<Sampler>),
    tex_man: Arc<Mutex<TextureManager>>,
    textures: HashMap<i32, u32>,
    pub samplers: Vec<(Arc<ImageView>, Arc<Sampler>)>,
    id: u32,
}
impl ParticleTextures {
    pub fn color_tex(
        colors: &[[[u8; 4]; 256]],
        vk: &Arc<VulkanManager>,
        builder: &mut utils::PrimaryCommandBuffer,
    ) -> (Arc<ImageView>, Arc<Sampler>) {
        // let image_view = {
        //     let dimensions = ImageDimensions::Dim2d {
        //         width: 256,
        //         height: colors.len() as u32,
        //         array_layers: 1,
        //     };
        //     let img_format = Format::R8G8B8A8_SRGB;

            let pixels: Vec<u8> = colors
                .iter() // for each array
                .flat_map(|p| p.iter().flat_map(|p| *p))
                .collect();
        //     // let command_buffer_allocator =
        //     //     StandardCommandBufferAllocator::new(vk.device.clone(), Default::default());
        //     // let mut uploads = AutoCommandBufferBuilder::primary(
        //     //     &command_buffer_allocator,
        //     //     vk.queue.queue_family_index(),
        //     //     CommandBufferUsage::OneTimeSubmit,
        //     // )
        //     // .unwrap();
        //     let image = ImmutableImage::from_iter(
        //         &vk.mem_alloc,
        //         pixels,
        //         dimensions,
        //         MipmapsCount::One,
        //         Format::R8G8B8A8_SRGB,
        //         builder,
        //     )
        //     .unwrap();

        //     // let _ = uploads.build().unwrap().execute(vk.queue.clone()).unwrap();

        //     ImageView::new_default(image).unwrap()
        // };

        // let sampler = Sampler::new(
        //     vk.device.clone(),
        //     SamplerCreateInfo {
        //         lod: 0.0..=LOD_CLAMP_NONE,
        //         mip_lod_bias: -0.2,
        //         // mag_filter: Filter::Nearest,
        //         // min_filter: Filter::Nearest,
        //         address_mode: [SamplerAddressMode::ClampToEdge; 3],
        //         // mip_lod_bias: 1.0,
        //         // anisotropy: Some(())
        //         ..Default::default()
        //     },
        // )
        // .unwrap();
        texture::texture_from_bytes(vk.clone(), &pixels, 256, colors.len() as u32)
        // (image_view, sampler)
    }
    pub fn new(
        tex_man: Arc<Mutex<TextureManager>>,
        color_tex: (Arc<ImageView>, Arc<Sampler>),
    ) -> Self {
        Self {
            color_tex,
            tex_man,
            textures: HashMap::new(),
            samplers: Vec::new(),
            id: 0,
        }
    }
    pub fn get_tex_id(&mut self, tex: &AssetInstance<Texture>) -> u32 {
        if let Some(id) = self.textures.get(&tex.id) {
            *id
        } else {
            let id = self.id;
            self.id += 1;
            self.textures.insert(tex.id, id);
            let tm = self.tex_man.lock();
            let a = tm.get_id(&tex.id).unwrap().lock();
            self.samplers.push((a.image.clone(), a.sampler.clone()));
            id
        }
    }
    // pub fn register(&mut self, tex: &AssetInstance<Texture>) {
    //     let id = self.id;
    //     self.id += 1;
    //     self.textures.insert(tex.id, id);
    //     let a = self.tex_man.lock().get_id(&tex.id).unwrap().lock();
    //     self.samplers.push((a.image.clone(),a.sampler.clone()));
    // }
}
