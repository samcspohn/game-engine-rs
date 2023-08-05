use std::{sync::Arc, path::Path};

use crossbeam::queue::SegQueue;
use force_send_sync::SendSync;
use substring::Substring;
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, PrimaryAutoCommandBuffer,
    SecondaryAutoCommandBuffer,
};

pub type SecondaryCommandBuffer =
    AutoCommandBufferBuilder<SecondaryAutoCommandBuffer, Arc<StandardCommandBufferAllocator>>;
pub type PrimaryCommandBuffer =
    AutoCommandBufferBuilder<PrimaryAutoCommandBuffer, Arc<StandardCommandBufferAllocator>>;
pub type GPUWork =
    SegQueue<SendSync<Box<dyn FnOnce(&mut PrimaryCommandBuffer, Arc<VulkanManager>)>>>;

pub use config;

use config::{Config, File};
use lazy_static::lazy_static;
use parking_lot::RwLock;

use super::rendering::vulkan_manager::VulkanManager;

lazy_static! {
    pub static ref SETTINGS: RwLock<Config> = {
        RwLock::new(
            Config::builder()
                .add_source(File::with_name("config.toml"))
                .build()
                .unwrap(),
        )
    };
}

pub const SEP: char = std::path::MAIN_SEPARATOR;

pub fn path_format(entry: &std::path::PathBuf) -> String {
    let f_name = String::from(entry.to_string_lossy());
    let f_name = f_name.replace(SEP, "/");
    let f_name = f_name.substring(2, f_name.len()).to_owned();
    f_name
}