use std::sync::Arc;

use crossbeam::queue::SegQueue;
use force_send_sync::SendSync;
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
