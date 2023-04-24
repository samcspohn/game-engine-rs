use std::sync::Arc;

use vulkano::{swapchain::Swapchain, image::SwapchainImage};

struct RenderPipeline {
    swapchain:Arc<Swapchain>, 
    images: Vec<Arc<SwapchainImage>>,
}