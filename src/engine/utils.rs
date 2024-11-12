use std::{path::Path, sync::Arc};

use crossbeam::queue::SegQueue;
use force_send_sync::SendSync;
use nalgebra_glm::{Mat3, Quat, Vec3};
use substring::Substring;
use vulkano::command_buffer::{
    allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, PrimaryAutoCommandBuffer,
    SecondaryAutoCommandBuffer,
};

pub type SecondaryCommandBuffer =
    AutoCommandBufferBuilder<SecondaryAutoCommandBuffer, Arc<StandardCommandBufferAllocator>>;
pub type PrimaryCommandBuffer = AutoCommandBufferBuilder<
    PrimaryAutoCommandBuffer<Arc<StandardCommandBufferAllocator>>,
    Arc<StandardCommandBufferAllocator>,
>;
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

pub mod pipeline;
pub mod radix_sort;
pub mod perf;
pub mod gpu_perf;
// fn mat3_quat(m: &Mat3) -> Quat {
//     let m00 = m[0];
//     let m01 = m[1];
//     let m02 = m[2];
//     let m10 = m[3];
//     let m11 = m[4];
//     let m12 = m[5];
//     let m20 = m[6];
//     let m21 = m[7];
//     let m22 = m[8];

//     let tr = m00 + m11 + m22;
//     let mut q = Quat::new(0., 0., 0., 0.);
//     if (tr > 0.) {
//         let S = (tr + 1.0).sqrt() * 2.; // S=4*qw
//         q.w = 0.25 * S;
//         q.i = (m21 - m12) / S;
//         q.j = (m02 - m20) / S;
//         q.k = (m10 - m01) / S;
//     } else if ((m00 > m11) && (m00 > m22)) {
//         let S = (1.0 + m00 - m11 - m22).sqrt() * 2.; // S=4*qx
//         q.w = (m21 - m12) / S;
//         q.i = 0.25 * S;
//         q.j = (m01 + m10) / S;
//         q.k = (m02 + m20) / S;
//     } else if (m11 > m22) {
//         let S = (1.0 + m11 - m00 - m22).sqrt() * 2.; // S=4*qy
//         q.w = (m02 - m20) / S;
//         q.i = (m01 + m10) / S;
//         q.j = 0.25 * S;
//         q.k = (m12 + m21) / S;
//     } else {
//         let S = (1.0 + m22 - m00 - m11).sqrt() * 2.; // S=4*qz
//         q.w = (m10 - m01) / S;
//         q.i = (m02 + m20) / S;
//         q.j = (m12 + m21) / S;
//         q.k = 0.25 * S;
//     }
//     return q;
// }
use nalgebra_glm as glm;
// pub fn look_at_mat3(look_at: &Vec3, up: &Vec3) -> Mat3 {
//     let mut forward = look_at.clone();
//     forward = glm::normalize(&forward);
//     let right = glm::normalize(&glm::cross(&up, &forward));
//     let up = glm::normalize(&glm::cross(&forward, &right));

//     let mut m = Mat3::default();
//     m.set_column(0, &right);
//     m.set_column(1, &up);
//     m.set_column(2, &forward);
//     m = glm::transpose(&m);
//     // mat3 m = {{right.x,up.x,forward.x},{right.y,up.y,forward.y},{right.z,up.z,forward.z}};
//     // transpose(m);
//     m
// }
// pub fn look_at(look_at: &Vec3, up: &Vec3) -> Quat {
//     return mat3_quat(&look_at_mat3(look_at, up));
// }

pub fn euler_to_quat(euler_angles: &Vec3) -> Quat {
    let c = glm::cos(&(euler_angles * 0.5));
    let s = glm::sin(&(euler_angles * 0.5));
    let w = c.x * c.y * c.z + s.x * s.y * s.z;
    let i = s.x * c.y * c.z - c.x * s.y * s.z;
    let j = c.x * s.y * c.z + s.x * c.y * s.z;
    let k = c.x * c.y * s.z - s.x * s.y * c.z;
    Quat::new(w, i, j, k)
}
