use std::sync::Arc;

use glm::{vec4, Vec3};
use nalgebra_glm::{self as glm, vec3};
use winit::event::VirtualKeyCode;

use crate::engine::{
    input::Input, prelude::VulkanManager, rendering::camera::CameraDataId, time::Time,
};

pub struct EditorCam {
    pub pos: glm::Vec3,
    pub rot: glm::Quat,
    pub speed: f32,
    pub camera: CameraDataId,
}

impl EditorCam {
    pub fn new(vk: Arc<VulkanManager>) -> EditorCam {
        EditorCam {
            pos: Vec3::zeros(),
            rot: glm::quat_look_at_lh(&Vec3::z(), &Vec3::y()),
            speed: 32f32,
            camera: CameraDataId::new(vk, 1),
        }
    }
    pub fn update(&mut self, input: &Input, time: &Time) {
        let speed = self.speed * time.dt;
        if !input.get_key(&VirtualKeyCode::LControl) && input.get_mouse_button(&2) {
            if input.get_key_down(&VirtualKeyCode::R) {
                self.speed *= 1.5;
                // cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * -speed;
            }
            if input.get_key_down(&VirtualKeyCode::F) {
                self.speed /= 1.5;
                // cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * speed;
            }

            // forward/backward
            if input.get_key(&VirtualKeyCode::W) {
                self.pos += glm::quat_rotate_vec3(&self.rot, &Vec3::z()) * speed;
            }
            if input.get_key(&VirtualKeyCode::S) {
                self.pos += glm::quat_rotate_vec3(&self.rot, &Vec3::z()) * -speed;
            }
            //left/right
            if input.get_key(&VirtualKeyCode::A) {
                self.pos += glm::quat_rotate_vec3(&self.rot, &Vec3::x()) * -speed;
            }
            if input.get_key(&VirtualKeyCode::D) {
                self.pos += glm::quat_rotate_vec3(&self.rot, &Vec3::x()) * speed;
            }
            // up/down
            if input.get_key(&VirtualKeyCode::E) {
                self.pos += glm::quat_rotate_vec3(&self.rot, &Vec3::y()) * speed;
            }
            if input.get_key(&VirtualKeyCode::Q) {
                self.pos += glm::quat_rotate_vec3(&self.rot, &Vec3::y()) * -speed;
            }

            self.rot = glm::quat_rotate(
                &self.rot,
                input.get_mouse_delta().0 as f32 * 0.01,
                &glm::quat_inv_cross_vec(&Vec3::y(), &self.rot),
            );
            self.rot = glm::quat_rotate(
                &self.rot,
                input.get_mouse_delta().1 as f32 * 0.01,
                &Vec3::x(),
            );
        }
    }
}
