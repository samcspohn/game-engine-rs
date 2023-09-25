use glm::{vec4, Vec3};
use winit::event::VirtualKeyCode;

use nalgebra_glm as glm;

use crate::engine::{input::Input, time::Time};

pub struct EditorCam {
    pub pos: glm::Vec3,
    pub rot: glm::Quat,
    pub speed: f32,
}

impl EditorCam {
    pub fn update(&mut self, input: &Input, time: &Time) {
        let speed = self.speed * time.dt;
        if !input.get_key(&VirtualKeyCode::LControl) && input.get_mouse_button(&2) {
            if input.get_key_press(&VirtualKeyCode::R) {
                self.speed *= 1.5;
                // cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * -speed;
            }
            if input.get_key_press(&VirtualKeyCode::F) {
                self.speed /= 1.5;
                // cam_pos += (glm::quat_to_mat4(&cam_rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * speed;
            }

            // forward/backward
            if input.get_key(&VirtualKeyCode::W) {
                self.pos += (glm::quat_to_mat4(&self.rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * speed;
            }
            if input.get_key(&VirtualKeyCode::S) {
                self.pos +=
                    (glm::quat_to_mat4(&self.rot) * vec4(0.0, 0.0, 1.0, 1.0)).xyz() * -speed;
            }
            //left/right
            if input.get_key(&VirtualKeyCode::A) {
                self.pos += (glm::quat_to_mat4(&self.rot) * vec4(1.0, 0.0, 0.0, 1.0)).xyz() * speed;
            }
            if input.get_key(&VirtualKeyCode::D) {
                self.pos +=
                    (glm::quat_to_mat4(&self.rot) * vec4(1.0, 0.0, 0.0, 1.0)).xyz() * -speed;
            }
            // up/down
            if input.get_key(&VirtualKeyCode::Space) {
                self.pos +=
                    (glm::quat_to_mat4(&self.rot) * vec4(0.0, 1.0, 0.0, 1.0)).xyz() * -speed;
            }
            if input.get_key(&VirtualKeyCode::LShift) {
                self.pos += (glm::quat_to_mat4(&self.rot) * vec4(0.0, 1.0, 0.0, 1.0)).xyz() * speed;
            }

            // if input.get_mouse_button(&2) {
            self.rot = glm::quat_rotate(
                &self.rot,
                input.get_mouse_delta().0 as f32 * 0.01,
                &(glm::inverse(&glm::quat_to_mat3(&self.rot)) * Vec3::y()),
            );
            self.rot = glm::quat_rotate(
                &self.rot,
                input.get_mouse_delta().1 as f32 * -0.01,
                &Vec3::x(),
            );
            // }
        }
    }
}
