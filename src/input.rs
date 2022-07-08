use std::collections::HashMap;

use winit::event::VirtualKeyCode;



#[derive(Default, Clone)]
pub struct Input {
    pub key_downs: HashMap<VirtualKeyCode, bool>,
    pub key_presses: HashMap<VirtualKeyCode, bool>,
    pub key_ups: HashMap<VirtualKeyCode, bool>,
    pub mouse_x: f64,
    pub mouse_y: f64,
}

impl Input {

    pub fn get_key(&self, key: &VirtualKeyCode) -> bool {
        *self.key_downs.get(key).unwrap_or(&false)
    }
    pub fn get_key_press(&self, key: &VirtualKeyCode) -> bool {
        *self.key_presses.get(key).unwrap_or(&false)
    }
    pub fn get_key_up(&self, key: &VirtualKeyCode) -> bool {
        *self.key_ups.get(key).unwrap_or(&false)
    }
    pub fn get_mouse_delta(&self,) -> (f64,f64) {
        (self.mouse_x,self.mouse_y)
    }
}