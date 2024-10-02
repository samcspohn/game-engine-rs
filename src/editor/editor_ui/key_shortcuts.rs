use std::collections::HashMap;

use winit::event::VirtualKeyCode;

use crate::engine::input::Input;


pub struct KeyCombination {
    pub key: VirtualKeyCode,
    pub modifiers: Vec<VirtualKeyCode>,
}

pub struct KeyShortcuts {
    pub key_shortcuts: HashMap<String, KeyCombination>,
    // pub key_shortcuts: HashMap<String, String>,
}

impl KeyShortcuts {
    pub fn new() -> Self {
        Self {
            key_shortcuts: HashMap::new(),
        }
    }

    pub fn add_key_shortcut(&mut self, name: &str, key: VirtualKeyCode, modifiers: Vec<VirtualKeyCode>) {
        self.key_shortcuts.insert(name.to_string(), KeyCombination { key, modifiers });
    }
    pub fn get_key_shortcut(&self, name: &str, input: &Input) -> bool {
        if let Some(key_combination) = self.key_shortcuts.get(name) {
            if input.get_key_up(&key_combination.key) {
                for modifier in key_combination.modifiers.iter() {
                    if !input.get_key(modifier) {
                        return false;
                    }
                }
                return true;
            }
        }
        false
    }
    // pub fn get_key_shortcut(&self, name: &str) -> Option<&KeyCombination> {
    //     self.key_shortcuts.get(name)
    // }
}