use std::collections::HashMap;

use super::time::Time;
use winit::event::{
    DeviceEvent, DeviceId, ElementState, Event, KeyboardInput, ModifiersState, MouseButton,
    VirtualKeyCode, WindowEvent, MouseScrollDelta, TouchPhase,
};

#[derive(Default, Clone)]
pub struct Input {
    // focused: bool,
    pub(crate) key_downs: HashMap<VirtualKeyCode, bool>,
    pub(crate) key_presses: HashMap<VirtualKeyCode, bool>,
    pub(crate) key_ups: HashMap<VirtualKeyCode, bool>,
    pub(crate) mouse_x: f64,
    pub(crate) mouse_y: f64,
    pub(crate) mouse_whl_vert: f32,
    pub(crate) mouse_whl_horz: f32,
    pub(crate) mouse_buttons: HashMap<u32, bool>,
}

#[allow(dead_code)]
impl Input {
    // #[inline]
    // pub fn get_time(&self) -> &Time {
    //     &self.time
    // }
    pub fn get_key(&self, key: &VirtualKeyCode) -> bool {
        *self.key_downs.get(key).unwrap_or(&false)
    }
    pub fn get_key_press(&self, key: &VirtualKeyCode) -> bool {
        *self.key_presses.get(key).unwrap_or(&false)
    }
    pub fn get_key_up(&self, key: &VirtualKeyCode) -> bool {
        *self.key_ups.get(key).unwrap_or(&false)
    }
    pub fn get_mouse_delta(&self) -> (f64, f64) {
        (self.mouse_x, self.mouse_y)
    }
    pub fn get_mouse_button(&self, id: &u32) -> bool {
        *self.mouse_buttons.get(id).unwrap_or(&false)
    }
    pub fn get_mouse_scroll(&self) -> (&f32,&f32) {
        (&self.mouse_whl_horz, &self.mouse_whl_vert)
    }
    pub(crate) fn reset(&mut self) {
        self.key_presses.clear();
        self.key_ups.clear();
        self.mouse_x = 0.;
        self.mouse_y = 0.;
        self.mouse_whl_horz = 0.;
        self.mouse_whl_vert = 0.;
    }
    pub(crate) fn process_device(&mut self, event: DeviceEvent, focused: bool) {
        if !focused {
            return;
        }
        match event {
            DeviceEvent::MouseMotion { delta } => {
                self.mouse_x += delta.0;
                self.mouse_y += delta.1;
            }
            DeviceEvent::Added => (),
            DeviceEvent::Removed => (),
            DeviceEvent::MouseWheel { delta } => match delta {
                winit::event::MouseScrollDelta::LineDelta(x, y) => self.mouse_whl_vert += y,
                winit::event::MouseScrollDelta::PixelDelta(_) => (),
            },
            DeviceEvent::Motion { axis, value } => (),
            // DeviceEvent::Button { button, state } => println!("Button: {} {:?}", button, state),
            // DeviceEvent::Key(key) => {
            //     let KeyboardInput {
            //         scancode,
            //         state,
            //         virtual_keycode,
            //         modifiers,
            //     } = key;
            //     println!("key: {} {:?}", scancode, state);
            //     if let Some(virtual_keycode) = virtual_keycode {
            //         match state {
            //             ElementState::Pressed => {
            //                 self.key_presses.insert(virtual_keycode, true);
            //                 self.key_downs.insert(virtual_keycode, true);
            //             }
            //             ElementState::Released => {
            //                 self.key_downs.insert(virtual_keycode, false);
            //                 self.key_ups.insert(virtual_keycode, true);
            //             }
            //         }
            //     }
            // }
            // DeviceEvent::Text { codepoint } => (),
            _ => (),
        }
    }
    pub(crate) fn process_mouse_input(
        &mut self,
        device_id: DeviceId,
        state: ElementState,
        button: MouseButton,
        modifiers: ModifiersState,
    ) {
        match state {
            ElementState::Pressed => {
                self.mouse_buttons.insert(
                    match button {
                        MouseButton::Left => 0,
                        MouseButton::Middle => 1,
                        MouseButton::Right => 2,
                        MouseButton::Other(x) => x as u32,
                    },
                    true,
                );
            }
            ElementState::Released => {
                self.mouse_buttons.insert(
                    match button {
                        MouseButton::Left => 0,
                        MouseButton::Middle => 1,
                        MouseButton::Right => 2,
                        MouseButton::Other(x) => x as u32,
                    },
                    false,
                );
            }
        }
    }
    pub(crate) fn process_mouse_wheel(
        &mut self,
        device_id: DeviceId, delta: MouseScrollDelta, phase: TouchPhase, modifiers: ModifiersState
    ) {
        match delta {
            MouseScrollDelta::LineDelta(x, y) => { self.mouse_whl_vert += y; self.mouse_whl_horz += x },
            MouseScrollDelta::PixelDelta(_) => {},
        }
    }
    pub(crate) fn process_keyboard(
        &mut self,
        device_id: DeviceId,
        input: KeyboardInput,
        is_synthetic: bool,
    ) {
        // if let WindowEvent::KeyboardInput { input: x, .. } = event {
        match input {
            KeyboardInput {
                state: ElementState::Released,
                virtual_keycode: Some(key),
                ..
            } => {
                self.key_downs.insert(key, false);
                self.key_ups.insert(key, true);
            }
            KeyboardInput {
                state: ElementState::Pressed,
                virtual_keycode: Some(key),
                ..
            } => {
                self.key_presses.insert(key, true);
                self.key_downs.insert(key, true);
            }
            _ => {}
        };
        // }
    }
    // pub(crate) fn process_event<T>(&mut self, event: Event<T>) {
    //     match event {
    //         Event::DeviceEvent { event, .. } => match event {
    //             DeviceEvent::MouseMotion { delta } => {
    //                 // if focused {
    //                 self.mouse_x += delta.0;
    //                 self.mouse_y += delta.1;
    //                 // }
    //             }
    //             _ => (),
    //         },
    //         Event::WindowEvent { event, .. } => {
    //             match event {
    //                 WindowEvent::Focused(foc) => {
    //                     focused = foc;
    //                     println!("main event_loop id: {:?}", thread::current().id());
    //                 }
    //                 WindowEvent::CloseRequested => {
    //                     *control_flow = ControlFlow::Exit;
    //                     running.store(false, Ordering::SeqCst);
    //                     let game_thread = game_thread.remove(0);
    //                     let _res = coms.1.send((input.clone(), playing_game));
    //                     game_thread.join().unwrap();
    //                     save_project(&file_watcher, &world.lock(), assets_manager.clone());
    //                     perf.print();
    //                 }
    //                 WindowEvent::MouseInput {
    //                     device_id: _,
    //                     state,
    //                     button,
    //                     ..
    //                 } => match state {
    //                     ElementState::Pressed => {
    //                         self.mouse_buttons.insert(
    //                             match button {
    //                                 MouseButton::Left => 0,
    //                                 MouseButton::Middle => 1,
    //                                 MouseButton::Right => 2,
    //                                 MouseButton::Other(x) => x as u32,
    //                             },
    //                             true,
    //                         );
    //                     }
    //                     ElementState::Released => {
    //                         self.mouse_buttons.insert(
    //                             match button {
    //                                 MouseButton::Left => 0,
    //                                 MouseButton::Middle => 1,
    //                                 MouseButton::Right => 2,
    //                                 MouseButton::Other(x) => x as u32,
    //                             },
    //                             false,
    //                         );
    //                     }
    //                 },
    //                 WindowEvent::KeyboardInput { input: x, .. } => {
    //                     let _ = match x {
    //                         KeyboardInput {
    //                             state: ElementState::Released,
    //                             virtual_keycode: Some(key),
    //                             ..
    //                         } => {
    //                             self.key_downs.insert(key, false);
    //                             self.key_ups.insert(key, true);
    //                         }
    //                         KeyboardInput {
    //                             state: ElementState::Pressed,
    //                             virtual_keycode: Some(key),
    //                             ..
    //                         } => {
    //                             self.key_presses.insert(key, true);
    //                             self.key_downs.insert(key, true);
    //                         }
    //                         _ => {}
    //                     };
    //                 }
    //                 WindowEvent::ModifiersChanged(m) => modifiers = m,
    //                 WindowEvent::Resized(_size) => {
    //                     recreate_swapchain = true;
    //                 }
    //                 _ => (),
    //             }

    //             if !input.get_key(&VirtualKeyCode::Space) {
    //                 gui.update(&event);
    //             }
    //         }
    //         _ => panic!(),
    //     }
    // }
}
