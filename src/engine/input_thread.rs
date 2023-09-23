use std::sync::Arc;

use crossbeam::channel::{Receiver, Sender};
use force_send_sync::SendSync;
use winit::{
    dpi::PhysicalSize,
    event::{Event, ModifiersState, WindowEvent},
    event_loop::{self, ControlFlow, EventLoop},
};

use crate::engine::rendering::vulkan_manager::VulkanManager;

use super::{input::Input, EngineEvent};

pub(crate) fn input_thread(
    event_loop: SendSync<EventLoop<EngineEvent>>,
    vk: Arc<VulkanManager>,
    coms: Sender<(
        Vec<WindowEvent<'static>>,
        Input,
        Option<PhysicalSize<u32>>,
        bool,
    )>,
) {
    let mut focused = true;
    let mut input = Input::default();
    let mut modifiers = ModifiersState::default();
    // let mut recreate_swapchain = true;
    let mut events = Vec::new();
    let mut event_loop = event_loop.unwrap();
    let mut size = None;
    let mut should_quit = false;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::DeviceEvent { event, .. } => input.process_device(event, focused),
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::Focused(foc) => {
                        focused = foc;
                        if !focused {
                            let _er = vk
                                .window()
                                .set_cursor_grab(winit::window::CursorGrabMode::None);
                            match _er {
                                Ok(_) => {}
                                Err(e) => {
                                }
                            }
                        }
                    }
                    WindowEvent::CloseRequested => {
                        should_quit = true;
                        // *control_flow = ControlFlow::Exit;
                        // engine.end();
                    }
                    WindowEvent::MouseInput {
                        device_id,
                        state,
                        button,
                        modifiers,
                    } => input.process_mouse_input(device_id, state, button, modifiers),
                    WindowEvent::MouseWheel {
                        device_id,
                        delta,
                        phase,
                        modifiers,
                    } => input.process_mouse_wheel(device_id, delta, phase, modifiers),

                    WindowEvent::KeyboardInput {
                        input: ky_input,
                        device_id,
                        is_synthetic,
                    } => input.process_keyboard(device_id, ky_input, is_synthetic),
                    WindowEvent::ModifiersChanged(m) => modifiers = m,
                    WindowEvent::Resized(_size) => {
                        // recreate_swapchain = true;
                        size = Some(_size);
                    }
                    _ => (),
                }
                if let Some(event) = event.to_static() {
                    events.push(event);
                }
            }
            Event::UserEvent(e) => {
                match e {
                    EngineEvent::Send => {
                        // let mut a = Vec::new();
                        // swap(&mut a, &mut events);
                        coms.send((events.clone(), input.clone(), size, should_quit));
                        // recreate_swapchain = false;
                        size = None;
                        events.clear();
                        input.reset();
                    }
                    EngineEvent::Quit => {
                        // todo!()
                        *control_flow = ControlFlow::Exit;
                        return;
                    }
                }
            }
            _ => {} // Event::RedrawEventsCleared => {}
        }
    });
}