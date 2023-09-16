use crossbeam::channel::{Receiver, Sender};
use force_send_sync::SendSync;
use winit::{
    event::{Event, WindowEvent, ModifiersState},
    event_loop::{self, ControlFlow, EventLoop}, dpi::PhysicalSize,
};

use super::input::Input;

pub(crate) fn input_thread(event_loop: SendSync<EventLoop<()>>, coms: Sender<(Vec<WindowEvent<'static>>,Input, Option<PhysicalSize<u32>>, bool)>) {
    let mut focused = true;
    let mut input = Input::default();
    let mut modifiers = ModifiersState::default();
    // let mut recreate_swapchain = true;
    let mut events = Vec::new();
    let mut event_loop = event_loop.unwrap();
    let mut size = None;
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::DeviceEvent { event, .. } => input.process_device(event, focused),
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::Focused(foc) => {
                        focused = foc;
                    }
                    WindowEvent::CloseRequested => {
                        *control_flow = ControlFlow::Exit;
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
                // if !input.get_key(&VirtualKeyCode::Space) {
                // gui.update(&event);
                // }
            }
            Event::MainEventsCleared => {
                coms.send((events.clone(),input.clone(), size, *control_flow != ControlFlow::Exit));
                // recreate_swapchain = false;
                size = None;
                events.clear();
                input.reset();
            },
            _ => {} // Event::RedrawEventsCleared => {}
        }
    });
}
