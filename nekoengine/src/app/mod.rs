use anyhow::Result;
use ash::vk::EXT_TOOLING_INFO_NAME;
use winit::application::ApplicationHandler;

use crate::app::engine::Engine;

pub mod engine;
pub mod game;

#[derive(Default)]
pub struct App {
    engine: Option<Engine>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        tracing::info!("resumed (engine exists: {})", self.engine.is_some());
        match Engine::new(event_loop) {
            Ok(engine) => {
                self.engine = Some(engine);
            }
            Err(err) => {
                tracing::error!("error: {}", err);
                event_loop.exit();
            }
        };
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let Some(engine) = self.engine.as_mut() {
            engine.device_event(event_loop, device_id, event);
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        if let Some(engine) = self.engine.as_mut() {
            engine.window_event(event_loop, window_id, event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let Some(engine) = self.engine.as_mut() {
            engine.request_redraw();
        }
    }

    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        tracing::info!("suspended");
        self.engine = None;
    }
}
