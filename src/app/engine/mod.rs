pub mod gpu;
pub mod input;
pub mod renderer;
pub mod resources;

use std::{sync::Arc, thread, time::Instant};

use anyhow::{Ok, Result};
use parking_lot::Mutex;
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::app::{
    engine::{
        gpu::context::RenderingContext,
        input::InputManager,
        renderer::Renderer,
        resources::{Resources, gltf},
    },
    game::{Game, world::World},
};

pub struct Engine {
    hide_cursor: bool,
    last_frame: Instant,
    world: Arc<World>,
    input_manager: Arc<InputManager>,
    renderer: Renderer,
    resources: Arc<Mutex<Resources>>,
    window: Arc<Window>,
    context: Arc<RenderingContext>,
}

impl Engine {
    pub fn new(event_loop: &ActiveEventLoop) -> Result<Self> {
        let input_manager = Arc::new(InputManager::new());
        let world = Arc::new(World::new(input_manager.clone()));
        let world_ = world.clone();
        thread::spawn(move || {
            let mut game = Game::new(world_);
            game.start_game();
        });

        let window = Arc::new(event_loop.create_window(Default::default())?);
        window.set_cursor_visible(false);
        window.set_cursor_grab(winit::window::CursorGrabMode::Locked)?;
        let context = Arc::new(RenderingContext::new(window.clone())?);
        let resources = Arc::new(Mutex::new(Resources::new(context.clone())?));
        let miku_data = gltf::load_model_glb(
            "annie",
            include_bytes!(concat!(env!("OUT_DIR"), "/res/models/annie.glb")),
        )?;
        {
            let mut res = resources.lock();
            res.upload_model(miku_data)?;
            res.flush()?;
        }
        let renderer = Renderer::new(
            context.clone(),
            window.clone(),
            resources.clone(),
            world.clone(),
        )?;
        Ok(Self {
            hide_cursor: true,
            last_frame: Instant::now(),
            world,
            input_manager,
            renderer,
            resources,
            window,
            context,
        })
    }

    pub fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        match event {
            DeviceEvent::MouseMotion { delta } if !self.renderer.egui_renderer.wants_pointer() => {
                if self.hide_cursor {
                    self.world
                        .player
                        .write()
                        .camera
                        .process_mouse_movement(delta.0 as f32, delta.1 as f32);
                }
            }
            DeviceEvent::Key(event) if !self.renderer.egui_renderer.wants_keyboard() => {
                match event.physical_key {
                    PhysicalKey::Code(code) => match event.state {
                        ElementState::Pressed => {
                            if code == KeyCode::KeyE {
                                if self.hide_cursor {
                                    self.window.set_cursor_visible(true);
                                    self.window
                                        .set_cursor_grab(winit::window::CursorGrabMode::None)
                                        .unwrap();
                                    self.hide_cursor = false;
                                } else {
                                    self.window.set_cursor_visible(false);
                                    self.window
                                        .set_cursor_grab(winit::window::CursorGrabMode::Locked)
                                        .unwrap();
                                    self.hide_cursor = true;
                                }
                            }

                            if self.hide_cursor {
                                self.input_manager.pressed_keys.write().insert(code);
                            }
                        }
                        ElementState::Released => {
                            self.input_manager.pressed_keys.write().remove(&code);
                        }
                    },
                    _ => {}
                }
            }
            _ => {}
        }
    }

    pub fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        let consumed = self.renderer.egui_renderer.on_window_event(&event);
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                self.renderer.resize().unwrap();
            }
            WindowEvent::ScaleFactorChanged { .. } => {
                self.renderer.resize().unwrap();
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let delta_time = now.duration_since(self.last_frame).as_secs_f32();
                self.last_frame = now;
                {
                    let mut player = self.world.player.write();
                    player.tick(delta_time);
                }
                self.renderer.render(delta_time).unwrap();
            }
            WindowEvent::KeyboardInput {
                device_id,
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                is_synthetic,
            } => {
                //self.renderer.process_camera_input(code, key_state);
            }
            _ => {}
        }
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }
}
