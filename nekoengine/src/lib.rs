use std::{collections::HashSet, sync::Arc, time::Instant};

use anyhow::{Ok, Result};
use bevy_ecs::{
    query::With,
    resource::Resource,
    schedule::{IntoScheduleConfigs, Schedule, SystemSet},
    system::{Query, Res, ResMut, ScheduleSystem},
    world::World,
};
use parking_lot::Mutex;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

use crate::{
    gpu::context::RenderingContext,
    input::InputManager,
    renderer::{
        ExtractedFrame, GlobalTransform, ModelInstance, RenderFrame, Renderer,
        ToonParamsUniformBufferObject, Transform, Visibility,
        camera::{Camera, CameraController, RenderCamera},
    },
    resources::{
        Resources,
        animation::Animator,
        data::{MaterialData, TextureData},
        gltf,
    },
};

pub mod gpu;
pub mod input;
pub mod renderer;
pub mod resources;

pub struct App {
    pub(crate) world: World,
    startup: Schedule,
    update: Schedule,
}

impl App {
    pub fn new() -> Self {
        Self {
            world: World::new(),
            startup: Schedule::default(),
            update: Schedule::default(),
        }
    }

    pub fn add_startup<M>(&mut self, s: impl IntoScheduleConfigs<ScheduleSystem, M>) -> &mut Self {
        self.startup.add_systems(s);
        self
    }
    pub fn add_systems<M>(&mut self, s: impl IntoScheduleConfigs<ScheduleSystem, M>) -> &mut Self {
        self.update.add_systems(s);
        self
    }
    pub fn insert_resource<R: Resource>(&mut self, r: R) -> &mut Self {
        self.world.insert_resource(r);
        self
    }
    pub(crate) fn run_startup(&mut self) {
        self.startup.run(&mut self.world);
    }
    pub(crate) fn tick(&mut self) {
        self.update.run(&mut self.world);
    }
}
pub trait Plugin {
    fn build(&self, engine: &mut App);
}

pub struct EngineBuilder {
    window_config: WindowConfig,
    plugins: Vec<Box<dyn Plugin>>,
}

impl EngineBuilder {
    pub fn new() -> Self {
        Self {
            window_config: WindowConfig {
                title: "Nekoengine Game".to_string(),
                size: (1920, 1080),
                grab_cursor: false,
            },
            plugins: Vec::new(),
        }
    }
    pub fn window(mut self, cfg: WindowConfig) -> Self {
        self.window_config = cfg;
        self
    }

    pub fn add_plugin(mut self, p: impl Plugin + 'static) -> Self {
        self.plugins.push(Box::new(p));
        self
    }

    pub fn run(self) -> anyhow::Result<()> {
        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut runner = Runner {
            builder: Some(self),
            engine: None,
            error: None,
        };
        event_loop.run_app(&mut runner)?;
        runner.error.map_or(Ok(()), Err)
    }
}

pub struct WindowConfig {
    pub title: String,
    pub size: (u32, u32),
    pub grab_cursor: bool,
}

struct Runner {
    builder: Option<EngineBuilder>,
    engine: Option<Engine>,
    error: Option<anyhow::Error>,
}

impl ApplicationHandler for Runner {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let Some(b) = self.builder.take() else { return };
        match Engine::new(el, &b.window_config, &b.plugins) {
            std::result::Result::Ok(e) => {
                self.engine = Some(e);
            }
            Err(err) => {
                self.error = Some(err);
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, id: WindowId, ev: WindowEvent) {
        if let Some(e) = &mut self.engine {
            if let Err(err) = e.window_event(el, id, ev) {
                self.error = Some(err);
                el.exit();
            }
        }
    }

    fn device_event(&mut self, el: &ActiveEventLoop, id: DeviceId, ev: DeviceEvent) {
        if let Some(e) = &mut self.engine {
            if let Err(err) = e.device_event(el, id, ev) {
                self.error = Some(err);
                el.exit();
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(e) = &self.engine {
            e.request_redraw();
        }
    }
}

#[derive(SystemSet, Hash, Eq, PartialEq, Clone, Debug)]
pub enum EngineSet {
    Game,
    Animate,
    Extract,
}

pub struct Engine {
    hide_cursor: bool,
    last_frame: Instant,
    input_manager: Arc<InputManager>,
    renderer: Renderer,
    resources: Arc<Mutex<Resources>>,
    window: Arc<Window>,
    context: Arc<RenderingContext>,
    app: App,
}

impl Engine {
    pub fn builder() -> EngineBuilder {
        EngineBuilder::new()
    }

    pub fn new(
        event_loop: &ActiveEventLoop,
        window_config: &WindowConfig,
        plugins: &[Box<dyn Plugin>],
    ) -> Result<Self> {
        let input_manager = Arc::new(InputManager::new());
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title(window_config.title.clone())
                    .with_inner_size(LogicalSize::new(window_config.size.0, window_config.size.1)),
            )?,
        );
        #[cfg(not(target_os = "android"))]
        {
            if window_config.grab_cursor {
                window.set_cursor_visible(false);
                if let Err(e) = window.set_cursor_grab(winit::window::CursorGrabMode::Locked) {
                    tracing::warn!("Locked cursor grab failed: {e}, trying Confined");
                    window.set_cursor_grab(winit::window::CursorGrabMode::Confined)?;
                }
            }
        }
        let context = Arc::new(RenderingContext::new(window.clone())?);
        let resources = Arc::new(Mutex::new(Resources::new(context.clone())?));
        {
            let mut res = resources.lock();
            res.upload_defaults()?;
        }

        let mut app = App::new();

        let mut render_frame = RenderFrame::default();
        render_frame.light_pos = glam::Vec3::new(20.0, 20.0, 20.0);
        render_frame.camera = RenderCamera::new(
            glam::Vec3 {
                x: 2.0,
                y: 2.0,
                z: 2.0,
            },
            glam::Vec3 {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            },
            225.0,
            -37.0,
            0.0,
        );
        render_frame.toon_ubo = ToonParamsUniformBufferObject {
            shadow_tint: glam::Vec4::new(0.6, 0.55, 0.8, 1.0),
            light_color: glam::Vec4::new(1.0, 1.0, 1.0, 0.1),
            step_threshold: 0.5,
            step_softness: 0.03,
            rim_power: 4.0,
            rim_strength: 0.4,
            rim_edge_0: 0.6,
            rim_edge_1: 0.8,
            spec_threshold: 0.5,
            shininess: 32.0,
            spec_strength: 0.3,
            _pad: 0.0,
        };
        app.insert_resource(Time { dt: 0.0 })
            .insert_resource(Assets {
                assets: resources.clone(),
            })
            .insert_resource(Input {
                input: input_manager.clone(),
            })
            .insert_resource(ExtractedFrame(render_frame));
        app.update
            .configure_sets((EngineSet::Game, EngineSet::Animate, EngineSet::Extract).chain());
        app.update
            .add_systems(Renderer::advance_animators.in_set(EngineSet::Animate));
        app.update.add_systems(
            (Renderer::extract_camera, Renderer::extract_instances).in_set(EngineSet::Extract),
        );
        for p in plugins {
            p.build(&mut app);
        }

        app.run_startup();
        resources.lock().flush()?;

        let renderer = Renderer::new(context.clone(), window.clone(), resources.clone())?;

        Ok(Self {
            hide_cursor: true,
            last_frame: Instant::now(),
            input_manager,
            renderer,
            resources,
            window,
            context,
            app,
        })
    }

    pub fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) -> Result<()> {
        match event {
            DeviceEvent::MouseMotion { delta } if !self.renderer.egui_renderer.wants_pointer() => {
                if self.hide_cursor {
                    let mut look = self.input_manager.mouse_look_delta.write();
                    look.0 += delta.0 as f32;
                    look.1 += delta.1 as f32;
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
                                        .set_cursor_grab(winit::window::CursorGrabMode::None)?;
                                    self.hide_cursor = false;
                                } else {
                                    self.window.set_cursor_visible(false);
                                    self.window
                                        .set_cursor_grab(winit::window::CursorGrabMode::Locked)?;
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
        Ok(())
    }

    pub fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) -> Result<()> {
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
                self.app.world.resource_mut::<Time>().dt = delta_time;
                self.app.tick();
                self.renderer
                    .render(&self.app.world.resource::<ExtractedFrame>().0, delta_time)
                    .unwrap();
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
            } => {}
            WindowEvent::Touch(touch) => {
                let width = self.window.inner_size().width as f64;
                self.input_manager
                    .touch_controls
                    .write()
                    .handle(&touch, width);
            }
            _ => {}
        }
        Ok(())
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }
}

#[derive(Resource)]
pub struct Input {
    pub input: Arc<InputManager>,
}

#[derive(Resource)]
pub struct Assets {
    pub assets: Arc<Mutex<Resources>>,
}

#[derive(Resource)]
pub struct Time {
    pub dt: f32,
}
