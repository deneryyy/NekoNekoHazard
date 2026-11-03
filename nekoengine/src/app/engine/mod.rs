use std::{collections::HashSet, sync::Arc, time::Instant};

use anyhow::{Ok, Result};
use bevy_ecs::{
    query::With,
    resource::Resource,
    schedule::{IntoScheduleConfigs, Schedule},
    system::{Query, Res, ResMut},
    world::World,
};
use parking_lot::Mutex;
use winit::{
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::app::engine::{
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

pub struct Engine {
    hide_cursor: bool,
    last_frame: Instant,
    startup: Schedule,
    update: Schedule,
    world: World,
    input_manager: Arc<InputManager>,
    renderer: Renderer,
    resources: Arc<Mutex<Resources>>,
    window: Arc<Window>,
    context: Arc<RenderingContext>,
}

const SPEED: f32 = 20.0;
const SENSITIVITY: f32 = 0.1;

impl Engine {
    pub fn new(event_loop: &ActiveEventLoop) -> Result<Self> {
        let input_manager = Arc::new(InputManager::new());
        let window = Arc::new(event_loop.create_window(Default::default())?);
        #[cfg(not(target_os = "android"))]
        {
            window.set_cursor_visible(false);
            if let Err(e) = window.set_cursor_grab(winit::window::CursorGrabMode::Locked) {
                tracing::warn!("Locked cursor grab failed: {e}, trying Confined");
                window.set_cursor_grab(winit::window::CursorGrabMode::Confined)?;
            }
        }
        let context = Arc::new(RenderingContext::new(window.clone())?);
        let resources = Arc::new(Mutex::new(Resources::new(context.clone())?));
        let annie_data = gltf::load_model_glb(
            "annie",
            include_bytes!(concat!(env!("OUT_DIR"), "/res/models/annie.glb")),
        )?;
        let chito_data = gltf::load_model_glb(
            "chito",
            include_bytes!(concat!(env!("OUT_DIR"), "/res/models/chito.glb")),
        )?;
        let mut annie_model_handle;
        let mut chito_model_handle;
        {
            let mut res = resources.lock();
            res.upload_defaults()?;
            let missing_texture = res.upload_texture(TextureData::decode(
                include_bytes!(concat!(env!("OUT_DIR"), "/res/textures/missing.png")),
                true,
            )?)?;
            let missing_material = res.upload_material(
                MaterialData {
                    name: "missing".to_string(),
                    base_color_image: Some(0),
                    base_color_factor: [1.0; 4],
                },
                &[missing_texture],
                missing_texture,
            )?;
            annie_model_handle = res.upload_model(annie_data, missing_texture, missing_material)?;
            chito_model_handle = res.upload_model(chito_data, missing_texture, missing_material)?;
            res.flush()?;
        }

        let mut world = World::new();
        world.insert_resource(Time { dt: 0.0 });
        world.insert_resource(Assets {
            assets: resources.clone(),
        });
        world.insert_resource(Input {
            input: input_manager.clone(),
        });
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
        world.insert_resource(ExtractedFrame(render_frame));
        {
            let res = resources.lock();
            let annie_model = res.models.get(annie_model_handle).unwrap();
            let mut e_annie = world.spawn((
                Visibility(true),
                ModelInstance(annie_model_handle),
                GlobalTransform(glam::Mat4::from_scale(glam::Vec3::new(5.0, 5.0, 5.0))),
            ));
            if !annie_model.clips.is_empty() {
                e_annie.insert(Animator::new(&annie_model.skeleton));
            }
            let chito_model = res.models.get(chito_model_handle).unwrap();
            let mut e_chito = world.spawn((
                Visibility(true),
                ModelInstance(chito_model_handle),
                GlobalTransform(
                    glam::Mat4::from_translation(glam::Vec3::new(15.0, 0.0, 0.0))
                        * glam::Mat4::from_scale(glam::Vec3::new(5.0, 5.0, 5.0)),
                ),
            ));
            if !chito_model.clips.is_empty() {
                e_chito.insert(Animator::new(&chito_model.skeleton));
            }
        }

        // Player entity
        world.spawn((
            Camera {
                fov_y: 45.0,
                near: 0.1,
                far: 1000.0,
            },
            CameraController {
                yaw: 225.0,
                pitch: -37.0,
            },
            Transform {
                translation: glam::Vec3::new(2.0, 2.0, 2.0),
                rotation: glam::Quat::IDENTITY,
                scale: glam::Vec3::new(1.0, 1.0, 1.0),
            },
        ));

        let mut update = Schedule::default();
        update.add_systems(
            (
                Self::player_controller,
                Renderer::advance_animators,
                Renderer::extract_camera,
                Renderer::extract_instances,
            )
                .chain(),
        );
        let renderer = Renderer::new(context.clone(), window.clone(), resources.clone())?;
        Ok(Self {
            hide_cursor: true,
            last_frame: Instant::now(),
            world,
            startup: Schedule::default(),
            update,
            input_manager,
            renderer,
            resources,
            window,
            context,
        })
    }

    pub fn run_startup(&mut self) {
        self.startup.run(&mut self.world);
    }

    pub fn player_controller(
        mut q: Query<(&mut CameraController, &mut Transform), With<Camera>>,
        input: Res<Input>,
        time: Res<Time>,
    ) {
        let im = &input.input;
        for (mut ctl, mut tf) in &mut q {
            let (dx, dy) = {
                let l = im.mouse_look_delta.read();
                (l.0, l.1)
            };
            let (tdx, tdy) = im.touch_controls.write().take_look_delta();

            ctl.yaw += (dx + tdx) * SENSITIVITY;
            ctl.pitch = (ctl.pitch - (dy + tdy) * SENSITIVITY).clamp(-89.0, 89.0);

            {
                let mut l = im.mouse_look_delta.write();
                l.0 = 0.0;
                l.1 = 0.0;
            }

            let front = ctl.front();
            let right = front.cross(glam::Vec3::Y).normalize();

            let mut movement = glam::Vec3::ZERO;
            for key in im.get_snapshot() {
                match key {
                    KeyCode::KeyW => movement += front,
                    KeyCode::KeyS => movement -= front,
                    KeyCode::KeyD => movement += right,
                    KeyCode::KeyA => movement -= right,
                    KeyCode::Space => movement += glam::Vec3::Y,
                    KeyCode::ShiftLeft => movement -= glam::Vec3::Y,
                    _ => {}
                }
            }
            tf.translation += movement.normalize_or_zero() * SPEED * time.dt;
        }
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
                self.world.resource_mut::<Time>().dt = delta_time;
                self.update.run(&mut self.world);
                self.renderer
                    .render(&self.world.resource::<ExtractedFrame>().0, delta_time)
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
    }

    pub fn request_redraw(&self) {
        self.window.request_redraw();
    }
}

#[derive(Resource)]
pub struct Input {
    input: Arc<InputManager>,
}

#[derive(Resource)]
pub struct Assets {
    assets: Arc<Mutex<Resources>>,
}

#[derive(Resource)]
pub struct Time {
    pub dt: f32,
}
