use std::{collections::HashSet, sync::Arc};

use anyhow::Result;
use ash::vk;
use egui::{Context, TextureId, ViewportId};
use egui_ash_renderer::{DynamicRendering, Options, RenderMode, Renderer, allocator::GpuAllocator};
use egui_winit::State;
use winit::{event::WindowEvent, window::Window};

use crate::app::engine::{
    gpu::context::RenderingContext,
    renderer::{
        self, Frame, ToonParamsUniformBufferObject, swapchain::Swapchain,
    },
};

pub struct EguiRenderer {
    pub light_pos: glam::Vec3,
    pub toon_params: ToonParamsUniformBufferObject,
    show_settings: bool,
    textures_to_free: HashSet<TextureId>,
    pub egui_ctx: Context,
    pub egui_winit: State,
    pub renderer: Renderer<GpuAllocator>,
    pub window: Arc<Window>,
}

impl EguiRenderer {
    pub fn new(
        window: Arc<Window>,
        ctx: Arc<RenderingContext>,
        swapchain: &Swapchain,
    ) -> Result<Self> {
        let egui_ctx = Context::default();
        let egui_winit = State::new(
            egui_ctx.clone(),
            ViewportId::ROOT,
            window.as_ref(),
            None,
            None,
            None,
        );

        let renderer = {
            let allocator = gpu_allocator::vulkan::Allocator::new(
                &gpu_allocator::vulkan::AllocatorCreateDesc {
                    instance: ctx.instance.clone(),
                    device: ctx.device.clone(),
                    physical_device: ctx.physical_device.handle,
                    debug_settings: Default::default(),
                    buffer_device_address: false,
                    allocation_sizes: Default::default(),
                },
            )?;

            let render_mode = if ctx.use_dynamic_rendering {
                RenderMode::DynamicRendering(DynamicRendering {
                    color_attachment_format: swapchain.format,
                    depth_attachment_format: Some(swapchain.depth_image.image_format),
                    stencil_attachment_format: Some(swapchain.depth_image.image_format),
                })
            } else {
                RenderMode::RenderPass(swapchain.main_render_pass.unwrap())
            };
            Renderer::with_allocator(
                GpuAllocator::new(std::sync::Arc::new(std::sync::Mutex::new(allocator))),
                ctx.device.clone(),
                render_mode,
                Options {
                    in_flight_frames: renderer::FRAME_OVERLAP,
                    srgb_framebuffer: true,
                    ..Default::default()
                },
            )?
        };

        Ok(Self {
            light_pos: glam::Vec3 {
                x: 20.0,
                y: 20.0,
                z: 20.0,
            },
            toon_params: ToonParamsUniformBufferObject {
                shadow_tint: glam::Vec4::from_array([0.6, 0.55, 0.8, 1.0]),
                light_color: glam::Vec4::from_array([1.0, 1.0, 1.0, 0.1]),
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
            },
            show_settings: true,
            textures_to_free: HashSet::new(),
            egui_ctx,
            egui_winit,
            renderer,
            window,
        })
    }

    pub fn record_commands(
        &mut self,
        frame: &Frame,
        extent: vk::Extent2D,
        queue: vk::Queue,
    ) -> Result<()> {
        for id in self.textures_to_free.drain() {
            self.renderer.free_texture(id)?;
        }

        let raw_input = self.egui_winit.take_egui_input(&self.window);

        let egui::FullOutput {
            platform_output,
            mut textures_delta,
            shapes,
            pixels_per_point,
            ..
        } = self.egui_ctx.run_ui(raw_input, |ui| {
            let ctx = ui.ctx().clone();

            egui::Window::new("Settings")
                .open(&mut self.show_settings) // adds a close button
                .default_pos([20.0, 20.0])
                .default_width(260.0)
                .resizable(true)
                .collapsible(true)
                .show(&ctx, |ui| {
                    ui.heading("Nekohazard Debug");
                    ui.add(egui::DragValue::new(&mut self.light_pos.x).prefix("light pos x: "));
                    ui.add(egui::DragValue::new(&mut self.light_pos.y).prefix("light pos y: "));
                    ui.add(egui::DragValue::new(&mut self.light_pos.z).prefix("light pos z: "));
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.shadow_tint.x)
                            .prefix("shadow tint r: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.shadow_tint.y)
                            .prefix("shadow tint g: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.shadow_tint.z)
                            .prefix("shadow tint b: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.light_color.x)
                            .prefix("light color r: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.light_color.y)
                            .prefix("light color b: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.light_color.z)
                            .prefix("light color g: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.light_color.w)
                            .prefix("light color a: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.step_threshold)
                            .prefix("step threshold: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.step_softness)
                            .prefix("step softness: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.rim_power).prefix("rim power: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.rim_strength)
                            .prefix("rim strength: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.rim_edge_0)
                            .prefix("rim edge 0: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.rim_edge_1)
                            .prefix("rim edge 1: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.spec_threshold)
                            .prefix("spec threshold: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.shininess).prefix("shininess: "),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.toon_params.spec_strength)
                            .prefix("spec strength: "),
                    );
                });
        });

        self.egui_winit
            .handle_platform_output(&self.window, platform_output);

        for (id, deltas) in textures_delta.set.drain() {
            for delta in deltas {
                self.renderer
                    .set_texture(queue, frame.command_pool, id, &delta);
            }
        }

        for id in textures_delta.free.drain() {
            self.textures_to_free.insert(id);
        }

        let clipped_primitives = self.egui_ctx.tessellate(shapes, pixels_per_point);
        self.renderer.cmd_draw(
            frame.command_buffer,
            extent,
            pixels_per_point,
            &clipped_primitives,
        )?;
        Ok(())
    }

    pub fn on_window_event(&mut self, event: &WindowEvent) -> bool {
        let response = self.egui_winit.on_window_event(&self.window, event);
        if response.repaint {
            self.window.request_redraw();
        }
        response.consumed
    }

    pub fn wants_keyboard(&self) -> bool {
        self.egui_ctx.egui_wants_keyboard_input()
    }

    pub fn wants_pointer(&self) -> bool {
        self.egui_ctx.egui_wants_pointer_input()
    }
}
