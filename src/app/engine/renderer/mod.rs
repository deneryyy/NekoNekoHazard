pub mod camera;
pub mod character_renderer;
pub mod context;
pub mod descriptor;
pub mod egui_renderer;
pub mod model;
pub mod staging;
pub mod swapchain;
pub mod texture;

use std::{sync::Arc, u32};

use anyhow::{Ok, Result};
use tracing::info;
use glam::Mat4;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use parking_lot::Mutex;
use winit::window::Window;

use crate::app::{
    engine::renderer::{
        character_renderer::CharacterRenderer,
        context::{AllocatedBuffer, RenderingContext},
        descriptor::{DescriptorAllocator, DescriptorLayoutBuilder, PoolSizeRatio},
        egui_renderer::EguiRenderer,
        model::Model,
        staging::RingStagingBuffer,
        swapchain::Swapchain,
        texture::Texture,
    },
    game::world::World,
};
use ash::vk::{self, CommandBufferUsageFlags};

pub struct Frame {
    character_descriptors: vk::DescriptorSet,
    character_descriptor_layout: vk::DescriptorSetLayout,
    toon_params_uniform_buffer: AllocatedBuffer,
    frame_uniform_buffer: AllocatedBuffer,
    in_flight_fence: vk::Fence,
    image_available_semaphore: vk::Semaphore,
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
}

#[repr(C)]
struct FrameUniformBufferObject {
    view: glam::Mat4,
    proj: glam::Mat4,
    light_pos: glam::Vec4,
    cam_pos: glam::Vec4,
}

#[repr(C)]
struct ToonParamsUniformBufferObject {
    shadow_tint: glam::Vec4,
    light_color: glam::Vec4,
    step_threshold: f32,
    step_softness: f32,
    rim_power: f32,
    rim_strength: f32,
    rim_edge_0: f32,
    rim_edge_1: f32,
    spec_threshold: f32,
    shininess: f32,
    spec_strength: f32,
    _pad: f32,
}

pub struct Renderer {
    pub egui_renderer: EguiRenderer,
    character_renderer: CharacterRenderer,
    texture_descriptors: vk::DescriptorSet,
    texture_descriptor_set_layout: vk::DescriptorSetLayout,
    ring_staging_buffer: RingStagingBuffer,
    allocator: Arc<Mutex<Allocator>>,
    world: Arc<World>,
    global_descriptor_allocator: DescriptorAllocator,
    render_finished_semaphores: Vec<vk::Semaphore>,
    frame_number: usize,
    frames: Vec<Frame>,
    swapchain: Swapchain,
    context: Arc<RenderingContext>,
}

pub const FRAME_OVERLAP: usize = 2;

impl Renderer {
    pub fn new(
        context: Arc<RenderingContext>,
        window: Arc<Window>,
        world: Arc<World>,
    ) -> Result<Self> {
        let mut swapchain = Swapchain::new(context.clone(), window.clone())?;
        swapchain.resize()?;

        let allocator = Arc::new(Mutex::new(Allocator::new(&AllocatorCreateDesc {
            instance: context.instance.clone(),
            device: context.device.clone(),
            physical_device: context.physical_device.handle,
            debug_settings: Default::default(),
            buffer_device_address: context.use_buffer_device_address,
            allocation_sizes: Default::default(),
        })?));

        let miku_model = model::load_model_glb(
            context.clone(),
            allocator.clone(),
            Vec::from(include_bytes!(concat!(
                env!("OUT_DIR"),
                "/res/models/hatsune_miku.glb"
            ))),
        )?;
        let models = Vec::from([miku_model]);

        let textures: Vec<&Texture> = models
            .iter()
            .flat_map(|model| &model.materials)
            .map(|material| &material.diffuse_texture)
            .collect();
        let supported_textures: u32 = textures.len() as u32;

        let sizes = Vec::from([
            PoolSizeRatio {
                desc_type: vk::DescriptorType::UNIFORM_BUFFER,
                ratio: 4,
            },
            PoolSizeRatio {
                desc_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                ratio: supported_textures,
            },
        ]);
        let global_descriptor_allocator =
            DescriptorAllocator::new(&context.device, FRAME_OVERLAP as u32 * 2 + 1, sizes)?;

        let mut texture_descriptor_layout_builder = DescriptorLayoutBuilder::new();
        texture_descriptor_layout_builder.add_binding(
            0,
            vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            supported_textures,
            vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
            vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT
                | vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        );

        let texture_descriptor_set_layout =
            texture_descriptor_layout_builder.build(&context.device)?;
        let texture_descriptors = global_descriptor_allocator.allocate(
            &context.device,
            texture_descriptor_set_layout,
            &[supported_textures],
        )?;
        let image_infos: Vec<vk::DescriptorImageInfo> = textures
            .iter()
            .map(|tex| {
                vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(tex.image.image_view)
                    .sampler(tex.sampler)
            })
            .collect();

        let tex_write = vk::WriteDescriptorSet::default()
            .dst_binding(0)
            .dst_set(texture_descriptors)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_infos);

        unsafe {
            context.device.update_descriptor_sets(&[tex_write], &[]);
        }

        unsafe {
            let mut render_finished_semaphores: Vec<vk::Semaphore> =
                Vec::with_capacity(swapchain.images.len() as usize);
            for _ in 0..swapchain.images.len() {
                let render_finished_semaphore = context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                render_finished_semaphores.push(render_finished_semaphore);
            }

            let mut frames: Vec<Frame> = Vec::with_capacity(FRAME_OVERLAP);

            for _ in 0..FRAME_OVERLAP {
                let image_available_semaphore = context
                    .device
                    .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
                let in_flight_fence = context.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?;
                let command_pool = context.device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(context.queue_families.graphics)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )?;

                let &command_buffer = context
                    .device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(command_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )?
                    .iter()
                    .next()
                    .unwrap();

                let frame_uniform_buffer = context.create_buffer(
                    "Uniform Buffer",
                    &mut allocator.lock(),
                    std::mem::size_of::<FrameUniformBufferObject>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )?;

                let toon_params_uniform_buffer = context.create_buffer(
                    "Toon Params Uniform Buffer",
                    &mut allocator.lock(),
                    std::mem::size_of::<ToonParamsUniformBufferObject>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )?;

                let mut character_descriptor_layout_builder = DescriptorLayoutBuilder::new();
                character_descriptor_layout_builder.add_binding(
                    0,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    1,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    vk::DescriptorBindingFlags::empty(),
                );
                character_descriptor_layout_builder.add_binding(
                    1,
                    vk::DescriptorType::UNIFORM_BUFFER,
                    1,
                    vk::ShaderStageFlags::FRAGMENT,
                    vk::DescriptorBindingFlags::empty(),
                );

                let character_descriptor_layout =
                    character_descriptor_layout_builder.build(&context.device)?;

                let character_descriptors = global_descriptor_allocator.allocate(
                    &context.device,
                    character_descriptor_layout,
                    &[],
                )?;
                let mut desc_writes: Vec<vk::WriteDescriptorSet> = Vec::new();
                let uniform_desc_info = &[vk::DescriptorBufferInfo::default()
                    .buffer(frame_uniform_buffer.buffer)
                    .range(std::mem::size_of::<FrameUniformBufferObject>() as u64)];
                desc_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_binding(0)
                        .dst_set(character_descriptors)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(uniform_desc_info),
                );
                let toon_params_desc_info = &[vk::DescriptorBufferInfo::default()
                    .buffer(toon_params_uniform_buffer.buffer)
                    .range(std::mem::size_of::<ToonParamsUniformBufferObject>() as u64)];
                desc_writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_binding(1)
                        .dst_set(character_descriptors)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .buffer_info(toon_params_desc_info),
                );
                desc_writes.push(tex_write);
                context.device.update_descriptor_sets(&desc_writes, &[]);

                frames.push(Frame {
                    character_descriptors,
                    character_descriptor_layout,
                    toon_params_uniform_buffer,
                    frame_uniform_buffer,
                    image_available_semaphore,
                    in_flight_fence,
                    command_buffer,
                    command_pool,
                });
            }

            let character_descriptor_set_layouts = frames
                .iter()
                .map(|frame| return frame.character_descriptor_layout)
                .collect::<Vec<_>>();
            let egui_renderer = EguiRenderer::new(window.clone(), context.clone(), &swapchain)?;
            Ok(Self {
                egui_renderer,
                character_renderer: CharacterRenderer::new(
                    context.clone(),
                    allocator.clone(),
                    world.clone(),
                    textures.len() as u32,
                    &swapchain,
                    &character_descriptor_set_layouts,
                    &texture_descriptor_set_layout,
                    models,
                )?,
                texture_descriptor_set_layout,
                texture_descriptors,
                ring_staging_buffer: RingStagingBuffer::new(
                    context.clone(),
                    &mut allocator.lock(),
                    64 * 1024 * 1024,
                )?,
                allocator: allocator.clone(),
                world,
                global_descriptor_allocator,
                render_finished_semaphores,
                frame_number: 0,
                frames,
                swapchain,
                context,
            })
        }
    }

    pub fn resize(&mut self) -> Result<()> {
        self.swapchain.resize()
    }

    fn upload_uniform_buffer(
        world: Arc<World>,
        swapchain: &Swapchain,
        frame: &Frame,
        light_pos: glam::Vec3,
        toon_ubo: &ToonParamsUniformBufferObject,
    ) {
        unsafe {
            let player = world.player.read();
            let mut proj = glam::Mat4::perspective_rh(
                45.0,
                swapchain.draw_image.extent.width as f32
                    / swapchain.draw_image.extent.height as f32,
                0.1,
                1000.0,
            );
            proj.y_axis.y *= -1.0;

            let view = player.camera.get_view_matrix();

            let ubo = FrameUniformBufferObject {
                view,
                proj,
                light_pos: light_pos.extend(0.0),
                cam_pos: player.camera.position.extend(0.0),
            };
            let ubo_ptr = frame
                .frame_uniform_buffer
                .allocation
                .mapped_ptr()
                .unwrap()
                .as_ptr();
            std::ptr::copy_nonoverlapping(&ubo, ubo_ptr.cast(), 1);

            let toon_ubo_ptr = frame
                .toon_params_uniform_buffer
                .allocation
                .mapped_ptr()
                .unwrap()
                .as_ptr();
            std::ptr::copy_nonoverlapping(toon_ubo, toon_ubo_ptr.cast(), 1);
        }
    }

    pub fn render(&mut self) -> Result<()> {
        unsafe {
            let frame_index = self.frame_number % FRAME_OVERLAP;
            let frame = &self.frames[frame_index];

            // Wait until gpu finished rendering last frame
            self.context
                .device
                .wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;
            self.ring_staging_buffer.retire_frame(frame_index)?;

            if self.swapchain.is_dirty {
                self.swapchain.resize()?;
                self.swapchain.is_dirty = false;
            }

            let image_index_result = self
                .swapchain
                .acquire_next_image(frame.image_available_semaphore);

            if image_index_result.is_err() {
                self.swapchain.resize()?;
                return Ok(());
            }
            let image_index = image_index_result?;

            self.context.device.reset_fences(&[frame.in_flight_fence])?;

            self.context
                .device
                .reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())?;

            self.context.device.begin_command_buffer(
                frame.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            Self::upload_uniform_buffer(
                self.world.clone(),
                &self.swapchain,
                frame,
                self.egui_renderer.light_pos,
                &self.egui_renderer.toon_params,
            );

            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.draw_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            );

            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.depth_image.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::DEPTH,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            );

            let extent = vk::Extent2D::default()
                .width(self.swapchain.draw_image.extent.width)
                .height(self.swapchain.draw_image.extent.height);

            self.context.begin_rendering(
                frame.command_buffer,
                self.swapchain.draw_image.image_view,
                self.swapchain.depth_image.image_view,
                vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
                vk::Rect2D::default().extent(extent),
                self.swapchain.framebuffer,
                self.swapchain.main_render_pass,
            );

            self.context.device.cmd_set_viewport(
                frame.command_buffer,
                0,
                &[vk::Viewport::default()
                    .width(extent.width as f32)
                    .height(extent.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0)],
            );

            self.context.device.cmd_set_scissor(
                frame.command_buffer,
                0,
                &[vk::Rect2D::default().extent(extent)],
            );

            self.character_renderer.record_commands(
                frame,
                &self.frames,
                extent,
                &self.swapchain,
                self.texture_descriptors,
            );

            self.egui_renderer.record_commands(
                frame,
                extent,
                self.context.queues[self.context.queue_families.graphics as usize],
            )?;

            if self.context.use_dynamic_rendering {
                self.context.device.cmd_end_rendering(frame.command_buffer);
            } else {
                self.context
                    .device
                    .cmd_end_render_pass(frame.command_buffer);
            }
            //transition the draw image and the swapchain image into their correct transfer layouts
            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.draw_image.image,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::TRANSFER,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::AccessFlags::TRANSFER_READ,
            );
            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.images[image_index as usize],
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            );

            // execute a copy from the draw image into the swapchain
            self.context.copy_image_to_image(
                frame.command_buffer,
                self.swapchain.draw_image.image,
                self.swapchain.images[image_index as usize],
                vk::Extent2D::default()
                    .width(self.swapchain.draw_image.extent.width)
                    .height(self.swapchain.draw_image.extent.height),
                self.swapchain.extent,
            );

            // set swapchain image layout to Present so we can show it on the screen
            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.images[image_index as usize],
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::PRESENT_SRC_KHR,
                vk::ImageAspectFlags::COLOR,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            );
            self.context
                .device
                .end_command_buffer(frame.command_buffer)?;

            let render_finished_semaphore = self.render_finished_semaphores[image_index as usize];

            if self.context.use_sync2 {
                self.context.device.queue_submit2(
                    self.context.queues[self.context.queue_families.graphics as usize],
                    &[vk::SubmitInfo2::default()
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(frame.image_available_semaphore)
                            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT_KHR)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(render_finished_semaphore)
                            .stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)])
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default()
                            .command_buffer(frame.command_buffer)])],
                    frame.in_flight_fence,
                )?;
            } else {
                let wait_semaphores = [frame.image_available_semaphore];
                let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                let signal_semaphores = [render_finished_semaphore];
                let command_buffers = [frame.command_buffer];

                let submit_info = vk::SubmitInfo::default()
                    .wait_semaphores(&wait_semaphores)
                    .wait_dst_stage_mask(&wait_stages)
                    .signal_semaphores(&signal_semaphores)
                    .command_buffers(&command_buffers);

                self.context.device.queue_submit(
                    self.context.queues[self.context.queue_families.graphics as usize],
                    &[submit_info],
                    frame.in_flight_fence,
                )?;
            }
            self.swapchain
                .present(image_index, render_finished_semaphore)?;

            self.frame_number += 1;

            Ok(())
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.context.device.device_wait_idle().unwrap();

            self.render_finished_semaphores
                .drain(..)
                .for_each(|semaphore| {
                    self.context.device.destroy_semaphore(semaphore, None);
                });

            self.frames.drain(..).for_each(|mut frame| {
                self.context
                    .device
                    .destroy_semaphore(frame.image_available_semaphore, None);
                self.context
                    .device
                    .destroy_fence(frame.in_flight_fence, None);
                self.context
                    .device
                    .free_command_buffers(frame.command_pool, &[frame.command_buffer]);
                self.context
                    .device
                    .destroy_command_pool(frame.command_pool, None);
                self.context.destroy_buffer(
                    &mut self.allocator.lock(),
                    &mut frame.toon_params_uniform_buffer,
                );
                self.context
                    .destroy_buffer(&mut self.allocator.lock(), &mut frame.frame_uniform_buffer);
                self.context
                    .device
                    .destroy_descriptor_set_layout(frame.character_descriptor_layout, None);
            });
            self.context
                .device
                .destroy_descriptor_set_layout(self.texture_descriptor_set_layout, None);

            self.global_descriptor_allocator
                .destroy_pool(&self.context.device);

            self.context.destroy_buffer(
                &mut self.allocator.lock(),
                &mut self.ring_staging_buffer.buffer,
            );
        }
    }
}
