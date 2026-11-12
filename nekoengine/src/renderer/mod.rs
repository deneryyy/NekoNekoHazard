use std::sync::Arc;

use anyhow::{Ok, Result};
use bevy_ecs::{
    component::Component,
    resource::Resource,
    system::{Query, Res, ResMut},
};
use glam::Mat4;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use parking_lot::Mutex;
use winit::window::Window;

use crate::{
    Assets, Time,
    gpu::{
        context::{AllocatedBuffer, RenderingContext},
        descriptor::{
            DescriptorAllocator, DescriptorLayoutBuilder, DescriptorWriter, PoolSizeRatio,
        },
        profiler::GpuProfiler,
        staging::RingStagingBuffer,
        sync::FRAME_OVERLAP,
    },
    renderer::{
        animation_data::{MAX_JOINTS, SkinDescriptors},
        camera::{Camera, CameraController, RenderCamera},
        character_renderer::CharacterRenderer,
        egui_renderer::EguiRenderer,
        shadow_renderer::ShadowRenderer,
        swapchain::Swapchain,
    },
    resources::{Resources, animation::Animator, pool::Handle, types::Model},
};
use ash::vk::{self, CommandBufferUsageFlags};

pub mod animation_data;
pub mod camera;
pub mod character_renderer;
pub mod egui_renderer;
pub mod shadow_renderer;
pub mod swapchain;

#[derive(Component, Clone, Copy)]
pub struct Transform {
    pub translation: glam::Vec3,
    pub rotation: glam::Quat,
    pub scale: glam::Vec3,
}

#[derive(Component, Clone, Copy)]
pub struct GlobalTransform(pub glam::Mat4);

#[derive(Component)]
pub struct ModelInstance(pub Handle<Model>);
#[derive(Component)]
pub struct Visibility(pub bool);

#[derive(Default)]
pub struct RenderFrame {
    pub camera: RenderCamera,
    pub light_pos: glam::Vec3,
    pub toon_ubo: ToonParamsUniformBufferObject,
    pub instances: Vec<RenderInstance>,
    pub joint_matrices: Vec<Mat4>,
}

pub struct RenderInstance {
    pub model: Handle<Model>,
    pub transform: Mat4,
    pub skin_offsets: Vec<u32>,
}

#[derive(Resource, Default)]
pub struct ExtractedFrame(pub RenderFrame);

pub struct Frame {
    frame_set: vk::DescriptorSet,
    frame_set_layout: vk::DescriptorSetLayout,
    joint_buffer: AllocatedBuffer,
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
    light_space_matrix: glam::Mat4,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct ToonParamsUniformBufferObject {
    pub shadow_tint: glam::Vec4,
    pub light_color: glam::Vec4,
    pub step_threshold: f32,
    pub step_softness: f32,
    pub rim_power: f32,
    pub rim_strength: f32,
    pub rim_edge_0: f32,
    pub rim_edge_1: f32,
    pub spec_threshold: f32,
    pub shininess: f32,
    pub spec_strength: f32,
    pub _pad: f32,
}

pub struct Renderer {
    shadow_renderer: ShadowRenderer,
    pub egui_renderer: EguiRenderer,
    character_renderer: CharacterRenderer,
    gpu_profiler: GpuProfiler,
    texture_descriptors: vk::DescriptorSet,
    texture_descriptor_set_layout: vk::DescriptorSetLayout,
    ring_staging_buffer: RingStagingBuffer,
    allocator: Arc<Mutex<Allocator>>,
    skin_descriptors: SkinDescriptors,
    global_descriptor_allocator: DescriptorAllocator,
    render_finished_semaphores: Vec<vk::Semaphore>,
    frame_number: usize,
    frames: Vec<Frame>,
    swapchain: Swapchain,
    resources: Arc<Mutex<Resources>>,
    context: Arc<RenderingContext>,
}

impl Renderer {
    pub fn new(
        context: Arc<RenderingContext>,
        window: Arc<Window>,
        resources: Arc<Mutex<Resources>>,
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

        let ratios = Vec::from([
            PoolSizeRatio {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                per_set: 2.0,
            },
            PoolSizeRatio {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                per_set: MAX_JOINTS as f32,
            },
        ]);
        let mut global_descriptor_allocator = DescriptorAllocator::new(
            ratios,
            FRAME_OVERLAP as u32,
            vk::DescriptorPoolCreateFlags::empty(),
        );

        let skin_descriptors = SkinDescriptors::new(context.clone(), resources.clone())?;

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

                let joint_buffer_size =
                    MAX_JOINTS as u64 * std::mem::size_of::<glam::Mat4>() as u64;
                let joint_buffer = context.create_buffer(
                    "Joint Buffer",
                    &mut allocator.lock(),
                    joint_buffer_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                )?;

                let frame_set_layout_builder = DescriptorLayoutBuilder::new()
                    .binding(
                        0,
                        vk::DescriptorType::UNIFORM_BUFFER,
                        1,
                        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        vk::DescriptorBindingFlags::empty(),
                    )
                    .binding(
                        1,
                        vk::DescriptorType::UNIFORM_BUFFER,
                        1,
                        vk::ShaderStageFlags::FRAGMENT,
                        vk::DescriptorBindingFlags::empty(),
                    )
                    .binding(
                        2,
                        vk::DescriptorType::STORAGE_BUFFER,
                        MAX_JOINTS,
                        vk::ShaderStageFlags::VERTEX,
                        vk::DescriptorBindingFlags::empty(),
                    );

                let frame_set_layout = frame_set_layout_builder.build(&context.device, false)?;

                let frame_set = global_descriptor_allocator.allocate(
                    &context.device,
                    frame_set_layout,
                    None,
                )?;
                DescriptorWriter::new()
                    .buffer(
                        0,
                        vk::DescriptorType::UNIFORM_BUFFER,
                        frame_uniform_buffer.buffer,
                        0,
                        std::mem::size_of::<FrameUniformBufferObject>() as u64,
                    )
                    .buffer(
                        1,
                        vk::DescriptorType::UNIFORM_BUFFER,
                        toon_params_uniform_buffer.buffer,
                        0,
                        std::mem::size_of::<ToonParamsUniformBufferObject>() as u64,
                    )
                    .buffer(
                        2,
                        vk::DescriptorType::STORAGE_BUFFER,
                        joint_buffer.buffer,
                        0,
                        vk::WHOLE_SIZE,
                    )
                    .update(&context.device, frame_set);
                frames.push(Frame {
                    frame_set,
                    frame_set_layout,
                    joint_buffer,
                    toon_params_uniform_buffer,
                    frame_uniform_buffer,
                    image_available_semaphore,
                    in_flight_fence,
                    command_buffer,
                    command_pool,
                });
            }

            let egui_renderer = EguiRenderer::new(window.clone(), context.clone(), &swapchain)?;

            let frame_set_layouts = frames
                .iter()
                .map(|frame| return frame.frame_set_layout)
                .collect::<Vec<_>>();

            let (texture_set, texture_set_layout) = {
                let res = resources.lock();
                (res.bindless.set, res.bindless.layout)
            };

            let shadow_renderer = ShadowRenderer::new(
                context.clone(),
                allocator.clone(),
                resources.clone(),
                &frame_set_layouts,
                skin_descriptors.skin_set_layout,
                skin_descriptors.skin_set,
            )?;

            Ok(Self {
                shadow_renderer,
                egui_renderer,
                character_renderer: CharacterRenderer::new(
                    context.clone(),
                    allocator.clone(),
                    resources.clone(),
                    &swapchain,
                    &frame_set_layouts,
                    skin_descriptors.skin_set_layout,
                    skin_descriptors.skin_set,
                )?,
                gpu_profiler: GpuProfiler::new(&context, FRAME_OVERLAP)?,
                texture_descriptors: texture_set,
                texture_descriptor_set_layout: texture_set_layout,
                ring_staging_buffer: RingStagingBuffer::new(
                    context.clone(),
                    &mut allocator.lock(),
                    64 * 1024 * 1024,
                )?,
                allocator: allocator.clone(),
                skin_descriptors,
                global_descriptor_allocator,
                render_finished_semaphores,
                frame_number: 0,
                frames,
                swapchain,
                resources,
                context,
            })
        }
    }

    pub fn resize(&mut self) -> Result<()> {
        self.swapchain.resize()
    }

    pub fn extract_instances(
        q: Query<(
            &GlobalTransform,
            &ModelInstance,
            &Visibility,
            Option<&Animator>,
        )>,
        assets: Res<Assets>,
        mut out: ResMut<ExtractedFrame>,
    ) {
        let _s = tracy_client::span!("extract_instances");
        let res = assets.assets.lock();
        let frame = &mut out.0;
        frame.instances.clear();
        frame.joint_matrices.clear();

        for (gt, model_instance, vis, anim) in &q {
            if !vis.0 {
                continue;
            }
            let Some(model) = res.models.get(model_instance.0) else {
                continue;
            };

            let mut offsets = vec![u32::MAX; model.skins.len()];
            if let Some(anim) = anim {
                for (s, skin) in model.skins.iter().enumerate() {
                    let n = skin.joints.len();
                    let start = frame.joint_matrices.len();
                    if start + n > MAX_JOINTS as usize {
                        continue; // buffer full: this skin draws in bind pose
                    }
                    frame.joint_matrices.resize(start + n, Mat4::IDENTITY);
                    anim.write_skin(skin, &mut frame.joint_matrices[start..]);
                    offsets[s] = start as u32;
                }
            }
            frame.instances.push(RenderInstance {
                model: model_instance.0,
                transform: gt.0,
                skin_offsets: offsets,
            });
        }
    }

    pub fn extract_camera(q: Query<(&Camera, &Transform)>, mut out: ResMut<ExtractedFrame>) {
        let _s = tracy_client::span!("extract_camera");
        for (cam, transform) in &q {
            let camera = &mut out.0.camera;
            let (yaw, pitch, roll) = transform.rotation.to_euler(glam::EulerRot::YXZ);
            camera.yaw = yaw.to_degrees();
            camera.pitch = pitch.to_degrees();
            camera.roll = roll.to_degrees();
            camera.position = transform.translation;
            camera.fov_y = cam.fov_y;
            camera.near = cam.near;
            camera.far = cam.far;
            camera.update_camera_vectors();
        }
    }

    pub fn advance_animators(
        mut q: Query<(&ModelInstance, &mut Animator)>,
        assets: Res<Assets>,
        time: Res<Time>,
    ) {
        let _s = tracy_client::span!("advance_animators");
        let res = assets.assets.lock();
        for (mi, mut anim) in &mut q {
            let Some(model) = res.models.get(mi.0) else {
                continue;
            };
            let a = &mut *anim;
            a.update(time.dt, &model.skeleton, model.clips.get(a.clip));
        }
    }

    fn upload_uniform_buffer(render_frame: &RenderFrame, swapchain: &Swapchain, frame: &Frame) {
        let _s = tracy_client::span!("upload_uniform_buffer");
        unsafe {
            let mut proj = glam::Mat4::perspective_rh(
                render_frame.camera.fov_y.to_radians(),
                swapchain.extent.width as f32 / swapchain.extent.height as f32,
                0.1,
                1000.0,
            );
            proj.y_axis.y *= -1.0;

            let view = render_frame.camera.get_view_matrix();

            let mut light_proj = glam::Mat4::orthographic_rh(-10.0, 10.0, -10.0, 10.0, 1.0, 7.5);
            light_proj.y_axis.y *= -1.0;
            let light_view = glam::Mat4::look_at_rh(
                render_frame.light_pos,
                glam::Vec3::new(0.0, 0.0, 0.0),
                glam::Vec3::new(0.0, 1.0, 0.0),
            );

            let ubo = FrameUniformBufferObject {
                view,
                proj,
                light_pos: render_frame.light_pos.extend(0.0),
                cam_pos: render_frame.camera.position.extend(0.0),
                light_space_matrix: light_proj * light_view,
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
            std::ptr::copy_nonoverlapping(&render_frame.toon_ubo, toon_ubo_ptr.cast(), 1);
        }
    }

    pub fn render(&mut self, render_frame: &RenderFrame, dt: f32) -> Result<()> {
        unsafe {
            let frame_index = self.frame_number % FRAME_OVERLAP;
            let frame = &self.frames[frame_index];

            {
                let _s = tracy_client::span!("wait_for_fences");
                // Wait until gpu finished rendering last frame
                self.context
                    .device
                    .wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;
            }
            self.ring_staging_buffer.retire_frame(frame_index);
            {
                let _s = tracy_client::span!("bindless collect");
                let mut res = self.resources.lock();
                res.bindless
                    .collect(self.frame_number as u64, FRAME_OVERLAP as u64);
            }

            {
                let _s = tracy_client::span!("joint buffer copy");
                let joints: &mut [Mat4] = {
                    let ptr =
                        frame.joint_buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut Mat4;
                    std::slice::from_raw_parts_mut(ptr, MAX_JOINTS as usize)
                };
                let n = render_frame.joint_matrices.len().min(joints.len());
                joints[..n].copy_from_slice(&render_frame.joint_matrices[..n]);
            }
            self.gpu_profiler
                .collect(&self.context.device, frame_index)?;
            self.gpu_profiler.sync();

            if self.swapchain.check_extent().is_err() {
                return Ok(());
            }

            if self.swapchain.is_dirty {
                self.swapchain.resize()?;
                self.swapchain.is_dirty = false;
                return Ok(());
            }

            let Some(image_index) = self
                .swapchain
                .acquire_next_image(frame.image_available_semaphore)?
            else {
                return Ok(());
            };

            self.context.device.reset_fences(&[frame.in_flight_fence])?;

            self.context
                .device
                .reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())?;

            self.context.device.begin_command_buffer(
                frame.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            self.gpu_profiler
                .begin_frame(&self.context.device, frame.command_buffer, frame_index);

            Self::upload_uniform_buffer(&render_frame, &self.swapchain, frame);

            self.shadow_renderer.record_commands(frame, &render_frame);

            self.context.transition_image_layout(
                frame.command_buffer,
                self.swapchain.draw_image.image,
                0,
                vk::REMAINING_MIP_LEVELS,
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
                0,
                vk::REMAINING_MIP_LEVELS,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                vk::ImageAspectFlags::DEPTH,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            );

            let extent = vk::Extent2D::default()
                .width(self.swapchain.extent.width)
                .height(self.swapchain.extent.height);

            self.context.begin_rendering(
                frame.command_buffer,
                Some(self.swapchain.draw_image.image_view),
                Some(self.swapchain.depth_image.image_view),
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

            let z = self.gpu_profiler.begin_zone(
                &self.context.device,
                frame.command_buffer,
                frame_index,
                tracy_client::span_location!("characters"),
            );
            self.character_renderer.record_commands(
                frame,
                &self.frames,
                extent,
                &self.swapchain,
                self.texture_descriptors,
                render_frame,
            );
            self.gpu_profiler
                .end_zone(&self.context.device, frame.command_buffer, frame_index, z);

            let z = self.gpu_profiler.begin_zone(
                &self.context.device,
                frame.command_buffer,
                frame_index,
                tracy_client::span_location!("egui"),
            );
            self.egui_renderer.record_commands(
                frame,
                extent,
                self.context.queues[self.context.queue_families.graphics as usize],
            )?;
            self.gpu_profiler
                .end_zone(&self.context.device, frame.command_buffer, frame_index, z);

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
                0,
                vk::REMAINING_MIP_LEVELS,
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
                0,
                vk::REMAINING_MIP_LEVELS,
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
                0,
                vk::REMAINING_MIP_LEVELS,
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

            self.ring_staging_buffer.end_frame(frame_index);
            {
                let _s = tracy_client::span!("queue_submit");
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
            }
            {
                let _s = tracy_client::span!("present");
                self.swapchain
                    .present(image_index, render_finished_semaphore)?;
            }

            self.frame_number += 1;
            tracy_client::frame_mark();

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
                self.context
                    .destroy_buffer(&mut self.allocator.lock(), &mut frame.joint_buffer);
                self.context.destroy_buffer(
                    &mut self.allocator.lock(),
                    &mut frame.toon_params_uniform_buffer,
                );
                self.context
                    .destroy_buffer(&mut self.allocator.lock(), &mut frame.frame_uniform_buffer);
                self.context
                    .device
                    .destroy_descriptor_set_layout(frame.frame_set_layout, None);
            });

            self.gpu_profiler.destroy(&self.context.device);

            self.global_descriptor_allocator
                .destroy(&self.context.device);

            self.context.destroy_buffer(
                &mut self.allocator.lock(),
                &mut self.ring_staging_buffer.buffer,
            );
        }
    }
}
