use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::vulkan::Allocator;
use parking_lot::Mutex;
use std::sync::Arc;

use crate::app::engine::gpu::{
    context::{AllocatedImage, RenderingContext},
    staging::RingStagingBuffer,
};
use tracing::info;

pub struct Uploader {
    recording: bool,
    fence: vk::Fence,
    cmd: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    ring: RingStagingBuffer,
    allocator: Arc<Mutex<Allocator>>,
    context: Arc<RenderingContext>,
}

const IMAGE_COPY_ALIGN: u64 = 16;

impl Uploader {
    pub fn new(context: Arc<RenderingContext>, allocator: Arc<Mutex<Allocator>>) -> Result<Self> {
        unsafe {
            let command_pool = context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(context.queue_families.graphics)
                    .flags(
                        vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                            | vk::CommandPoolCreateFlags::TRANSIENT,
                    ),
                None,
            )?;

            let &cmd = context
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

            let fence = context.device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::empty()),
                None,
            )?;

            Ok(Self {
                recording: false,
                fence,
                cmd,
                command_pool,
                ring: RingStagingBuffer::new(
                    context.clone(),
                    &mut allocator.clone().lock(),
                    256 * 1024 * 1024,
                )?,
                allocator,
                context,
            })
        }
    }

    fn ensure_recording(&mut self) -> Result<()> {
        if !self.recording {
            let info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { self.context.device.begin_command_buffer(self.cmd, &info)? };
            self.recording = true;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<()> {
        if !self.recording {
            return Ok(());
        }

        unsafe {
            self.context.device.end_command_buffer(self.cmd)?;

            if self.context.use_sync2 {
                self.context.device.queue_submit2(
                    self.context.queues[self.context.queue_families.graphics as usize],
                    &[vk::SubmitInfo2::default().command_buffer_infos(&[
                        vk::CommandBufferSubmitInfo::default().command_buffer(self.cmd),
                    ])],
                    self.fence,
                )?;
            } else {
                let submit_info =
                    vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.cmd));

                self.context.device.queue_submit(
                    self.context.queues[self.context.queue_families.graphics as usize],
                    &[submit_info],
                    self.fence,
                )?;
            }
            self.context
                .device
                .wait_for_fences(&[self.fence], true, u64::MAX)?;

            self.context.device.reset_fences(&[self.fence])?;
            self.context
                .device
                .reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
            self.recording = false;

            self.ring.reset();
        }
        Ok(())
    }

    pub fn upload_buffer(&mut self, dst: vk::Buffer, dst_offset: u64, data: &[u8]) -> Result<()> {
        info!("uploading buffer, size: {}", data.len());
        let mut done = 0usize;
        while done < data.len() {
            self.ensure_recording()?;
            let chunk = (data.len() - done).min(self.ring.capacity() as usize);
            let Some(src_off) = self.ring.try_alloc(chunk as u64, 16) else {
                self.flush()?;
                continue;
            };
            self.ring.write_at(src_off, &data[done..done + chunk]);
            let region = vk::BufferCopy {
                src_offset: src_off,
                dst_offset: dst_offset + done as u64,
                size: chunk as u64,
            };
            unsafe {
                self.context.device.cmd_copy_buffer(
                    self.cmd,
                    self.ring.buffer.buffer,
                    dst,
                    &[region],
                )
            };
            done += chunk;
        }
        Ok(())
    }

    pub fn upload_image(
        &mut self,
        rgba: &[u8],
        width: u32,
        height: u32,
        srgb: bool,
    ) -> Result<AllocatedImage> {
        info!("uploading image, size: {}", rgba.len());
        let len = rgba.len() as u64;
        let src_off = match self.ring.try_alloc(len, IMAGE_COPY_ALIGN) {
            Some(o) => o,
            None => {
                self.flush()?;
                self.ring
                    .try_alloc(len, IMAGE_COPY_ALIGN)
                    .context("image is larger than the staging ring")?
            }
        };
        self.ring.write_at(src_off, rgba);
        self.ensure_recording()?;
        let format = if srgb {
            vk::Format::R8G8B8A8_SRGB
        } else {
            vk::Format::R8G8B8A8_UNORM
        };
        let extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let image = self.context.create_image(
            &mut self.allocator.lock(),
            extent,
            format,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::COLOR,
        )?;
        unsafe {
            self.context.transition_image_layout(
                self.cmd,
                image.image,
                0,
                vk::REMAINING_MIP_LEVELS,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ,
            );
            self.context.device.cmd_copy_buffer_to_image(
                self.cmd,
                self.ring.buffer.buffer,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .buffer_offset(src_off)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(extent)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .base_array_layer(0)
                            .layer_count(1),
                    )],
            );
            self.context.transition_image_layout(
                self.cmd,
                image.image,
                0,
                vk::REMAINING_MIP_LEVELS,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::ImageAspectFlags::COLOR,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_WRITE,
                vk::AccessFlags::MEMORY_WRITE | vk::AccessFlags::MEMORY_READ,
            );
        }

        Ok(image)
    }
}

impl Drop for Uploader {
    fn drop(&mut self) {
        unsafe {
            self.context.destroy_buffer(
                &mut self.allocator.lock(),
                &mut self.ring.buffer,
            );
            self.context.device.destroy_fence(self.fence, None);
            self.context.device.destroy_command_pool(self.command_pool, None);
        }
    }
}
