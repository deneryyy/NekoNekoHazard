use std::sync::Arc;

use anyhow::{Context, Ok, Result};
use ash::vk;
use gpu_allocator::vulkan::Allocator;

use crate::app::engine::gpu::{
    context::{AllocatedBuffer, RenderingContext},
    sync::FRAME_OVERLAP,
};

pub struct RingStagingBuffer {
    pub buffer: AllocatedBuffer,
    size: u64,
    head: u64,
    tail: u64,
    frame_end: [u64; FRAME_OVERLAP],
}

fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) & !(a - 1)
}

impl RingStagingBuffer {
    pub fn new(
        context: Arc<RenderingContext>,
        allocator: &mut Allocator,
        size: u64,
    ) -> Result<Self> {
        let buffer = context.create_buffer(
            "Ring staging buffer",
            allocator,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            gpu_allocator::MemoryLocation::CpuToGpu,
        )?;
        Ok(Self {
            buffer,
            size,
            head: 0,
            tail: 0,
            frame_end: [0; FRAME_OVERLAP],
        })
    }

    pub fn try_alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        if size >= self.size {
            return None;
        }

        let mut start = align_up(self.head, align);
        let phys = start % self.size;
        if phys + size > self.size {
            start += self.size - phys; // skip the tail, restart at offset 0
        }
        let end = start + size;
        if end - self.tail >= self.size {
            return None;
        }

        self.head = end;
        Some(start % self.size)
    }

    pub fn capacity(&self) -> u64 {
        self.size
    }

    pub fn write<T: Copy>(&mut self, data: &[T], align: u64) -> Result<u64> {
        let bytes = std::mem::size_of_val(data) as u64;
        let offset = self
            .try_alloc(bytes, align)
            .context("Failed to allocate ring buffer for writing!")?;
        self.write_at(offset, data);
        Ok(offset)
    }

    pub fn write_at<T: Copy>(&mut self, offset: u64, data: &[T]) {
        let bytes = std::mem::size_of_val(data) as u64;
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr() as *const u8,
                self.buffer
                    .allocation
                    .mapped_ptr()
                    .unwrap()
                    .as_ptr()
                    .cast::<u8>()
                    .add(offset as usize),
                bytes as usize,
            );
        }
    }

    pub fn end_frame(&mut self, frame: usize) {
        self.frame_end[frame] = self.head;
    }

    pub fn retire_frame(&mut self, frame: usize) {
        self.tail = self.tail.max(self.frame_end[frame]);
    }

    pub fn reset(&mut self) {
        self.head = 0;
        self.tail = 0;
    }
}
