use anyhow::{Result, anyhow, bail, ensure};
use ash::vk;
use parking_lot::Mutex;
use std::sync::Arc;
use tracing::info;

use gpu_allocator::{
    MemoryLocation,
    vulkan::{Allocator as GpuAllocator, AllocatorCreateDesc},
};

use crate::app::engine::{
    gpu::context::{AllocatedBuffer, RenderingContext},
    resources::data::ModelVertex,
};
use offset_allocator::{Allocation, Allocator as OffsetAllocator, ext::min_allocator_size};

pub const VERTEX_SIZE: u64 = std::mem::size_of::<ModelVertex>() as u64;
pub const INDEX_SIZE: u64 = std::mem::size_of::<u32>() as u64;
pub const VERTEX_PAGE_UNITS: u32 = 8 * 1024 * 1024;
pub const INDEX_PAGE_UNITS: u32 = 32 * 1024 * 1024;

pub struct BufferPage {
    unit_size: u64,
    address: Option<vk::DeviceAddress>,
    allocator: offset_allocator::Allocator,
    buffer: AllocatedBuffer,
    gpu_allocator: Arc<Mutex<GpuAllocator>>,
    context: Arc<RenderingContext>,
}

pub enum PoolKind {
    VERTEX,
    INDEX,
}

#[derive(Clone, Copy)]
pub struct PoolAlloc {
    pub page: u32,
    pub alloc: offset_allocator::Allocation,
    pub units: u32,
}

pub struct TypedPool {
    unit_size: u64,
    page_units: u32,
    pages: Vec<BufferPage>,
    allocator: Arc<Mutex<GpuAllocator>>,
    context: Arc<RenderingContext>,
}

pub struct GeometryPool {
    pub vertices: TypedPool,
    pub indices: TypedPool,
}

impl BufferPage {
    pub fn vertex(
        context: Arc<RenderingContext>,
        allocator: Arc<Mutex<GpuAllocator>>,
        unit_size: u64,
        capacity: u32,
    ) -> Result<Self> {
        let usage = if context.use_buffer_device_address {
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
        } else {
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER
        };

        Self::new(
            context,
            allocator,
            "Vertex page",
            unit_size,
            capacity,
            usage,
            true,
        )
    }

    pub fn index(
        context: Arc<RenderingContext>,
        allocator: Arc<Mutex<GpuAllocator>>,
        unit_size: u64,
        capacity: u32,
    ) -> Result<Self> {
        let usage = vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;
        Self::new(
            context,
            allocator,
            "Index page",
            unit_size,
            capacity,
            usage,
            false,
        )
    }

    fn new(
        context: Arc<RenderingContext>,
        gpu: Arc<Mutex<GpuAllocator>>,
        name: &str,
        unit_size: u64,
        capacity: u32,
        usage: vk::BufferUsageFlags,
        enable_address: bool,
    ) -> Result<Self> {
        info!("Allocating new geometry page {} with size {}", name, unit_size * capacity as u64);
        let buffer = context.create_buffer(
            name,
            &mut gpu.lock(),
            capacity as u64 * unit_size,
            usage,
            gpu_allocator::MemoryLocation::GpuOnly,
        )?;
        let address = unsafe {
            if context.use_buffer_device_address && enable_address {
                Some(context.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(buffer.buffer),
                ))
            } else {
                None
            }
        };
        Ok(Self {
            address,
            unit_size,
            allocator: OffsetAllocator::new(capacity),
            buffer,
            gpu_allocator: gpu,
            context,
        })
    }

    pub fn vk_buffer(&self) -> vk::Buffer {
        self.buffer.buffer
    }

    pub fn address(&self) -> Option<vk::DeviceAddress> {
        self.address
    }
}

impl Drop for BufferPage {
    fn drop(&mut self) {
        self.context
            .destroy_buffer(&mut self.gpu_allocator.lock(), &mut self.buffer);
    }
}

impl PoolAlloc {
    pub fn offset_units(&self) -> u32 {
        self.alloc.offset
    }
}

impl TypedPool {
    fn new(
        context: Arc<RenderingContext>,
        allocator: Arc<Mutex<GpuAllocator>>,
        unit_size: u64,
        page_units: u32,
    ) -> Self {
        Self {
            unit_size,
            page_units,
            pages: Vec::new(),
            allocator,
            context,
        }
    }

    pub fn alloc(&mut self, units: u32, kind: PoolKind) -> Result<PoolAlloc> {
        for (i, page) in self.pages.iter_mut().enumerate() {
            if let Some(a) = page.allocator.allocate(units) {
                return Ok(PoolAlloc {
                    page: i as u32,
                    alloc: a,
                    units,
                });
            }
        }
        // No room: add a page (oversized meshes get a dedicated page of exactly their size)
        let cap = units.max(self.page_units);
        let page = match kind {
            PoolKind::VERTEX => BufferPage::vertex(self.context.clone(), self.allocator.clone(), VERTEX_SIZE, cap)?,
            PoolKind::INDEX => BufferPage::index(self.context.clone(), self.allocator.clone(), INDEX_SIZE, cap)?,
        };
        self.pages.push(page);
        let i = self.pages.len() - 1;
        let a = self.pages[i]
            .allocator
            .allocate(units)
            .expect("fresh page must fit");
        Ok(PoolAlloc {
            page: i as u32,
            alloc: a,
            units,
        })
    }

    pub fn free(&mut self, a: PoolAlloc) {
        self.pages[a.page as usize].allocator.free(a.alloc);
    }

    pub fn byte_offset(&self, a: &PoolAlloc) -> u64 {
        a.alloc.offset as u64 * self.unit_size
    }

    pub fn get_page(&self, index: usize) -> &BufferPage {
        &self.pages[index]
    }
}

impl GeometryPool {
    pub fn new(
        context: Arc<RenderingContext>,
        allocator: Arc<Mutex<GpuAllocator>>,
    ) -> Result<Self> {
        Ok(Self {
            vertices: TypedPool::new(context.clone(), allocator.clone(), VERTEX_SIZE, VERTEX_PAGE_UNITS),
            indices: TypedPool::new(context, allocator, INDEX_SIZE, INDEX_PAGE_UNITS),
        })
    }
}
