use std::sync::Arc;

use anyhow::Result;
use ash::vk;
use glam::Mat4;
use parking_lot::Mutex;

use crate::app::engine::{
    gpu::{
        context::{AllocatedBuffer, RenderingContext}, descriptor::{
            DescriptorAllocator, DescriptorLayoutBuilder, DescriptorWriter, PoolSizeRatio,
        },
    }, resources::Resources,
};

pub const MAX_JOINTS: u32 = 4096;

pub struct SkinDescriptors {
    pub skin_set_layout: vk::DescriptorSetLayout,
    pub skin_set: vk::DescriptorSet,
    descriptor_allocator: DescriptorAllocator,
    context: Arc<RenderingContext>,
}

impl SkinDescriptors {
    pub fn new(context: Arc<RenderingContext>, resources: Arc<Mutex<Resources>>) -> Result<Self> {
        let skin_buffer_count = {
            let res = resources.lock();
            (res.geometry().skins.get_page_count() as u32).min(
                context
                    .physical_device
                    .vulkan12_properties
                    .max_descriptor_set_update_after_bind_storage_buffers,
            )
        };
        let ratios = Vec::from([PoolSizeRatio {
            ty: vk::DescriptorType::STORAGE_BUFFER,
            per_set: 1.0,
        }]);
        let mut descriptor_allocator = DescriptorAllocator::new(
            ratios,
            skin_buffer_count,
            vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND,
        );

        let skin_set_layout = DescriptorLayoutBuilder::new()
            .binding(
                0,
                vk::DescriptorType::STORAGE_BUFFER,
                skin_buffer_count,
                vk::ShaderStageFlags::VERTEX,
                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                    | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
            )
            .build(&context.device)?;
        let skin_set = descriptor_allocator.allocate(
            &context.device,
            skin_set_layout,
            Some(skin_buffer_count),
        )?;
        {
            let res = resources.lock();
            let mut writes = DescriptorWriter::new();
            for page in res.geometry().skins.iter() {
                writes.buffer(
                    0,
                    vk::DescriptorType::STORAGE_BUFFER,
                    page.vk_buffer(),
                    0,
                    page.alloc_size(),
                );
            }

            writes.update(&context.device, skin_set);
        }
        Ok(Self {
            skin_set_layout,
            skin_set,
            descriptor_allocator,
            context,
        })
    }
}

impl Drop for SkinDescriptors {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_descriptor_set_layout(self.skin_set_layout, None);
        }
        self.descriptor_allocator.destroy(&self.context.device);
    }
}