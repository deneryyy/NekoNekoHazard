use anyhow::Result;
use ash::vk;
use std::sync::Arc;

use crate::app::engine::gpu::{context::RenderingContext, descriptor::DescriptorLayoutBuilder};

pub struct BindlessTextures {
    pub layout: vk::DescriptorSetLayout,
    pub set: vk::DescriptorSet,
    pool: vk::DescriptorPool,
    sampler: vk::Sampler,
    capacity: u32,
    next: u32,
    free: Vec<u32>,
    /// (frame_when_released, slot)
    pending: Vec<(u64, u32)>,
    context: Arc<RenderingContext>,
}

impl BindlessTextures {
    pub fn new(context: Arc<RenderingContext>, want: u32) -> Result<Self> {
        let capacity = want.min(context.physical_device.vulkan12_properties.max_descriptor_set_update_after_bind_sampled_images);

        let uab = vk::DescriptorBindingFlags::UPDATE_AFTER_BIND;
        let layout = DescriptorLayoutBuilder::new()
            .binding(0, vk::DescriptorType::SAMPLER, 1,
                vk::ShaderStageFlags::ALL, uab)
            .binding(1, vk::DescriptorType::SAMPLED_IMAGE, capacity,
                vk::ShaderStageFlags::ALL,
                uab | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                    | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT)
            .build(&context.device)?;

        let sizes = [
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLER, descriptor_count: 1 },
            vk::DescriptorPoolSize { ty: vk::DescriptorType::SAMPLED_IMAGE, descriptor_count: capacity },
        ];
        let pool = unsafe {
            context.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?
        };

        let layouts = [layout];
        let counts = [capacity];
        let mut var = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&counts);
        let set = unsafe {
            context.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts)
                    .push_next(&mut var),
            )?[0]
        };

        let sampler = unsafe {
            context.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::REPEAT)
                    .address_mode_v(vk::SamplerAddressMode::REPEAT)
                    .address_mode_w(vk::SamplerAddressMode::REPEAT)
                    .max_lod(vk::LOD_CLAMP_NONE)
                    .anisotropy_enable(true)
                    .max_anisotropy(8.0),
                None,
            )?
        };
        let info = [vk::DescriptorImageInfo { sampler, ..Default::default() }];
        unsafe {
            context.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(set).dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&info)],
                &[],
            );
        }

        Ok(Self { layout, set, pool, sampler, capacity, next: 0, free: vec![], pending: vec![], context })
    }

    pub fn register(&mut self, device: &ash::Device, view: vk::ImageView) -> Result<u32> {
        let slot = self.free.pop()
            .or_else(|| (self.next < self.capacity).then(|| { self.next += 1; self.next - 1 }))
            .ok_or_else(|| anyhow::anyhow!("bindless texture table full"))?;

        let info = [vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        unsafe {
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(self.set).dst_binding(1).dst_array_element(slot)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&info)],
                &[],
            );
        }
        Ok(slot)
    }

    /// Schedule a slot for reuse once the GPU can no longer reference it.
    pub fn release(&mut self, slot: u32, current_frame: u64) {
        self.pending.push((current_frame, slot));
    }

    /// Call once per frame after the fence wait. `frames_in_flight` is typically 2 or 3.
    pub fn collect(&mut self, current_frame: u64, frames_in_flight: u64) {
        let free = &mut self.free;
        self.pending.retain(|&(f, s)| {
            if current_frame >= f + frames_in_flight { free.push(s); false } else { true }
        });
    }
}

impl Drop for BindlessTextures {
    fn drop(&mut self) {
        unsafe {
            self.context.device.destroy_sampler(self.sampler, None);
            self.context.device.destroy_descriptor_pool(self.pool, None);
            self.context.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}