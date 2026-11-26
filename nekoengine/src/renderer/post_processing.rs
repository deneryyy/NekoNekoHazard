use std::sync::Arc;

use anyhow::Result;
use ash::vk;
use shaderc::ShaderKind;

use crate::{
    gpu::{
        context::RenderingContext,
        descriptor::{
            DescriptorAllocator, DescriptorLayoutBuilder, DescriptorWriter, PoolSizeRatio,
        },
    },
    renderer::{Frame, swapchain::Swapchain},
    resources,
};

pub struct PostProcessing {
    descriptor_allocator: DescriptorAllocator,
    post_process_set: vk::DescriptorSet,
    post_process_set_layout: vk::DescriptorSetLayout,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    context: Arc<RenderingContext>,
}

#[repr(C)]
struct ShaderConstants {
    samples: u32,
}

#[repr(C, align(16))]
struct PostProcessPushConstants {
    exposure: f32,
}

impl PostProcessing {
    pub fn new(context: Arc<RenderingContext>, swapchain: &Swapchain) -> Result<Self> {
        let ratios = Vec::from([
            PoolSizeRatio {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                per_set: 1.0,
            },
            PoolSizeRatio {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                per_set: 1.0,
            },
        ]);
        let mut descriptor_allocator =
            DescriptorAllocator::new(ratios, 1, vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
        let post_process_layout_builder = DescriptorLayoutBuilder::new()
            .binding(
                0,
                vk::DescriptorType::SAMPLED_IMAGE,
                1,
                vk::ShaderStageFlags::COMPUTE,
                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            )
            .binding(
                1,
                vk::DescriptorType::STORAGE_IMAGE,
                1,
                vk::ShaderStageFlags::COMPUTE,
                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
            );
        let post_process_set_layout = post_process_layout_builder.build(&context.device, false)?;
        let post_process_set =
            descriptor_allocator.allocate(&context.device, post_process_set_layout, None)?;
        DescriptorWriter::new()
            .image(
                0,
                0,
                vk::DescriptorType::SAMPLED_IMAGE,
                swapchain.msaa_draw_image.image_view,
                vk::Sampler::null(),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )
            .image(
                1,
                0,
                vk::DescriptorType::STORAGE_IMAGE,
                swapchain.resolve_image.image_view,
                vk::Sampler::null(),
                vk::ImageLayout::GENERAL,
            )
            .update(&context.device, post_process_set);

        let post_process_shader = resources::load_shader_module(
            include_str!(concat!(env!("OUT_DIR"), "/res/shaders/post_process.comp")),
            "post_process.comp",
            ShaderKind::Compute,
            &context.device,
        )?;

        let mut descriptors = Vec::new();
        descriptors.push(post_process_set_layout);
        let push_constant_range = vk::PushConstantRange::default()
            .offset(0)
            .size(std::mem::size_of::<PostProcessPushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let layout = unsafe {
            context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&descriptors)
                    .push_constant_ranges(&[push_constant_range]),
                None,
            )?
        };
        let spec_data = ShaderConstants {
            samples: context.msaa_samples.1,
        };
        let map_entries = [vk::SpecializationMapEntry::default()
            .constant_id(0)
            .offset(std::mem::offset_of!(ShaderConstants, samples) as u32)
            .size(size_of::<u32>())];
        let data_bytes = unsafe {
            std::slice::from_raw_parts(
                (&spec_data as *const ShaderConstants) as *const u8,
                size_of::<ShaderConstants>(),
            )
        };
        let pipelines = unsafe {
            context
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .layout(layout)
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .specialization_info(
                                    &vk::SpecializationInfo::default()
                                        .map_entries(&map_entries)
                                        .data(data_bytes),
                                )
                                .module(post_process_shader)
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .name(c"main"),
                        )],
                    None,
                )
                .map_err(|(_, e)| e)?
        };
        unsafe {
            context
                .device
                .destroy_shader_module(post_process_shader, None);
        }
        Ok(Self {
            descriptor_allocator,
            post_process_set,
            post_process_set_layout,
            pipeline: pipelines[0],
            layout,
            context,
        })
    }

    pub fn resize(&self, swapchain: &Swapchain) {
        DescriptorWriter::new()
            .image(
                0,
                0,
                vk::DescriptorType::SAMPLED_IMAGE,
                swapchain.msaa_draw_image.image_view,
                vk::Sampler::null(),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            )
            .image(
                1,
                0,
                vk::DescriptorType::STORAGE_IMAGE,
                swapchain.resolve_image.image_view,
                vk::Sampler::null(),
                vk::ImageLayout::GENERAL,
            )
            .update(&self.context.device, self.post_process_set);
    }

    pub fn record_commands(&self, frame: &Frame, swapchain: &Swapchain) {
        let _main = tracy_client::span!("post processing record commands");
        let device = &self.context.device;
        let cb = frame.command_buffer;
        unsafe {
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.post_process_set],
                &[],
            );
            let post_process_push_constants = PostProcessPushConstants { exposure: 1.0 };
            device.cmd_push_constants(
                cb,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                std::slice::from_raw_parts(
                    &post_process_push_constants as *const _ as *const u8,
                    std::mem::size_of::<PostProcessPushConstants>(),
                ),
            );

            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_dispatch(
                cb,
                swapchain.extent.width.div_ceil(8),
                swapchain.extent.height.div_ceil(8),
                1,
            );
        }
    }
}

impl Drop for PostProcessing {
    fn drop(&mut self) {
        unsafe {
            self.descriptor_allocator.destroy(&self.context.device);
            self.context
                .device
                .destroy_descriptor_set_layout(self.post_process_set_layout, None);
            self.context
                .device
                .destroy_pipeline_layout(self.layout, None);
            self.context.device.destroy_pipeline(self.pipeline, None);
        }
    }
}
