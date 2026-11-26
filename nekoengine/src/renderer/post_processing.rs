use std::sync::Arc;

use anyhow::Result;
use ash::vk;
use shaderc::ShaderKind;

use crate::{
    gpu::{context::RenderingContext, descriptor::DescriptorLayoutBuilder},
    renderer::Frame,
    resources,
};

pub struct PostProcessing {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    context: Arc<RenderingContext>,
}

struct ShaderConstants {
    samples: u32,
}

impl PostProcessing {
    pub fn new(context: Arc<RenderingContext>) -> Result<Self> {
        let post_process_layout_builder = DescriptorLayoutBuilder::new()
            .binding(
                0,
                vk::DescriptorType::SAMPLED_IMAGE,
                1,
                vk::ShaderStageFlags::COMPUTE,
                vk::DescriptorBindingFlags::empty(),
            )
            .binding(
                1,
                vk::DescriptorType::STORAGE_IMAGE,
                1,
                vk::ShaderStageFlags::COMPUTE,
                vk::DescriptorBindingFlags::empty(),
            );
        let post_process_set_layout = post_process_layout_builder.build(&context.device, false)?;

        let post_process_shader = resources::load_shader_module(
            include_str!(concat!(env!("OUT_DIR"), "/res/shaders/post_process.comp")),
            "post_process.comp",
            ShaderKind::Compute,
            &context.device,
        )?;

        let mut descriptors = Vec::new();
        descriptors.push(post_process_set_layout);
        let layout = unsafe {
            context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptors),
                None,
            )?
        };
        let spec_data = ShaderConstants { samples: context.msaa_samples.1 };
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
            pipeline: pipelines[0],
            layout,
            context,
        })
    }

    pub fn record_commands(&self, frame: &Frame) {
        let _main = tracy_client::span!("post processing record commands");
        let device = &self.context.device;
        let cb = frame.command_buffer;
        let mem = |src, dst, src_stage, dst_stage| unsafe {
            device.cmd_pipeline_barrier(
                cb,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(src)
                    .dst_access_mask(dst)],
                &[],
                &[],
            );
        };
    }
}

impl Drop for PostProcessing {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_pipeline_layout(self.layout, None);
            self.context.device.destroy_pipeline(self.pipeline, None);
        }
    }
}
