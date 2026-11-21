use std::sync::Arc;

use anyhow::Result;
use ash::vk;
use bevy_ecs::component::Component;
use glam::{Mat4, UVec3, UVec4, Vec2, Vec3, Vec4};
use shaderc::ShaderKind;

use crate::{gpu::context::RenderingContext, renderer::Frame, resources};

pub const MAX_LIGHTS: usize = 1024;
pub const CLUSTER_GRID: UVec3 = UVec3::new(16, 9, 24);
pub const CLUSTER_COUNT: usize = (16 * 9 * 24) as usize;
pub const MAX_LIGHTS_PER_CLUSTER: usize = 64;

#[derive(Component)]
pub struct DirectionalLight {
    pub color: Vec3,
    pub intensity: f32,
}

#[derive(Component)]
pub struct PointLight {
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
}

#[derive(Component)]
pub struct SpotLight {
    pub color: Vec3,
    pub intensity: f32,
    pub range: f32,
    pub inner_angle: f32,
    pub outer_angle: f32,
}

pub struct ClusterCuller {
    cull_pipeline: vk::Pipeline,
    build_pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    context: Arc<RenderingContext>,
}

impl ClusterCuller {
    pub fn new(
        context: Arc<RenderingContext>,
        lighting_set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self> {
        let cluster_build_shader = resources::load_shader_module(
            include_str!(concat!(env!("OUT_DIR"), "/res/shaders/cluster_build.comp")),
            "cluster_build.comp",
            ShaderKind::Compute,
            &context.device,
        )?;
        let light_cull_shader = resources::load_shader_module(
            include_str!(concat!(env!("OUT_DIR"), "/res/shaders/light_cull.comp")),
            "light_cull.comp",
            ShaderKind::Compute,
            &context.device,
        )?;

        let mut cluster_cull_descriptors = Vec::new();
        cluster_cull_descriptors.push(lighting_set_layout);
        let layout = unsafe {
            context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&cluster_cull_descriptors),
                None,
            )?
        };
        let pipelines = unsafe {
            context
                .device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[
                        vk::ComputePipelineCreateInfo::default()
                            .layout(layout)
                            .stage(
                                vk::PipelineShaderStageCreateInfo::default()
                                    .module(cluster_build_shader)
                                    .stage(vk::ShaderStageFlags::COMPUTE)
                                    .name(c"main"),
                            ),
                        vk::ComputePipelineCreateInfo::default()
                            .layout(layout)
                            .stage(
                                vk::PipelineShaderStageCreateInfo::default()
                                    .module(light_cull_shader)
                                    .stage(vk::ShaderStageFlags::COMPUTE)
                                    .name(c"main"),
                            ),
                    ],
                    None,
                )
                .map_err(|(_, e)| e)?
        };
        unsafe {
            context
                .device
                .destroy_shader_module(cluster_build_shader, None);
            context
                .device
                .destroy_shader_module(light_cull_shader, None);
        }
        Ok(Self {
            cull_pipeline: pipelines[1],
            build_pipeline: pipelines[0],
            layout,
            context,
        })
    }

    pub fn record_commands(&self, frame: &Frame) {
        let _main = tracy_client::span!("clustered light culler record commands");
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
        unsafe {
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[frame.lighting_set],
                &[],
            );

            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.build_pipeline);
            device.cmd_dispatch(
                cb,
                CLUSTER_GRID.x.div_ceil(8),
                CLUSTER_GRID.y.div_ceil(8),
                CLUSTER_GRID.z,
            );
        }
        mem(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
        );
        unsafe {
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.cull_pipeline);
            device.cmd_dispatch(cb, (CLUSTER_COUNT as u32).div_ceil(64), 1, 1);
        }
        mem(
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
    }
}

impl Drop for ClusterCuller {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_pipeline_layout(self.layout, None);
            self.context
                .device
                .destroy_pipeline(self.build_pipeline, None);
            self.context
                .device
                .destroy_pipeline(self.cull_pipeline, None);
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GpuLight {
    pub position_range: Vec4,  // xyz world pos, w range
    pub color_intensity: Vec4, // rgb color, w intensity
    pub direction_type: Vec4,  // xyz direction light points TO, w: 0 = point, 1 = spot
    pub spot_params: Vec4,     // x cos(inner), y cos(outer), z shadow index (-1 none), w unused
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GpuDirectional {
    pub color_intensity: Vec4, // rgb color, w intensity
    pub direction: Vec4,       // xyz direction light points TO
}

#[repr(C)]
pub struct LightingParamsUniformBufferObject {
    pub view: Mat4,
    pub inv_proj: Mat4,
    pub screen_size: Vec2, // pixels
    pub near: f32,
    pub far: f32,
    pub grid: UVec4,               // x,y,z = cluster counts
    pub dir_direction: Vec4,       // xyz = direction TO the light, w = 1.0 if present
    pub dir_color_intensity: Vec4, // rgb color, w intensity
    pub light_count: u32,
    pub _pad: [u32; 3],
}
