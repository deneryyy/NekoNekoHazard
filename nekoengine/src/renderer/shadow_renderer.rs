use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::Allocator;
use parking_lot::Mutex;
use shaderc::ShaderKind;
use std::sync::Arc;

use crate::{
    gpu::{
        context::{AllocatedImage, RenderingContext},
        descriptor::DescriptorLayoutBuilder,
    },
    renderer::{Frame, RenderFrame, swapchain::Swapchain},
    resources::{self, Resources, data::ModelVertex},
};

const SHADOW_WIDTH: u32 = 1024;
const SHADOW_HEIGHT: u32 = 1024;

struct DrawItem {
    index_page: u32,
    vertex_page: u32,
    skin_page: u32,
    first_index: u32,
    index_count: u32,
    vertex_offset: i32,
    mesh_transform: glam::Mat4,
    skin_delta: i32,
    joint_offset: u32,
}

#[repr(C)]
struct ShadowPushConstants {
    model: glam::Mat4,
    skin_page: u32,
    skin_delta: i32,
    joint_offset: u32,
}

pub struct ShadowRenderer {
    skin_set: vk::DescriptorSet,
    skin_set_layout: vk::DescriptorSetLayout,
    shadow_pipeline: vk::Pipeline,
    shadow_pipeline_layout: vk::PipelineLayout,
    framebuffer: Option<vk::Framebuffer>,
    render_pass: Option<vk::RenderPass>,
    depth_map: AllocatedImage,
    resources: Arc<Mutex<Resources>>,
    allocator: Arc<Mutex<Allocator>>,
    context: Arc<RenderingContext>,
}

impl ShadowRenderer {
    pub fn new<'a>(
        context: Arc<RenderingContext>,
        allocator: Arc<Mutex<Allocator>>,
        resources: Arc<Mutex<Resources>>,
        frame_set_layouts: &'a [vk::DescriptorSetLayout],
        skin_set_layout: vk::DescriptorSetLayout,
        skin_set: vk::DescriptorSet,
    ) -> Result<Self> {
        let depth_map = context.create_image(
            &mut allocator.lock(),
            vk::Extent3D {
                width: SHADOW_WIDTH,
                height: SHADOW_HEIGHT,
                depth: 1,
            },
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::ImageAspectFlags::DEPTH,
        )?;

        let mut render_pass = None;
        if !context.use_dynamic_rendering {
            render_pass = Some(context.create_render_pass(None, Some(depth_map.image_format))?);
        }

        let extent = vk::Extent2D {
            width: SHADOW_WIDTH,
            height: SHADOW_HEIGHT,
        };

        let mut framebuffer = None;
        if !context.use_dynamic_rendering {
            framebuffer = Some(context.create_framebuffer(
                None,
                Some(depth_map.image_view),
                extent,
                render_pass.unwrap(),
            )?);
        }
        let push_constant_range = vk::PushConstantRange::default()
            .offset(0)
            .size(std::mem::size_of::<ShadowPushConstants>() as u32)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let mut shadow_set_layouts = Vec::new();
        shadow_set_layouts.extend_from_slice(frame_set_layouts);
        shadow_set_layouts.push(skin_set_layout);
        let shadow_pipeline_layout = unsafe {
            context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&shadow_set_layouts)
                    .push_constant_ranges(&[push_constant_range]),
                None,
            )?
        };

        let vertex_shader = resources::load_shader_module(
            include_str!(concat!(env!("OUT_DIR"), "/res/shaders/shadow_map.vert")),
            "character.vert",
            ShaderKind::Vertex,
            &context.device,
        )?;

        let fragment_shader = resources::load_shader_module(
            include_str!(concat!(env!("OUT_DIR"), "/res/shaders/shadow_map.frag")),
            "character.frag",
            ShaderKind::Fragment,
            &context.device,
        )?;

        let binding_descriptions = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<ModelVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];

        let attribute_descriptions = [vk::VertexInputAttributeDescription::default()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(std::mem::offset_of!(ModelVertex, position) as u32)];

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let shadow_pipeline = context.create_graphics_pipeline(
            extent,
            vk::Format::UNDEFINED,
            depth_map.image_format,
            vertex_shader,
            fragment_shader,
            shadow_pipeline_layout,
            Default::default(),
            render_pass,
            vertex_input_state,
        )?;

        unsafe {
            context.device.destroy_shader_module(vertex_shader, None);
            context.device.destroy_shader_module(fragment_shader, None);
        }

        Ok(Self {
            skin_set,
            skin_set_layout,
            shadow_pipeline,
            shadow_pipeline_layout,
            framebuffer,
            render_pass,
            depth_map,
            resources,
            allocator,
            context,
        })
    }

    pub fn record_commands(&self, frame: &Frame, render_frame: &RenderFrame) {
        let _main = tracy_client::span!("shadow renderer record commands");
        self.context.transition_image_layout(
            frame.command_buffer,
            self.depth_map.image,
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

        let extent = vk::Extent2D {
            width: SHADOW_WIDTH,
            height: SHADOW_HEIGHT,
        };

        self.context.begin_rendering(
            frame.command_buffer,
            None,
            Some(self.depth_map.image_view),
            vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
            vk::Rect2D::default().extent(extent),
            self.framebuffer,
            self.render_pass,
        );

        unsafe {
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

            self.context.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.shadow_pipeline,
            );

            self.context.device.cmd_bind_descriptor_sets(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.shadow_pipeline_layout,
                0,
                &[frame.frame_set],
                &[],
            );

            self.context.device.cmd_bind_descriptor_sets(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.shadow_pipeline_layout,
                2,
                &[self.skin_set],
                &[],
            );

            let mut items = Vec::new();
            let res = self.resources.lock();
            let geo = res.geometry();
            {
                let _s = tracy_client::span!("build draw items");
                for inst in render_frame.instances.iter() {
                    let Some(model) = res.models.get(inst.model) else {
                        continue;
                    };

                    for &mh in &model.meshes {
                        let Some(mesh) = res.meshes.get(mh) else {
                            continue;
                        };
                        let Some(mat) = res.material(mesh.material) else {
                            continue;
                        };
                        let Some(tex) = res.texture(mat.base_color) else {
                            continue;
                        };

                        let (model_matrix, joint_offset, skin_delta) = match &mesh.skin_alloc {
                            Some((alloc, skin_idx)) => (
                                inst.transform,
                                inst.skin_offsets[*skin_idx],
                                alloc.offset_units() as i32
                                    - mesh.vertex_alloc.offset_units() as i32,
                            ),
                            None => (inst.transform * mesh.transform, u32::MAX, 0),
                        };

                        items.push(DrawItem {
                            index_page: mesh.index_alloc.page,
                            vertex_page: mesh.vertex_alloc.page,
                            skin_page: mesh.skin_alloc.as_ref().map(|(a, _)| a.page).unwrap_or(0),
                            first_index: mesh.index_alloc.offset_units(),
                            index_count: mesh.index_alloc.units,
                            vertex_offset: mesh.vertex_alloc.offset_units() as i32,
                            mesh_transform: model_matrix,
                            skin_delta,
                            joint_offset,
                        });
                    }
                }
            }

            {
                let _s = tracy_client::span!("sort");
                items.sort_unstable_by_key(|i| (i.index_page, i.vertex_page));
            }
            let mut bound_index_page = u32::MAX;
            let mut bound_vertex_page = u32::MAX;

            {
                let _s = tracy_client::span!("record draws");
                for item in &items {
                    if item.index_page != bound_index_page {
                        let page = geo.indices.get_page(item.index_page as usize);
                        self.context.device.cmd_bind_index_buffer(
                            frame.command_buffer,
                            page.vk_buffer(),
                            0,
                            vk::IndexType::UINT32,
                        );
                        bound_index_page = item.index_page;
                    }
                    let page = geo.vertices.get_page(item.vertex_page as usize);

                    let legacy_push_constants = ShadowPushConstants {
                        model: item.mesh_transform,
                        skin_page: item.skin_page,
                        skin_delta: item.skin_delta,
                        joint_offset: item.joint_offset,
                    };

                    self.context.device.cmd_push_constants(
                        frame.command_buffer,
                        self.shadow_pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        std::slice::from_raw_parts(
                            &legacy_push_constants as *const _ as *const u8,
                            std::mem::size_of::<ShadowPushConstants>(),
                        ),
                    );

                    if bound_vertex_page != item.vertex_page {
                        self.context.device.cmd_bind_vertex_buffers(
                            frame.command_buffer,
                            0,
                            &[page.vk_buffer()],
                            &[0],
                        );
                        bound_vertex_page = item.vertex_page;
                    }

                    self.context.device.cmd_draw_indexed(
                        frame.command_buffer,
                        item.index_count,
                        1,
                        item.first_index,
                        item.vertex_offset,
                        0,
                    );
                }
            }

            if self.context.use_dynamic_rendering {
                self.context.device.cmd_end_rendering(frame.command_buffer);
            } else {
                self.context
                    .device
                    .cmd_end_render_pass(frame.command_buffer);
            }
        }

        self.context.transition_image_layout(
            frame.command_buffer,
            self.depth_map.image,
            0,
            vk::REMAINING_MIP_LEVELS,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::ImageAspectFlags::DEPTH,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
    }
}

impl Drop for ShadowRenderer {
    fn drop(&mut self) {
        unsafe {
            if !self.context.use_dynamic_rendering {
                self.context
                    .device
                    .destroy_render_pass(self.render_pass.unwrap(), None);
            }
            if let Some(fb) = self.framebuffer.take() {
                self.context.device.destroy_framebuffer(fb, None);
            }
            self.context
                .device
                .destroy_pipeline_layout(self.shadow_pipeline_layout, None);
            self.context
                .device
                .destroy_pipeline(self.shadow_pipeline, None);
            self.context
                .destroy_image(&mut self.allocator.lock(), &mut self.depth_map);
        }
    }
}
