use std::{
    collections::{HashMap, VecDeque},
    ops::Mul,
    sync::Arc,
};

use anyhow::Result;
use ash::vk;
use glam::IVec3;
use gpu_allocator::vulkan::Allocator;
use parking_lot::Mutex;
use shaderc::ShaderKind;
use std::result::Result::Ok as stdOk;
use tracing::info;

use crate::app::{
    engine::{
        gpu::context::RenderingContext,
        renderer::{Frame, swapchain::Swapchain},
        resources::{self, Resources, data::ModelVertex, pool::Handle, types::Model},
    },
    game::{self, world::World},
};

#[repr(C, align(16))]
struct CharacterPushConstants {
    vertex_buffer: vk::DeviceAddress,
    material_id: u32,
    model: glam::Mat4,
}

#[repr(C, align(16))]
struct LegacyCharacterPushConstants {
    material_id: u32,
    model: glam::Mat4,
}

struct DrawItem {
    index_page: u32,
    vertex_page: u32,
    first_index: u32,
    index_count: u32,
    vertex_offset: i32,
    mesh_transform: glam::Mat4,
    bindless_index: u32,
}

pub struct CharacterRenderer {
    pub character_pipeline: vk::Pipeline,
    pub character_pipeline_layout: vk::PipelineLayout,
    pub world: Arc<World>,
    pub resources: Arc<Mutex<Resources>>,
    pub allocator: Arc<Mutex<Allocator>>,
    pub context: Arc<RenderingContext>,
}

impl CharacterRenderer {
    pub fn new<'a>(
        context: Arc<RenderingContext>,
        allocator: Arc<Mutex<Allocator>>,
        world: Arc<World>,
        swapchain: &Swapchain,
        character_descriptor_set_layouts: &'a [vk::DescriptorSetLayout],
        resources: Arc<Mutex<Resources>>,
    ) -> Result<Self> {
        let texture_descriptor_set_layout = {
            let res = resources.lock();
            res.bindless.layout
        };
        let mut character_set_layouts = Vec::new();
        character_set_layouts.extend_from_slice(character_descriptor_set_layouts);
        character_set_layouts.push(texture_descriptor_set_layout);
        let character_push_constant_range = if context.use_buffer_device_address {
            vk::PushConstantRange::default()
                .offset(0)
                .size(std::mem::size_of::<CharacterPushConstants>() as u32)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
        } else {
            vk::PushConstantRange::default()
                .offset(0)
                .size(std::mem::size_of::<LegacyCharacterPushConstants>() as u32)
                .stage_flags(vk::ShaderStageFlags::VERTEX)
        };
        let character_pipeline_layout = unsafe {
            context.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&character_set_layouts)
                    .push_constant_ranges(&[character_push_constant_range]),
                None,
            )?
        };
        let bda_character_shader =
            include_str!(concat!(env!("OUT_DIR"), "/res/shaders/character.vert"));
        let legacy_character_shader = include_str!(concat!(
            env!("OUT_DIR"),
            "/res/shaders/character_legacy.vert"
        ));

        let character_shader = if context.use_buffer_device_address {
            resources::load_shader_module(
                bda_character_shader,
                "character.vert",
                ShaderKind::Vertex,
                &context.device,
            )?
        } else {
            resources::load_shader_module(
                legacy_character_shader,
                "character_legacy.vert",
                ShaderKind::Vertex,
                &context.device,
            )?
        };
        let fragment_shader = resources::load_shader_module(
            include_str!(concat!(env!("OUT_DIR"), "/res/shaders/character.frag")),
            "character.frag",
            ShaderKind::Fragment,
            &context.device,
        )?;

        let binding_descriptions = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<ModelVertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];

        let attribute_descriptions = [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(ModelVertex, position) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32_SFLOAT)
                .offset(std::mem::offset_of!(ModelVertex, uv1) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(std::mem::offset_of!(ModelVertex, normal) as u32),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(3)
                .format(vk::Format::R32_SFLOAT)
                .offset(std::mem::offset_of!(ModelVertex, uv2) as u32),
        ];

        let vertex_input_state = if context.use_buffer_device_address {
            vk::PipelineVertexInputStateCreateInfo::default()
        } else {
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&binding_descriptions)
                .vertex_attribute_descriptions(&attribute_descriptions)
        };

        let character_pipeline = context.create_graphics_pipeline(
            vk::Extent2D::default()
                .width(swapchain.draw_image.extent.width)
                .height(swapchain.draw_image.extent.height),
            swapchain.draw_image.image_format,
            swapchain.depth_image.image_format,
            character_shader,
            fragment_shader,
            character_pipeline_layout,
            Default::default(),
            swapchain.main_render_pass,
            vertex_input_state,
        )?;

        unsafe {
            context.device.destroy_shader_module(character_shader, None);
            context.device.destroy_shader_module(fragment_shader, None);
        }

        Ok(Self {
            character_pipeline,
            character_pipeline_layout,
            world,
            resources,
            allocator,
            context,
        })
    }

    pub fn record_commands(
        &self,
        frame: &Frame,
        frames: &Vec<Frame>,
        extent: vk::Extent2D,
        swapchain: &Swapchain,
        texture_set: vk::DescriptorSet,
    ) {
        unsafe {
            self.context.device.cmd_bind_pipeline(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.character_pipeline,
            );

            self.context.device.cmd_bind_descriptor_sets(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.character_pipeline_layout,
                0,
                &[frame.character_descriptors],
                &[],
            );

            self.context.device.cmd_bind_descriptor_sets(
                frame.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.character_pipeline_layout,
                2,
                &[texture_set],
                &[],
            );

            let res = self.resources.lock();
            let geo = res.geometry();
            let mut items: Vec<DrawItem> = res
                .meshes
                .iter()
                .filter_map(|mesh_tuple| {
                    let mesh = mesh_tuple.1;
                    let mat = res.material(mesh.material)?;
                    let tex = res.texture(mat.base_color)?;
                    let vpage = geo.vertices.get_page(mesh.vertex_alloc.page as usize);

                    Some(DrawItem {
                        index_page: mesh.index_alloc.page,
                        vertex_page: mesh.vertex_alloc.page,
                        first_index: mesh.index_alloc.offset_units(),
                        index_count: mesh.index_alloc.units,
                        vertex_offset: mesh.vertex_alloc.offset_units() as i32,
                        mesh_transform: mesh.transform,
                        bindless_index: tex.bindless_index,
                    })
                })
                .collect();

            items.sort_unstable_by_key(|i| (i.index_page, i.vertex_page));
            let mut bound_index_page = u32::MAX;
            let mut bound_vertex_page = u32::MAX;
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
                let scale = glam::Mat4::from_scale(glam::Vec3 {
                    x: 20.0,
                    y: 20.0,
                    z: 20.0,
                });

                let page = geo.vertices.get_page(item.vertex_page as usize);
                if self.context.use_buffer_device_address {
                    let push_constants = CharacterPushConstants {
                        vertex_buffer: page.address().unwrap(),
                        material_id: item.bindless_index,
                        model: scale.mul_mat4(&item.mesh_transform),
                    };

                    self.context.device.cmd_push_constants(
                        frame.command_buffer,
                        self.character_pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        std::slice::from_raw_parts(
                            &push_constants as *const _ as *const u8,
                            std::mem::size_of::<CharacterPushConstants>(),
                        ),
                    );
                } else {
                    let legacy_push_constants = LegacyCharacterPushConstants {
                        material_id: item.bindless_index,
                        model: scale.mul_mat4(&item.mesh_transform),
                    };

                    self.context.device.cmd_push_constants(
                        frame.command_buffer,
                        self.character_pipeline_layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        std::slice::from_raw_parts(
                            &legacy_push_constants as *const _ as *const u8,
                            std::mem::size_of::<LegacyCharacterPushConstants>(),
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
    }
}

impl Drop for CharacterRenderer {
    fn drop(&mut self) {
        unsafe {
            self.context
                .device
                .destroy_pipeline_layout(self.character_pipeline_layout, None);
            self.context
                .device
                .destroy_pipeline(self.character_pipeline, None);
            /*
            for model in &mut self.models {
                for mesh in &mut model.meshes {
                    self.context
                        .destroy_buffer(&mut self.allocator.lock(), &mut mesh.vertex_buffer);
                    self.context
                        .destroy_buffer(&mut self.allocator.lock(), &mut mesh.index_buffer);
                }

                model
                    .materials
                    .iter_mut()
                    .map(|material| &mut material.diffuse_texture)
                    .for_each(|texture| {
                        Texture::destroy_texture(
                            self.context.clone(),
                            &mut self.allocator.lock(),
                            texture,
                        );
                    });
            }
            */
        }
    }
}
