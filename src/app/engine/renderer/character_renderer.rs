use std::{
    collections::{HashMap, VecDeque},
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
        renderer::{
            Frame, context::{AllocatedBuffer, RenderingContext}, model::Model, staging::RingStagingBuffer, swapchain::Swapchain, texture::Texture,
        }, resources,
    }, game::{self, world::World},
};

#[repr(C, align(16))]
struct CharacterPushConstants {
    vertex_buffer: vk::DeviceAddress,
    material_id: u32,
}

#[repr(C, align(16))]
struct LegacyCharacterPushConstants {
    material_id: u32,
}

pub struct CharacterRenderer {
    pub models: Vec<Model>,
    pub character_pipeline: vk::Pipeline,
    pub character_pipeline_layout: vk::PipelineLayout,
    pub world: Arc<World>,
    pub allocator: Arc<Mutex<Allocator>>,
    pub context: Arc<RenderingContext>,
}

impl CharacterRenderer {
    pub fn new<'a>(
        context: Arc<RenderingContext>,
        allocator: Arc<Mutex<Allocator>>,
        world: Arc<World>,
        texture_num: u32,
        swapchain: &Swapchain,
        vertex_character_descriptor_set_layouts: &'a [vk::DescriptorSetLayout],
        texture_descriptor_set_layout: &'a vk::DescriptorSetLayout,
        models: Vec<Model>,
    ) -> Result<Self> {
        let mut character_set_layouts = Vec::new();
        character_set_layouts.extend_from_slice(vertex_character_descriptor_set_layouts);
        character_set_layouts.push(*texture_descriptor_set_layout);
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
            "terrain.frag",
            ShaderKind::Fragment,
            &context.device,
        )?;
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
        )?;

        unsafe {
            context.device.destroy_shader_module(character_shader, None);
            context.device.destroy_shader_module(fragment_shader, None);
        }

        Ok(Self {
            models,
            character_pipeline,
            character_pipeline_layout,
            world,
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
            for model in self.models.iter() {
                for mesh in model.meshes.iter() {
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

                    if self.context.use_buffer_device_address {
                        let push_constants = CharacterPushConstants {
                            vertex_buffer: mesh.vertex_buffer_address.unwrap(),
                            material_id: mesh.material,
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
                            material_id: mesh.material,
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

                        self.context.device.cmd_bind_vertex_buffers(
                            frame.command_buffer,
                            0,
                            &[mesh.vertex_buffer.buffer],
                            &[0],
                        );
                    }

                    self.context.device.cmd_bind_index_buffer(
                        frame.command_buffer,
                        mesh.index_buffer.buffer,
                        0,
                        vk::IndexType::UINT32,
                    );

                    self.context.device.cmd_draw_indexed(
                        frame.command_buffer,
                        mesh.index_count,
                        1,
                        0,
                        0,
                        0,
                    );
                }
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
            for model in &mut self.models {
                for mesh in &mut model.meshes {
                    self.context.destroy_buffer(
                        &mut self.allocator.lock(),
                        &mut mesh.vertex_buffer,
                    );
                    self.context.destroy_buffer(
                        &mut self.allocator.lock(),
                        &mut mesh.index_buffer,
                    );
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
        }
    }
}
