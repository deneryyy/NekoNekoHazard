use std::{
    io::{BufReader, Cursor},
    sync::Arc,
};

use anyhow::Result;
use ash::vk;
use gltf::{Gltf, buffer};
use gpu_allocator::vulkan::Allocator;
use image::ImageBuffer;
use parking_lot::Mutex;

use crate::app::engine::renderer::{
    context::{AllocatedBuffer, RenderingContext},
    texture::Texture,
};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub uv1: f32,
    pub normal: [f32; 3],
    pub uv2: f32,
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
}

pub struct Material {
    pub name: String,
    pub diffuse_texture: Texture,
}

pub struct Mesh {
    pub vertex_buffer_address: Option<vk::DeviceAddress>,
    pub vertex_buffer: AllocatedBuffer,
    pub index_buffer: AllocatedBuffer,
    pub index_count: u32,
    pub material: u32,
}

pub fn load_model_glb(
    context: Arc<RenderingContext>,
    allocator: Arc<Mutex<Allocator>>,
    data: Vec<u8>,
) -> Result<Model> {
    let gltf_cursor = Cursor::new(data);
    let gltf_reader = BufReader::new(gltf_cursor);

    let gltf = Gltf::from_reader(gltf_reader)?;
    let buffer_data = gltf::import_buffers(&gltf.document, None, gltf.blob.clone())?;
    let mut meshes = Vec::new();
    for scene in gltf.scenes() {
        for node in scene.nodes() {
            parse_node(
                context.clone(),
                allocator.clone(),
                &node,
                &mut meshes,
                &buffer_data,
            );
        }
    }

    // Load materials
    let mut materials = Vec::new();
    for material in gltf.materials() {
        match &material
            .pbr_metallic_roughness()
            .base_color_texture()
            .map(|tex| {
                tex.texture().source()
            }) {
            None => {}
            Some(image) => match image.source() {
                gltf::image::Source::View { view, mime_type } => {
                    let image_data =
                        gltf::image::Data::from_source(image.source(), None, &buffer_data)?;

                    use gltf::image::Format;
                    let diffuse_texture = match image_data.format {
                        Format::R8G8B8 => {
                            let image_buf: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
                                ImageBuffer::from_raw(
                                    image_data.width as u32,
                                    image_data.height as u32,
                                    image_data.pixels,
                                )
                                .unwrap();
                            let image_rgb8 = image::DynamicImage::ImageRgb8(image_buf);
                            Some(Texture::from_image(
                                context.clone(),
                                &mut allocator.lock(),
                                &image_rgb8,
                            )?)
                        }
                        Format::R8G8B8A8 => {
                            let image_buf: image::ImageBuffer<image::Rgba<u8>, Vec<u8>> =
                                ImageBuffer::from_raw(
                                    image_data.width as u32,
                                    image_data.height as u32,
                                    image_data.pixels,
                                )
                                .unwrap();
                            let image_rgba8 = image::DynamicImage::ImageRgba8(image_buf);
                            Some(Texture::from_image(
                                context.clone(),
                                &mut allocator.lock(),
                                &image_rgba8,
                            )?)
                        }
                        _ => None,
                    }
                    .expect("Unsupported texture format detected while loading gltf model!");

                    materials.push(Material {
                        name: material.name().unwrap_or("Default Material").to_string(),
                        diffuse_texture,
                    });
                }
                gltf::image::Source::Uri { uri, mime_type } => {}
            },
        }
    }
    Ok(Model { meshes, materials })
}

fn parse_node(
    context: Arc<RenderingContext>,
    allocator: Arc<Mutex<Allocator>>,
    node: &gltf::Node,
    meshes: &mut Vec<Mesh>,
    buffer_data: &Vec<buffer::Data>,
) {
    match node.mesh() {
        None => {}
        Some(mesh) => {
            let primitives = mesh.primitives();
            primitives.for_each(|primitive| {
                let reader = primitive.reader(|buffer| Some(&buffer_data[buffer.index()]));

                let mut vertices = Vec::new();
                if let Some(vertex_attribute) = reader.read_positions() {
                    vertex_attribute.for_each(|vertex| {
                        vertices.push(ModelVertex {
                            position: vertex,
                            uv1: 0.0,
                            normal: Default::default(),
                            uv2: 0.0,
                        })
                    });
                }
                if let Some(normal_attribute) = reader.read_normals() {
                    let mut normal_index = 0;
                    normal_attribute.for_each(|normal| {
                        vertices[normal_index].normal = normal;

                        normal_index += 1;
                    });
                }
                if let Some(tex_coord_attribute) = reader.read_tex_coords(0).map(|v| v.into_f32()) {
                    let mut tex_coord_index = 0;
                    tex_coord_attribute.for_each(|tex_coord| {
                        vertices[tex_coord_index].uv1 = tex_coord[0];
                        vertices[tex_coord_index].uv2 = tex_coord[1];

                        tex_coord_index += 1;
                    });
                }

                let mut indices = Vec::new();
                if let Some(indices_raw) = reader.read_indices() {
                    indices.append(&mut indices_raw.into_u32().collect::<Vec<u32>>());
                }

                let vertex_usage = if context.use_buffer_device_address {
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                } else {
                    vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER
                };

                let vertex_buffer_size =
                    (vertices.len() * std::mem::size_of::<ModelVertex>()) as u64;
                let vertex_buffer = context
                    .create_buffer(
                        "Vertex buffer",
                        &mut allocator.lock(),
                        vertex_buffer_size,
                        vertex_usage,
                        gpu_allocator::MemoryLocation::GpuOnly,
                    )
                    .unwrap();

                let index_buffer_size = (indices.len() * std::mem::size_of::<u32>()) as u64;
                let index_buffer = context
                    .create_buffer(
                        "Index buffer",
                        &mut allocator.lock(),
                        index_buffer_size,
                        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                        gpu_allocator::MemoryLocation::GpuOnly,
                    )
                    .unwrap();
                let material = primitive.material();
                let vertex_buffer_address = unsafe {
                    if context.use_buffer_device_address {
                        Some(context.device.get_buffer_device_address(
                            &vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer.buffer),
                        ))
                    } else {
                        None
                    }
                };

                let mut staging_buffer = context.create_buffer(
                    "Staging buffer",
                    &mut allocator.lock(),
                    vertex_buffer_size + index_buffer_size,
                    vk::BufferUsageFlags::TRANSFER_SRC,
                    gpu_allocator::MemoryLocation::CpuToGpu,
                ).unwrap();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        vertices.as_ptr(),
                        staging_buffer
                            .allocation
                            .mapped_ptr()
                            .unwrap()
                            .as_ptr()
                            .cast(),
                        vertices.len(),
                    );
                    std::ptr::copy_nonoverlapping(
                        indices.as_ptr(),
                        staging_buffer
                            .allocation
                            .mapped_ptr()
                            .unwrap()
                            .as_ptr()
                            .byte_add(vertex_buffer_size as usize)
                            .cast(),
                        indices.len(),
                    );
                }
                let cmd = context.begin_immediate_submit().unwrap();
                unsafe {
                    context.device.cmd_copy_buffer(
                        cmd,
                        staging_buffer.buffer,
                        vertex_buffer.buffer,
                        &[vk::BufferCopy::default()
                            .size(vertex_buffer_size)
                            .src_offset(0)
                            .dst_offset(0)],
                    );
                    context.device.cmd_copy_buffer(
                        cmd,
                        staging_buffer.buffer,
                        index_buffer.buffer,
                        &[vk::BufferCopy::default()
                            .size(index_buffer_size)
                            .src_offset(vertex_buffer_size)
                            .dst_offset(0)],
                    );
                }
                context.end_immediate_submit().unwrap();
                context.destroy_buffer(&mut allocator.lock(), &mut staging_buffer);

                meshes.push(Mesh {
                    vertex_buffer_address,
                    vertex_buffer,
                    index_buffer,
                    index_count: indices.len() as u32,
                    material: material.index().unwrap() as u32,
                });
            });
        }
    }

    for child in node.children() {
        parse_node(
            context.clone(),
            allocator.clone(),
            &child,
            meshes,
            buffer_data,
        );
    }
}
