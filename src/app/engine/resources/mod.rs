use std::sync::Arc;

use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use parking_lot::Mutex;

use crate::app::engine::gpu::context::RenderingContext;
use crate::app::engine::resources::bindless_textures::BindlessTextures;
use crate::app::engine::resources::data::{MaterialData, MeshData, ModelData, TextureData};
use crate::app::engine::resources::geometry_pool::{GeometryPool, PoolKind};
use crate::app::engine::resources::pool::{Handle, Pool};
use crate::app::engine::resources::types::{Material, Mesh, Model, Texture};
use crate::app::engine::resources::uploader::Uploader;

pub mod bindless_textures;
pub mod data;
pub mod geometry_pool;
pub mod gltf;
pub mod pool;
pub mod types;
pub mod uploader;

pub struct Resources {
    pub textures: Pool<Texture>,
    pub materials: Pool<Material>,
    pub meshes: Pool<Mesh>,
    pub models: Pool<Model>,
    pub bindless: BindlessTextures,
    geometry_pool: GeometryPool,
    uploader: Uploader,
    allocator: Arc<Mutex<Allocator>>,
    context: Arc<RenderingContext>,
}

impl Resources {
    pub fn new(context: Arc<RenderingContext>) -> Result<Self> {
        let allocator = Arc::new(Mutex::new(Allocator::new(&AllocatorCreateDesc {
            instance: context.instance.clone(),
            device: context.device.clone(),
            physical_device: context.physical_device.handle,
            debug_settings: Default::default(),
            buffer_device_address: context.use_buffer_device_address,
            allocation_sizes: Default::default(),
        })?));
        Ok(Self {
            textures: Pool::new(),
            materials: Pool::new(),
            meshes: Pool::new(),
            models: Pool::new(),
            bindless: BindlessTextures::new(context.clone(), 65536)?,
            geometry_pool: GeometryPool::new(context.clone(), allocator.clone())?,
            uploader: Uploader::new(context.clone(), allocator.clone())
                .expect("failed to create uploader!"),
            allocator,
            context: context,
        })
    }

    pub fn upload_texture(&mut self, data: TextureData) -> Result<Handle<Texture>> {
        let image = self
            .uploader
            .upload_image(&data.rgba, data.width, data.height, data.srgb)?;
        let bindless_index = self
            .bindless
            .register(&self.context.device, image.image_view)?;
        Ok(self.textures.insert(Texture {
            image,
            bindless_index,
        }))
    }

    pub fn upload_material(&mut self, data: MaterialData) -> Result<Handle<Material>> {
        let tex_index = data.base_color_image.unwrap_or(0);
        Ok(self.materials.insert(Material {
            name: data.name,
            base_color: self
                .textures
                .get_at_index(tex_index)
                .expect("Couldn't find texture for material creation"),
            base_color_factor: data.base_color_factor,
        }))
    }

    pub fn upload_mesh(&mut self, data: MeshData) -> Result<Handle<Mesh>> {
        let vertex_alloc = self
            .geometry_pool
            .vertices
            .alloc(data.vertices.len() as u32, PoolKind::VERTEX)?;
        let index_alloc = self
            .geometry_pool
            .indices
            .alloc(data.indices.len() as u32, PoolKind::INDEX)?;

        let vertex_page = self
            .geometry_pool
            .vertices
            .get_page(vertex_alloc.page as usize);
        self.uploader.upload_buffer(
            vertex_page.vk_buffer(),
            self.geometry_pool.vertices.byte_offset(&vertex_alloc),
            bytemuck::cast_slice(&data.vertices),
        )?;
        let index_page = self
            .geometry_pool
            .indices
            .get_page(index_alloc.page as usize);
        self.uploader.upload_buffer(
            index_page.vk_buffer(),
            self.geometry_pool.indices.byte_offset(&index_alloc),
            bytemuck::cast_slice(&data.indices),
        )?;
        let mesh = Mesh {
            vertex_alloc,
            index_alloc,
            material: self
                .materials
                .get_at_index(data.material.unwrap_or(0))
                .expect("Couldn't get material while uploading mesh"),
            transform: data.transform,
            node: data.node,
        };
        Ok(self.meshes.insert(mesh))
    }

    pub fn upload_model(&mut self, data: ModelData) -> Result<Handle<Model>> {
        let mut textures = Vec::new();
        for texture in data.images {
            textures.push(self.upload_texture(texture)?);
        }
        let mut materials = Vec::new();
        for material in data.materials {
            materials.push(self.upload_material(material)?);
        }
        let mut meshes = Vec::new();
        for mesh in data.meshes {
            meshes.push(self.upload_mesh(mesh)?);
        }
        Ok(self.models.insert(Model {
            name: data.name,
            meshes,
            materials,
            textures,
        }))
    }

    pub fn mesh(&self, h: Handle<Mesh>) -> Option<&Mesh> {
        self.meshes.get(h)
    }

    pub fn material(&self, h: Handle<Material>) -> Option<&Material> {
        self.materials.get(h)
    }

    pub fn texture(&self, h: Handle<Texture>) -> Option<&Texture> {
        self.textures.get(h)
    }

    pub fn geometry(&self) -> &GeometryPool {
        &self.geometry_pool
    }

    pub fn flush(&mut self) -> Result<()> {
        self.uploader.flush()
    }
}

impl Drop for Resources {
    fn drop(&mut self) {
        for mut texture in self.textures.drain() {
            self.context
                .destroy_image(&mut self.allocator.lock(), &mut texture.image);
        }
    }
}

pub fn load_string(file_name: &str) -> anyhow::Result<String> {
    let txt = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read_to_string(path)?
    };

    Ok(txt)
}

pub fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    let data = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read(path)?
    };

    Ok(data)
}

pub fn load_spirv(data: Vec<u8>) -> anyhow::Result<Vec<u32>> {
    Ok(ash::util::read_spv(&mut std::io::Cursor::new(data))?)
}

pub fn load_shader_module(
    code: &str,
    input_file_name: &str,
    shader_kind: shaderc::ShaderKind,
    device: &ash::Device,
) -> anyhow::Result<vk::ShaderModule> {
    let binary_data = load_spirv(compile_shader(code, input_file_name, shader_kind)?)?;

    unsafe {
        Ok(device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&binary_data),
            None,
        )?)
    }
}

fn compile_shader(
    source: &str,
    input_file_name: &str,
    shader_kind: shaderc::ShaderKind,
) -> anyhow::Result<Vec<u8>> {
    let compiler = shaderc::Compiler::new().unwrap();
    let options = shaderc::CompileOptions::new().unwrap();
    //options.add_macro_definition("EP", Some("main"));
    let binary_result = compiler
        .compile_into_spirv(
            source,
            shader_kind,
            &input_file_name,
            "main",
            Some(&options),
        )
        .unwrap();

    Ok(Vec::from(binary_result.as_binary_u8()))
}
