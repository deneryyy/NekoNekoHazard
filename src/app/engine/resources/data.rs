use std::io::Cursor;

use anyhow::Result;
use image::ImageReader;

pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub srgb: bool,
}

pub struct MaterialData {
    pub name: String,
    pub base_color_image: Option<usize>,
    pub base_color_factor: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub uv1: f32,
    pub normal: [f32; 3],
    pub uv2: f32,
}

pub struct MeshData {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>,
    pub material: Option<usize>,
    pub transform: glam::Mat4,
    pub node: usize,
}

pub struct ModelData {
    pub name: String,
    pub images: Vec<TextureData>,
    pub materials: Vec<MaterialData>,
    pub meshes: Vec<MeshData>,
}

impl TextureData {
    pub fn decode(bytes: &[u8], srgb: bool) -> Result<Self> {
        let img = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()?
            .decode()?;
        let rgba = img.to_rgba8();
        Ok(Self {
            width: rgba.width(),
            height: rgba.height(),
            rgba: rgba.into_raw(),
            srgb,
        })
    }
 
    pub fn mip_levels(&self) -> u32 {
        32 - self.width.max(self.height).leading_zeros() // floor(log2(max)) + 1
    }
}
