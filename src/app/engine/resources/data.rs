use std::io::Cursor;

use anyhow::Result;
use glam::{Mat4, Quat, Vec3, Vec4};
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinVertex {
    pub joints: [u32; 4],   
    pub weights: [f32; 4],
}

pub struct MeshData {
    pub vertices: Vec<ModelVertex>,
    pub indices: Vec<u32>,
    pub skin: Option<(usize, Vec<SkinVertex>)>,
    pub material: Option<usize>,
    pub transform: glam::Mat4,
    pub node: usize,
}

pub struct ModelData {
    pub name: String,
    pub images: Vec<TextureData>,
    pub materials: Vec<MaterialData>,
    pub meshes: Vec<MeshData>,
    pub skeleton: Skeleton,
    pub skins: Vec<SkinData>,
    pub clips: Vec<Clip>,
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

#[derive(Clone, Copy, Debug)]
pub struct Trs {
    pub t: Vec3,
    pub r: Quat,
    pub s: Vec3,
}

impl Trs {
    pub const IDENTITY: Trs = Trs { t: Vec3::ZERO, r: Quat::IDENTITY, s: Vec3::ONE };

    pub fn to_mat4(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.s, self.r, self.t)
    }

    pub fn lerp(&self, other: &Trs, a: f32) -> Trs {
        Trs {
            t: self.t.lerp(other.t, a),
            r: self.r.slerp(other.r, a),
            s: self.s.lerp(other.s, a),
        }
    }
}

pub struct Skeleton {
    pub parent: Vec<Option<usize>>,
    pub rest: Vec<Trs>,
    pub order: Vec<usize>,
}

pub struct SkinData {
    pub joints: Vec<usize>,
    pub inverse_bind: Vec<Mat4>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathKind {
    Translation,
    Rotation,
    Scale,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interp {
    Linear,
    Step,
    CubicSpline,
}

pub struct Channel {
    pub node: usize,
    pub path: PathKind,
    pub interp: Interp,
    pub times: Vec<f32>,
    pub values: Vec<Vec4>,
}

pub struct Clip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<Channel>,
}