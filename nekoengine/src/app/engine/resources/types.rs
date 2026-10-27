use crate::app::engine::{gpu::context::AllocatedImage, resources::{data::{Clip, Skeleton, SkinData}, geometry_pool::PoolAlloc, pool::Handle}};

pub struct Texture {
    pub image: AllocatedImage,
    pub bindless_index: u32,
}

pub struct Material {
    pub name: String,
    pub base_color: Handle<Texture>,
    pub base_color_factor: [f32; 4],
}

pub struct Mesh {
    pub vertex_alloc: PoolAlloc,
    pub index_alloc: PoolAlloc,
    pub skin_alloc: Option<(PoolAlloc, usize)>,
    pub material: Handle<Material>,
    pub transform: glam::Mat4,
    pub node: usize,
}

/// A model owns everything it created, so unload_model needs no bookkeeping elsewhere.
pub struct Model {
    pub name: String,
    pub meshes: Vec<Handle<Mesh>>,
    pub materials: Vec<Handle<Material>>,
    pub textures: Vec<Handle<Texture>>, // only textures created for this model, not defaults
    pub skeleton: Skeleton,
    pub skins: Vec<SkinData>,
    pub clips: Vec<Clip>,
}
