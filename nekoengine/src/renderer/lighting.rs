use bevy_ecs::component::Component;
use glam::{Mat4, UVec4, Vec2, Vec3, Vec4};

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

pub const MAX_LIGHTS: usize = 1024;

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
    pub direction: Vec4,  // xyz direction light points TO
}

#[repr(C)]
pub struct LightingParamsUniformBufferObject {
    pub inv_proj: Mat4,
    pub screen_size: Vec2,      // pixels
    pub near: f32,
    pub far: f32,
    pub grid: UVec4,            // x,y,z = cluster counts, w = tile size in px
    pub dir_direction: Vec4,         // xyz = direction TO the light, w = 1.0 if present
    pub dir_color_intensity: Vec4,   // rgb color, w intensity
    pub light_count: u32,
    pub _pad: [u32; 3],
}