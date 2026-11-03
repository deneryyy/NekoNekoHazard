use bevy_ecs::component::Component;

#[derive(Component)] pub struct Camera { pub fov_y: f32, pub near: f32, pub far: f32 }

#[derive(Component)]
pub struct CameraController { pub yaw: f32, pub pitch: f32 }  

#[derive(Default)]
pub struct RenderCamera {
    pub position: glam::Vec3,
    pub front: glam::Vec3,
    pub up: glam::Vec3,
    pub right: glam::Vec3,
    pub world_up: glam::Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub roll: f32,
    pub movement_speed: f32,
    pub mouse_sensitivity: f32,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
}

const SENSIVITY: f32 = 0.1;
const SPEED: f32 = 20.0;
const FOV: f32 = 45.0;
const NEAR: f32 = 0.1;
const FAR: f32 = 1000.0;

impl CameraController {
    pub fn front(&self) -> glam::Vec3 {
        let (y, p) = (self.yaw.to_radians(), self.pitch.to_radians());
        glam::Vec3::new(y.cos() * p.cos(), p.sin(), y.sin() * p.cos()).normalize()
    }
}

impl RenderCamera {
    pub fn new(pos: glam::Vec3, up: glam::Vec3, yaw: f32, pitch: f32, roll: f32) -> Self {
        let front = glam::Vec3 {
            x: f32::cos(yaw.to_radians()) * f32::cos(pitch.to_radians()),
            y: f32::sin(pitch.to_radians()),
            z: f32::sin(yaw.to_radians()) * f32::cos(pitch.to_radians()),
        }
        .normalize();
        let right = front.cross(up).normalize();
        Self {
            position: pos,
            front: front,
            up: right.cross(front).normalize(),
            right,
            world_up: up,
            yaw,
            pitch,
            roll,
            movement_speed: SPEED,
            mouse_sensitivity: SENSIVITY,
            fov_y: FOV,
            near: NEAR,
            far: FAR,
        }
    }

    pub fn get_view_matrix(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(self.position, self.position + self.front, self.world_up)
    }

    pub fn update_camera_vectors(&mut self) {
        self.front = glam::Vec3 {
            x: f32::cos(self.yaw.to_radians()) * f32::cos(self.pitch.to_radians()),
            y: f32::sin(self.pitch.to_radians()),
            z: f32::sin(self.yaw.to_radians()) * f32::cos(self.pitch.to_radians()),
        }
        .normalize();
        self.right = self.front.cross(self.world_up).normalize();
        self.up = self.right.cross(self.front).normalize()
    }
}