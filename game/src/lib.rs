use bevy_ecs::{
    error::Result,
    query::With,
    schedule::IntoScheduleConfigs,
    system::{Commands, Query, Res},
};

use nekoengine::{
    Assets, EngineSet, Input, Plugin, Time,
    renderer::{
        GlobalTransform, ModelInstance, Transform, Visibility,
        camera::{Camera, CameraController},
    },
    resources::{
        animation::Animator,
        data::{MaterialData, TextureData},
        gltf,
    },
};
use winit::keyboard::KeyCode;

pub struct GamePlugin;

impl Plugin for GamePlugin {
    fn build(&self, app: &mut nekoengine::App) {
        app.add_startup(spawn_world);
        app.add_systems(player_controller.in_set(EngineSet::Game));
    }
}

const SPEED: f32 = 20.0;
const SENSITIVITY: f32 = 0.1;

pub fn spawn_world(mut commands: Commands, assets: Res<Assets>) -> Result<()> {
    let resources = assets.assets.clone();
    let annie_data = gltf::load_model_glb(
        "annie",
        include_bytes!(concat!(env!("OUT_DIR"), "/res/models/annie.glb")),
    )?;
    let chito_data = gltf::load_model_glb(
        "chito",
        include_bytes!(concat!(env!("OUT_DIR"), "/res/models/chito.glb")),
    )?;
    let mut annie_model_handle;
    let mut chito_model_handle;
    {
        let mut res = resources.lock();
        let missing_texture = res.upload_texture(TextureData::decode(
            include_bytes!(concat!(env!("OUT_DIR"), "/res/textures/missing.png")),
            true,
        )?)?;
        let missing_material = res.upload_material(
            MaterialData {
                name: "missing".to_string(),
                base_color_image: Some(0),
                base_color_factor: [1.0; 4],
            },
            &[missing_texture],
            missing_texture,
        )?;
        annie_model_handle = res.upload_model(annie_data, missing_texture, missing_material)?;
        chito_model_handle = res.upload_model(chito_data, missing_texture, missing_material)?;

        let annie_model = res.models.get(annie_model_handle).unwrap();
        let mut e_annie = commands.spawn((
            Visibility(true),
            ModelInstance(annie_model_handle),
            GlobalTransform(glam::Mat4::from_scale(glam::Vec3::new(5.0, 5.0, 5.0))),
        ));
        if !annie_model.clips.is_empty() {
            e_annie.insert(Animator::new(&annie_model.skeleton));
        }
        let chito_model = res.models.get(chito_model_handle).unwrap();
        let mut e_chito = commands.spawn((
            Visibility(true),
            ModelInstance(chito_model_handle),
            GlobalTransform(
                glam::Mat4::from_translation(glam::Vec3::new(15.0, 0.0, 0.0))
                    * glam::Mat4::from_scale(glam::Vec3::new(5.0, 5.0, 5.0)),
            ),
        ));
        if !chito_model.clips.is_empty() {
            e_chito.insert(Animator::new(&chito_model.skeleton));
        }
    }

    commands.spawn((
        Camera {
            fov_y: 45.0,
            near: 0.1,
            far: 1000.0,
        },
        CameraController {
            yaw: 225.0,
            pitch: -37.0,
            roll: 0.0,
        },
        Transform {
            translation: glam::Vec3::new(2.0, 2.0, 2.0),
            rotation: glam::Quat::IDENTITY,
            scale: glam::Vec3::new(1.0, 1.0, 1.0),
        },
    ));
    Ok(())
}

pub fn player_controller(
    mut q: Query<(&mut CameraController, &mut Transform), With<Camera>>,
    input: Res<Input>,
    time: Res<Time>,
) {
    let im = &input.input;
    for (mut ctl, mut tf) in &mut q {
        let (dx, dy) = {
            let l = im.mouse_look_delta.read();
            (l.0, l.1)
        };
        let (tdx, tdy) = im.touch_controls.write().take_look_delta();

        ctl.yaw += (dx + tdx) * SENSITIVITY;
        ctl.pitch = (ctl.pitch - (dy + tdy) * SENSITIVITY).clamp(-89.0, 89.0);

        {
            let mut l = im.mouse_look_delta.write();
            l.0 = 0.0;
            l.1 = 0.0;
        }

        let front = ctl.front();
        let right = front.cross(glam::Vec3::Y).normalize();

        let mut movement = glam::Vec3::ZERO;
        for key in im.get_snapshot() {
            match key {
                KeyCode::KeyW => movement += front,
                KeyCode::KeyS => movement -= front,
                KeyCode::KeyD => movement += right,
                KeyCode::KeyA => movement -= right,
                KeyCode::Space => movement += glam::Vec3::Y,
                KeyCode::ShiftLeft => movement -= glam::Vec3::Y,
                _ => {}
            }
        }
        tf.translation += movement.normalize_or_zero() * SPEED * time.dt;
        tf.rotation = glam::Quat::from_euler(
            glam::EulerRot::YXZ,
            ctl.yaw.to_radians(),
            ctl.pitch.to_radians(),
            ctl.roll.to_radians(),
        );
    }
}
