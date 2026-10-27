use glam::Mat4;

use crate::app::engine::{
    gpu::context::AllocatedBuffer,
    renderer::animation_data::MAX_JOINTS,
    resources::{
        Resources,
        animation::Animator,
        pool::{Handle, Pool},
        types::Model,
    },
};

pub struct Instance {
    pub model: Handle<Model>,
    pub transform: Mat4,
    pub animator: Option<Animator>,
    pub visible: bool,
    pub skin_offsets: Vec<u32>,
}

pub struct Scene {
    pub instances: Pool<Instance>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            instances: Pool::new(),
        }
    }

    pub fn spawn(
        &mut self,
        res: &Resources,
        model: Handle<Model>,
        transform: Mat4,
    ) -> Handle<Instance> {
        let m = res.models.get(model).unwrap();
        let animator = (!m.clips.is_empty()).then(|| Animator::new(&m.skeleton));
        self.instances.insert(Instance {
            model,
            transform,
            animator,
            visible: true,
            skin_offsets: vec![u32::MAX; m.skins.len()],
        })
    }

    pub fn prepare_joints(&mut self, dt: f32, res: &Resources, joints: &mut [Mat4]) {
        let mut cursor = 0usize;

        for (_, inst) in self.instances.iter_mut() {
            if !inst.visible {
                continue;
            }
            let Some(model) = res.models.get(inst.model) else {
                continue;
            };
            let Some(anim) = inst.animator.as_mut() else {
                continue;
            };

            anim.update(dt, &model.skeleton, model.clips.get(anim.clip));

            for (s, skin) in model.skins.iter().enumerate() {
                let n = skin.joints.len();
                match joints.get_mut(cursor..cursor + n) {
                    Some(out) => {
                        anim.write_skin(skin, out);
                        inst.skin_offsets[s] = cursor as u32;
                        cursor += n;
                    }
                    None => {
                        // Joint buffer full: the shader draws this skin in bind pose.
                        inst.skin_offsets[s] = u32::MAX;
                    }
                }
            }
        }
    }
}
