use glam::{Mat4, Quat, Vec3, Vec4};

use crate::app::engine::resources::data::{Channel, Clip, Interp, PathKind, Skeleton, SkinData, Trs};

pub fn sample(ch: &Channel, t: f32) -> Vec4 {
    let n = ch.times.len();
    let cubic = ch.interp == Interp::CubicSpline;

    // Keyframe accessors that hide the 3-values-per-key layout of cubic splines.
    let value = |k: usize| if cubic { ch.values[3 * k + 1] } else { ch.values[k] };
    let in_tan = |k: usize| ch.values[3 * k];
    let out_tan = |k: usize| ch.values[3 * k + 2];

    if n == 1 || t <= ch.times[0] {
        return value(0);
    }
    if t >= ch.times[n - 1] {
        return value(n - 1);
    }

    // Index of the keyframe at or before t. Safe: times[0] < t < times[n-1].
    let i = ch.times.partition_point(|&k| k <= t) - 1;
    let dt = ch.times[i + 1] - ch.times[i];
    let a = (t - ch.times[i]) / dt;
    let (v0, v1) = (value(i), value(i + 1));

    match ch.interp {
        Interp::Step => v0,
        Interp::Linear => match ch.path {
            PathKind::Rotation => {
                Vec4::from(Quat::from_vec4(v0).slerp(Quat::from_vec4(v1), a))
            }
            _ => v0.lerp(v1, a),
        },
        Interp::CubicSpline => {
            // Hermite spline; tangents are scaled by the keyframe interval (glTF spec, appendix C).
            let (a2, a3) = (a * a, a * a * a);
            let b0 = out_tan(i) * dt;
            let a1 = in_tan(i + 1) * dt;
            let r = v0 * (2.0 * a3 - 3.0 * a2 + 1.0)
                + b0 * (a3 - 2.0 * a2 + a)
                + v1 * (-2.0 * a3 + 3.0 * a2)
                + a1 * (a3 - a2);
            if ch.path == PathKind::Rotation { r.normalize() } else { r }
        }
    }
}

/// Applies every channel of `clip` at `time` on top of `pose` (which should start as the rest pose).
pub fn apply_clip(clip: &Clip, time: f32, pose: &mut [Trs]) {
    for ch in &clip.channels {
        let v = sample(ch, time);
        let node = &mut pose[ch.node];
        match ch.path {
            PathKind::Translation => node.t = v.truncate(),
            PathKind::Scale => node.s = v.truncate(),
            PathKind::Rotation => node.r = Quat::from_vec4(v).normalize(),
        }
    }
}

pub struct Animator {
    pub clip: usize,
    pub time: f32,
    pub speed: f32,
    pub looping: bool,
    pub playing: bool,
    local: Vec<Trs>,
    world: Vec<Mat4>,
}

impl Animator {
    pub fn new(skel: &Skeleton) -> Self {
        let world = Self::rest_world(skel);
        Self {
            clip: 0,
            time: 0.0,
            speed: 1.0,
            looping: true,
            playing: true,
            local: skel.rest.clone(),
            world,
        }
    }

    fn rest_world(skel: &Skeleton) -> Vec<Mat4> {
        let mut world = vec![Mat4::IDENTITY; skel.rest.len()];
        for &i in &skel.order {
            let p = skel.parent[i].map_or(Mat4::IDENTITY, |p| world[p]);
            world[i] = p * skel.rest[i].to_mat4();
        }
        world
    }

    /// Switches clip and restarts it.
    pub fn play(&mut self, clip: usize, looping: bool) {
        self.clip = clip;
        self.time = 0.0;
        self.looping = looping;
        self.playing = true;
    }

    /// Advances time, samples the clip and recomputes world matrices of every node.
    /// `clip` may be `None` (model without animations): the rest pose is used.
    pub fn update(&mut self, dt: f32, skel: &Skeleton, clip: Option<&Clip>) {
        self.local.copy_from_slice(&skel.rest);

        if let Some(clip) = clip {
            if self.playing {
                self.time += dt * self.speed;
            }
            if clip.duration > 0.0 {
                if self.looping {
                    self.time = self.time.rem_euclid(clip.duration);
                } else if self.time >= clip.duration {
                    self.time = clip.duration;
                    self.playing = false;
                }
            }
            apply_clip(clip, self.time, &mut self.local);
        }

        self.compute_world(skel);
    }

    /// Lets you pose the skeleton yourself (blending, procedural tweaks), then call this.
    pub fn local_pose_mut(&mut self) -> &mut [Trs] {
        &mut self.local
    }

    /// Recomputes world matrices from the current `local` pose.
    pub fn compute_world(&mut self, skel: &Skeleton) {
        for &i in &skel.order {
            let p = skel.parent[i].map_or(Mat4::IDENTITY, |p| self.world[p]);
            self.world[i] = p * self.local[i].to_mat4();
        }
    }

    /// World matrix of any node (use for animated meshes that have no skin).
    pub fn node_world(&self, node: usize) -> Mat4 {
        self.world[node]
    }

    /// Writes `world[joint] * inverse_bind` for each joint of `skin`. `out.len()` must be `skin.joints.len()`.
    pub fn write_skin(&self, skin: &SkinData, out: &mut [Mat4]) {
        debug_assert_eq!(out.len(), skin.joints.len());
        for (k, &j) in skin.joints.iter().enumerate() {
            out[k] = self.world[j] * skin.inverse_bind[k];
        }
    }
}

/// Crossfade helper: `out = a` blended toward `b` by `t` (0..1). All slices must have equal length.
pub fn blend_poses(a: &[Trs], b: &[Trs], t: f32, out: &mut [Trs]) {
    for ((o, a), b) in out.iter_mut().zip(a).zip(b) {
        *o = a.lerp(b, t);
    }
}