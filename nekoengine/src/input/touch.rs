use std::collections::HashMap;
use winit::dpi::PhysicalPosition;
use winit::event::{Touch, TouchPhase};

enum Role {
    Look,
    Move { origin: PhysicalPosition<f64> },
}

struct Finger {
    role: Role,
    last: PhysicalPosition<f64>,
}

#[derive(Default)]
pub struct TouchControls {
    fingers: HashMap<u64, Finger>,
    look_delta: (f32, f32),
    pub move_axis: (f32, f32),
}

impl TouchControls {
    const JOYSTICK_RADIUS: f64 = 150.0; 

    pub fn handle(&mut self, touch: &Touch, screen_width: f64) {
        let pos = touch.location;
        match touch.phase {
            TouchPhase::Started => {
                let role = if pos.x < screen_width * 0.4 {
                    Role::Move { origin: pos }
                } else {
                    Role::Look
                };
                self.fingers.insert(touch.id, Finger { role, last: pos });
            }
            TouchPhase::Moved => {
                if let Some(f) = self.fingers.get_mut(&touch.id) {
                    match f.role {
                        Role::Look => {
                            self.look_delta.0 += (pos.x - f.last.x) as f32;
                            self.look_delta.1 += (pos.y - f.last.y) as f32;
                        }
                        Role::Move { origin } => {
                            let r = Self::JOYSTICK_RADIUS;
                            self.move_axis = (
                                ((pos.x - origin.x) / r).clamp(-1.0, 1.0) as f32,
                                (-(pos.y - origin.y) / r).clamp(-1.0, 1.0) as f32, // screen y is down
                            );
                        }
                    }
                    f.last = pos;
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                if let Some(f) = self.fingers.remove(&touch.id) {
                    if matches!(f.role, Role::Move { .. }) {
                        self.move_axis = (0.0, 0.0);
                    }
                }
            }
        }
    }

    pub fn take_look_delta(&mut self) -> (f32, f32) {
        std::mem::take(&mut self.look_delta)
    }
}