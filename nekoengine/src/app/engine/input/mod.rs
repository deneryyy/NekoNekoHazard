use std::{
    collections::HashSet,
};

use parking_lot::RwLock;
use winit::keyboard::KeyCode;

use crate::app::engine::input::touch::TouchControls;

mod touch;

pub struct InputManager {
    pub pressed_keys: RwLock<HashSet<KeyCode>>,
    pub touch_controls: RwLock<TouchControls>,
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            pressed_keys: RwLock::new(HashSet::new()),
            touch_controls: RwLock::new(TouchControls::default())
        }
    }

    pub fn get_snapshot(&self) -> HashSet<KeyCode> {
        self.pressed_keys.read().clone()
    }
}