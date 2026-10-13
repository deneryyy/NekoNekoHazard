pub mod player;
pub mod world;

use std::{sync::Arc, time::Instant};

use crate::app::game::world::World;

pub struct Game {
    world: Arc<World>,
}

impl Game {
    pub fn new(world: Arc<World>) -> Self {
        Self { world }
    }

    pub fn start_game(&mut self) {
        let mut last_frame = Instant::now();
        loop {
            let now = Instant::now();
            let delta_time = now.duration_since(last_frame).as_secs_f32();
            last_frame = now;
            std::thread::sleep(std::time::Duration::from_secs_f32(0.001));
        }
    }
}
