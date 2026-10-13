use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use crossbeam_channel::{Receiver, Sender, bounded};
use glam::{IVec3, Vec3};
use parking_lot::RwLock;
use rayon::{
    ThreadPool, ThreadPoolBuilder,
    iter::{IntoParallelIterator, ParallelIterator},
};

use crate::app::{
    engine::{
        input::InputManager,
        renderer::{
            camera::Camera,
        },
    },
    game::{
        self,
        player::Player,
    },
};

pub struct World {
    thread_pool: ThreadPool,
    pub player: RwLock<Player>,
}

impl World {
    pub fn new(input_manager: Arc<InputManager>) -> Self {
        Self {
            thread_pool: ThreadPoolBuilder::new().num_threads(16).build().unwrap(),
            player: RwLock::new(Player::new(
                input_manager.clone(),
                Vec3 {
                    x: 2.0,
                    y: 2.0,
                    z: 2.0,
                },
                Camera::new(
                    Vec3 {
                        x: 2.0,
                        y: 2.0,
                        z: 2.0,
                    },
                    Vec3 {
                        x: 0.0,
                        y: 1.0,
                        z: 0.0,
                    },
                    225.0,
                    -37.0,
                    0.0,
                ),
                4,
            )),
        }
    }

    pub fn tick(&self, delta_time: f32) {
        
    }

} 