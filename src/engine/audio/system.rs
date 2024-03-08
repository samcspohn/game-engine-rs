use std::sync::Arc;

use kira::{
    manager::{backend::DefaultBackend, AudioManager, AudioManagerSettings},
    spatial::scene::{SpatialSceneHandle, SpatialSceneSettings},
};
use parking_lot::Mutex;

#[derive(Clone)]
pub struct AudioSystem {
    pub m: Arc<Mutex<AudioManager>>,
    pub scene: Arc<Mutex<SpatialSceneHandle>>,
}

impl AudioSystem {
    pub fn new() -> Self {
        let m = Arc::new(Mutex::new(
            AudioManager::<DefaultBackend>::new(AudioManagerSettings::default()).unwrap(),
        ));
        let scene = Arc::new(Mutex::new(m.lock().add_spatial_scene(SpatialSceneSettings::default()).unwrap()));
        Self { m, scene }
    }
}
