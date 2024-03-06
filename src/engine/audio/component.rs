use glium::buffer::Content;
use serde::{Deserialize, Serialize};

use crate::engine::{prelude, project::asset_manager::AssetInstance};
use prelude::*;

use super::asset::{AudioAsset, AudioManager};

#[derive(ComponentID, Clone, Deserialize, Serialize, Default)]
#[serde(default)]
pub struct AudioSource {
    sound: AssetInstance<AudioAsset>,
}
impl Component for AudioSource {
    fn inspect(&mut self, transform: &Transform, id: i32, ui: &mut egui::Ui, sys: &Sys) {
        // let temp = self.sound.clone();
        Ins(&mut self.sound).inspect("sound", ui, sys);
    }
    // fn init(&mut self, transform: &Transform, id: i32, sys: &Sys) {
    //     let audio_asset_manager = sys.assets.get_manager::<AudioAsset>();
    //     let guard = audio_asset_manager.lock();
    //     let audio_asset_manager_downcast: &AudioManager =
    //         unsafe { guard.as_any().downcast_ref_unchecked() };
    //     if let Some(sound_data) = audio_asset_manager_downcast.assets_id.get(&self.sound.id) {
    //         unsafe {
    //             sys.audio
    //                 .lock()
    //                 .play(sound_data.lock().d.assume_init_ref().clone());
    //         }
    //     }
    //     // self.sound.
    // }
    fn on_start(&mut self, transform: &Transform, sys: &System) {
        let audio_asset_manager = sys.assets.get_manager::<AudioAsset>();
        let guard = audio_asset_manager.lock();
        let audio_asset_manager_downcast: &AudioManager =
            unsafe { guard.as_any().downcast_ref_unchecked() };
        if let Some(sound_data) = audio_asset_manager_downcast.assets_id.get(&self.sound.id) {
            unsafe { sys.audio.lock().play(sound_data.lock().d.assume_init_ref().clone()); }
        }
        // self.sound.
    }
}
