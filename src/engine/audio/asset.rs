use std::mem::MaybeUninit;
use std::ptr::null;
use std::sync::Arc;

use crate::engine::{
    prelude::Inspectable_,
    project::asset_manager::{Asset, AssetManager},
};
use id::{ID_trait, ID};
use kira::{
    manager::{backend::DefaultBackend, AudioManagerSettings},
    sound::static_sound::{StaticSoundData, StaticSoundHandle, StaticSoundSettings},
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

pub type Param = Arc<Mutex<kira::manager::AudioManager>>;

#[derive(ID, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioAsset {
    file: String,
    #[serde(skip_serializing, skip_deserializing)]
    pub(crate) d: MaybeUninit<StaticSoundData>,
    #[serde(skip_serializing, skip_deserializing)]
    m: MaybeUninit<Arc<Mutex<kira::manager::AudioManager>>>,
}
// impl Clone for AudioAsset {
//     fn clone(&self) -> Self {
//         unsafe {
//             Self {
//                 d: MaybeUninit::new(self.d.assume_init_ref().clone()),
//                 m: MaybeUninit::new(self.m.assume_init_ref().clone()),
//             }
//         }
//     }
// }

impl Default for AudioAsset {
    fn default() -> Self {
        Self {
            file: "".into(),
            d: MaybeUninit::uninit(),
            m: MaybeUninit::uninit(),
        }
    }
}

impl Asset<AudioAsset, Param> for AudioAsset {
    fn from_file(file: &str, params: &Param) -> AudioAsset {
        AudioAsset {
            file: file.into(),
            d: MaybeUninit::new(
                StaticSoundData::from_file(file, StaticSoundSettings::default()).unwrap(),
            ),
            m: MaybeUninit::new(params.clone()),
        }
    }

    fn reload(&mut self, file: &str, params: &Param) {
        *self = Self::from_file(file, params);
    }
}

impl AudioAsset {
    pub fn play(&self) -> StaticSoundHandle {
        unsafe {
            self.m
                .assume_init_ref()
                .lock()
                .play(self.d.assume_init_ref().clone())
                .unwrap()
        }
    }
    pub fn get_sound_data(&self) -> &StaticSoundData {
        unsafe { self.d.assume_init_ref() }
    }
}
impl Inspectable_ for AudioAsset {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn inspect(&mut self, ui: &mut egui::Ui, world: &mut crate::engine::World) -> bool {
        ui.horizontal(|ui| {
            ui.label(&self.file);
            if ui.button("play").clicked() {
                self.play();
            }
        });
        true
        // unsafe {
        //     let dur = self.d.assume_init_ref().duration().as_secs_f64();
        //     let settings = &mut self.d.assume_init_mut().settings;
        //     let mut looping: bool = settings.loop_region.is_some();
        //     ui.checkbox(&mut looping, "loop");
        //     if looping {
        //         *settings = settings.loop_region(0.0..dur);
        //     } else {
        //         settings.loop_region = None;
        //     }
        // }
    }
}
pub type AudioManager = AssetManager<Param, AudioAsset>;
