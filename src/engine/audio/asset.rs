use std::mem::MaybeUninit;
use std::ptr::null;
use std::sync::Arc;

use crate::engine::project::asset_manager::_AssetID;
use crate::engine::{
    prelude::Inspectable_,
    project::asset_manager::{Asset, AssetManager},
};
use component_derive::AssetID;
use kira::{
    manager::{backend::DefaultBackend, AudioManagerSettings},
    sound::static_sound::{StaticSoundData, StaticSoundHandle, StaticSoundSettings},
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

pub type Param = Arc<Mutex<kira::manager::AudioManager>>;

#[derive(AssetID, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioAsset {
    #[serde(skip_serializing, skip_deserializing)]
    pub(crate) d: MaybeUninit<StaticSoundData>,
    #[serde(skip_serializing, skip_deserializing)]
    m: MaybeUninit<Arc<Mutex<kira::manager::AudioManager>>>,
}
impl Clone for AudioAsset {
    fn clone(&self) -> Self {
        unsafe {
            Self {
                d: MaybeUninit::new(self.d.assume_init_ref().clone()),
                m: MaybeUninit::new(self.m.assume_init_ref().clone()),
            }
        }
    }
}

impl Default for AudioAsset {
    fn default() -> Self {
        Self {
            d: MaybeUninit::uninit(),
            m: MaybeUninit::uninit(),
        }
    }
}

impl Asset<AudioAsset, Param> for AudioAsset {
    fn from_file(file: &str, params: &Param) -> AudioAsset {
        AudioAsset {
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
}
impl Inspectable_ for AudioAsset {
    fn inspect(&mut self, ui: &mut egui::Ui, world: &mut crate::engine::World) {}
}
pub type AudioManager = AssetManager<Param, AudioAsset>;
