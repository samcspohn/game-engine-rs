use std::{default, time::Duration};
use id::*;
use glium::buffer::Content;
use kira::{
    sound::static_sound::{StaticSoundData, StaticSoundHandle, StaticSoundSettings},
    spatial::{
        emitter::{EmitterHandle, EmitterSettings},
        listener::{ListenerHandle, ListenerSettings},
    },
    tween::Tween,
};
use serde::{Deserialize, Serialize};

use crate::engine::{prelude, project::asset_manager::AssetInstance};
use prelude::*;

use super::asset::{AudioAsset, AudioManager};

#[derive(ID, Deserialize, Serialize, Default)]
#[serde(default)]
pub struct AudioSource {
    id: AssetInstance<AudioAsset>,
    #[serde(skip_serializing, skip_deserializing)]
    data: Option<StaticSoundData>,
    #[serde(skip_serializing, skip_deserializing)]
    emitter: Option<EmitterHandle>,
    looping: bool,
    // #[serde(skip_serializing, skip_deserializing)]
    //     sound_handle: Option<StaticSoundHandle>
}
impl Clone for AudioSource {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            data: None,
            emitter: None,
            looping: self.looping,
            // sound_handle: None,
        }
    }
}
impl Component for AudioSource {
    fn inspect(&mut self, transform: &Transform, id: i32, ui: &mut egui::Ui, sys: &Sys) {
        // let temp = self.sound.clone();
        Ins(&mut self.id).inspect("sound", ui, sys);
        Ins(&mut self.looping).inspect("looping", ui, sys);
    }

    fn on_start(&mut self, transform: &Transform, sys: &System) {
        sys.assets.get_manager(|audio: &AudioManager| {
            if let Some(sound_data) = audio.assets_id.get(&self.id.id) {
                let pos: [f32; 3] = transform.get_position().into();
                let emitter = sys
                    .audio
                    .scene
                    .lock()
                    .add_emitter(pos, EmitterSettings::default())
                    .unwrap();
                unsafe {
                    let asset = sound_data.lock();
                    let dur = asset.get_sound_data().duration().as_secs_f64();
                    let mut m_sound = asset.get_sound_data().with_modified_settings(|settings| {
                        let mut new_settings = settings.output_destination(&emitter);
                        if self.looping {
                            new_settings = new_settings.loop_region(0.0..dur);
                        } else {
                            new_settings.loop_region = None;
                        }
                        new_settings
                    });
                    self.data = Some(m_sound);
                    self.emitter = Some(emitter);
                    sys.audio.m.lock().play(self.data.as_ref().unwrap().clone());
                }
            }
        });
    }
    fn update(&mut self, transform: &Transform, sys: &System, world: &crate::engine::World) {
        if let (Some(mut sound), Some(mut emitter)) = (self.data.as_mut(), self.emitter.as_mut()) {
            let pos: [f32; 3] = transform.get_position().into();
            emitter
                .set_position(
                    pos,
                    Tween {
                        start_time: kira::StartTime::Immediate,
                        duration: Duration::from_secs_f32(sys.time.dt),
                        ..Default::default()
                    },
                )
                .unwrap();
        } else {
            self.on_start(transform, sys);
        }
    }
    // fn deinit(&mut self, transform: &Transform, _id: i32, sys: &Sys) {
    //     if let (Some(mut sound), Some(mut emitter)) = (self.data.as_mut(), self.emitter.as_mut()) {
    //         sound
    //     }
    // }
}

#[derive(ID, Deserialize, Serialize, Default)]
#[serde(default)]
pub struct AudioListener {
    #[serde(skip_serializing, skip_deserializing)]
    l: Option<ListenerHandle>,
}

impl Clone for AudioListener {
    fn clone(&self) -> Self {
        Self { l: None }
    }
}

impl Component for AudioListener {
    fn inspect(&mut self, transform: &Transform, id: i32, ui: &mut egui::Ui, sys: &Sys) {}
    fn on_start(&mut self, transform: &Transform, sys: &System) {
        unsafe {
            let pos: [f32; 3] = transform.get_position().into();
            let rot: [f32; 4] = transform.get_rotation().coords.into();
            let listener = sys
                .audio
                .scene
                .lock()
                .add_listener(pos, rot, ListenerSettings::default())
                .unwrap();
            self.l = Some(listener);
        }
    }
    fn update(&mut self, transform: &Transform, sys: &System, world: &crate::engine::World) {
        if let Some(mut listener) = self.l.as_mut() {
            let pos: [f32; 3] = transform.get_position().into();
            let rot: [f32; 4] = transform.get_rotation().coords.into();
            let t = Tween {
                start_time: kira::StartTime::Immediate,
                duration: Duration::from_secs_f32(sys.time.dt),
                ..Default::default()
            };
            listener.set_position(pos, t.clone()).unwrap();
            listener.set_orientation(rot, t).unwrap()
        } else {
            self.on_start(transform, sys);
        }
    }
}
