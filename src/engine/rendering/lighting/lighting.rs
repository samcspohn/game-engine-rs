use std::sync::Arc;

use crossbeam::queue::SegQueue;
use glm::Vec3;
use nalgebra_glm as glm;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use vulkano::{
    buffer::{allocator::SubbufferAllocator, BufferUsage, Subbuffer},
    command_buffer::{BufferCopy, CopyBufferInfo},
    memory::allocator::MemoryUsage,
    DeviceSize,
};

use crate::{
    engine::{
        atomic_vec::AtomicVec,
        prelude::*,
        project::asset_manager::AssetInstance,
        rendering::{pipeline::fs, vulkan_manager::VulkanManager},
        storage::_Storage,
    },
    // prelude::utils::PrimaryCommandBuffer,
};
use component_derive::ComponentID;

use super::{lighting_asset::LightTemplate, lighting_compute::cs};
pub struct LightingSystem {
    pub(crate) light_templates: Arc<Mutex<_Storage<fs::lightTemplate>>>,
    // pub(crate) lights: Arc<Mutex<_Storage<fs::light>>>,
    pub(crate) lights: Mutex<Subbuffer<[cs::light]>>,
    pub(crate) lights_buffer: Arc<Mutex<SubbufferAllocator>>, //Subbuffer<[super::pipeline::fs::light]>,
    pub light_inits: SegQueue<cs::light_init>,
    pub light_deinits: SegQueue<cs::light_deinit>,
    vk: Arc<VulkanManager>,
}
impl LightingSystem {
    pub fn new(vk: Arc<VulkanManager>) -> Self {
        Self {
            light_templates: Arc::new(Mutex::new(_Storage::new())),
            lights: Mutex::new(vk.buffer_array(1, MemoryUsage::Download)),
            lights_buffer: Arc::new(Mutex::new(
                vk.sub_buffer_allocator_with_usage(BufferUsage::STORAGE_BUFFER),
            )),
            light_deinits: SegQueue::new(),
            light_inits: SegQueue::new(),
            vk,
        }
    }
    pub fn get_light_buffer(
        &self,
        light_len: usize,
        builder: &mut utils::PrimaryCommandBuffer,
    ) -> (
        Subbuffer<[fs::lightTemplate]>,
        Option<Subbuffer<[cs::light_deinit]>>,
        Option<Subbuffer<[cs::light_init]>>,
    ) {
        let mut lights = self.lights.lock();
        if light_len > lights.len() as usize {
            let buf = self.vk.buffer_array(
                light_len.next_power_of_two() as DeviceSize,
                MemoryUsage::Download,
            );
            builder.copy_buffer(CopyBufferInfo::buffers(lights.clone(), buf.clone())).unwrap();
            *lights = buf;
        }
        let buf: Subbuffer<[fs::lightTemplate]> = self
            .lights_buffer
            .lock()
            .allocate_unsized(self.light_templates.lock().data.len().max(1) as DeviceSize)
            .unwrap();
        {
            let mut b = buf.write().unwrap();
            if self.light_templates.lock().data.len() > 0 {
                b[..].copy_from_slice(&self.light_templates.lock().data);
            }
        }
        let mut deinits = None;
        if self.light_deinits.len() > 0 {
            let mut v = Vec::with_capacity(self.light_deinits.len());
            while let Some(i) = self.light_deinits.pop() {
                v.push(i);
            }
            // let v = self.light_deinits.into_iter().collect::<Vec<cs::light_deinit>>();
            let buf2: Subbuffer<[cs::light_deinit]> = self
                .lights_buffer
                .lock()
                .allocate_unsized(v.len() as DeviceSize)
                .unwrap();
            {
                let mut b = buf2.write().unwrap();
                b[..].copy_from_slice(v.as_slice());
            }
            deinits = Some(buf2);
        }
        let mut inits = None;
        if self.light_inits.len() > 0 {
            let mut v = Vec::with_capacity(self.light_inits.len());
            while let Some(i) = self.light_inits.pop() {
                v.push(i);
            }
            // let v = self.light_inits.into_iter().collect::<Vec<cs::light_init>>();
            let buf2: Subbuffer<[cs::light_init]> = self
                .lights_buffer
                .lock()
                .allocate_unsized(v.len() as DeviceSize)
                .unwrap();
            {
                let mut b = buf2.write().unwrap();
                b[..].copy_from_slice(v.as_slice());
            }
            inits = Some(buf2);
        }
        (buf, deinits, inits)
    }
}

#[derive(ComponentID, Serialize, Deserialize, Default, Clone)]
#[serde(default)]
pub struct Light {
    l_id: i32,
    l_t: AssetInstance<LightTemplate>,
}
impl Light {
    pub fn new(templ: AssetInstance<LightTemplate>) -> Self {
        Self {
            l_id: -1,
            l_t: templ,
        }
    }
}

impl Component for Light {
    fn init(&mut self, transform: &Transform, id: i32, sys: &Sys) {
        // self.l_id = sys.lighting_system.lights.lock().emplace(fs::light {
        //     templ: self.l_t.id,
        //     t_id: transform.id,
        // });
        sys.lighting_system.light_inits.push(cs::light_init {
            templ_id: self.l_t.id,
            t_id: transform.id,
            id,
            p1: 0,
        });
    }
    fn deinit(&mut self, transform: &Transform, id: i32, sys: &Sys) {
        // sys.lighting_system.lights.lock().erase(self.l_id);
        sys.lighting_system
            .light_deinits
            .push(cs::light_deinit { id });
    }
    fn inspect(&mut self, transform: &Transform, id: i32, ui: &mut egui::Ui, sys: &Sys) {
        Ins(&mut self.l_t).inspect("template", ui, sys);
    }
}
