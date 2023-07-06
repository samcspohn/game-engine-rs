use std::sync::{
    atomic::{AtomicI32, AtomicUsize, Ordering},
    Arc,
};

use crate::{
    color_gradient::ColorGradient,
    editor::inspectable::{Inpsect, Ins, Inspectable, Inspectable_},
    engine::{world::{transform::Transform, Sys, World}, component::{Component, _ComponentID}, storage::_Storage, project::asset_manager::{AssetInstance, Asset, AssetManager, AssetManagerBase}, rendering::{vulkan_manager::VulkanManager, renderer_component::buffer_usage_all}, transform_compute::cs::ty::transform},
};
// use lazy_static::lazy::Lazy;

use component_derive::{ComponentID, AssetID};
use nalgebra_glm as glm;
use parking_lot::{Mutex, MutexGuard};
use serde::{Deserialize, Serialize};
use sync_unsafe_cell::SyncUnsafeCell;
use vulkano::{
    buffer::{
        cpu_pool::CpuBufferPoolSubbuffer, CpuAccessibleBuffer, CpuBufferPool, DeviceLocalBuffer,
        TypedBufferAccess,
    },
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, DispatchIndirectCommand, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::Device,
    format::Format,
    image::{view::ImageView, ImageDimensions, ImmutableImage, MipmapsCount},
    memory::allocator::{MemoryUsage, StandardMemoryAllocator},
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            depth_stencil::{CompareOp, DepthState, DepthStencilState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            rasterization::{CullMode, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        ComputePipeline, GraphicsPipeline, Pipeline, PipelineBindPoint, StateMode,
    },
    render_pass::{RenderPass, Subpass},
    sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo},
    sync::{self, FlushError, GpuFuture},
    DeviceSize,
};

use self::cs::ty::{emitter_init, particle_template};

pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/particle_shaders/particles.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/particle_shaders/particles.vert",
        types_meta: {


            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
pub mod gs {
    vulkano_shaders::shader! {
        ty: "geometry",
        path: "src/particle_shaders/particles.geom",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/particle_shaders/particles.frag"
    }
}

pub const MAX_PARTICLES: i32 = 1024 * 1024 * 8 * 2;
// pub const NUM_EMITTERS: i32 = 1_200_000;



// #[component]
#[derive(ComponentID, Clone, Deserialize, Serialize)]
#[serde(default)]
#[repr(C)]

pub struct ParticleEmitter {
    template: AssetInstance<ParticleTemplate>,
}
impl Default for ParticleEmitter {
    fn default() -> Self {
        Self { template: AssetInstance::new(0) }
    }
}

// impl Default for cs::ty::particle_template {
//     fn default() -> Self {
//         gen_particle_template(&ParticleTemplate::default())
//     }
// }

impl Inspectable for ParticleEmitter {
    fn inspect(&mut self, _transform: &Transform, _id: i32, ui: &mut egui::Ui, sys: &Sys) {
        // ui.add(egui::DragValue::new(&mut self.template));
        // ui.add(egui::Label::new("Particle Emitter"));
        // egui::CollapsingHeader::new("Particle Emitter")
        //     .default_open(true)
        //     .show(ui, |ui| {
        Ins(&mut self.template).inspect("template", ui, sys);
        // });
    }
}
impl ParticleEmitter {
    pub fn new(template: i32) -> ParticleEmitter {
        let inst = AssetInstance::<ParticleTemplate>::new(template);
        ParticleEmitter { template: inst }
    }
}

impl Component for ParticleEmitter {
    fn init(&mut self, transform: &Transform, id: i32, sys: &Sys) {
        let d = cs::ty::emitter_init {
            transform_id: transform.id,
            alive: 1,
            template_id: self.template.id,
            e_id: id,
        };
        match sys.particles_system.emitter_inits.try_push(d) {
            None => {}
            Some(i) => {
                sys.particles_system.emitter_inits.push(i, d);
            }
        }
    }
    fn deinit(&mut self, transform: &Transform, id: i32, sys: &Sys) {
        let d = cs::ty::emitter_init {
            transform_id: transform.id,
            alive: 0,
            template_id: self.template.id,
            e_id: id,
        };

        match sys.particles_system.emitter_inits.try_push(d) {
            None => {}
            Some(i) => {
                sys.particles_system.emitter_inits.push(i, d);
            }
        }
    }
}

use crate::engine::project::asset_manager::_AssetID;

use super::particle_sort::ParticleSort;
#[derive(AssetID, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ParticleTemplate {
    color: [f32; 4],
    speed: f32,
    emission_rate: f32,
    life_time: f32,
    color_over_life: ColorGradient,
    trail: bool,
}

impl Default for ParticleTemplate {
    fn default() -> Self {
        Self {
            color: [1.; 4],
            speed: 1.,
            emission_rate: 10.,
            life_time: 1.,
            color_over_life: ColorGradient::new(),
            trail: false,
        }
    }
}

fn gen_particle_template(t: &ParticleTemplate) -> particle_template {
    particle_template {
        color: t.color,
        speed: t.speed,
        emission_rate: t.emission_rate,
        life_time: t.life_time,
        // color_life: [[f32;4];200],
        color_life: t.color_over_life.to_color_array(),
        trail: if t.trail { 1 } else { 0 },
        _dummy0: Default::default(),
        size: 1f32,
    }
}

fn field<F>(ui: &mut egui::Ui, name: &str, func: F)
where
    F: FnOnce(&mut egui::Ui),
{
    ui.horizontal(|ui| {
        ui.add(egui::Label::new(name));
        func(ui);
        // ui.add(egui::DragValue::new(self.0));s
    });
}
impl Inspectable_ for ParticleTemplate {
    fn inspect(&mut self, ui: &mut egui::Ui, _world: &parking_lot::Mutex<World>) {
        field(ui, "color", |ui| {
            ui.color_edit_button_rgba_premultiplied(&mut self.color);
        });
        field(ui, "emission rate", |ui| {
            ui.add(egui::DragValue::new(&mut self.emission_rate));
        });
        field(ui, "speed", |ui| {
            ui.add(egui::DragValue::new(&mut self.speed));
        });
        field(ui, "life time", |ui| {
            ui.add(egui::DragValue::new(&mut self.life_time));
        });
        field(ui, "color over life", |ui| {
            // static mut  cg: Lazy<ColorGradient> = Lazy::new( || ColorGradient::new());
            self.color_over_life.edit(ui);
        });
        // ui.add()
        // field(ui, "trail", |ui| {
        ui.checkbox(&mut self.trail, "trail");
        // })
        // ui.color_edit_button_rgba_premultiplied(&mut self.color);
        // ui.add(egui::DragValue::new(&mut self.emission_rate));
        // ui.add(egui::DragValue::new(&mut self.speed));
        // ui.label(self.color)
    }
}

impl Asset<ParticleTemplate, Arc<Mutex<_Storage<cs::ty::particle_template>>>> for ParticleTemplate {
    fn from_file(file: &str, params: &Arc<Mutex<_Storage<particle_template>>>) -> ParticleTemplate {
        let mut t = ParticleTemplate::default();
        if let Ok(s) = std::fs::read_to_string(file) {
            t = serde_yaml::from_str(s.as_str()).unwrap();
        }
        // let p_t = particle_template { color: t.color, speed: t.speed, emission_rate: t.emission_rate, _dummy0: Default::default() };
        let p_t = gen_particle_template(&t);
        params.lock().emplace(p_t);
        t
    }

    fn reload(&mut self, file: &str, _params: &Arc<Mutex<_Storage<particle_template>>>) {
        // let mut t = ParticleTemplate::default();
        if let Ok(s) = std::fs::read_to_string(file) {
            *self = serde_yaml::from_str(s.as_str()).unwrap();
        }
        // let id = self.id;
        // *params.lock().get_mut(&id) = gen_particle_template(&t);
        // *self = t;
        // self.id = id;
        // let p_t = particle_template { color: t.color, speed: t.speed, emission_rate: t.emission_rate, _dummy0: Default::default() };
        // t.id = params.emplace(p_t);
        // t
    }
    fn save(&mut self, file: &str, _params: &Arc<Mutex<_Storage<particle_template>>>) {
        if let Ok(s) = serde_yaml::to_string(self) {
            match std::fs::write(file, s.as_bytes()) {
                Ok(_) => (),
                Err(a) => {println!("{}: failed for file: {}",a, file)}
            }
        }
    }
    fn new(
        _file: &str,
        params: &Arc<Mutex<_Storage<cs::ty::particle_template>>>,
    ) -> Option<ParticleTemplate> {
        let t = ParticleTemplate::default();
        // if let Ok(s) = std::fs::read_to_string(file) {
        //     t = serde_yaml::from_str(s.as_str()).unwrap();
        // }
        // let p_t = particle_template { color: t.color, speed: t.speed, emission_rate: t.emission_rate, _dummy0: Default::default() };
        let p_t = gen_particle_template(&t);
        params.lock().emplace(p_t);
        Some(t)
    }
}

pub type ParticleTemplateManager =
    AssetManager<Arc<Mutex<_Storage<cs::ty::particle_template>>>, ParticleTemplate>;

pub struct PerformanceCounters {
    pub update_particles: i32,
    pub update_emitters: i32,
    pub init_emitters: i32,
    pub sort_particles: i32,
}
pub struct ParticleBuffers {
    pub particles: Arc<DeviceLocalBuffer<[cs::ty::particle]>>,
    pub particle_positions_lifes: Arc<DeviceLocalBuffer<[cs::ty::pos_lif]>>,
    pub particle_next: Arc<DeviceLocalBuffer<[i32]>>,
    pub particle_template_ids: Arc<DeviceLocalBuffer<[i32]>>,
    pub alive: Arc<DeviceLocalBuffer<[u32]>>,
    pub alive_count: Arc<DeviceLocalBuffer<i32>>,
    pub emitters: Mutex<Arc<DeviceLocalBuffer<[cs::ty::emitter]>>>,
    pub particle_template: Mutex<Arc<DeviceLocalBuffer<[cs::ty::particle_template]>>>,
    pub avail: Arc<DeviceLocalBuffer<[u32]>>,
    pub avail_count: Arc<DeviceLocalBuffer<i32>>,
    pub buffer_0: Arc<DeviceLocalBuffer<i32>>,
    pub indirect: Arc<DeviceLocalBuffer<[DispatchIndirectCommand]>>,
    pub alive_b: Arc<DeviceLocalBuffer<[cs::ty::b]>>,
}

pub struct AtomicVec<T: Copy> {
    lock: Mutex<()>,
    data: SyncUnsafeCell<Vec<T>>,
    index: AtomicUsize,
}
impl<T: Copy> AtomicVec<T> {
    pub fn new() -> Self {
        let mut data = Vec::with_capacity(4);
        unsafe {
            data.set_len(4);
        }
        Self {
            lock: Mutex::new(()),
            data: SyncUnsafeCell::new(data),
            index: AtomicUsize::new(0),
        }
    }
    pub fn push_multi<'a>(&mut self, count: usize) -> ( MutexGuard<()>, &'a [T]) {
        let _l = self.lock.lock();
        unsafe {
            let index = self.index.load(Ordering::Relaxed);
            (*self.data.get()).reserve(count);
            (*self.data.get()).set_len(index + count);
            self.index.fetch_add(count, Ordering::Relaxed);
            (_l, &(*self.data.get())[index..(index+count)])
        }
        
    }
    pub fn try_push(&self, d: T) -> Option<usize> {
        let index = self.index.fetch_add(1, Ordering::Relaxed);
        if index >= unsafe { (*self.data.get()).len() } {
            Some(index)
        } else {
            unsafe { (*self.data.get())[index] = d };
            None
        }
    }
    pub fn push(&self, i: usize, d: T) {
        // let index = self.index.fetch_add(1, Ordering::Relaxed);
        unsafe {
            let _l = self.lock.lock();
            if i >= (*self.data.get()).len() {
                let data = &mut (*self.data.get());
                let len = data.len() + 1;
                let new_len = (data.len() + 1).next_power_of_two();
                data.reserve_exact(new_len - len + 1);
                data.set_len(new_len);
            }
            (*self.data.get())[i] = d;
            // drop(l);
        };
    }
    pub fn get_vec(&self) -> Vec<T> {
        let _l = self.lock.lock();
        // let mut v = Vec::new();
        let len = self.index.load(Ordering::Relaxed);
        // unsafe {
        //     let cap = (*self.data.get()).capacity();
        //     std::mem::swap(&mut v, &mut *self.data.get());
        //     v.set_len(len);
        //     *self.data.get() = Vec::with_capacity(cap);
        //     (*self.data.get()).set_len(cap);
        // }
        // v
        let mut v = Vec::with_capacity(len);
        for i in (0..len).into_iter() {
            unsafe { v.push((*self.data.get())[i]) };
        }
        self.index.store(0, Ordering::Relaxed);
        v
    }
}

pub struct ParticleCompute {
    pub sort: ParticleSort,
    pub emitter_inits: AtomicVec<cs::ty::emitter_init>,
    pub particle_templates: Arc<Mutex<_Storage<cs::ty::particle_template>>>,
    pub particle_template_manager: Arc<Mutex<ParticleTemplateManager>>,
    pub particle_buffers: ParticleBuffers,
    pub compute_pipeline: Arc<ComputePipeline>,
    pub compute_uniforms: CpuBufferPool<cs::ty::Data>,
    pub render_uniforms: CpuBufferPool<gs::ty::Data>,
    pub def_texture: Arc<ImageView<ImmutableImage>>,
    pub def_sampler: Arc<Sampler>,
    pub vk: Arc<VulkanManager>,
    pub performance: PerformanceCounters,
}
pub struct ParticleRenderPipeline {
    pub arc: Arc<GraphicsPipeline>,
}
impl ParticleRenderPipeline {
    pub fn new(vk: Arc<VulkanManager>, render_pass: Arc<RenderPass>) -> Self {
        let vs = vs::load(vk.device.clone()).unwrap();
        let fs = fs::load(vk.device.clone()).unwrap();
        let gs = gs::load(vk.device.clone()).unwrap();
        let subpass = Subpass::from(render_pass, 0).unwrap();
        let blend_state = ColorBlendState::new(subpass.num_color_attachments()).blend_alpha();
        let mut depth_stencil_state = DepthStencilState::simple_depth_test();
        depth_stencil_state.depth = Some(DepthState {
            enable_dynamic: false,
            write_enable: StateMode::Fixed(false),
            compare_op: StateMode::Fixed(CompareOp::Less),
        });

        let render_pipeline = GraphicsPipeline::start()
            .vertex_input_state(BuffersDefinition::new())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            // .input_assembly_state(InputAssemblyState::new())
            .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::PointList))
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .geometry_shader(gs.entry_point("main").unwrap(), ())
            // .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::LineStrip))
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
            .depth_stencil_state(depth_stencil_state)
            .color_blend_state(blend_state)
            .render_pass(subpass)
            .build(vk.device.clone())
            .unwrap();
        Self {
            arc: render_pipeline,
        }
    }
}

impl ParticleCompute {
    pub fn new(device: Arc<Device>, vk: Arc<VulkanManager>) -> ParticleCompute {
        let performance = PerformanceCounters {
            update_particles: vk.new_query(),
            update_emitters: vk.new_query(),
            init_emitters: vk.new_query(),
            sort_particles: vk.new_query(),
        };

        let particles = DeviceLocalBuffer::<[cs::ty::particle]>::array(
            &vk.mem_alloc,
            MAX_PARTICLES as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        let particle_template_ids = DeviceLocalBuffer::<[i32]>::array(
            &vk.mem_alloc,
            MAX_PARTICLES as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        let mut builder = AutoCommandBufferBuilder::primary(
            &vk.comm_alloc,
            vk.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        println!("pos_lif: {}", std::mem::size_of::<cs::ty::pos_lif>());
        println!("b: {}", std::mem::size_of::<cs::ty::b>());

        let particle_positions_lifes: Vec<cs::ty::pos_lif> = (0..MAX_PARTICLES)
            .map(|_| cs::ty::pos_lif {
                pos: [0., 0., 0.],
                life: 0.,
            })
            .collect();
        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &vk.mem_alloc,
            buffer_usage_all(),
            false,
            particle_positions_lifes,
        )
        .unwrap();
        let particle_positions_lifes = DeviceLocalBuffer::<[cs::ty::pos_lif]>::array(
            &vk.mem_alloc,
            MAX_PARTICLES as vulkano::DeviceSize,
            buffer_usage_all(),
            vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_buffer,
                particle_positions_lifes.clone(),
            ))
            .unwrap();

        let particle_next = DeviceLocalBuffer::<[i32]>::array(
            &vk.mem_alloc,
            MAX_PARTICLES as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        // avail
        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &vk.mem_alloc,
            buffer_usage_all(),
            false,
            0..MAX_PARTICLES,
        )
        .unwrap();
        let avail = DeviceLocalBuffer::<[u32]>::array(
            &vk.mem_alloc,
            MAX_PARTICLES as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, avail.clone()))
            .unwrap();

        // alive_b
        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &vk.mem_alloc,
            buffer_usage_all(),
            false,
            (0..MAX_PARTICLES).map(|_| 0),
        )
        .unwrap();
        let alive_b = DeviceLocalBuffer::<[cs::ty::b]>::array(
            &vk.mem_alloc,
            MAX_PARTICLES as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, alive_b.clone()))
            .unwrap();
        let alive = DeviceLocalBuffer::<[u32]>::array(
            &vk.mem_alloc,
            MAX_PARTICLES as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        // avail_count
        let copy_buffer =
            CpuAccessibleBuffer::from_data(&vk.mem_alloc, buffer_usage_all(), false, 0i32).unwrap();
        let avail_count = DeviceLocalBuffer::<i32>::new(
            &vk.mem_alloc,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        let alive_count = DeviceLocalBuffer::<i32>::new(
            &vk.mem_alloc,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        // buffer 0
        let buffer_0 = DeviceLocalBuffer::<i32>::new(
            &vk.mem_alloc,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_buffer.clone(),
                avail_count.clone(),
            ))
            .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_buffer.clone(),
                alive_count.clone(),
            ))
            .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, buffer_0.clone()))
            .unwrap();

        // emitters
        let emitters = DeviceLocalBuffer::<[cs::ty::emitter]>::array(
            &vk.mem_alloc,
            1 as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        // indirect
        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &vk.mem_alloc,
            buffer_usage_all(),
            false,
            [DispatchIndirectCommand { x: 0, y: 1, z: 1 }],
        )
        .unwrap();
        let indirect = DeviceLocalBuffer::<[DispatchIndirectCommand]>::array(
            &vk.mem_alloc,
            1 as DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, indirect.clone()))
            .unwrap();

        let particle_templates = _Storage::new();
        let particle_templates = Arc::new(Mutex::new(particle_templates));
        let particle_template_manager = Arc::new(Mutex::new(ParticleTemplateManager::new(
            particle_templates.clone(),
            &["ptem"],
        )));
        particle_template_manager.lock().new_asset("res/default.ptem");

        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &vk.mem_alloc,
            buffer_usage_all(),
            false,
            particle_templates.lock().data.clone(),
        )
        .unwrap();
        let templates = DeviceLocalBuffer::<[cs::ty::particle_template]>::array(
            &vk.mem_alloc,
            copy_buffer.len() as vulkano::DeviceSize,
            buffer_usage_all(),
            device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, templates.clone()))
            .unwrap();

        let def_texture = {
            let dimensions = ImageDimensions::Dim2d {
                width: 1,
                height: 1,
                array_layers: 1,
            };
            let image_data = vec![255_u8, 255, 255, 255];
            let image = ImmutableImage::from_iter(
                &vk.mem_alloc,
                image_data,
                dimensions,
                MipmapsCount::One,
                Format::R8G8B8A8_SRGB,
                &mut builder,
            )
            .unwrap();

            ImageView::new_default(image).unwrap()
        };
        let def_sampler = Sampler::new(
            device.clone(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                ..Default::default()
            },
        )
        .unwrap();

        // build buffer
        let command_buffer = builder.build().unwrap();

        let execute = sync::now(device.clone())
            .boxed()
            .then_execute(vk.queue.clone(), command_buffer);

        match execute {
            Ok(execute) => {
                let future = execute.then_signal_fence_and_flush();
                match future {
                    Ok(_) => {}
                    Err(FlushError::OutOfDate) => {}
                    Err(_e) => {}
                }
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
            }
        };
        let cs = cs::load(device.clone()).unwrap();
        let compute_pipeline = vulkano::pipeline::ComputePipeline::new(
            device.clone(),
            cs.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create compute shader");

        let uniforms = CpuBufferPool::<cs::ty::Data>::new(
            vk.mem_alloc.clone(),
            buffer_usage_all(),
            MemoryUsage::Upload,
        );
        let render_uniforms = CpuBufferPool::<gs::ty::Data>::new(
            vk.mem_alloc.clone(),
            buffer_usage_all(),
            MemoryUsage::Upload,
        );
        ParticleCompute {
            sort: ParticleSort::new(
                device,
                vk.queue.clone(),
                vk.mem_alloc.clone(),
                &vk.comm_alloc,
                vk.desc_alloc.clone(),
            ),
            emitter_inits: AtomicVec::new(),
            particle_templates,
            particle_template_manager,
            particle_buffers: ParticleBuffers {
                particles,
                particle_next,
                particle_positions_lifes,
                particle_template_ids,
                emitters: Mutex::new(emitters),
                particle_template: Mutex::new(templates),
                alive,
                alive_count,
                avail,
                avail_count,
                buffer_0,
                indirect,
                alive_b,
            },
            // render_pipeline,
            compute_pipeline,
            compute_uniforms: uniforms,
            render_uniforms,
            def_texture,
            def_sampler,
            vk,
            performance,
        }
    }
    pub fn update(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        emitter_inits: (usize, Vec<emitter_init>),
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        dt: f32,
        time: f32,
        cam_pos: [f32; 3],
        cam_rot: [f32; 4],
    ) {
        self.emitter_init(
            builder,
            transform.clone(),
            emitter_inits.1.clone(),
            emitter_inits.0,
            dt,
            time,
            cam_pos,
            cam_rot,
        );
        self.emitter_update(
            builder,
            transform.clone(),
            emitter_inits.0,
            dt,
            time,
            cam_pos,
            cam_rot,
        );
        self.particle_update(builder, transform, dt, time, cam_pos, cam_rot);
    }
    pub fn emitter_init(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        emitter_inits: Vec<cs::ty::emitter_init>,
        emitter_len: usize,
        dt: f32,
        time: f32,
        cam_pos: [f32; 3],
        cam_rot: [f32; 4],
    ) {
        // let mut emitter_inits = emitter_inits.lock();

        let pb = &self.particle_buffers;
        {
            let mut pt = self.particle_templates.lock();
            for (id, a) in self.particle_template_manager.lock().assets_id.iter() {
                let a = a.lock();
                *pt.get_mut(id) = gen_particle_template(&a);
            }
        }

        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &self.vk.mem_alloc,
            buffer_usage_all(),
            false,
            self.particle_templates.lock().data.clone(),
        )
        .unwrap();

        if copy_buffer.len() > pb.particle_template.lock().len() {
            *pb.particle_template.lock() = DeviceLocalBuffer::<[cs::ty::particle_template]>::array(
                &self.vk.mem_alloc,
                copy_buffer.len() as vulkano::DeviceSize,
                buffer_usage_all(),
                self.vk.device.active_queue_family_indices().iter().copied(),
            )
            .unwrap();
        }
        builder
            .copy_buffer(CopyBufferInfo::buffers(
                copy_buffer,
                pb.particle_template.lock().clone(),
            ))
            .unwrap();

        if emitter_inits.is_empty() {
            return;
        };
        let len = emitter_inits.len();
        // let mut ei = Vec::<emitter_init>::new();
        // std::mem::swap(&mut ei, &mut emitter_inits);

        let copy_buffer = CpuAccessibleBuffer::from_iter(
            &self.vk.mem_alloc,
            buffer_usage_all(),
            false,
            emitter_inits,
        )
        .unwrap();
        let emitter_inits = DeviceLocalBuffer::<[cs::ty::emitter_init]>::array(
            &self.vk.mem_alloc,
            len as vulkano::DeviceSize,
            buffer_usage_all(),
            self.vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap(); // TODO: cache

        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, emitter_inits.clone()))
            .unwrap();

        // let max_len = (emitter_len as f32 + 1.).log2().ceil();
        let max_len = emitter_len.next_power_of_two();
        // let mut self_emitters = self.emitters.clone();
        if pb.emitters.lock().len() < max_len as u64 {
            let emitters = DeviceLocalBuffer::<[cs::ty::emitter]>::array(
                &self.vk.mem_alloc,
                max_len as vulkano::DeviceSize,
                buffer_usage_all(),
                self.vk.device.active_queue_family_indices().iter().copied(),
            )
            .unwrap();

            builder
                .copy_buffer(CopyBufferInfo::buffers(
                    pb.emitters.lock().clone(),
                    emitters.clone(),
                ))
                .unwrap();
            *pb.emitters.lock() = emitters;
        }

        let uniform_sub_buffer = {
            let uniform_data = cs::ty::Data {
                num_jobs: len as i32,
                dt,
                time,
                stage: 0,
                cam_pos,
                cam_rot,
                MAX_PARTICLES,
                _dummy0: Default::default(),
            };
            self.compute_uniforms.from_data(uniform_data).unwrap()
        };
        let descriptor_set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.compute_pipeline
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, transform),
                WriteDescriptorSet::buffer(1, pb.particles.clone()),
                WriteDescriptorSet::buffer(2, pb.particle_positions_lifes.clone()),
                WriteDescriptorSet::buffer(3, pb.particle_next.clone()),
                WriteDescriptorSet::buffer(4, pb.avail.clone()),
                WriteDescriptorSet::buffer(5, pb.emitters.lock().clone()),
                WriteDescriptorSet::buffer(6, pb.avail_count.clone()),
                WriteDescriptorSet::buffer(7, pb.particle_template.lock().clone()),
                WriteDescriptorSet::buffer(8, uniform_sub_buffer),
                WriteDescriptorSet::buffer(9, pb.particle_template_ids.clone()),
                WriteDescriptorSet::buffer(10, emitter_inits),
                WriteDescriptorSet::buffer(11, pb.alive.clone()),
                WriteDescriptorSet::buffer(12, pb.alive_count.clone()),
                WriteDescriptorSet::buffer(13, pb.indirect.clone()),
                WriteDescriptorSet::buffer(14, pb.alive_b.clone()),
            ],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set,
            );
        // self.vk.query(&self.performance.init_emitters, builder);
        builder.dispatch([len as u32 / 1024 + 1, 1, 1]).unwrap();
        // self.vk.end_query(&self.performance.init_emitters, builder)
    }

    pub fn emitter_update(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        emitter_len: usize,
        dt: f32,
        time: f32,
        cam_pos: [f32; 3],
        cam_rot: [f32; 4],
    ) {
        let pb = &self.particle_buffers;

        let emitter_len = emitter_len.max(1);
        let emitter_inits = DeviceLocalBuffer::<[cs::ty::emitter_init]>::array(
            &self.vk.mem_alloc,
            1 as vulkano::DeviceSize,
            buffer_usage_all(),
            self.vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();

        let uniform_sub_buffer = {
            let uniform_data = cs::ty::Data {
                num_jobs: emitter_len as i32,
                dt,
                time,
                stage: 1,
                cam_pos,
                cam_rot,
                MAX_PARTICLES,
                _dummy0: Default::default(),
            };
            self.compute_uniforms.from_data(uniform_data).unwrap()
        };
        let descriptor_set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.compute_pipeline
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, transform),
                WriteDescriptorSet::buffer(1, pb.particles.clone()),
                WriteDescriptorSet::buffer(2, pb.particle_positions_lifes.clone()),
                WriteDescriptorSet::buffer(3, pb.particle_next.clone()),
                WriteDescriptorSet::buffer(4, pb.avail.clone()),
                WriteDescriptorSet::buffer(5, pb.emitters.lock().clone()),
                WriteDescriptorSet::buffer(6, pb.avail_count.clone()),
                WriteDescriptorSet::buffer(7, pb.particle_template.lock().clone()),
                WriteDescriptorSet::buffer(8, uniform_sub_buffer),
                WriteDescriptorSet::buffer(9, pb.particle_template_ids.clone()),
                WriteDescriptorSet::buffer(10, emitter_inits),
                WriteDescriptorSet::buffer(11, pb.alive.clone()),
                WriteDescriptorSet::buffer(12, pb.alive_count.clone()),
                WriteDescriptorSet::buffer(13, pb.indirect.clone()),
                WriteDescriptorSet::buffer(14, pb.alive_b.clone()),
            ],
        )
        .unwrap();

        builder
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .dispatch([emitter_len as u32 / 1024 + 1, 1, 1])
            .unwrap();
    }

    pub fn particle_update(
        &self,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        dt: f32,
        time: f32,
        cam_pos: [f32; 3],
        cam_rot: [f32; 4],
    ) {
        let pb = &self.particle_buffers;

        let emitter_inits = DeviceLocalBuffer::<[cs::ty::emitter_init]>::array(
            &self.vk.mem_alloc,
            1 as vulkano::DeviceSize,
            buffer_usage_all(),
            self.vk.device.active_queue_family_indices().iter().copied(),
        )
        .unwrap();
        let mut uniform_data = cs::ty::Data {
            num_jobs: MAX_PARTICLES,
            dt,
            time,
            stage: 2,
            cam_pos,
            cam_rot,
            MAX_PARTICLES,
            _dummy0: Default::default(),
        };
        let uniform_sub_buffer = self.compute_uniforms.from_data(uniform_data).unwrap();
        let get_descriptors = |ub: Arc<CpuBufferPoolSubbuffer<cs::ty::Data>>| {
            [
                WriteDescriptorSet::buffer(0, transform.clone()),
                WriteDescriptorSet::buffer(1, pb.particles.clone()),
                WriteDescriptorSet::buffer(2, pb.particle_positions_lifes.clone()),
                WriteDescriptorSet::buffer(3, pb.particle_next.clone()),
                WriteDescriptorSet::buffer(4, pb.avail.clone()),
                WriteDescriptorSet::buffer(5, pb.emitters.lock().clone()),
                WriteDescriptorSet::buffer(6, pb.avail_count.clone()),
                WriteDescriptorSet::buffer(7, pb.particle_template.lock().clone()),
                WriteDescriptorSet::buffer(8, ub),
                WriteDescriptorSet::buffer(9, pb.particle_template_ids.clone()),
                WriteDescriptorSet::buffer(10, emitter_inits.clone()),
                WriteDescriptorSet::buffer(11, pb.alive.clone()),
                WriteDescriptorSet::buffer(12, pb.alive_count.clone()),
                WriteDescriptorSet::buffer(13, pb.indirect.clone()),
                WriteDescriptorSet::buffer(14, pb.alive_b.clone()),
            ]
        };
        // count alive particles
        let descriptor_set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.compute_pipeline
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            get_descriptors(uniform_sub_buffer),
        )
        .unwrap();

        builder
            .copy_buffer(CopyBufferInfo::buffers(
                pb.buffer_0.clone(),
                pb.alive_count.clone(),
            ))
            .unwrap()
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set,
            )
            .dispatch([MAX_PARTICLES as u32 / 1024 + 1, 1, 1])
            .unwrap();
        // set indirect
        uniform_data.num_jobs = 1;
        uniform_data.stage = 3;
        let uniform_sub_buffer = self.compute_uniforms.from_data(uniform_data).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.compute_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap()
                .clone(),
            get_descriptors(uniform_sub_buffer),
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0,
                descriptor_set,
            )
            .dispatch([1, 1, 1])
            .unwrap();
        // dispatch indirect particle update
        uniform_data.num_jobs = -1;
        uniform_data.stage = 4;
        let uniform_sub_buffer = self.compute_uniforms.from_data(uniform_data).unwrap();
        let descriptor_set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.compute_pipeline
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            get_descriptors(uniform_sub_buffer),
        )
        .unwrap();
        builder
            .bind_pipeline_compute(self.compute_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.compute_pipeline.layout().clone(),
                0, // Bind this descriptor set to index 0.
                descriptor_set,
            )
            .dispatch_indirect(pb.indirect.clone())
            .unwrap();
    }
    pub fn render_particles(
        &self,
        particle_render_pipeline: &ParticleRenderPipeline,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        view: glm::Mat4,
        proj: glm::Mat4,
        cam_rot: [f32; 4],
        cam_pos: [f32; 3],
        transform: Arc<DeviceLocalBuffer<[transform]>>,
        _mem: Arc<StandardMemoryAllocator>,
        _command_allocator: &StandardCommandBufferAllocator,
        desc_allocator: Arc<StandardDescriptorSetAllocator>,
    ) {
        let pb = &self.particle_buffers;

        let layout = particle_render_pipeline
            .arc
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();

        let mut descriptors = Vec::new();
        let uniform_sub_buffer = {
            let uniform_data = gs::ty::Data {
                view: view.into(),
                proj: proj.into(),
                cam_rot,
                cam_pos,
                _dummy0: Default::default(),
            };
            self.render_uniforms.from_data(uniform_data).unwrap()
        };

        descriptors.push(WriteDescriptorSet::buffer(
            0,
            pb.particle_positions_lifes.clone(),
        ));
        // descriptors.push(WriteDescriptorSet::buffer(2, self.particles.clone()));
        descriptors.push(WriteDescriptorSet::buffer(
            3,
            pb.particle_template.lock().clone(),
        ));
        descriptors.push(WriteDescriptorSet::buffer(4, uniform_sub_buffer));
        descriptors.push(WriteDescriptorSet::buffer(5, self.sort.a2.clone()));
        descriptors.push(WriteDescriptorSet::buffer(
            6,
            pb.particle_template_ids.clone(),
        ));
        descriptors.push(WriteDescriptorSet::buffer(7, pb.particle_next.clone()));
        descriptors.push(WriteDescriptorSet::buffer(8, transform));

        // if let Some(texture) = mesh.texture.as_ref() {
        //     descriptors.push(WriteDescriptorSet::image_view_sampler(
        //         1,
        //         texture.image.clone(),
        //         texture.sampler.clone(),
        //     ));
        // } else {
        //     descriptors.push(WriteDescriptorSet::image_view_sampler(
        //         1,
        //         self.def_texture.clone(),
        //         self.def_sampler.clone(),
        //     ));
        // }
        // descriptors.push(WriteDescriptorSet::buffer(2, instance_buffer.clone()));
        let set =
            PersistentDescriptorSet::new(&desc_allocator, layout.clone(), descriptors).unwrap();

        builder
            .bind_pipeline_graphics(particle_render_pipeline.arc.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                particle_render_pipeline.arc.layout().clone(),
                0,
                set,
            )
            .draw_indirect(self.sort.draw.clone())
            .unwrap();
        // .draw(MAX_PARTICLES as u32, 1, 0, 0)
        // .unwrap();

        // let temp = self.sort.a1.clone();
        // self.sort.a1 = self.sort.a2.clone();
        // self.sort.a2 = temp.clone();
    }
}
