use std::{ops::Mul, sync::Arc};

use glm::vec3;
use nalgebra_glm as glm;
use nalgebra_glm::{Mat4, Vec3};
use parking_lot::Mutex;
use rapier3d::na::ComplexField;
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::pipeline::graphics::vertex_input::VertexBufferDescription;
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::{
    buffer::{allocator::SubbufferAllocator, Subbuffer},
    command_buffer::{CopyBufferInfo, DispatchIndirectCommand, DrawIndirectCommand},
    descriptor_set::{self, DescriptorSet, WriteDescriptorSet},
    memory::allocator::MemoryTypeFilter,
    padded::Padded,
    pipeline::{
        graphics::{
            color_blend::ColorBlendState,
            depth_stencil::{CompareOp, DepthState, DepthStencilState},
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            rasterization::{CullMode, RasterizationState},
            vertex_input::BuffersDefinition,
            viewport::ViewportState,
        },
        ComputePipeline, GraphicsPipeline, Pipeline,
    },
    render_pass::{RenderPass, Subpass},
};

use crate::engine::rendering::component::ur::r;
use crate::engine::rendering::component::Indirect;
use crate::engine::utils::radix_sort;
use crate::engine::{
    prelude::{
        utils::{self, PrimaryCommandBuffer},
        VulkanManager,
    },
    rendering::{camera::CameraViewData, pipeline::fs},
    transform_compute::cs::transform,
};
pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/lighting.comp",
    }
}
pub mod lt {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/shaders/light_tile.comp",
    }
}

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/light_debug.vert",
    }
}

pub mod gs {
    vulkano_shaders::shader! {
        ty: "geometry",
        path: "src/shaders/light_debug.geom",
    }
}

pub mod lfs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/light_debug.frag",
    }
}

pub struct LightingCompute {
    pipeline: Arc<ComputePipeline>,
    pipeline2: Arc<ComputePipeline>,
    // pub(crate) debug: Arc<GraphicsPipeline>,
    // uniforms: Mutex<SubbufferAllocator>,
    vk: Arc<VulkanManager>,
    dummy_buffer: Subbuffer<[u8]>,
    pub(crate) tiles: Mutex<Subbuffer<[lt::tile]>>,
    light_list: Mutex<Subbuffer<[u32]>>,
    light_tile_ids: Mutex<Subbuffer<[u32]>>,
    pub(crate) light_list2: Mutex<Subbuffer<[u32]>>,
    light_tile_ids2: Mutex<Subbuffer<[u32]>>,
    light_offsets: Subbuffer<[u32]>,
    light_counter: Subbuffer<radix_sort::cs1::PC>,
    pub(crate) visible_lights: Mutex<Subbuffer<[u32]>>,
    pub(crate) visible_lights_c: Subbuffer<u32>,
    pub(crate) radix_sort: Arc<Mutex<crate::engine::utils::radix_sort::RadixSort>>,
}
pub const NUM_TILES: u64 = 64 * 64 + 32 * 32 + 16 * 16 + 8 * 8 + 4 * 4 + 2 * 2 + 1;
impl LightingCompute {
    pub fn new_pipeline(
        vk: Arc<VulkanManager>,
        render_pass: Arc<RenderPass>,
    ) -> Arc<GraphicsPipeline> {
        let vs = vs::load(vk.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let fs = lfs::load(vk.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();
        let gs = gs::load(vk.device.clone())
            .unwrap()
            .entry_point("main")
            .unwrap();

        // let subpass = Subpass::from(render_pass, 0).unwrap();
        // let blend_state = ColorBlendState::new(subpass.num_color_attachments()).blend_alpha();
        // let mut depth_stencil_state = DepthStencilState::simple_depth_test();
        // depth_stencil_state.depth = Some(DepthState {
        //     write_enable: StateMode::Fixed(false),
        //     compare_op: StateMode::Fixed(CompareOp::Less),
        // });

        // let render_pipeline = GraphicsPipeline::start()
        //     .vertex_input_state(BuffersDefinition::new())
        //     .vertex_shader(vs.entry_point("main").unwrap(), ())
        //     // .input_assembly_state(InputAssemblyState::new())
        //     .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::PointList))
        //     .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        //     .geometry_shader(gs.entry_point("main").unwrap(), ())
        //     // .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::LineStrip))
        //     .fragment_shader(fs.entry_point("main").unwrap(), ())
        //     .rasterization_state(RasterizationState::new().cull_mode(CullMode::None))
        //     .depth_stencil_state(depth_stencil_state)
        //     .color_blend_state(blend_state)
        //     .render_pass(subpass)
        //     .build(vk.device.clone())
        //     .unwrap();
        let render_pipeline = utils::pipeline::graphics_pipeline(
            vk.clone(),
            &[vs, gs, fs],
            &[],
            |info| {
                info.input_assembly_state = Some(InputAssemblyState {
                    topology: PrimitiveTopology::PointList,
                    primitive_restart_enable: false,
                    ..Default::default()
                });
                info.rasterization_state = Some(RasterizationState {
                    cull_mode: CullMode::None,
                    ..Default::default()
                });
            },
            render_pass,
        );
        render_pipeline
    }
    pub fn new(vk: Arc<VulkanManager>, render_pass: Arc<RenderPass>) -> LightingCompute {
        Self {
            // pipeline: vulkano::pipeline::ComputePipeline::new(
            //     vk.device.clone(),
            //     cs::load(vk.device.clone())
            //         .unwrap()
            //         .entry_point("main")
            //         .unwrap(),
            //     &(),
            //     None,
            //     |_| {},
            // )
            // .expect("Failed to create compute shader"),
            pipeline: utils::pipeline::compute_pipeline(
                vk.clone(),
                cs::load(vk.device.clone()).unwrap(),
            ),
            // pipeline2: vulkano::pipeline::ComputePipeline::new(
            //     vk.device.clone(),
            //     lt::load(vk.device.clone())
            //         .unwrap()
            //         .entry_point("main")
            //         .unwrap(),
            //     &(),
            //     None,
            //     |_| {},
            // )
            // .expect("Failed to create compute shader"),
            pipeline2: utils::pipeline::compute_pipeline(
                vk.clone(),
                lt::load(vk.device.clone()).unwrap(),
            ),
            // uniforms: Mutex::new(vk.sub_buffer_allocator()),
            dummy_buffer: vk.buffer_array(1, MemoryTypeFilter::PREFER_DEVICE),
            light_list: Mutex::new(vk.buffer_array(4, MemoryTypeFilter::PREFER_DEVICE)),
            light_tile_ids: Mutex::new(vk.buffer_array(4, MemoryTypeFilter::PREFER_DEVICE)),
            light_list2: Mutex::new(vk.buffer_array(4, MemoryTypeFilter::PREFER_DEVICE)),
            light_tile_ids2: Mutex::new(vk.buffer_array(4, MemoryTypeFilter::PREFER_DEVICE)),
            light_offsets: vk.buffer_array(NUM_TILES, MemoryTypeFilter::PREFER_DEVICE),
            light_counter: vk.buffer(MemoryTypeFilter::PREFER_DEVICE),
            visible_lights: Mutex::new(vk.buffer_array(1, MemoryTypeFilter::PREFER_DEVICE)),
            visible_lights_c: vk.buffer(MemoryTypeFilter::PREFER_DEVICE),
            tiles: Mutex::new(vk.buffer_array(NUM_TILES, MemoryTypeFilter::PREFER_DEVICE)),
            radix_sort: Arc::new(Mutex::new(
                crate::engine::utils::radix_sort::RadixSort::new(vk.clone()),
            )),
            vk: vk,
        }
    }
    fn get_descriptors<W, V, U, T>(
        &self,
        num_jobs: i32,
        stage: i32,
        lights: Subbuffer<[lt::light]>,
        deinits: Subbuffer<[U]>,
        inits: Subbuffer<[T]>,
        transforms: Subbuffer<[transform]>,
        light_templates: Subbuffer<[fs::lightTemplate]>,
        tiles: Subbuffer<[V]>,
        indirect: Subbuffer<[W]>,
    ) -> Arc<PersistentDescriptorSet> {
        // let visble_lights = self.visible_lights.lock();
        let uniforms = self.vk.allocate(cs::Data {
            num_jobs: num_jobs as i32,
            stage: stage.into(),
        });
        // {
        //     let uniform_data = cs::Data {
        //         num_jobs: num_jobs as i32,
        //         stage: stage.into(),
        //     };
        //     let ub = self.uniforms.lock().allocate_sized().unwrap();
        //     *ub.write().unwrap() = uniform_data;
        //     ub
        //     // self.uniforms.from_data(uniform_data).unwrap()
        // };
        PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.pipeline
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, uniforms),
                WriteDescriptorSet::buffer(1, lights),
                WriteDescriptorSet::buffer(2, deinits),
                WriteDescriptorSet::buffer(3, inits),
                WriteDescriptorSet::buffer(4, transforms),
                WriteDescriptorSet::buffer(6, light_templates.clone()),
                WriteDescriptorSet::buffer(7, tiles.clone()),
                WriteDescriptorSet::buffer(8, self.light_list.lock().clone()),
                WriteDescriptorSet::buffer(15, self.light_tile_ids.lock().clone()),
                // WriteDescriptorSet::buffer(9, self.light_list2.lock().clone()),
                // WriteDescriptorSet::buffer(10, self.light_offsets.clone()),
                WriteDescriptorSet::buffer(11, self.light_counter.clone()),
                WriteDescriptorSet::buffer(12, self.visible_lights.lock().clone()),
                WriteDescriptorSet::buffer(13, self.visible_lights_c.clone()),
                WriteDescriptorSet::buffer(14, indirect.clone()),
            ],
            [],
        )
        .unwrap()
    }

    pub fn update_lights_1(
        &self,
        builder: &mut PrimaryCommandBuffer,
        deinits: Option<Subbuffer<[cs::light_deinit]>>,
        inits: Option<Subbuffer<[cs::light_init]>>,
        lights: Subbuffer<[lt::light]>,
        transforms: Subbuffer<[transform]>,
        light_templates: Subbuffer<[fs::lightTemplate]>,
        // screen_dims: [u32; 2],
    ) {
        builder.bind_pipeline_compute(self.pipeline.clone());
        // builder.fill_buffer(self.buckets_count.clone(), 0).unwrap();
        // builder.fill_buffer(visible_lights.clone(), 0).unwrap();
        // builder.update_buffer(self.visible_lights_index.clone(), &0);
        // let _builder = builder;
        let mut build_stage =
            |builder: &mut utils::PrimaryCommandBuffer,
             num_jobs: i32,
             stage: i32,
             inits: Option<Subbuffer<[cs::light_init]>>,
             deinits: Option<Subbuffer<[cs::light_deinit]>>| {
                let descriptor_set = if let Some(deinit) = deinits {
                    self.get_descriptors(
                        num_jobs,
                        stage,
                        lights.clone(),
                        deinit,
                        self.dummy_buffer.clone(),
                        transforms.clone(),
                        light_templates.clone(),
                        self.dummy_buffer.clone(),
                        self.dummy_buffer.clone(),
                    )
                } else if let Some(init) = inits {
                    self.get_descriptors(
                        num_jobs,
                        stage,
                        lights.clone(),
                        self.dummy_buffer.clone(),
                        init,
                        transforms.clone(),
                        light_templates.clone(),
                        self.dummy_buffer.clone(),
                        self.dummy_buffer.clone(),
                    )
                } else {
                    self.get_descriptors(
                        num_jobs,
                        stage,
                        lights.clone(),
                        self.dummy_buffer.clone(),
                        self.dummy_buffer.clone(),
                        transforms.clone(),
                        light_templates.clone(),
                        self.dummy_buffer.clone(),
                        self.dummy_buffer.clone(),
                    )
                };
                builder
                    .bind_descriptor_sets(
                        self.pipeline.bind_point(),
                        self.pipeline.layout().clone(),
                        0,
                        descriptor_set,
                    )
                    .unwrap()
                    .dispatch([(num_jobs as u32).div_ceil(128), 1, 1])
                    .unwrap();
            };

        if let Some(deinits) = deinits {
            build_stage(builder, deinits.len() as i32, 0, None, Some(deinits));
        }
        if let Some(inits) = inits {
            build_stage(builder, inits.len() as i32, 1, Some(inits), None);
        }
        build_stage(builder, lights.len() as i32, 2, None, None);
    }

    pub fn update_lights_2(
        &self,
        builder: &mut PrimaryCommandBuffer,
        lights: Subbuffer<[lt::light]>,
        cvd: &CameraViewData,
        transforms: Subbuffer<[transform]>,
        light_templates: Subbuffer<[fs::lightTemplate]>,
        num_lights: i32,
    ) {
        {
            let mut light_list = self.light_list.lock();
            let mut light_tile_ids = self.light_tile_ids.lock();
            let mut light_list2 = self.light_list2.lock();
            let mut light_tile_ids2 = self.light_tile_ids2.lock();
            let mut visible_lights = self.visible_lights.lock();
            if (num_lights > visible_lights.len() as i32) {
                let buf = self.vk.buffer_array(
                    (num_lights as u64).next_power_of_two() * 4,
                    MemoryTypeFilter::PREFER_DEVICE,
                );
                *light_list = buf;

                let buf = self.vk.buffer_array(
                    (num_lights as u64).next_power_of_two() * 4,
                    MemoryTypeFilter::PREFER_DEVICE,
                );
                *light_tile_ids = buf;

                let buf = self.vk.buffer_array(
                    (num_lights as u64).next_power_of_two() * 4,
                    MemoryTypeFilter::PREFER_DEVICE,
                );
                *light_list2 = buf;

                let buf = self.vk.buffer_array(
                    (num_lights as u64).next_power_of_two() * 4,
                    MemoryTypeFilter::PREFER_DEVICE,
                );
                *light_tile_ids2 = buf;

                let buf = self.vk.buffer_array(
                    (num_lights as u64).next_power_of_two(),
                    MemoryTypeFilter::PREFER_DEVICE,
                );
                *visible_lights = buf;
            }
        }
        let mut uni = lt::Data {
            // num_jobs: 0,
            vp: { cvd.proj * cvd.view }.into(),
            cam_pos: cvd.cam_pos.into(),
            num_lights: num_lights as i32,
        };
        builder.bind_pipeline_compute(self.pipeline2.clone());
        let uniforms = self.vk.allocate(uni);
        //  {
        //     let ub = self.uniforms.lock().allocate_sized().unwrap();
        //     *ub.write().unwrap() = uni;
        //     ub
        //     // self.uniforms.from_data(uniform_data).unwrap()
        // };
        let set = PersistentDescriptorSet::new(
            &self.vk.desc_alloc,
            self.pipeline2
                .layout()
                .set_layouts()
                .get(0) // 0 is the index of the descriptor set.
                .unwrap()
                .clone(),
            [
                WriteDescriptorSet::buffer(0, uniforms),
                WriteDescriptorSet::buffer(2, self.tiles.lock().clone()),
            ],
            [],
        )
        .unwrap();
        builder
            .bind_descriptor_sets(
                self.pipeline2.bind_point(),
                self.pipeline2.layout().clone(),
                0,
                set,
            )
            .unwrap()
            .dispatch([NUM_TILES.div_ceil(64) as u32, 1, 1])
            .unwrap();

        builder.bind_pipeline_compute(self.pipeline.clone());

        let tiles = self.tiles.lock();
        let mut build_stage = |builder: &mut utils::PrimaryCommandBuffer,
                               num_jobs: i32,
                               indirect: Option<Subbuffer<[DispatchIndirectCommand]>>,
                               indirect_write: Option<Subbuffer<[DispatchIndirectCommand]>>,
                               stage: i32| {
            let descriptor_set = if let Some(indirect_write) = indirect_write {
                self.get_descriptors(
                    num_jobs,
                    stage,
                    lights.clone(),
                    self.dummy_buffer.clone(),
                    self.dummy_buffer.clone(),
                    transforms.clone(),
                    light_templates.clone(),
                    tiles.clone(),
                    indirect_write.clone(),
                )
            } else {
                self.get_descriptors(
                    num_jobs,
                    stage,
                    lights.clone(),
                    self.dummy_buffer.clone(),
                    self.dummy_buffer.clone(),
                    transforms.clone(),
                    light_templates.clone(),
                    tiles.clone(),
                    self.dummy_buffer.clone(),
                )
            };
            builder.bind_descriptor_sets(
                self.pipeline.bind_point(),
                self.pipeline.layout().clone(),
                0,
                descriptor_set,
            );
            if (num_jobs >= 0) {
                builder
                    .dispatch([(num_jobs as u32).div_ceil(128), 1, 1])
                    .unwrap();
            } else {
                if let Some(indirect_buffer) = indirect {
                    builder.dispatch_indirect(indirect_buffer).unwrap();
                }
            }
        };
        let indirect = self.vk.buffer_from_iter([
            DispatchIndirectCommand { x: 1, y: 1, z: 1 },
            DispatchIndirectCommand { x: 1, y: 1, z: 1 },
        ]);
        // build_stage(builder, 32 * 32, 3);
        let light_jobs = (num_lights as u32).div_ceil(128).mul(128) as i32;
        builder.update_buffer(self.visible_lights_c.clone(), &0);
        builder.update_buffer(
            self.light_counter.clone(),
            &radix_sort::cs1::PC {
                g_num_elements: 0,
                g_num_workgroups: 0,
            },
        );

        build_stage(
            builder,
            num_lights,
            None,
            Some(indirect.clone().slice(0..1)),
            3,
        );
        build_stage(
            builder,
            -1,
            Some(indirect.clone().slice(0..1)),
            Some(indirect.clone().slice(1..2)),
            4,
        );
        // radix sort
        {
            let mut light_list = self.light_list.lock();
            let mut light_tile_ids = self.light_tile_ids.lock();
            let mut light_list2 = self.light_list2.lock();
            let mut light_tile_ids2 = self.light_tile_ids2.lock();
            // let mut visible_lights = self.visible_lights.lock();
            self.radix_sort.lock().sort(
                self.vk.clone(),
                num_lights as u32,
                indirect.clone().slice(1..2),
                self.light_counter.clone(),
                &mut *light_tile_ids,
                &mut *light_list,
                &mut *light_tile_ids2,
                &mut *light_list2,
                builder,
            );
        }
        // build_stage(builder, 128, None, None, 4);
        // build_stage(builder, -1, Some(indirect.clone().slice(1..2)), None, 5);
        // build_stage(builder, 32 * 32, 5);

        // let tiles_curr_len = { tiles.lock().len() };
        // let tiles_should_be_len = (((screen_dims[0] / 16.).ceil() + 1.) * ((screen_dims[1].abs() / 16.).ceil() + 1.)).max(1.) as u64;
        // let tiles_should_be_len = tiles_should_be_len.min(14_651);
        // if tiles_curr_len != tiles_should_be_len {
        //     let buf = self.vk.buffer_array(
        //         tiles_should_be_len,
        //         MemoryTypeFilter::PREFER_DEVICE,
        //     );
        //     *tiles.lock() = buf;
        // }
        // let num_threads = [(screen_dims[0] / 16.).ceil().min(160.0) as u32, (screen_dims[1].abs() / 16.).ceil().min(90.0) as u32];

        // // prefix sum
        // builder
        //     .copy_buffer(CopyBufferInfo::buffers(
        //         self.buckets_count.clone(),
        //         self.buckets.clone().slice(0..NUM_BUCKETS),
        //     ))
        //     .unwrap();
        // build_stage(builder, 256, 3, None, None);
        // build_stage(builder, 1, 5, None, None);
        // build_stage(builder, 256, 6, None, None);
        // builder.copy_buffer(CopyBufferInfo::buffers(
        //     self.buckets.clone().slice(0..NUM_BUCKETS),
        //     self.buckets_2.clone(),
        // ));
        // //////
        // build_stage(builder, lights.len() as i32, 4, None, None);
    }
}
