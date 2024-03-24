use std::{default, sync::Arc};

use parking_lot::Mutex;
use vulkano::{
    buffer::{allocator::SubbufferAllocator, Buffer, Subbuffer},
    command_buffer::{
        allocator::StandardCommandBufferAllocator, AutoCommandBufferBuilder, CommandBufferUsage,
        CopyBufferInfo, DispatchIndirectCommand, DrawIndirectCommand, PrimaryAutoCommandBuffer,
    },
    descriptor_set::{
        allocator::StandardDescriptorSetAllocator, PersistentDescriptorSet, WriteDescriptorSet,
    },
    device::{Device, Queue},
    memory::allocator::{MemoryUsage, StandardMemoryAllocator},
    padded::Padded,
    pipeline::{ComputePipeline, Pipeline, PipelineBindPoint},
    sync::{self, FlushError, GpuFuture},
    DeviceSize,
};
use winit::event::VirtualKeyCode;

use crate::engine::{
    input::Input,
    perf::{self, Perf},
    rendering::{camera::CameraViewData, component::buffer_usage_all},
    transform_compute::cs::transform,
    VulkanManager,
};

use super::{
    particles::{ParticleBuffers, _MAX_PARTICLES},
    shaders::scs,
};

pub struct ParticleSort {
    vk: Arc<VulkanManager>,
    pub a1: Subbuffer<[scs::_a]>,
    pub a2: Subbuffer<[u32]>,
    pub buckets: Subbuffer<[u32]>,
    pub avail_count: Subbuffer<u32>,
    pub indirect: Vec<Subbuffer<[DispatchIndirectCommand]>>,
    pub draw: Subbuffer<[DrawIndirectCommand]>,
    // pub uniforms: Mutex<SubbufferAllocator>,
    pub compute_pipeline: Arc<ComputePipeline>,
}
impl ParticleSort {
    pub fn new(vk: Arc<VulkanManager>) -> ParticleSort {
        let mut builder = AutoCommandBufferBuilder::primary(
            &vk.comm_alloc,
            vk.queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        let max_particles: i32 = *_MAX_PARTICLES;

        let a1 = vk.buffer_array(
            max_particles as vulkano::DeviceSize,
            MemoryUsage::DeviceOnly,
        );
        let a2 = vk.buffer_array(
            max_particles as vulkano::DeviceSize,
            MemoryUsage::DeviceOnly,
        );
        let buckets = vk.buffer_array(65536 as vulkano::DeviceSize, MemoryUsage::DeviceOnly);
        let avail_count = vk.buffer_from_data(0u32);
        // indirect
        let indirect = (0..2)
            .map(|_| {
                let copy_buffer =
                    vk.buffer_from_iter([DispatchIndirectCommand { x: 0, y: 1, z: 1 }]);
                let indirect = vk.buffer_array(1, MemoryUsage::DeviceOnly);
                builder
                    .copy_buffer(CopyBufferInfo::buffers(copy_buffer, indirect.clone()))
                    .unwrap();
                indirect
            })
            .collect();

        // draw
        let copy_buffer = vk.buffer_from_iter([DrawIndirectCommand {
            vertex_count: 0,
            instance_count: 1,
            first_vertex: 0,
            first_instance: 0,
        }]);
        let draw = vk.buffer_array(1 as DeviceSize, MemoryUsage::DeviceOnly);
        builder
            .copy_buffer(CopyBufferInfo::buffers(copy_buffer, draw.clone()))
            .unwrap();

        // let uniforms = Mutex::new(vk.sub_buffer_allocator());

        // build buffer
        let command_buffer = builder.build().unwrap();

        let execute = Some(sync::now(vk.device.clone()).boxed())
            .take()
            .unwrap()
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

        let scs = scs::load(vk.device.clone()).unwrap();
        let compute_pipeline = vulkano::pipeline::ComputePipeline::new(
            vk.device.clone(),
            scs.entry_point("main").unwrap(),
            &(),
            None,
            |_| {},
        )
        .expect("Failed to create compute shader");

        ParticleSort {
            a1,
            a2,
            buckets,
            avail_count,
            indirect,
            draw,
            // uniforms,
            compute_pipeline,
            vk,
        }
    }
    pub fn sort(
        &self,
        cvd: &CameraViewData,
        transform: Subbuffer<[transform]>,
        pb: &ParticleBuffers,
        _device: Arc<Device>,
        _queue: Arc<Queue>,
        builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
        desc_allocator: &StandardDescriptorSetAllocator,
        perf: &Perf,
        input: &Input,
    ) {
        let max_particles: i32 = *_MAX_PARTICLES;
        static mut FRUSTUM: scs::Frustum = scs::Frustum {
            planes: [[0., 0., 0., 0.]; 6],
            points: [Padded([0., 0., 0.]); 8],
        };
        static mut LOCK_FRUSTUM: bool = false;

        if input.get_key_up(&VirtualKeyCode::C) {
            unsafe {
                LOCK_FRUSTUM = !LOCK_FRUSTUM;
            }
            println!("lock frustum: {}", unsafe { LOCK_FRUSTUM });
        }
        if unsafe { !LOCK_FRUSTUM } {
            unsafe {
                FRUSTUM = cvd.frustum.clone().into();
            }
        }
        let mut uniform_data = scs::Data {
            num_jobs: max_particles,
            stage: 0.into(),
            view: cvd.view.into(),
            proj: cvd.proj.into(),
            cam_pos: Padded(cvd.cam_pos.into()),
            frustum: unsafe { FRUSTUM },
        };

        let layout = self
            .compute_pipeline
            .layout()
            .set_layouts()
            .get(0) // 0 is the index of the descriptor set.
            .unwrap()
            .clone();

        let mut build_stage = |builder: &mut AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer,
            Arc<StandardCommandBufferAllocator>,
        >,
                               stage: i32,
                               num_jobs: i32| {
            let indirect = match stage {
                // dispatch
                1 => pb.indirect.clone(),
                3 => self.indirect[0].clone(),
                5 => self.indirect[0].clone(),
                _ => self.indirect[1].clone(),
            };
            let bound_indirect = match stage {
                // update
                2 => self.indirect[0].clone(),
                _ => self.indirect[1].clone(),
            };

            uniform_data.num_jobs = num_jobs;
            uniform_data.stage = stage.into();
            let uniform_sub_buffer = self.vk.allocate(uniform_data);
            let descriptor_set = PersistentDescriptorSet::new(
                desc_allocator,
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, self.a1.clone()),
                    WriteDescriptorSet::buffer(1, self.a2.clone()),
                    WriteDescriptorSet::buffer(2, pb.particles.clone()),
                    WriteDescriptorSet::buffer(3, pb.particle_positions_lifes.clone()),
                    WriteDescriptorSet::buffer(4, bound_indirect),
                    WriteDescriptorSet::buffer(5, self.avail_count.clone()),
                    WriteDescriptorSet::buffer(6, uniform_sub_buffer),
                    WriteDescriptorSet::buffer(7, self.buckets.clone()),
                    WriteDescriptorSet::buffer(8, self.draw.clone()),
                    WriteDescriptorSet::buffer(9, pb.particle_next.clone()),
                    WriteDescriptorSet::buffer(10, pb.particle_templates.lock().clone()),
                    WriteDescriptorSet::buffer(11, pb.emitters.lock().clone()),
                    WriteDescriptorSet::buffer(12, transform.clone()),
                    WriteDescriptorSet::buffer(13, pb.alive.clone()),
                    WriteDescriptorSet::buffer(14, pb.alive_count.clone()),
                    // WriteDescriptorSet::buffer(15, pb.pos_life_compressed.clone()),
                ],
            )
            .unwrap();

            if num_jobs < 0 {
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        self.compute_pipeline.layout().clone(),
                        0, // Bind this descriptor set to index 0.
                        descriptor_set,
                    )
                    .dispatch_indirect(indirect)
                    .unwrap();
            } else {
                builder
                    .bind_descriptor_sets(
                        PipelineBindPoint::Compute,
                        self.compute_pipeline.layout().clone(),
                        0, // Bind this descriptor set to index 0.
                        descriptor_set,
                    )
                    .dispatch([num_jobs as u32 / 1024 + 1, 1, 1])
                    .unwrap();
            }
        };
        builder.bind_pipeline_compute(self.compute_pipeline.clone());
        builder.update_buffer(self.avail_count.clone(), &0).unwrap();
        // .copy_buffer(CopyBufferInfo::buffers(
        //     pb.buffer_0.clone(),
        //     self.avail_count.clone(),
        // ))
        // .unwrap();

        // stage 0
        // build_stage(builder, 0, 1);

        // stage 1
        build_stage(builder, 1, -1); // number of active particles

        // stage 2
        builder.fill_buffer(self.buckets.clone(), 0);
        build_stage(builder, 2, 1);

        // stage 3
        build_stage(builder, 3, -1); // number of visible particles

        // stage 4
        build_stage(builder, 4, 256); // prefix sum

        // stage 5
        build_stage(builder, 5, -1); // number of visible particles

        // stage 6
        build_stage(builder, 6, 1); // set draw data

        // let temp = self.a1.clone();
        // self.a1 = self.a2.clone();
        // self.a2 = temp.clone();
        // swap(&mut self.a1, &mut self.a2);
    }
}
