
pub mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/engine/particles/particle_shaders/particles.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/engine/particles/particle_shaders/particles.vert",
        types_meta: {


            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
pub mod gs {
    vulkano_shaders::shader! {
        ty: "geometry",
        path: "src/engine/particles/particle_shaders/particles.geom",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}

pub mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/engine/particles/particle_shaders/particles.frag"
    }
}


pub mod scs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/engine/particles/particle_shaders/particle_sort.comp",
        types_meta: {
            use bytemuck::{Pod, Zeroable};

            #[derive(Clone, Copy, Zeroable, Pod)]
        },
    }
}
