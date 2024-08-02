#version 450
#include "util.glsl"

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
// layout(location = 3) in uint bone_weight_offset;
// layout(location = 4) in uint bone_weight_count;
// layout(location = 3) in int id;

// The per-instance data.
// layout(location = 3) in vec3 pos;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 coords;
layout(location = 2) out vec3 v_pos;
layout(location = 3) out vec3 _v;

layout(set = 0, binding = 0) buffer tr { MVP mvp[]; };
layout(set = 0, binding = 2) buffer id { ivec2 ids[]; };
layout(set = 0, binding = 10) buffer b { vec4 bones[]; };
struct bone_weight {
    int bone_id;
    float weight;
};
layout(set = 0, binding = 11) buffer bvw { bone_weight bone_vertex_weights[]; };

layout(set = 0, binding = 12) uniform UniformBufferObject { int has_skeleton; int num_bones; };
layout(set = 0, binding = 13) buffer bwo { uvec2 bone_weight_offsets_counts[]; };
layout(set = 0, binding = 14) buffer bwc { uint bone_weight_counts[]; };
// layout(location = 4) in uint bone_weight_count;
// mat4 world;
void main() {
    mat4 vertex_offset = identity();
    int id = gl_InstanceIndex;
    id = ids[id].x;

    if (has_skeleton == 1) {
        mat4 z = {
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 0},
            {0, 0, 0, 1}
        };
        vertex_offset = z;
        for (uint i = bone_weight_offsets_counts[gl_VertexIndex].x; i < bone_weight_offsets_counts[gl_VertexIndex].x + bone_weight_offsets_counts[gl_VertexIndex].y; ++i) {
            int raw_offset = ids[gl_InstanceIndex].y * num_bones + bone_vertex_weights[i].bone_id;
            int b = raw_offset * 3; // 3 vec4 per matrix
            vertex_offset[0] += bones[b] * bone_vertex_weights[i].weight;
            vertex_offset[1] += bones[b + 1] * bone_vertex_weights[i].weight;
            vertex_offset[2] += bones[b + 2] * bone_vertex_weights[i].weight;
        }
        vertex_offset = transpose(vertex_offset);
    }
    // mat4 worldview = uniforms.view;
    // v_normal = transpose(inverse(mat3(worldview))) * normal;
    coords = uv;
    v_normal = mat3(mvp[id].n) * normal;
    v_pos = (mvp[id].m * vertex_offset * vec4(position, 1.0)).xyz;
    vec4 v = (mvp[id].mvp * vertex_offset * vec4(position, 1.0));
    // _v = get_cluster_idx(v);
    _v = v.xyz;
    // mat4 mvp = mvp[id];
    gl_Position = mvp[id].mvp * vertex_offset * vec4(position, 1.0);

    // gl_Position.z *= -1;
    // gl_Position.y *= -1;
}
