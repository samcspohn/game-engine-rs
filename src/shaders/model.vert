#version 450
#include "util.glsl"

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
// layout(location = 3) in int id;

// The per-instance data.
// layout(location = 3) in vec3 pos;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec2 coords;
layout(location = 2) out vec3 v_pos;
layout(location = 3) out vec3 _v;

struct MVP {
    mat4 mvp;
    mat4 mv;
    mat4 m;
};

layout(set = 0, binding = 0) buffer tr {
    MVP mvp[];
};
layout(set = 0, binding = 2) buffer id {
    int ids[];
};

// layout(set = 0, binding = 3) uniform UniformBufferObject {
//     int offset;
// }ubo;

    // mat4 world;
void main() {
    int id = gl_InstanceIndex;
    // mat4 worldview = uniforms.view;
        // v_normal = transpose(inverse(mat3(worldview))) * normal;
    coords = uv;
    v_normal = mat3(mvp[ids[id]].m) * normal;
    v_pos = (mvp[ids[id]].m * vec4(position, 1.0)).xyz;
    vec4 v = (mvp[ids[id]].mvp * vec4(position, 1.0));
    // _v = get_cluster_idx(v);
    _v = v.xyz;
    // mat4 mvp = mvp[id];
    gl_Position = mvp[ids[id]].mvp * vec4(position, 1.0);

    // gl_Position.z *= -1;
    // gl_Position.y *= -1;
}
