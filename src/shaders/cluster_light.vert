#version 450
#include "util.glsl"

layout(location = 0) in vec3 position;


layout(set = 0, binding = 0) uniform UniformBufferObject { mat4 vp; };
layout(set = 0, binding = 1) buffer l { light lights[]; };
layout(set = 0, binding = 2) buffer vl { uint visible_lights[]; };
// layout(set = 0, binding = 2) buffer id {
//     int ids[];
// };

const vec3 verts[4] = {vec3(-1, -1, 0), vec3(1, -1, 0), vec3(-1, 1, 0),
                       vec3(1, 1, 0)};
// mat4 world;
void main() {
    int id = gl_InstanceIndex;
    int v_id = gl_VertexIndex;
    // mat4 worldview = uniforms.view;
    // v_normal = transpose(inverse(mat3(worldview))) * normal;
    // coords = uv;
    // v_normal = mat3(mvp[ids[id]].m) * normal;
    // v_pos = (mvp[ids[id]].m * vec4(position, 1.0)).xyz;
    // mat4 mvp = mvp[id];
    gl_Position = vp * vec4(position, 1.0) +
                  vec4(verts[v_id], 0) * lights[visible_lights[id]].radius;
    // gl_Position.z *= -1;
    // gl_Position.y *= -1;
}
