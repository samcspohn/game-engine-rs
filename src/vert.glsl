#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

// The per-instance data.
layout(location = 2) in vec3 pos;

layout(location = 0) out vec3 v_normal;

layout(set = 0, binding = 0) uniform Data {
    mat4 view;
    mat4 proj;
} uniforms;

    // mat4 world;
void main() {
    mat4 worldview = uniforms.view;
    // v_normal = transpose(inverse(mat3(worldview))) * normal;
    v_normal = normal;
    gl_Position = uniforms.proj * worldview * vec4(pos + position, 1.0);
}
