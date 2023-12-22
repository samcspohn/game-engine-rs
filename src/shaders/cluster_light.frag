#version 450

// layout(location = 0) in vec3 v_normal;
// layout(location = 1) in vec2 coords;
// layout(location = 0) in vec3 v_pos;
// layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 3) buffer l { uint clusters[16][9][24]; };
void main() {
    vec3 v = gl_FragCoord.xyz;
    atomicAdd(clusters[uint(v.x)][uint(v.y)][uint(v.z * 24)], 1);
}