#version 450
#include "util.glsl"
#include "lighting.glsl"

#define BLOCK_SIZE 16

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 coords;
layout(location = 2) in vec3 v_pos;
layout(location = 3) in vec3 _v;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 7) uniform sampler2D tex;
layout(set = 0, binding = 8) uniform Data { vec2 screen_dims; };

layout(push_constant) uniform Constants { vec3 cam_pos; };

void main() {
    vec4 total_light = vec4(vec3(0.05), 1.0f);
    float brightness = dot(normalize(v_normal), normalize(LIGHT)) * 0.3;
    total_light += vec4(vec3(brightness), 1.0f);
    total_light.a = 1.0f;
    total_light.rgb += calc_light(v_pos, v_normal, cam_pos, screen_dims);
    f_color = texture(tex, coords) * total_light;   // min(brightness + 0.8, 1.0);
}
