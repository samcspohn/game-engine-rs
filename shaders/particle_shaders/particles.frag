#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#include "../util.glsl"
#include "../lighting/lighting.glsl"
#include "particle.glsl"


layout(location = 0) in float life;
layout(location = 1) in flat int templ_id;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec2 uv2;
layout(location = 4) in vec3 v_pos;
layout(location = 0) out vec4 FragColor;

layout(set = 0, binding = 3) buffer pt { particle_template templates[]; };
layout(set = 0, binding = 4) uniform Data {
    mat4 view;
    mat4 cam_inv_rot;
    mat4 proj;
    vec3 cam_pos;
    uint num_templates;
    vec4 cam_rot;
    vec2 screen_dims;
};
layout(set = 0, binding = 10) uniform sampler2D color_over_life;

// layout(set = 0, binding = 11) buffer lt { lightTemplate light_templates[]; };

// layout(set = 0, binding = 12) buffer l { light lights[]; };
// layout(set = 0, binding = 13) buffer lid { uint light_ids[]; };
// layout(set = 0, binding = 14) buffer b { uint buckets[]; };
// layout(set = 0, binding = 15) buffer bc { uint buckets_count[]; };

layout(set = 0, binding = 16) uniform sampler2D[] s;
void main() {
#define _templ templates[templ_id]
    // particle_template templ = templates[templ_id];
    vec4 col = texture(nonuniformEXT(s[_templ.tex_id]), uv);
    vec4 col2 = texture(color_over_life, uv2);
    if (col.a < 0.01 || col2.a < 0.01) {
        discard;
    }
    // vec4 col3 = templates[templ_id].color_life[int(life * 255)];
    vec4 total_light = vec4(1.0);
    if (_templ.recieve_lighting == 1) {
        total_light = vec4(vec3(0.3), 1.0f);
        total_light.rgb += calc_light_p(v_pos, cam_pos, screen_dims);
    }

    FragColor = col * col2 * total_light;
    // FragColor = vec4(1,0,0,1);
}