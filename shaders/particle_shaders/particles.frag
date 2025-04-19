#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_ARB_separate_shader_objects : enable
#include "../lighting/lighting.glsl"
#include "../util.glsl"
#include "particle.glsl"

layout(location = 0) in float life;
layout(location = 1) in flat int templ_id;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec2 uv2;
layout(location = 4) in vec3 v_pos;
layout(location = 5) in flat uint num_lights;
layout(location = 6) in flat uint offset;
layout(location = 7) in float y;
// layout(location = 6) in flat uint[MAX_LIGHTS_PER_PARTICLE] light_ids;
layout(location = 0) out vec4 FragColor;

const int BM_NONE = 0;
const int BM_BLEND = 1;
const int BM_ADD = 2;
const int BM_ADDX2 = 3;

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
layout(set = 0, binding = 10) buffer b10 {
    uint offset;
    uint particle_lighting[];
}
_pl_;
layout(set = 0, binding = 11) uniform sampler2D color_over_life;

layout(set = 0, binding = 12) uniform sampler2D[] s;
void main() {
#define _templ templates[templ_id]
    // particle_template templ = templates[templ_id];
    float _y = 1.0f;
    if (_templ.trail == 1) {
        float size_l = _templ.size_over_lifetime[int(life * 255)];
        _y = abs(y * 2.0f - 1.0f);
        if (_y > size_l) {
            discard;
        }
        _y = y * 2.0f - 1.0f;
        _y = _y / size_l;
        _y = (_y + 1.0f) / 2.0f;
    }
    vec4 col = texture(nonuniformEXT(s[_templ.tex_id]), uv);
    vec4 col2 = texture(color_over_life, uv2);
    // if (col.a < 0.01 || col2.a < 0.01) {
    //     discard;
    // }
    // vec4 col3 = templates[templ_id].color_life[int(life * 255)];
    vec4 total_light = vec4(1.0);
    uint lit_times = 0;
    int num_iter = 0;
    if (_templ.recieve_lighting == 1) {
        total_light = vec4(vec3(0.3), 1.0f);

        for (int i = 0; i < num_lights; ++i) {
            uint l_id = _pl_.particle_lighting[(offset + i) % (1 << 22)];
            total_light.rgb += CalcPointLight_p(l_id, v_pos).rgb;
        }
        // total_light = vec4(vec3(num_lights / 255), 1.0f);
        // total_light = vec4(vec3(0.3), 1.0f);
        // total_light.rgb += calc_light_p(v_pos, cam_pos, screen_dims);


        // int stack[32];
        // int stack_ptr = 0;
        // stack[stack_ptr++] = 0;
        // uvec3 _v_pos = uvec3(float_to_uint(v_pos.x), float_to_uint(v_pos.y), float_to_uint(v_pos.z));

        // // float rp2 = (radius + p_size) * (radius + p_size);
        // // float r2 = p_size * p_size;
        // while (stack_ptr > 0) {
        //     uint bvh_ptr = stack[--stack_ptr];

        //     if (bvh_ptr >= bvh.length()) break;   // Check for out-of-bounds access

        //     // AABB point intersection test - check if the position is within the bounding box
        //     if (_v_pos.x >= bvh[bvh_ptr].min.x && _v_pos.x <= bvh[bvh_ptr].max.x &&
        //         _v_pos.y >= bvh[bvh_ptr].min.y && _v_pos.y <= bvh[bvh_ptr].max.y &&
        //         _v_pos.z >= bvh[bvh_ptr].min.z && _v_pos.z <= bvh[bvh_ptr].max.z) {
                
        //         int front = bvh[bvh_ptr].left;
        //         int back = bvh[bvh_ptr].right;

        //         if (front < -1) {   // front is a light id
        //             int l_id = -front - 2;

        //             vec3 l_pos = v_pos - lights[l_id].pos;
        //             float radius = lights[l_id].radius;
        //             // Light testing still uses spheres
        //             if (dot(l_pos, l_pos) < sqr(radius)) {
        //                 lit_times++;
        //                 total_light += CalcPointLight_p(l_id, v_pos);
        //             }
        //         } else if (front >= 0) {
        //             if (stack_ptr < 32) stack[stack_ptr++] = front;   // Check for stack overflow
        //         }
                
        //         if (back < -1) {   // back is a light id
        //             int l_id = -back - 2;

        //             vec3 l_pos = v_pos - lights[l_id].pos;
        //             float radius = lights[l_id].radius;
        //             if (dot(l_pos, l_pos) < sqr(radius)) {
        //                 lit_times++;
        //                 total_light += CalcPointLight_p(l_id, v_pos);
        //             }
        //         } else if (back >= 0) {
        //             if (stack_ptr < 32) stack[stack_ptr++] = back;   // Check for stack overflow
        //         }
        //     } 
        //     // else {
        //     //     // Optional optimization: Potentially add distance-based prioritization for stack
        //     //     // For point queries with AABBs, you might want to push closer nodes first
                
        //     //     // Simple method: still test children if they might contain relevant lights
        //     //     int front = bvh[bvh_ptr].left;
        //     //     int back = bvh[bvh_ptr].right;
                
        //     //     if (front >= 0) {
        //     //         if (stack_ptr < 32) stack[stack_ptr++] = front;
        //     //     }
        //     //     if (back >= 0) {
        //     //         if (stack_ptr < 32) stack[stack_ptr++] = back;
        //     //     }
        //     // }
        //     num_iter++;
        // }

        total_light.a = 1.0f;
    }
    vec4 color = col * col2 * total_light;
    // if (num_iter == 1) {
    //     color.rgb = vec3(1.0, 0.0, 0.0);
    // }
    // float opacity = 0.0f;
    // // int blendMode = _templ.blend_mode;
    // int blendMode = BM_BLEND;
    // if(blendMode == BM_NONE) {
    //     opacity = 1.0f;
    //     color.a = 1.0f;
    // } else if(blendMode == BM_BLEND) {
    //     opacity = color.a;
    // } else if(blendMode == BM_ADD) {
    //     opacity = 0.0f;
    // } else if(blendMode == BM_ADDX2) {
    //     opacity = 0.0f;
    //     color *= 2.0f;
    // }
    // color.rgb *= color.a;
    // color.a = color.a * opacity;

    FragColor = color;
    //     fragColor = color;
    // fragTexCoord = inTexCoord;
    // fragOpacity = color.a;
    // FragColor = vec4(1,0,0,1);
}