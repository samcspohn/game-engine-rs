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
    // vec4 col3 = templates[templ_id].color_life[int(life * 255)];
    vec3 total_light = vec3(1);

    // uint hashes[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    // uint curr_hash = 0;
    // for (int x = -1; x <= 1; x += 2) {
    //     for (int y = -1; y <= 1; y += 2) {
    //         for (int z = -1; z <= 1; z += 2) {
    //             uint hash = hash_pos(v_pos + vec3(x, y, z) * 8);
    //             uint hash_hash = hash % 8;
    //             for (int i = 0; i < 8; ++i) {

    //                 if (hashes[hash_hash] == -1 || hashes[hash_hash] == hash) {
    //                     hashes[hash_hash] = hash;
    //                     break;
    //                 }
    //                 hash_hash = (hash_hash + 1) % 8;
    //             }
    //         }
    //     }
    // }
    // for (int i = 0; i < 8; ++i) {
    //     if (hashes[i] == -1) continue;
    //     uint hash = hashes[i];
    //     uint count = buckets_count[hash];
    //     if (count > 0) {
    //         uint offset = buckets[hash];
    //         for (uint i = 0; i < count; ++i) {
    //             uint l_id = light_ids[offset + i];
    //             vec3 v = v_pos - lights[l_id].pos;
    //             float radius = lights[l_id].radius;
    //             if (dot(v, v) < radius * radius) {
    //                 uint templ_id = lights[l_id].templ;
    //                 total_light += light_templates[templ_id].Color *
    //                                light_templates[templ_id].atten.brightness *
    //                                10 / length(v);
    //                 // total_light += CalcPointLight(l_id, v_normal);
    //             }
    //         }
    //     }
    // }

    // uint hash = hash_pos(v_pos);
    // uint count = buckets_count[hash];
    // // uint lights_in_range[8];
    // // uint num_lights = 0;
    // if (count > 0) {
    //     uint offset = buckets[hash];
    //     for (uint i = 0; i < count; ++i) {
    //         uint l_id = light_ids[offset + i];
    //         vec3 v = v_pos - lights[l_id].pos;
    //         float radius = lights[l_id].radius;
    //         if (dot(v, v) < radius * radius) {
    //             // lights_in_range[num_lights] = l_id;
    //             // num_lights += 1;
    //             // if (num_lights >= 8) break;
    //             // total_light += CalcPointLight(l_id, v_normal);
    //             uint templ_id = lights[l_id].templ;
    //             total_light += light_templates[templ_id].Color *
    //             light_templates[templ_id].atten.brightness * 10 / length(v);
    //         }
    //     }
    //     // for (int i = 0; i < num_lights; ++i) {
    //     //     total_light += CalcPointLight(lights_in_range[i], v_normal);
    //     // }
    // }

    FragColor = col * col2 * vec4(total_light, 1.0);
    // FragColor = vec4(1,0,0,1);
}