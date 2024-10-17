#version 450
#include "util.glsl"

#define BLOCK_SIZE 16

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 coords;
layout(location = 2) in vec3 v_pos;
layout(location = 3) in vec3 _v;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform sampler2D tex;
layout(set = 0, binding = 3) buffer lt { lightTemplate light_templates[]; };
layout(set = 0, binding = 4) buffer l { light lights[]; };
layout(set = 0, binding = 5) buffer c { tile tiles[]; };
layout(set = 0, binding = 6) uniform Data { vec2 screen_dims; };
layout(set = 0, binding = 7) buffer ll { uint light_list[]; };
layout(set = 0, binding = 8) buffer vl { uint visible_lights[]; };
layout(set = 0, binding = 9) buffer vlc { uint visible_lights_count; };
layout(set = 0, binding = 10) buffer blh_ { BoundingLine blh; };
layout(push_constant) uniform Constants { vec3 cam_pos; };

vec4 CalcLightInternal(lightTemplate Light, vec3 LightDirection, vec3 Normal) {
    // vec4 AmbientColor = vec4(Light.color, 1.0f);
    float DiffuseFactor = dot(Normal, -LightDirection);

    vec4 DiffuseColor = vec4(0, 0, 0, 0);
    vec4 SpecularColor = vec4(0, 0, 0, 0);

    if (DiffuseFactor > 0) {
        DiffuseColor = vec4(Light.color * DiffuseFactor, 1.0f);
        // vec3 VertexToEye = normalize(cam_pos - v_pos);
        // vec3 LightReflect = normalize(reflect(LightDirection, Normal));
        // float SpecularFactor = dot(VertexToEye, LightReflect);
        // if (SpecularFactor > 0) {
        //     SpecularFactor = pow(SpecularFactor, gSpecularPower);
        //     SpecularColor = vec4(Light.Color * gMatSpecularIntensity *
        //     SpecularFactor, 1.0f);
        // }
    }

    return (DiffuseColor + SpecularColor);
}

vec4 CalcPointLight(uint Index, vec3 Normal) {

#define _l    lights[Index]
#define templ light_templates[_l.templ]

    vec3 LightDirection = v_pos - _l.pos;
    float Distance = length(LightDirection);
    LightDirection = normalize(LightDirection);

    vec4 Color = CalcLightInternal(templ, LightDirection, Normal);
    float Attenuation = templ.atten.constant + templ.atten.linear * Distance + templ.atten.exponential * Distance * Distance;

    return Color * templ.atten.brightness / Attenuation;
}

const vec3 LIGHT = vec3(1.0, 1.0, -0.7);
const uint MAX_LIT = 256;
const uint MAX_ITER = 1024;

void main() {
    vec4 total_light = vec4(vec3(0.05), 1.0f);
    float brightness = dot(normalize(v_normal), normalize(LIGHT)) * 0.3;
    total_light += vec4(vec3(brightness), 1.0f);
    vec2 coord = gl_FragCoord.xy;
    coord.y = abs(screen_dims.y) - coord.y - 1;
    vec2 screen_ratio = coord.xy / screen_dims;
    uint lit_times = 0;
    float z = distance(cam_pos, v_pos);
    uint light_ids[256];
    uint lit_times = 0;
    // uint iters = 0;
    for (int l = 1; l < MAX_LEVEL; ++l) {   // iterate through light quadtree levels
        ivec2 ti = ivec2(screen_ratio * _light_quadtree_widths[l]);
        uint tileIndex = _light_quadtree_offsets[l] + uint(ti.x + (-ti.y) * _light_quadtree_widths[l]);
        if (tiles[tileIndex].count == 0) continue;

        uint stack[32];
        int stack_ptr = 0;
        stack[stack_ptr++] = tiles[tileIndex].BLH_offset;

        while (stack_ptr > 0) {
            uint blh_ptr = stack[--stack_ptr];
            if (blh[blh_ptr].front > z && blh[blh_ptr].back > z) {
                uint front = blh[blh_ptr].front;
                uint back = blh[blh_ptr].back;

                if (front < 1) {   // front is a light id
                    uint l_id = -front;
                    light_ids[lit_times++] = l_id;
                    if (lit_times > MAX_LIT) break;
                } else {
                    stack[stack_ptr++] = front;
                }
                if (back < 1) {   // back is a light id
                    uint l_id = -back;
                    light_ids[lit_times++] = l_id;
                    if (lit_times > MAX_LIT) break;
                } else {
                    stack[stack_ptr++] = back;
                }
            }
        }
    }
    for (uint i = 0; i < lit_times; ++i) {
        uint l_id = light_ids[i];
        vec3 l_pos = v_pos - lights[l_id].pos;
        float radius = lights[l_id].radius;
        if (dot(l_pos, l_pos) < radius * radius) {
            total_light += CalcPointLight(l_id, v_normal);
            lit_times++;
        }
        total_light += CalcPointLight(lights[i], v_normal);
    }
    // uint count = tiles[tileIndex].count;
    // uint offset = tiles[tileIndex].offset;
    // for (int i = 0; i < count; ++i) {
    //     uint l_id = light_list[offset + i];
    //     vec3 l_pos = v_pos - lights[l_id].pos;
    //     float radius = lights[l_id].radius;
    //     if (dot(l_pos, l_pos) < radius * radius) {
    //         total_light += CalcPointLight(l_id, v_normal);
    //         lit_times++;
    //     }
    //     // iters++;
    //     if (lit_times > MAX_LIT) {
    //         break;
    //     }
    // }
    // if (lit_times > MAX_LIT) {
    //     break;
    // }
    // for (uint i = 0; i < visible_lights_count; ++i) {
    //         uint l_id = visible_lights[i];
    //         // vec3 l_pos = v_pos - lights[l_id].pos;
    //         // float radius = lights[l_id].radius;
    //         // if (dot(l_pos, l_pos) < radius * radius) {
    //             total_light += CalcPointLight(l_id, v_normal);
    //         // lit_times++;
    //         // }
    //         // iters++;
    //         // if (iters > MAX_ITER || lit_times > MAX_LIT) {
    //         //     break;
    //         // }
    //     }

    total_light.a = 1.0f;
    f_color = texture(tex, coords) * total_light;   // min(brightness + 0.8, 1.0);
    // f_color = vec4(_v, 1);
}
