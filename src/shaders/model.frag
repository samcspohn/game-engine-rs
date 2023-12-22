#version 450
#include "util.glsl"

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 coords;
layout(location = 2) in vec3 v_pos;
layout(location = 3) in vec3 _v;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform sampler2D tex;
// layout(set = 0, binding = 3) buffer tr1 { transform transforms[]; };
layout(set = 0, binding = 3) buffer lt { lightTemplate light_templates[]; };
// layout(set = 0, binding = 5) uniform Data {
//     vec3 cam_pos;
//     uint num_lights;
// };
layout(set = 0, binding = 4) buffer l { light lights[]; };
// layout(set = 0, binding = 7) buffer lid { uint light_ids[]; };
// layout(set = 0, binding = 8) buffer b { uint buckets[]; };
// layout(set = 0, binding = 9) buffer bc { uint buckets_count[]; };
// const float bucket_size = 120.0;
layout(set = 0, binding = 5) buffer c { cluster clusters[16][9][32]; };

vec4 CalcLightInternal(lightTemplate Light, vec3 LightDirection, vec3 Normal) {
    vec4 AmbientColor = vec4(Light.Color, 1.0f);
    float DiffuseFactor = dot(Normal, -LightDirection);

    vec4 DiffuseColor = vec4(0, 0, 0, 0);
    vec4 SpecularColor = vec4(0, 0, 0, 0);

    if (DiffuseFactor > 0) {
        DiffuseColor = vec4(Light.Color * DiffuseFactor, 1.0f);
        // vec3 VertexToEye = normalize(cam_pos - v_pos);
        // vec3 LightReflect = normalize(reflect(LightDirection, Normal));
        // float SpecularFactor = dot(VertexToEye, LightReflect);
        // if (SpecularFactor > 0) {
        //     SpecularFactor = pow(SpecularFactor, gSpecularPower);
        //     SpecularColor = vec4(Light.Color * gMatSpecularIntensity *
        //     SpecularFactor, 1.0f);
        // }
    }

    return (AmbientColor + DiffuseColor + SpecularColor);
}

vec4 CalcPointLight(uint Index, vec3 Normal) {

#define _l    lights[Index]
#define templ light_templates[_l.templ]

    vec3 LightDirection = v_pos - _l.pos;
    float Distance = length(LightDirection);
    LightDirection = normalize(LightDirection);

    vec4 Color = CalcLightInternal(templ, LightDirection, Normal);
    float Attenuation = templ.atten.constant + templ.atten.linear * Distance +
                        templ.atten.exponential * Distance * Distance;

    return Color * templ.atten.brightness / Attenuation;
}

const vec3 LIGHT = vec3(1.0, 1.0, -0.7);

void main() {
    vec4 total_light = vec4(vec3(0.05), 1.0f);
    float brightness = dot(normalize(v_normal), normalize(LIGHT)) * 0.3;
    total_light += vec4(vec3(brightness), 1.0f);

    // uint hashes[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    // uint curr_hash = 0;
    // for (int x = -1; x <= 1; x += 2) {
    //     for (int y = -1; y <= 1; y += 2) {
    //         for (int z = -1; z <= 1; z += 2) {
    //             uint hash = hash_pos(v_pos + vec3(x, y, z) * 8);
    //             uint hash_hash = hash % 8;
    //             for (int i = 0; i < 8; ++i) {

    //                 if (hashes[hash_hash] == -1 || hashes[hash_hash] == hash)
    //                 {
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
    //                 total_light += CalcPointLight(l_id, v_normal);
    //             }
    //         }
    //     }
    // }
    // uint hash = hash_pos(v_pos);
    // uint count = buckets_count[hash];
    // // uint lights_in_range[8];
    // // uint num_lights = 0;
    uvec3 v = uvec3(_v.x, _v.y, _v.z);
#define _cluster clusters[v.x][v.y][v.z]
    uint count = _cluster.count;
    for (int i = 0; i < count; ++i) {
        uint l_id = _cluster.lights[i];
        vec3 l_pos = v_pos - lights[l_id].pos;
        float radius = lights[l_id].radius;
        if (dot(l_pos, l_pos) < radius * radius) {
            // lights_in_range[num_lights] = l_id;
            // num_lights += 1;
            // if (num_lights >= 8) break;
            total_light += CalcPointLight(l_id, v_normal);
        }
    }
    // for (int i = 0; i < num_lights; ++i) {
    //     total_light += CalcPointLight(lights_in_range[i], v_normal);
    // }
    total_light.a = 1.0f;
    f_color =
        texture(tex, coords) * total_light;   // min(brightness + 0.8, 1.0);
    // f_color = vec4(_v, 1);
}
