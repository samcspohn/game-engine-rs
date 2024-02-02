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

vec4 CalcLightInternal(lightTemplate Light, vec3 LightDirection, vec3 Normal) {
    vec4 AmbientColor = vec4(Light.color, 1.0f);
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

    return (AmbientColor + DiffuseColor + SpecularColor);
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

void main() {
    vec4 total_light = vec4(vec3(0.05), 1.0f);
    float brightness = dot(normalize(v_normal), normalize(LIGHT)) * 0.3;
    total_light += vec4(vec3(brightness), 1.0f);
    vec2 screen_ratio = gl_FragCoord.xy / screen_dims;
    for (int l = 0; l < MAX_LEVEL; ++l) {   // iterate through light quadtree levels

        ivec2 ti = ivec2(screen_ratio * widths[l]);
        uint tileIndex = offsets[l] + uint(ti.x + (ti.y) * -widths[l]);

#define _cluster tiles[tileIndex]
        uint count = min(_cluster.count, MAX_LIGHTS_PER_TILE);
        for (int i = 0; i < count; ++i) {
            uint l_id = _cluster.lights[i];
            vec3 l_pos = v_pos - lights[l_id].pos;
            float radius = lights[l_id].radius;
            if (dot(l_pos, l_pos) < radius * radius) {
                total_light += CalcPointLight(l_id, v_normal);
            }
        }
    }

    total_light.a = 1.0f;
    f_color = texture(tex, coords) * total_light;   // min(brightness + 0.8, 1.0);
    // f_color = vec4(_v, 1);
}
