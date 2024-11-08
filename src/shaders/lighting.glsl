
layout(set = 1, binding = 0) buffer lt { lightTemplate light_templates[]; };
layout(set = 1, binding = 1) buffer l { light lights[]; };
layout(set = 1, binding = 2) buffer c { tile tiles[]; };
layout(set = 1, binding = 3) buffer ll { uint light_list[]; };
layout(set = 1, binding = 4) buffer blh_ { BoundingLine blh[]; };

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

vec4 CalcPointLight(uint Index, vec3 v_pos, vec3 Normal) {

#define _l    lights[Index]
#define templ light_templates[_l.templ]

    vec3 LightDirection = v_pos - _l.pos;
    float Distance = length(LightDirection);
    LightDirection = normalize(LightDirection);

    vec4 Color = CalcLightInternal(templ, LightDirection, Normal);
    float Attenuation = templ.atten.constant + templ.atten.linear * Distance + templ.atten.exponential * Distance * Distance;

    return Color * templ.atten.brightness / Attenuation;
#undef _l
#undef templ
}

const vec3 LIGHT = vec3(1.0, 1.0, -0.7);
const uint MAX_LIT = 256;
const uint MAX_ITER = 1024;
// const uint MAX_ITER = 1024;
const bool USE_BLH = true;

vec3 calc_light(vec3 v_pos, vec3 v_normal, vec3 cam_pos, vec2 screen_dims) {
    // vec4 total_light = vec4(vec3(0.05), 1.0f);
    // float brightness = dot(normalize(v_normal), normalize(LIGHT)) * 0.3;
    // total_light += vec4(vec3(brightness), 1.0f);
    vec4 total_light = vec4(0.0f);
    vec2 coord = gl_FragCoord.xy;
    coord.y = abs(screen_dims.y) - coord.y - 1;
    vec2 screen_ratio = coord.xy / screen_dims;
    uint lit_times = 0;
    uint light_ids[256];
    float z = distance(cam_pos, v_pos);
    int num_iter = 0;
    for (int l = 0; l < MAX_LEVEL; ++l) {   // iterate through light quadtree levels
        ivec2 ti = ivec2(screen_ratio * _light_quadtree_widths[l]);
        uint tileIndex = _light_quadtree_offsets[l] + uint(ti.x + (-ti.y) * _light_quadtree_widths[l]);

        if (tiles[tileIndex].count == 0) continue;

        if (!USE_BLH) {
            for (uint i = tiles[tileIndex].offset; i < tiles[tileIndex].offset + tiles[tileIndex].count && num_iter < MAX_ITER; ++i) {
                uint l_id = light_list[i];
                vec3 l_pos = v_pos - lights[l_id].pos;
                float radius = lights[l_id].radius;
                if (dot(l_pos, l_pos) < radius * radius) {
                    light_ids[lit_times++] = l_id;
                    if (lit_times > MAX_LIT) break;
                }
                num_iter++;
            }
            continue;
        }

        if (tiles[tileIndex].BLH_offset < -1) {
            uint l_id = -tiles[tileIndex].BLH_offset - 2;
            vec3 l_pos = v_pos - lights[l_id].pos;
            float radius = lights[l_id].radius;
            if (dot(l_pos, l_pos) < radius * radius) {
                light_ids[lit_times++] = l_id;
                if (lit_times > MAX_LIT) break;
            }
            continue;
        }
        if (tiles[tileIndex].BLH_offset == -1) {
            continue;
        }

        int stack[32];
        int stack_ptr = 0;
        stack[stack_ptr++] = tiles[tileIndex].BLH_offset;

        uint _z = float_to_uint(z);
        while (stack_ptr > 0 && lit_times < MAX_LIT && num_iter < MAX_ITER) {
            uint blh_ptr = stack[--stack_ptr];

            if (blh_ptr >= blh.length()) break;   // Check for out-of-bounds access

            if (blh[blh_ptr].start <= _z && blh[blh_ptr].end >= _z) {
                int front = blh[blh_ptr].front;
                int back = blh[blh_ptr].back;

                if (front < -1) {   // front is a light id
                    int l_id = -front - 2;
                    vec3 l_pos = v_pos - lights[l_id].pos;
                    float radius = lights[l_id].radius;
                    if (dot(l_pos, l_pos) < radius * radius) {
                        light_ids[lit_times++] = l_id;
                        if (lit_times > MAX_LIT) break;
                    }
                } else if (front >= 0) {
                    if (stack_ptr < 32) stack[stack_ptr++] = front;   // Check for stack overflow
                }
                if (back < -1) {   // back is a light id
                    int l_id = -back - 2;
                    vec3 l_pos = v_pos - lights[l_id].pos;
                    float radius = lights[l_id].radius;
                    if (dot(l_pos, l_pos) < radius * radius) {
                        light_ids[lit_times++] = l_id;
                        if (lit_times > MAX_LIT) break;
                    }
                } else if (back >= 0) {
                    if (stack_ptr < 32) stack[stack_ptr++] = back;   // Check for stack overflow
                }
            }
            num_iter++;
        }
    }
    for (uint i = 0; i < lit_times; ++i) {
        uint l_id = light_ids[i];
        total_light += CalcPointLight(l_id, v_pos, v_normal);
    }
    total_light.a = 1.0f;
    return total_light.rgb;
    // f_color = texture(tex, coords) * total_light;   // min(brightness + 0.8, 1.0);
}
