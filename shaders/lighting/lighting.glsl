#include "../util.glsl"
// #extension GL_EXT_control_flow_attributes: enable
layout(set = 1, binding = 0) buffer lt { lightTemplate light_templates[]; };
layout(set = 1, binding = 1) buffer l { light lights[]; };
layout(set = 1, binding = 2) buffer c { tile tiles[]; };
layout(set = 1, binding = 3) buffer ll { uint light_list[]; };
layout(set = 1, binding = 4) buffer blh_ { BoundingLine blh[]; };
layout(set = 1, binding = 5) buffer bvh_ { BoundingBox bvh[]; };

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

#ifndef GEOM
vec3 calc_light(vec3 v_pos, vec3 v_normal, vec3 cam_pos, vec2 screen_dims) {
    // vec4 total_light = vec4(vec3(0.05), 1.0f);
    // float brightness = dot(normalize(v_normal), normalize(LIGHT)) * 0.3;
    // total_light += vec4(vec3(brightness), 1.0f);
    vec4 total_light = vec4(0.0f);
    vec2 coord = gl_FragCoord.xy;
    coord.y = abs(screen_dims.y) - coord.y - 1;
    vec2 screen_ratio = coord.xy / abs(screen_dims);
    uint lit_times = 0;
    uint light_ids[MAX_LIT];
    float z = distance(cam_pos, v_pos);
    int num_iter = 0;
    for (int l = 0; l < MAX_LEVEL; ++l) {   // iterate through light quadtree levels
        ivec2 ti = ivec2(screen_ratio * _light_quadtree_widths[l]);
        // uint tileIndex = _light_quadtree_offsets[l] + uint(ti.x + (ti.y) * _light_quadtree_widths[l]);
        uint tileIndex = get_tile(ti.x, ti.y, l);

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
#endif

// vec3 calc_light_p(vec3 v_pos, vec3 cam_pos, vec2 screen_dims) {
//     vec3 norm = normalize(v_pos - cam_pos);
//     return calc_light(v_pos, norm, cam_pos, screen_dims);
// }

// vec4 CalcLightInternal_p(lightTemplate Light, vec3 LightDirection) {
//     // vec4 AmbientColor = vec4(Light.color, 1.0f);
//     // float DiffuseFactor = dot(Normal, -LightDirection);

//     // vec4 DiffuseColor = vec4(0, 0, 0, 0);
//     // vec4 SpecularColor = vec4(0, 0, 0, 0);

//     if (DiffuseFactor > 0) {
//         DiffuseColor = vec4(Light.color, 1.0f);
//         // vec3 VertexToEye = normalize(cam_pos - v_pos);
//         // vec3 LightReflect = normalize(reflect(LightDirection, Normal));
//         // float SpecularFactor = dot(VertexToEye, LightReflect);
//         // if (SpecularFactor > 0) {
//         //     SpecularFactor = pow(SpecularFactor, gSpecularPower);
//         //     SpecularColor = vec4(Light.Color * gMatSpecularIntensity *
//         //     SpecularFactor, 1.0f);
//         // }
//     }

//     return (DiffuseColor + SpecularColor);
// }

vec4 CalcPointLight_p(uint Index, vec3 v_pos) {

#define _l    lights[Index]
#define templ light_templates[_l.templ]

    vec3 LightDirection = v_pos - _l.pos;
    float Distance = length(LightDirection);
    // LightDirection = normalize(LightDirection);

    // vec4 Color = CalcLightInternal_p(templ, LightDirection, Normal);
    vec4 Color = vec4(templ.color, 1.0f);
    float Attenuation = templ.atten.constant + templ.atten.linear * Distance + templ.atten.exponential * Distance * Distance;

    return Color * templ.atten.brightness / Attenuation;
#undef _l
#undef templ
}
#ifndef GEOM
vec3 calc_light_p(vec3 v_pos, vec3 cam_pos, vec2 screen_dims) {
    // vec4 total_light = vec4(vec3(0.05), 1.0f);
    // float brightness = dot(normalize(v_normal), normalize(LIGHT)) * 0.3;
    // total_light += vec4(vec3(brightness), 1.0f);
    vec4 total_light = vec4(0.0f);
    vec2 coord = gl_FragCoord.xy;
    coord.y = abs(screen_dims.y) - coord.y - 1;
    vec2 screen_ratio = coord.xy / abs(screen_dims);
    uint lit_times = 0;
    const uint max_p_lit = 32;
    uint light_ids[max_p_lit];
    float z = distance(cam_pos, v_pos);
    int num_iter = 0;
    const float max_brightness = 10.f;
    for (int l = 0; l < MAX_LEVEL; ++l) {   // iterate through light quadtree levels
        ivec2 ti = ivec2(screen_ratio * _light_quadtree_widths[l]);
        // uint tileIndex = _light_quadtree_offsets[l] + uint(ti.x + (ti.y) * _light_quadtree_widths[l]);
        uint tileIndex = get_tile(ti.x, ti.y, l);

        if (tiles[tileIndex].count == 0) continue;

        if (tiles[tileIndex].BLH_offset < -1) {
            uint l_id = -tiles[tileIndex].BLH_offset - 2;
            vec3 l_pos = v_pos - lights[l_id].pos;
            float radius = lights[l_id].radius;
            if (dot(l_pos, l_pos) < radius * radius) {
                light_ids[lit_times++] = l_id;
                // total_light += CalcPointLight_p(l_id, v_pos);
                // if (lit_times > max_p_lit || length(total_light) > max_brightness) return total_light.rgb;
                if (lit_times > max_p_lit) break;
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
        while (stack_ptr > 0 && lit_times < max_p_lit && num_iter < MAX_ITER) {
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
                        // total_light += CalcPointLight_p(l_id, v_pos);
                        // if (lit_times > max_p_lit || length(total_light) > max_brightness) return total_light.rgb;
                        if (lit_times > max_p_lit) break;
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
                        // total_light += CalcPointLight_p(l_id, v_pos);
                        // if (lit_times > max_p_lit || length(total_light) > max_brightness) return total_light.rgb;
                        if (lit_times > max_p_lit) break;
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
        total_light += CalcPointLight_p(l_id, v_pos);
    }
    // total_light.a = 1.0f;
    return total_light.rgb;
    // f_color = texture(tex, coords) * total_light;   // min(brightness + 0.8, 1.0);
}
#endif

const uint MAX_LIGHTS_PER_PARTICLE = 128;
float sqr(float x) { return x * x; }
void get_light_list_bvh(vec2 screen_ratio, AABB aabb, vec3 cam_pos, inout uint light_ids[MAX_LIGHTS_PER_PARTICLE], inout uint lit_times) {
    lit_times = 0;
    const uint max_p_lit = 8;
    // float z = distance(cam_pos, v_pos);
    int num_iter = 0;
    const float max_brightness = 10.f;
    // p_size *= p_size;
    int stack[32 + 16];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    uvec3 aabb_min = uvec3(float_to_uint(aabb._min.x), float_to_uint(aabb._min.y), float_to_uint(aabb._min.z));
    uvec3 aabb_max = uvec3(float_to_uint(aabb._max.x), float_to_uint(aabb._max.y), float_to_uint(aabb._max.z));
    vec3 aabb_center = (aabb._min + aabb._max) * 0.5;

    while (stack_ptr > 0 && lit_times < MAX_LIGHTS_PER_PARTICLE) {
        uint bvh_ptr = stack[--stack_ptr];

        if (bvh_ptr >= bvh.length()) break;   // Check for out-of-bounds access

        // AABB-AABB intersection test - check if the AABBs overlap
        if (aabb_min.x <= bvh[bvh_ptr].max.x && aabb_max.x >= bvh[bvh_ptr].min.x &&
            aabb_min.y <= bvh[bvh_ptr].max.y && aabb_max.y >= bvh[bvh_ptr].min.y &&
            aabb_min.z <= bvh[bvh_ptr].max.z && aabb_max.z >= bvh[bvh_ptr].min.z) {

            int front = bvh[bvh_ptr].left;
            int back = bvh[bvh_ptr].right;

            // if (front >= 0 && back >= 0) {
            //     vec3 front_aabb_min = vec3(uint_to_float(bvh[front].min.x), uint_to_float(bvh[front].min.y), uint_to_float(bvh[front].min.z));
            //     vec3 front_aabb_max = vec3(uint_to_float(bvh[front].max.x), uint_to_float(bvh[front].max.y), uint_to_float(bvh[front].max.z));
            //     vec3 back_aabb_min = vec3(uint_to_float(bvh[back].min.x), uint_to_float(bvh[back].min.y), uint_to_float(bvh[back].min.z));
            //     vec3 back_aabb_max = vec3(uint_to_float(bvh[back].max.x), uint_to_float(bvh[back].max.y), uint_to_float(bvh[back].max.z));

            //     vec3 front_closest_point = max(aabb._min, min(aabb_center, front_aabb_max));
            //     vec3 back_closest_point = max(aabb._min, min(aabb_center, back_aabb_max));
            //     vec3 front_dist = front_closest_point - aabb_center;
            //     vec3 back_dist = back_closest_point - aabb_center;

            //     if(dot(front_dist, front_dist) < dot(back_dist, back_dist)) {
            //         if (stack_ptr < 32) stack[stack_ptr++] = front;   // Check for stack overflow
            //         if (stack_ptr < 32) stack[stack_ptr++] = back;
            //     } else {
            //         if (stack_ptr < 32) stack[stack_ptr++] = back;   // Check for stack overflow
            //         if (stack_ptr < 32) stack[stack_ptr++] = front;
            //     }
            //     continue;
            // }

            if (front < -1) {   // front is a light id
                int l_id = -front - 2;
                vec3 light_pos = lights[l_id].pos;
                float radius = lights[l_id].radius;
                
                // Test if light sphere intersects with query AABB
                vec3 closest_point = max(aabb._min, min(light_pos, aabb._max));
                // vec3 closest_point = (aabb._min + aabb._max) * 0.5;
                vec3 dist = closest_point - light_pos;
                
                if (dot(dist, dist) <= sqr(radius)) {
                    light_ids[lit_times++] = l_id;
                }
            } else if (front >= 0) {
                if (stack_ptr < 32) stack[stack_ptr++] = front;   // Check for stack overflow
            }

            if (back < -1) {   // back is a light id
                int l_id = -back - 2;
                vec3 light_pos = lights[l_id].pos;
                float radius = lights[l_id].radius;
                
                // Test if light sphere intersects with query AABB
                vec3 closest_point = max(aabb._min, min(light_pos, aabb._max));
                // vec3 closest_point = (aabb._min + aabb._max) * 0.5;
                vec3 dist = closest_point - light_pos;
                
                if (dot(dist, dist) <= sqr(radius)) {
                    light_ids[lit_times++] = l_id;
                }
            } else if (back >= 0) {
                if (stack_ptr < 32) stack[stack_ptr++] = back;   // Check for stack overflow
            }
        }
        // else {
        //     // Optimization: If parent doesn't intersect, we might still need to test children
        //     // as they could have tighter bounds. This is especially important for BVHs
        //     // created using Morton codes or other methods where child AABBs might not be
        //     // strictly contained within parent AABBs
        //     int front = bvh[bvh_ptr].left;
        //     int back = bvh[bvh_ptr].right;
            
        //     if (front >= 0) {
        //         if (stack_ptr < 32) stack[stack_ptr++] = front;
        //     }
        //     if (back >= 0) {
        //         if (stack_ptr < 32) stack[stack_ptr++] = back;
        //     }
        // }
        num_iter++;
    }
}