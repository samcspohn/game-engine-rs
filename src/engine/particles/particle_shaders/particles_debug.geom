#version 450
#include "../../../shaders/util.glsl"
#include "particle.glsl"

layout(points) in;
layout(triangle_strip, max_vertices = 48) out;

layout(location = 0) in int[] id;
layout(location = 0) out vec4 _color;

// layout(set = 0, binding = 0) buffer _p { pos_lif p_l[]; };
// layout(set = 0, binding = 0) buffer pl_c { pos_life_comp p_l[]; };
layout(set = 0, binding = 0) buffer pl_c { pos_lif p_l[]; };
layout(set = 0, binding = 3) buffer pt { particle_template templates[]; };
layout(set = 0, binding = 5) buffer a { int sorted[]; };
layout(set = 0, binding = 6) buffer pti { int template_ids[]; };
layout(set = 0, binding = 7) buffer _n { int next[]; };
layout(set = 0, binding = 8) buffer t { transform transforms[]; };
layout(set = 0, binding = 9) buffer p { particle particles[]; };
layout(set = 0, binding = 4) uniform Data {
    mat4 view;
    mat4 cam_inv_rot;
    mat4 proj;
    vec3 cam_pos;
    vec4 cam_rot;
    uint num_templates;
};
vec4 look_at = lookAt(rotate3(cam_rot) * vec3(0, 0, 1), rotate3(cam_rot) * vec3(0, 1, 0));

const vec4 vert_pos[4] = {vec4(-1., 1., 0.0, 1.0), vec4(-1., -1., 0.0, 1.0), vec4(1, 1., 0.0, 1.0), vec4(1., -1., 0.0, 1.0)};
const vec2 vert_uv[4] = {vec2(0., 1.), vec2(1, 1.), vec2(0., 0.), vec2(1., 0.)};
vec4 get_position(in mat4 vp, int vert_id) {
    vec4 vert = vert_pos[vert_id];
    return vp * vert;   // vec4(-.5, .5, 0.0, 1.0);
}

void draw_line(in mat4 vp, vec3 a, vec3 b, vec4 color, float _width) {
    vec4 _a = vp * vec4(a, 1);
    vec4 _b = vp * vec4(b, 1);
    vec4 z = vec4(0, 0, 1, 1);
    vec3 width_dir = normalize(cross(normalize(_a.xyz - _b.xyz), z.xyz));
    float desired_width = _width * (_a.w + _b.w) / 2;
    vec4 width = vec4(width_dir, 0) * (desired_width + _width) / 2;
    // vec3 offset = width_dir;
    _color = color;
    gl_Position = _a + width;
    EmitVertex();
    _color = color;
    gl_Position = _a - width;
    EmitVertex();
    _color = color;
    gl_Position = _b + width;
    EmitVertex();
    _color = color;
    gl_Position = _b - width;
    EmitVertex();
    EndPrimitive();
}

void main() {
#define templ templates[_templ_id]
#define p     particles[i]
    int _i = id[0];
    // #define jump 4
    //     for (int n = _i * jump; n < _i * jump + jump; ++n) {

    int i = sorted[_i];
    int _templ_id = template_ids[i];

    mat4 model;
    float l1;
    float l2;
    // vec3 pos = get_pos(p_l[i]);
    vec3 pos = p_l[i].pos;
    vec3 next_pos = pos;
    if (templ.trail == 1) {   // trail
        // l2 = 1. - get_life(p_l[i]);
        l2 = 1. - p_l[i].life;
        if (next[i] >= 0) {   // next is particle
            // next_pos = get_pos(p_l[next[i]]);
            // l1 = 1. - get_life(p_l[next[i]]);
            next_pos = p_l[next[i]].pos;
            l1 = 1. - p_l[next[i]].life;
        } else if (next[i] < -1) {   // next is emitter / transform
            next_pos = transforms[-next[i] - 2].position;
            l1 = 0.;
        } else {   // next is invalid
            next_pos = pos;
            l1 = l2;
        }
        vec3 v = (next_pos - cam_pos) - (pos - cam_pos);
        vec3 x = cross(v, (pos - cam_pos));
        x = cross(x, v);
        vec4 l = lookAt(x, v);
        model = translate(pos + v / 2.f) * rotate(l) * scale(vec3(templ.scale.x, length(v) / 2 * templ.scale.y, 1));
    } else {   // not trail / billboard / aligned to velocity
        // l1 = l2 = 1. - get_life(p_l[i]);
        l1 = l2 = 1. - p_l[i].life;
        vec4 rot;
        // uint a = templ.billboard & templ.align_vel << 1;
        // switch (a) {
        // case 1:   // billboard
        //     rot = look_at;
        //     break;
        // // case 002: // align_vel
        // //     rot =
        // case 3:   // billboard & align_vel
        //     rot = lookAt(cam_pos, p.vel);
        //     break;
        // default:
        //     rot = p.rot;
        //     break;
        // }
        if (templ.billboard == 1 && templ.align_vel == 1) {
            vec3 a = pos - cam_pos;
            vec3 b = cross(p.vel, a);
            vec3 c = cross(p.vel, b);

            // mat3 m;
            // m[0] = b;
            // m[1] = p.vel;
            // m[2] = c;
            // m = transpose(m);

            rot = lookAt(c, p.vel);
        } else if (templ.billboard == 1) {
            rot = look_at;
        } else {
            rot = particles[i].rot;
        }
        model = translate(pos) * rotate(rot) * scale(vec3(templ.scale.x, templ.scale.y, 1));
    }
    mat4 vp = proj * view;
    float radius = length(templ.scale);
    // AABB a;
    // a._min = min(pos, next_pos) - radius;
    // a._max = max(pos, next_pos) + radius;
    // float size = length(a._max - a._min);
    // a._max += size * 2;
    // a._min -= size * 2;
    // vec3 center = (pos + next_pos) / 2.0;
    // a._min = a._max = center;
    AABB a = p.aabb;
    vec3 corners[8] = {
        {a._min.x, a._min.y, a._min.z}, // 0, | 0, 0, 0|
        {a._min.x, a._min.y, a._max.z}, // 1, | 0, 0, 1|
        {a._min.x, a._max.y, a._min.z}, // 2, | 0, 1, 0|
        {a._min.x, a._max.y, a._max.z}, // 3, | 0, 1, 1|

        {a._max.x, a._min.y, a._min.z}, // 4, | 1, 0, 0|
        {a._max.x, a._min.y, a._max.z}, // 5, | 1, 0, 1|
        {a._max.x, a._max.y, a._min.z}, // 6, | 1, 1, 0|
        {a._max.x, a._max.y, a._max.z}  // 7, | 1, 1, 1|
    };

    vec4 color = vec4(1, 0, 0, 1);
    draw_line(vp, corners[0], corners[1], color, 0.006);   // left side
    draw_line(vp, corners[1], corners[3], color, 0.006);
    draw_line(vp, corners[3], corners[2], color, 0.006);
    draw_line(vp, corners[0], corners[2], color, 0.006);

    draw_line(vp, corners[4], corners[5], color, 0.006);   // right side
    draw_line(vp, corners[5], corners[7], color, 0.006);
    draw_line(vp, corners[7], corners[6], color, 0.006);
    draw_line(vp, corners[4], corners[6], color, 0.006);

    draw_line(vp, corners[0], corners[4], color, 0.006);   // left to right
    draw_line(vp, corners[1], corners[5], color, 0.006);
    draw_line(vp, corners[2], corners[6], color, 0.006);
    draw_line(vp, corners[3], corners[7], color, 0.006);

    // float offset = 0.5 / float(num_templates);
    // float tid = float(_templ_id + 0.5);
    // float color_id = tid / float(num_templates);
    // templ_id = template_ids[i];
    // gl_Position = get_position(vp, 0);
    // v_pos = (model * vert_pos[0]).xyz;
    // uv = vert_uv[0];
    // uv2 = vec2(l1, color_id);
    // life = l1;
    // EmitVertex();

    // gl_Position = get_position(vp, 1);
    // v_pos = (model * vert_pos[1]).xyz;
    // templ_id = template_ids[i];
    // uv = vert_uv[1];
    // uv2 = vec2(l2, color_id);
    // life = l2;
    // EmitVertex();

    // gl_Position = get_position(vp, 2);
    // v_pos = (model * vert_pos[2]).xyz;
    // templ_id = template_ids[i];
    // uv = vert_uv[2];
    // uv2 = vec2(l1, color_id);
    // life = l1;
    // EmitVertex();

    // gl_Position = get_position(vp, 3);
    // v_pos = (model * vert_pos[3]).xyz;
    // templ_id = template_ids[i];
    // uv = vert_uv[3];
    // uv2 = vec2(l2, color_id);
    // life = l2;
    // EmitVertex();

    // EndPrimitive();
    // }

    // // DEBUG POINT
    // model = translate(p_l[i].pos) * rotate(look_at) *
    // scale(vec3(0.5,0.5,0.5)); vp = proj * view * model;

    // gl_Position = get_position(vp, 0);
    // life = 0;
    // EmitVertex();

    // gl_Position = get_position(vp, 1);
    // life = 0;
    // EmitVertex();

    // gl_Position = get_position(vp, 2);
    // life = 0;
    // EmitVertex();

    // gl_Position = get_position(vp, 3);
    // life = 0;
    // EmitVertex();

    // EndPrimitive();
}