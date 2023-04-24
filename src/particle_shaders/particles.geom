#version 450
#include "../util.glsl"
#include "particle.glsl"

struct transform {
    vec3 position;
    vec4 rotation;
    vec3 scale;
};

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) in int[] id;
layout(location = 0) out float life;
layout(location = 1) out int templ_id;

layout(set = 0, binding = 0) buffer _p { pos_lif p_l[]; };
layout(set = 0, binding = 3) buffer pt { particle_template templates[]; };
layout(set = 0, binding = 5) buffer a { int sorted[]; };
layout(set = 0, binding = 6) buffer pti { int template_ids[]; };
layout(set = 0, binding = 7) buffer _n { int next[]; };
layout(set = 0, binding = 8) buffer t { transform transforms[]; };
layout(set = 0, binding = 4) uniform Data {
    mat4 view;
    mat4 proj;
    vec3 cam_pos;
    vec4 cam_rot;
};
vec4 look_at =
    lookAt(rotate3(cam_rot) * vec3(0, 0, 1), rotate3(cam_rot) * vec3(0, 1, 0));

const vec4 vert_pos[4] = {vec4(-1., 1., 0.0, 1.0), vec4(-1., -1., 0.0, 1.0),
                          vec4(1, 1., 0.0, 1.0), vec4(1., -1., 0.0, 1.0)};
vec4 get_position(in mat4 mvp, in particle_template templ, int vert_id) {
    vec4 vert = vert_pos[vert_id];
    return mvp * vert;   // vec4(-.5, .5, 0.0, 1.0);
}
#define templ templates[template_ids[i]]

void main() {
    int _i = id[0];
    int i = sorted[_i];
    // particle_template templ = templates[template_ids[i]];

    mat4 model;
    float color1;
    float color2;
    if (templ.trail == 1) {
        vec3 next_pos;
        color2 = 1. - p_l[i].life;
        if (next[i] >= 0) { // next is particle
            next_pos = p_l[next[i]].pos;
            color1 = 1. - p_l[next[i]].life;
        } else if (next[i] < -1) { // next is emitter / transform
            next_pos = transforms[-next[i] - 2].position;
            color1 = 0.;
        } else { // next is invalid
            next_pos = p_l[i].pos;
            color1 = color2;
        }
        vec3 v = next_pos - p_l[i].pos;
        vec3 x = cross(v, cam_pos - p_l[i].pos);
        x = cross(x, v);
        vec4 l = lookAt(x, v);
        model = translate(p_l[i].pos + v / 2.f) * rotate(l) *
                scale(vec3(1, length(v) / 2, 1));
    } else {
        color1 = color2 = 1. - p_l[i].life;//templ.color_life[int((1. - p_l[i].life) * 199)];
        model = translate(p_l[i].pos) * rotate(look_at);
    }
    mat4 mvp = proj * view * model;
    templ_id = template_ids[i];
    gl_Position = get_position(mvp, templ, 0);
    life = color1;
    EmitVertex();

    gl_Position = get_position(mvp, templ, 1);
    life = color2;
    EmitVertex();

    gl_Position = get_position(mvp, templ, 2);
    life = color1;
    EmitVertex();

    gl_Position = get_position(mvp, templ, 3);
    life = color2;
    EmitVertex();

    EndPrimitive();
}