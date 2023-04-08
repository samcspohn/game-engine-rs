#version 450
#include "../util.glsl"
#include "particle.glsl"

struct transform {
    vec3 position;
    vec4 rotation;
    vec3 scale;
};

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

layout (location = 0) in int[] id;
layout (location = 0) out vec4 color;


layout(set = 0, binding = 0) buffer _p {
    pos_lif p_l[];
};
layout(set = 0, binding = 3) buffer pt {
    particle_template templates[];
};
layout(set = 0, binding = 5) buffer a {
    int sorted[];
};
layout(set = 0, binding = 6) buffer pti {
    int template_ids[];
};
layout(set = 0, binding = 7) buffer _n {
    int next[];
};
layout(set = 0, binding = 8) buffer t {
    transform transforms[];
};
layout(set = 0, binding = 4) uniform Data {
    mat4 view;
    mat4 proj;
    vec3 cam_pos;
    vec4 cam_rot;
};
vec4 look_at = lookAt(rotate3(cam_rot) * vec3(0,0,1), rotate3(cam_rot) * vec3(0,1,0));

const vec4 vert_pos[4] = {vec4(-1., 1., 0.0, 1.0),vec4( -1., -1., 0.0, 1.0),vec4( 1, 1., 0.0, 1.0),vec4( 1., -1., 0.0, 1.0)};
vec4 get_position(in mat4 mvp, in particle_template templ, int vert_id) {
    vec4 vert = vert_pos[vert_id];
    return mvp * vert; //vec4(-.5, .5, 0.0, 1.0);
}
void main() {
    // int i = id[0];
    // mat4 model = translate(sorted[i].pos) * rotate(sorted[i].rot);
    // mat4 mvp = proj * view * model;
    // particle_template templ = templates[sorted[i].proto_id];
    // gl_Position = mvp * vec4(-.5, .5, 0.0, 1.0);
    // color = vec4(templ.color, sorted[i].life * 0.5);
    // EmitVertex();

    // gl_Position = mvp * vec4( -.5, -.5, 0.0, 1.0);
    // color = vec4(templ.color, sorted[i].life * 0.5);
    // EmitVertex();
    
    // gl_Position = mvp * vec4( .5, .5, 0.0, 1.0);
    // color = vec4(templ.color, sorted[i].life * 0.5);
    // EmitVertex();

    // gl_Position = mvp * vec4( .5, -.5, 0.0, 1.0);
    // color = vec4(templ.color, sorted[i].life * 0.5);
    // EmitVertex();
    
    // EndPrimitive();

    int _i = id[0];
    int i = sorted[_i];
    particle_template templ = templates[template_ids[i]];

    mat4 model;
    vec4 color1;
    vec4 color2;
    if(templ.trail == 1) {
        vec3 next_pos;
        color2 = templ.color_life[int((1. - p_l[i].life) * 199)];
        if (next[i] >= 0) {
            next_pos = p_l[next[i]].pos;
            color1 = templ.color_life[int((1. - p_l[next[i]].life) * 199)];
        } else if (next[i] < -1) {
            next_pos = transforms[-next[i] - 2].position;
            color1 = templ.color_life[0];
        } else {
            next_pos = p_l[i].pos;
            color1 = color2;
        }
        vec3 v = next_pos - p_l[i].pos;
        vec3 x = cross(v, cam_pos);
        x = cross(x,v);
        vec4 l = lookAt(x, v);
        model = translate(p_l[i].pos + v / 2.f) * rotate(l) * scale(vec3(1,length(v) / 2,1)) ;
    } else {
        color1 = color2 = templ.color_life[int((1. - p_l[i].life) * 199)];
        model = translate(p_l[i].pos) * rotate(look_at);
    }
    mat4 mvp = proj * view * model;
    gl_Position = get_position(mvp,templ,0);//mvp * vec4(-.5, .5, 0.0, 1.0);
    color = color1;
    EmitVertex();

    gl_Position = get_position(mvp,templ,1);
    color = color2;
    EmitVertex();
    
    gl_Position = get_position(mvp,templ,2);
    color = color1;
    EmitVertex();

    gl_Position = get_position(mvp,templ,3);
    color = color2;
    EmitVertex();
    
    EndPrimitive();

}  