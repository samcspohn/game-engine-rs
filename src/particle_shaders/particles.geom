#version 450
#include "../util.glsl"
#include "particle.glsl"
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

layout (location = 0) in int[] id;
layout (location = 0) out vec4 color;


layout(set = 0, binding = 0) buffer _p {
    pos_lif p_l[];
};
layout(set = 0, binding = 6) buffer pti {
    int template_ids[];
};
// layout(set = 0, binding = 2) buffer p {
//     particle particles[];
// };
layout(set = 0, binding = 3) buffer pt {
    particle_template templates[];
};
layout(set = 0, binding = 4) uniform Data {
    mat4 view;
    mat4 proj;
    vec3 cam_pos;
    vec4 cam_rot;
};
layout(set = 0, binding = 5) buffer a {
    int sorted[];
};
vec4 look_at = lookAt(rotate3(cam_rot) * vec3(0,0,1), rotate3(cam_rot) * vec3(0,1,0));

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

    mat4 model = translate(p_l[i].pos) * rotate(look_at);
    mat4 mvp = proj * view * model;
    particle_template templ = templates[template_ids[i]];
    gl_Position = mvp * vec4(-.5, .5, 0.0, 1.0);
    color = vec4(templ.color, p_l[i].life * 0.5);
    EmitVertex();

    gl_Position = mvp * vec4( -.5, -.5, 0.0, 1.0);
    color = vec4(templ.color, p_l[i].life * 0.5);
    EmitVertex();
    
    gl_Position = mvp * vec4( .5, .5, 0.0, 1.0);
    color = vec4(templ.color, p_l[i].life * 0.5);
    EmitVertex();

    gl_Position = mvp * vec4( .5, -.5, 0.0, 1.0);
    color = vec4(templ.color, p_l[i].life * 0.5);
    EmitVertex();
    
    EndPrimitive();

}  