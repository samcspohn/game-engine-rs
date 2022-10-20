#version 450
#include "../util.glsl"
layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

layout (location = 0) in int[] id;
layout (location = 0) out vec4 color;

struct _a {
    vec4 rot;
    vec3 pos;
    uint key;
    int proto_id;
    float life;
};
struct particle {
    vec3 vel;
    int emitter_id;
    vec4 rot;
    int proto_id;
    int sorted;
};
struct particle_template {
    vec3 color;
    float speed;
    float emission_rate;
};
struct pos_lif {
    vec3 pos;
    float life;
};

layout(set = 0, binding = 0) buffer _p {
    pos_lif p_l[];
};
layout(set = 0, binding = 1) buffer l {
    float life[];
};
layout(set = 0, binding = 2) buffer p {
    particle particles[];
};
layout(set = 0, binding = 3) buffer pt {
    particle_template templates[];
};
layout(set = 0, binding = 4) uniform Data {
    mat4 view;
    mat4 proj;
};
layout(set = 0, binding = 5) buffer a {
    _a sorted[];
};

void main() {
    int i = id[0];
    mat4 model = translate(sorted[i].pos) * rotate(sorted[i].rot);
    mat4 mvp = proj * view * model;
    particle_template templ = templates[sorted[i].proto_id];
    gl_Position = mvp * vec4(-.5, .5, 0.0, 1.0);
    color = vec4(templ.color, sorted[i].life * 0.5);
    EmitVertex();

    gl_Position = mvp * vec4( -.5, -.5, 0.0, 1.0);
    color = vec4(templ.color, sorted[i].life * 0.5);
    EmitVertex();
    
    gl_Position = mvp * vec4( .5, .5, 0.0, 1.0);
    color = vec4(templ.color, sorted[i].life * 0.5);
    EmitVertex();

    gl_Position = mvp * vec4( .5, -.5, 0.0, 1.0);
    color = vec4(templ.color, sorted[i].life * 0.5);
    EmitVertex();
    
    EndPrimitive();

    // int i = id[0];
    // int _i = sorted[0].proto_id;

    // if (p_l[i].life > 0){

    //     mat4 model = translate(p_l[i].pos) * rotate(particles[i].rot);
    //     mat4 mvp = proj * view * model;
    //     particle_template templ = templates[particles[i].proto_id];
    //     gl_Position = mvp * vec4(-.5, .5, 0.0, 1.0);
    //     color = vec4(templ.color, p_l[i].life * 0.5);
    //     EmitVertex();

    //     gl_Position = mvp * vec4( -.5, -.5, 0.0, 1.0);
    //     color = vec4(templ.color, p_l[i].life * 0.5);
    //     EmitVertex();
        
    //     gl_Position = mvp * vec4( .5, .5, 0.0, 1.0);
    //     color = vec4(templ.color, p_l[i].life * 0.5);
    //     EmitVertex();

    //     gl_Position = mvp * vec4( .5, -.5, 0.0, 1.0);
    //     color = vec4(templ.color, p_l[i].life * 0.5);
    //     EmitVertex();
        
    //     EndPrimitive();
    // }
}  