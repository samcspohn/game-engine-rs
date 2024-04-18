#version 450
#include "util.glsl"
// #include "drawable.glsl"
struct DrawFrustum {
    Frustum f;
    vec4 color;
};

layout(points) in;
layout(triangle_strip, max_vertices = 48) out;

layout(location = 0) in int[] id;
layout(location = 0) out vec4 _color;


layout(set = 0, binding = 0) buffer d { DrawFrustum frustums[]; };
layout(set = 0, binding = 1) uniform Data {
    mat4 view;
    // mat4 cam_inv_rot;
    mat4 proj;
    // vec3 cam_pos;
    // vec4 cam_rot;
    // uint num_templates;
};
// vec4 look_at = lookAt(rotate3(cam_rot) * vec3(0, 0, 1), rotate3(cam_rot) * vec3(0, 1, 0));

const vec4 vert_pos[4] = {vec4(-1., 1., 0.0, 1.0), vec4(-1., -1., 0.0, 1.0), vec4(1, 1., 0.0, 1.0), vec4(1., -1., 0.0, 1.0)};
const vec2 vert_uv[4] = {vec2(0., 1.), vec2(1, 1.), vec2(0., 0.), vec2(1., 0.)};
vec4 get_position(in mat4 vp, int vert_id) {
    vec4 vert = vert_pos[vert_id];
    return vp * vert;   // vec4(-.5, .5, 0.0, 1.0);
}

void draw_line(in mat4 vp, vec3 a, vec3 b, vec4 color, float _width) {
    vec4 _a = vp * vec4(a,1);
    vec4 _b = vp * vec4(b,1);
    vec4 z = vec4(0,0,1,1);
    vec3 width_dir = normalize(cross(normalize(_a.xyz - _b.xyz), z.xyz));
    float desired_width = _width * (_a.w + _b.w) / 2;
    vec4 width = vec4(width_dir,0) * (desired_width + _width) / 2;
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

    mat4 vp = proj * view;

    vec3 corners[8] = frustums[_i].f.points;
    vec4 color = frustums[_i].color;
    draw_line(vp, corners[0], corners[1], color, 0.002);   // left side
    draw_line(vp, corners[1], corners[2], color, 0.002);
    draw_line(vp, corners[2], corners[3], color, 0.002);
    draw_line(vp, corners[3], corners[0], color, 0.002);

    draw_line(vp, corners[4], corners[5], color, 0.002);   // right side
    draw_line(vp, corners[5], corners[6], color, 0.002);
    draw_line(vp, corners[6], corners[7], color, 0.002);
    draw_line(vp, corners[7], corners[4], color, 0.002);

    draw_line(vp, corners[0], corners[4], color, 0.002);   // left to right
    draw_line(vp, corners[1], corners[5], color, 0.002);
    draw_line(vp, corners[2], corners[6], color, 0.002);
    draw_line(vp, corners[3], corners[7], color, 0.002);
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