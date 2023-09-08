#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#include "particle.glsl"
layout (location = 0) in float life;
layout (location = 1) in flat int templ_id;
layout (location = 2) in vec2 uv;
layout (location = 3) in vec2 uv2;
layout (location = 0) out vec4 FragColor;

layout(set = 0, binding = 3) buffer pt { particle_template templates[]; };
layout(set = 0, binding = 10) uniform sampler2D color_over_life;
layout(set = 0, binding = 11) uniform sampler2D[] s;
void main()
{
    #define templ templates[templ_id]
    // particle_template templ = templates[templ_id];
    vec4 col = texture(nonuniformEXT(s[templ.tex_id]), uv);
    vec4 col2 = texture(color_over_life, uv2);
    // vec4 col3 = templates[templ_id].color_life[int(life * 255)];
    FragColor = col * col2;
    // FragColor = vec4(1,0,0,1);
}  