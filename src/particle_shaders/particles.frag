#version 450
#include "particle.glsl"
layout (location = 0) in float life;
layout (location = 1) in flat int templ_id;
layout (location = 0) out vec4 FragColor;

layout(set = 0, binding = 3) buffer pt { particle_template templates[]; };
void main()
{
    // particle_template templ = templates[templ_id];
    FragColor = templates[templ_id].color_life[int((life) * 199)];//color;//vec4(color.xyz,1.0);
}  