#version 450

layout (location = 0) out int id;
#include "util.glsl"

void main()
{   
    id = gl_VertexIndex;
}