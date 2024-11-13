#version 450

layout (location = 0) out int id;
void main()
{   
    id = gl_VertexIndex;
}