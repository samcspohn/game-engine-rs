#version 450
// layout (location = 0) in vec2 aPos;
layout (location = 0) out int id;


void main()
{   
    // mat4 v = view;
    id = gl_VertexIndex;
    // vec3 pos = position[gl_VertexIndex];
    // gl_Position = proj * view * vec4(pos, 1.0);
    // live = int(life[gl_VertexIndex] > 0.0);
}