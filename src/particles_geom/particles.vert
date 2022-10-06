#version 450
// layout (location = 0) in vec2 aPos;

layout(set = 0, binding = 0) buffer _p {
    vec3 position[];
};
layout(set = 0, binding = 1) uniform Data {
    mat4 view;
    mat4 proj;
};

void main()
{   
    // mat4 v = view;
    vec3 pos = position[gl_VertexIndex];
    gl_Position = proj * view * vec4(pos, 1.0); 
}