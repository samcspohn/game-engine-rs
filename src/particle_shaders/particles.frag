#version 450
layout (location = 0) in vec4 color;
layout (location = 0) out vec4 FragColor;

void main()
{
    FragColor = vec4(color.xyz,1.0);
}  