#version 450
#extension GL_EXT_nonuniform_qualifier : enable
#include "../../../shaders/util.glsl"
#include "particle.glsl"
layout(location = 0) in vec4 _color;
layout(location = 0) out vec4 FragColor;

void main() {
    FragColor = _color;
}