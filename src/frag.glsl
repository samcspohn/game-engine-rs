#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec2 coords;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform sampler2D tex;

const vec3 LIGHT = vec3(0.0, 1.0, -1.0);

void main() {
    float brightness = 2 * dot(normalize(v_normal), normalize(LIGHT));
    // vec3 dark_color = vec3(0.6, 0.0, 0.0);
    // vec3 regular_color = vec3(1.0, 0.0, 0.0);

    f_color = texture(tex, coords);// * brightness;//min(brightness + 0.8, 1.0);
}
