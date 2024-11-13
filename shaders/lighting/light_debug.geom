#version 450
#include "../util.glsl"

layout(points) in;
layout(triangle_strip, max_vertices = 8) out;

layout(location = 0) in int[] id;
layout(location = 0) out vec4 color;

layout(set = 0, binding = 0) buffer t { tile tiles[]; };
layout (push_constant, std430) uniform p { vec2 screen_dims; };

const vec4 colors[2] = {vec4(0, 1, 1, 0.05), vec4(1, 0, 0, 0.4)};
const vec4 colors2[2] = {vec4(0, 1, 1, 0.05), vec4(0.7, 0, 0, 0.6)};
void main() {
    int id = id[0];
    // if (id >= _light_quadtree_offsets[MAX_LEVEL]) return;
    int _id = id;
    ivec3 tile_idx;
    for (int i = MAX_LEVEL - 1; i >= 0; --i) {
        if (id >= _light_quadtree_offsets[i]) {
            id -= _light_quadtree_offsets[i];
            tile_idx = ivec3(id % _light_quadtree_widths[i], id / _light_quadtree_widths[i], i);
            break;
        }
    }
    vec2 size = vec2(1.0 / _light_quadtree_widths[tile_idx.z]);
    size.x -= 0.001 * abs(screen_dims.x) * 0.001; // / screen_dims.y * screen_dims.y / screen_dims.x;
    size.y -= 0.001 * abs(screen_dims.y) * 0.001;
    // size.x -= 0.0026;// * tile_idx.z;
    // size.y -= 0.004 ;// * tile_idx.z;
    // tile_idx.y = _light_quadtree_widths[tile_idx.z] - tile_idx.y - 1;
    // tile_idx.xy = tile_idx.yx;
    vec2 pos = 2 * (tile_idx.xy + 0.5) / _light_quadtree_widths[tile_idx.z] - 1;
    uint count = tiles[_id].count;
    // color = vec4(min(1, count / 512), min(1, max(0, (512 - count) / 512)), min(1, max(0, (512 - count) / 512)), 0.2);
    vec4 Color = count == 0 ? vec4(0, 0, 0, 0) : mix(colors[0], colors[1], min(1, float(count) / 512.0));

    color = Color;
    gl_Position = vec4(pos + size * ivec2(-1, -1), 0, 1);
    EmitVertex();
    color = Color;
    gl_Position = vec4(pos + size * ivec2(1, -1), 0, 1);
    EmitVertex();
    color = Color;
    gl_Position = vec4(pos + size * ivec2(-1, 1), 0, 1);
    EmitVertex();
    color = Color;
    gl_Position = vec4(pos + size * ivec2(1, 1), 0, 1);
    EmitVertex();
    EndPrimitive();

    size = vec2(0.01);
    count = tiles[_id].travel_through;
    Color = count == 0 ? vec4(0, 0, 0, 0) : mix(colors2[0], colors2[1], min(1, float(count) / 5000.0));

    color = Color;
    gl_Position = vec4(pos + size * ivec2(-1, -1), 0, 1);
    EmitVertex();
    color = Color;
    gl_Position = vec4(pos + size * ivec2(1, -1), 0, 1);
    EmitVertex();
    color = Color;
    gl_Position = vec4(pos + size * ivec2(-1, 1), 0, 1);
    EmitVertex();
    color = Color;
    gl_Position = vec4(pos + size * ivec2(1, 1), 0, 1);
    EmitVertex();
}
