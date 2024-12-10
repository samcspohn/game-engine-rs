
struct emitter {
    vec3 prev_pos;
    int alive;

    vec4 prev_rot;

    int transform_id;
    float emission;
    int template_id;
    int last;
    int num_particles;
    uint gen;
};
struct emitter_init {
    int transform_id;
    int alive;
    int template_id;
    int e_id;
};
struct burst {
    vec3 pos;
    int template_id;
    vec3 dir;
    uint count;
};
struct particle {
    vec3 vel;
    int emitter_id;
    uvec2 rot;
    float l;
    uint gen;

};
    void set_rot(inout particle p, vec4 r) {
        p.rot = uvec2(packHalf2x16(r.xy), packHalf2x16(r.zw));
    }
    vec4 get_rot(inout particle p) {
        return vec4(unpackHalf2x16(p.rot.x), unpackHalf2x16(p.rot.y));
    }
struct particle_template {
    vec4 color;

    float emission_rate;
    float emission_radius;
    float min_speed;
    float max_speed;

    vec2 scale;
    float min_lifetime;
    float max_lifetime;

    int trail;
    int billboard;
    int align_vel;
    float dispersion;

    uint tex_id;
    uint recieve_lighting;
    uint padding2;
    uint padding3;
    float size_over_lifetime[256];
    // vec4 color_life[256];
};
struct pos_lif {
    vec3 pos;     // 12
    float life;   // 4
    // int template_id; // 4
    //_dummy0 //
};
struct _a {
    uint key;
    uint p_id;
};
const uint hi = 0xffff0000;   // use to keep hi/lo bits
const uint lo = 0x0000ffff;

// ----------
//  l  |  x
// ----------
//  hi -- lo
// ----------
//  y  |  z
// ----------
struct pos_life_comp {
    uint lx;
    uint yz;
};
float get_life(inout pos_life_comp pl) {
    vec2 lx = unpackHalf2x16(pl.lx);
    return lx.x;
    // uint l = pl.lx >> 16;
    // return l * 65536;
}
void set_life(inout pos_life_comp pl, float _l) {
    vec2 _lx = unpackHalf2x16(pl.lx);
    uint lx = packHalf2x16(vec2(_l,_lx.y));
    pl.lx = lx;
    // uint l = uint(_l * 65536) << 16;
    // pl.lx = (pl.lx & lo) & l;
}
vec3 get_pos(inout pos_life_comp pl) {
    vec2 lx = unpackHalf2x16(pl.lx);
    vec2 yz = unpackHalf2x16(pl.yz);
    return vec3(lx.y, yz.x, yz.y);
    // uint x = pl.lx & lo << 16;
    // uint y = pl.yz & hi;
    // uint z = pl.yz & lo << 16;
    // return vec3(uintBitsToFloat(x),uintBitsToFloat(y),uintBitsToFloat(z));
}
void set_pos(inout pos_life_comp pl, vec3 v) {
    // uint lx = (packHalf2x16(vec2(0, v.x)) & lo) & (pl.lx & hi);
    vec2 _lx = unpackHalf2x16(pl.lx);
    uint lx = packHalf2x16(vec2(_lx.x, v.x));
    uint yz = packHalf2x16(v.yz);
    pl.lx = lx;
    pl.yz = yz;
    // uint x = floatBitsToUint(v.x) >> 16;
    // pl.lx = pl.lx & hi & x;
    // uint y = floatBitsToUint(v.y);
    // uint z = floatBitsToUint(v.z) >> 16;
    // pl.yz = y & hi & z;
}