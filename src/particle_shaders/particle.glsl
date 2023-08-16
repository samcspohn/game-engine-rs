struct emitter {
    vec3 prev_pos;
    vec4 prev_rot;
    int alive;
    int transform_id;
    float emission;
    int template_id;
    int last;
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
    vec4 rot;
    float l;
};
struct particle_template {
    vec4 color;

    float emission_rate;
    float emission_radius;
    float min_speed;
    float max_speed;

    float min_lifetime;
    float max_lifetime;
    vec2 scale;
    
    int trail;
    int billboard;
    int align_vel;
    float dispersion;
    uint tex_id;
    // vec4 color_life[256];

};
struct pos_lif {
    vec3 pos; // 12
    float life; // 4
    // int template_id; // 4
    //_dummy0 // 
};
struct _a {
    // vec4 rot;
    // vec3 pos;
    uint key;
    // int proto_id;
    // float life;
    uint p_id;
};
