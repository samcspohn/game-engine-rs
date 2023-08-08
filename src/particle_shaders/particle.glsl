struct emitter {
    // vec3 pos;
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
    vec4 rot;
    uint count;
};
struct particle {
    vec3 vel;
    int emitter_id;
    vec4 rot;
    int sorted;
};
struct particle_template {
    vec4 color;
    float speed;
    float emission_rate;
    float life_time;
    float size;
    vec4 color_life[200];
    int trail;
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
