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
    int p_id;
};