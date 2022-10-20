struct particle {
    vec3 vel;
    int emitter_id;
    int sorted;
};
struct particle_template {
    vec3 color;
    float speed;
    float emission_rate;
};
struct pos_lif {
    vec3 pos;
    float life;
    vec4 rot;
    int template_id;
};
struct _a {
    // vec4 rot;
    // vec3 pos;
    uint key;
    // int proto_id;
    // float life;
    int p_id;
};