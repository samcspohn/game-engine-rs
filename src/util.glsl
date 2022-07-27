
mat4 translate(mat4 m, vec3 translation){
		mat4 t = {{1,0,0,translation.x},
		{0,1,0,translation.y,},
		{0,0,1,translation.z}
		,{0,0,0,1}};
		return m * transpose(t);
}
mat4 translate(vec3 translation){
		mat4 t = {{1,0,0,translation.x},
		{0,1,0,translation.y,},
		{0,0,1,translation.z}
		,{0,0,0,1}};
		return transpose(t);
}
mat4 scale(mat4 m, vec3 scale){
	mat4 s = {{scale.x,0,0,0}, {0,scale.y,0,0},{0,0,scale.z,0},{0,0,0,1}};
	return m * transpose(s);
}
mat4 scale(vec3 scale){
	mat4 s = {{scale.x,0,0,0}, {0,scale.y,0,0},{0,0,scale.z,0},{0,0,0,1}};
	return transpose(s);
}

mat4 rotate(mat4 m, vec4 q){
	mat4 r1 = {{q.w,q.z,-q.y,q.x},{-q.z,q.w,q.x,q.y},{q.y,-q.x,q.w,q.z},{-q.x,-q.y,-q.z,q.w}};
	mat4 r2 = {{q.w,q.z,-q.y,-q.x},{-q.z,q.w,q.x,-q.y},{q.y,-q.x,q.w,-q.z},{q.x,q.y,q.z,q.w}};
	mat4 r = r1 * r2;
	r[0][3] = r[1][3] = r[2][3] = r[3][0] = r[3][1] = r[3][2] = 0;
	r[3][3] = 1;
	return m * r;
}
mat4 rotate(vec4 q){
	mat4 r1 = {{q.w,q.z,-q.y,q.x},{-q.z,q.w,q.x,q.y},{q.y,-q.x,q.w,q.z},{-q.x,-q.y,-q.z,q.w}};
	mat4 r2 = {{q.w,q.z,-q.y,-q.x},{-q.z,q.w,q.x,-q.y},{q.y,-q.x,q.w,-q.z},{q.x,q.y,q.z,q.w}};
	mat4 r = r1 * r2;
	r[0][3] = r[1][3] = r[2][3] = r[3][0] = r[3][1] = r[3][2] = 0;
	r[3][3] = 1;
	return r;
}
mat4 identity(){
	mat4 i = {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
	return i;
}
vec4 lookAt(in vec3 lookAt, in vec3 up) {
	vec3 forward = lookAt;
	forward = normalize(forward);
	vec3 right = normalize(cross(up, forward));
	up = normalize(cross(forward,right));

	#define m00 right.x
	#define m01 up.x
	#define m02 forward.x
	#define m10 right.y
	#define m11 up.y
	#define m12 forward.y
	#define m20 right.z
	#define m21 up.z
	#define m22 forward.z

	vec4 ret;
	ret.w = sqrt(1.0f + m00 + m11 + m22) * 0.5f;
	float w4_recip = 1.0f / (4.0f * ret.w);
	ret.x = (m21 - m12) * w4_recip;
	ret.y = (m02 - m20) * w4_recip;
	ret.z = (m10 - m01) * w4_recip;

	#undef m00
	#undef m01
	#undef m02
	#undef m10
	#undef m11
	#undef m12
	#undef m20
	#undef m21
	#undef m22
	return ret;
}

mat4 rotationMatrix(vec3 axis, float angle)
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}

vec3 rotate(vec3 axis, float angle, vec3 vec){
	if(angle == 0){
		return vec;
	}
	return (rotationMatrix(axis, angle) * vec4(vec,1)).xyz;
}

const uint RIGHT = 0x0000ffff;
const uint LEFT = 0xffff0000;
uint getLeft(uint field) {
	return field >> 16; // >>> operator 0-fills from left
}
uint getRight(uint field) {
	return field & RIGHT;
}

void setHighBits(inout uint o, uint left){
	o = (left << 16) | (o & RIGHT);
}
void setLowBits(inout uint o, uint right){
	o = (o & LEFT) | (right);
}
uint getHighBits(inout uint o){
	return getLeft(o);
}
uint getLowBits(inout uint o){
	return getRight(o);
}

#define M_PI 3.1415926535897932384626433832795
float PHI = 1.61803398874989484820459;  // Φ = Golden Ratio   
float rand(in vec2 xy, in float seed){
    xy += vec2(1);
    return fract(tan(distance(xy*PHI, xy)*sin(seed))*xy.x);
}
struct rng{
	vec2 r;
	float s;

};
void setSeed(inout rng g, vec2 rin, float seedin){
	g.r = rin;
	g.s = seedin;
}

float gen(inout rng g){
	float ret = rand(g.r, g.s);
	g.r = g.r + vec2(1.21212121,2.12121212);
	g.s += 2.121212112f;
	return ret;
}
vec4 randVec4(inout rng g){
	return vec4(
    gen(g) - 0.5f,
    gen(g) - 0.5f,
    gen(g) - 0.5f,
    gen(g) - 0.5f
    );
}
vec3 randVec3(inout rng g){
	return vec3(
    gen(g) - 0.5f,
    gen(g) - 0.5f,
    gen(g) - 0.5f
    );
}

struct smquat{
	uvec2 d;
};

vec4 get(smquat q){
	return normalize(vec4((float(getHighBits(q.d.x)) - 32768) / 32768,
                (float(getLowBits(q.d.x)) - 32768) / 32768,
                (float(getHighBits(q.d.y)) - 32768) / 32768,
                (float(getLowBits(q.d.y)) - 32768) / 32768));
}

void set(inout smquat q,vec4 quat){
	quat = normalize(quat);
	q.d.x = (uint(quat.x * 32768 + 32768) << 16) | uint(quat.y * 32768 + 32768);
	q.d.y = (uint(quat.z * 32768 + 32768) << 16) | uint(quat.w * 32768 + 32768);
}

struct smvec3{
	uint xy;
	float z;
};

uint getAngle(vec3 a, vec3 b,vec2 quadrant){
    float angle = acos(dot(normalize(a),normalize(b)));
    if(quadrant.x > 0){
        angle = 6.28318530718 - angle;
    }
    return uint((angle / 6.28318530718) * 65536);
}

void set(inout smvec3 v, vec3 a){
	vec3 newVec = vec3(a.x,a.y,-a.z);
    uint xAxisAngle = getAngle(newVec,vec3(a.x,0,-a.z),vec2(a.y,a.z));
    
    uint yAxisAngle = getAngle(vec3(a.x,0, -a.z), vec3(0,0, 1),vec2(a.x,a.z));
	v.xy = (xAxisAngle << 16) | yAxisAngle;
	v.z = length(a);

	// v.xy = (floatBitsToUint(tan(a.x / a.z)) & LEFT) | (floatBitsToUint(tan(a.y / a.z)) >> 16);
	// v.z = a.z;
}

void rotateX(inout vec3 vec, float angle){
    float y = vec.y;
    float z = vec.z;
    vec.y = y * cos(angle) - z * sin(angle);
    vec.z = y * sin(angle) + z * cos(angle);
}

void rotateY(inout vec3 vec, float angle){
    float x = vec.x;
    float z = vec.z;
    vec.x = x * cos(angle) + z * sin(angle);
    vec.z = -x * sin(angle) + z * cos(angle);
}

float getAngle(uint a){
    return float(a) / 65536 * 6.28318530718;
}

vec3 get(smvec3 v){
	float anglex = getAngle(getHighBits(v.xy));
    float angley = getAngle(getLowBits(v.xy));
    vec3 p = vec3(0,0, -v.z);
    rotateX(p,-anglex);
    rotateY(p,angley);
	// vec3 p;
	// p.x = atan(uintBitsToFloat(v.xy & LEFT));
	// p.x *= (v.z > 0 && p.x > 0 ? -1 : 1) * v.z;
	// p.y = atan(uintBitsToFloat(v.xy << 16));
	// p.y *= (v.z > 0 && p.y > 0 ? -1 : 1) * v.z;
	// p.z = v.z;
    return p;
}