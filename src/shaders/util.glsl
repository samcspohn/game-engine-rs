
// #define MAX_LIGHTS_PER_TILE 512
#define MAX_LEVEL 7

const int _light_quadtree_offsets[8] = {0, 1, 5, 21, 85, 341, 1365, 5461};
const int _light_quadtree_widths[7] = {1, 2, 4, 8, 16, 32, 64};
// const int half_widths_r[5] = {16, 8, 4, 2, 1};

// some credit to https://www.3dgep.com/forward-plus/#Forward Jeremiah van Oosten
struct Plane {
    vec3 N;    // Plane normal.
    float d;   // Distance to origin.
};
struct Frustum {
    vec4 planes[6];
    vec3 points[8];   // 0-3 near 4-7 far
};
// struct Sphere {
//     vec3 c;    // Center point.
//     float r;   // Radius.
// };
struct Cone {
    vec3 T;    // Cone tip.
    float h;   // Height of the cone.
    vec3 d;    // Direction of the cone.
    float r;   // bottom radius of the cone.
};
struct LineSegment {
    vec3 p1;
    vec3 p2;
};
struct transform {
    vec3 position;
    int padding;
    vec4 rotation;
    vec3 scale;
    int padding2;
};
struct attenuation {
    float constant;
    float linear;
    float exponential;
    float brightness;
};

struct lightTemplate {
    vec3 color;
    int p1;
    attenuation atten;
};
struct light {
    int templ;
    int t_id;
    int enabled;
    float radius;
    // 4
    vec3 pos;
    int p;
    // 8
};

struct light_init {
    int templ_id;
    int t_id;
    int id;
    int p1;
};
struct light_deinit {
    int id;
};
struct tile {
    // Frustum frustum;
    uint count;
    uint offset;
    int BLH_offset;
    uint travel_through;
    // vec2 p;
    // uint p2;
    // uint lights[MAX_LIGHTS_PER_TILE];
};
struct BoundingLine {
    uint flag;
    uint start;
    uint end;
    int front;   // positive value points to BoundingLine, negative points to light
    int back;
};

uint float_to_uint(float f) {
    // uint _f = floatBitsToUint(f);
    // uint mask = -int(_f >> 31) | 0x80000000;
    // return _f ^ mask;
    // Get the raw bits as uint
    uint u = floatBitsToUint(f);
    
    // If the float is negative, we need to flip all bits to maintain sorting
    // This works because IEEE-754 negative floats have reverse ordering
    if ((u & 0x80000000u) != 0u) {
        u = ~u;  // Flip all bits for negative numbers
    } else {
        u |= 0x80000000u;  // Set sign bit for positive numbers
    }
    
    return u;
}
float uint_to_float(uint u) {
     // Reverse the bit manipulation from floatToUint
    if ((u & 0x80000000u) != 0u) {
        u &= 0x7FFFFFFFu;  // Clear sign bit for positive numbers
    } else {
        u = ~u;  // Flip all bits back for negative numbers
    }
    
    return uintBitsToFloat(u);
    // uint mask = (-int(i) >> 31) | 0x80000000;
    // return uintBitsToFloat(i ^ mask);
}

// Converts a float to a uint while preserving sorting order
uint floatToUint(float f) {
    // Get the raw bits as uint
    uint u = floatBitsToUint(f);
    
    // If the float is negative, we need to flip all bits to maintain sorting
    // This works because IEEE-754 negative floats have reverse ordering
    if ((u & 0x80000000u) != 0u) {
        u = ~u;  // Flip all bits for negative numbers
    } else {
        u |= 0x80000000u;  // Set sign bit for positive numbers
    }
    
    return u;
}

// Converts the uint back to the original float
float uintToFloat(uint u) {
    // Reverse the bit manipulation from floatToUint
    if ((u & 0x80000000u) != 0u) {
        u &= 0x7FFFFFFFu;  // Clear sign bit for positive numbers
    } else {
        u = ~u;  // Flip all bits back for negative numbers
    }
    
    return uintBitsToFloat(u);
}

struct AABB {
    vec3 _min;
    vec3 _max;
};

struct MVP {
    mat4 mvp;
    mat4 mv;
    mat4 m;
    mat4 n;
};

// Structure to represent a sphere
struct Sphere {
    vec3 center;
    float radius;
};

// Structure to hold the transformed sphere in NDC space
struct NDCSphere {
    vec3 center;
    float radius;
};

// Structure to represent a rectangle in screen space
struct ScreenRect {
    vec2 min;  // Bottom-left in pixels
    vec2 max;  // Top-right in pixels
};


// Transform sphere from world space to NDC space using a single view-projection matrix
NDCSphere transformSphereToNDC(Sphere sphere, mat4 viewProjMatrix) {
    NDCSphere ndcSphere;
    
    // Transform sphere center to clip space
    vec4 clipCenter = viewProjMatrix * vec4(sphere.center, 1.0);
    
    // Perspective divide to get NDC coordinates
    ndcSphere.center = clipCenter.xyz / clipCenter.w;
    
    // Calculate projected radius
    // We'll use two points to determine the projected radius
    // First, the sphere center point
    vec4 clipCenterRadius = viewProjMatrix * vec4(sphere.center + vec3(sphere.radius, 0.0, 0.0), 1.0);
    
    // Perspective divide for the radius point
    vec3 ndcRadius = clipCenterRadius.xyz / clipCenterRadius.w;
    
    // Calculate radius in NDC space
    // Use the distance between center and radius point
    ndcSphere.radius = distance(ndcSphere.center.xy, ndcRadius.xy);
    
    // Optional: Adjust for perspective distortion
    // This can make the radius more accurate at different view distances
    float depthFactor = abs(clipCenter.w);  // Depth in view space
    ndcSphere.radius *= (clipCenter.w / depthFactor);
    
    return ndcSphere;
}

float halfSpaceTest(vec4 plane, vec3 p) { return dot(plane.xyz, p) - plane.w; }
bool frustumAABBIntersect(in Frustum f, AABB a) {
    vec3 mins, maxs;
    mins = a._min;
    maxs = a._max;
    vec3 axis_vert;
    vec4 planes[6] = f.planes;

    for (int i = 0; i < 6; ++i) {
        vec4 p = planes[i];
        axis_vert = vec3(p.x > 0.0 ? maxs.x : mins.x, p.y > 0.0 ? maxs.y : mins.y, p.z > 0.0 ? maxs.z : mins.z);
        if (dot(planes[i].xyz, axis_vert) - planes[i].w < 0) return false;
    }
    return true;
}

bool frustumSegmentIntersect(in Frustum f, LineSegment s) {

    for (int i = 0; i < 6; ++i) {
        if (dot(normalize(f.planes[i].xyz), s.p1) - f.planes[i].w > 0 || dot(normalize(f.planes[i].xyz), s.p2) - f.planes[i].w > 0) {
            continue;
        }
        return false;
    }
    return true;
}

int get_tile(int x, int y, int level) { return _light_quadtree_offsets[level] + (x + (y) *_light_quadtree_widths[level]); }

float DistanceToPlane(vec4 vPlane, vec3 vPoint) { return dot(vec4(vPoint, 1.0), vPlane); }

// Frustum cullling on a sphere. Returns > 0 if visible, <= 0 otherwise
bool CullSphere(vec4 vPlanes[6], vec3 vCenter, float fRadius) {
    float dist01 = min(DistanceToPlane(vPlanes[0], vCenter), DistanceToPlane(vPlanes[1], vCenter));
    float dist23 = min(DistanceToPlane(vPlanes[2], vCenter), DistanceToPlane(vPlanes[3], vCenter));
    float dist45 = min(DistanceToPlane(vPlanes[4], vCenter), DistanceToPlane(vPlanes[5], vCenter));

    return min(min(dist01, dist23), dist45) + fRadius < 0;
}

bool sphere_frustum(vec3 pos, float radius, in Frustum frustum) {
    // if(t.contains_origin == 1) {
    // 	return false;
    // }

    // AABB aabb = {light_bbox_min, light_bbox_max};
    // bool result = frustumAABBIntersect(frustum, aabb);
    // return result;
    // if (!frustumAABBIntersect(frustum, aabb)) {
    //     return false;
    // }
    bool result = true;

    // Step1: sphere-plane test
    for (int i = 0; i < 6; i++) {
        if (dot(frustum.planes[i].xyz, pos) - frustum.planes[i].w < -radius) {
            result = false;   // <--
            break;            //   |
        }   //   |
    }   //   |
        //   |
    if (!result) {   // if false ---
        return false;
    }

    // Step2: bbox corner test (to reduce false positive)
    vec3 light_bbox_max = pos + vec3(radius);
    vec3 light_bbox_min = pos - vec3(radius);
    int probe;
    probe = 0;
    for (int i = 0; i < 8; i++)
        probe += ((frustum.points[i].x > light_bbox_max.x) ? 1 : 0);
    if (probe == 8) return false;
    probe = 0;
    for (int i = 0; i < 8; i++)
        probe += ((frustum.points[i].x < light_bbox_min.x) ? 1 : 0);
    if (probe == 8) return false;
    probe = 0;
    for (int i = 0; i < 8; i++)
        probe += ((frustum.points[i].y > light_bbox_max.y) ? 1 : 0);
    if (probe == 8) return false;
    probe = 0;
    for (int i = 0; i < 8; i++)
        probe += ((frustum.points[i].y < light_bbox_min.y) ? 1 : 0);
    if (probe == 8) return false;
    probe = 0;
    for (int i = 0; i < 8; i++)
        probe += ((frustum.points[i].z > light_bbox_max.z) ? 1 : 0);
    if (probe == 8) return false;
    probe = 0;
    for (int i = 0; i < 8; i++)
        probe += ((frustum.points[i].z < light_bbox_min.z) ? 1 : 0);
    if (probe == 8) return false;

    return true;
}
// Compute a plane from 3 noncollinear points that form a triangle.
// This equation assumes a right-handed (counter-clockwise winding order)
// coordinate system to determine the direction of the plane normal.
// Plane ComputePlane( vec3 p0, vec3 p1, vec3 p2 )
// {
//     Plane plane;

//     vec3 v0 = p1 - p0;
//     vec3 v2 = p2 - p0;

//     plane.N = normalize( cross( v0, v2 ) );

//     // Compute the distance to the origin using p0.
//     plane.d = dot( plane.N, p0 );

//     return plane;
// }
// bool SphereInsidePlane( Sphere sphere, Plane plane )
// {
//     return dot( plane.N, sphere.c ) - plane.d < -sphere.r;
// }

// // Check to see of a light is partially contained within the frustum.
// bool SphereInsideFrustum( Sphere sphere, Frustum frustum, float zNear, float zFar )
// {
//     bool result = true;

//     // First check depth
//     // Note: Here, the view vector points in the -Z axis so the
//     // far depth value will be approaching -infinity.
//     if ( sphere.c.z - sphere.r > zNear || sphere.c.z + sphere.r < zFar )
//     {
//         result = false;
//     }

//     // Then check frustum planes
//     for ( int i = 0; i < 4 && result; i++ )
//     {
//         if ( SphereInsidePlane( sphere, frustum.planes[i] ) )
//         {
//             result = false;
//         }
//     }

//     return result;
// }

// // Check to see if a point is fully behind (inside the negative halfspace of) a plane.
// bool PointInsidePlane( vec3 p, Plane plane )
// {
//     return dot( plane.N, p ) - plane.d < 0;
// }
// // Check to see if a cone if fully behind (inside the negative halfspace of) a plane.
// // Source: Real-time collision detection, Christer Ericson (2005)
// bool ConeInsidePlane( Cone cone, Plane plane )
// {
//     // Compute the farthest point on the end of the cone to the positive space of the plane.
//     vec3 m = cross( cross( plane.N, cone.d ), cone.d );
//     vec3 Q = cone.T + cone.d * cone.h - m * cone.r;

//     // The cone is in the negative halfspace of the plane if both
//     // the tip of the cone and the farthest point on the end of the cone to the
//     // positive halfspace of the plane are both inside the negative halfspace
//     // of the plane.
//     return PointInsidePlane( cone.T, plane ) && PointInsidePlane( Q, plane );
// }
// bool ConeInsideFrustum( Cone cone, Frustum frustum, float zNear, float zFar )
// {
//     bool result = true;

//     Plane nearPlane = { vec3( 0, 0, -1 ), -zNear };
//     Plane farPlane = { vec3( 0, 0, 1 ), zFar };

//     // First check the near and far clipping planes.
//     if ( ConeInsidePlane( cone, nearPlane ) || ConeInsidePlane( cone, farPlane ) )
//     {
//         result = false;
//     }

//     // Then check frustum planes
//     for ( int i = 0; i < 4 && result; i++ )
//     {
//         if ( ConeInsidePlane( cone, frustum.planes[i] ) )
//         {
//             result = false;
//         }
//     }

//     return result;
// }

float modify(float x, float m) { return (x * 0.5 + 2) * m; }
vec3 get_tile_idx(vec4 v) {
    // vec3 _v = vec3(1, 1, 0);
    // if (v.x > 0.5) {
    // 	_v.x = 0.;
    // }
    // if (v.y > 0.5) {
    // 	_v.y = 0.;
    // }
    // vec3 _v = vec3(0, 0, (v.z + 1) * -24);
    // vec3 _v = vec3(modify(v.x,16 / 5), modify(v.y, 9 / 5), z * 24);
    // vec3 _v = vec3(v.x * 16, v.y * 9, 0);
    // vec3 _v = vec3(mod(v.x,16), mod(v.y,9), 0);
    // vec3 _v = vec3(v.xy / v.w, v.z / 2000);
    vec3 _v = v.xyz / v.w;
    _v = vec3((_v.x + 1) * 16, (_v.y + 1) * 9, 0);
    return _v;
}
uint hash_pos(vec3 p) {
    const float bucket_size = 32.0;
    ivec3 pos_hash = ivec3(p / bucket_size);
    // uvec3 hash_vec =
    // uvec3(floatBitsToUint(pos_hash.x),floatBitsToUint(pos_hash.y),floatBitsToUint(pos_hash.z));
    return uint(pos_hash.x * (pos_hash.y * 32) * (pos_hash.z * 32 * 32)) % 65536;
    // return (pos_hash.x * pos_hash.y * pos_hash.z) % 0x00ff;
}
struct DispatchIndirectCommand {
    uint x;
    uint y;
    uint z;
};
struct VkDrawIndirectCommand {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

mat4 translate(mat4 m, vec3 translation) {
    mat4 t = {
        {1, 0, 0, translation.x},
        {0, 1, 0, translation.y},
        {0, 0, 1, translation.z},
        {0, 0, 0, 1            }
    };
    return m * transpose(t);
}
mat4 translate(vec3 translation) {
    mat4 t = {
        {1, 0, 0, translation.x},
        {0, 1, 0, translation.y},
        {0, 0, 1, translation.z},
        {0, 0, 0, 1            }
    };
    return transpose(t);
}
mat4 scale(mat4 m, vec3 scale) {
    mat4 s = {
        {scale.x, 0,       0,       0},
        {0,       scale.y, 0,       0},
        {0,       0,       scale.z, 0},
        {0,       0,       0,       1}
    };
    return m * transpose(s);
}
mat4 scale(vec3 scale) {
    mat4 s = {
        {scale.x, 0,       0,       0},
        {0,       scale.y, 0,       0},
        {0,       0,       scale.z, 0},
        {0,       0,       0,       1}
    };
    return transpose(s);
}

mat4 rotate(mat4 m, vec4 q) {
    mat4 r1 = {
        {q.w,  q.z,  -q.y, q.x},
        {-q.z, q.w,  q.x,  q.y},
        {q.y,  -q.x, q.w,  q.z},
        {-q.x, -q.y, -q.z, q.w}
    };
    mat4 r2 = {
        {q.w,  q.z,  -q.y, -q.x},
        {-q.z, q.w,  q.x,  -q.y},
        {q.y,  -q.x, q.w,  -q.z},
        {q.x,  q.y,  q.z,  q.w }
    };
    mat4 r = r1 * r2;
    r[0][3] = r[1][3] = r[2][3] = r[3][0] = r[3][1] = r[3][2] = 0;
    r[3][3] = 1;
    return m * r;
}
mat4 rotate(vec4 q) {
    // mat4 r1 = {{q.w,q.z,-q.y,q.x},{-q.z,q.w,q.x,q.y},{q.y,-q.x,q.w,q.z},{-q.x,-q.y,-q.z,q.w}};
    // mat4 r2 = {{q.w,q.z,-q.y,-q.x},{-q.z,q.w,q.x,-q.y},{q.y,-q.x,q.w,-q.z},{q.x,q.y,q.z,q.w}};
    // mat4 r = r1 * r2;
    // r[0][3] = r[1][3] = r[2][3] = r[3][0] = r[3][1] = r[3][2] = 0;
    // r[3][3] = 1;
    // return r;
    float x = q.x;
    float y = q.y;
    float z = q.z;
    float s = q.w;
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;

    mat4 r = {
        {1.f - 2.f * y2 - 2.f * z2, 2.f * x * y - 2.f * s * z, 2.f * x * z + 2.f * s * y, 0.f},
        {2.f * x * y + 2.f * s * z, 1.f - 2.f * x2 - 2.f * z2, 2.f * y * z - 2.f * s * x, 0.f},
        {2.f * x * z - 2.f * s * y, 2.f * y * z + 2.f * s * x, 1.f - 2.f * x2 - 2.f * y2, 0.f},
        {0.f,                       0.f,                       0.f,                       1.f}
    };
    return transpose(r);
}
mat3 rotate3(vec4 q) {
    float qx2 = q.x * q.x;
    float qy2 = q.y * q.y;
    float qz2 = q.z * q.z;
    mat3 r = {
        {1 - 2 * qy2 - 2 * qz2,         2 * q.x * q.y - 2 * q.z * q.w, 2 * q.x * q.z + 2 * q.y * q.w},
        {2 * q.x * q.y + 2 * q.z * q.w, 1 - 2 * qx2 - 2 * qz2,         2 * q.y * q.z - 2 * q.x * q.w},
        {2 * q.x * q.z - 2 * q.y * q.w, 2 * q.y * q.z + 2 * q.x * q.w, 1 - 2 * qx2 - 2 * qy2        }
    };
    // r = transpose(r);
    return transpose(r);
}
mat4 identity() {
    mat4 i = {
        {1, 0, 0, 0},
        {0, 1, 0, 0},
        {0, 0, 1, 0},
        {0, 0, 0, 1}
    };
    return i;
}

vec4 angleAxis(float rad, vec3 axis) {
    vec4 q;
    q.x = axis.x * sin(rad / 2);
    q.y = axis.y * sin(rad / 2);
    q.z = axis.z * sin(rad / 2);
    q.w = cos(rad / 2);
    return q;
}

vec4 RotationBetweenVectors(vec3 start, vec3 dest) {
    start = normalize(start);
    dest = normalize(dest);

    float cosTheta = dot(start, dest);
    vec3 rotationAxis;

    if (cosTheta < -1 + 0.001f) {
        // special case when vectors in opposite directions:
        // there is no "ideal" rotation axis
        // So guess one; any will do as long as it's perpendicular to start
        rotationAxis = cross(vec3(0.0f, 0.0f, 1.0f), start);
        if ((rotationAxis.x * rotationAxis.x + rotationAxis.y * rotationAxis.y + rotationAxis.z * rotationAxis.z) <
            0.01)   // bad luck, they were parallel, try again!
            rotationAxis = cross(vec3(1.0f, 0.0f, 0.0f), start);

        rotationAxis = normalize(rotationAxis);
        return angleAxis(radians(180.0f), rotationAxis);
    }

    rotationAxis = cross(start, dest);

    float s = sqrt((1 + cosTheta) * 2);
    float invs = 1 / s;

    return vec4(s * 0.5f, rotationAxis.x * invs, rotationAxis.y * invs, rotationAxis.z * invs);
}

vec4 mat3_quat(mat3 m) {
#define m00 m[0][0]
#define m01 m[0][1]
#define m02 m[0][2]
#define m10 m[1][0]
#define m11 m[1][1]
#define m12 m[1][2]
#define m20 m[2][0]
#define m21 m[2][1]
#define m22 m[2][2]

    float tr = m00 + m11 + m22;
    vec4 q;
    if (tr > 0) {
        float S = sqrt(tr + 1.0) * 2;   // S=4*qw
        q.w = 0.25 * S;
        q.x = (m21 - m12) / S;
        q.y = (m02 - m20) / S;
        q.z = (m10 - m01) / S;
    } else if (bool(uint(m00 > m11) & uint(m00 > m22))) {
        float S = sqrt(1.0 + m00 - m11 - m22) * 2;   // S=4*qx
        q.w = (m21 - m12) / S;
        q.x = 0.25 * S;
        q.y = (m01 + m10) / S;
        q.z = (m02 + m20) / S;
    } else if (m11 > m22) {
        float S = sqrt(1.0 + m11 - m00 - m22) * 2;   // S=4*qy
        q.w = (m02 - m20) / S;
        q.x = (m01 + m10) / S;
        q.y = 0.25 * S;
        q.z = (m12 + m21) / S;
    } else {
        float S = sqrt(1.0 + m22 - m00 - m11) * 2;   // S=4*qz
        q.w = (m10 - m01) / S;
        q.x = (m02 + m20) / S;
        q.y = (m12 + m21) / S;
        q.z = 0.25 * S;
    }
    return q;
#undef m00
#undef m01
#undef m02
#undef m10
#undef m11
#undef m12
#undef m20
#undef m21
#undef m22
}
vec4 lookAt(in vec3 lookAt, in vec3 up) {

    // #define m00 right.x
    // #define m01 up.x
    // #define m02 forward.x
    // #define m10 right.y
    // #define m11 up.y
    // #define m12 forward.y
    // #define m20 right.z
    // #define m21 up.z
    // #define m22 forward.z

    vec3 forward = lookAt;
    forward = normalize(forward);
    vec3 right = normalize(cross(up, forward));
    up = normalize(cross(forward, right));

    mat3 m;
    m[0] = right;
    m[1] = up;
    m[2] = forward;
    m = transpose(m);
    // mat3 m = {{right.x,up.x,forward.x},{right.y,up.y,forward.y},{right.z,up.z,forward.z}};
    // transpose(m);
    return mat3_quat(m);
}

mat4 rotationMatrix(vec3 axis, float angle) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;

    mat4 r = {
        {oc * axis.x * axis.x + c,          oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s, 0.0},
        {oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c,          oc * axis.y * axis.z - axis.x * s, 0.0},
        {oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c,          0.0},
        {0.0,                               0.0,                               0.0,                               1.0},
    };
    return r;
}

vec3 rotate(vec3 axis, float angle, vec3 vec) {
    if (angle == 0) {
        return vec;
    }
    return (rotationMatrix(axis, angle) * vec4(vec, 1)).xyz;
}

const uint RIGHT = 0x0000ffff;
const uint LEFT = 0xffff0000;
uint getLeft(uint field) {
    return field >> 16;   // >>> operator 0-fills from left
}
uint getRight(uint field) { return field & RIGHT; }

void setHighBits(inout uint o, uint left) { o = (left << 16) | (o & RIGHT); }
void setLowBits(inout uint o, uint right) { o = (o & LEFT) | (right); }
uint getHighBits(inout uint o) { return getLeft(o); }
uint getLowBits(inout uint o) { return getRight(o); }

#define M_PI 3.1415926535897932384626433832795
float PHI = 1.61803398874989484820459;   // Φ = Golden Ratio
float rand(in vec2 xy, in float seed) {
    xy += vec2(1);
    return fract(tan(distance(xy * PHI, xy) * sin(seed)) * xy.x);
}
struct rng {
    vec2 r;
    float s;
};
void setSeed(inout rng g, vec2 rin, float seedin) {
    g.r = rin;
    g.s = seedin;
}

float gen(inout rng g) {
    float ret = rand(g.r, g.s);
    g.r = g.r + vec2(1.21212121, 2.12121212);
    g.s += 2.121212112f;
    return ret;
}
vec4 randVec4(inout rng g) { return vec4(gen(g) - 0.5f, gen(g) - 0.5f, gen(g) - 0.5f, gen(g) - 0.5f); }
vec3 randVec3(inout rng g) { return vec3(gen(g) - 0.5f, gen(g) - 0.5f, gen(g) - 0.5f); }

struct smquat {
    uvec2 d;
};

vec4 get(smquat q) {
    return normalize(vec4((float(getHighBits(q.d.x)) - 32768) / 32768, (float(getLowBits(q.d.x)) - 32768) / 32768, (float(getHighBits(q.d.y)) - 32768) / 32768,
                          (float(getLowBits(q.d.y)) - 32768) / 32768));
}

void set(inout smquat q, vec4 quat) {
    quat = normalize(quat);
    q.d.x = (uint(quat.x * 32768 + 32768) << 16) | uint(quat.y * 32768 + 32768);
    q.d.y = (uint(quat.z * 32768 + 32768) << 16) | uint(quat.w * 32768 + 32768);
}

struct smvec3 {
    uint xy;
    float z;
};

uint getAngle(vec3 a, vec3 b, vec2 quadrant) {
    float angle = acos(dot(normalize(a), normalize(b)));
    if (quadrant.x > 0) {
        angle = 6.28318530718 - angle;
    }
    return uint((angle / 6.28318530718) * 65536);
}

void set(inout smvec3 v, vec3 a) {
    vec3 newVec = vec3(a.x, a.y, -a.z);
    uint xAxisAngle = getAngle(newVec, vec3(a.x, 0, -a.z), vec2(a.y, a.z));

    uint yAxisAngle = getAngle(vec3(a.x, 0, -a.z), vec3(0, 0, 1), vec2(a.x, a.z));
    v.xy = (xAxisAngle << 16) | yAxisAngle;
    v.z = length(a);

    // v.xy = (floatBitsToUint(tan(a.x / a.z)) & LEFT) | (floatBitsToUint(tan(a.y / a.z)) >> 16);
    // v.z = a.z;
}

void rotateX(inout vec3 vec, float angle) {
    float y = vec.y;
    float z = vec.z;
    vec.y = y * cos(angle) - z * sin(angle);
    vec.z = y * sin(angle) + z * cos(angle);
}

void rotateY(inout vec3 vec, float angle) {
    float x = vec.x;
    float z = vec.z;
    vec.x = x * cos(angle) + z * sin(angle);
    vec.z = -x * sin(angle) + z * cos(angle);
}

float getAngle(uint a) { return float(a) / 65536 * 6.28318530718; }

vec3 get(smvec3 v) {
    float anglex = getAngle(getHighBits(v.xy));
    float angley = getAngle(getLowBits(v.xy));
    vec3 p = vec3(0, 0, -v.z);
    rotateX(p, -anglex);
    rotateY(p, angley);
    // vec3 p;
    // p.x = atan(uintBitsToFloat(v.xy & LEFT));
    // p.x *= (v.z > 0 && p.x > 0 ? -1 : 1) * v.z;
    // p.y = atan(uintBitsToFloat(v.xy << 16));
    // p.y *= (v.z > 0 && p.y > 0 ? -1 : 1) * v.z;
    // p.z = v.z;
    return p;
}