#pragma once

#include "../hitable/hitable.h"

class Triangle : public Hitable {
public:
    __device__ Triangle() : EPSILON(0.00001) {}
    __device__ Triangle(vec3 vs[3], Material* mat, bool flip, Transform* t, const bool cull = false) :
        Hitable(t), flipNormal(flip) ,EPSILON(0.000001) {
        for (int i = 0; i < 3; i++) {
            vertices[i] = vs[i];
        }
        material = mat;
        vec3 edge1, edge2;
        edge1 = vertices[1] - vertices[0];
        edge2 = vertices[2] - vertices[0];
        normal = unit_vector(cross(edge1, edge2));
        backCulling = cull;
    };

    __device__ Triangle(vec3 vs[3], vec3 triNormal,  Material* mat, bool flip, Transform* t, const bool cull = false) :
        Hitable(t), flipNormal(flip), EPSILON(0.000001) {
        for (int i = 0; i < 3; i++) {
            vertices[i] = vs[i];
        }
        normal = triNormal;
        material = mat;
        backCulling = cull;
    };

    __device__ virtual bool collision_detection(const Ray& r,
        float t_min,
        float t_max,
        HitRecord& rec, int frameIndex) const;

    __device__ virtual bool bounding_box(float t0,
        float t1,
        AABB& box) const;

    __device__ void SetVertices(vec3 vs[3]) {
        for (int vi = 0; vi < 3; vi++) 
        {
            vertices[vi] = vs[vi];
        }
    }

    const float EPSILON;

    vec3 vertices[3];
    vec3 normal;
    bool flipNormal;
    bool backCulling;
    Material* material;
};


__device__ bool Triangle::collision_detection(const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec, int frameIndex) const {
    if (dot(r.direction(), normal) < 0)return false;
    vec3 vertex0 = vertices[0];
    vec3 vertex1 = vertices[1];
    vec3 vertex2 = vertices[2];

    vec3 edge1, edge2, h, s, q;
    float a, f, u, v;

    edge1 = vertex1 - vertex0;
    edge2 = vertex2 - vertex0;
    h = cross(r.direction(), edge2);
    a = dot(edge1, h);

    if (a < EPSILON && backCulling)
        return false;

    if (a > -EPSILON && a < EPSILON)
        return false;

    f = 1.0 / a;
    s = r.origin() - vertex0;
    u = f * dot(s, h);

    if (u < 0.0 || u > 1.0)
        return false;

    q = cross(s, edge1);
    v = f * dot(r.direction(), q);
    if (v < 0.0 || u + v > 1.0)
        return false;

    float t = f * dot(edge2, q);

    rec.t = t;
    rec.p = r.point_at_t(rec.t);
    rec.normal = normal;
    rec.mat_ptr = material;

    return true;
}

__device__ bool Triangle::bounding_box(float t0,
    float t1,
    AABB& bbox) const {
    float minX = min(vertices[0][0], min(vertices[1][0], vertices[2][0]));
    float minY = min(vertices[0][1], min(vertices[1][1], vertices[2][1]));
    float minZ = min(vertices[0][2], min(vertices[1][2], vertices[2][2]));

    float maxX = max(vertices[0][0], max(vertices[1][0], vertices[2][0]));
    float maxY = max(vertices[0][1], max(vertices[1][1], vertices[2][1]));
    float maxZ = max(vertices[0][2], max(vertices[1][2], vertices[2][2]));

    bbox = AABB(vec3(minX, minY, minZ), vec3(maxX, maxY, maxZ));
    return true;
}
