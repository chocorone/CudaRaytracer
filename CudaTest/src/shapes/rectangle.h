#pragma once
#include "../hitable/hitable.h"

class Rectangle : public Hitable {
public:
    __device__ Rectangle() {};
    __device__ Rectangle(Material* mat, bool flip) :mat_ptr(mat), flipNormal(flip) {};
    __device__ Rectangle(Material* mat,bool flip, Transform t) : Hitable(t),flipNormal(flip) ,mat_ptr(mat) {
    };

    __device__ virtual bool collision_detection(const Ray& r, float t0, float t1, HitRecord& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const {
        box = AABB(vec3(-0.5, -0.5, -0.0001), vec3(0.5, 0.5, 0.0001));
        return true;
    }
    Material* mat_ptr;
    bool flipNormal;
};


__device__ bool Rectangle::collision_detection(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
    
    vec3 normal = flipNormal ? vec3(0, 0, -1 ) : vec3(0, 0, 1);
    if (dot(r.direction(), normal) > 0)return false;
    float t = -r.origin().z() / r.direction().z();
    if (t < t_min || t > t_max) return false;

    float x0 = -0.5;
    float x1 = 0.5;
    float y0 = -0.5;
    float y1 = 0.5;

    float x = r.origin().x() + t * r.direction().x();
    float y = r.origin().y() + t * r.direction().y();
    if (x < x0 || x > x1 || y < y0 || y > y1) return false;

    rec.u = (x - x0) / (x1 - x0);
    rec.v = (y - y0) / (y1 - y0);
    rec.t = t;
    rec.mat_ptr = mat_ptr;
    rec.p = r.point_at_t(t);
    rec.normal = normal;
    return true;
}