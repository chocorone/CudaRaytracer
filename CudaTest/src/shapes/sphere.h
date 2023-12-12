#pragma once

#include "../hitable/hitable.h"


class Sphere : public Hitable {
public:
    __device__ Sphere(Transform t):Hitable(t) {}
    __device__ Sphere(Transform t,
        float r,
        Material* mat) :Hitable(t), radius(r), material(mat) {};

    __device__ virtual bool collision_detection(const Ray& r,
        float t_min,
        float t_max,
        HitRecord& rec, int frameIndex) const;

    __device__ virtual bool bounding_box(float t0,
        float t1,
        AABB& box) const;

    float radius;
    Material* material;
};


__device__ bool Sphere::collision_detection(const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec, int frameIndex) const {
    vec3 oc = r.origin();
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float tmp = (-b - sqrt(discriminant)) / a;
        if (tmp < t_max && tmp > t_min) {
            rec.t = tmp;
            rec.p = r.point_at_t(rec.t);
            rec.normal = (rec.p) / radius;
            rec.mat_ptr = material;
            return true;
        }
        tmp = (-b + sqrt(discriminant)) / a;
        if (tmp < t_max && tmp > t_min) {
            rec.t = tmp;
            rec.p = r.point_at_t(rec.t);
            rec.normal = (rec.p) / radius;
            rec.mat_ptr = material;
            return true;
        }
    }
    return false;
}


__device__ bool Sphere::bounding_box(float t0,
    float t1,
    AABB& box) const {
    box = AABB( - vec3(radius, radius, radius),vec3(radius, radius, radius));
    return true;
}
