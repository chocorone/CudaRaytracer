#pragma once

#include "../core/ray.h"
#include "../core/aabb.h"


class Material;

// 衝突結果のデータを格納する
struct HitRecord {
    float t;
    float u;
    float v;
    vec3  p;
    vec3  normal;
    Material* mat_ptr;
};
class Transform {
public:
    __device__ Transform() { position = vec3(); rotation = vec3(); scale = vec3(1); }
    __device__ Transform(vec3 p, vec3 r, vec3 s) : position(p), rotation(r), scale(s) { }
    __device__ Ray TransformRay(const Ray& r)
    {
        return  TranslateRay(ScaleRay(r));
    }

private:
    __device__ Ray TranslateRay(const Ray& r) const {
        Ray moved_r(r.origin() - position, r.direction(), r.time());
        return moved_r;
    }
    __device__ Ray ScaleRay(const Ray& r) const {
        vec3 dir = r.direction()/ scale;
        Ray scaled_r(r.origin(), dir, r.time());
        return scaled_r;
    }

    //publicにしたい
    vec3 position;
    vec3 rotation;
    vec3 scale;
};

// レンダリングで使用されるオブジェクト
// リストやBVHなども同様に扱う

class Hitable {
public:
    __device__ Hitable(Transform* t) : transform(t) {}
    __device__ Hitable() { transform = new Transform(); }

    __device__ bool hit(const Ray& r,
        float t_min,
        float t_max,
        HitRecord& rec) 
        {
            Ray transformedRay = transform->TransformRay(r);
            return collision_detection(transformedRay, t_min, t_max, rec);
        }

    __device__ virtual bool collision_detection(const Ray& r,
        float t_min,
        float t_max,
        HitRecord& rec) const = 0;

    __device__ virtual bool bounding_box(float t0,
        float t1,
        AABB& box) const = 0;

    Transform* transform;
};


/*class FlipNormals : public Hitable {
public:
    __device__ FlipNormals(Hitable* p, Transform* t) : Hitable(t), ptr(p) {}

    __device__ virtual bool hit(const Ray& r,
        float t_min,
        float t_max,
        HitRecord& rec) const {
        if (ptr->hit(r, t_min, t_max, rec)) {
            rec.normal = -rec.normal;
            return true;
        }
        else {
            return false;
        }
    }

    __device__ virtual bool bounding_box(float t0,
        float t1,
        AABB& box) const {
        return ptr->bounding_box(t0, t1, box);
    }

    Hitable* ptr;
};
*/