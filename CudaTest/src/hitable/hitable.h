#pragma once

#include "../core/ray.h"
#include "../core/aabb.h"
#include "transform.h"


class Material;

// �Փˌ��ʂ̃f�[�^���i�[����
struct HitRecord {
    float t;
    float u;
    float v;
    vec3  p;
    vec3  normal;
    Material* mat_ptr;
};


// �����_�����O�Ŏg�p�����I�u�W�F�N�g
// ���X�g��BVH�Ȃǂ����l�Ɉ���

class Hitable {
public:
    __device__ Hitable(Transform t) : transform(t) {}
    __device__ Hitable() { transform = Transform(); }

    __device__ bool hit(const Ray& r,
        float t_min,
        float t_max,
        HitRecord& rec,int frameIndex) 
        {
            Ray transformedRay = transform.TransformRay(r);
            bool flag = collision_detection(transformedRay, t_min, t_max, rec,frameIndex);
            rec.normal = rotate(rec.normal, transform.rotation);
            return flag;
        }

    __device__ virtual bool collision_detection(const Ray& r,
        float t_min,
        float t_max,
        HitRecord& rec, int frameIndex) const = 0;

    __device__ virtual bool bounding_box(float t0,
        float t1,
        AABB& box) const = 0;

    Transform transform;
};
