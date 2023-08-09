#pragma once
#include "hitable.h"


class HitableList : public Hitable {
public:
    __device__ HitableList() {}
    __device__ HitableList(Hitable** l, int n) { list = l; list_size = n; }
    __device__ virtual bool hit(const Ray& r,
        float t_min,
        float t_max,
        HitRecord& rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const;

    Hitable** list;
    int list_size;
};


__device__ bool HitableList::hit(const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec) const {
    HitRecord tmp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, tmp_rec) && tmp_rec.t < closest_so_far) {
            // if(list[i]->hit(r, t_min, closest_so_far, tmp_rec)){  // origin one, maybe bug existing
            hit_anything = true;
            closest_so_far = tmp_rec.t;
            rec = tmp_rec;
        }
    }
    return hit_anything;
}


__device__ bool HitableList::bounding_box(float t0,
    float t1,
    AABB& box) const {
    if (list_size < 1) return false;

    AABB tmp_box;
    bool first_true = list[0]->bounding_box(t0, t1, tmp_box);

    if (!first_true) {
        return false;
    }
    else {
        box = tmp_box;
    }

    for (int i = 1; i < list_size; i++) {
        if (list[i]->bounding_box(t0, t1, tmp_box)) {
            box = surrounding_box(box, tmp_box);
        }
        else {
            return false;
        }
    }
    return true;
}