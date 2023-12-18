#pragma once
#include "hitable.h"


// 複数のオブジェクトを格納するリスト
class HitableList : public Hitable {
public:
    __device__ HitableList() { list = new Hitable * (); list_size = 0; }
    __device__ HitableList(Hitable** l, int n){ list = l; list_size = n; }
    __device__ HitableList(Hitable** l, int n, Transform* t) : Hitable(t) { list = l; list_size = n; }
    __device__ virtual bool collision_detection(const Ray& r,
        float t_min,
        float t_max,
        HitRecord& rec, int frameIndex) const;
    __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const;
    __device__ void append(Hitable* data)
    {
        Hitable** tmp = (Hitable**)malloc(sizeof(Hitable*) * list_size);

        for (int i = 0; i < list_size; i++)
        {
            tmp[i] = list[i];
        }

        free(list);

        list_size++;

        list = (Hitable**)malloc(sizeof(Hitable*) * list_size);

        for (int i = 0; i < list_size - 1; i++)
        {
            list[i] = tmp[i];
        }
        list[list_size - 1] = data;

        free(tmp); 
    }
    __device__  void freeMemory()
    {
        free(list);

        list_size = 0;
    }

    Hitable** list;
    int list_size;
};


__device__ bool HitableList::collision_detection(const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec, int frameIndex) const {
    HitRecord tmp_rec;
    bool hit_anything = false;
    double closest_so_far = t_max;
    for (int i = 0; i < list_size; i++) {
        if (list[i]->hit(r, t_min, closest_so_far, tmp_rec, frameIndex) && tmp_rec.t < closest_so_far) {
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
    bool first_true = list[0]->GetBV(t0, t1, tmp_box);

    if (!first_true) {
        return false;
    }
    else {
        box = tmp_box;
    }

    for (int i = 1; i < list_size; i++) {
        if (list[i]->GetBV(t0, t1, tmp_box)) {
            box = surrounding_box(box, tmp_box);
        }
        else {
            return false;
        }
    }
    return true;
}

