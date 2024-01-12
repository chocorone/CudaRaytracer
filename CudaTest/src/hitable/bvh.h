#pragma once

#include <thrust/sort.h>
#include <curand.h>
#include <curand_kernel.h>
#include "hitable.h"


struct BoxCompare {
    __device__ BoxCompare(int m) : mode(m) {}
    __device__ bool operator()(Hitable* a, Hitable* b) const {
        // return true;

        AABB box_left, box_right;
        Hitable* ah = a;
        Hitable* bh = b;

        if (!ah->GetBV(0, 0, box_left) || !bh->GetBV(0, 0, box_right)) {
            return false;
        }

        float val1, val2;
        if (mode == 1) {
            val1 = box_left.min().x();
            val2 = box_right.min().x();
        }
        else if (mode == 2) {
            val1 = box_left.min().y();
            val2 = box_right.min().y();
        }
        else if (mode == 3) {
            val1 = box_left.min().z();
            val2 = box_right.min().z();
        }

        if (val1 - val2 < 0.0) {
            return false;
        }
        else {
            return true;
        }
    }
    // mode: 1, x; 2, y; 3, z
    int mode;
};


class BVHNode : public Hitable {
public:
    __device__ BVHNode() {}
    __device__ BVHNode(Hitable** l,
        int n,
        float time0,
        float time1,
        curandState* state) ;

    __device__ virtual bool collision_detection(const Ray& r,
        float t_min,
        float t_max,
        HitRecord& rec, int frameIndex) const;

    __device__ virtual bool bounding_box(float t0,
        float t1,
        AABB& b) const;

    __device__ void UpdateBVH();

    BVHNode* left;
    BVHNode* right;
    HitableList* childList;
    AABB box;
    bool isLeaf;
};


__device__ BVHNode::BVHNode(Hitable** l,
    int n,
    float time0,
    float time1,
    curandState* state) {
    transform->ResetTransform();
    

    int axis = int(3 * curand_uniform(state));
    if (axis == 0) {
        thrust::sort(l, l + n, BoxCompare(1));
    }
    else if (axis == 1) {
        thrust::sort(l, l + n, BoxCompare(2));
    }
    else {
        thrust::sort(l, l + n, BoxCompare(3));
    }

    if (n == 1) {
        childList = new HitableList(1);
        childList->transform->ResetTransform();
        childList->list[0] = l[0];
        isLeaf = true;
        childList->GetBV(0,1,box);
    }
    else if (n == 2) {
        childList = new HitableList(2);
        childList->transform->ResetTransform();
        childList->list[0] = l[0];
        childList->list[1] = l[1];
        isLeaf = true;
        childList->GetBV(0, 1, box);
    }
    else {
        left = new BVHNode(l, n / 2, time0, time1, state);
        right = new BVHNode(l + n / 2, n - n / 2, time0, time1, state);
        isLeaf = false;

        AABB box_left, box_right;
        if (!left->GetBV(time0, time1, box_left) ||
            !right->GetBV(time0, time1, box_right)) {
            return;
            // std::cerr << "no bounding box in BVHNode constructor \n";
        }
        box = surrounding_box(box_left, box_right);
    }

    //printf("box mix:%f,%f,%f\nbox max:%f,%f,%f\n",box.min().x(), box.min().y(), box.min().z(),box.max().x(), box.max().y(), box.max().z());
}


__device__ bool BVHNode::bounding_box(float t0,
    float t1,
    AABB& b) const {
    b = box;
    return true;
}

__device__ void BVHNode::UpdateBVH()
{
    if (isLeaf) 
    {
        childList->bounding_box(0, 1, box);
    }
    else {
        left->UpdateBVH();
        right->UpdateBVH();

        AABB box_left, box_right;
        if (!left->GetBV(0, 1, box_left) ||
            !right->GetBV(0, 1, box_right)) {
            return;
        }
        //拡大
        //box = surrounding_box(box_left, box);
        //box = surrounding_box(box_right, box);

        // 再フィット
        box = surrounding_box(box_right, box_left);
    }
    
}

__device__ bool BVHNode::collision_detection(const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec, int frameIndex) const {
   
    if (!box.hit(r, t_min, t_max)) return false;
    if (isLeaf)return childList->hit(r, t_min, t_max, rec, frameIndex);
    
    HitRecord left_rec, right_rec;
    bool hit_left = left->hit(r, t_min, t_max, left_rec, frameIndex);
    bool hit_right = right->hit(r, t_min, t_max, right_rec, frameIndex);
    if (hit_left && hit_right) {
        if (left_rec.t < right_rec.t) {
            rec = left_rec;
        }
        else {
            rec = right_rec;
        }
        return true;
    }
    else if (hit_left) {
        rec = left_rec;
        return true;
    }
    else if (hit_right) {
        rec = right_rec;
        return true;
    }
        
    return false;
}
