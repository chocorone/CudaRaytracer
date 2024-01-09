#pragma once

#include "bvh.h"

class BoneBVHNode : public Hitable {
public:
    __device__ BoneBVHNode() {}
    __device__ BoneBVHNode(bool empty) { isEmpty = true; }
    __device__ BoneBVHNode(Hitable** l,
        int n,
        float time0,
        float time1,
        curandState* state, Bone* b,bool root);

    __device__ virtual bool collision_detection(const Ray& r,
        float t_min,
        float t_max,
        HitRecord& rec, int frameIndex) const;

    __device__ virtual bool bounding_box(float t0,
        float t1,
        AABB& b) const;

    __device__ void UpdateBVH();

    Hitable* left;
    Hitable* right;
    AABB box;
    Bone* bone;
    bool childIsNode;
    bool isRoot;
    bool isEmpty = false;
};


// 与えられたボーン情報、三角形のリストからBVHを作成する
__device__ BoneBVHNode::BoneBVHNode(Hitable** l,
    int n,
    float time0,
    float time1,
    curandState* state,Bone* b,bool root) {
    transform->ResetTransform();
    isRoot = root;
    bone = b;

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
        left = right = l[0];
        childIsNode = false;
    }
    else if (n == 2) {
        left = l[0];
        right = l[1];
        childIsNode = false;
    }
    else {
        left = new BoneBVHNode(l, n / 2, time0, time1, state,bone,false);
        right = new BoneBVHNode(l + n / 2, n - n / 2, time0, time1, state, bone, false);
        childIsNode = true;
    }

    AABB box_left, box_right;
    if (!left->GetBV(time0, time1, box_left) ||
        !right->GetBV(time0, time1, box_right)) {
        return;
        // std::cerr << "no bounding box in BVHNode constructor \n";
    }

    box = surrounding_box(box_left, box_right);

    if (!childIsNode)
    {
        // bvを移動or回転
        box = moveAABB(box, -bone->defaultTransform);
    }
}


__device__ bool BoneBVHNode::bounding_box(float t0,
    float t1,
    AABB& b) const {
    b = box;
    return true;
}

__device__ void BoneBVHNode::UpdateBVH()
{
    if (isEmpty)return;
    if (childIsNode) {
        ((BoneBVHNode*)left)->UpdateBVH();
        ((BoneBVHNode*)right)->UpdateBVH();
    }

    AABB box_left, box_right;
    if (!left->GetBV(0, 1, box_left) ||
        !right->GetBV(0, 1, box_right)) {
        return;
        // std::cerr << "no bounding box in BVHNode constructor \n";
    }

    //拡大
    //box = surrounding_box(box_left, box);
    //box = surrounding_box(box_right, box);

    // 再フィット
    box = surrounding_box(box_right, box_left);
    if (!childIsNode)
    {
        // bvを移動or回転
        box = moveAABB(box, -bone->nowTransform);
    }

}

// 光線をボーンと同じだけ回転させて衝突
__device__ bool BoneBVHNode::collision_detection(const Ray& r,
    float t_min,
    float t_max,
    HitRecord& rec, int frameIndex) const {
    if (isEmpty)return false;

    Ray moved_r(r.origin(), r.direction(), r.time());
    //ボーンの座標分光線を変形
    if (isRoot) 
    {
        moved_r = Ray(r.origin() - bone->nowTransform, r.direction(), r.time());
    }

    //葉の親と衝突判定時は光線の変形を戻す
    if (box.hit(moved_r, t_min, t_max)) {
        //rec.normal = vec3(0,0,1);
        //return true;


        if (!childIsNode) {
            //printf("hit to node\n");
            moved_r = Ray(r.origin() + bone->nowTransform, r.direction(), r.time());
        }
        HitRecord left_rec, right_rec;
        bool hit_left = left->hit(moved_r, t_min, t_max, left_rec, frameIndex);
        bool hit_right = right->hit(moved_r, t_min, t_max, right_rec, frameIndex);

        if (hit_left && hit_right) {
            if (left_rec.t < right_rec.t) {
                rec = left_rec;
            }
            else {
                rec = right_rec;
            }

            //if (!childIsNode) printf("hit to child\n");

            return true;
        }
        else if (hit_left) {
            rec = left_rec;
            //if (!childIsNode) printf("hit to child\n");


            return true;
        }
        else if (hit_right) {
            rec = right_rec;
            //if (!childIsNode) printf("hit to child\n");

            return true;
        }
        else {
            return false;
        }
    }
    return false;
}
