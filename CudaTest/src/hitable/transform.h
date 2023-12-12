#pragma once

#include <float.h>


struct Transform {
public:
    __device__ Transform() { position = vec3(); rotation = vec3(); scale = vec3(1); }
    __device__ Transform(vec3 p, vec3 r, vec3 s) : position(p), rotation(r), scale(s)
    {
        //printf("set transform\n");
    }
    __device__ Ray TransformRay(const Ray& r)
    {
        return  TranslateRay(RotateRay(ScaleRay(r)));
    }

    vec3 position;
    vec3 rotation;
    vec3 scale;

private:
    __device__ Ray TranslateRay(const Ray& r) const {
        Ray moved_r(r.origin() - position, r.direction(), r.time());
        return moved_r;
    }

    __device__ Ray RotateRay(const Ray& r) const {
        
        Ray rotate_r(rotate(r.origin(),rotation), rotate(r.direction(),rotation), r.time());

        return rotate_r;
    }

    __device__ Ray ScaleRay(const Ray& r) const {
        vec3 dir = r.direction() / scale;
        Ray scaled_r(r.origin(), unit_vector(dir), r.time()*dir.length());
        return scaled_r;
    }

};

struct KeyFrame {
public:
    int frame;
    Transform transform;
};

/*
__host__ __device__ inline Transform get_key(KeyFrame* keys,int nowFrame) {
    return v / v.length();
}

__host__ __device__ inline int get_next_frameIndex(KeyFrame* keys, int nowFrameIndex,int nowFrame) {
    int size = sizeof(keys) / sizeof(keys);
    if (size) {
        return nowFrameIndex;
    }
 
    return keys[nowFrameIndex + 1].frame <= nowFrame?nowFrameIndex + 1:nowFrameIndex;
}*/