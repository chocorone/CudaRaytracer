#pragma once

#include <float.h>


class Transform {
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

struct AnimationData {
public:
    KeyFrame* keyframs;
    int currentFrameIndex;
    __device__ Transform Get_NextTransform(int nextFrame) {
        if (sizeof(keyframs) / sizeof(KeyFrame) <= currentFrameIndex + 1)return keyframs[currentFrameIndex].transform;

        //•âŠ®‚µ‚½Transform‚ð•Ô‚·
        Transform begin = keyframs[currentFrameIndex].transform;
        Transform end = keyframs[currentFrameIndex + 1].transform;
        float t = (nextFrame-keyframs[currentFrameIndex].frame) / (keyframs[currentFrameIndex + 1].frame - keyframs[currentFrameIndex].frame);
        vec3 position = SLerp(begin.position, end.position, t);
        vec3 rotation = SLerp(begin.rotation, end.rotation, t);
        vec3 scale = SLerp(begin.scale, end.scale, t);

        return Transform(position, rotation, scale);
    }

    __device__ void SetNext(int nextFrame) {
        if (sizeof(keyframs) / sizeof(KeyFrame) <= currentFrameIndex+1) return;
        if (keyframs[currentFrameIndex + 1].frame >= nextFrame)currentFrameIndex++;
    }
};