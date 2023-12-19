#pragma once
#include "transform.h"

class Bone {
public:
    __host__ __device__ Bone() {}
    __host__ __device__ Bone(const char* name, vec3 defaultT, vec3 defaultR, vec3 t, vec3 r, int* indices, double* weight) {
        boneName = name;
        defaultTransform = defaultT;
        defaultRotation = defaultR;
        nowTransform = t;
        nowRotation = r;
        weightIndices = indices;
        weights = weight;
    }

    __host__ __device__ vec3 GetDiffTransform() {
        return nowTransform - defaultTransform;
    }

    __host__ __device__ vec3 GetDiffRotation() {
        return nowRotation - defaultRotation;
    }

    const char* boneName;
    vec3 defaultTransform;
    vec3 defaultRotation;
    vec3 nowTransform;
    vec3 nowRotation;
    int* weightIndices;
    double* weights;
};

class BonePoseData {
public:
    __host__ __device__ BonePoseData() {}
    __host__ __device__ BonePoseData(vec3 t,vec3 r) {
        nowLclTransforom = t;
        nowLclRatation = r;
        childData = new BonePoseData * (); childCount = 0;
    }

    __host__ __device__ void ResisterChild(BonePoseData** children,int count) {
        childData = children;
        childCount = count;
    }

    __host__ __device__ void freeMemory()
    {
        free(childData);

        childCount = 0;
    }

    int childCount;
    BonePoseData** childData;
    vec3 nowLclTransforom;
    vec3 nowLclRatation;
};


struct KeyFrame {
public:
    int frame;
    Transform* transform;
    __host__ __device__ KeyFrame() {
        frame = 0;
        transform = new Transform();
    }

    __host__ __device__ KeyFrame(int frameNum, Transform* t) :transform(t), frame(frameNum) {

    }
};

class KeyFrameList {
public:
    __host__ __device__ KeyFrameList() { list = new KeyFrame * (); list_size = 0; }
    __host__ __device__ KeyFrameList(KeyFrame** l, int n) { list = l; list_size = n; }
    __host__ __device__ void append(KeyFrame* data)
    {
        KeyFrame** tmp = (KeyFrame**)malloc(sizeof(KeyFrame*) * list_size);

        for (int i = 0; i < list_size; i++)
        {
            tmp[i] = list[i];
        }

        free(list);

        list_size++;

        list = (KeyFrame**)malloc(sizeof(KeyFrame*) * list_size);

        for (int i = 0; i < list_size - 1; i++)
        {
            list[i] = tmp[i];
        }
        list[list_size - 1] = data;

        free(tmp);
    }
    __host__ __device__  void freeMemory()
    {
        free(list);
        list_size = 0;
    }

    KeyFrame** list;
    int list_size;
};



struct AnimationData {
public:
    KeyFrameList* keyframs;
    int currentFrameIndex;
    __host__ __device__ AnimationData() {
        currentFrameIndex = 0;
        keyframs = new KeyFrameList();
    }
    __host__ __device__ AnimationData(KeyFrameList* frames) :keyframs(frames) {
        currentFrameIndex = 0;
    }
    __host__ __device__ Transform Get_NextTransform(int nextFrame) {
        int cnt = sizeof(keyframs) / sizeof(KeyFrame);
        if (keyframs->list_size <= currentFrameIndex + 1)
        {
            //printf("return:%d\n",cnt);
            return *keyframs->list[currentFrameIndex]->transform;
        }

        //•âŠ®‚µ‚½Transform‚ð•Ô‚·
        Transform* begin = keyframs->list[currentFrameIndex]->transform;
        Transform* end = keyframs->list[currentFrameIndex+1]->transform;
        float t = ((float)(nextFrame - keyframs->list[currentFrameIndex]->frame)) / (keyframs->list[currentFrameIndex+1]->frame - keyframs->list[currentFrameIndex]->frame);
        //printf("t:%f\n", t);
        
        //vec3 position = SLerp(begin.position, end.position, t);
        vec3 position = lerp(t, begin->position, end->position);
        //vec3 rotation = SLerp(begin.rotation, end.rotation, t);
        vec3 rotation = lerp(t, begin->rotation, end->rotation);
        //vec3 scale = SLerp(begin.scale, end.scale, t);
        vec3 scale = lerp(t, begin->scale, end->scale);
        //printf("%f\n", scale.y());
        return  Transform(position, rotation, scale);
    }

    __host__ __device__ void SetNext(int nextFrame) {
        if (sizeof(keyframs) / sizeof(KeyFrame) <= currentFrameIndex + 1) return;
        if (keyframs->list[currentFrameIndex+1]->frame >= nextFrame)currentFrameIndex++;
    }
};


class AnimationDataList {
public:
    __host__ __device__ AnimationDataList() { list = new AnimationData * (); list_size = 0; }
    __host__ __device__ AnimationDataList(AnimationData** l, int n) { list = l; list_size = n; }
    __host__ __device__ void append(AnimationData* data)
    {
        AnimationData** tmp = (AnimationData**)malloc(sizeof(AnimationData*) * list_size);

        for (int i = 0; i < list_size; i++)
        {
            tmp[i] = list[i];
        }

        free(list);

        list_size++;

        list = (AnimationData**)malloc(sizeof(AnimationData*) * list_size);

        for (int i = 0; i < list_size - 1; i++)
        {
            list[i] = tmp[i];
        }
        list[list_size - 1] = data;

        free(tmp);
    }
    __host__ __device__  void freeMemory()
    {
        free(list);
        list_size = 0;
    }

    AnimationData** list;
    int list_size;
};