#pragma once
#include "../core/vec3.h"

class MeshData {
public:
	int nPoints;
	int nTriangles;
	vec3* points;
	vec3* idxVertex;
	vec3* normals;
};

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

class FBXObject {
public:
	MeshData* mesh;
    Triangle** triangleData;
	Bone* boneList;
    int boneCount;
};


class BonePoseData {
public:
    __host__ __device__ BonePoseData() {}
    int boneCount;

    vec3* nowLclTransforom;
    vec3* nowLclRatation;
};

class FBXAnimationData {
public:
    __host__ __device__ FBXAnimationData() {}
    int frameCount;
    FBXObject* object;
    BonePoseData* animation;
};