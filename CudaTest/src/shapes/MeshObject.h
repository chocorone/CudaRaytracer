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
    __host__ __device__ Bone(const char* name, vec3 defaultT, vec3 defaultR, vec3 t, vec3 r,int wcount) {
        boneName = name;
        defaultTransform = defaultT;
        defaultRotation = defaultR;
        nowTransform = t;
        nowRotation = r;
        weightCount = wcount;
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

    int weightCount;
    int* weightIndices;
    double* weights;
};


class BonePoseData {
public:
    __host__ __device__ BonePoseData() {}
    int boneCount;

    vec3* nowTransforom;
    vec3* nowRatation;
};

class FBXAnimationData {
public:
    __host__ __device__ FBXAnimationData() {}
    int frameCount;
    BonePoseData* animation;
};

class FBXObject {
public:
	MeshData* mesh;
    Triangle** d_triangleData;
	Bone* boneList;
    int boneCount;
    FBXAnimationData* fbxAnimationData;
    FBXObject() 
    {
        mesh = new MeshData();
        fbxAnimationData = new FBXAnimationData();
    }
};

