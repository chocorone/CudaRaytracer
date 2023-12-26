#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include <float.h>


#include "hitable/hitable.h"
#include "hitable/bvh.h"
#include "hitable/hitable_list.h"
#include "core/camera.h"
#include "shapes/sphere.h"
#include "shapes/triangle.h"
#include "shapes/box.h"
#include "material/material.h"
#include "hitable/animationData.h"
#include "shapes/MeshObject.h"
#include "core/deviceManage.h"

__device__ float rand(curandState* state) {
    return float(curand_uniform(state));
}

__global__ void create_TransformList(TransformList** transformPointer)
{
    *transformPointer = new TransformList();
}
__global__ void create_List(HitableList** list)
{
    *list = new HitableList();
}


//BVHの作成
__global__ void create_BVH(HitableList** list, BVHNode** bvh,curandState* state) {
    *bvh = new BVHNode((*list)->list, (*list)->list_size, 0, 1, state);
    //(*bvh)->transform->rotation = vec3(0, 45, 0);
}

__global__ void UpdateBVH(BVHNode** bvh) {
    (*bvh)->UpdateBVH();
}


// オブジェクトの生成
__global__ void add_object(HitableList** list,  TransformList** transformPointer)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {

        Texture* checker = new CheckerTexture(new ConstantTexture(vec3(0.2, 0.3, 0.1)),
            new ConstantTexture(vec3(0.9, 0.9, 0.9)));
        //(*list)->append(new Sphere(new Transform(vec3(-10,-10, 0), vec3(0), vec3(1)), 1, new Lambertian(new ConstantTexture(vec3(0.8, 0.1, 0.1)))));
        //(*list)->append(new Sphere(new Transform(), 1, new Lambertian(new ConstantTexture(vec3(0.1, 0.8, 0.1)))));
        //(*list)->append(new Sphere(new Transform(vec3(5, 5, 0), vec3(0), vec3(1)), 1, new Lambertian(new ConstantTexture(vec3(0.1, 0.1, 0.8)))));
       

        /*Transform* transform1 = new Transform(vec3(0, 1, 0), vec3(0), vec3(1));
        Transform* transform2 = new Transform(vec3(-4, 1, 0), vec3(0), vec3(1));
        Transform* transform3 = new Transform(vec3(4, 1, 0), vec3(0), vec3(1));

        (*list)->append(new Sphere(transform1, 1.0, new Lambertian(checker)));
        (*list)->append(new Sphere(transform2, 1.0, new Dielectric(1.5)));
        (*list)->append(new Sphere(transform3, 1.0, new Metal(vec3(0.7, 0.6, 0.5), 0.0)));

        (*transformPointer)->append(transform1);
        (*transformPointer)->append(transform2);
        (*transformPointer)->append(transform3);*/
    }
}

__global__ void add_mesh_withNormal(HitableList** list, MeshData* data, TransformList** transformPointer)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        Material* mat = new Lambertian(new ConstantTexture(vec3(0.65, 0.05, 0.05)));
        for (int i = 0; i < data->nTriangles; i++) {
            vec3 idx = data->idxVertex[i];
            vec3 v[3] = { data->points[int(idx[2])], data->points[int(idx[1])], data->points[int(idx[0])] };
            Transform* transform = new Transform(vec3(0), vec3(0,0,0), vec3(1));
            //(*transformPointer)->append(transform);//とりあえずなしで
            (*list)->append(new Triangle(v, data->normals[i], mat, false, transform, true));
        }
    }
}

__global__ void add_mesh_withNormal(HitableList** list, FBXObject* data)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        Material* mat = new Lambertian(new ConstantTexture(vec3(0.65, 0.05, 0.05)));
        for (int i = 0; i < data->mesh->nTriangles; i++) {
            vec3 idx = data->mesh->idxVertex[i];
            vec3 v[3] = { data->mesh->points[int(idx[2])], data->mesh->points[int(idx[1])], data->mesh->points[int(idx[0])] };
            Transform* transform = new Transform(vec3(0), vec3(0, 0, 0), vec3(1));
            //(*transformPointer)->append(transform);//とりあえずなしで
            (*list)->append(new Triangle(v, data->mesh->normals[i], mat, false, transform, true));
        }
    }
}

__global__ void add_mesh_fromPoseData(HitableList** list, FBXObject* data,BonePoseData pose) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        data->triangleData = (Triangle**)malloc(data->mesh->nTriangles*sizeof(Triangle*));
        Material* mat = new Lambertian(new ConstantTexture(vec3(0.65, 0.05, 0.05)));
        for (int i = 0; i < data->mesh->nTriangles; i++) {
            vec3 idx = data->mesh->idxVertex[i];
            vec3 v[3] = { data->mesh->points[int(idx[2])], data->mesh->points[int(idx[1])], data->mesh->points[int(idx[0])] };
            Triangle* triagnle = new Triangle(v, data->mesh->normals[i], mat, false, new Transform(), true);
            data->triangleData[i] = triagnle;
            (*list)->append(triagnle);
        }
    }
}

__device__ void CalcFBXVertexPos(FBXObject* data, BonePoseData pose,vec3* newPos) 
{
    for (int i = 0; i < data->mesh->nPoints; i++) {
        newPos[i] = data->mesh->points[i];
    }

    for (int boneIndex = 0; boneIndex < data->boneCount; boneIndex++)
    {
        for (int weightIndex = 0; weightIndex < data->boneList[boneIndex].weightCount; weightIndex++)
        {
            int vertexIndex = data->boneList[boneIndex].weightIndices[weightIndex];
            double weight = data->boneList[boneIndex].weights[weightIndex];
            // ボーンのデフォルトの位置を中心にウェイト分だけ頂点を回転
            const vec3 verticesBasedOrigin = data->mesh->points[vertexIndex] - data->boneList[boneIndex].defaultTransform;
            const vec3 boneRotateDiff = pose.nowRatation[boneIndex] - data->boneList[boneIndex].defaultRotation;
            const vec3 rotatedPosBasedOrigin = rotate(verticesBasedOrigin, boneRotateDiff * weight);
            const vec3 rotatedPos = rotatedPosBasedOrigin + data->boneList[boneIndex].defaultTransform;
            // ウェイト分だけボーンと同様に移動
            const vec3 movedPos = rotatedPos + (pose.nowTransforom[boneIndex] - data->boneList[boneIndex].defaultTransform) * weight;
            //それぞれのボーンの差分を加算
            newPos[vertexIndex] += (pose.nowTransforom[boneIndex] - data->boneList[boneIndex].defaultTransform) * weight;
        }
    }
}

__global__ void update_mesh_fromPoseData(FBXObject* data, BonePoseData pose,float f) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        vec3* newPos = (vec3*)malloc(sizeof(vec3) * data->mesh->nPoints);
        CalcFBXVertexPos(data,pose,newPos);

        for (int i = 0; i < data->mesh->nTriangles; i++) {
            vec3 idx = data->mesh->idxVertex[i];
            vec3 v[3] = { newPos[int(idx[2])], newPos[int(idx[1])], newPos[int(idx[0])]};
            data->triangleData[i]->SetVertices(v);
            
            //Lambertian* mat = (Lambertian*)data->triangleData[i]->material;
            //ConstantTexture* tex = (ConstantTexture*)mat->albedo;
            //tex->color = vec3(tex->color.r(), tex->color.g(), f/10);
        }
        printf("update %f\n", f);
    }
}


__global__ void create_camera(Camera** camera, int nx, int ny,
    vec3 lookfrom, vec3 lookat, float dist_to_focus, float aperture, float vfov)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *camera = new Camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            vfov,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
    }
}

void init_camera(Camera** camera, int nx, int ny,CudaPointerList* pointerList) {
    create_camera << <1, 1 >> > (camera, nx, ny, vec3(0, 20, 400), vec3(0, 20, 0), 10.0, 0.0, 60);
    //create_camera << <1, 1 >> > (camera, nx, ny, vec3(278, 278, -700), vec3(278, 278, 0), 10.0, 0.0, 40);
    pointerList->append((void**)camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void init_TransformList(TransformList** transformPointer, CudaPointerList* pointerList) {
    create_TransformList << <1, 1 >> > (transformPointer);
    pointerList->append((void**)transformPointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void init_List(HitableList** list, CudaPointerList* pointerList)
{
    create_List << <1, 1 >> > (list);
    pointerList->append((void**)list);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

}