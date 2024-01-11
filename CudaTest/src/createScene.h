#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include <float.h>
#include <set>


#include "hitable/hitable.h"
#include "hitable/hitable_list.h"
#include "core/camera.h"
#include "shapes/sphere.h"
#include "shapes/triangle.h"
#include "shapes/box.h"
#include "material/material.h"
#include "hitable/animationData.h"
#include "shapes/MeshObject.h"
#include "core/deviceManage.h"
#include "Loader/FbxLoader.h"
#include "hitable/BoneBVH.h"

__device__ float rand(curandState* state) {
    return float(curand_uniform(state));
}

__global__ void create_TransformList(TransformList** transformPointer)
{
    *transformPointer = new TransformList();
}
__global__ void create_List(HitableList** list)
{
    (*list) = new HitableList();
}
__global__ void create_List(HitableList** list,int n)
{
    (*list) = new HitableList(n);
}


//BVHの作成
__global__ void create_BVH(HitableList** list, BVHNode** bvh,curandState* state) {
    *bvh = new BVHNode((*list)->list, (*list)->list_size, 0, 1, state);
    //(*bvh)->transform->rotation = vec3(0, 45, 0);
}

__global__ void UpdateBVH(BVHNode** bvh) {
    (*bvh)->UpdateBVH();
}

__global__ void UpdateBVH(HitableList** list) {

    for (size_t i = 0; i < (*list)->list_size; i++)
    {
        ((BoneBVHNode*)((*list)->list[i]))->UpdateBVH();
    }
    
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
        (*list)->Reseize(data->mesh->nTriangles);
        for (int i = 0; i < data->mesh->nTriangles; i++) {
            vec3 idx = data->mesh->idxVertex[i];
            vec3 v[3] = { data->mesh->points[int(idx[2])], data->mesh->points[int(idx[1])], data->mesh->points[int(idx[0])] };
            Triangle* triagnle = new Triangle(v, data->mesh->normals[i], mat, false, new Transform(), false);
            data->triangleData[i] = triagnle;
            (*list)->list[i] = (Hitable*)triagnle;
        }
    }
}

__device__ void CalcFBXVertexPos(FBXObject* data, BonePoseData pose,vec3* newPos) 
{
    for (int i = 0; i < data->mesh->nPoints; i++) {
        newPos[i] = data->mesh->points[i];
    }

    for (int boneIndex = 0; boneIndex < data->boneCount; boneIndex++)
    //for (int boneIndex = 3; boneIndex < 7; boneIndex++)
    {
        data->boneList[boneIndex].nowTransform = pose.nowTransforom[boneIndex];
        data->boneList[boneIndex].nowRotation = pose.nowRatation[boneIndex];

       for (int weightIndex = 0; weightIndex < data->boneList[boneIndex].weightCount; weightIndex++)
       {
            int vertexIndex = data->boneList[boneIndex].weightIndices[weightIndex];
            double weight = data->boneList[boneIndex].weights[weightIndex];
            //とりあえず動くが関節が微妙
            const vec3 posePos = (pose.nowTransforom[boneIndex] - data->boneList[boneIndex].defaultTransform) * weight;
            
            //SLerpで試す
            //const vec3 posePos = SLerp(vec3(0), pose.nowTransforom[boneIndex] - data->boneList[boneIndex].defaultTransform,weight);

            //回転を試してみたい
            //頂点をボーンのデフォルトの位置分移動
            //const vec3 localPos =  data->boneList[boneIndex].nowTransform- data->boneList[boneIndex].defaultTransform;
            //原点で回転させる
            //const vec3 rotateDiff = data->boneList[boneIndex].nowRotation - data->boneList[boneIndex].defaultRotation;
            //const vec3 rotateDiff = data->boneList[boneIndex].nowRotation;
            //const vec3 localRotatedPos = rotate(localPos * weight, rotateDiff);
            //移動分を加算
            //const vec3 posePos = localPos * weight ;

            //テスト
            newPos[vertexIndex] += posePos;
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
        //printf("update %f\n", f);
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
    create_camera << <1, 1 >> > (camera, nx, ny, vec3(0, 150, 400), vec3(0, 150, 0), 10.0, 0.0, 40);//low_walk
    //create_camera << <1, 1 >> > (camera, nx, ny, vec3(200, 200, 400), vec3(0, 200, 0), 10.0, 0.0, 60);//low_stand

    //create_camera << <1, 1 >> > (camera, nx, ny, vec3(200, 250, 200), vec3(0, 200, 0), 10.0, 0.0, 60);//high_walk
    //create_camera << <1, 1 >> > (camera, nx, ny, vec3(200, 250, 300), vec3(0, 200, 0), 10.0, 0.0, 60);//新しい
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
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    pointerList->append((void**)list);
}

void create_FBXObject(const std::string& filePath, FBXObject* fbxData, FBXAnimationData* fbxAnimationData,int &endFrame, CudaPointerList* pointerList) {
    pointerList->append((void**)fbxData);
    CreateFBXData(filePath, fbxData, fbxAnimationData, endFrame);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

void create_FBXMesh(HitableList** list, FBXObject* data, FBXAnimationData* fbxAnimationData) 
{
    add_mesh_fromPoseData << <1, 1 >> > (list, data, fbxAnimationData->animation[0]); //メッシュの移動と作成
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

void create_BVHfromList(BVHNode** bvh,HitableList** list, curandState* curand_state, CudaPointerList* pointerList)
{
    pointerList->append((void**)bvh);
    create_BVH << <1, 1 >> > (list, bvh, curand_state);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void create_BoneBVH(HitableList** list, FBXObject* d_FBXdata, bool* d_hasTriangle, int boneIndex, curandState* curand_state,int hasTriangleNum)
{
    // ボーンの三角形のHitableListを作成
    HitableList* triangleList = new HitableList(hasTriangleNum);
    int i = 0;
    //三角形がsetに含まれるか判定、含まれてたらリストに入れる
    for (int triangleIndex = 0; triangleIndex < d_FBXdata->mesh->nTriangles; triangleIndex++)
    {
        if (d_hasTriangle[triangleIndex]) {
            triangleList->list[i]=d_FBXdata->triangleData[triangleIndex];
            i++;
        }
    }

    //リストからBoneBVHNodeを作成する
    BoneBVHNode* node = new BoneBVHNode(triangleList->list, triangleList->list_size, 0, 1, curand_state, &d_FBXdata->boneList[boneIndex],true);
    //BoneBVHNodeをListに追加
    (*list)->list[boneIndex]=node;
    triangleList->freeMemory();
}



__global__ void create_emptyBoneBVH(HitableList** list,int boneIndex)
{
    BoneBVHNode* node = new BoneBVHNode(true);
    (*list)->list[boneIndex] = node;
}




__global__ void CopyBoneCount(FBXObject* d_FBXdata, int* d_BoneCount, int* d_triangleCount) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_BoneCount[0] = d_FBXdata->boneCount;
        d_triangleCount[0] = d_FBXdata->mesh->nTriangles;
    }
}

__global__ void CopyIdxVertices(FBXObject* d_FBXdata, vec3* d_idxVertices,int vertexNum) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < vertexNum; i++) 
        {
            d_idxVertices[i] = d_FBXdata->mesh->idxVertex[i];
        }
    }
}

__global__ void CopyWeightCount(FBXObject* d_FBXdata,int boneIndex, int* d_WeightCount) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_WeightCount[0] = d_FBXdata->boneList[boneIndex].weightCount;
    }
}

__global__ void CopyBoneVertices(FBXObject* d_FBXdata, int boneIndex, int* d_vertices,int weightCount) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < weightCount; i++)
        {
            d_vertices[i] = d_FBXdata->boneList[boneIndex].weightIndices[i];
        }

    }
}

void createBoneBVH(HitableList** list, FBXObject* d_FBXdata, curandState* curand_state, CudaPointerList* pointerList)
{
    int *h_boneCount = (int*)malloc(1 * sizeof(int));
    h_boneCount[0] = 0;
    int *d_boneCount;
    cudaMalloc(&d_boneCount, 1 * sizeof(int));
    int* h_triangleCount = (int*)malloc(1 * sizeof(int));
    h_triangleCount[0] = 0;
    int* d_triangleCount;
    cudaMalloc(&d_triangleCount, 1 * sizeof(int));
    CopyBoneCount << <1, 1 >> > (d_FBXdata, d_boneCount,d_triangleCount);
    cudaMemcpy(h_boneCount, d_boneCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_triangleCount, d_triangleCount, sizeof(int), cudaMemcpyDeviceToHost);

    create_List << <1, 1 >> > (list, h_boneCount[0]);

    vec3* h_idxVertices = (vec3*)malloc(h_triangleCount[0] * sizeof(vec3));
    vec3* d_idxVertices;
    cudaMalloc(&d_idxVertices, h_triangleCount[0] * sizeof(vec3));
    CopyIdxVertices << <1, 1 >> > (d_FBXdata, d_idxVertices, h_triangleCount[0]);
    cudaMemcpy(h_idxVertices, d_idxVertices, h_triangleCount[0] * sizeof(vec3), cudaMemcpyDeviceToHost);
    std::vector<bool> triangleAdded(h_triangleCount[0],false);

    // BoneBVHNodeをボーン分作成
    for (int boneIndex = 0; boneIndex < h_boneCount[0]; boneIndex++)
    {
        //printf("BVH作成 %d\n", boneIndex);
        bool* h_hasTriangle = (bool*)malloc(h_triangleCount[0] * sizeof(bool));
        //setにボーンの頂点のインデックスを格納
        std::set<int> boneVerticesIndexes;

        //weightIndicesとweightCountを取得
        int* h_weightCount = (int*)malloc(1 * sizeof(int));
        h_weightCount[0] = 0;
        int* d_weightCount;
        cudaMalloc(&d_weightCount, 1 * sizeof(int));
        CopyWeightCount << <1, 1 >> > (d_FBXdata, boneIndex,d_weightCount);
        cudaMemcpy(h_weightCount, d_weightCount, 1, cudaMemcpyDeviceToHost);
        //printf("weightCount:%d\n", h_weightCount[0]);

        int* h_weightIndecis = (int*)malloc(h_weightCount[0] * sizeof(int));
        int* d_weightIndecis;
        cudaMalloc(&d_weightIndecis, h_weightCount[0] * sizeof(int));
        CopyBoneVertices << <1, 1 >> > (d_FBXdata, boneIndex, d_weightIndecis, h_weightCount[0]);
        cudaMemcpy(h_weightIndecis, d_weightIndecis, h_weightCount[0] * sizeof(int), cudaMemcpyDeviceToHost);

        for (int weightIndex = 0; weightIndex < h_weightCount[0]; weightIndex++)
        {
            boneVerticesIndexes.insert(h_weightIndecis[weightIndex]);
        }

        int hasTriangleNum=0;
        //printf("三角形の数 %d\n", h_triangleCount[0]);
        //三角形がsetに含まれるか判定、含まれてたらリストに入れる
        for (int triangleIndex = 0; triangleIndex < h_triangleCount[0]; triangleIndex++)
        {
            h_hasTriangle[triangleIndex] = false;
            if (triangleAdded[triangleIndex])continue;
            vec3 pointsIndex = h_idxVertices[triangleIndex];
            if (boneVerticesIndexes.find((int)pointsIndex[0]) != boneVerticesIndexes.end()
                && boneVerticesIndexes.find((int)pointsIndex[1]) != boneVerticesIndexes.end()
                && boneVerticesIndexes.find((int)pointsIndex[2]) != boneVerticesIndexes.end())
            {
                h_hasTriangle[triangleIndex]=true;
                triangleAdded[triangleIndex] = true;
                hasTriangleNum++;
            }
        }
        bool* d_hasTriangle;
        cudaMalloc(&d_hasTriangle, h_triangleCount[0] * sizeof(bool));
        cudaMemcpy(d_hasTriangle, h_hasTriangle, h_triangleCount[0] * sizeof(bool), cudaMemcpyHostToDevice);

        if (hasTriangleNum==0) {
            create_emptyBoneBVH << <1, 1 >> > (list, boneIndex);
        }
        else {
            create_BoneBVH << <1, 1 >> > (list, d_FBXdata, d_hasTriangle, boneIndex, curand_state, hasTriangleNum);
        }
        CHECK(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        cudaFree(d_hasTriangle);
        free(h_hasTriangle);

        //weightIndicesとweightCountを解放
        cudaFree(d_weightCount);
        cudaFree(d_weightIndecis);
        free(h_weightCount);
        free(h_weightIndecis);
    }

}