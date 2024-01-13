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
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*bvh)->UpdateBVH();
    }
}

void Update_BVH(BVHNode** d_bvhNode)
{
    UpdateBVH << <1, 1 >> > (d_bvhNode);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
}

__global__ void UpdateBVH(HitableList** list,vec3* nowT) {

    for (size_t i = 0; i < (*list)->list_size; i++)
    {
        ((BoneBVHNode*)((*list)->list[i]))->UpdateBVH(nowT[i]);
    }
    
}


__global__ void UpdateBVH(HitableList** list, vec3* d_nowTransform, int boneCount) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < boneCount)
    {
        ((BoneBVHNode*)((*list)->list[i]))->UpdateBVH(d_nowTransform[i]);
    }

}


void Update_BVH(HitableList** d_boneBvhNode, FBXObject* obj)
{
    vec3* h_nowTransform = (vec3*)malloc(sizeof(vec3) * obj->boneCount);
    for (int i = 0; i < obj->boneCount; i++)
    {
        h_nowTransform[i] = obj->boneList[i].nowTransform;
    }

    vec3* d_nowTransform;
    cudaMalloc(&d_nowTransform, sizeof(vec3) * obj->boneCount);
    cudaMemcpy(d_nowTransform, h_nowTransform, obj->boneCount * sizeof(vec3), cudaMemcpyHostToDevice);
    UpdateBVH << <1, obj->boneCount >> > (d_boneBvhNode, d_nowTransform, obj->boneCount);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_nowTransform));
    free(h_nowTransform);
}


__global__ void update_pose(Triangle** tris, vec3* newPos, vec3* idxVertices, int triangleNum)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        for (int i = 0; i < triangleNum; i++) {
            vec3 idx = idxVertices[i];
            vec3 v[3] = { newPos[int(idx[2])], newPos[int(idx[1])], newPos[int(idx[0])] };
            tris[i]->SetVertices(v);
            //Lambertian* mat = (Lambertian*)tris[i]->material;
            //ConstantTexture* tex = (ConstantTexture*)mat->albedo;
            //tex->color = vec3(0,0, (float)i / (float)triangleNum);
        }
    }
}

void calcPose(int frame, const FBXObject* data, vec3*& newPos, const vec3* idxVertices)
{
    newPos = (vec3*)malloc(sizeof(vec3) * data->mesh->nPoints);
    for (int i = 0; i < data->mesh->nPoints; i++) {
        newPos[i] = data->mesh->points[i];
    }

    BonePoseData pose = data->fbxAnimationData->animation[frame];

    for (int boneIndex = 0; boneIndex < data->boneCount; boneIndex++)
    {
        for (int weightIndex = 0; weightIndex < data->boneList[boneIndex].weightCount; weightIndex++)
        {
            int vertexIndex = data->boneList[boneIndex].weightIndices[weightIndex];
            double weight = data->boneList[boneIndex].weights[weightIndex];
            //とりあえず動くが関節が微妙
            const vec3 posePos = (pose.nowTransforom[boneIndex] - data->boneList[boneIndex].defaultTransform) * weight;
            newPos[vertexIndex] += posePos;
        }
    }
}

void updateFBXObj(int frameIndex, FBXObject* obj, Triangle** triangleList) {
    vec3* h_pointPos = (vec3*)malloc(sizeof(vec3) * obj->mesh->nPoints);
    calcPose(frameIndex, obj, h_pointPos, obj->mesh->idxVertex);
    vec3* d_newPos;
    cudaMalloc(&d_newPos, sizeof(vec3) * obj->mesh->nPoints);
    cudaMemcpy(d_newPos, h_pointPos, obj->mesh->nPoints * sizeof(vec3), cudaMemcpyHostToDevice);
    vec3* d_idxVertices;
    cudaMalloc(&d_idxVertices, sizeof(vec3) * obj->mesh->nTriangles);
    cudaMemcpy(d_idxVertices, obj->mesh->idxVertex, obj->mesh->nTriangles * sizeof(vec3), cudaMemcpyHostToDevice);
    update_pose << <1, 1 >> > (triangleList, d_newPos, d_idxVertices, obj->mesh->nTriangles);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_newPos));
    checkCudaErrors(cudaFree(d_idxVertices));
    free(h_pointPos);
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

__device__ void CalcFBXVertexPos(FBXObject* data, BonePoseData pose,vec3* newPos) 
{
    for (int i = 0; i < data->mesh->nPoints; i++) {
        newPos[i] = data->mesh->points[i];
    }

    for (int boneIndex = 0; boneIndex < data->boneCount; boneIndex++)
    {
        data->boneList[boneIndex].nowTransform = pose.nowTransforom[boneIndex];
        data->boneList[boneIndex].nowRotation = pose.nowRatation[boneIndex];
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
    //create_camera << <1, 1 >> > (camera, nx, ny, vec3(0, 200, 2000), vec3(0, 200, 0), 10.0, 0.0, 40);//dragon
    //create_camera << <1, 1 >> > (camera, nx, ny, vec3(200, 250, 200), vec3(0, 200, 0), 10.0, 0.0, 60);//high_walk
    //create_camera << <1, 1 >> > (camera, nx, ny, vec3(200, 200, 400), vec3(0, 200, 0), 10.0, 0.0, 60);
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

__global__ void add_mesh_withNormal(HitableList** list, Triangle** triangles, vec3* points, vec3* normal, vec3* idxVertex, int nTriangles)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        (*list) = new HitableList(nTriangles);
        Material* mat = new Lambertian(new ConstantTexture(vec3(0.65, 0.05, 0.05)));
        for (int i = 0; i < nTriangles; i++) {
            vec3 idx = idxVertex[i];
            vec3 v[3] = { points[int(idx[2])], points[int(idx[1])], points[int(idx[0])] };
            Transform* transform = new Transform(vec3(0), vec3(0, 0, 0), vec3(1));
            //Triangle* tri = new Triangle(v, normal[i], new Lambertian(new ConstantTexture(vec3(0.65, 0.05, 0.05))), false, transform, true);
            Triangle* tri = new Triangle(v, normal[i], mat, false, transform, true);
            (*list)->list[i] = tri;
            triangles[i] = tri;
        }
    }
}

void create_FBXMesh(HitableList** list, FBXObject* data) 
{
    vec3* d_point;
    cudaMalloc(&d_point, sizeof(vec3) * data->mesh->nPoints);
    cudaMemcpy(d_point, data->mesh->points, data->mesh->nPoints * sizeof(vec3), cudaMemcpyHostToDevice);
    vec3* d_idxVertices;
    cudaMalloc(&d_idxVertices, sizeof(vec3) * data->mesh->nTriangles);
    cudaMemcpy(d_idxVertices, data->mesh->idxVertex, data->mesh->nTriangles * sizeof(vec3), cudaMemcpyHostToDevice);
    vec3* d_normals;
    cudaMalloc(&d_normals, sizeof(vec3) * data->mesh->nTriangles);
    cudaMemcpy(d_normals, data->mesh->normals, data->mesh->nTriangles * sizeof(vec3), cudaMemcpyHostToDevice);
    cudaMalloc(&data->d_triangleData, sizeof(Triangle*) * data->mesh->nTriangles);
    add_mesh_withNormal << <1, 1 >> > (list, data->d_triangleData, d_point, d_normals, d_idxVertices, data->mesh->nTriangles); //メッシュの移動と作成
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_point));
    checkCudaErrors(cudaFree(d_idxVertices));
    checkCudaErrors(cudaFree(d_normals));
}

void create_BVHfromList(BVHNode** bvh,HitableList** list, curandState* curand_state, CudaPointerList* pointerList)
{
    pointerList->append((void**)bvh);
    create_BVH << <1, 1 >> > (list, bvh, curand_state);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__global__ void create_BoneBVH(HitableList** bvh_list, bool* d_hasTriangle, int boneIndex,
    curandState* curand_state, int hasTriangleNum, int nTriangles, vec3 defaultTransform, vec3 nowTransform, Triangle** fbxList)
{
    // ボーンの三角形のHitableListを作成
    HitableList* triangleList = new HitableList(hasTriangleNum);
    int i = 0;
    //三角形がsetに含まれるか判定、含まれてたらリストに入れる
    for (int triangleIndex = 0; triangleIndex < nTriangles; triangleIndex++)
    {
        if (d_hasTriangle[triangleIndex]) {
            triangleList->list[i] = fbxList[triangleIndex];
            i++;
        }
    }

    //リストからBoneBVHNodeを作成する
    BoneBVHNode* node = new BoneBVHNode(triangleList->list, triangleList->list_size, 0, 1, curand_state, defaultTransform, nowTransform, true);
    //BoneBVHNodeをListに追加
    (*bvh_list)->list[boneIndex] = node;
    triangleList->freeMemory();
}




__global__ void create_emptyBoneBVH(HitableList** list,int boneIndex)
{
    //リストからBoneBVHNodeを作成する
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

void createBoneBVH(HitableList** list, FBXObject* fbxData, curandState* curand_state, CudaPointerList* pointerList)
{
    create_List << <1, 1 >> > (list, fbxData->boneCount);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    std::vector<bool> IsTriangleAdded(fbxData->mesh->nTriangles, false);

    // BoneBVHNodeをボーン分作成
    for (int boneIndex = 0; boneIndex < fbxData->boneCount; boneIndex++)
    {
        bool* h_hasTriangle = (bool*)malloc(fbxData->mesh->nTriangles * sizeof(bool));
        //setにボーンの頂点のインデックスを格納
        std::set<int> boneVerticesIndexes;

        for (int weightIndex = 0; weightIndex < fbxData->boneList[boneIndex].weightCount; weightIndex++)
        {
            boneVerticesIndexes.insert(int(fbxData->boneList[boneIndex].weightIndices[weightIndex]));
        }

        int hasTriangleNum = 0;
        //三角形がsetに含まれるか判定、含まれてたらリストに入れる
        for (int triangleIndex = 0; triangleIndex < fbxData->mesh->nTriangles; triangleIndex++)
        {
            h_hasTriangle[triangleIndex] = false;
            if (IsTriangleAdded[triangleIndex])continue;
            vec3 pointsIndex = fbxData->mesh->idxVertex[triangleIndex];
            if (boneVerticesIndexes.find((int)pointsIndex[0]) != boneVerticesIndexes.end()
                && boneVerticesIndexes.find((int)pointsIndex[1]) != boneVerticesIndexes.end()
                && boneVerticesIndexes.find((int)pointsIndex[2]) != boneVerticesIndexes.end())
            {
                h_hasTriangle[triangleIndex] = true;
                IsTriangleAdded[triangleIndex] = true;
                hasTriangleNum++;
            }
        }
        bool* d_hasTriangle;
        cudaMalloc(&d_hasTriangle, fbxData->mesh->nTriangles * sizeof(bool));
        cudaMemcpy(d_hasTriangle, h_hasTriangle, fbxData->mesh->nTriangles * sizeof(bool), cudaMemcpyHostToDevice);

        if (hasTriangleNum == 0) {
            create_emptyBoneBVH << <1, 1 >> > (list, boneIndex);
        }
        else {
            create_BoneBVH << <1, 1 >> > (list, d_hasTriangle, boneIndex,
                curand_state, hasTriangleNum, fbxData->mesh->nTriangles, fbxData->boneList[boneIndex].defaultTransform, fbxData->boneList[boneIndex].nowTransform, fbxData->d_triangleData);
        }

        CHECK(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        cudaFree(d_hasTriangle);
        free(h_hasTriangle);
    }

}