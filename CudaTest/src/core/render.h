#pragma once
#define _USE_MATH_DEFINES
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../stb-master/stb_image_write.h"

#include <cstddef>
#include <memory>
#include <new>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>


#include "../test_scene.h"

#include "../Loader/FbxLoader.h"


#define RESOLUTION 1

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

#define CHECK(call) {const cudaError_t error = call;  if (error != cudaSuccess)  { printf("Error: %s:%d, ", __FILE__, __LINE__); printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); exit(1); } }

void check_cuda(cudaError_t result,
    char const* const func,
    const char* const file,
    int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void destroy(HitableList** world,
    Camera** camera, TransformList** transformPointer) {

    (*world)->freeMemory();
    (*transformPointer)->freeMemory();
    delete* world;
    delete* camera;
    delete* transformPointer;
    
}

__global__ void random_init(int nx,
    int ny,
    curandState* state) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= nx) || (y >= ny)) return;
    int pixel_index = y * nx + x;
    curand_init(0, pixel_index, 0, &state[pixel_index]);
}

struct RGB {
    unsigned char r, g, b, a; //赤, 緑, 青, 透過
    RGB() = default;
    constexpr RGB(const unsigned char r_, const unsigned char g_, const unsigned char b_, const unsigned char a_) :r(r_), g(g_), b(b_), a(a_) {}
};

__device__ vec3 backgroundSky(const vec3& d)
{
    vec3 v = unit_vector(d);
    float t = 0.5f * (v[1] + 1.0f);
    return lerp(t, vec3(1), vec3(0.5f, 0.7f, 1.0f));
}

__device__ vec3 shade(const Ray& r,
    HitableList** world,
    int depth,
    curandState* state, int frameIndex) {
    HitRecord rec;
    if ((*world)->hit(r, 0.001, FLT_MAX, rec, frameIndex)) {
        Ray scattered;
        vec3 attenuation;
        vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
        if (depth > 0 && rec.mat_ptr->scatter(r, rec, attenuation, scattered, state)) {
            return emitted + attenuation * shade(scattered, world, depth - 1, state, frameIndex);
        }
        else {
            return emitted + vec3(0.1, 0.1, 0.1);
        }
    }
    else {
        return backgroundSky(r.direction());
    }
}

//ランバートシェードでのテスト
__device__ vec3 LambertShade(const Ray& r,
    HitableList** world,
    int depth,
    curandState* state, int frameIndex) {
    HitRecord rec;
    if ((*world)->hit(r, 0.001, FLT_MAX, rec, frameIndex)) {
        Ray scattered;
        vec3 attenuation;
        vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
        rec.mat_ptr->scatter(r, rec, attenuation, scattered, state);
        float t = dot(r.direction(), rec.normal);
        if (t < 0)t = 0;
        return attenuation * t * backgroundSky(r.direction()) * 0.2 + emitted;
    }
    else {
        return backgroundSky(r.direction());
    }
}

// 法線のテスト
__device__ vec3 shade_normal(const Ray& r,
    HitableList** world,
    int depth,
    curandState* state, int frameIndex) {
    HitRecord rec;
    if ((*world)->hit(r, 0.001, FLT_MAX, rec, frameIndex)) {
        Ray scattered;
        vec3 attenuation;
        return rec.normal;
    }
    else {
        return backgroundSky(r.direction());
    }
}

__global__ void render(vec3* colorBuffer,
    HitableList** world,
    Camera** camera,
    curandState* state,
    int nx,
    int ny,
    int samples,
    int max_depth,
    int frameIndex) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= nx) || (y >= ny)) return;

    int pixel_index = y * nx + x;

    int ns = samples;
    vec3 col(0, 0, 0);
    for (int i = 0; i < ns; i++) {
        float u = float(x + curand_uniform(&(state[pixel_index]))) / float(nx);
        float v = float(y + curand_uniform(&(state[pixel_index]))) / float(ny);
        Ray r = (*camera)->get_ray(u, v, state);
        col += shade(r, world, max_depth, &(state[pixel_index]), frameIndex);
        //col += LambertShade(r, world, max_depth, &(state[pixel_index]),frameIndex);
        //col += shade_normal(r, world, 0, &(state[pixel_index]),frameIndex);
    }
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);

    colorBuffer[pixel_index] = clip(col);
}

__global__ void SetTransform(Transform transform, TransformList** transformPointer,int i) {
    *((*transformPointer)->list[i]) = transform;
}

void renderAnimation(int nx,int ny,int samples,int max_depth,int minFrame,int maxFrame,
    vec3* colorBuffer, HitableList** world,  Camera** camera, AnimationDataList* animationData, TransformList** transformPointer,
    dim3 blocks, dim3 threads, curandState* curand_state) {
    // レンダリング
    for (int frameIndex = 0; frameIndex <= maxFrame; frameIndex++)
    {
        // 位置更新処理
        for (int i = 0; i < animationData->list_size; i++)
        {
            SetTransform << <1, 1 >> > (animationData->list[i]->Get_NextTransform(frameIndex), transformPointer, i);
            animationData->list[i]->SetNext(frameIndex);
        }

        render << <blocks, threads >> > (colorBuffer, world, camera, curand_state, nx, ny, samples, max_depth, frameIndex);
        CHECK(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        //png書き出し
        RGB* rgb = new RGB[nx * ny];
        for (int i = 0; i < ny; i++) {
            for (int j = 0; j < nx; j++) {
                size_t pixel_index = (ny - 1 - i) * nx + j;
                rgb[i * nx + j].r = char(255.99 * colorBuffer[pixel_index].r());
                rgb[i * nx + j].g = char(255.99 * colorBuffer[pixel_index].g());
                rgb[i * nx + j].b = char(255.99 * colorBuffer[pixel_index].b());
                rgb[i * nx + j].a = 255;
            }
        }

        char* folderPath = "images/moveTest/picture_";
        char* pathname = new char[strlen(folderPath) + sizeof(frameIndex) + strlen(".png") + 1];
        strcpy(pathname, folderPath);
        sprintf(pathname + strlen(folderPath), "%d", frameIndex);
        strcat(pathname, ".png");
        stbi_write_png(pathname, nx, ny, sizeof(RGB), rgb, 0);

        printf("%dフレーム目:画像書き出し\n", frameIndex);
        delete[] pathname; 
    }
}

void BuildAnimatedSphere(HitableList** world, AnimationDataList* animationData, TransformList** transformPointer) {
    add_object << <1, 1 >> > (world, transformPointer);

    //アニメーション準備
    KeyFrameList* keyFrames = new KeyFrameList();
    keyFrames->append(new KeyFrame(0, new Transform(vec3(0, 1, 0), vec3(0), vec3(1))));
    keyFrames->append(new KeyFrame(3, new Transform(vec3(0, 5, 0), vec3(0), vec3(1.5))));
    KeyFrameList* keyFrames2 = new KeyFrameList();
    keyFrames2->append(new KeyFrame(0, new Transform(vec3(-4, 1, 0), vec3(0), vec3(1))));
    keyFrames2->append(new KeyFrame(3, new Transform(vec3(-4, -5, 0), vec3(0), vec3(10))));
    KeyFrameList* keyFrames3 = new KeyFrameList();
    keyFrames3->append(new KeyFrame(0, new Transform(vec3(4, 1, 0), vec3(0), vec3(1))));

    animationData->append(new AnimationData(keyFrames));
    animationData->append(new AnimationData(keyFrames2));
    animationData->append(new AnimationData(keyFrames3));
}

void AddFBXMesh(const std::string& filePath, HitableList** world, AnimationDataList* animationData, TransformList** transformPointer)
{
    vec3* points;
    vec3* idxVertex;
    vec3* normals;

    checkCudaErrors(cudaMallocManaged((void**)&points, 2600 * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged((void**)&idxVertex, 5000 * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged((void**)&normals, 5000 * sizeof(vec3)));

    int nPoints, nTriangles;
    if (!FBXLoad(filePath, points, idxVertex, normals, nPoints, nTriangles))
    {
        std::cout << "fbx load failed" << std::endl;
        return;
    }

    add_mesh_withNormal << <1, 1 >> > (world, points, idxVertex, normals, nPoints, nTriangles, transformPointer);
}

void BuildSceneData(HitableList** world, Camera** camera,AnimationDataList* animationData, TransformList** transformPointer, int nx, int ny)
{
    create_camera << <1, 1 >> > (camera, nx, ny, vec3(0, 0, 20),vec3(0, 0, 0), 10.0,0.0,60);

    init_data << <1, 1 >> > (world, transformPointer);
    BuildAnimatedSphere(world, animationData, transformPointer);
    //AddFBXMesh("./objects/bunny2.fbx",world, animationData, transformPointer);

    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("シーン作成完了\n");
}


int BuildRandomWorld(Hitable** world, Hitable** obj_list, Camera** camera, curandState* state, int nx, int ny)
{
    int obj_cnt = 488;
    checkCudaErrors(cudaMallocManaged((void**)&obj_list, obj_cnt * sizeof(Hitable*)));

    create_camera << <1, 1 >> > (camera, nx, ny, vec3(0, 0, 20), vec3(0, 0, 0), 10.0, 0.0, 60);
    random_scene << <1, 1 >> > (world, obj_list, state);

    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
     
    printf("シーン作成完了\n");

    return obj_cnt;
}

/*
int BuildMesh(Hitable** world, Hitable** obj_list, Camera** camera, curandState* state, int nx, int ny)
{
    vec3* points;
    vec3* idxVertex;

    checkCudaErrors(cudaMallocManaged((void**)&points, 2600 * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged((void**)&idxVertex, 5000 * sizeof(vec3)));

    int nPoints, nTriangles;
    parseObjByName("./objects/small_bunny.obj", points, idxVertex, nPoints, nTriangles);

    std::cout << "# of points: " << nPoints << std::endl;
    std::cout << "# of triangles: " << nTriangles << std::endl;

    // scale
    for (int i = 0; i < nPoints; i++) { points[i] *= 100.0; }
    //for (int i = 0; i < nPoints; i++) { std::cout << points[i] << std::endl; }

    int obj_cnt = nTriangles + 10;
    printf("obj_cnt %d\n", obj_cnt);
    checkCudaErrors(cudaMallocManaged((void**)&obj_list, obj_cnt * sizeof(Hitable*)));

    create_camera_origin << <1, 1 >> > (camera, nx, ny);
    draw_one_mesh_withoutBVH << <1, 1 >> > (world, obj_list, points, idxVertex, nPoints, nTriangles, state);

    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    printf("シーン作成完了\n");

    return obj_cnt;
}
*/


int BuildCornellBox(Hitable** world, Hitable** obj_list, Camera** camera, curandState* state, int nx, int ny)
{
    int obj_cnt = 2;
    checkCudaErrors(cudaMallocManaged((void**)&obj_list, obj_cnt * sizeof(Hitable*)));
    //    create_camera_for_cornelbox << <1, 1 >> > (camera, nx, ny);
    create_camera << <1, 1 >> > (camera, nx, ny, vec3(278, 278, -700), vec3(278, 278, 0), 10.0, 0.0, 40);
    cornell_box_scene << <1, 1 >> > (world, obj_list, state);

    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    printf("シーン作成完了\n");

    return obj_cnt;
}
/*
int BuildBVHTest(Hitable** world, Hitable** obj_list, Camera** camera, curandState* state, int nx, int ny) {
    vec3* points;
    vec3* idxVertex;

    checkCudaErrors(cudaMallocManaged((void**)&points, 2600 * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged((void**)&idxVertex, 5000 * sizeof(vec3)));

    int nPoints, nTriangles;
    parseObjByName("./objects/small_bunny.obj", points, idxVertex, nPoints, nTriangles);

    std::cout << "# of points: " << nPoints << std::endl;
    std::cout << "# of triangles: " << nTriangles << std::endl;

    // scale
    for (int i = 0; i < nPoints; i++) { points[i] *= 100.0; }
    for (int i = 0; i < nPoints; i++) { std::cout << points[i] << std::endl; }

    int obj_cnt = nTriangles + 10;
    printf("obj_cnt %d\n", obj_cnt);
    checkCudaErrors(cudaMallocManaged((void**)&obj_list, obj_cnt * sizeof(Hitable*)));


    //create_camera_for_cornelbox << <1, 1 >> > (camera, nx, ny);
    //bunny_inside_cornell_box << <1, 1 >> > (world, obj_list, points,idxVertex, nPoints, nTriangles, state);

    create_camera_origin << <1, 1 >> > (camera, nx, ny);
    draw_one_mesh << <1, 1 >> > (world, obj_list, points, idxVertex, nPoints, nTriangles, state);

    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    printf("シーン作成完了\n");

    return obj_cnt;
}
*/

