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


#include "../createScene.h"

#include "../Loader/FbxLoader.h"
#include "core/deviceManage.h"

#define RESOLUTION 1


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
    Hitable** world,
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
    Hitable** world,
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
    Hitable** world,
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

__global__ void render(vec3* colorBuffer, Hitable** world,Camera** camera,curandState* state,
    int nx,int ny,int samples,int max_depth,int frameIndex) {
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
        //col += shade(r, world, max_depth, &(state[pixel_index]), frameIndex);
        col += LambertShade(r, world, max_depth, &(state[pixel_index]),frameIndex);
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

void renderAnimation(int nx,int ny,int samples,int max_depth,int beginFrame,int endFrame,
    Hitable** world,  Camera** camera, AnimationDataList* animationData, TransformList** transformPointer, FBXAnimationData* fbxAnimationData,
    dim3 blocks, dim3 threads, curandState* curand_state) {

    // 画素のメモリ確保
    const int num_pixel = nx * ny;
    vec3* colorBuffer = (vec3*)malloc(nx * ny * sizeof(vec3));
    for (int i = 0; i < nx * ny; i++)
    {
        colorBuffer[i] = vec3(0);
    }
    vec3* d_colorBuffer;
    cudaMalloc(&d_colorBuffer, nx * ny * sizeof(vec3));
    cudaMemcpy(d_colorBuffer, colorBuffer, nx * ny * sizeof(vec3), cudaMemcpyHostToDevice);
    // レンダリング
    for (int frameIndex = beginFrame; frameIndex <= endFrame; frameIndex++)
    {
        // 位置更新処理
        /*for (int i = 0; i < animationData->list_size; i++)
        {
            SetTransform << <1, 1 >> > (animationData->list[i]->Get_NextTransform(frameIndex), transformPointer, i);
            animationData->list[i]->SetNext(frameIndex);
        }*/

        //メッシュの位置の更新
        update_mesh_fromPoseData << <1, 1 >> > (fbxAnimationData->object, fbxAnimationData->animation[frameIndex], frameIndex);
        CHECK(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        //BVHの更新
        UpdateBVH << <1, 1 >> > ((BVHNode**)world);
        CHECK(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        printf("BVH更新完了\n");


        render << <blocks, threads >> > (d_colorBuffer, world, camera, curand_state, nx, ny, samples, max_depth, frameIndex);
        CHECK(cudaDeviceSynchronize());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        cudaMemcpy(colorBuffer, d_colorBuffer, nx * ny * sizeof(vec3), cudaMemcpyDeviceToHost);

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
    checkCudaErrors(cudaFree(d_colorBuffer));
    free(colorBuffer);
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


