#define _USE_MATH_DEFINES
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb-master/stb_image_write.h"

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


#include "test_scene.h"

#include "Loader/FbxLoader.h"


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



__global__ void random_init(int nx,
    int ny,
    curandState* state) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= nx) || (y >= ny)) return;
    int pixel_index = y * nx + x;
    curand_init(0, pixel_index, 0, &state[pixel_index]);
}

__global__ void destroy(Hitable** obj_list,
    Hitable** world,
    Camera** camera,
    int obj_cnt) {
    for (int i = 0; i < obj_cnt; i++) {
        delete* (obj_list + i);
    }
    delete* world;
    delete* camera;
}

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
            return emitted + attenuation * shade(scattered, world, depth -1, state,frameIndex);
        }
        else {
            return emitted+vec3(0.1,0.1,0.1);
        }
    }
    else {
        //if (depth != 16)printf("depth:%d\n", depth);
        return backgroundSky(r.direction());
    }
}

//ランバートシェードでのテスト
__device__ vec3 LambertShade(const Ray& r,
    Hitable** world,
    int depth,
    curandState* state,int frameIndex) {
    HitRecord rec;
    if ((*world)->hit(r, 0.001, FLT_MAX, rec, frameIndex)) {
        Ray scattered;
        vec3 attenuation;
        vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
        rec.mat_ptr->scatter(r, rec, attenuation, scattered, state);
        float t = dot(r.direction(), rec.normal);
        if (t < 0)t = 0;
        return attenuation * t * backgroundSky(r.direction()) * 0.2+ emitted;
    }
    else {
        return backgroundSky(r.direction());
    }
}

__device__ vec3 shade_normal(const Ray& r,
    Hitable** world,
    int depth,
    curandState* state, int frameIndex) {
    HitRecord rec;
    if ((*world)->hit(r, 0.001, FLT_MAX, rec,frameIndex)) {
        Ray scattered;
        vec3 attenuation;
        return rec.normal;
    }
    else {
        return backgroundSky(r.direction());
    }
}

__global__ void render(vec3* colorBuffer,
    Hitable** world,
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
        //col += shade(r, world, max_depth, &(state[pixel_index]),frameIndex);
        col += LambertShade(r, world, max_depth, &(state[pixel_index]),frameIndex);
        //col += shade_normal(r, world, 0, &(state[pixel_index]),frameIndex);
    }
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    
    colorBuffer[pixel_index] = clip(col);
}

int BuildRandomWorld(Hitable** world, Hitable** obj_list, Camera** camera, curandState* state, int nx, int ny)
{
    int obj_cnt = 488;
    checkCudaErrors(cudaMallocManaged((void**)&obj_list, obj_cnt * sizeof(Hitable*)));

    create_camera_origin << <1, 1 >> > (camera, nx, ny);
    random_scene << <1, 1 >> > (world, obj_list, state);

    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    printf("シーン作成完了\n");

    return obj_cnt;
}


int BuildFBXMesh(Hitable** world, Hitable** obj_list, Camera** camera, curandState* state, int nx, int ny)
{
    vec3* points;
    vec3* idxVertex;
    vec3* normals;

    checkCudaErrors(cudaMallocManaged((void**)&points, 2600 * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged((void**)&idxVertex, 5000 * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged((void**)&normals, 5000 * sizeof(vec3)));

    int nPoints, nTriangles;
    if (!FBXLoad("./objects/bunny2.fbx", points, idxVertex,normals, nPoints, nTriangles)) 
    {
        std::cout << "fbx load failed" << std::endl;
        return 0;
    }

    int obj_cnt = nTriangles + 10;
    printf("obj_cnt %d\n", obj_cnt);
    checkCudaErrors(cudaMallocManaged((void**)&obj_list, obj_cnt * sizeof(Hitable*)));

    create_camera_origin << <1, 1 >> > (camera, nx, ny);
    draw_one_mesh_withNormal << <1, 1 >> > (world, obj_list, points, idxVertex, normals, nPoints, nTriangles, state);

    //create_camera_for_cornelbox << <1, 1 >> > (camera, nx, ny);
    //bunny_inside_cornell_box << <1, 1 >> > (world, obj_list, points, idxVertex, normals, nPoints, nTriangles, state);
    //cornell_box_scene << <1, 1 >> > (world, obj_list, state);

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
    create_camera_origin << <1, 1 >> > (camera, nx, ny);
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
struct RGB {
    unsigned char r, g, b, a; //赤, 緑, 青, 透過
    RGB() = default;
    constexpr RGB(const unsigned char r_, const unsigned char g_, const unsigned char b_, const unsigned char a_) :r(r_), g(g_), b(b_), a(a_) {}
};

int main()
{
    // パラメーター設定
    const int nx = 1024 * RESOLUTION;
    const int ny = 512 * RESOLUTION;  
    const int tx = 16;
    const int ty = 16;
    const int max_depth = 8;
    const int samples = 4;

    const int maxFrame = 2;

    //ヒープサイズ・スタックサイズ指定
    size_t heapSize = 1024 * 1024 * 1024;
    size_t stackSize = 4096 * 2;

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
    cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);
    printf("Heap Size=%ld\n", heapSize);
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    printf("Stack Size=%ld\n", stackSize);

    const int num_pixel = nx * ny;


    // 画素のメモリ確保
    vec3* colorBuffer;
    checkCudaErrors(cudaMallocManaged((void**)&colorBuffer, num_pixel * sizeof(vec3)));

    // 乱数列生成用のメモリ確保
    curandState* curand_state;
    checkCudaErrors(cudaMallocManaged((void**)&curand_state, num_pixel * sizeof(curandState)));

    //シーン保存用の変数のメモリ確保
    Hitable** world;
    Camera** camera;
    Hitable** obj_list;
    checkCudaErrors(cudaMallocManaged((void**)&obj_list, 10 * sizeof(Hitable*)));
    checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(Hitable*)));
    checkCudaErrors(cudaMallocManaged((void**)&camera, sizeof(Camera*)));

    
    // 画素ごとに乱数を初期化
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    random_init <<<blocks, threads >>> (nx, ny, curand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    
    // オブジェクト、カメラの生成
    // ランダムな球
    //int obj_count = BuildRandomWorld(world,obj_list,camera, curand_state,nx, ny);
    // カーネルボックス
    // int obj_count = BuildCornellBox(world, obj_list, camera, curand_state, nx, ny);
    // objのテスト（BVHなし）
    //int obj_count = BuildMesh(world, obj_list, camera, curand_state, nx, ny);
    // obj+BVHのテスト
    //int obj_count = BuildBVHTest(world, obj_list, camera, curand_state, nx, ny);
    int obj_count = BuildFBXMesh(world, obj_list, camera, curand_state, nx, ny);

    
    // レンダリング
    for (int frameIndex = 0; frameIndex < maxFrame; frameIndex++)
    {
        render << <blocks, threads >> > (colorBuffer, world, camera, curand_state, nx, ny, samples, max_depth,frameIndex);
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

        char* path = "images/moveTest/picture_";
        char* png = " .png";
        char* D = new char[strlen(path) + sizeof(frameIndex) + strlen(png) + 1];
        strcpy(D, path);
        sprintf(D + strlen(path), "%d", frameIndex);
        strcat(D, png);
        stbi_write_png(D, nx, ny, sizeof(RGB), rgb, 0);

        printf("%dフレーム目:画像書き出し\n", frameIndex);
        delete[] D;
    }


   
    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    destroy << <1, 1 >> > (obj_list, world, camera,obj_count);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(obj_list));
    checkCudaErrors(cudaFree(camera));
    checkCudaErrors(cudaFree(curand_state));
    checkCudaErrors(cudaFree(colorBuffer));
    

    cudaDeviceReset();

    printf("実行完了\n");

    return 0;
}

