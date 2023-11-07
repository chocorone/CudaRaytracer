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

#include "mesh/obj_loader.h"
#include "test_scene.h"

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

__global__ void build_random_world(Hitable** list,
    Hitable** world,
    Camera** camera,
    curandState* state,
    int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        //random_scene(list, world, state);
        cornell_box_scene(list,world);

        vec3 lookfrom(278, 278, -700);
        vec3 lookat(278, 278, 0);
        float dist_to_focus = 10.0;
        float aperture = 0.0;
        float vfov = 40.0;

        *camera = new Camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            vfov,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
    }
}

__global__ void build_mesh(Hitable** mesh,
    Camera** camera,
    Hitable** triangles,
    vec3* points,
    vec3* idxVertex,
    int np, int nt,
    curandState* state,
    int nx, int ny, int cnt) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {

        //draw_one_mesh(mesh, triangles, points, idxVertex, np, nt, state);
        bunny_inside_cornell_box(mesh, triangles, points, idxVertex, np, nt, state);

        vec3 lookfrom(278, 278, -700);
        vec3 lookat(278, 278, 0);
        float dist_to_focus = 10.0;
        float aperture = 0.0;
        float vfov = 40.0;

        *camera = new Camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            vfov,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
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
    //return vec3(0.5f, 0.7f, 1.0f);
}

__device__ vec3 shade(const Ray& r,
    Hitable** world,
    int depth,
    curandState* state) {
    HitRecord rec;
    if ((*world)->hit(r, 0.001, FLT_MAX, rec)) {
        Ray scattered;
        vec3 attenuation;
        vec3 emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);
        if (depth > 0 && rec.mat_ptr->scatter(r, rec, attenuation, scattered, state)) {
            return emitted + attenuation * shade(scattered, world, depth -1, state);
        }
        else {
            return emitted;
        }
        /*if (rec.mat_ptr->scatter(r, rec, attenuation, scattered, state)) {
            return attenuation;
        }
        else {
            return vec3(1, 1, 1);
        }*/
    }
    else {
        return backgroundSky(r.direction());
    }
}

__device__ vec3 shade_nolight(const Ray& r,
    Hitable** world,
    int depth,
    curandState* state) {
    HitRecord rec;
    if ((*world)->hit(r, 0.001, FLT_MAX, rec)) {
        Ray scattered;
        vec3 attenuation;
        if (depth > 0&& rec.mat_ptr->scatter(r, rec, attenuation, scattered, state)) {
            return attenuation * shade_nolight(scattered, world, depth - 1, state);
        }
        else {
            return vec3(0, 0, 0);
        }
    }
    else {
        return vec3(1.0, 1.0, 1.0);
    }
}

__global__ void render(vec3* colorBuffer,
    Hitable** world,
    Camera** camera,
    curandState* state,
    int nx,
    int ny,
    int samples,
    int max_depth) {
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
        col += shade(r, world, max_depth, &(state[pixel_index]));
        //col += shade_nolight(r, world, 0, &(state[pixel_index]));
        
    }
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    
    colorBuffer[pixel_index] = clip(col);
}


void AllocateMesh(Hitable** mesh,
    Camera** camera, 
    vec3* points,
    vec3* idxVertex) {

    checkCudaErrors(cudaMallocManaged((void**)&points, 2600 * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged((void**)&idxVertex, 5000 * sizeof(vec3)));

    int nPoints, nTriangles;
    parseObjByName("./objects/small_bunny.obj", points, idxVertex, nPoints, nTriangles);

    std::cout << "# of points: " << nPoints << std::endl;
    std::cout << "# of triangles: " << nTriangles << std::endl;

    // 大きくしてる
    for (int i = 0; i < nPoints; i++) { points[i] *= 30.0; }
    //for (int i = 0; i < nPoints; i++) { std::cout << points[i] << std::endl; }
    

    Hitable** triangles;
    checkCudaErrors(cudaMallocManaged((void**)&triangles, nTriangles * sizeof(Hitable*)));
}

struct RGB {
    unsigned char r, g, b, a; //赤, 緑, 青, 透過
    RGB() = default;
    constexpr RGB(const unsigned char r_, const unsigned char g_, const unsigned char b_, const unsigned char a_) :r(r_), g(g_), b(b_), a(a_) {}
};

int main()
{
    int nx = 1024 * RESOLUTION;
    int ny = 512 * RESOLUTION;
    int tx = 16;
    int ty = 16;
    int max_depth = 10;
    int samples = 50;

    int num_pixel = nx * ny;

    size_t heapSize = 1024*1024*1024;
    size_t stackSize = 4096;

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);

    //確認
    cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);
    printf("Heap Size=%ld\n", heapSize);
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    printf("Stack Size=%ld\n", stackSize);

    // 画素のメモリ確保
    vec3* colorBuffer;

    checkCudaErrors(cudaMallocManaged((void**)&colorBuffer, num_pixel * sizeof(vec3)));

    // 乱数列生成用のメモリ確保
    curandState* curand_state;
    checkCudaErrors(cudaMallocManaged((void**)&curand_state, num_pixel * sizeof(curandState)));

    // シーン作成
    int obj_cnt = 488;
    Hitable** obj_list;
    Hitable** world;
    Camera** camera;
    checkCudaErrors(cudaMallocManaged((void**)&obj_list, obj_cnt * sizeof(Hitable*)));
    checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(Hitable*)));
    checkCudaErrors(cudaMallocManaged((void**)&camera, sizeof(Camera*)));

    
    // 画素ごとに乱数を初期化
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    random_init << <blocks, threads >> > (nx, ny, curand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    // --------------------------- allocate the mesh ----------------------------------------
    vec3* points;
    vec3* idxVertex;

    // NOTE: must pre-allocate before initialize the elements
    checkCudaErrors(cudaMallocManaged((void**)&points, 2600 * sizeof(vec3)));
    checkCudaErrors(cudaMallocManaged((void**)&idxVertex, 5000 * sizeof(vec3)));

    int nPoints, nTriangles;
    parseObjByName("./objects/small_bunny.obj", points, idxVertex, nPoints, nTriangles);

    std::cout << "# of points: " << nPoints << std::endl;
    std::cout << "# of triangles: " << nTriangles << std::endl;

    // scale
    for (int i = 0; i < nPoints; i++) { points[i] *= 30.0; }
    for (int i = 0; i < nPoints; i++) { std::cout << points[i] << std::endl; }

    Hitable** triangles;
    checkCudaErrors(cudaMallocManaged((void**)&triangles, nTriangles * sizeof(Hitable*)));
    // --------------------------- ! allocate the mesh ---------------------------------------

    // オブジェクト、カメラの生成
    //build_random_world << <1, 1 >> > (obj_list, world, camera, curand_state, nx, ny);

    //AllocateMesh(obj_list, camera, curand_state, nx, ny, obj_cnt);
    build_mesh << <1, 1 >> > (world, camera, triangles, points,
        idxVertex, nPoints, nTriangles, curand_state, nx, ny, obj_cnt);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    printf("シーン作成完了\n");
    

    // レンダリング
    render <<<blocks, threads >>> (colorBuffer, world, camera, curand_state, nx, ny, samples,max_depth);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    printf("レンダリング終了\n");

    //png書き出し
    RGB* rgb = new RGB[nx * ny];
    for (int i = 0; i < ny ; i++) {
        for (int j = 0; j < nx; j++) {
            size_t pixel_index = (ny - 1 - i) * nx + j;
            rgb[i * nx + j].r = char(255.99 * colorBuffer[pixel_index].r());
            rgb[i * nx + j].g = char(255.99 * colorBuffer[pixel_index].g());
            rgb[i * nx + j].b = char(255.99 * colorBuffer[pixel_index].b());
            rgb[i * nx + j].a = 255;
        }
    }

    stbi_write_png("images/picture_1.png",nx, ny,sizeof(RGB), rgb,0);

    printf("画像書き出し\n");
    
    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    destroy << <1, 1 >> > (obj_list, world, camera, obj_cnt);

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

