#define _USE_MATH_DEFINES

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
#define SAMPLES 100

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

        random_scene(list, world, state);

        vec3 lookfrom(0, 0, 10);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        float aperture = 0.0;
        float vfov = 60.0;

        *camera = new MotionCamera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            vfov,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.0,
            1.0);
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

        random_scene(mesh, triangles, state);
        //draw_one_mesh(mesh, triangles, points, idxVertex, np, nt, state);
        // bunny_inside_cornell_box(mesh, triangles, points, idxVertex, np, nt, state);

        vec3 lookfrom(0, 0, 10);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        float aperture = 0.0;
        float vfov = 60.0;

        *camera = new MotionCamera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            vfov,
            float(nx) / float(ny),
            aperture,
            dist_to_focus,
            0.0,
            1.0);
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
    printf("%d,%d 完了\n", x, y);
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    //col[0] = float(x) / float(nx);
    //col[1] = float(y) / float(ny);
    //col[2] = 0.5f;
    
    colorBuffer[pixel_index] = clip(col);
}

__global__ void renderTest(vec3* colorBuffer,
    curandState* state,
    int nx,
    int ny) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= nx) || (y >= ny)) return;

    int pixel_index = y * nx + x;
    vec3 col(0, 0, 0);

    col[0] = float(x) / float(nx);
    col[1] = float(y) / float(ny);
    col[2] = 0.5f;

    colorBuffer[pixel_index] = clip(col);
}


cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

/*void load_mesh() {
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

    // 大きくしてる？
    for (int i = 0; i < nPoints; i++) { points[i] *= 30.0; }
    for (int i = 0; i < nPoints; i++) { std::cout << points[i] << std::endl; }


    Hitable** triangles;
    checkCudaErrors(cudaMallocManaged((void**)&triangles, nTriangles * sizeof(Hitable*)));
    // --------------------------- ! allocate the mesh ---------------------------------------
}*/

int main()
{
    std::ofstream imgWrite("images/image.ppm");

    int nx = 1024 * RESOLUTION;
    int ny = 512 * RESOLUTION;
    int tx = 16;
    int ty = 16;
    int max_depth = 15;

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

    //load_mesh();
    

    // オブジェクト、カメラの生成
    build_random_world << <1, 1 >> > (obj_list, world, camera, curand_state, nx, ny);
    //build_mesh << <1, 1 >> > (world, camera, triangles, points,idxVertex, nPoints, nTriangles, curand_state, nx, ny, obj_cnt);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    

    // レンダリング
    render << <blocks, threads >> > (colorBuffer, world, camera, curand_state, nx, ny, SAMPLES,max_depth);
    //renderTest <<<blocks,threads>>> (colorBuffer,curand_state, nx, ny);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // 画像書き出し
    imgWrite << "P3\n" << nx << " " << ny << "\n255\n";
    for (int i = ny - 1; i >= 0; i--) {
        for (int j = 0; j < nx; j++) {
            size_t pixel_index = i * nx + j;
            int ir = int(255.99 * colorBuffer[pixel_index].r());
            int ig = int(255.99 * colorBuffer[pixel_index].g());
            int ib = int(255.99 * colorBuffer[pixel_index].b());
            imgWrite << ir << " " << ig << " " << ib << "\n";
        }
    }
    
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

    


    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
