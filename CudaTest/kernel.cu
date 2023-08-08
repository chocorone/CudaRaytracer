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

#include "src/core/hitable.h"
#include "src/core/camera.h"
#include "src/mesh/obj_loader.h"

#define RESOLUTION 1
#define SAMPLES 100

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)


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

int main()
{
    std::ofstream imgWrite("images/image.ppm");

    int nx = 1024 * RESOLUTION;
    int ny = 512 * RESOLUTION;
    int tx = 16;
    int ty = 16;

    int num_pixel = nx * ny;


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

    // 大きくしてる？
    for (int i = 0; i < nPoints; i++) { points[i] *= 30.0; }
    for (int i = 0; i < nPoints; i++) { std::cout << points[i] << std::endl; }

    /*
    Hitable** triangles;
    checkCudaErrors(cudaMallocManaged((void**)&triangles, nTriangles * sizeof(Hitable*)));
    // --------------------------- ! allocate the mesh ---------------------------------------

    // オブジェクト、カメラの生成
    build_mesh << <1, 1 >> > (world, camera, triangles, points,
        idxVertex, nPoints, nTriangles, curand_state, nx, ny, obj_cnt);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    */

    // レンダリング
    //render << <blocks, threads >> > (colorBuffer, world, camera, curand_state, nx, ny, SAMPLES);
    renderTest <<<blocks,threads>>> (colorBuffer,curand_state, nx, ny);
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
