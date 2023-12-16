#pragma once

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

void ChangeHeapSize(size_t heapSize) {
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
    cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);
    printf("Heap Size=%ld\n", heapSize);

}

void ChangeStackSize(size_t stackSize){
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    printf("Stack Size=%ld\n", stackSize);
}

__global__ void destroy(HitableList** world,
    Camera** camera, TransformList** transformPointer) {

    (*world)->freeMemory();
    (*transformPointer)->freeMemory();
    delete* world;
    delete* camera;
    delete* transformPointer;

}

__global__ void destroy(MeshData* meshData) {

    delete meshData->points;
    delete meshData->normals;
    delete meshData->idxVertex;

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