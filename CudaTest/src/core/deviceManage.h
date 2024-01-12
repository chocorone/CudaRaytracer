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


__global__ void destroy(Camera** camera) {

    delete* camera;

}
__global__ void destroy(HitableList** world) {

    (*world)->freeMemory();
    delete* world;

}
__global__ void destroy(TransformList** transformPointer) {

    (*transformPointer)->freeMemory();
    delete* transformPointer;

}



class CudaPointerList
{
public:
    CudaPointerList() { list = new void* (); list_size = 0; }
    CudaPointerList(void** l, int n) { list = l; list_size = n; }
    void append(void** data)
    {
        void** tmp = (void**)malloc(sizeof(void*) * list_size);

        for (int i = 0; i < list_size; i++)
        {
            tmp[i] = list[i];
        }

        free(list);

        list_size++;

        list = (void**)malloc(sizeof(void*) * list_size);

        for (int i = 0; i < list_size - 1; i++)
        {
            list[i] = tmp[i];
        }
        list[list_size - 1] = data;

        free(tmp);
    }
    void freeMemory()
    {
        for (int i = 0; i < list_size; i++)
        {
            printf("free %d\n", i);

            if (typeid(list[i]) == typeid(HitableList**)) 
            {
                destroy << <1, 1 >> > ((HitableList**)list[i]);
                CHECK(cudaDeviceSynchronize());
                checkCudaErrors(cudaGetLastError());
            }
            if (typeid(list[i]) == typeid(TransformList**))
            {
                destroy << <1, 1 >> > ((TransformList**)list[i]);
                CHECK(cudaDeviceSynchronize());
                checkCudaErrors(cudaGetLastError());
            }

            if (typeid(list[i]) == typeid(Camera**))
            {
                destroy << <1, 1 >> > ((Camera**)list[i]);
                CHECK(cudaDeviceSynchronize());
                checkCudaErrors(cudaGetLastError());
            }

            checkCudaErrors(cudaFree(list[i]));
            CHECK(cudaDeviceSynchronize());
            checkCudaErrors(cudaGetLastError());

        }
        free(list);
        list_size = 0;
    }
    void** list;
    int list_size;
};


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

__global__ void random_init(int nx,
    int ny,
    curandState* state) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= nx) || (y >= ny)) return;
    int pixel_index = y * nx + x;
    curand_init(0, pixel_index, 0, &state[pixel_index]);
}

void SetCurandState(curandState* curand_state,int nx,int ny,dim3 blocks,dim3 threads, CudaPointerList* pointerList) {
    // âÊëfÇ≤Ç∆Ç…óêêîÇèâä˙âª
    random_init << <blocks, threads >> > (nx, ny, curand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    pointerList->append((void**)curand_state);
}

__global__ void destroy(HitableList** world,
    Camera** camera, TransformList** transformPointer) {

    (*world)->freeMemory();
    (*transformPointer)->freeMemory();
    delete* world;
    delete* camera;
    delete* transformPointer;

}
