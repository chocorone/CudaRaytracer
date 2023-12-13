#include "core/render.h"

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
    HitableList** world;
    Camera** camera;
    TransformList** transformPointer;

    checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(HitableList*)));
    checkCudaErrors(cudaMallocManaged((void**)&camera, sizeof(Camera*)));
    checkCudaErrors(cudaMallocManaged((void**)&transformPointer, sizeof(TransformList*)));
    
    // 画素ごとに乱数を初期化
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    random_init <<<blocks, threads >>> (nx, ny, curand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // オブジェクト、カメラの生成
    //int obj_count = BuildRandomWorld(world,obj_list,camera, curand_state,nx, ny);
    AnimationDataList* animationData = new AnimationDataList();
    BuildAppendTest(world, camera, curand_state, animationData, transformPointer, nx, ny);

    renderAnimation(nx, ny, samples, max_depth, 0, maxFrame,
        colorBuffer,world, camera,animationData,transformPointer,blocks,threads,curand_state);
    
    animationData->freeMemory();
    // clean up
    checkCudaErrors(cudaDeviceSynchronize());
    destroy << <1, 1 >> > ( world, camera,transformPointer);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(camera));
    checkCudaErrors(cudaFree(transformPointer));
    checkCudaErrors(cudaFree(curand_state));
    checkCudaErrors(cudaFree(colorBuffer));
    
    cudaDeviceReset();


    return 0;
}

