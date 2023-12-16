#include "core/render.h"

struct Parameter 
{
public:
    static const int nx = 1024 * RESOLUTION;
    static const int ny = 512 * RESOLUTION;
    static const int tx = 16;
    static const int ty = 16;
    static const int max_depth = 16;
    static const int samples = 8;
    static const int beginFrame = 0;
    static const int endFrame = 0;
};

void SetDeviceHeapSize() {
    //ヒープサイズ・スタックサイズ指定
    size_t heapSize = 1024 * 1024 * 1024;
    size_t stackSize = 4096 * 2;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, heapSize);
    cudaDeviceSetLimit(cudaLimitStackSize, stackSize);
    cudaDeviceGetLimit(&heapSize, cudaLimitMallocHeapSize);
    printf("Heap Size=%ld\n", heapSize);
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    printf("Stack Size=%ld\n", stackSize);
}

int main()
{
    SetDeviceHeapSize();

    // 画素のメモリ確保
    const int num_pixel = Parameter::nx * Parameter::ny;

    // 乱数列生成用のメモリ確保
    curandState* curand_state;
    checkCudaErrors(cudaMallocManaged((void**)&curand_state, num_pixel * sizeof(curandState)));

    //シーン保存用の変数のメモリ確保
    HitableList** world;
    Camera** camera;
    TransformList** transformPointer;
    //Hitable** bvh;
    checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(HitableList*)));
    checkCudaErrors(cudaMallocManaged((void**)&camera, sizeof(Camera*)));
    checkCudaErrors(cudaMallocManaged((void**)&transformPointer, sizeof(TransformList*)));
    //checkCudaErrors(cudaMallocManaged((void**)&bvh, sizeof(Hitable*)));

    // 画素ごとに乱数を初期化
    dim3 blocks(Parameter::nx / Parameter::tx + 1, Parameter::ny / Parameter::ty + 1);
    dim3 threads(Parameter::tx, Parameter::ty);
    random_init <<<blocks, threads >>> (Parameter::nx, Parameter::ny, curand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //FBXファイル読み込み
    MeshData* meshData;
    checkCudaErrors(cudaMallocManaged((void**)&meshData, sizeof(MeshData*)));
    //CreateFBXMeshData("./objects/HipHopDancing.fbx", meshData);
    CreateFBXMeshData("./objects/bunny2.fbx", meshData);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // オブジェクト、カメラの生成
    AnimationDataList* animationData = new AnimationDataList();
    create_camera << <1, 1 >> > (camera, Parameter::nx, Parameter::ny, vec3(0, 400, 20), vec3(0, 0, 20), 10.0, 0.0, 60);
    //create_camera << <1, 1 >> > (camera, nx, ny, vec3(278, 278, -700), vec3(278, 278, 0), 10.0, 0.0, 40);
    init_data << <1, 1 >> > (world, transformPointer);
    add_mesh_withNormal << <1, 1 >> > (world, meshData, transformPointer);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    //create_BVH << <1, 1 >> > (world,bvh, curand_state);


    /*CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());*/
    printf("シーン作成完了\n");

    //レンダリング
    renderAnimation(Parameter::nx, Parameter::ny, Parameter::samples, Parameter::max_depth, Parameter::beginFrame, Parameter::endFrame,
        world, camera, animationData, transformPointer, blocks, threads, curand_state);
    

    
    //メモリ解放
    animationData->freeMemory();
    checkCudaErrors(cudaDeviceSynchronize());
    destroy << <1, 1 >> > (world, camera, transformPointer);
    destroy << <1, 1 >> > (meshData);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(camera));
    //checkCudaErrors(cudaFree(bvh));
    checkCudaErrors(cudaFree(transformPointer));
    checkCudaErrors(cudaFree(curand_state));
    checkCudaErrors(cudaFree(meshData));

    
    cudaDeviceReset();


    return 0;
}

