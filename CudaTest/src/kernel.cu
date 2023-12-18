#include "core/render.h"


int main()
{
    // パラメーター設定
    const int nx = 1024 * RESOLUTION;
    const int ny = 512 * RESOLUTION;  
    const int threadX = 16;
    const int threadY = 16;
    const int max_depth = 8;
    const int samples = 4;
    const int beginFrame = 0;
    const int endFrame = 0;

    const int num_pixel = nx * ny;

    //ヒープサイズ・スタックサイズ指定
    ChangeHeapSize(1024 * 1024 * 1024);
    ChangeStackSize(4096 * 2);
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
    BVHNode** bvh;
    checkCudaErrors(cudaMallocManaged((void**)&bvh, sizeof(BVHNode*)));
    
    // 画素ごとに乱数を初期化
    dim3 blocks(nx / threadX + 1, ny / threadY + 1);
    dim3 threads(threadX, threadY);
    random_init <<<blocks, threads >>> (nx, ny, curand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //FBXファイル読み込み
    MeshData* meshData;
    checkCudaErrors(cudaMallocManaged((void**)&meshData, sizeof(MeshData*)));
    CreateFBXMeshData("./objects/HipHopDancing.fbx", meshData);
    CreateFBXMeshData("./objects/bunny2.fbx", meshData);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // オブジェクト、カメラの生成
    AnimationDataList* animationData = new AnimationDataList();
    create_camera << <1, 1 >> > (camera, nx, ny, vec3(0,0,20), vec3(0, 0, 0), 10.0, 0.0, 60);
    //create_camera << <1, 1 >> > (camera, nx, ny, vec3(278, 278, -700), vec3(278, 278, 0), 10.0, 0.0, 40);
    init_data << <1, 1 >> > (world, transformPointer);
    //BuildAnimatedSphere(world,animationData, transformPointer);
    add_mesh_withNormal << <1, 1 >> > (world, meshData, transformPointer);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("シーン作成完了\n");

    //BVHの作成
    create_BVH << <1, 1 >> > (world, bvh, curand_state);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("BVH作成完了\n");

    //レンダリング
    //renderAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)world, camera, animationData, transformPointer, blocks, threads, curand_state);
    renderAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)bvh, camera,animationData,transformPointer,blocks,threads,curand_state);
    
    //メモリ解放
    animationData->freeMemory();
    checkCudaErrors(cudaDeviceSynchronize());
    destroy << <1, 1 >> > (world, camera, transformPointer);
    destroy << <1, 1 >> > (meshData);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(camera));
    checkCudaErrors(cudaFree(transformPointer));
    checkCudaErrors(cudaFree(curand_state));
    checkCudaErrors(cudaFree(meshData));
    checkCudaErrors(cudaFree(bvh));

    
    cudaDeviceReset();


    return 0;
}

