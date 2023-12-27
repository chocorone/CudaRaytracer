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
    const int endFrame = 30;

    const int num_pixel = nx * ny;
    dim3 blocks(nx / threadX + 1, ny / threadY + 1);
    dim3 threads(threadX, threadY);
    CudaPointerList* pointerList = new CudaPointerList();//あとで破棄するデバイス用ポインターのリスト

    //ヒープサイズ・スタックサイズ指定
    ChangeHeapSize(1024 * 1024 * 1024);
    ChangeStackSize(4096 * 2);
    // 乱数列生成用のメモリ確保
    curandState* curand_state;
    checkCudaErrors(cudaMallocManaged((void**)&curand_state, nx * ny * sizeof(curandState)));
    SetCurandState(curand_state, nx, ny, blocks, threads,pointerList);

    //カメラ作成
    Camera** camera;
    checkCudaErrors(cudaMallocManaged((void**)&camera, sizeof(Camera*)));
    init_camera(camera, nx, ny, pointerList);

    //オブジェクト作成
    TransformList** transformPointer;
    checkCudaErrors(cudaMallocManaged((void**)&transformPointer, sizeof(TransformList*)));
    init_TransformList(transformPointer, pointerList);
    AnimationDataList* animationData = new AnimationDataList();
    //BuildAnimatedSphere(world,animationData, transformPointer);
    
    //FBXオブジェクト作成
    HitableList** fbxList;
    checkCudaErrors(cudaMallocManaged((void**)&fbxList, sizeof(HitableList*)));
    init_List(fbxList, pointerList);
    //FBXファイル読み込み
    FBXObject* fbxData = new FBXObject();//モデルデータ
    checkCudaErrors(cudaMallocManaged((void**)&fbxData, sizeof(FBXObject*)));
    FBXAnimationData* fbxAnimationData;//アニメーションデータ
    fbxAnimationData = (FBXAnimationData*)malloc(sizeof(FBXAnimationData*));
    create_FBXObject("./objects/human_light.fbx", fbxData, fbxAnimationData, pointerList);
    // メッシュの生成
    create_FBXMesh(fbxList, fbxData, fbxAnimationData);
    //BVHの作成
    HitableList** boneBVHList;
    checkCudaErrors(cudaMallocManaged((void**)&boneBVHList, sizeof(BVHNode*)));
    init_List(boneBVHList, pointerList);
    createBoneBVH(boneBVHList, fbxData, curand_state, pointerList);

    //レンダリング
    //renderAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)world, camera, animationData, transformPointer, fbxAnimationData, blocks, threads, curand_state);
    //renderAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)FBXBVH, camera, animationData, transformPointer, fbxAnimationData, blocks, threads, curand_state);
    renderAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)boneBVHList, camera,animationData,transformPointer, fbxAnimationData,blocks,threads,curand_state);
    
    //メモリ解放
    checkCudaErrors(cudaDeviceSynchronize());
    pointerList->freeMemory();
    free(animationData);
    free(fbxAnimationData);
    cudaDeviceReset();
    checkCudaErrors(cudaGetLastError());

    return 0;
}

