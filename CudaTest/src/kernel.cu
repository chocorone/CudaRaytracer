#include "Loader/CSVWriter.h"
#include "core/render.h"


void renderBoneBVH(int nx, int ny, int samples, int max_depth, int beginFrame, int endFrame,
    Hitable** world, Camera** camera, AnimationDataList* animationData, FBXAnimationData* fbxAnimationData,
    dim3 blocks, dim3 threads, curandState* curand_state, std::vector<std::vector<std::string>>& data,
    CudaPointerList* pointerList, FBXObject* fbxData)
{
    StopWatch sw;
    sw.Reset();
    sw.Start();
    HitableList** boneBVHList;
    checkCudaErrors(cudaMallocManaged((void**)&boneBVHList, sizeof(HitableList**)));
    init_List(boneBVHList, pointerList);
    createBoneBVH(boneBVHList, fbxData, curand_state, pointerList);
    sw.Stop();
    data.push_back({ "", "", "",std::to_string(sw.GetTime()) });
    renderBVHNodeAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)boneBVHList, camera, animationData, fbxAnimationData, blocks, threads, curand_state, data);

}


void renderBVH(int nx, int ny, int samples, int max_depth, int beginFrame, int endFrame,
    Hitable** world, Camera** camera, AnimationDataList* animationData, FBXAnimationData* fbxAnimationData,
    dim3 blocks, dim3 threads, curandState* curand_state, std::vector<std::vector<std::string>>& data,
    CudaPointerList* pointerList, HitableList** fbxList)
{
    StopWatch sw;
    sw.Reset();
    sw.Start();
    BVHNode** bvhNode;
    checkCudaErrors(cudaMallocManaged((void**)&bvhNode, sizeof(HitableList**)));
    create_BVHfromList(bvhNode, fbxList, curand_state, pointerList);
    sw.Stop();
    data.push_back({ "", "", "",std::to_string(sw.GetTime()) });
    renderBVHAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)bvhNode, camera, animationData, fbxAnimationData, blocks, threads, curand_state, data);
}

int main()
{
    cudaDeviceReset();

    // パラメーター設定
    const int nx = 1024 * RESOLUTION;
    const int ny = 512 * RESOLUTION;  
    const int threadX = 16;
    const int threadY = 16;
    const int max_depth = 8;
    const int samples = 4;
    const int beginFrame = 0;
    int endFrame = 30;

    const int num_pixel = nx * ny;
    dim3 blocks(nx / threadX + 1, ny / threadY + 1);
    dim3 threads(threadX, threadY);
    CudaPointerList* pointerList = new CudaPointerList();//あとで破棄するデバイス用ポインターのリスト

    //計測用データ
    StopWatch sw;
    std::vector<std::vector<std::string>> data;
    data.push_back({ "frame", "rendering", "update","build"});

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

    //FBXオブジェクト作成
    HitableList** fbxList;
    checkCudaErrors(cudaMallocManaged((void**)&fbxList, sizeof(HitableList*)));
    init_List(fbxList, pointerList);
    //FBXファイル読み込み
    FBXObject* fbxData = new FBXObject();//モデルデータ
    checkCudaErrors(cudaMallocManaged((void**)&fbxData, sizeof(FBXObject*)));
    FBXAnimationData* fbxAnimationData;//アニメーションデータ
    fbxAnimationData = new FBXAnimationData();
    create_FBXObject("./objects/high_Walking2.fbx", fbxData, fbxAnimationData, endFrame, pointerList);
    // メッシュの生成
    create_FBXMesh(fbxList, fbxData, fbxAnimationData);

    endFrame = 0;
    //ただのリスト
    renderListAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)fbxList, camera, animationData, fbxAnimationData, blocks, threads, curand_state);
    //ボーンによるBVH
    //renderBoneBVH(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)fbxList, camera, animationData, fbxAnimationData, blocks, threads, curand_state, data, pointerList, fbxData);
    //BVH
    //renderBVH(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)fbxList, camera, animationData, fbxAnimationData, blocks, threads, curand_state, data, pointerList, fbxList);


    // CSVファイルに書き出す
    writeCSV("output.csv", data);
    printf("csv書き出し完了\n");

    //メモリ解放
    checkCudaErrors(cudaDeviceSynchronize());
    pointerList->freeMemory();
    cudaDeviceReset();
    checkCudaErrors(cudaGetLastError());

    free(animationData);
    free(fbxAnimationData);

    return 0;
}

