#include "Loader/CSVWriter.h"
#include "core/render.h"


void renderBoneBVH(int nx, int ny, int samples, int max_depth, int beginFrame, int endFrame,
    Camera** camera, FBXAnimationData* fbxAnimationData,
    dim3 blocks, dim3 threads, curandState* curand_state, std::vector<std::vector<std::string>>& data,
    CudaPointerList* pointerList, FBXObject* fbxData)
{
    StopWatch sw;
    sw.Reset();
    sw.Start();
    HitableList** boneBVHList;
    checkCudaErrors(cudaMallocManaged((void**)&boneBVHList, sizeof(HitableList**)));
    //init_List(boneBVHList, pointerList);
    createBoneBVH(boneBVHList, fbxData, curand_state, pointerList);
    sw.Stop();
    printf("BVH作成完了\n");
    data.push_back({ "", "", "",std::to_string(sw.GetTime()) });
    renderBVHNodeAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)boneBVHList, camera, fbxAnimationData, blocks, threads, curand_state, data);

}


void renderBVH(int nx, int ny, int samples, int max_depth, int beginFrame, int endFrame,
    HitableList** fbxList, Camera** camera, FBXAnimationData* fbxAnimationData,
    dim3 blocks, dim3 threads, curandState* curand_state, std::vector<std::vector<std::string>>& data,
    CudaPointerList* pointerList)
{
    StopWatch sw;
    sw.Reset();
    sw.Start();
    BVHNode** bvhNode;
    checkCudaErrors(cudaMallocManaged((void**)&bvhNode, sizeof(HitableList**)));
    create_BVHfromList(bvhNode, fbxList, curand_state, pointerList);
    sw.Stop();
    printf("BVH作成完了\n");
    data.push_back({ "", "", "",std::to_string(sw.GetTime()) });
    renderBVHAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)bvhNode, camera, fbxAnimationData, blocks, threads, curand_state, data);
}

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
    int endFrame = 30;

    const int num_pixel = nx * ny;
    dim3 blocks(nx / threadX + 1, ny / threadY + 1);
    dim3 threads(threadX, threadY);
    CudaPointerList* pointerList = new CudaPointerList();//あとで破棄するデバイス用ポインターのリスト

    //計測用データ
    std::vector<std::vector<std::string>> data;
    data.push_back({ "frame", "rendering", "update","build"});

    //ヒープサイズ・スタックサイズ指定
    ChangeHeapSize(1024 * 1024 * 1024 * 128);
    ChangeStackSize(1024 * 128);
    // 乱数列生成用のメモリ確保
    curandState* d_curand_state;
    cudaMalloc(&d_curand_state, nx * ny * sizeof(curandState));
    SetCurandState(d_curand_state, nx, ny, blocks, threads,pointerList);

    //カメラ作成
    Camera** d_camera;
    cudaMalloc(&d_camera, sizeof(Camera*));
    init_camera(d_camera, nx, ny, pointerList);

    //FBXオブジェクト作成
    HitableList** fbxList;
    checkCudaErrors(cudaMallocManaged((void**)&fbxList, sizeof(HitableList*)));
    init_List(fbxList, pointerList);
    //FBXファイル読み込み
    FBXObject* fbxData = new FBXObject();//モデルデータ
    checkCudaErrors(cudaMallocManaged((void**)&fbxData, sizeof(FBXObject*)));
    FBXAnimationData* fbxAnimationData;//アニメーションデータ
    fbxAnimationData = new FBXAnimationData();
    //create_FBXObject("./objects/high_Walking3.fbx", fbxData, fbxAnimationData, endFrame, pointerList);
    create_FBXObject("./objects/low_walking.fbx", fbxData, fbxAnimationData, endFrame, pointerList);
    //create_FBXObject("./objects/low_standUp.fbx", fbxData, fbxAnimationData, endFrame, pointerList);
    
    // メッシュの生成
    create_FBXMesh(fbxList, fbxData, fbxAnimationData);
    printf("シーン作成完了\n");

    //ただのリスト
    //renderListAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)fbxList, camera, fbxAnimationData, blocks, threads, curand_state);
    //ボーンによるBVH
    //renderBoneBVH(nx, ny, samples, max_depth, beginFrame, endFrame, camera, fbxAnimationData, blocks, threads, curand_state, data, pointerList, fbxData);
    //BVH
    renderBVH(nx, ny, samples, max_depth, beginFrame, endFrame, fbxList, d_camera, fbxAnimationData, blocks, threads, d_curand_state, data, pointerList);


    // CSVファイルに書き出す
    writeCSV("output.csv", data);
    printf("csv書き出し完了\n");

    //メモリ解放
    checkCudaErrors(cudaDeviceSynchronize());
    pointerList->freeMemory();
    cudaDeviceReset();
    checkCudaErrors(cudaGetLastError());
    free(fbxAnimationData);

    return 0;
}

