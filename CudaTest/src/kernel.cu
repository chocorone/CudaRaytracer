#include "Loader/CSVWriter.h"
#include "core/render.h"

void renderFBXList(int nx, int ny, int samples, int max_depth, int beginFrame, int endFrame,
    Camera** camera,dim3 blocks, dim3 threads, curandState* curand_state, FBXObject* obj,HitableList** fbxList)
{
    // レンダリング
    for (int frameIndex = beginFrame; frameIndex <= endFrame; frameIndex++)
    {
        //メッシュの位置の更新
        updateFBXObj(frameIndex, obj,fbxList);
        renderImage(nx, ny, samples, max_depth, frameIndex, (Hitable**)fbxList,camera, blocks,threads,curand_state);
    }
}

void renderBoneBVH(int nx, int ny, int samples, int max_depth, int beginFrame, int endFrame,
    Camera** camera, dim3 blocks, dim3 threads, curandState* curand_state, FBXObject* obj, HitableList** fbxList,
    std::vector<std::vector<std::string>>& data)
{
    StopWatch sw;
    sw.Reset();
    sw.Start();
    HitableList** d_boneBVHList;
    cudaMalloc(&d_boneBVHList, sizeof(HitableList*));
    createBoneBVH(d_boneBVHList, obj, curand_state,fbxList);
    sw.Stop();
    printf("BVH作成完了\n");
    data.push_back({ "", "", "",std::to_string(sw.GetTime()) });

    // レンダリング
    for (int frameIndex = beginFrame; frameIndex <= endFrame; frameIndex++)
    {
        //メッシュの位置の更新
        updateFBXObj(frameIndex, obj,fbxList);
        Update_BVH(d_boneBVHList,obj);
        //std::string updateTime = std::to_string(sw.GetTime());
        renderImage(nx, ny, samples, max_depth, frameIndex, (Hitable**)d_boneBVHList, camera, blocks, threads, curand_state);
    }
    checkCudaErrors(cudaFree(d_boneBVHList));
}


void renderBVH(int nx, int ny, int samples, int max_depth, int beginFrame, int endFrame, 
    Camera** camera, dim3 blocks, dim3 threads, curandState* curand_state, FBXObject* obj, HitableList** fbxList,
    std::vector<std::vector<std::string>>& data)
{
    StopWatch sw;
    sw.Reset();
    sw.Start();
    BVHNode** d_bvhNode;
    cudaMalloc(&d_bvhNode, sizeof(BVHNode*));
    create_BVHfromList(d_bvhNode, fbxList, curand_state);
    sw.Stop();
    printf("BVH作成完了\n");
    data.push_back({ "", "", "",std::to_string(sw.GetTime()) });

    // レンダリング
    for (int frameIndex = beginFrame; frameIndex <= endFrame; frameIndex++)
    {
        //メッシュの位置の更新
        updateFBXObj(frameIndex, obj, fbxList);
        Update_BVH(d_bvhNode);
        renderImage(nx, ny, samples, max_depth, frameIndex, (Hitable**)d_bvhNode, camera, blocks, threads, curand_state);
    }
    checkCudaErrors(cudaFree(d_bvhNode));
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
    printf("シーン準備完了\n");

    //FBXファイル読み込み
    FBXObject* h_fbxData = new FBXObject();//モデルデータ
    //create_FBXObject("./objects/high_Walking3.fbx", fbxData, fbxAnimationData, endFrame, pointerList);
    CreateFBXData("./objects/low_walking.fbx", h_fbxData, endFrame);
    //create_FBXObject("./objects/low_standUp.fbx", fbxData, fbxAnimationData, endFrame, pointerList);
    printf("FBXロード完了\n");

    //FBXオブジェクト作成
    HitableList** d_fbxList;
    cudaMalloc(&d_fbxList, sizeof(HitableList*));
    create_FBXMesh(d_fbxList, h_fbxData);
    printf("FBX作成完了\n");

    endFrame = 5;
    //ただのリスト
    //renderFBXList(nx, ny, samples, max_depth, beginFrame, endFrame, d_camera, blocks, threads, d_curand_state,h_fbxData,d_fbxList);
    //ボーンによるBVH
    renderBoneBVH(nx, ny, samples, max_depth, beginFrame, endFrame, d_camera, blocks, threads, d_curand_state, h_fbxData, d_fbxList, data);
    //BVH
    //renderBVH(nx, ny, samples, max_depth, beginFrame, endFrame,  d_camera, blocks, threads, d_curand_state, h_fbxData, d_fbxList,data);


    // CSVファイルに書き出す
    writeCSV("output.csv", data);
    printf("csv書き出し完了\n");

    //メモリ解放
    checkCudaErrors(cudaDeviceSynchronize());
    pointerList->freeMemory();
    cudaDeviceReset();
    checkCudaErrors(cudaGetLastError());

    return 0;
}

