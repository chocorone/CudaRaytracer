#include "Loader/CSVWriter.h"
#include "core/render.h"

void renderFBXList(int nx, int ny, int samples, int max_depth, int beginFrame, int endFrame,Hitable** world,
    Camera** camera,dim3 blocks, dim3 threads, curandState* curand_state, FBXObject* obj,HitableList** fbxList)
{
    vec3* h_pointPos = (vec3*)malloc(sizeof(vec3) * obj->mesh->nPoints);

    vec3* d_idxVertices;
    cudaMalloc(&d_idxVertices, sizeof(vec3) * obj->mesh->nTriangles);
    cudaMemcpy(d_idxVertices, obj->mesh->idxVertex, obj->mesh->nTriangles * sizeof(vec3), cudaMemcpyHostToDevice);

    // レンダリング
    for (int frameIndex = beginFrame; frameIndex <= endFrame; frameIndex++)
    {
        //メッシュの位置の更新
        updateFBXObj(frameIndex, obj, h_pointPos, d_idxVertices,fbxList);
        renderImage(nx, ny, samples, max_depth, frameIndex, world,camera, blocks,threads,curand_state);
    }
    checkCudaErrors(cudaFree(d_idxVertices));
    free(h_pointPos);
}

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

    vec3* h_pointPos = (vec3*)malloc(sizeof(vec3) * obj->mesh->nPoints);
    vec3* d_idxVertices;
    cudaMalloc(&d_idxVertices, sizeof(vec3) * obj->mesh->nTriangles);
    cudaMemcpy(d_idxVertices, obj->mesh->idxVertex, obj->mesh->nTriangles * sizeof(vec3), cudaMemcpyHostToDevice);
    // レンダリング
    for (int frameIndex = beginFrame; frameIndex <= endFrame; frameIndex++)
    {
        //メッシュの位置の更新
        updateFBXObj(frameIndex, obj, h_pointPos, d_idxVertices, fbxList);
        Update_BVH(d_bvhNode);
        renderImage(nx, ny, samples, max_depth, frameIndex, (Hitable**)d_bvhNode, camera, blocks, threads, curand_state);
    }
    checkCudaErrors(cudaFree(d_idxVertices));
    checkCudaErrors(cudaFree(d_bvhNode));
    free(h_pointPos);
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

    //ただのリスト
    //renderFBXList(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)d_fbxList, d_camera, blocks, threads, d_curand_state,h_fbxData,d_fbxList);
    //ボーンによるBVH
    //renderBoneBVH(nx, ny, samples, max_depth, beginFrame, endFrame, camera, fbxAnimationData, blocks, threads, curand_state, data, pointerList, fbxData);
    //BVH
    renderBVH(nx, ny, samples, max_depth, beginFrame, endFrame,  d_camera, blocks, threads, d_curand_state, h_fbxData, d_fbxList,data);


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

