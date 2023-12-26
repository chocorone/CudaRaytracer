#include "core/render.h"


class HostPointerList
{
public:
    HostPointerList() { list = new void* (); list_size = 0; }
    HostPointerList(void** l, int n) { list = l; list_size = n; }
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
        for (size_t i = 0; i < list_size; i++)
        {
            free(list[i]);
        }
        free(list);
        list_size = 0;
    }
    void** list;
    int list_size;
};

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
    const int endFrame = 90;

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
    SetCurandState(curand_state, nx, ny, blocks, threads);
    pointerList->append((void**)curand_state);

    //カメラ作成
    Camera** camera;
    checkCudaErrors(cudaMallocManaged((void**)&camera, sizeof(Camera*)));
    pointerList->append((void**)camera);
    create_camera << <1, 1 >> > (camera, nx, ny, vec3(0, 20, 400), vec3(0, 20, 0), 10.0, 0.0, 60);
    //create_camera << <1, 1 >> > (camera, nx, ny, vec3(278, 278, -700), vec3(278, 278, 0), 10.0, 0.0, 40);

    //あとで修正
    HitableList** world;
    TransformList** transformPointer;
    checkCudaErrors(cudaMallocManaged((void**)&world, sizeof(HitableList*)));
    checkCudaErrors(cudaMallocManaged((void**)&transformPointer, sizeof(TransformList*)));
    init_data << <1, 1 >> > (world, transformPointer);
    AnimationDataList* animationData = new AnimationDataList();
    //BuildAnimatedSphere(world,animationData, transformPointer);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    
    //FBXオブジェクト作成
    FBXObject* fbxData = new FBXObject();//モデルデータ
    checkCudaErrors(cudaMallocManaged((void**)&fbxData, sizeof(FBXObject*)));
    pointerList->append((void**)fbxData);
    FBXAnimationData* fbxAnimationData;//アニメーションデータ
    fbxAnimationData = (FBXAnimationData*)malloc(sizeof(FBXAnimationData*));
    CreateFBXData("./objects/human_light.fbx", fbxData, fbxAnimationData);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    add_mesh_fromPoseData << <1, 1 >> > (world, fbxData, fbxAnimationData->animation[0]); //メッシュの移動と作成
    BVHNode** bvh;
    checkCudaErrors(cudaMallocManaged((void**)&bvh, sizeof(BVHNode*)));
    pointerList->append((void**)bvh);
    create_BVH << <1, 1 >> > (world, bvh, curand_state);
    CHECK(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    printf("BVH作成完了\n");

    //レンダリング
    //renderAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)world, camera, animationData, transformPointer, fbxAnimationData, blocks, threads, curand_state);
    renderAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)bvh, camera,animationData,transformPointer, fbxAnimationData,blocks,threads,curand_state);
    
    //メモリ解放
    checkCudaErrors(cudaDeviceSynchronize());
    //destroy << <1, 1 >> > (world, camera, transformPointer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(world));
    checkCudaErrors(cudaFree(transformPointer));
    pointerList->freeMemory();
    free(pointerList);
    free(animationData);
    free(fbxAnimationData);

    
    cudaDeviceReset();


    return 0;
}

