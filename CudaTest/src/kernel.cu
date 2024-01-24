#include "Loader/CSVWriter.h"
#include "core/render.h"


void renderBoneBVH(int nx, int ny, int samples, int max_depth, int beginFrame, int endFrame,
    Camera** camera, 
    dim3 blocks, dim3 threads, curandState* curand_state, std::vector<std::vector<std::string>>& data,
    CudaPointerList* pointerList, FBXObject* fbxData)
{
    StopWatch sw;
    sw.Reset();
    sw.Start();
    HitableList** boneBVHList;
    cudaMalloc(&boneBVHList, sizeof(HitableList*));
    createBoneBVH(boneBVHList, fbxData, curand_state, pointerList);
    sw.Stop();
    printf("BVH�쐬����\n");
    data.push_back({ "", "", "",std::to_string(sw.GetTime()) });
    renderBoneBVHAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)boneBVHList, camera, fbxData, blocks, threads, curand_state, data);

}


void renderBVH(int nx, int ny, int samples, int max_depth, int beginFrame, int endFrame,
    HitableList** fbxList, Camera** camera,
    dim3 blocks, dim3 threads, curandState* curand_state, std::vector<std::vector<std::string>>& data,
    CudaPointerList* pointerList, FBXObject* fbxData)
{
    StopWatch sw;
    sw.Reset();
    sw.Start();
    BVHNode** bvhNode;
    cudaMalloc(&bvhNode, sizeof(BVHNode*));
    create_BVHfromList(bvhNode, fbxList, curand_state, pointerList);
    sw.Stop();
    printf("BVH�쐬����\n");
    data.push_back({ "", "", "",std::to_string(sw.GetTime()) });
    renderBVHAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)bvhNode, camera, fbxData, blocks, threads, curand_state, data);
}

int main()
{
    // �p�����[�^�[�ݒ�
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
    CudaPointerList* pointerList = new CudaPointerList();//���ƂŔj������f�o�C�X�p�|�C���^�[�̃��X�g

    //�v���p�f�[�^
    StopWatch sw;
    std::vector<std::vector<std::string>> data;
    data.push_back({ "frame", "rendering", "update","build"});

    //�q�[�v�T�C�Y�E�X�^�b�N�T�C�Y�w��
    //ChangeHeapSize(1024 * 1024 * 1024*4);
    cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1048576ULL * 2048);

    ChangeStackSize(1024 * 16);
    // �����񐶐��p�̃������m��
    curandState* curand_state;
    checkCudaErrors(cudaMallocManaged((void**)&curand_state, nx * ny * sizeof(curandState)));
    SetCurandState(curand_state, nx, ny, blocks, threads,pointerList);

    //�J�����쐬
    Camera** camera;
    checkCudaErrors(cudaMallocManaged((void**)&camera, sizeof(Camera*)));
    init_camera(camera, nx, ny, pointerList);

    //FBX�I�u�W�F�N�g�쐬
    HitableList** fbxList;
    checkCudaErrors(cudaMallocManaged((void**)&fbxList, sizeof(HitableList*)));
    init_List(fbxList, pointerList);
    //FBX�t�@�C���ǂݍ���
    FBXObject* fbxData = new FBXObject();//���f���f�[�^
    CreateFBXData("./objects/low_walking.fbx", fbxData, endFrame);
    //CreateFBXData("./objects/low_standUp.fbx", fbxData, endFrame);
    //CreateFBXData("./objects/high_Walking5.fbx", fbxData, endFrame);
    //CreateFBXData("./objects/high_StandUp2.fbx", fbxData, endFrame);
    //CreateFBXData("./objects/Land2.fbx", fbxData, endFrame);
    // ���b�V���̐���
    create_FBXMesh(fbxList, fbxData);
    printf("�V�[���쐬����\n");
    //�����̃��X�g
    //renderListAnimation(nx, ny, samples, max_depth, beginFrame, endFrame, (Hitable**)fbxList, camera, fbxAnimationData, blocks, threads, curand_state);
    //�{�[���ɂ��BVH
    //renderBoneBVH(nx, ny, samples, max_depth, beginFrame, endFrame, camera, blocks, threads, curand_state, data, pointerList, fbxData);
    //BVH
    renderBVH(nx, ny, samples, max_depth, beginFrame, endFrame, fbxList, camera, blocks, threads, curand_state, data, pointerList,fbxData);


    // CSV�t�@�C���ɏ����o��
    writeCSV("output.csv", data);
    printf("csv�����o������\n");

    //���������
    checkCudaErrors(cudaDeviceSynchronize());
    pointerList->freeMemory();
    cudaDeviceReset();
    checkCudaErrors(cudaGetLastError());

    return 0;
}

