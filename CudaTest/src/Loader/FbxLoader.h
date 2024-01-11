#pragma once
#include <fbxsdk.h>
#include <vector>
#include <array>
#include <string>
#include "../shapes/MeshObject.h"
#include "../core/deviceManage.h"
#include <map>
#pragma comment(lib, "libfbxsdk-md.lib")
#pragma comment(lib, "libxml2-md.lib")
#pragma comment(lib, "zlib-md.lib")
#pragma comment(lib, "zlib-md.lib")

void printAllNode(fbxsdk::FbxNode* object,int index) 
{

	printf("%*s%s\n", index,"", object->GetNameOnly());
	
	for (int i = 0; i < object->GetChildCount(); i++)
	{
		printAllNode(object->GetChild(i), index+1);
	}
}


bool GetMeshData(fbxsdk::FbxManager* manager,fbxsdk::FbxScene* scene, FBXObject* fbxData) {
	// �O�p�|���S���ւ̃R���o�[�g
	FbxGeometryConverter geometryConverter(manager);
	if (!geometryConverter.Triangulate(scene, true))
	{
		printf("��غ�ݎ擾���s\n");
		return false;
	}
	// ���b�V���擾
	auto mesh = scene->GetSrcObject<FbxMesh>();
	if (!mesh)
	{
		printf("���b�V���擾���s\n");
		return false;
	}
	int nPoints, nTriangles;
	nPoints = mesh->GetControlPointsCount();
	cudaMallocManaged((void**)&fbxData->mesh, sizeof(MeshData*));
	fbxData->mesh->nPoints = nPoints;
	nTriangles = mesh->GetPolygonCount();
	fbxData->mesh->nTriangles = nTriangles;

	printf("�f�[�^���擾����\n");

	cudaMallocManaged((void**)&fbxData->mesh->points, nPoints * sizeof(vec3));
	cudaDeviceSynchronize();
	cudaMallocManaged((void**)&fbxData->mesh->idxVertex, nTriangles * sizeof(vec3));
	cudaDeviceSynchronize();
	cudaMallocManaged((void**)&fbxData->mesh->normals, nTriangles * sizeof(vec3));
	cudaDeviceSynchronize();

	// ���_���W���̃��X�g�𐶐�
	for (int i = 0; i < nPoints; i++)
	{
		// ���_���W��ǂݍ���Őݒ�
		auto point = mesh->GetControlPointAt(i);

		vec3 vertex = vec3(point[0], point[1], point[2]);
		fbxData->mesh->points[i] = vertex;
	}
	printf("���_�擾����\n");

	// ���_���̏����擾����
	// 3�p�`��غ�݂Ɍ��肷��
	for (int polIndex = 0; polIndex < nTriangles; polIndex++) // �|���S�����̃��[�v
	{
		// �C���f�b�N�X���W
		int one = mesh->GetPolygonVertex(polIndex, 0);
		int two = mesh->GetPolygonVertex(polIndex, 1);
		int three = mesh->GetPolygonVertex(polIndex, 2);
		fbxData->mesh->idxVertex[polIndex] = vec3(one, two, three);
		//�@��
		FbxVector4 normalVec4;
		mesh->GetPolygonVertexNormal(polIndex, 0, normalVec4);
		fbxData->mesh->normals[polIndex] = vec3(normalVec4[0], normalVec4[1], normalVec4[2]);
	}
	printf("��غ�ݎ擾����\n");
	return true;
}

void GetBoneData(fbxsdk::FbxImporter* importer, fbxsdk::FbxScene* scene, FBXObject* fbxData) {
	auto mesh = scene->GetSrcObject<FbxMesh>();
	fbxsdk::FbxSkin* pSkin = static_cast<fbxsdk::FbxSkin*>(mesh->GetDeformer(0));

	if (mesh->GetDeformer(0)->GetDeformerType() != fbxsdk::FbxDeformer::EDeformerType::eSkin) {
		printf("�f�t�H�[�}�[�̎�ނ��قȂ�܂�\n");
		return;
	}

	int ClusterCount = pSkin->GetClusterCount();

	printf("�{�[����%d\n", ClusterCount);
	fbxData->boneCount = ClusterCount;
	cudaMallocManaged((void**)&fbxData->boneList, sizeof(Bone) * ClusterCount);
	cudaDeviceSynchronize();

	for (int i = 0; i < ClusterCount; i++)
	{
		fbxsdk::FbxCluster* pCluster = pSkin->GetCluster(i);
		//�{�[���̃f�t�H���g�̃O���[�o�����W���擾
		FbxNode* node = pCluster->GetLink();
		fbxsdk::FbxAMatrix amat = node->EvaluateGlobalTransform();
		vec3 defaultTransform = vec3(amat.GetT()[0], amat.GetT()[1], amat.GetT()[2]);
		vec3 defaultRotation = vec3(amat.GetR()[0], amat.GetR()[1], amat.GetR()[2]);
		
		fbxData->boneList[i] = Bone(node->GetName(), defaultTransform, defaultRotation,
			defaultTransform, defaultRotation,pCluster->GetControlPointIndicesCount());

		cudaMallocManaged((void**)&fbxData->boneList[i].weightIndices, sizeof(int) * pCluster->GetControlPointIndicesCount());
		cudaMallocManaged((void**)&fbxData->boneList[i].weights, sizeof(double) * pCluster->GetControlPointIndicesCount());
		cudaDeviceSynchronize();
		for (int weightIndex = 0; weightIndex < pCluster->GetControlPointIndicesCount(); weightIndex++) 
		{
			fbxData->boneList[i].weightIndices[weightIndex] = pCluster->GetControlPointIndices()[weightIndex];
			fbxData->boneList[i].weights[weightIndex] = pCluster->GetControlPointWeights()[weightIndex];
			
			//������2�t���[��������̒��_�̈ʒu���v�Z���ē����
		}

		printf("%s\n", pCluster->GetName());
		
		
		printf("     t = (%8.3f, %8.3f, %8.3f)\n     r = (%8.3f, %8.3f, %8.3f)\n",
			amat.GetT()[0],
			amat.GetT()[1],
			amat.GetT()[2],
			amat.GetR()[0],
			amat.GetR()[1],
			amat.GetR()[2]);
	}

}

void GetAnimationData(fbxsdk::FbxImporter* importer, fbxsdk::FbxScene* scene, FBXAnimationData* animationData,FBXObject* fbxData, int& endFrame) {
	//�A�j���[�V�������擾
	int animStackCount = importer->GetAnimStackCount();
	FbxTakeInfo* pFbxTakeInfo = importer->GetTakeInfo(0);
	FbxLongLong start = pFbxTakeInfo->mLocalTimeSpan.GetStart().Get();
	FbxLongLong stop = pFbxTakeInfo->mLocalTimeSpan.GetStop().Get();
	FbxLongLong oneFrameValue = FbxTime::GetOneFrameValue(FbxTime::eFrames60);
	int framecount = (stop - start) / oneFrameValue;
	printf("�A�j���[�V�����̍��v�t���[����%d\n", framecount);
	endFrame = framecount-1;
	
	auto mesh = scene->GetSrcObject<FbxMesh>();
	fbxsdk::FbxSkin* pSkin = static_cast<fbxsdk::FbxSkin*>(mesh->GetDeformer(0));

	if (mesh->GetDeformer(0)->GetDeformerType() != fbxsdk::FbxDeformer::EDeformerType::eSkin) {
		printf("�f�t�H�[�}�[�̎�ނ��قȂ�܂�\n");
		return;
	}

	animationData->frameCount = framecount;
	animationData->animation = (BonePoseData*)malloc(sizeof(BonePoseData) * framecount);
	for (int frameIndex = 0; frameIndex < framecount; frameIndex++) 
	{
		BonePoseData pose = BonePoseData();
		pose.boneCount = fbxData->boneCount;
		cudaMallocManaged((void**)&pose.nowTransforom, sizeof(vec3) * fbxData->boneCount);
		cudaMallocManaged((void**)&pose.nowRatation, sizeof(vec3) * fbxData->boneCount);
		cudaDeviceSynchronize();

		for (int i = 0; i < fbxData->boneCount; i++)
		{
			fbxsdk::FbxCluster* pCluster = pSkin->GetCluster(i);
			FbxNode* node = pCluster->GetLink();
			fbxsdk::FbxAMatrix amat = node->EvaluateGlobalTransform(frameIndex* oneFrameValue);
			pose.nowTransforom[i] = vec3(amat.GetT()[0], amat.GetT()[1], amat.GetT()[2]);
			pose.nowRatation[i] = vec3(amat.GetR()[0], amat.GetR()[1], amat.GetR()[2]);
		}
		animationData->animation[frameIndex] = pose;
		printf("%d�t���[���ړǂݍ��݊���\n", frameIndex);
	}	
}

bool CreateFBXData(const std::string& filePath, FBXObject* fbxData, FBXAnimationData* animationData, int& endFrame)
{
	auto manager = FbxManager::Create();

	// �C���|�[�^�[������
	auto importer = FbxImporter::Create(manager, "");
	if (!importer->Initialize(filePath.c_str(), -1, manager->GetIOSettings()))
	{
		printf("�C���|�[�^�[���������s\n");
		return false;
	}

	// �V�[���쐬
	auto scene = FbxScene::Create(manager, "");
	importer->Import(scene);

	if (!GetMeshData(manager, scene, fbxData)) {
		printf("���b�V�����̎擾�Ɏ��s\n");
		return false;
	}

	GetBoneData(importer, scene,fbxData);
	animationData->object = fbxData;
	GetAnimationData(importer, scene,animationData,fbxData, endFrame);

	// �}�l�[�W���[�A�V�[���̔j��
	importer->Destroy();
	scene->Destroy();
	manager->Destroy();
	return true;
}

