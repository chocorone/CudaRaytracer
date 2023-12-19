#pragma once
#include <fbxsdk.h>
#include <vector>
#include <array>
#include <string>
#include "../shapes/MeshObject.h"
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

BonePoseData* createBonePoseData(fbxsdk::FbxNode* object,int frame) 
{
	BonePoseData** childList = (BonePoseData**)malloc(sizeof(BonePoseData) * object->GetChildCount());
	printf("%s\n", object->GetName());
	printf("local transform %f, %f, %f\n", object->EvaluateLocalTranslation(frame)[0], object->EvaluateLocalTranslation(frame)[1], object->EvaluateLocalTranslation(frame)[2]);
	printf("local rotation  %f, %f, %f\n", object->EvaluateLocalRotation(frame)[0], object->EvaluateLocalRotation(frame)[1], object->EvaluateLocalRotation(frame)[2]);

	for (int i = 0; i < object->GetChildCount(); i++)
	{
		childList[i]=createBonePoseData(object->GetChild(i),frame);
	}

	vec3 transform = vec3(object->EvaluateLocalTranslation(frame)[0], object->EvaluateLocalTranslation(frame)[1], object->EvaluateLocalTranslation(frame)[2]);
	vec3 rotation = vec3(object->EvaluateLocalRotation(frame)[0], object->EvaluateLocalRotation(frame)[1], object->EvaluateLocalRotation(frame)[2]);
	BonePoseData* poseData = new BonePoseData(transform,rotation);
	poseData->ResisterChild(childList, object->GetChildCount());
	return poseData;
}

bool GetMeshData(fbxsdk::FbxManager* manager,fbxsdk::FbxScene* scene, MeshData* data) {
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
	data->nPoints = nPoints;
	nTriangles = mesh->GetPolygonCount();
	data->nTriangles = nTriangles;

	printf("�f�[�^���擾����\n");

	cudaMallocManaged((void**)&data->points, nPoints * sizeof(vec3));
	cudaMallocManaged((void**)&data->idxVertex, nTriangles * sizeof(vec3));
	cudaMallocManaged((void**)&data->normals, nTriangles * sizeof(vec3));

	cudaDeviceSynchronize();
	cudaGetLastError();
	cudaDeviceSynchronize();

	// ���_���W���̃��X�g�𐶐�
	for (int i = 0; i < nPoints; i++)
	{
		// ���_���W��ǂݍ���Őݒ�
		auto point = mesh->GetControlPointAt(i);

		vec3 vertex = vec3(point[0], point[1], point[2]);
		data->points[i] = vertex;
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
		data->idxVertex[polIndex] = vec3(one, two, three);
		//�@��
		FbxVector4 normalVec4;
		mesh->GetPolygonVertexNormal(polIndex, 0, normalVec4);
		data->normals[polIndex] = vec3(normalVec4[0], normalVec4[1], normalVec4[2]);
	}
	printf("��غ�ݎ擾����\n");
	return true;
}

void GetBoneData(fbxsdk::FbxImporter* importer, fbxsdk::FbxScene* scene) {
	auto mesh = scene->GetSrcObject<FbxMesh>();
	fbxsdk::FbxSkin* pSkin = static_cast<fbxsdk::FbxSkin*>(mesh->GetDeformer(0));

	if (mesh->GetDeformer(0)->GetDeformerType() != fbxsdk::FbxDeformer::EDeformerType::eSkin) {
		printf("�f�t�H�[�}�[�̎�ނ��قȂ�܂�\n");
		return;
	}

	int ClusterCount = pSkin->GetClusterCount();

	Bone* boneList = (Bone*)malloc(sizeof(Bone) * ClusterCount);

	for (int i = 0; i < ClusterCount; i++)
	{
		fbxsdk::FbxCluster* pCluster = pSkin->GetCluster(i);
		//fbxsdk::FbxAMatrix initMat =FbxAMatrix();
		//pCluster->GetTransformLinkMatrix(initMat);

		//�{�[���̃f�t�H���g�̃��[�J�����W���擾
		FbxNode* node = pCluster->GetLink();
		//printf("%s:\n", node->GetName());
		//printf("local transform %f,%f,%f\n", node->LclTranslation.Get()[0], node->LclTranslation.Get()[1], node->LclTranslation.Get()[2]);
		//printf("local rotation %f,%f,%f\n", node->LclRotation.Get()[0], node->LclRotation.Get()[1], node->LclRotation.Get()[2]);
		// �����̓O���[�o�����W���~��������
		vec3 defaultTransform = vec3(node->LclTranslation.Get()[0], node->LclTranslation.Get()[1], node->LclTranslation.Get()[2]);
		vec3 defaultRotation = vec3(node->LclRotation.Get()[0], node->LclRotation.Get()[1], node->LclRotation.Get()[2]);
		boneList[i] = Bone(node->GetName(), defaultTransform, defaultRotation,
			defaultTransform, defaultRotation,
			pCluster->GetControlPointIndices(), pCluster->GetControlPointWeights());
	}
}

void GetAnimationData(fbxsdk::FbxImporter* importer, fbxsdk::FbxScene* scene) {
	auto mesh = scene->GetSrcObject<FbxMesh>();
	fbxsdk::FbxSkin* pSkin = static_cast<fbxsdk::FbxSkin*>(mesh->GetDeformer(0));

	if (mesh->GetDeformer(0)->GetDeformerType() != fbxsdk::FbxDeformer::EDeformerType::eSkin) {
		printf("�f�t�H�[�}�[�̎�ނ��قȂ�܂�\n");
		return;
	}

	//�A�j���[�V�������擾
	int animStackCount = importer->GetAnimStackCount();
	FbxTakeInfo* pFbxTakeInfo = importer->GetTakeInfo(0);
	FbxLongLong start = pFbxTakeInfo->mLocalTimeSpan.GetStart().Get();
	FbxLongLong stop = pFbxTakeInfo->mLocalTimeSpan.GetStop().Get();
	FbxLongLong oneFrameValue = FbxTime::GetOneFrameValue(FbxTime::eFrames60);
	int framecount = (stop - start) / oneFrameValue;
	printf("�A�j���[�V�����̍��v�t���[����%d\n", framecount);



	BonePoseData** animationData = (BonePoseData**)malloc(sizeof(BonePoseData*) * framecount);
	for (int i = 0; i < framecount; i++) {
		//�w��t���[���ł̉�]���擾�H�ŏ��͉�]����������
		int frame = oneFrameValue * i;
		animationData[i] = createBonePoseData(pSkin->GetCluster(0)->GetLink(),frame);
		//�e�q�֌W�擾
		//printAllNode(pSkin->GetCluster(0)->GetLink(), 0);
	}
	
}

bool CreateFBXMeshData(const std::string& filePath, MeshData* data)
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

	if (!GetMeshData(manager, scene, data)) {
		printf("���b�V�����̎擾�Ɏ��s\n");
		return false;
	}

	GetBoneData(importer, scene);
	GetAnimationData(importer, scene);	

	// �}�l�[�W���[�A�V�[���̔j��
	importer->Destroy();
	scene->Destroy();
	manager->Destroy();
	return true;
}

