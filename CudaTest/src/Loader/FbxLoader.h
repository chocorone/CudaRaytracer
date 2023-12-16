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
	data->nPoints =nPoints;
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

	//�{�[���擾
	/*int DeformerCount = mesh->GetDeformerCount();

	fbxsdk::FbxSkin* pSkin = static_cast<fbxsdk::FbxSkin*>(mesh->GetDeformer(0));
	//�e�q�֌W�擾
	printAllNode(pSkin->GetCluster(0)->GetLink(), 0);

	/*
	printf("�łӂ��[�܁[���F%d\n", DeformerCount);
	for (int i = 0; i < DeformerCount; ++i)
	{
		printf("%s\n", mesh->GetDeformer(i)->GetName());
		if (mesh->GetDeformer(i)->GetDeformerType() == fbxsdk::FbxDeformer::EDeformerType::eSkin)
		{
			//�{�[���擾
			fbxsdk::FbxSkin* pSkin = static_cast<fbxsdk::FbxSkin*>(mesh->GetDeformer(i));
			int ClusterCount = pSkin->GetClusterCount();
			for (int i = 0; i < ClusterCount; ++i)
			{
				fbxsdk::FbxCluster* pCluster = pSkin->GetCluster(i);

				if (pCluster->GetLinkMode() != fbxsdk::FbxCluster::eTotalOne)
				{
					int ControlPointIndicesCount = pCluster->GetControlPointIndicesCount();
					printf("�{�[�����F%s�@�e�����_���F%d\n", pCluster->GetNameOnly(), ControlPointIndicesCount);
					//�{�[���̊e���_�ւ̉e���擾
					for (int j = 0; j < ControlPointIndicesCount; ++j)
					{
						printf("%f\n", (pCluster->GetControlPointWeights())[j]);
					}
				}

			}
		}
	}

	//�A�j���[�V�������擾
	int animStackCount = importer->GetAnimStackCount();
	FbxTakeInfo* pFbxTakeInfo = importer->GetTakeInfo(0);
	FbxLongLong start = pFbxTakeInfo->mLocalTimeSpan.GetStart().Get();
	FbxLongLong stop = pFbxTakeInfo->mLocalTimeSpan.GetStop().Get();

	FbxLongLong oneFrameValue = FbxTime::GetOneFrameValue(FbxTime::eFrames60);

	int framecount = (stop - start) / oneFrameValue;
	printf("�A�j���[�V�����̍��v�t���[����%d\n", framecount);

	//�|�[�Y���擾
	printf("�{�[���擾����\n");
	*/

	// �}�l�[�W���[�A�V�[���̔j��
	importer->Destroy();
	scene->Destroy();
	manager->Destroy();
	return true;
}

