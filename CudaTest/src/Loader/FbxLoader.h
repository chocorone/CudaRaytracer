#pragma once
#include <fbxsdk.h>
#include <vector>
#include <array>
#include <string>
#pragma comment(lib, "libfbxsdk-md.lib")
#pragma comment(lib, "libxml2-md.lib")
#pragma comment(lib, "zlib-md.lib")
#pragma comment(lib, "zlib-md.lib")

bool GetFBXVertexCount(const std::string& filePath, int& nPoints, int& nTriangles)
{
	// �}�l�[�W���[������
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
	importer->Destroy();

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

	nPoints = mesh->GetControlPointsCount();
	nTriangles = mesh->GetPolygonCount();
	
	printf("�f�[�^���擾����\n");

	// �}�l�[�W���[�A�V�[���̔j��
	scene->Destroy();
	manager->Destroy();
	return true;
}

bool FBXLoad(const std::string& filePath, vec3* points, vec3* idxVertex,vec3* normal)
{
	printf("FBX�ǂݍ��݊J�n\n");

	// �}�l�[�W���[������
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
	importer->Destroy();

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

	int nPoints = mesh->GetControlPointsCount();
	// ���_���W���̃��X�g�𐶐�
	for (int i = 0; i < nPoints; i++)
	{
		// ���_���W��ǂݍ���Őݒ�
		auto point = mesh->GetControlPointAt(i);

		vec3 vertex = vec3(point[0], point[1], point[2]);
		points[i] = vertex;
	}
	printf("���_�擾����\n");

	int nTriangles = mesh->GetPolygonCount();
	// ���_���̏����擾����
	// 3�p�`��غ�݂Ɍ��肷��
	for (int polIndex = 0; polIndex < nTriangles; polIndex++) // �|���S�����̃��[�v
	{
		// �C���f�b�N�X���W
		int one = mesh->GetPolygonVertex(polIndex, 0);
		int two = mesh->GetPolygonVertex(polIndex, 1);
		int three = mesh->GetPolygonVertex(polIndex, 2);
		idxVertex[polIndex] = vec3(one, two, three);
		//�@��
		FbxVector4 normalVec4;
		mesh->GetPolygonVertexNormal(polIndex, 0, normalVec4);
		normal[polIndex] = vec3(normalVec4[0], normalVec4[1], normalVec4[2]);
	}
	printf("��غ�ݎ擾����\n");

	//�f�t�H�[�}�[�擾
	int DeformerCount = mesh->GetDeformerCount();
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
					printf("�{�[�����F%s�@�e�����_���F%d\n", pCluster->GetName(), ControlPointIndicesCount);
					//�{�[���̊e���_�ւ̉e���擾
					for (int j = 0; j < ControlPointIndicesCount; ++j) 
					{
						printf("%f\n", (pCluster->GetControlPointWeights())[j]);
					}
				}

			}
		}
	}
	printf("�{�[���擾����\n");

	// �}�l�[�W���[�A�V�[���̔j��
	scene->Destroy();
	manager->Destroy();
	return true;
}
