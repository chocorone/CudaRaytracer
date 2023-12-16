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

	// インポーター初期化
	auto importer = FbxImporter::Create(manager, "");
	if (!importer->Initialize(filePath.c_str(), -1, manager->GetIOSettings()))
	{
		printf("インポーター初期化失敗\n");
		return false;
	}

	// シーン作成
	auto scene = FbxScene::Create(manager, "");
	importer->Import(scene);

	// 三角ポリゴンへのコンバート
	FbxGeometryConverter geometryConverter(manager);
	if (!geometryConverter.Triangulate(scene, true))
	{
		printf("ﾎﾟﾘｺﾞﾝ取得失敗\n");
		return false;
	}
	// メッシュ取得
	auto mesh = scene->GetSrcObject<FbxMesh>();
	if (!mesh)
	{
		printf("メッシュ取得失敗\n");
		return false;
	}
	int nPoints, nTriangles;
	nPoints = mesh->GetControlPointsCount();
	data->nPoints =nPoints;
	nTriangles = mesh->GetPolygonCount();
	data->nTriangles = nTriangles;

	printf("データ数取得完了\n");

	cudaMallocManaged((void**)&data->points, nPoints * sizeof(vec3));
	cudaMallocManaged((void**)&data->idxVertex, nTriangles * sizeof(vec3));
	cudaMallocManaged((void**)&data->normals, nTriangles * sizeof(vec3));

	cudaDeviceSynchronize();
	cudaGetLastError();
	cudaDeviceSynchronize();

	// 頂点座標情報のリストを生成
	for (int i = 0; i < nPoints; i++)
	{
		// 頂点座標を読み込んで設定
		auto point = mesh->GetControlPointAt(i);

		vec3 vertex = vec3(point[0], point[1], point[2]);
		data->points[i] = vertex;
	}
	printf("頂点取得完了\n");

	// 頂点毎の情報を取得する
	// 3角形ﾎﾟﾘｺﾞﾝに限定する
	for (int polIndex = 0; polIndex < nTriangles; polIndex++) // ポリゴン毎のループ
	{
		// インデックス座標
		int one = mesh->GetPolygonVertex(polIndex, 0);
		int two = mesh->GetPolygonVertex(polIndex, 1);
		int three = mesh->GetPolygonVertex(polIndex, 2);
		data->idxVertex[polIndex] = vec3(one, two, three);
		//法線
		FbxVector4 normalVec4;
		mesh->GetPolygonVertexNormal(polIndex, 0, normalVec4);
		data->normals[polIndex] = vec3(normalVec4[0], normalVec4[1], normalVec4[2]);
	}
	printf("ﾎﾟﾘｺﾞﾝ取得完了\n");	

	//ボーン取得
	/*int DeformerCount = mesh->GetDeformerCount();

	fbxsdk::FbxSkin* pSkin = static_cast<fbxsdk::FbxSkin*>(mesh->GetDeformer(0));
	//親子関係取得
	printAllNode(pSkin->GetCluster(0)->GetLink(), 0);

	/*
	printf("でふぉーまー数：%d\n", DeformerCount);
	for (int i = 0; i < DeformerCount; ++i)
	{
		printf("%s\n", mesh->GetDeformer(i)->GetName());
		if (mesh->GetDeformer(i)->GetDeformerType() == fbxsdk::FbxDeformer::EDeformerType::eSkin)
		{
			//ボーン取得
			fbxsdk::FbxSkin* pSkin = static_cast<fbxsdk::FbxSkin*>(mesh->GetDeformer(i));
			int ClusterCount = pSkin->GetClusterCount();
			for (int i = 0; i < ClusterCount; ++i)
			{
				fbxsdk::FbxCluster* pCluster = pSkin->GetCluster(i);

				if (pCluster->GetLinkMode() != fbxsdk::FbxCluster::eTotalOne)
				{
					int ControlPointIndicesCount = pCluster->GetControlPointIndicesCount();
					printf("ボーン名：%s　影響頂点数：%d\n", pCluster->GetNameOnly(), ControlPointIndicesCount);
					//ボーンの各頂点への影響取得
					for (int j = 0; j < ControlPointIndicesCount; ++j)
					{
						printf("%f\n", (pCluster->GetControlPointWeights())[j]);
					}
				}

			}
		}
	}

	//アニメーション情報取得
	int animStackCount = importer->GetAnimStackCount();
	FbxTakeInfo* pFbxTakeInfo = importer->GetTakeInfo(0);
	FbxLongLong start = pFbxTakeInfo->mLocalTimeSpan.GetStart().Get();
	FbxLongLong stop = pFbxTakeInfo->mLocalTimeSpan.GetStop().Get();

	FbxLongLong oneFrameValue = FbxTime::GetOneFrameValue(FbxTime::eFrames60);

	int framecount = (stop - start) / oneFrameValue;
	printf("アニメーションの合計フレーム数%d\n", framecount);

	//ポーズ情報取得
	printf("ボーン取得完了\n");
	*/

	// マネージャー、シーンの破棄
	importer->Destroy();
	scene->Destroy();
	manager->Destroy();
	return true;
}

