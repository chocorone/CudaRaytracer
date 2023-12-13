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
	// マネージャー初期化
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
	importer->Destroy();

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

	nPoints = mesh->GetControlPointsCount();
	nTriangles = mesh->GetPolygonCount();
	
	printf("データ数取得完了\n");

	// マネージャー、シーンの破棄
	scene->Destroy();
	manager->Destroy();
	return true;
}

bool FBXLoad(const std::string& filePath, vec3* points, vec3* idxVertex,vec3* normal)
{
	printf("FBX読み込み開始\n");

	// マネージャー初期化
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
	importer->Destroy();

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

	int nPoints = mesh->GetControlPointsCount();
	// 頂点座標情報のリストを生成
	for (int i = 0; i < nPoints; i++)
	{
		// 頂点座標を読み込んで設定
		auto point = mesh->GetControlPointAt(i);

		vec3 vertex = vec3(point[0], point[1], point[2]);
		points[i] = vertex;
	}
	printf("頂点取得完了\n");

	int nTriangles = mesh->GetPolygonCount();
	// 頂点毎の情報を取得する
	// 3角形ﾎﾟﾘｺﾞﾝに限定する
	for (int polIndex = 0; polIndex < nTriangles; polIndex++) // ポリゴン毎のループ
	{
		// インデックス座標
		int one = mesh->GetPolygonVertex(polIndex, 0);
		int two = mesh->GetPolygonVertex(polIndex, 1);
		int three = mesh->GetPolygonVertex(polIndex, 2);
		idxVertex[polIndex] = vec3(one, two, three);
		//法線
		FbxVector4 normalVec4;
		mesh->GetPolygonVertexNormal(polIndex, 0, normalVec4);
		normal[polIndex] = vec3(normalVec4[0], normalVec4[1], normalVec4[2]);
	}
	printf("ﾎﾟﾘｺﾞﾝ取得完了\n");

	//デフォーマー取得
	int DeformerCount = mesh->GetDeformerCount();
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
					printf("ボーン名：%s　影響頂点数：%d\n", pCluster->GetName(), ControlPointIndicesCount);
					//ボーンの各頂点への影響取得
					for (int j = 0; j < ControlPointIndicesCount; ++j) 
					{
						printf("%f\n", (pCluster->GetControlPointWeights())[j]);
					}
				}

			}
		}
	}
	printf("ボーン取得完了\n");

	// マネージャー、シーンの破棄
	scene->Destroy();
	manager->Destroy();
	return true;
}
