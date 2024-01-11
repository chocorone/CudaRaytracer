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
	cudaMallocManaged((void**)&fbxData->mesh, sizeof(MeshData*));
	fbxData->mesh->nPoints = nPoints;
	nTriangles = mesh->GetPolygonCount();
	fbxData->mesh->nTriangles = nTriangles;

	printf("データ数取得完了\n");

	cudaMallocManaged((void**)&fbxData->mesh->points, nPoints * sizeof(vec3));
	cudaDeviceSynchronize();
	cudaMallocManaged((void**)&fbxData->mesh->idxVertex, nTriangles * sizeof(vec3));
	cudaDeviceSynchronize();
	cudaMallocManaged((void**)&fbxData->mesh->normals, nTriangles * sizeof(vec3));
	cudaDeviceSynchronize();

	// 頂点座標情報のリストを生成
	for (int i = 0; i < nPoints; i++)
	{
		// 頂点座標を読み込んで設定
		auto point = mesh->GetControlPointAt(i);

		vec3 vertex = vec3(point[0], point[1], point[2]);
		fbxData->mesh->points[i] = vertex;
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
		fbxData->mesh->idxVertex[polIndex] = vec3(one, two, three);
		//法線
		FbxVector4 normalVec4;
		mesh->GetPolygonVertexNormal(polIndex, 0, normalVec4);
		fbxData->mesh->normals[polIndex] = vec3(normalVec4[0], normalVec4[1], normalVec4[2]);
	}
	printf("ﾎﾟﾘｺﾞﾝ取得完了\n");
	return true;
}

void GetBoneData(fbxsdk::FbxImporter* importer, fbxsdk::FbxScene* scene, FBXObject* fbxData) {
	auto mesh = scene->GetSrcObject<FbxMesh>();
	fbxsdk::FbxSkin* pSkin = static_cast<fbxsdk::FbxSkin*>(mesh->GetDeformer(0));

	if (mesh->GetDeformer(0)->GetDeformerType() != fbxsdk::FbxDeformer::EDeformerType::eSkin) {
		printf("デフォーマーの種類が異なります\n");
		return;
	}

	int ClusterCount = pSkin->GetClusterCount();

	printf("ボーン数%d\n", ClusterCount);
	fbxData->boneCount = ClusterCount;
	cudaMallocManaged((void**)&fbxData->boneList, sizeof(Bone) * ClusterCount);
	cudaDeviceSynchronize();

	for (int i = 0; i < ClusterCount; i++)
	{
		fbxsdk::FbxCluster* pCluster = pSkin->GetCluster(i);
		//ボーンのデフォルトのグローバル座標を取得
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
			
			//ここで2フレームあたりの頂点の位置を計算して入れる
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
	//アニメーション情報取得
	int animStackCount = importer->GetAnimStackCount();
	FbxTakeInfo* pFbxTakeInfo = importer->GetTakeInfo(0);
	FbxLongLong start = pFbxTakeInfo->mLocalTimeSpan.GetStart().Get();
	FbxLongLong stop = pFbxTakeInfo->mLocalTimeSpan.GetStop().Get();
	FbxLongLong oneFrameValue = FbxTime::GetOneFrameValue(FbxTime::eFrames60);
	int framecount = (stop - start) / oneFrameValue;
	printf("アニメーションの合計フレーム数%d\n", framecount);
	endFrame = framecount-1;
	
	auto mesh = scene->GetSrcObject<FbxMesh>();
	fbxsdk::FbxSkin* pSkin = static_cast<fbxsdk::FbxSkin*>(mesh->GetDeformer(0));

	if (mesh->GetDeformer(0)->GetDeformerType() != fbxsdk::FbxDeformer::EDeformerType::eSkin) {
		printf("デフォーマーの種類が異なります\n");
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
		printf("%dフレーム目読み込み完了\n", frameIndex);
	}	
}

bool CreateFBXData(const std::string& filePath, FBXObject* fbxData, FBXAnimationData* animationData, int& endFrame)
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

	if (!GetMeshData(manager, scene, fbxData)) {
		printf("メッシュ情報の取得に失敗\n");
		return false;
	}

	GetBoneData(importer, scene,fbxData);
	animationData->object = fbxData;
	GetAnimationData(importer, scene,animationData,fbxData, endFrame);

	// マネージャー、シーンの破棄
	importer->Destroy();
	scene->Destroy();
	manager->Destroy();
	return true;
}

