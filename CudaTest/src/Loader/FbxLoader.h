#pragma once
#include <fbxsdk.h>
#include <vector>
#include <array>
#include <string>
#include "../shapes/MeshObject.h"
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

BonePoseData* createBonePoseData(fbxsdk::FbxNode* object,int frame) 
{
	BonePoseData** childList = (BonePoseData**)malloc(sizeof(BonePoseData) * object->GetChildCount());
	printf("%s\n", object->GetName());
	// とりあえずGlobalを取得
	fbxsdk::FbxAMatrix amat = object->EvaluateGlobalTransform(frame);
	printf("global transform %f, %f, %f\n", amat.GetT()[0], amat.GetT()[1], amat.GetT()[2]);
	printf("global rotation  %f, %f, %f\n", amat.GetR()[0], amat.GetR()[1], amat.GetR()[2]);

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
	data->nPoints = nPoints;
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
	return true;
}

void GetBoneData(fbxsdk::FbxImporter* importer, fbxsdk::FbxScene* scene) {
	auto mesh = scene->GetSrcObject<FbxMesh>();
	fbxsdk::FbxSkin* pSkin = static_cast<fbxsdk::FbxSkin*>(mesh->GetDeformer(0));

	if (mesh->GetDeformer(0)->GetDeformerType() != fbxsdk::FbxDeformer::EDeformerType::eSkin) {
		printf("デフォーマーの種類が異なります\n");
		return;
	}

	int ClusterCount = pSkin->GetClusterCount();

	Bone* boneList = (Bone*)malloc(sizeof(Bone) * ClusterCount);
	std::map<const char*, int> boneIndex;

	for (int i = 0; i < ClusterCount; i++)
	{
		fbxsdk::FbxCluster* pCluster = pSkin->GetCluster(i);
		//ボーンのデフォルトのグローバル座標を取得
		FbxNode* node = pCluster->GetLink();
		fbxsdk::FbxAMatrix amat = node->EvaluateGlobalTransform();
		vec3 defaultTransform = vec3(amat.GetT()[0], amat.GetT()[1], amat.GetT()[2]);
		vec3 defaultRotation = vec3(amat.GetR()[0], amat.GetR()[1], amat.GetR()[2]);
		
		boneList[i] = Bone(node->GetName(), defaultTransform, defaultRotation,
			defaultTransform, defaultRotation,
			pCluster->GetControlPointIndices(), pCluster->GetControlPointWeights());
		boneIndex[node->GetName()] = i;

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

void GetAnimationData(fbxsdk::FbxImporter* importer, fbxsdk::FbxScene* scene, BonePoseData** animationData) {
	auto mesh = scene->GetSrcObject<FbxMesh>();
	fbxsdk::FbxSkin* pSkin = static_cast<fbxsdk::FbxSkin*>(mesh->GetDeformer(0));

	if (mesh->GetDeformer(0)->GetDeformerType() != fbxsdk::FbxDeformer::EDeformerType::eSkin) {
		printf("デフォーマーの種類が異なります\n");
		return;
	}

	//アニメーション情報取得
	int animStackCount = importer->GetAnimStackCount();
	FbxTakeInfo* pFbxTakeInfo = importer->GetTakeInfo(0);
	FbxLongLong start = pFbxTakeInfo->mLocalTimeSpan.GetStart().Get();
	FbxLongLong stop = pFbxTakeInfo->mLocalTimeSpan.GetStop().Get();
	FbxLongLong oneFrameValue = FbxTime::GetOneFrameValue(FbxTime::eFrames60);
	int framecount = (stop - start) / oneFrameValue;
	printf("アニメーションの合計フレーム数%d\n", framecount);

	//animationData = (BonePoseData**)malloc(sizeof(BonePoseData*) * framecount);
	cudaMallocManaged((void**)&animationData, sizeof(BonePoseData*) * framecount);
	for (int i = 0; i < framecount; i++) {
		//指定フレームでの回転を取得？最初は回転だけ正しい
		int frame = oneFrameValue * i;
		animationData[i] = createBonePoseData(pSkin->GetCluster(0)->GetLink(),frame);
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

	if (!GetMeshData(manager, scene, data)) {
		printf("メッシュ情報の取得に失敗\n");
		return false;
	}

	GetBoneData(importer, scene);
	BonePoseData** animationData;
	cudaMallocManaged((void**)&animationData, sizeof(BonePoseData**));
	GetAnimationData(importer, scene,animationData);	

	// マネージャー、シーンの破棄
	importer->Destroy();
	scene->Destroy();
	manager->Destroy();
	return true;
}

