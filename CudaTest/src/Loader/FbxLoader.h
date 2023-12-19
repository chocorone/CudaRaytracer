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

	for (int i = 0; i < ClusterCount; i++)
	{
		fbxsdk::FbxCluster* pCluster = pSkin->GetCluster(i);
		//fbxsdk::FbxAMatrix initMat =FbxAMatrix();
		//pCluster->GetTransformLinkMatrix(initMat);

		//ボーンのデフォルトのローカル座標を取得
		FbxNode* node = pCluster->GetLink();
		//printf("%s:\n", node->GetName());
		//printf("local transform %f,%f,%f\n", node->LclTranslation.Get()[0], node->LclTranslation.Get()[1], node->LclTranslation.Get()[2]);
		//printf("local rotation %f,%f,%f\n", node->LclRotation.Get()[0], node->LclRotation.Get()[1], node->LclRotation.Get()[2]);
		// ここはグローバル座標が欲しいかも
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



	BonePoseData** animationData = (BonePoseData**)malloc(sizeof(BonePoseData*) * framecount);
	for (int i = 0; i < framecount; i++) {
		//指定フレームでの回転を取得？最初は回転だけ正しい
		int frame = oneFrameValue * i;
		animationData[i] = createBonePoseData(pSkin->GetCluster(0)->GetLink(),frame);
		//親子関係取得
		//printAllNode(pSkin->GetCluster(0)->GetLink(), 0);
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
	GetAnimationData(importer, scene);	

	// マネージャー、シーンの破棄
	importer->Destroy();
	scene->Destroy();
	manager->Destroy();
	return true;
}

