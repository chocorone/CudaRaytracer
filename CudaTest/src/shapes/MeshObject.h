#pragma once
#include "../core/vec3.h"

class MeshData {
public:
	int nPoints;
	int nTriangles;
	vec3* points;
	vec3* idxVertex;
	vec3* normals;
};