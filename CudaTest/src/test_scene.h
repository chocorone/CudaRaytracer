#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include <float.h>

#include "hitable/bvh.h"
#include "core/camera.h"
#include "shapes/triangle.h"
#include "material/material.h"



__device__ void draw_one_mesh(Hitable** mesh,
    Hitable** triangles,
    vec3* points,
    vec3* idxVertex,
    int np, int nt,
    curandState* state) {

    Material* mat = new DiffuseLight(new ConstantTexture(vec3(0.4, 0.7, 0.5)));

    int l = 0;
    for (int i = 0; i < nt; i++) {
        vec3 idx = idxVertex[i];
        vec3 v[3] = { points[int(idx[2])], points[int(idx[1])], points[int(idx[0])] };
        triangles[l++] = new Triangle(v, mat, true);
    }
    *mesh = new BVHNode(triangles, l, 0, 1, state);
    
}