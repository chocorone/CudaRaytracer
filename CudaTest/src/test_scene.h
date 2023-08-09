#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include <float.h>

#include "hitable/bvh.h"
#include "hitable/hitable_list.h"
#include "core/camera.h"
#include "shapes/sphere.h"
#include "shapes/triangle.h"
#include "material/material.h"

__device__ float rand(curandState* state) {
    return float(curand_uniform(state));
}


/* It works */
__device__ void random_scene(Hitable** list,
    Hitable** world,
    curandState* state) {
    Texture* checker = new CheckerTexture(new ConstantTexture(vec3(0.2, 0.3, 0.1)),
        new ConstantTexture(vec3(0.9, 0.9, 0.9)));
    list[0] = new Sphere(vec3(0, -1000.0, -1), 1000, new Lambertian(checker));
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = rand(state);
            vec3 center(a + 0.9 * rand(state), 0.2, b + 0.9 * rand(state));
            if (choose_mat < 0.8f) {
                list[i++] = new MovingSphere(center, center + vec3(0, 0.5 * rand(state), 0), 0.0, 1.0, 0.2,
                    new Lambertian(new ConstantTexture(vec3(rand(state), rand(state), rand(state)))));
                continue;
            }
            else if (choose_mat < 0.95f) {
                list[i++] = new Sphere(center, 0.2,
                    new Metal(vec3(0.5f * (1.0f + rand(state)),
                        0.5f * (1.0f + rand(state)),
                        0.5f * (1.0f + rand(state))),
                        0.5f * rand(state)));
            }
            else {
                list[i++] = new Sphere(center, 0.2, new Dielectric(rand(state) * 2));
            }
        }
    }
    list[i++] = new Sphere(vec3(0, 1, 0), 1.0, new Dielectric(1.5));
    list[i++] = new Sphere(vec3(-4, 1, 0), 1.0, new Lambertian(checker));
    list[i++] = new Sphere(vec3(4, 1, 0), 1.0, new Metal(vec3(0.7, 0.6, 0.5), 0.0));
    *world = new HitableList(list, i);
}


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