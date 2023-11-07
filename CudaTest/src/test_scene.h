#pragma once

#include <curand.h>
#include <curand_kernel.h>

#include <float.h>


#include "hitable/hitable.h"
#include "hitable/bvh.h"
#include "hitable/hitable_list.h"
#include "core/camera.h"
#include "shapes/sphere.h"
#include "shapes/triangle.h"
#include "shapes/transform.h"
#include "shapes/box.h"
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

/*
__device__ void bvh_scene(Hitable** list,
    Hitable** world,
    curandState* state) {
    int nb = 10;
    Hitable** boxlist1 = new Hitable * [1000];
    Material* ground = new Lambertian(new ConstantTexture(vec3(0.48, 0.83, 0.53)));

    int b = 0;
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < nb; j++) {
            float w = 100;
            float x0 = -1000 + i * w;
            float z0 = -1000 + j * w;
            float y0 = 0;
            float x1 = x0 + w;
            float y1 = 100 * (rand(state) + 0.01);
            float z1 = z0 + w;
            boxlist1[b++] = new Box(vec3(x0, y0, z0), vec3(x1, y1, z1), ground);
        }
    }

    int l = 0;
    list[l++] = new BVHNode(boxlist1, b, 0, 1, state);

    *world = new HitableList(list, l);
}*/

__device__ void cornell_box_scene(Hitable** list,
    Hitable** world) {
    int i = 0;
    Material* red = new   Lambertian(new ConstantTexture(vec3(0.65, 0.05, 0.05)));
    Material* white = new   Lambertian(new ConstantTexture(vec3(0.73, 0.73, 0.73)));
    Material* green = new   Lambertian(new ConstantTexture(vec3(0.12, 0.45, 0.15)));
    Material* light = new DiffuseLight(new ConstantTexture(vec3(15, 15, 15)));

    Material* diele = new Dielectric(0.5);
    Material* metal = new Metal(vec3(0.7, 0.6, 0.5), 0.3);

    list[i++] = new FlipNormals(new RectangleYZ(0, 555, 0, 555, 555, green));
    list[i++] = (new RectangleYZ(0, 555, 0, 555, 0, red));
    list[i++] = (new RectangleXZ(213, 343, 227, 332, 554, light));
    list[i++] = new FlipNormals(new RectangleXZ(0, 555, 0, 555, 555, white));
    list[i++] = (new RectangleXZ(0, 555, 0, 555, 0, white));
    list[i++] = new FlipNormals(new RectangleXY(0, 555, 0, 555, 555, white));

    list[i++] = new Translate(new Rotate(new Box(vec3(0, 0, 0), vec3(165, 165, 165), metal), -18), vec3(130, 0, 65));
    list[i++] = new Translate(new Rotate(new Box(vec3(0, 0, 0), vec3(165, 330, 165), diele), 18), vec3(265, 0, 295));


    *world = new HitableList(list, i);
}

__device__ void bunny_inside_cornell_box(Hitable** world,
    Hitable** list,
    vec3* points,
    vec3* idxVertex,
    int np, int nt,
    curandState* state) {
    int i = 0;
    Material* red = new   Lambertian(new ConstantTexture(vec3(0.65, 0.05, 0.05)));
    Material* white = new   Lambertian(new ConstantTexture(vec3(0.73, 0.73, 0.73)));
    Material* green = new   Lambertian(new ConstantTexture(vec3(0.12, 0.45, 0.15)));
    Material* light = new DiffuseLight(new ConstantTexture(vec3(15, 15, 15)));

    Material* diele = new Dielectric(0.5);
    Material* metal = new Metal(vec3(0.7, 0.6, 0.5), 0.3);

    list[i++] = new FlipNormals(new RectangleYZ(0, 555, 0, 555, 555, green));
    list[i++] = (new RectangleYZ(0, 555, 0, 555, 0, red));
    list[i++] = (new RectangleXZ(213, 343, 227, 332, 554, light));
    list[i++] = new FlipNormals(new RectangleXZ(0, 555, 0, 555, 555, white));
    list[i++] = (new RectangleXZ(0, 555, 0, 555, 0, white));
    list[i++] = new FlipNormals(new RectangleXY(0, 555, 0, 555, 555, white));

    list[i++] = new Translate(new Rotate(new Box(vec3(0, 0, 0), vec3(165, 165, 165), metal), -18), vec3(130, 0, 65));
    list[i++] = new Translate(new Rotate(new Box(vec3(0, 0, 0), vec3(165, 330, 165), diele), 18), vec3(265, 0, 295));

    Material* bunny = new Metal(vec3(0.4, 0.7, 0.5), 0.5f * rand(state));

    int l = 0;
    for (int i = 0; i < nt; i++) {
        vec3 idx = idxVertex[i];
        vec3 v[3] = { points[int(idx[2])], points[int(idx[1])], points[int(idx[0])] };
        list[l++] = new Triangle(v, bunny, true);
    }

    *world = new HitableList(list, i);
}