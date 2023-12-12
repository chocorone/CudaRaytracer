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
#include "shapes/box.h"
#include "material/material.h"

__device__ float rand(curandState* state) {
    return float(curand_uniform(state));
}

// オブジェクトの生成
__global__ void append_test(HitableList** world)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *world = new HitableList();

        Texture* checker = new CheckerTexture(new ConstantTexture(vec3(0.2, 0.3, 0.1)),
            new ConstantTexture(vec3(0.9, 0.9, 0.9)));
        (*world)->append(new Sphere(new Transform(vec3(0, 1, 0), vec3(0), vec3(1)), 1.0, new Dielectric(1.5)));
        (*world)->append(new Sphere(new Transform(vec3(-4, 1, 0), vec3(0), vec3(1)), 1.0, new Lambertian(checker)));
        (*world)->append(new Sphere(new Transform(vec3(4, 1, 0), vec3(0), vec3(1)), 1.0, new Metal(vec3(0.7, 0.6, 0.5), 0.0)));
    }
}



//デバイス側の処理
__global__ void random_scene(Hitable** world, Hitable** list,curandState* state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        Texture* checker = new CheckerTexture(new ConstantTexture(vec3(0.2, 0.3, 0.1)),
            new ConstantTexture(vec3(0.9, 0.9, 0.9)));
        int i = 0;
        list[i++] = new Sphere(new Transform(vec3(0, -1000.0, -1), vec3(0), vec3(1)), 1000, new Lambertian(checker));
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = rand(state);
                vec3 center(a + 0.9 * rand(state), 0.2, b + 0.9 * rand(state));
                if (choose_mat < 0.8f) {
                    list[i++] = new Sphere(new Transform(center,vec3(0),vec3(1)), 0.2,
                        new Lambertian(new ConstantTexture(vec3(rand(state), rand(state), rand(state)))));
                    continue;
                }
                else if (choose_mat < 0.95f) {
                    list[i++] = new Sphere(new Transform(center, vec3(0), vec3(1)), 0.2,
                        new Metal(vec3(0.5f * (1.0f + rand(state)),
                            0.5f * (1.0f + rand(state)),
                            0.5f * (1.0f + rand(state))),
                            0.5f * rand(state)));
                }
                else {
                    list[i++] = new Sphere(new Transform(center, vec3(0), vec3(1)), 0.2, new Dielectric(rand(state) * 2));
                }
            }
        }
        list[i++] = new Sphere(new Transform(vec3(0, 1, 0), vec3(0), vec3(1)), 1.0, new Dielectric(1.5));
        list[i++] = new Sphere(new Transform(vec3(-4, 1, 0), vec3(0), vec3(1)), 1.0, new Lambertian(checker));
        list[i++] = new Sphere(new Transform(vec3(4, 1, 0), vec3(0), vec3(1)), 1.0, new Metal(vec3(0.7, 0.6, 0.5), 0.0));
        *world = new HitableList(list, i);
    }
}

/*
__global__ void draw_one_mesh(Hitable** world,Hitable** list,vec3* points,vec3* idxVertex,
    int np, int nt,curandState* state) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        Material* mat = new DiffuseLight(new ConstantTexture(vec3(0.4, 0.7, 0.5)));

        int l = 0;
        for (int i = 0; i < nt; i++) {
            vec3 idx = idxVertex[i];
            vec3 v[3] = { points[int(idx[2])], points[int(idx[1])], points[int(idx[0])] };
            list[l++] = new Triangle(v, mat, true);
        }

        *world = new BVHNode(list, l, 0, 1, state);
    }
}
*/


__global__ void draw_one_mesh_withNormal(Hitable** world, Hitable** list, vec3* points, vec3* idxVertex,vec3* normal,
    int np, int nt, curandState* state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        Material* mat = new Lambertian(new ConstantTexture(vec3(0.65, 0.05, 0.05)));
        //Material* mat = new DiffuseLight(new ConstantTexture(vec3(0.4, 0.7, 0.5)));
        
        int l = 0;
        for (int i = 0; i < nt; i++) {
            vec3 idx = idxVertex[i];
            vec3 v[3] = { points[int(idx[2])], points[int(idx[1])], points[int(idx[0])] };
            list[l++] = new Triangle(v,  normal[i], mat,false,new Transform(), true);
        }
        *world = new HitableList(list, l,new Transform(vec3(0,-10,0),vec3(-20,180,0),vec3(1.5)));
    }
}

__global__ void cornell_box_scene(Hitable** world, Hitable** list, curandState* state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        Material* red = new   Lambertian(new ConstantTexture(vec3(0.65, 0.05, 0.05)));
        Material* white = new   Lambertian(new ConstantTexture(vec3(0.73, 0.73, 0.73)));
        Material* green = new   Lambertian(new ConstantTexture(vec3(0.12, 0.45, 0.15)));
        Material* light = new DiffuseLight(new ConstantTexture(vec3(15, 15, 15)));

        Material* diele = new Dielectric(0.5);
        Material* metal = new Metal(vec3(0.7, 0.6, 0.5), 0.3);

        int listIndex = 0;
        //list[listIndex++] = new FlipNormals(new RectangleYZ(0, 555, 0, 555, 555, green));
        //list[listIndex++] = (new RectangleYZ(0, 555, 0, 555, 0, red));
        //list[listIndex++] = (new RectangleXZ(213, 343, 227, 332, 554, light));
        //list[listIndex++] = new FlipNormals(new RectangleXZ(0, 555, 0, 555, 555, white));
        //list[listIndex++] = (new RectangleXZ(0, 555, 0, 555, 0, white));
        //list[listIndex++] = new FlipNormals(new RectangleXY(0, 555, 0, 555, 555, white));
        list[listIndex++] = new Rectangle(red,true,new Transform(vec3(0,0,10),vec3(0,0,90),vec3(6,3,1)));
        //list[listIndex++] = new Sphere(new Transform(vec3(0),vec3(0),vec3(10,10,1)), 1, red);



        *world = new HitableList(list, listIndex);
    }
}

__global__ void create_camera_origin(Camera** camera, int nx, int ny) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        vec3 lookfrom(0, 0, 20);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0;
        float aperture = 0.0;
        float vfov = 60.0;


        *camera = new Camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            vfov,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
    }
}

__global__ void create_camera_for_cornelbox(Camera** camera, int nx, int ny) 
{
    if (threadIdx.x == 0 && blockIdx.x == 0) 
    {
        vec3 lookfrom(278, 278, -700);
        vec3 lookat(278, 278, 0);
        float dist_to_focus = 10.0;
        float aperture = 0.0;
        float vfov = 40.0;


        *camera = new Camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            vfov,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
    }
}

