#pragma once

#include <float.h>


class Transform {
public:
    __host__ __device__ Transform() { position = vec3(); rotation = vec3(); scale = vec3(1); }
    __host__ __device__ Transform(vec3 p, vec3 r, vec3 s) : position(p), rotation(r), scale(s)
    {
        //printf("set transform\n");
    }
     __device__ Ray TransformRay(const Ray& r)
    {
        return  TranslateRay(RotateRay(ScaleRay(r)));
    }

    vec3 position;
    vec3 rotation;
    vec3 scale;

private:
    __device__ Ray TranslateRay(const Ray& r) const {
        Ray moved_r(r.origin() - position, r.direction(), r.time());
        return moved_r;
    }

    __device__ Ray RotateRay(const Ray& r) const {
        
        Ray rotate_r(rotate(r.origin(),rotation), rotate(r.direction(),rotation), r.time());

        return rotate_r;
    }

    __device__ Ray ScaleRay(const Ray& r) const {
        vec3 dir = r.direction() / scale;
        Ray scaled_r(r.origin(), unit_vector(dir), r.time()*dir.length());
        return scaled_r;
    }

};

class TransformList {
public:
     __device__ TransformList() { list = new Transform * (); list_size = 0; }
     __device__ TransformList(Transform** l, int n) { list = l; list_size = n; }
     __device__ void append(Transform* data)
    {
        Transform** tmp = (Transform**)malloc(sizeof(Transform*) * list_size);

        for (int i = 0; i < list_size; i++)
        {
            tmp[i] = list[i];
        }

        free(list);

        list_size++;

        list = (Transform**)malloc(sizeof(Transform*) * list_size);

        for (int i = 0; i < list_size - 1; i++)
        {
            list[i] = tmp[i];
        }
        list[list_size - 1] = data;

        free(tmp);
    }
     __device__  void freeMemory()
    {
        free(list);
        /*for (int i = 0; i < list_size; i++) {
            delete* (list + i);
        }*/
        list_size = 0;
    }

    Transform** list;
    int list_size;
};