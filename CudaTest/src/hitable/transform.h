#pragma once

#include <float.h>


class Transform {
public:
    __host__ __device__ Transform() { position = vec3(); rotation = vec3(); scale = vec3(1); }
    __host__ __device__ Transform(vec3 p, vec3 r, vec3 s) : position(p), rotation(r), scale(s)
    {}
     __device__ Ray TransformRay(const Ray& r)
    {
        return  TranslateRay(RotateRay(ScaleRay(r)));
    }

     __device__ void ResetTransform() {
         position = vec3(0);
         rotation = vec3(0);
         scale = vec3(1);
     }

     __device__ void TransformAABB(AABB& aabb) {
         //âÒì]?ä‘à·Ç¡ÇƒÇ¢ÇÈÅH
         vec3 rotetedMin = rotate(aabb.min(), rotation);
         vec3 rotetedMax = rotate(aabb.max(), rotation);
         vec3 min = minVec3(rotetedMin, rotetedMax);
         vec3 max = maxVec3(rotetedMin,rotetedMax);

         //à íuà⁄ìÆ
         aabb =  AABB(min + position, max + position);
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
        list_size = 0;
    }

    Transform** list;
    int list_size;
};