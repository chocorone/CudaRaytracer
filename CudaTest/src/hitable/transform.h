#pragma once

#include <float.h>


class Transform {
public:
    __device__ Transform() { position = vec3(); rotation = vec3(); scale = vec3(1); }
    __device__ Transform(vec3 p, vec3 r, vec3 s) : position(p), rotation(r), scale(s)
    {
        //printf("set transform\n");
    }
    __device__ Ray TransformRay(const Ray& r)
    {
        return  TranslateRay(RotateRay(ScaleRay(r)));
    }

private:
    __device__ Ray TranslateRay(const Ray& r) const {
        Ray moved_r(r.origin() - position, r.direction(), r.time());
        return moved_r;
    }

    __device__ Ray RotateRay(const Ray& r) const {
        //xŽ²‰ñ“]
        float radiansX = (M_PI / 180.) * rotation.x();
        float sin_X = sin(radiansX);
        float cos_X = cos(radiansX);
        float radiansY = (M_PI / 180.) * rotation.y();
        float sin_Y = sin(radiansY);
        float cos_Y = cos(radiansY);
        float radiansZ = -(M_PI / 180.) * rotation.z();
        float sin_Z = sin(radiansZ);
        float cos_Z = cos(radiansZ);

        vec3 rotate0 = vec3(cos_Y*cos_Z,-cos_Y*sin_Z,sin_Y);
        vec3 rotate1 = vec3(sin_X * sin_Y * cos_Z + cos_X * sin_Z, -sin_X * sin_Y * sin_Z + cos_X * cos_Z, -sin_X * cos_Y);
        vec3 rotate2 = vec3(-cos_X * sin_Y * cos_Z + sin_X * sin_Z, cos_X * sin_Y * sin_Z + sin_X * cos_Z, cos_X * cos_Y);

        vec3 origin = r.origin();
        vec3 direction = r.direction();
        
        origin = vec3((origin * rotate0).sum(), (origin * rotate1).sum(), (origin * rotate2).sum());
        direction = vec3((direction *rotate0).sum(), (direction * rotate1).sum(), (direction * rotate2).sum());

        Ray rotate_r(origin, direction, r.time());

        return rotate_r;
    }

    __device__ Ray ScaleRay(const Ray& r) const {
        vec3 dir = r.direction() / scale;
        Ray scaled_r(r.origin(), unit_vector(dir), r.time()*dir.length());
        return scaled_r;
    }

    //public‚É‚µ‚½‚¢
    vec3 position;
    vec3 rotation;
    vec3 scale;
};
