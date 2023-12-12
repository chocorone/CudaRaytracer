#pragma once

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3 {
public:
    __host__ __device__ vec3() {}
    __host__ __device__ vec3(float e0) {
        e[0] = e0; e[1] = e0; e[2] = e0;
    }
    __host__ __device__ vec3(float e0, float e1, float e2) {
        e[0] = e0; e[1] = e1; e[2] = e2;
    }
    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; }

    __host__ __device__ inline vec3& operator+=(const vec3& v2);
    __host__ __device__ inline vec3& operator-=(const vec3& v2);
    __host__ __device__ inline vec3& operator*=(const vec3& v2);
    __host__ __device__ inline vec3& operator/=(const vec3& v2);

    __host__ __device__ inline vec3& operator+=(const float t);
    __host__ __device__ inline vec3& operator-=(const float t);
    __host__ __device__ inline vec3& operator*=(const float t);
    __host__ __device__ inline vec3& operator/=(const float t);

    __host__ __device__ inline float length() const {
        return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    }

    __host__ __device__ inline float squared_length() const {
        return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
    }

    __host__ __device__ inline void make_unit_vector();
    __host__ __device__ inline float sum();

    float e[3];
};


inline std::istream& operator>>(std::istream& is, vec3& t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream& os, const vec3& t) {
    os << "(" << t[0] << ", " << t[1] << ", " << t[2] << ")";
    return os;
}

__host__ __device__ inline void vec3::make_unit_vector() {
    float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

inline __host__ __device__ float vec3::sum()
{
    return e[0] + e[1] + e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline vec3 operator+(const vec3& v1, const float t) {
    return vec3(v1.e[0] + t, v1.e[1] + t, v1.e[2] + t);
}

__host__ __device__ inline vec3 operator-(const vec3& v1, const float t) {
    return vec3(v1.e[0] - t, v1.e[1] - t, v1.e[2] - t);
}

__host__ __device__ inline vec3 operator*(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3& v, const float t) {
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline float dot(const vec3& v1, const vec3& v2) {
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& v1, const vec3& v2) {
    return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
        (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
        (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__host__ __device__ float clip_single(float f, int min, int max) {
    if (f > max) return max;
    else if (f < min) return min;
    return f;
}

__host__ __device__ inline vec3 clip(const vec3& v, int min = 0.0f, int max = 1.0f) {
    vec3 vr(0, 0, 0);
    vr[0] = clip_single(v[0], min, max);
    vr[1] = clip_single(v[1], min, max);
    vr[2] = clip_single(v[2], min, max);
    return vr;
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3& v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3& v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline vec3& vec3::operator+=(const float t) {
    e[0] += t;
    e[1] += t;
    e[2] += t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const float t) {
    e[0] -= t;
    e[1] -= t;
    e[2] -= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t) {
    float k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}


__host__ __device__ inline vec3 lerp(float t,vec3 from, vec3 to) {
    return from + t*(to-from);
}

__host__ __device__ inline vec3 rotate(vec3 origin,vec3 rotation) {
    float radiansX = (M_PI / 180.) * rotation.x();
    float sin_X = sin(radiansX);
    float cos_X = cos(radiansX);
    float radiansY = (M_PI / 180.) * rotation.y();
    float sin_Y = sin(radiansY);
    float cos_Y = cos(radiansY);
    float radiansZ = -(M_PI / 180.) * rotation.z();
    float sin_Z = sin(radiansZ);
    float cos_Z = cos(radiansZ);

    vec3 rotate0 = vec3(cos_Y * cos_Z, -cos_Y * sin_Z, sin_Y);
    vec3 rotate1 = vec3(sin_X * sin_Y * cos_Z + cos_X * sin_Z, -sin_X * sin_Y * sin_Z + cos_X * cos_Z, -sin_X * cos_Y);
    vec3 rotate2 = vec3(-cos_X * sin_Y * cos_Z + sin_X * sin_Z, cos_X * sin_Y * sin_Z + sin_X * cos_Z, cos_X * cos_Y);

    origin = vec3((origin * rotate0).sum(), (origin * rotate1).sum(), (origin * rotate2).sum());
    return origin;
}

__host__ __device__ inline vec3 SLerp(vec3 start, vec3 end, float t) {

    // 2ƒxƒNƒgƒ‹ŠÔ‚ÌŠp“xi‰sŠp‘¤j
    float angle = acos(dot(start, end));

    // sinƒÆ
    float SinTh = sin(angle);

    // •âŠÔŒW”
    float Ps = sin(angle * (1 - t));
    float Pe = sin(angle * t);

    vec3 out = (Ps * start + Pe * end) / SinTh;

    return unit_vector(out);
}