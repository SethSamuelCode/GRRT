#ifndef GRRT_VEC3_H
#define GRRT_VEC3_H

#include <cmath>

namespace grrt {

struct Vec3 {
    double data[3]{};

    double& operator[](int i) { return data[i]; }
    const double& operator[](int i) const { return data[i]; }

    double r() const { return data[0]; }
    double g() const { return data[1]; }
    double b() const { return data[2]; }

    Vec3 operator+(const Vec3& o) const {
        return {{data[0]+o[0], data[1]+o[1], data[2]+o[2]}};
    }

    Vec3 operator-(const Vec3& o) const {
        return {{data[0]-o[0], data[1]-o[1], data[2]-o[2]}};
    }

    Vec3 operator*(double s) const {
        return {{data[0]*s, data[1]*s, data[2]*s}};
    }

    // Component-wise multiply (color modulation)
    Vec3 operator*(const Vec3& o) const {
        return {{data[0]*o[0], data[1]*o[1], data[2]*o[2]}};
    }

    Vec3& operator+=(const Vec3& o) {
        data[0] += o[0]; data[1] += o[1]; data[2] += o[2];
        return *this;
    }

    double max_component() const {
        return std::fmax(data[0], std::fmax(data[1], data[2]));
    }
};

inline Vec3 operator*(double s, const Vec3& v) {
    return v * s;
}

} // namespace grrt

#endif
