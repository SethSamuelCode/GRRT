#ifndef GRRT_VEC4_H
#define GRRT_VEC4_H

namespace grrt {

struct Vec4 {
    double data[4]{};

    double& operator[](int i) { return data[i]; }
    const double& operator[](int i) const { return data[i]; }

    Vec4 operator+(const Vec4& o) const {
        return {{data[0]+o[0], data[1]+o[1], data[2]+o[2], data[3]+o[3]}};
    }

    Vec4 operator-(const Vec4& o) const {
        return {{data[0]-o[0], data[1]-o[1], data[2]-o[2], data[3]-o[3]}};
    }

    Vec4 operator*(double s) const {
        return {{data[0]*s, data[1]*s, data[2]*s, data[3]*s}};
    }
};

inline Vec4 operator*(double s, const Vec4& v) {
    return v * s;
}

} // namespace grrt

#endif
