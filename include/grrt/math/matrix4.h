#ifndef GRRT_MATRIX4_H
#define GRRT_MATRIX4_H

#include "grrt/math/vec4.h"

namespace grrt {

struct Matrix4 {
    double m[4][4]{};

    // Contract: result_μ = Σ_ν M_μν v^ν
    // Used for both lowering (g_lower.contract(v)) and raising (g_upper.contract(v))
    Vec4 contract(const Vec4& v) const {
        Vec4 result;
        for (int mu = 0; mu < 4; ++mu) {
            double sum = 0.0;
            for (int nu = 0; nu < 4; ++nu) {
                sum += m[mu][nu] * v[nu];
            }
            result[mu] = sum;
        }
        return result;
    }

    // Create a diagonal matrix
    static Matrix4 diagonal(double a, double b, double c, double d) {
        Matrix4 mat;
        mat.m[0][0] = a;
        mat.m[1][1] = b;
        mat.m[2][2] = c;
        mat.m[3][3] = d;
        return mat;
    }

    // Inverse of a diagonal matrix (sufficient for Schwarzschild)
    // For general metrics (Kerr), this will need a full 4x4 inverse later.
    Matrix4 inverse_diagonal() const {
        return diagonal(
            1.0 / m[0][0],
            1.0 / m[1][1],
            1.0 / m[2][2],
            1.0 / m[3][3]
        );
    }
};

} // namespace grrt

#endif
