#ifndef GRRT_CUDA_MATH_H
#define GRRT_CUDA_MATH_H

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/// @file cuda_math.h
/// @brief CUDA device-compatible math types mirroring the CPU grrt math types.
///
/// All structs use geometrized units (G = c = 1) and are compatible with
/// both host and device code via __host__ __device__ qualifiers.
/// Vec4 indices: (t=0, r=1, theta=2, phi=3) in Boyer-Lindquist coordinates.

#include <cmath>
#include <cuda_runtime.h>

namespace cuda {

// ---------------------------------------------------------------------------
// Vec3: 3-component double vector (used for RGB color / spatial directions)
// ---------------------------------------------------------------------------
struct Vec3 {
    double data[3];

    __host__ __device__ double& operator[](int i) { return data[i]; }
    __host__ __device__ const double& operator[](int i) const { return data[i]; }

    __host__ __device__ double r() const { return data[0]; }
    __host__ __device__ double g() const { return data[1]; }
    __host__ __device__ double b() const { return data[2]; }

    __host__ __device__ Vec3 operator+(const Vec3& o) const {
        return {data[0] + o.data[0], data[1] + o.data[1], data[2] + o.data[2]};
    }

    __host__ __device__ Vec3 operator-(const Vec3& o) const {
        return {data[0] - o.data[0], data[1] - o.data[1], data[2] - o.data[2]};
    }

    __host__ __device__ Vec3 operator*(double s) const {
        return {data[0] * s, data[1] * s, data[2] * s};
    }

    /// Component-wise multiply (color modulation)
    __host__ __device__ Vec3 operator*(const Vec3& o) const {
        return {data[0] * o.data[0], data[1] * o.data[1], data[2] * o.data[2]};
    }

    __host__ __device__ Vec3& operator+=(const Vec3& o) {
        data[0] += o.data[0];
        data[1] += o.data[1];
        data[2] += o.data[2];
        return *this;
    }

    /// Returns the largest of the three components.
    __host__ __device__ double max_component() const {
        return fmax(data[0], fmax(data[1], data[2]));
    }
};

__host__ __device__ inline Vec3 operator*(double s, const Vec3& v) {
    return v * s;
}

// ---------------------------------------------------------------------------
// Vec4: 4-component double vector for spacetime 4-vectors / covectors.
// Index convention: [0]=t, [1]=r, [2]=theta, [3]=phi (Boyer-Lindquist).
// ---------------------------------------------------------------------------
struct Vec4 {
    double data[4];

    __host__ __device__ double& operator[](int i) { return data[i]; }
    __host__ __device__ const double& operator[](int i) const { return data[i]; }

    __host__ __device__ Vec4 operator+(const Vec4& o) const {
        return {data[0] + o.data[0], data[1] + o.data[1],
                data[2] + o.data[2], data[3] + o.data[3]};
    }

    __host__ __device__ Vec4 operator-(const Vec4& o) const {
        return {data[0] - o.data[0], data[1] - o.data[1],
                data[2] - o.data[2], data[3] - o.data[3]};
    }

    __host__ __device__ Vec4 operator*(double s) const {
        return {data[0] * s, data[1] * s, data[2] * s, data[3] * s};
    }
};

__host__ __device__ inline Vec4 operator*(double s, const Vec4& v) {
    return v * s;
}

// ---------------------------------------------------------------------------
// Matrix4: 4x4 double matrix for metric tensors.
// Exploits the block-diagonal structure of Boyer-Lindquist metrics:
//   - (t, phi) entries form a 2x2 block: [0][0], [0][3], [3][0], [3][3]
//   - r and theta are diagonal:          [1][1], [2][2]
//   - All other entries are zero.
// ---------------------------------------------------------------------------
struct Matrix4 {
    double m[4][4];

    /// Contract: result_mu = sum_nu M_mu_nu * v^nu
    /// Used for lowering (g.contract(v)) and raising (g_upper.contract(v)).
    __host__ __device__ Vec4 contract(const Vec4& v) const {
        Vec4 result{};
        for (int mu = 0; mu < 4; ++mu) {
            double sum = 0.0;
            for (int nu = 0; nu < 4; ++nu) {
                sum += m[mu][nu] * v.data[nu];
            }
            result.data[mu] = sum;
        }
        return result;
    }

    /// Create a diagonal matrix with entries (a, b, c, d) on the diagonal.
    __host__ __device__ static Matrix4 diagonal(double a, double b, double c, double d) {
        Matrix4 mat{};
        mat.m[0][0] = a;
        mat.m[1][1] = b;
        mat.m[2][2] = c;
        mat.m[3][3] = d;
        return mat;
    }

    /// Inverse of a purely diagonal matrix (sufficient for Schwarzschild metric).
    __host__ __device__ Matrix4 inverse_diagonal() const {
        return diagonal(
            1.0 / m[0][0],
            1.0 / m[1][1],
            1.0 / m[2][2],
            1.0 / m[3][3]
        );
    }

    /// General inverse exploiting the block-diagonal structure of BL metrics.
    /// (t, phi) form a 2x2 block; r and theta are diagonal.
    /// Also works for fully diagonal matrices (off-diagonal block entries = 0).
    __host__ __device__ Matrix4 inverse() const {
        Matrix4 inv{};

        // Invert the (t, phi) 2x2 block using Cramer's rule:
        //   | m[0][0]  m[0][3] |^-1 = (1/det) * | m[3][3]  -m[0][3] |
        //   | m[3][0]  m[3][3] |                 | -m[3][0]  m[0][0] |
        double det_tf = m[0][0] * m[3][3] - m[0][3] * m[3][0];
        inv.m[0][0] =  m[3][3] / det_tf;
        inv.m[3][3] =  m[0][0] / det_tf;
        inv.m[0][3] = -m[0][3] / det_tf;
        inv.m[3][0] = -m[3][0] / det_tf;

        // Diagonal r and theta entries
        inv.m[1][1] = 1.0 / m[1][1];
        inv.m[2][2] = 1.0 / m[2][2];

        return inv;
    }
};

} // namespace cuda

#endif // GRRT_CUDA_MATH_H
