#ifndef GRRT_CUDA_NOISE_H
#define GRRT_CUDA_NOISE_H

/// @file cuda_noise.h
/// @brief CUDA device-compatible 3D simplex noise.
///
/// Mirrors the CPU grrt::SimplexNoise3D::evaluate() algorithm for GPU execution.
/// The permutation table is stored in __constant__ memory (512 ints = 2 KB),
/// uploaded from the host before rendering via cudaMemcpyToSymbol().
///
/// Output range: approximately [-1, 1] for single evaluation,
/// [-1.5, 1.5] for turbulent (two-octave) evaluation.

#include <cuda_runtime.h>

namespace cuda {

// ---------------------------------------------------------------------------
// Constant memory: permutation table (defined in cuda_render.cu)
// ---------------------------------------------------------------------------

/// @brief Permutation table for simplex noise hash function.
///
/// 512 entries: perm[0..255] is a shuffled identity permutation,
/// perm[256..511] = perm[0..255] repeated for overflow-safe indexing.
/// Uploaded once from SimplexNoise3D::permutation_table() before rendering.
extern __constant__ int d_noise_perm[512];

// ---------------------------------------------------------------------------
// Compile-time gradient vectors for 3D simplex noise
// ---------------------------------------------------------------------------

/// @brief 12 gradient vectors for 3D simplex noise.
///
/// Each gradient is an edge midpoint of a unit cube, giving 12 directions.
/// These are the same vectors used in the CPU implementation.
__device__ constexpr double d_grad3[12][3] = {
    {1,1,0}, {-1,1,0}, {1,-1,0}, {-1,-1,0},
    {1,0,1}, {-1,0,1}, {1,0,-1}, {-1,0,-1},
    {0,1,1}, {0,-1,1}, {0,1,-1}, {0,-1,-1}
};

// ---------------------------------------------------------------------------
// Helper: dot product of gradient with displacement
// ---------------------------------------------------------------------------

/// @brief Dot product of a gradient vector with a 3D displacement.
__device__ inline double cuda_dot3(const double g[3], double x, double y, double z) {
    return g[0]*x + g[1]*y + g[2]*z;
}

// ---------------------------------------------------------------------------
// 3D simplex noise evaluation
// ---------------------------------------------------------------------------

/// @brief Evaluate 3D simplex noise at position (x, y, z).
///
/// Implements Ken Perlin's simplex noise algorithm. The 3D input space
/// is skewed into a tetrahedral lattice, and the contribution of each
/// of the 4 surrounding simplex vertices is summed. The radial falloff
/// kernel t^4 (where t = 0.6 - |d|^2) ensures C1 continuity.
///
/// @return Value in approximately [-1, 1].
__device__ inline double cuda_simplex_noise_3d(double x, double y, double z) {
    // Skewing factors for 3D simplex grid
    constexpr double F3 = 1.0 / 3.0;
    constexpr double G3 = 1.0 / 6.0;

    // Skew input space to find simplex cell
    double s = (x + y + z) * F3;
    int i = (int)floor(x + s);
    int j = (int)floor(y + s);
    int k = (int)floor(z + s);

    // Unskew back to find cell origin in (x, y, z) space
    double t = (i + j + k) * G3;
    double X0 = i - t, Y0 = j - t, Z0 = k - t;
    double x0 = x - X0, y0 = y - Y0, z0 = z - Z0;

    // Determine which simplex we are in (6 possible orderings of x0, y0, z0)
    int i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0)      { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
        else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
        else               { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    } else {
        if (y0 < z0)       { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
        else if (x0 < z0)  { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
        else               { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }

    // Offsets for remaining corners
    double x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
    double x2 = x0 - i2 + 2.0*G3, y2 = y0 - j2 + 2.0*G3, z2 = z0 - k2 + 2.0*G3;
    double x3 = x0 - 1.0 + 3.0*G3, y3 = y0 - 1.0 + 3.0*G3, z3 = z0 - 1.0 + 3.0*G3;

    // Hash coordinates to gradient indices
    int ii = i & 255, jj = j & 255, kk = k & 255;

    // Contribution from each of the 4 simplex corners
    // Each uses the radial falloff kernel: (0.6 - |d|^2)^4 * dot(grad, d)
    double n0, n1, n2, n3;

    // Corner 0
    {
        double t0 = 0.6 - x0*x0 - y0*y0 - z0*z0;
        if (t0 < 0.0) { n0 = 0.0; }
        else {
            int gi = d_noise_perm[ii + d_noise_perm[jj + d_noise_perm[kk]]] % 12;
            t0 *= t0;
            n0 = t0 * t0 * cuda_dot3(d_grad3[gi], x0, y0, z0);
        }
    }

    // Corner 1
    {
        double t1 = 0.6 - x1*x1 - y1*y1 - z1*z1;
        if (t1 < 0.0) { n1 = 0.0; }
        else {
            int gi = d_noise_perm[ii+i1 + d_noise_perm[jj+j1 + d_noise_perm[kk+k1]]] % 12;
            t1 *= t1;
            n1 = t1 * t1 * cuda_dot3(d_grad3[gi], x1, y1, z1);
        }
    }

    // Corner 2
    {
        double t2 = 0.6 - x2*x2 - y2*y2 - z2*z2;
        if (t2 < 0.0) { n2 = 0.0; }
        else {
            int gi = d_noise_perm[ii+i2 + d_noise_perm[jj+j2 + d_noise_perm[kk+k2]]] % 12;
            t2 *= t2;
            n2 = t2 * t2 * cuda_dot3(d_grad3[gi], x2, y2, z2);
        }
    }

    // Corner 3
    {
        double t3 = 0.6 - x3*x3 - y3*y3 - z3*z3;
        if (t3 < 0.0) { n3 = 0.0; }
        else {
            int gi = d_noise_perm[ii+1 + d_noise_perm[jj+1 + d_noise_perm[kk+1]]] % 12;
            t3 *= t3;
            n3 = t3 * t3 * cuda_dot3(d_grad3[gi], x3, y3, z3);
        }
    }

    // Scale to [-1, 1]
    return 32.0 * (n0 + n1 + n2 + n3);
}

// ---------------------------------------------------------------------------
// Turbulent (two-octave) noise
// ---------------------------------------------------------------------------

/// @brief Evaluate turbulent simplex noise at (x, y, z).
///
/// Combines the base frequency with a second octave at 3x frequency
/// and 0.5x amplitude, producing a rougher, cloud-like appearance.
/// Used for density perturbations in the volumetric accretion disk.
///
/// @return Value in approximately [-1.5, 1.5].
__device__ inline double cuda_simplex_noise_turbulent(double x, double y, double z) {
    return cuda_simplex_noise_3d(x, y, z)
         + 0.5 * cuda_simplex_noise_3d(x * 3.0, y * 3.0, z * 3.0);
}

} // namespace cuda

#endif // GRRT_CUDA_NOISE_H
