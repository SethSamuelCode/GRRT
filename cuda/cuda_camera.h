#ifndef GRRT_CUDA_CAMERA_H
#define GRRT_CUDA_CAMERA_H

/// @file cuda_camera.h
/// @brief CUDA device-compatible camera: tetrad construction and pixel-to-ray mapping.
///
/// Constructs an orthonormal tetrad at the observer's position in Boyer-Lindquist
/// coordinates, then maps pixel (i, j) to an initial null geodesic state.
///
/// For Schwarzschild: uses the static observer 4-velocity.
/// For Kerr: uses the ZAMO (Zero Angular Momentum Observer) 4-velocity.
///
/// Tetrad convention:
///   e0: timelike (observer 4-velocity)
///   e1: spacelike, right (φ direction)
///   e2: spacelike, up (−θ direction)
///   e3: spacelike, forward (radially inward, −r direction)
///
/// Units: geometrized (G = c = 1). Coordinates: Boyer-Lindquist (t, r, θ, φ).

#include "cuda_math.h"
#include "cuda_metric.h"
#include "cuda_geodesic.h"
#include "cuda_types.h"
#include <cmath>

namespace cuda {

// ---------------------------------------------------------------------------
// metric_dot: g_{ab} u^a v^b using the covariant metric
// ---------------------------------------------------------------------------

/// @brief Compute the metric inner product g_{ab} u^a v^b.
///
/// @param type  MetricType dispatch tag
/// @param M     Mass
/// @param a     Spin parameter
/// @param x     Boyer-Lindquist position
/// @param u     First contravariant vector
/// @param v     Second contravariant vector
/// @return      g_{ab} u^a v^b
__host__ __device__ inline double metric_dot(MetricType type, double M, double a,
                                              const Vec4& x,
                                              const Vec4& u, const Vec4& v) {
    Matrix4 g = metric_lower(type, M, a, x);
    double sum = 0.0;
    for (int mu = 0; mu < 4; ++mu)
        for (int nu = 0; nu < 4; ++nu)
            sum += g.m[mu][nu] * u.data[mu] * v.data[nu];
    return sum;
}

// ---------------------------------------------------------------------------
// metric_normalize: v / sqrt(|g(v,v)|)
// ---------------------------------------------------------------------------

/// @brief Normalize a contravariant vector with respect to the metric.
///
/// Returns v / sqrt(|g(v,v)|). The absolute value handles both spacelike and
/// timelike vectors (whose norm-squared may be negative).
///
/// @param type  MetricType dispatch tag
/// @param M     Mass
/// @param a     Spin parameter
/// @param x     Boyer-Lindquist position
/// @param v     Contravariant vector to normalize
/// @return      Normalized vector
__host__ __device__ inline Vec4 metric_normalize(MetricType type, double M, double a,
                                                  const Vec4& x, const Vec4& v) {
    double norm2 = metric_dot(type, M, a, x, v, v);
    double norm  = sqrt(fabs(norm2));
    return v * (1.0 / norm);
}

// ---------------------------------------------------------------------------
// project_out: Gram-Schmidt step, remove component of v along e
// ---------------------------------------------------------------------------

/// @brief Gram-Schmidt projection: remove the component of v along e.
///
/// result = v - (g(v,e) / e_norm2) * e
///
/// @param type     MetricType dispatch tag
/// @param M        Mass
/// @param a        Spin parameter
/// @param x        Boyer-Lindquist position
/// @param v        Vector to project
/// @param e        Direction to project out
/// @param e_norm2  Precomputed g(e,e) (avoids recomputation)
/// @return         Projected vector
__host__ __device__ inline Vec4 project_out(MetricType type, double M, double a,
                                             const Vec4& x,
                                             const Vec4& v, const Vec4& e,
                                             double e_norm2) {
    double ve = metric_dot(type, M, a, x, v, e);
    return v - e * (ve / e_norm2);
}

// ---------------------------------------------------------------------------
// build_tetrad: construct orthonormal tetrad at observer's position
// ---------------------------------------------------------------------------

/// @brief Build an orthonormal tetrad at the observer's location in params.
///
/// Populates params.cam_e0 through params.cam_e3 using Gram-Schmidt
/// orthogonalization. Modifies params in place.
///
/// Observer models:
///   - Schwarzschild: static observer, u^μ = (1/√(−g_tt), 0, 0, 0)
///   - Kerr: ZAMO (Zero Angular Momentum Observer)
///     ω = −g^{tφ}/g^{φφ}, u^t = 1/√(−g_tt − ω g_tφ) ... (see code)
///
/// Tetrad:
///   e3 (forward) = radially inward (−r hat), projected out e0, normalized
///   e2 (up)      = −θ hat, projected out e0 and e3, normalized
///   e1 (right)   = +φ hat, projected out e0, e3, e2, normalized
///
/// @param params  RenderParams with cam_position, metric_type, mass, spin set
__host__ __device__ inline void build_tetrad(RenderParams& params) {
    MetricType type = params.metric_type;
    double M        = params.mass;
    double a        = params.spin;
    const Vec4& x   = params.cam_position;

    Matrix4 g_low = metric_lower(type, M, a, x);

    // ------------------------------------------------------------------
    // e0: observer 4-velocity (timelike, normalized to g(e0,e0) = -1)
    // ------------------------------------------------------------------
    Vec4 e0{};
    if (type == MetricType::Schwarzschild) {
        // Static observer: u^μ = (1/√(−g_tt), 0, 0, 0)
        double g_tt = g_low.m[0][0];  // negative
        e0.data[0] = 1.0 / sqrt(-g_tt);
        e0.data[1] = 0.0;
        e0.data[2] = 0.0;
        e0.data[3] = 0.0;
    } else {
        // ZAMO: zero angular momentum observer for Kerr spacetime.
        // Frame-dragging angular velocity: ω = −g^{tφ} / g^{φφ}
        // (using upper metric dispatched through our helper).
        Matrix4 g_up = metric_upper(type, M, a, x);
        double g_up_tphi = g_up.m[0][3];  // g^{tφ} (= g^{φt} by symmetry)
        double g_up_pp   = g_up.m[3][3];  // g^{φφ}
        double omega = -g_up_tphi / g_up_pp;

        // u^μ = N (1, 0, 0, ω) where N = 1/√(−g_tt − 2ω g_tφ − ω² g_φφ)
        double g_tt   = g_low.m[0][0];
        double g_tphi = g_low.m[0][3];
        double g_pp   = g_low.m[3][3];
        double norm2_inv = -(g_tt + 2.0 * omega * g_tphi + omega * omega * g_pp);
        double N = 1.0 / sqrt(fabs(norm2_inv));
        e0.data[0] = N;
        e0.data[1] = 0.0;
        e0.data[2] = 0.0;
        e0.data[3] = N * omega;
    }

    double e0_norm2 = metric_dot(type, M, a, x, e0, e0);

    // ------------------------------------------------------------------
    // e3: forward = radially inward (−r direction)
    // ------------------------------------------------------------------
    Vec4 r_dir{};
    r_dir.data[0] =  0.0;
    r_dir.data[1] = -1.0;  // −r hat
    r_dir.data[2] =  0.0;
    r_dir.data[3] =  0.0;

    Vec4 v3 = project_out(type, M, a, x, r_dir, e0, e0_norm2);
    Vec4 e3 = metric_normalize(type, M, a, x, v3);
    double e3_norm2 = metric_dot(type, M, a, x, e3, e3);

    // ------------------------------------------------------------------
    // e2: up = −θ direction
    // ------------------------------------------------------------------
    Vec4 theta_dir{};
    theta_dir.data[0] =  0.0;
    theta_dir.data[1] =  0.0;
    theta_dir.data[2] = -1.0;  // −θ hat
    theta_dir.data[3] =  0.0;

    Vec4 v2 = project_out(type, M, a, x, theta_dir, e0, e0_norm2);
    double v2_e3_norm2 = metric_dot(type, M, a, x, e3, e3);  // = e3_norm2
    v2 = project_out(type, M, a, x, v2, e3, v2_e3_norm2);
    Vec4 e2 = metric_normalize(type, M, a, x, v2);
    double e2_norm2 = metric_dot(type, M, a, x, e2, e2);

    // ------------------------------------------------------------------
    // e1: right = +φ direction
    // ------------------------------------------------------------------
    Vec4 phi_dir{};
    phi_dir.data[0] = 0.0;
    phi_dir.data[1] = 0.0;
    phi_dir.data[2] = 0.0;
    phi_dir.data[3] = 1.0;  // +φ hat

    Vec4 v1 = project_out(type, M, a, x, phi_dir, e0, e0_norm2);
    v1 = project_out(type, M, a, x, v1, e3, e3_norm2);
    v1 = project_out(type, M, a, x, v1, e2, e2_norm2);
    Vec4 e1 = metric_normalize(type, M, a, x, v1);

    params.cam_e0 = e0;
    params.cam_e1 = e1;
    params.cam_e2 = e2;
    params.cam_e3 = e3;
}

// ---------------------------------------------------------------------------
// ray_for_pixel: map pixel (i, j) to an initial GeodesicState
// ---------------------------------------------------------------------------

/// @brief Generate the initial null geodesic state for pixel (i, j).
///
/// Screen-space angles:
///   alpha = (i + 0.5 − width/2) * fov / width   (horizontal)
///   beta  = (j + 0.5 − height/2) * fov / width  (vertical)
///
/// Local direction in tetrad frame:
///   ca = cos(beta) * sin(alpha)
///   cb = sin(beta)
///   cc = cos(beta) * cos(alpha)
///   d  = −ca * e1 − cb * e2 + cc * e3
///
/// Null 4-momentum (contravariant): p^μ = −e0 + d
/// Lowered: p_μ = g_{μν} p^ν
///
/// @param params  RenderParams with cam_position, tetrad, fov, width, height
/// @param i       Pixel column index (0-based)
/// @param j       Pixel row index (0-based)
/// @return        GeodesicState with position = cam_position, momentum = p_cov
__host__ __device__ inline GeodesicState ray_for_pixel(const RenderParams& params,
                                                        int i, int j) {
    double alpha = (i + 0.5 - params.width  * 0.5) * params.fov / params.width;
    double beta  = (j + 0.5 - params.height * 0.5) * params.fov / params.width;

    double ca = cos(beta) * sin(alpha);
    double cb = sin(beta);
    double cc = cos(beta) * cos(alpha);

    // d^μ = −ca * e1 − cb * e2 + cc * e3
    Vec4 d{};
    for (int mu = 0; mu < 4; ++mu) {
        d.data[mu] = -ca * params.cam_e1.data[mu]
                     - cb * params.cam_e2.data[mu]
                     + cc * params.cam_e3.data[mu];
    }

    // Null 4-momentum (contravariant): p^μ = −e0 + d
    Vec4 p_contra{};
    for (int mu = 0; mu < 4; ++mu) {
        p_contra.data[mu] = -params.cam_e0.data[mu] + d.data[mu];
    }

    // Lower: p_μ = g_{μν} p^ν
    Matrix4 g_low = metric_lower(params.metric_type, params.mass, params.spin,
                                  params.cam_position);
    Vec4 p_cov = g_low.contract(p_contra);

    GeodesicState state;
    state.position = params.cam_position;
    state.momentum = p_cov;
    return state;
}

} // namespace cuda

#endif // GRRT_CUDA_CAMERA_H
