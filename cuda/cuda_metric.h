#ifndef GRRT_CUDA_METRIC_H
#define GRRT_CUDA_METRIC_H

/// @file cuda_metric.h
/// @brief CUDA device-compatible metric functions for Schwarzschild and Kerr spacetimes.
///
/// Mirrors the CPU grrt::Schwarzschild and grrt::Kerr implementations but uses
/// enum+switch dispatch instead of virtual functions (which are not supported on
/// CUDA device code in general). All functions are __host__ __device__ so they
/// compile for both CPU and GPU.
///
/// Units: geometrized (G = c = 1), M sets the length scale.
/// Coordinates: Boyer-Lindquist (t, r, theta, phi), Vec4 index [0..3].

#include "cuda_math.h"
#include <cmath>

namespace cuda {

// ---------------------------------------------------------------------------
// Metric type tag — used for enum+switch dispatch on device
// ---------------------------------------------------------------------------
enum class MetricType { Schwarzschild, Kerr };

// ===========================================================================
// Schwarzschild helpers
// ===========================================================================

/// @brief Covariant Schwarzschild metric g_μν (diagonal).
///
/// g_tt = -(1 - 2M/r)
/// g_rr = 1 / (1 - 2M/r)
/// g_θθ = r²
/// g_φφ = r² sin²θ
///
/// @param M  Mass (geometrized units)
/// @param x  Boyer-Lindquist position (t, r, θ, φ)
__host__ __device__ inline Matrix4 schwarzschild_g_lower(double M, const Vec4& x) {
    const double r     = x[1];
    const double theta = x[2];
    // Clamp sin(theta) away from zero to avoid g_phiphi = 0 at poles
    double sin_t = sin(theta);
    if (fabs(sin_t) < 1e-10) sin_t = (sin_t >= 0.0) ? 1e-10 : -1e-10;
    const double sin2 = sin_t * sin_t;

    const double f = 1.0 - 2.0 * M / r;  // lapse²

    return Matrix4::diagonal(
        -f,               // g_tt
        1.0 / f,          // g_rr
        r * r,            // g_θθ
        r * r * sin2      // g_φφ
    );
}

/// @brief Contravariant Schwarzschild metric g^μν (diagonal inverse).
///
/// For a diagonal metric g^μν = 1 / g_μν component-wise.
///
/// @param M  Mass
/// @param x  Boyer-Lindquist position
__host__ __device__ inline Matrix4 schwarzschild_g_upper(double M, const Vec4& x) {
    return schwarzschild_g_lower(M, x).inverse_diagonal();
}

// ===========================================================================
// Kerr helpers
// ===========================================================================

/// @brief Kerr-Schild Σ = r² + a²cos²θ
__host__ __device__ inline double kerr_sigma(double r, double theta, double a) {
    double cos_t = cos(theta);
    return r * r + a * a * cos_t * cos_t;
}

/// @brief Kerr-Schild Δ = r² - 2Mr + a²
__host__ __device__ inline double kerr_delta(double r, double M, double a) {
    return r * r - 2.0 * M * r + a * a;
}

/// @brief Covariant Kerr metric g_μν in Boyer-Lindquist coordinates.
///
/// Non-zero components:
///   g_tt   = -(1 - 2Mr/Σ)
///   g_tφ   = -2Mar sin²θ / Σ   (frame-dragging off-diagonal terms)
///   g_φt   = g_tφ
///   g_rr   = Σ/Δ
///   g_θθ   = Σ
///   g_φφ   = (r² + a² + 2Ma²r sin²θ/Σ) sin²θ
///
/// Reference: Kerr (1963); Boyer & Lindquist (1967).
///
/// @param M  Mass
/// @param a  Spin parameter (|a| < M for sub-extremal black hole)
/// @param x  Boyer-Lindquist position (t, r, θ, φ)
__host__ __device__ inline Matrix4 kerr_g_lower(double M, double a, const Vec4& x) {
    const double r     = x[1];
    const double theta = x[2];

    double sin_t = sin(theta);
    double sin2  = sin_t * sin_t;
    // Clamp to avoid degenerate metric at poles
    if (sin2 < 1e-20) sin2 = 1e-20;

    const double S = fmax(kerr_sigma(r, theta, a), 1e-20);
    const double D = kerr_delta(r, M, a);

    Matrix4 g{};
    g.m[0][0] = -(1.0 - 2.0 * M * r / S);                                // g_tt
    g.m[0][3] = -2.0 * M * a * r * sin2 / S;                              // g_tφ
    g.m[3][0] = g.m[0][3];                                                 // g_φt (symmetric)
    g.m[1][1] = S / D;                                                     // g_rr
    g.m[2][2] = S;                                                         // g_θθ
    g.m[3][3] = (r * r + a * a + 2.0 * M * a * a * r * sin2 / S) * sin2; // g_φφ

    return g;
}

/// @brief Contravariant Kerr metric g^μν.
///
/// Computed via block-diagonal inverse of g_lower (see Matrix4::inverse).
///
/// @param M  Mass
/// @param a  Spin parameter
/// @param x  Boyer-Lindquist position
__host__ __device__ inline Matrix4 kerr_g_upper(double M, double a, const Vec4& x) {
    return kerr_g_lower(M, a, x).inverse();
}

// ===========================================================================
// Dispatch functions
// ===========================================================================

/// @brief Covariant metric g_μν dispatched by MetricType.
///
/// @param type  MetricType::Schwarzschild or MetricType::Kerr
/// @param M     Mass
/// @param a     Spin parameter (ignored for Schwarzschild)
/// @param x     Boyer-Lindquist position
__host__ __device__ inline Matrix4 metric_lower(MetricType type, double M, double a, const Vec4& x) {
    switch (type) {
        case MetricType::Schwarzschild:
            return schwarzschild_g_lower(M, x);
        case MetricType::Kerr:
            return kerr_g_lower(M, a, x);
        default:
            return schwarzschild_g_lower(M, x);
    }
}

/// @brief Contravariant metric g^μν dispatched by MetricType.
///
/// @param type  MetricType::Schwarzschild or MetricType::Kerr
/// @param M     Mass
/// @param a     Spin parameter (ignored for Schwarzschild)
/// @param x     Boyer-Lindquist position
__host__ __device__ inline Matrix4 metric_upper(MetricType type, double M, double a, const Vec4& x) {
    switch (type) {
        case MetricType::Schwarzschild:
            return schwarzschild_g_upper(M, x);
        case MetricType::Kerr:
            return kerr_g_upper(M, a, x);
        default:
            return schwarzschild_g_upper(M, x);
    }
}

/// @brief Event horizon radius dispatched by MetricType.
///
/// Schwarzschild: r_h = 2M
/// Kerr:          r_h = M + sqrt(M² - a²)
///
/// @param type  MetricType
/// @param M     Mass
/// @param a     Spin parameter (ignored for Schwarzschild)
__host__ __device__ inline double horizon_radius(MetricType type, double M, double a) {
    switch (type) {
        case MetricType::Schwarzschild:
            return 2.0 * M;
        case MetricType::Kerr:
            return M + sqrt(M * M - a * a);
        default:
            return 2.0 * M;
    }
}

/// @brief Innermost stable circular orbit (ISCO) radius dispatched by MetricType.
///
/// Schwarzschild: r_ISCO = 6M
/// Kerr (prograde): Bardeen, Press & Teukolsky (1972) formula.
///
/// @param type  MetricType
/// @param M     Mass
/// @param a     Spin parameter (ignored for Schwarzschild)
__host__ __device__ inline double isco_radius(MetricType type, double M, double a) {
    switch (type) {
        case MetricType::Schwarzschild:
            return 6.0 * M;
        case MetricType::Kerr: {
            // Dimensionless spin a* = a/M
            const double a_star = a / M;
            // Z1 = 1 + (1 - a*²)^(1/3) * [(1 + a*)^(1/3) + (1 - a*)^(1/3)]
            const double Z1 = 1.0 + cbrt(1.0 - a_star * a_star)
                                  * (cbrt(1.0 + a_star) + cbrt(1.0 - a_star));
            // Z2 = sqrt(3a*² + Z1²)
            const double Z2 = sqrt(3.0 * a_star * a_star + Z1 * Z1);
            return M * (3.0 + Z2 - sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)));
        }
        default:
            return 6.0 * M;
    }
}

} // namespace cuda

#endif // GRRT_CUDA_METRIC_H
