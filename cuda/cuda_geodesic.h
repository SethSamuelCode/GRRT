#ifndef GRRT_CUDA_GEODESIC_H
#define GRRT_CUDA_GEODESIC_H

/// @file cuda_geodesic.h
/// @brief CUDA device-compatible RK4 geodesic integrator for null geodesics.
///
/// Implements Hamiltonian formulation for null geodesics in curved spacetime.
/// State vector: (x^μ, p_μ) — 8 components total.
///
/// Equations of motion:
///   dx^μ/dλ = g^{μν} p_ν         (raise momentum with inverse metric)
///   dp_μ/dλ = -½ (∂g^{αβ}/∂x^μ) p_α p_β  (geodesic force via FD)
///
/// Mirrors the CPU RK4 implementation in src/rk4.cpp but uses device-compatible
/// functions (no std::, no virtual dispatch). All functions are __device__ inline.
///
/// Units: geometrized (G = c = 1). Coordinates: Boyer-Lindquist (t, r, θ, φ).

#include "cuda_math.h"
#include "cuda_metric.h"

namespace cuda {

// ---------------------------------------------------------------------------
// GeodesicState: position x^μ (contravariant) + momentum p_μ (covariant)
// ---------------------------------------------------------------------------

/// @brief State vector for a photon geodesic.
///
/// position: x^μ = (t, r, θ, φ) — Boyer-Lindquist contravariant coordinates.
/// momentum: p_μ — covariant 4-momentum (lowered index).
struct GeodesicState {
    Vec4 position;
    Vec4 momentum;
};

/// @brief Result of an adaptive RK4 step.
struct AdaptiveResult {
    GeodesicState state;       ///< Accepted new state
    double        next_dlambda; ///< Recommended step size for next iteration
};

// ---------------------------------------------------------------------------
// Helper: add scaled derivative to state
// ---------------------------------------------------------------------------
__device__ inline GeodesicState geodesic_add(const GeodesicState& s,
                                              const GeodesicState& ds,
                                              double h) {
    GeodesicState result;
    result.position = s.position + ds.position * h;
    result.momentum = s.momentum + ds.momentum * h;
    return result;
}

// ---------------------------------------------------------------------------
// derivatives: compute (dx^μ/dλ, dp_μ/dλ) at given state
// ---------------------------------------------------------------------------

/// @brief Compute geodesic derivatives at the given state.
///
/// dx^μ/dλ = g^{μν} p_ν
/// dp_μ/dλ = -½ Σ_{α,β} (∂g^{αβ}/∂x^μ) p_α p_β
///
/// The metric derivatives are approximated by central finite differences:
///   ∂g^{αβ}/∂x^μ ≈ (g^{αβ}(x + ε e_μ) - g^{αβ}(x - ε e_μ)) / (2ε)
///
/// @param type    MetricType::Schwarzschild or MetricType::Kerr
/// @param M       Mass in geometrized units
/// @param a       Spin parameter (ignored for Schwarzschild)
/// @param state   Current geodesic state (x^μ, p_μ)
/// @return        Derivatives (dx^μ/dλ, dp_μ/dλ) packaged as a GeodesicState
__device__ inline GeodesicState derivatives(MetricType type, double M, double a,
                                             const GeodesicState& state) {
    constexpr double fd_eps = 1e-6;
    constexpr double inv_2eps = 1.0 / (2.0 * fd_eps);

    const Vec4& x = state.position;
    const Vec4& p = state.momentum;

    // dx^μ/dλ = g^{μν} p_ν — exploit Boyer-Lindquist sparsity
    Matrix4 g_inv = metric_upper(type, M, a, x);
    Vec4 dx{};
    if (type == MetricType::Schwarzschild) {
        // Diagonal: g^{μν} = 0 for μ≠ν
        for (int mu = 0; mu < 4; ++mu)
            dx.data[mu] = g_inv.m[mu][mu] * p[mu];
    } else {
        // Kerr: diagonal + (t,φ) off-diagonal block
        dx.data[0] = g_inv.m[0][0] * p[0] + g_inv.m[0][3] * p[3];
        dx.data[1] = g_inv.m[1][1] * p[1];
        dx.data[2] = g_inv.m[2][2] * p[2];
        dx.data[3] = g_inv.m[3][0] * p[0] + g_inv.m[3][3] * p[3];
    }

    // dp_μ/dλ = -½ Σ_{α,β} (∂g^{αβ}/∂x^μ) p_α p_β
    // Process one coordinate at a time to minimize register pressure:
    // only two Matrix4 live simultaneously (plus/minus), immediately reduced to scalar.
    Vec4 dp{};
    for (int mu = 0; mu < 4; ++mu) {
        Vec4 x_plus  = x;
        Vec4 x_minus = x;
        x_plus.data[mu]  += fd_eps;
        x_minus.data[mu] -= fd_eps;

        Matrix4 g_plus  = metric_upper(type, M, a, x_plus);
        Matrix4 g_minus = metric_upper(type, M, a, x_minus);

        // Exploit sparsity: only sum over non-zero metric components
        double force;
        if (type == MetricType::Schwarzschild) {
            // Diagonal only: α==β terms
            force = 0.0;
            for (int i = 0; i < 4; ++i) {
                double dg = (g_plus.m[i][i] - g_minus.m[i][i]) * inv_2eps;
                force += dg * p[i] * p[i];
            }
        } else {
            // Kerr: diagonal + (0,3) and (3,0) off-diagonal
            force = 0.0;
            for (int i = 0; i < 4; ++i) {
                double dg = (g_plus.m[i][i] - g_minus.m[i][i]) * inv_2eps;
                force += dg * p[i] * p[i];
            }
            // Off-diagonal (t,φ) block: contributes 2 * dg^{tφ} * p_t * p_φ
            double dg_tphi = (g_plus.m[0][3] - g_minus.m[0][3]) * inv_2eps;
            force += 2.0 * dg_tphi * p[0] * p[3];
        }
        dp.data[mu] = -0.5 * force;
    }

    GeodesicState deriv;
    deriv.position = dx;
    deriv.momentum = dp;
    return deriv;
}

// ---------------------------------------------------------------------------
// rk4_step: classic 4-stage Runge-Kutta step
// ---------------------------------------------------------------------------

/// @brief Advance geodesic state by one RK4 step of size dlambda.
///
/// Uses the standard 4-stage formula:
///   k1 = f(state)
///   k2 = f(state + h/2 * k1)
///   k3 = f(state + h/2 * k2)
///   k4 = f(state + h * k3)
///   result = state + (h/6)(k1 + 2k2 + 2k3 + k4)
///
/// @param type      MetricType dispatch tag
/// @param M         Mass
/// @param a         Spin parameter
/// @param state     Current state
/// @param dlambda   Step size in affine parameter λ
/// @return          New state after one RK4 step
__device__ inline GeodesicState rk4_step(MetricType type, double M, double a,
                                          const GeodesicState& state,
                                          double dlambda) {
    GeodesicState k1 = derivatives(type, M, a, state);
    GeodesicState k2 = derivatives(type, M, a, geodesic_add(state, k1, dlambda * 0.5));
    GeodesicState k3 = derivatives(type, M, a, geodesic_add(state, k2, dlambda * 0.5));
    GeodesicState k4 = derivatives(type, M, a, geodesic_add(state, k3, dlambda));

    GeodesicState result;
    result.position = state.position
        + (k1.position + k2.position * 2.0 + k3.position * 2.0 + k4.position)
          * (dlambda / 6.0);
    result.momentum = state.momentum
        + (k1.momentum + k2.momentum * 2.0 + k3.momentum * 2.0 + k4.momentum)
          * (dlambda / 6.0);
    return result;
}

// ---------------------------------------------------------------------------
// rk4_adaptive_step: step doubling with error control
// ---------------------------------------------------------------------------

/// @brief Adaptive RK4 step using step-doubling error estimation.
///
/// Compares one full step of size h against two half-steps of size h/2.
/// The error estimate is the maximum absolute difference in position components.
///
/// Acceptance logic (mirrors CPU rk4.cpp adaptive_step):
///   - err <= tolerance: accept two-half-step result
///     - err < 0.01 * tolerance: grow step by 2x
///     - Clamp max step to 5 * max(r, 1)
///   - err > tolerance: halve step, retry (up to 20 times)
///   - If step shrinks below 1e-6: force-accept at minimum step
///
/// @param type       MetricType dispatch tag
/// @param M          Mass
/// @param a          Spin parameter
/// @param state      Current state
/// @param dlambda    Initial step size
/// @param tolerance  Error tolerance (e.g. 1e-8)
/// @return           AdaptiveResult with accepted state and next step suggestion
__device__ inline AdaptiveResult rk4_adaptive_step(MetricType type, double M, double a,
                                                    const GeodesicState& state,
                                                    double dlambda,
                                                    double tolerance) {
    constexpr double dl_min     = 1e-6;
    constexpr int    max_retries = 20;
    constexpr double eps_scale  = 1e-10;  // scale guard to avoid div-by-zero in relative error

    double dl = dlambda;

    for (int retry = 0; retry < max_retries; ++retry) {
        // One full step
        GeodesicState s_full = rk4_step(type, M, a, state, dl);

        // Two half steps
        GeodesicState s_mid  = rk4_step(type, M, a, state, dl * 0.5);
        GeodesicState s_half = rk4_step(type, M, a, s_mid, dl * 0.5);

        // Compute max error across spatial position (r, θ, φ) and all momentum
        // (mirrors CPU: skip t for position, use all 4 for momentum)
        double err = 0.0;
        for (int i = 1; i < 4; ++i) {  // position: skip t (index 0)
            double diff  = fabs(s_full.position[i] - s_half.position[i]);
            double scale = fabs(s_half.position[i]) + eps_scale;
            err = fmax(err, diff / scale);
        }
        for (int i = 0; i < 4; ++i) {  // all momentum components
            double diff  = fabs(s_full.momentum[i] - s_half.momentum[i]);
            double scale = fabs(s_half.momentum[i]) + eps_scale;
            err = fmax(err, diff / scale);
        }

        if (err <= tolerance) {
            // Accept — decide whether to grow step
            double next_dl = dl;
            if (err < tolerance * 0.01) {
                next_dl = dl * 2.0;
            }
            // Clamp to maximum step size based on current radius
            double r_cur = fabs(s_half.position[1]);
            double max_step = 5.0 * fmax(r_cur, 1.0);
            if (next_dl > max_step) next_dl = max_step;

            AdaptiveResult res;
            res.state       = s_half;
            res.next_dlambda = next_dl;
            return res;
        }

        // Reject — shrink step and retry
        dl *= 0.5;
        if (dl < dl_min) {
            // Force-accept at minimum step to avoid infinite loop
            GeodesicState s_mid2  = rk4_step(type, M, a, state, dl * 0.5);
            GeodesicState s_half2 = rk4_step(type, M, a, s_mid2, dl * 0.5);
            double r_cur = fabs(s_half2.position[1]);
            double next_dl = fmin(dl_min, 5.0 * fmax(r_cur, 1.0));

            AdaptiveResult res;
            res.state        = s_half2;
            res.next_dlambda = next_dl;
            return res;
        }
    }

    // Exhausted retries — force-accept whatever we have at current dl
    GeodesicState s_mid  = rk4_step(type, M, a, state, dl * 0.5);
    GeodesicState s_half = rk4_step(type, M, a, s_mid, dl * 0.5);

    AdaptiveResult res;
    res.state        = s_half;
    res.next_dlambda = dl;
    return res;
}

// ---------------------------------------------------------------------------
// hamiltonian: compute H = 0.5 * g^{ab} p_a p_b (should be 0 for null rays)
// ---------------------------------------------------------------------------

/// @brief Compute the null geodesic Hamiltonian H = ½ g^{αβ} p_α p_β.
///
/// For a null geodesic this should remain zero throughout integration.
/// Drift from zero measures integration error.
///
/// @param type   MetricType dispatch tag
/// @param M      Mass
/// @param a      Spin parameter
/// @param state  Current geodesic state
/// @return       H (should be ≈ 0 for a well-integrated null ray)
__device__ inline double hamiltonian(MetricType type, double M, double a,
                                      const GeodesicState& state) {
    Matrix4 g_inv = metric_upper(type, M, a, state.position);
    const Vec4& p = state.momentum;
    double H;
    if (type == MetricType::Schwarzschild) {
        H = g_inv.m[0][0] * p[0] * p[0] + g_inv.m[1][1] * p[1] * p[1]
          + g_inv.m[2][2] * p[2] * p[2] + g_inv.m[3][3] * p[3] * p[3];
    } else {
        H = g_inv.m[0][0] * p[0] * p[0] + g_inv.m[1][1] * p[1] * p[1]
          + g_inv.m[2][2] * p[2] * p[2] + g_inv.m[3][3] * p[3] * p[3]
          + 2.0 * g_inv.m[0][3] * p[0] * p[3];
    }
    return 0.5 * H;
}

} // namespace cuda

#endif // GRRT_CUDA_GEODESIC_H
