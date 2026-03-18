#include "grrt/geodesic/rk4.h"
#include <cmath>
#include <algorithm>

namespace grrt {

GeodesicState RK4::derivatives(const Metric& metric, const GeodesicState& state) {
    const Vec4& x = state.position;
    const Vec4& p = state.momentum;

    // dx^μ/dλ = g^μν p_ν  (raise momentum with inverse metric)
    Matrix4 g_inv = metric.g_upper(x);
    Vec4 dx = g_inv.contract(p);

    // dp_μ/dλ = -½ Σ_{α,β} (∂g^αβ/∂x^μ) p_α p_β
    // Compute ∂g^αβ/∂x^μ by central finite differences
    Vec4 dp;
    for (int mu = 0; mu < 4; ++mu) {
        // Perturb position along coordinate mu
        Vec4 x_plus = x;
        Vec4 x_minus = x;
        x_plus[mu] += fd_epsilon;
        x_minus[mu] -= fd_epsilon;

        Matrix4 g_inv_plus = metric.g_upper(x_plus);
        Matrix4 g_inv_minus = metric.g_upper(x_minus);

        // Sum over all α, β
        double force = 0.0;
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                double dg = (g_inv_plus.m[a][b] - g_inv_minus.m[a][b]) / (2.0 * fd_epsilon);
                force += dg * p[a] * p[b];
            }
        }
        dp[mu] = -0.5 * force;
    }

    return {dx, dp};
}

GeodesicState RK4::step(const Metric& metric, const GeodesicState& state, double dl) const {
    // Classic RK4: compute 4 derivative evaluations, combine
    auto add = [](const GeodesicState& s, const GeodesicState& ds, double h) -> GeodesicState {
        return {s.position + ds.position * h, s.momentum + ds.momentum * h};
    };

    GeodesicState k1 = derivatives(metric, state);
    GeodesicState k2 = derivatives(metric, add(state, k1, dl * 0.5));
    GeodesicState k3 = derivatives(metric, add(state, k2, dl * 0.5));
    GeodesicState k4 = derivatives(metric, add(state, k3, dl));

    Vec4 new_pos = state.position
        + (k1.position + k2.position * 2.0 + k3.position * 2.0 + k4.position) * (dl / 6.0);
    Vec4 new_mom = state.momentum
        + (k1.momentum + k2.momentum * 2.0 + k3.momentum * 2.0 + k4.momentum) * (dl / 6.0);

    return {new_pos, new_mom};
}

AdaptiveResult RK4::adaptive_step(const Metric& metric, const GeodesicState& state,
                                  double dlambda, double tolerance) const {
    constexpr double dl_min = 1e-6;
    constexpr int max_retries = 20;
    constexpr double eps = 1e-10;

    double dl = dlambda;

    for (int retry = 0; retry < max_retries; ++retry) {
        // One full step
        GeodesicState s_full = step(metric, state, dl);

        // Two half steps
        GeodesicState s_mid = step(metric, state, dl * 0.5);
        GeodesicState s_half = step(metric, s_mid, dl * 0.5);

        // Compute max relative error across spatial position (r,θ,φ) and all momentum
        double err = 0.0;
        for (int i = 1; i < 4; ++i) {  // Skip t (index 0) for position
            double diff = std::abs(s_full.position[i] - s_half.position[i]);
            double scale = std::abs(s_half.position[i]) + eps;
            err = std::max(err, diff / scale);
        }
        for (int i = 0; i < 4; ++i) {  // All momentum components
            double diff = std::abs(s_full.momentum[i] - s_half.momentum[i]);
            double scale = std::abs(s_half.momentum[i]) + eps;
            err = std::max(err, diff / scale);
        }

        if (err <= tolerance) {
            // Accept — decide whether to grow step
            double next_dl = dl;
            if (err < tolerance * 0.01) {
                next_dl = dl * 2.0;
            }
            // Clamp max
            double r = std::abs(s_half.position[1]);
            next_dl = std::min(next_dl, 5.0 * std::max(r, 1.0));
            return {s_half, next_dl};
        }

        // Reject — shrink step
        dl *= 0.5;
        if (dl < dl_min) {
            // Force-accept at minimum step size to prevent infinite loop
            double r = std::abs(s_half.position[1]);
            double next_dl = std::min(dl_min, 5.0 * std::max(r, 1.0));
            return {s_half, next_dl};
        }
    }

    // Exhausted retries — force-accept whatever we have
    GeodesicState s_mid = step(metric, state, dl * 0.5);
    GeodesicState s_half = step(metric, s_mid, dl * 0.5);
    return {s_half, dl};
}

} // namespace grrt
