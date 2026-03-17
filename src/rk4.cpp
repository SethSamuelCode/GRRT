#include "grrt/geodesic/rk4.h"

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

} // namespace grrt
