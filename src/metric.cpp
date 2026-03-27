#include "grrt/spacetime/metric.h"
#include <cmath>

namespace grrt {

Vec4 Metric::geodesic_force(const Vec4& x, const Vec4& velocity) const {
    // Default: finite differences on covariant metric
    constexpr double eps = 1e-6;
    constexpr double inv_2eps = 1.0 / (2.0 * eps);

    Vec4 dp;
    for (int mu = 0; mu < 4; ++mu) {
        Vec4 x_plus = x, x_minus = x;
        x_plus[mu] += eps;
        x_minus[mu] -= eps;

        Matrix4 g_plus = g_lower(x_plus);
        Matrix4 g_minus = g_lower(x_minus);

        double force = 0.0;
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                double dg = (g_plus.m[a][b] - g_minus.m[a][b]) * inv_2eps;
                force += dg * velocity[a] * velocity[b];
            }
        }
        dp[mu] = 0.5 * force;
    }
    return dp;
}

Metric::DerivResult Metric::compute_derivatives(const Vec4& x, const Vec4& p) const {
    // Default: separate calls (no trig sharing)
    Matrix4 g_inv = g_upper(x);
    Vec4 dx = g_inv.contract(p);
    Vec4 dp = geodesic_force(x, dx);
    return {dx, dp};
}

} // namespace grrt
