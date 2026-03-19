#include "grrt/spacetime/schwarzschild.h"
#include <cmath>
#include <algorithm>

namespace grrt {

Schwarzschild::Schwarzschild(double mass) : mass_(mass) {}

Matrix4 Schwarzschild::g_lower(const Vec4& x) const {
    const double r = x[1];
    const double theta = x[2];
    const double sin_theta = std::max(std::abs(std::sin(theta)), 1e-10);

    const double f = 1.0 - 2.0 * mass_ / r;  // 1 - 2M/r

    return Matrix4::diagonal(
        -f,              // g_tt
        1.0 / f,         // g_rr
        r * r,           // g_θθ
        r * r * sin_theta * sin_theta  // g_φφ
    );
}

Matrix4 Schwarzschild::g_upper(const Vec4& x) const {
    // For diagonal metric, g^μν = 1/g_μν
    return g_lower(x).inverse_diagonal();
}

double Schwarzschild::horizon_radius() const {
    return 2.0 * mass_;
}

double Schwarzschild::isco_radius() const {
    return 6.0 * mass_;
}

Vec4 Schwarzschild::geodesic_force(const Vec4& x, const Vec4& velocity) const {
    const double M = mass_;
    const double r = x[1];
    const double theta = x[2];
    const double r2 = r * r;
    const double f = 1.0 - 2.0 * M / r;

    double sin_t = std::sin(theta);
    if (std::abs(sin_t) < 1e-10) sin_t = (sin_t >= 0.0) ? 1e-10 : -1e-10;
    const double sin2 = sin_t * sin_t;

    const double vt = velocity[0], vr = velocity[1];
    const double vth = velocity[2], vph = velocity[3];

    Vec4 dp;
    // dp_t = dp_φ = 0 (Killing symmetries)
    dp[0] = 0.0;
    dp[3] = 0.0;

    // dp_r = ½[∂g_tt/∂r·vt² + ∂g_rr/∂r·vr² + ∂g_θθ/∂r·vθ² + ∂g_φφ/∂r·vφ²]
    const double two_M_over_r2 = 2.0 * M / r2;
    dp[1] = 0.5 * (
        -two_M_over_r2 * vt * vt
        - two_M_over_r2 / (f * f) * vr * vr
        + 2.0 * r * vth * vth
        + 2.0 * r * sin2 * vph * vph
    );

    // dp_θ = ½[∂g_φφ/∂θ·vφ²]  where ∂g_φφ/∂θ = r²sin(2θ)
    dp[2] = 0.5 * r2 * std::sin(2.0 * theta) * vph * vph;

    return dp;
}

} // namespace grrt
