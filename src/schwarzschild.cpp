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

} // namespace grrt
