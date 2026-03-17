#include "grrt/spacetime/kerr.h"
#include <cmath>
#include <algorithm>

namespace grrt {

Kerr::Kerr(double mass, double spin) : mass_(mass), spin_(spin) {}

double Kerr::sigma(double r, double theta) const {
    double a = spin_;
    double cos_t = std::cos(theta);
    return r * r + a * a * cos_t * cos_t;
}

double Kerr::delta(double r) const {
    return r * r - 2.0 * mass_ * r + spin_ * spin_;
}

Matrix4 Kerr::g_lower(const Vec4& x) const {
    double r = x[1];
    double theta = x[2];
    double sin_t = std::sin(theta);
    double sin2 = sin_t * sin_t;
    sin2 = std::max(sin2, 1e-20);  // Avoid division by zero at poles

    double M = mass_;
    double a = spin_;
    double S = std::max(sigma(r, theta), 1e-20);
    double D = delta(r);

    Matrix4 g;
    g.m[0][0] = -(1.0 - 2.0 * M * r / S);                              // g_tt
    g.m[0][3] = -2.0 * M * a * r * sin2 / S;                            // g_tφ
    g.m[3][0] = g.m[0][3];                                               // g_φt
    g.m[1][1] = S / D;                                                    // g_rr
    g.m[2][2] = S;                                                        // g_θθ
    g.m[3][3] = (r * r + a * a + 2.0 * M * a * a * r * sin2 / S) * sin2; // g_φφ

    return g;
}

Matrix4 Kerr::g_upper(const Vec4& x) const {
    return g_lower(x).inverse();
}

double Kerr::horizon_radius() const {
    double M = mass_;
    double a = spin_;
    return M + std::sqrt(M * M - a * a);
}

double Kerr::isco_radius() const {
    double M = mass_;
    double a = spin_;
    double a_star = a / M;  // Dimensionless spin

    // Prograde ISCO (Bardeen, Press & Teukolsky 1972)
    double Z1 = 1.0 + std::cbrt(1.0 - a_star * a_star)
                     * (std::cbrt(1.0 + a_star) + std::cbrt(1.0 - a_star));
    double Z2 = std::sqrt(3.0 * a_star * a_star + Z1 * Z1);

    return M * (3.0 + Z2 - std::sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)));
}

} // namespace grrt
