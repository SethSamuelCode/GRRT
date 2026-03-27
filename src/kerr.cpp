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

Vec4 Kerr::geodesic_force(const Vec4& x, const Vec4& velocity) const {
    const double M = mass_;
    const double a = spin_;
    const double r = x[1];
    const double theta = x[2];
    const double r2 = r * r;
    const double a2 = a * a;

    double sin_t = std::sin(theta);
    double cos_t = std::cos(theta);
    if (std::abs(sin_t) < 1e-10) sin_t = (sin_t >= 0.0) ? 1e-10 : -1e-10;
    const double sin2 = sin_t * sin_t;
    const double sin4 = sin2 * sin2;
    const double sin2theta = 2.0 * sin_t * cos_t;  // sin(2θ)
    const double cos2 = cos_t * cos_t;

    const double Sigma = r2 + a2 * cos2;
    const double Delta = r2 - 2.0 * M * r + a2;
    const double Sigma2 = Sigma * Sigma;
    const double Delta2 = Delta * Delta;
    const double Mr = M * r;
    const double Sigma_m2r2 = Sigma - 2.0 * r2;  // = a²cos²θ - r²

    const double vt = velocity[0], vr = velocity[1];
    const double vth = velocity[2], vph = velocity[3];

    // ---- r-derivatives of covariant metric ----
    const double dg_tt_dr = 2.0 * M * Sigma_m2r2 / Sigma2;
    const double dg_tphi_dr = -2.0 * M * a * sin2 * Sigma_m2r2 / Sigma2;
    const double dg_rr_dr = (2.0 * r * Delta - 2.0 * Sigma * (r - M)) / Delta2;
    const double dg_phph_dr = 2.0 * r * sin2
                              + 2.0 * M * a2 * sin4 * Sigma_m2r2 / Sigma2;

    const double dp_r = 0.5 * (
        dg_tt_dr * vt * vt
        + 2.0 * dg_tphi_dr * vt * vph
        + dg_rr_dr * vr * vr
        + 2.0 * r * vth * vth
        + dg_phph_dr * vph * vph
    );

    // ---- θ-derivatives of covariant metric ----
    const double dg_tt_dth = 2.0 * Mr * a2 * sin2theta / Sigma2;
    const double dg_tphi_dth = -2.0 * Mr * a * sin2theta * (Sigma + a2 * sin2) / Sigma2;
    const double dg_rr_dth = -a2 * sin2theta / Delta;
    const double dg_phph_dth = (r2 + a2) * sin2theta
        + 2.0 * Mr * a2 * sin2theta * sin2 * (2.0 * Sigma + a2 * sin2) / Sigma2;

    const double dp_th = 0.5 * (
        dg_tt_dth * vt * vt
        + 2.0 * dg_tphi_dth * vt * vph
        + dg_rr_dth * vr * vr
        + (-a2 * sin2theta) * vth * vth
        + dg_phph_dth * vph * vph
    );

    Vec4 dp;
    dp[0] = 0.0;
    dp[1] = dp_r;
    dp[2] = dp_th;
    dp[3] = 0.0;
    return dp;
}

Metric::DerivResult Kerr::compute_derivatives(const Vec4& x, const Vec4& p) const {
    const double M = mass_;
    const double a = spin_;
    const double r = x[1];
    const double theta = x[2];
    const double r2 = r * r;
    const double a2 = a * a;

    // Trig — computed ONCE for both g_upper and geodesic_force
    double sin_t = std::sin(theta);
    double cos_t = std::cos(theta);
    if (std::abs(sin_t) < 1e-10) sin_t = (sin_t >= 0.0) ? 1e-10 : -1e-10;
    const double sin2 = sin_t * sin_t;
    const double cos2 = cos_t * cos_t;
    const double sin4 = sin2 * sin2;
    const double sin2theta = 2.0 * sin_t * cos_t;

    const double Sigma = r2 + a2 * cos2;
    const double Delta = r2 - 2.0 * M * r + a2;
    const double Sigma2 = Sigma * Sigma;
    const double Delta2 = Delta * Delta;
    const double Mr = M * r;

    // ---- g^μν (inverse metric) directly, without building g_lower first ----
    // g_tt = -(1 - 2Mr/Σ), g_tφ = -2Ma r sin²θ/Σ
    // g_rr = Σ/Δ, g_θθ = Σ, g_φφ = (r²+a²+2Ma²r sin²θ/Σ)sin²θ
    const double g_tt = -(1.0 - 2.0 * Mr / Sigma);
    const double g_tphi = -2.0 * M * a * r * sin2 / Sigma;
    const double g_rr = Sigma / Delta;
    const double g_thth = Sigma;
    const double g_phph = (r2 + a2 + 2.0 * M * a2 * r * sin2 / Sigma) * sin2;

    // Block inverse of (t,φ) 2×2
    const double det_tf = g_tt * g_phph - g_tphi * g_tphi;
    const double ginv_tt = g_phph / det_tf;
    const double ginv_tphi = -g_tphi / det_tf;
    const double ginv_rr = 1.0 / g_rr;
    const double ginv_thth = 1.0 / g_thth;
    const double ginv_phph = g_tt / det_tf;

    // dx^μ/dλ = g^μν p_ν
    Vec4 dx;
    dx[0] = ginv_tt * p[0] + ginv_tphi * p[3];
    dx[1] = ginv_rr * p[1];
    dx[2] = ginv_thth * p[2];
    dx[3] = ginv_tphi * p[0] + ginv_phph * p[3];

    // ---- Geodesic force (reuses sin/cos/Sigma/Delta from above) ----
    const double Sigma_m2r2 = Sigma - 2.0 * r2;

    // r-derivatives of covariant metric
    const double dg_tt_dr = 2.0 * M * Sigma_m2r2 / Sigma2;
    const double dg_tphi_dr = -2.0 * M * a * sin2 * Sigma_m2r2 / Sigma2;
    const double dg_rr_dr = (2.0 * r * Delta - 2.0 * Sigma * (r - M)) / Delta2;
    const double dg_phph_dr = 2.0 * r * sin2
                              + 2.0 * M * a2 * sin4 * Sigma_m2r2 / Sigma2;

    const double dp_r = 0.5 * (
        dg_tt_dr * dx[0] * dx[0]
        + 2.0 * dg_tphi_dr * dx[0] * dx[3]
        + dg_rr_dr * dx[1] * dx[1]
        + 2.0 * r * dx[2] * dx[2]
        + dg_phph_dr * dx[3] * dx[3]
    );

    // θ-derivatives of covariant metric
    const double dg_tt_dth = 2.0 * Mr * a2 * sin2theta / Sigma2;
    const double dg_tphi_dth = -2.0 * Mr * a * sin2theta * (Sigma + a2 * sin2) / Sigma2;
    const double dg_rr_dth = -a2 * sin2theta / Delta;
    const double dg_phph_dth = (r2 + a2) * sin2theta
        + 2.0 * Mr * a2 * sin2theta * sin2 * (2.0 * Sigma + a2 * sin2) / Sigma2;

    const double dp_th = 0.5 * (
        dg_tt_dth * dx[0] * dx[0]
        + 2.0 * dg_tphi_dth * dx[0] * dx[3]
        + dg_rr_dth * dx[1] * dx[1]
        + (-a2 * sin2theta) * dx[2] * dx[2]
        + dg_phph_dth * dx[3] * dx[3]
    );

    Vec4 dp;
    dp[0] = 0.0;
    dp[1] = dp_r;
    dp[2] = dp_th;
    dp[3] = 0.0;

    return {dx, dp};
}

} // namespace grrt
