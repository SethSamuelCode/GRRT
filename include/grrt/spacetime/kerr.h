#ifndef GRRT_KERR_H
#define GRRT_KERR_H

#include "grrt/spacetime/metric.h"
#include "grrt/math/fast_trig.h"
#include "grrt_export.h"
#include <cmath>

namespace grrt {

class GRRT_EXPORT Kerr : public Metric {
public:
    Kerr(double mass, double spin);

    Matrix4 g_lower(const Vec4& x) const override;
    Matrix4 g_upper(const Vec4& x) const override;
    double horizon_radius() const override;
    double isco_radius() const override;
    Vec4 geodesic_force(const Vec4& x, const Vec4& velocity) const override;
    DerivResult compute_derivatives(const Vec4& x, const Vec4& p) const override;

    double mass() const { return mass_; }
    double spin() const { return spin_; }

    /// Hot-path fused derivative — force-inlined into the RK4 loop
#ifdef _MSC_VER
    __forceinline
#else
    __attribute__((always_inline)) inline
#endif
    DerivResult compute_derivatives_inline(const Vec4& x, const Vec4& p) const {
        const double M = mass_;
        const double a = spin_;
        const double r = x[1];
        const double theta = x[2];
        const double r2 = r * r;
        const double a2 = a * a;

        double sin_t, cos_t;
        fast_sincos(theta, sin_t, cos_t);
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

        const double g_tt = -(1.0 - 2.0 * Mr / Sigma);
        const double g_tphi = -2.0 * M * a * r * sin2 / Sigma;
        const double g_rr = Sigma / Delta;
        const double g_thth = Sigma;
        const double g_phph = (r2 + a2 + 2.0 * M * a2 * r * sin2 / Sigma) * sin2;

        const double det_tf = g_tt * g_phph - g_tphi * g_tphi;
        const double ginv_tt = g_phph / det_tf;
        const double ginv_tphi = -g_tphi / det_tf;
        const double ginv_rr = 1.0 / g_rr;
        const double ginv_thth = 1.0 / g_thth;
        const double ginv_phph = g_tt / det_tf;

        Vec4 dx;
        dx[0] = ginv_tt * p[0] + ginv_tphi * p[3];
        dx[1] = ginv_rr * p[1];
        dx[2] = ginv_thth * p[2];
        dx[3] = ginv_tphi * p[0] + ginv_phph * p[3];

        const double Sigma_m2r2 = Sigma - 2.0 * r2;

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

private:
    double mass_;
    double spin_;

    double sigma(double r, double theta) const;
    double delta(double r) const;
};

} // namespace grrt

#endif
