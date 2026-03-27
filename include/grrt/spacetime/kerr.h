#ifndef GRRT_KERR_H
#define GRRT_KERR_H

#include "grrt/spacetime/metric.h"

namespace grrt {

class Kerr : public Metric {
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

private:
    double mass_;  // M
    double spin_;  // a, with |a| < M

    double sigma(double r, double theta) const;
    double delta(double r) const;
};

} // namespace grrt

#endif
