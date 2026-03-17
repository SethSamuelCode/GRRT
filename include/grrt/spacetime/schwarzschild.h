#ifndef GRRT_SCHWARZSCHILD_H
#define GRRT_SCHWARZSCHILD_H

#include "grrt/spacetime/metric.h"

namespace grrt {

class Schwarzschild : public Metric {
public:
    explicit Schwarzschild(double mass);

    Matrix4 g_lower(const Vec4& x) const override;
    Matrix4 g_upper(const Vec4& x) const override;
    double horizon_radius() const override;

private:
    double mass_;
};

} // namespace grrt

#endif
