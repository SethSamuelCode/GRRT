#ifndef GRRT_RK4_H
#define GRRT_RK4_H

#include "grrt/geodesic/integrator.h"

namespace grrt {

class RK4 : public Integrator {
public:
    GeodesicState step(const Metric& metric,
                       const GeodesicState& state,
                       double dlambda) const override;

private:
    // Compute derivatives: (dx^μ/dλ, dp_μ/dλ)
    static GeodesicState derivatives(const Metric& metric,
                                     const GeodesicState& state);

    // Finite-difference epsilon for metric derivatives
    static constexpr double fd_epsilon = 1e-6;
};

} // namespace grrt

#endif
