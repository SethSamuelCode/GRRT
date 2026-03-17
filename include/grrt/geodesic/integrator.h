#ifndef GRRT_INTEGRATOR_H
#define GRRT_INTEGRATOR_H

#include "grrt/math/vec4.h"
#include "grrt/spacetime/metric.h"

namespace grrt {

struct GeodesicState {
    Vec4 position;  // x^μ = (t, r, θ, φ) — contravariant
    Vec4 momentum;  // p_μ — covariant
};

class Integrator {
public:
    virtual ~Integrator() = default;

    // Advance state by one step of size dlambda
    virtual GeodesicState step(const Metric& metric,
                               const GeodesicState& state,
                               double dlambda) const = 0;
};

} // namespace grrt

#endif
