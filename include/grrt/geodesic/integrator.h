#ifndef GRRT_INTEGRATOR_H
#define GRRT_INTEGRATOR_H

#include "grrt/math/vec4.h"
#include "grrt/spacetime/metric.h"
#include "grrt_export.h"

namespace grrt {

struct GeodesicState {
    Vec4 position;  // x^μ = (t, r, θ, φ) — contravariant
    Vec4 momentum;  // p_μ — covariant
};

struct AdaptiveResult {
    GeodesicState state;    // Accepted new state (always valid)
    double next_dlambda;    // Recommended step size for next iteration
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
