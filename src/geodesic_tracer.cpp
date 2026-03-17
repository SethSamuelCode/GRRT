#include "grrt/geodesic/geodesic_tracer.h"

namespace grrt {

GeodesicTracer::GeodesicTracer(const Metric& metric, const Integrator& integrator,
                               int max_steps, double r_escape)
    : metric_(metric), integrator_(integrator),
      max_steps_(max_steps), r_escape_(r_escape) {}

RayTermination GeodesicTracer::trace(GeodesicState& state) const {
    const double r_horizon = metric_.horizon_radius();

    for (int step = 0; step < max_steps_; ++step) {
        const double r = state.position[1];

        // Check termination conditions
        if (r < r_horizon + horizon_epsilon_) {
            return RayTermination::Horizon;
        }
        if (r > r_escape_) {
            return RayTermination::Escaped;
        }

        // Step size scales with r: smaller near the hole
        const double dlambda = 0.005 * r;

        state = integrator_.step(metric_, state, dlambda);
    }

    return RayTermination::MaxSteps;
}

} // namespace grrt
