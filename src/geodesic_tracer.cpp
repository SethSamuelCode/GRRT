#include "grrt/geodesic/geodesic_tracer.h"
#include "grrt/scene/accretion_disk.h"
#include "grrt/color/spectrum.h"
#include <cmath>
#include <numbers>

namespace grrt {

GeodesicTracer::GeodesicTracer(const Metric& metric, const Integrator& integrator,
                               double observer_r, int max_steps, double r_escape)
    : metric_(metric), integrator_(integrator),
      observer_r_(observer_r), max_steps_(max_steps), r_escape_(r_escape) {}

TraceResult GeodesicTracer::trace(GeodesicState state,
                                  const AccretionDisk* disk,
                                  const SpectrumLUT* spectrum) const {
    const double r_horizon = metric_.horizon_radius();
    const double half_pi = std::numbers::pi / 2.0;
    Vec3 color;

    GeodesicState prev = state;

    for (int step = 0; step < max_steps_; ++step) {
        const double r = state.position[1];

        // Check termination
        if (r < r_horizon + horizon_epsilon_) {
            return {RayTermination::Horizon, color, state.position, state.momentum};
        }
        if (r > r_escape_) {
            return {RayTermination::Escaped, color, state.position, state.momentum};
        }

        prev = state;

        const double dlambda = 0.005 * r;
        state = integrator_.step(metric_, state, dlambda);

        // Check for disk crossing (θ crosses π/2)
        if (disk && spectrum) {
            double theta_prev = prev.position[2];
            double theta_new = state.position[2];

            if ((theta_prev - half_pi) * (theta_new - half_pi) < 0.0) {
                double frac = (half_pi - theta_prev) / (theta_new - theta_prev);

                double r_cross = prev.position[1] + frac * (state.position[1] - prev.position[1]);

                Vec4 p_cross;
                for (int mu = 0; mu < 4; ++mu) {
                    p_cross[mu] = prev.momentum[mu] + frac * (state.momentum[mu] - prev.momentum[mu]);
                }

                if (r_cross >= disk->r_inner() && r_cross <= disk->r_outer()) {
                    color += disk->emission(r_cross, p_cross, observer_r_, *spectrum);
                }
            }
        }
    }

    return {RayTermination::MaxSteps, color, state.position, state.momentum};
}

} // namespace grrt
