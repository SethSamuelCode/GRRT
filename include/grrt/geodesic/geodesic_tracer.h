#ifndef GRRT_GEODESIC_TRACER_H
#define GRRT_GEODESIC_TRACER_H

#include "grrt/geodesic/integrator.h"

namespace grrt {

enum class RayTermination {
    Horizon,   // Hit the event horizon
    Escaped,   // Escaped to large radius
    MaxSteps   // Exceeded step limit
};

class GeodesicTracer {
public:
    GeodesicTracer(const Metric& metric, const Integrator& integrator,
                   int max_steps = 10000, double r_escape = 1000.0);

    RayTermination trace(GeodesicState& state) const;

private:
    const Metric& metric_;
    const Integrator& integrator_;
    int max_steps_;
    double r_escape_;
    double horizon_epsilon_ = 0.01;
};

} // namespace grrt

#endif
