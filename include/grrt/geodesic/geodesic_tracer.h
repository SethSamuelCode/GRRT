#ifndef GRRT_GEODESIC_TRACER_H
#define GRRT_GEODESIC_TRACER_H

#include "grrt/geodesic/integrator.h"
#include "grrt/math/vec3.h"

namespace grrt {

// Forward declarations
class AccretionDisk;
class SpectrumLUT;
class VolumetricDisk;

enum class RayTermination {
    Horizon,
    Escaped,
    MaxSteps
};

struct TraceResult {
    RayTermination termination;
    Vec3 accumulated_color;  // Sum of disk crossing emissions (linear HDR)
    Vec4 final_position;
    Vec4 final_momentum;
};

class GeodesicTracer {
public:
    GeodesicTracer(const Metric& metric, const Integrator& integrator,
                   double observer_r, int max_steps = 10000, double r_escape = 1000.0,
                   double tolerance = 1e-8,
                   const VolumetricDisk* vol_disk = nullptr);

    TraceResult trace(GeodesicState state,
                      const AccretionDisk* disk,
                      const SpectrumLUT* spectrum) const;

private:
    const Metric& metric_;
    const Integrator& integrator_;
    double observer_r_;
    int max_steps_;
    double r_escape_;
    double tolerance_;
    double horizon_epsilon_ = 0.01;
    const VolumetricDisk* vol_disk_ = nullptr;

    void raymarch_volumetric(GeodesicState& state, Vec3& color) const;
};

} // namespace grrt

#endif
