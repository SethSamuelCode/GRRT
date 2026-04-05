#ifndef GRRT_CAMERA_H
#define GRRT_CAMERA_H

#include "grrt/math/vec4.h"
#include "grrt/spacetime/metric.h"
#include "grrt/geodesic/integrator.h"
#include "grrt_export.h"

namespace grrt {

class GRRT_EXPORT Camera {
public:
    Camera(const Metric& metric, double r_obs, double theta_obs, double phi_obs,
           double fov, int width, int height);

    // Generate initial geodesic state for pixel (i, j) at pixel center
    GeodesicState ray_for_pixel(int i, int j) const;

    // Generate initial geodesic state for fractional pixel coordinate (x, y)
    // where (0.5, 0.5) is the center of the top-left pixel
    GeodesicState ray_for_pixel(double x, double y) const;

private:
    const Metric& metric_;
    Vec4 position_;  // Observer's 4-position
    double fov_;
    int width_;
    int height_;

    // Orthonormal tetrad at observer's position
    Vec4 e0_;  // Timelike (normalized 4-velocity)
    Vec4 e1_;  // Right (φ direction)
    Vec4 e2_;  // Up (θ direction)
    Vec4 e3_;  // Forward (radially inward, toward hole)

    void build_tetrad();
};

} // namespace grrt

#endif
