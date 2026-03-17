#ifndef GRRT_CELESTIAL_SPHERE_H
#define GRRT_CELESTIAL_SPHERE_H

#include "grrt/math/vec3.h"
#include "grrt/math/vec4.h"
#include <vector>

namespace grrt {

class CelestialSphere {
public:
    explicit CelestialSphere(int num_stars = 5000, unsigned int seed = 42);

    // Sample the sky at the given escaped ray position
    Vec3 sample(const Vec4& position) const;

private:
    struct Star {
        double theta;
        double phi;
        double brightness;
    };

    std::vector<Star> stars_;
    double angular_tolerance_ = 0.01;

    static constexpr int grid_theta = 180;
    static constexpr int grid_phi = 360;
    std::vector<std::vector<int>> grid_;

    int grid_index(int t_bin, int p_bin) const;
    void build_grid();
};

} // namespace grrt

#endif
