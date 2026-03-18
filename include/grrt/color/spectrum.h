#ifndef GRRT_SPECTRUM_H
#define GRRT_SPECTRUM_H

#include "grrt/math/vec3.h"
#include <vector>

namespace grrt {

class SpectrumLUT {
public:
    SpectrumLUT(double t_min = 1000.0, double t_max = 100000.0, int num_entries = 1000);

    // Look up color × luminosity for a given temperature
    Vec3 temperature_to_color(double temperature) const;

    // Look up chromaticity only (no luminosity scaling) for a given temperature
    Vec3 chromaticity(double temperature) const;

    // Look up relative luminosity (σT⁴ normalized) for a given temperature
    double luminosity(double temperature) const;

private:
    double t_min_;
    double t_max_;
    int num_entries_;
    std::vector<Vec3> color_lut_;
    std::vector<double> luminosity_lut_;

    static double planck(double wavelength_m, double temperature);
    Vec3 blackbody_to_rgb(double temperature) const;
};

} // namespace grrt

#endif
