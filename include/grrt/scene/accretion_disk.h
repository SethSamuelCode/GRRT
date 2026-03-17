#ifndef GRRT_ACCRETION_DISK_H
#define GRRT_ACCRETION_DISK_H

#include "grrt/math/vec3.h"
#include "grrt/math/vec4.h"
#include "grrt/color/spectrum.h"
#include <vector>

namespace grrt {

class AccretionDisk {
public:
    AccretionDisk(double mass, double spin, double r_isco,
                  double r_outer, double peak_temperature, int flux_lut_size = 500);

    double r_inner() const { return r_inner_; }
    double r_outer() const { return r_outer_; }

    double temperature(double r) const;

    Vec3 emission(double r_cross, const Vec4& p_cross,
                  double observer_r, const SpectrumLUT& spectrum) const;

private:
    double mass_;
    double spin_;     // a (Kerr spin parameter, 0 for Schwarzschild)
    double r_inner_;  // ISCO
    double r_outer_;
    double peak_temperature_;

    std::vector<double> flux_lut_;
    double flux_max_;
    double flux_r_min_;
    double flux_r_max_;
    int flux_lut_size_;

    // Kerr-aware circular orbit quantities
    double omega_kepler(double r) const;
    double Omega(double r) const;
    double E_circ(double r) const;
    double L_circ(double r) const;

    void build_flux_lut();
    double flux(double r) const;
    double redshift(double r_cross, const Vec4& p_cross, double observer_r) const;
};

} // namespace grrt

#endif
