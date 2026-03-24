#ifndef GRRT_VOLUMETRIC_DISK_H
#define GRRT_VOLUMETRIC_DISK_H

#include "grrt/color/opacity.h"
#include "grrt/math/noise.h"
#include "grrt_export.h"
#include <vector>
#include <cstdint>

namespace grrt {

/// Parameters for the volumetric accretion disk model.
struct VolumetricParams {
    double alpha      = 0.1;    ///< Shakura-Sunyaev viscosity parameter
    double turbulence = 0.4;    ///< Noise amplitude delta
    uint32_t seed     = 42;     ///< Noise seed
    double tau_mid    = 100.0;  ///< Midplane optical depth normalization at peak-flux radius
};

/// Volumetric accretion disk with Shakura-Sunyaev vertical structure,
/// Eddington radiative equilibrium, frequency-dependent opacity, and
/// plunging-region dynamics inside the ISCO.
///
/// The disk is constructed in Boyer-Lindquist coordinates around a Kerr
/// black hole. The vertical density and temperature profiles are solved
/// via numerical hydrostatic equilibrium (iterated with the Eddington
/// T-tau relation) and stored as 2D lookup tables. The midplane density
/// is normalized so that the vertical optical depth at the peak-flux
/// radius equals tau_mid.
///
/// Physical units: coordinates and H(r) are in geometric units (G=c=1,
/// M sets scale). Temperature is in Kelvin. Density is stored in
/// geometric normalization and converted to CGS via rho_scale when needed.
class GRRT_EXPORT VolumetricDisk {
public:
    /// Construct the volumetric disk. This builds all LUTs (flux, radial
    /// structure, vertical profiles, opacity) and may take several seconds.
    /// @param mass   Black hole mass M (geometric units, typically 1.0)
    /// @param spin   Kerr spin parameter a, |a| < M
    /// @param r_outer Outer edge of the disk [M]
    /// @param peak_temperature Peak effective temperature [K]
    /// @param params Additional configuration (viscosity, noise, etc.)
    VolumetricDisk(double mass, double spin, double r_outer,
                   double peak_temperature, const VolumetricParams& params = {});

    // --- Accessors for raymarching ---

    /// Scale height H(r) [geometric units]. Frozen at H(r_isco) for r < r_isco.
    double scale_height(double r) const;

    /// Total density at (r, z, phi) including taper and turbulent noise [geometric, scaled].
    double density(double r, double z, double phi) const;

    /// Total density in CGS [g/cm^3].
    double density_cgs(double r, double z, double phi) const;

    /// Temperature at (r, z) from precomputed vertical profile LUT [K].
    double temperature(double r, double z) const;

    /// ISCO taper factor: 1 for r >= r_isco, Gaussian decay inside.
    double taper(double r) const;

    /// Whether the point (r, |z|) is within the disk volume bounds.
    bool inside_volume(double r, double z) const;

    // --- Kerr orbital mechanics ---

    /// Kerr prograde orbital frequency Omega_orb(r).
    double omega_orb(double r) const;

    /// Kerr vertical epicyclic frequency squared Omega_z^2(r).
    double omega_z_sq(double r) const;

    // --- 4-velocity for redshift computation ---

    /// Circular orbit 4-velocity at r >= r_isco.
    /// Writes u^t and u^phi; u^r = u^theta = 0.
    void circular_velocity(double r, double& ut, double& uphi) const;

    /// Plunging geodesic 4-velocity at r < r_isco (BPT72 constants of motion).
    /// Writes u^t, u^r (negative, infall), u^phi.
    void plunging_velocity(double r, double theta,
                           double& ut, double& ur, double& uphi) const;

    // --- Opacity LUT access ---
    const OpacityLUTs& opacity_luts() const { return opacity_luts_; }

    // --- CUDA data accessors ---
    const std::vector<double>& scale_height_lut() const { return H_lut_; }
    const std::vector<double>& rho_mid_lut() const { return rho_mid_lut_; }
    const std::vector<double>& density_profile_lut() const { return rho_profile_lut_; }
    const std::vector<double>& temperature_profile_lut() const { return T_profile_lut_; }
    int radial_bins() const { return n_r_; }
    int vertical_bins() const { return n_z_; }
    double r_min() const { return r_min_; }
    double r_max() const { return r_outer_; }
    double r_isco() const { return r_isco_; }
    double r_horizon() const { return r_horizon_; }
    double rho_scale() const { return rho_scale_; }
    const SimplexNoise3D& noise() const { return noise_; }
    double E_isco() const { return E_isco_; }
    double L_isco() const { return L_isco_; }
    double taper_width() const { return taper_width_; }
    double turbulence() const { return params_.turbulence; }
    double peak_temperature() const { return peak_temperature_; }

private:
    double mass_, spin_, r_outer_, peak_temperature_;
    double r_isco_, r_horizon_;
    double r_min_;         ///< Inner bound (slightly outside horizon)
    double taper_width_;   ///< Gaussian taper width inside ISCO
    VolumetricParams params_;
    SimplexNoise3D noise_;

    /// BPT72 conserved quantities at ISCO (for plunging 4-velocity)
    double E_isco_, L_isco_;

    // Radial LUTs (n_r_ bins from r_min_ to r_outer_)
    int n_r_ = 500;
    std::vector<double> H_lut_;        ///< scale height H(r) [geometric]
    std::vector<double> rho_mid_lut_;  ///< midplane density [geometric, scaled]
    std::vector<double> T_eff_lut_;    ///< effective temperature T_eff(r) [K]

    // 2D vertical structure LUTs (n_r_ x n_z_)
    int n_z_ = 64;
    std::vector<double> rho_profile_lut_;  ///< rho(r,z)/rho_mid(r) [normalized]
    std::vector<double> T_profile_lut_;    ///< T(r,z) [K]

    // Opacity LUTs
    OpacityLUTs opacity_luts_;

    // Density normalization factor
    double rho_scale_ = 1.0;

    // --- Construction helpers ---
    void build_flux_lut(std::vector<double>& flux, double& flux_max) const;
    void compute_radial_structure();
    void compute_vertical_profiles();
    void normalize_density();

    // --- LUT interpolation helpers ---
    double interp_radial(const std::vector<double>& lut, double r) const;
    double interp_2d(const std::vector<double>& lut, double r, double z_abs) const;
};

} // namespace grrt

#endif // GRRT_VOLUMETRIC_DISK_H
