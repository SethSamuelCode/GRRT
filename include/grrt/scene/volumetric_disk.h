#ifndef GRRT_VOLUMETRIC_DISK_H
#define GRRT_VOLUMETRIC_DISK_H

#include "grrt/color/opacity.h"
#include "grrt/math/noise.h"
#include "grrt_export.h"
#include <utility>
#include <vector>
#include <cstdint>
#include <string>

namespace grrt {

/// Parameters for the volumetric accretion disk model.
struct VolumetricParams {
    // --- Physical (unchanged) ---
    double alpha          = 0.1;
    uint32_t seed         = 42;
    double tau_mid        = 100.0;
    double opacity_nu_min = 1e14;
    double opacity_nu_max = 1e16;
    int noise_octaves     = 2;

    // --- Noise composition (CHANGED semantics) ---
    double turbulence  = 1.0;   ///< Dimensionless boost on physically-derived σ_s.
                                ///< 1.0 = pure physical. 0.0 = axisymmetric.
    double noise_scale = 0.0;   ///< Multiplier on c_corr·H(r). 0 = auto.

    // --- Noise physics (NEW — data-derived defaults) ---
    double noise_compressive_b              = 0.0;  ///< 0 = derive from peak β
    double noise_correlation_length_factor  = 0.5;  ///< c_corr; eddy length / H(r)

    // --- Smooth volumetric envelope (NEW) ---
    double outer_taper_width        = 0.0;   ///< 0 = auto = 2·H(r_outer); units M
    double plunging_h_decay_exponent = 0.5;  ///< H(r<r_isco) = H_isco·taper(r)^p

    // --- LUT sizing (NEW — data-driven with manual override) ---
    int bins_per_h         = 0;          ///< 0 = auto via Richardson refinement
    int bins_per_gradient  = 0;          ///< 0 = auto via Richardson refinement
    double target_lut_eps  = 1e-3;       ///< Refinement tolerance (relative)
    int min_n_r            = 256;
    int min_n_z            = 64;
    int max_n_r            = 4096;
    int max_n_z            = 1024;
    int refine_num_frequencies = 8;      ///< Frequency samples for max-envelope
};

enum class WarningSeverity {
    Info       = 0,
    Warning    = 1,
    Promptable = 2,
    Severe     = 3
};

struct ConstructionWarning {
    WarningSeverity severity;
    std::string code;
    std::string message;
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
    double outer_taper_width() const { return outer_taper_width_; }
    double r_isco() const { return r_isco_; }
    double r_horizon() const { return r_horizon_; }
    double rho_scale() const { return rho_scale_; }
    const SimplexNoise3D& noise() const { return noise_; }
    double E_isco() const { return E_isco_; }
    double L_isco() const { return L_isco_; }
    double taper_width() const { return taper_width_; }
    double turbulence() const { return params_.turbulence; }
    double peak_temperature() const { return peak_temperature_; }
    /// Legacy accessor — returns the noise correlation length at peak-flux radius
    /// for compatibility with the CUDA host data layout. Computes on demand.
    double noise_scale() const {
        if (params_.noise_scale > 0.0) return params_.noise_scale;
        const double c_corr = (params_.noise_correlation_length_factor > 0.0)
                            ? params_.noise_correlation_length_factor : 0.5;
        return c_corr * H_lut_[n_r_ / 2];
    }

    /// Physical noise amplitude σ_s = b·√(ln(1+α)). Set during construction.
    double sigma_s_phys() const { return sigma_s_phys_; }

    /// Atmosphere extent z_max(r) [geometric]. Interpolated from z_max_lut_.
    double z_max_at(double r) const;
    const std::vector<double>& z_max_lut() const { return z_max_lut_; }

    /// Warnings collected during construction. Pointer-stable for the lifetime of
    /// this VolumetricDisk instance.
    const std::vector<ConstructionWarning>& warnings() const { return warnings_; }

    /// Number of warnings with severity >= Promptable.
    int promptable_count() const;

private:
    double mass_, spin_, r_outer_, peak_temperature_;
    double r_isco_, r_horizon_;
    double r_min_;              ///< Inner bound (slightly outside horizon)
    double taper_width_;        ///< Gaussian taper width inside ISCO
    double outer_taper_width_ = 0.0;   ///< Resolved width of the outer radial taper [M]
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
    int n_z_ = 128;
    std::vector<double> rho_profile_lut_;  ///< rho(r,z)/rho_mid(r) [normalized]
    std::vector<double> T_profile_lut_;    ///< T(r,z) [K]
    std::vector<double> z_max_lut_;       ///< atmosphere extent z_max(r) [geometric]

    // Opacity LUTs
    OpacityLUTs opacity_luts_;

    // Density normalization factor
    double rho_scale_ = 1.0;

    double sigma_s_phys_ = 0.0;

    std::vector<ConstructionWarning> warnings_;
    void emit(WarningSeverity sev, std::string code, std::string message);

    struct ColumnSolution {
        double z_max = 0.0;
        std::vector<double> rho_z;   // size n_z, normalized so rho_z[0] = 1
        std::vector<double> T_z;     // size n_z, in Kelvin
        double max_delta = 0.0;  ///< Final iteration-to-iteration relative density delta
    };

    /// Solve the hydrostatic-equilibrium ODE for one radial column at a given vertical
    /// resolution. Iteratively extends z_max until rho(z_max) < CONV_FLOOR or hits cap.
    ColumnSolution solve_column(double r, double H, double T_eff,
                                 double rho_mid_proportional, int n_z) const;

    /// Compare two ColumnSolutions (lo=coarse, hi=fine) using an optical-depth-weighted
    /// max-envelope metric. Returns a scalar error estimate for Richardson refinement.
    double compare_columns(const ColumnSolution& lo, const ColumnSolution& hi) const;

    /// Richardson refinement loop for vertical resolution. Doubles n_z until
    /// compare_columns() falls below target_lut_eps or max_n_z is reached.
    /// Mutates n_z_, z_max_lut_, rho_profile_lut_, T_profile_lut_.
    /// Emits a Promptable/Warning "n_z_cap" warning if capped before converging.
    [[maybe_unused]] int refine_n_z_globally();

    /// Richardson refinement loop for radial resolution. Doubles n_r until
    /// compare_radial() falls below target_lut_eps or max_n_r is reached.
    /// Mutates n_r_, H_lut_, rho_mid_lut_, T_eff_lut_.
    /// Emits a Promptable/Warning "n_r_cap" warning if capped before converging.
    [[maybe_unused]] int refine_n_r();

    /// Nested refinement: alternates refine_n_z_globally() and refine_n_r() until
    /// both converge to a fixed point or MAX_NESTED_ITERS is reached.
    /// Returns {n_r, n_z}.
    [[maybe_unused]] std::pair<int, int> nested_refine();

    // --- Construction helpers ---
    void build_flux_lut(std::vector<double>& flux, double& flux_max) const;
    void compute_radial_structure();
    void compute_plunging_region_decay();
    void apply_outer_radial_taper();
    void compute_vertical_profiles();
    void normalize_density();
    void compute_sigma_s_phys();

    bool validate_luts();

    // --- LUT interpolation helpers ---
    double interp_radial(const std::vector<double>& lut, double r) const;
    double interp_2d(const std::vector<double>& lut, double r, double z_abs) const;
};

} // namespace grrt

#endif // GRRT_VOLUMETRIC_DISK_H
