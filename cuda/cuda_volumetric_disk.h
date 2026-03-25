#ifndef GRRT_CUDA_VOLUMETRIC_DISK_H
#define GRRT_CUDA_VOLUMETRIC_DISK_H

/// @file cuda_volumetric_disk.h
/// @brief CUDA device functions for volumetric accretion disk raymarching.
///
/// Mirrors the CPU grrt::VolumetricDisk and GeodesicTracer::raymarch_volumetric()
/// implementations, adapted for GPU execution. Uses device pointers to LUTs
/// stored in global memory (uploaded from host before rendering).
///
/// The disk model includes:
/// - Shakura-Sunyaev radial structure with Novikov-Thorne flux profile
/// - Numerical hydrostatic equilibrium vertical profiles (precomputed on CPU)
/// - Frequency-dependent opacity via 3D/2D LUT interpolation
/// - Kerr plunging-region dynamics inside the ISCO
/// - Turbulent density perturbations via simplex noise
/// - Covariant radiative transfer with per-channel raymarching
///
/// Units: geometrized (G = c = 1) for coordinates and scale heights.
/// CGS for density, temperature, opacity. Coordinates: Boyer-Lindquist.

#include "cuda_math.h"
#include "cuda_metric.h"
#include "cuda_types.h"
#include "cuda_geodesic.h"
#include "cuda_noise.h"

namespace cuda {

// ---------------------------------------------------------------------------
// Device memory declarations for volumetric LUTs (defined in cuda_render.cu)
// ---------------------------------------------------------------------------

/// @name Radial LUTs (1D, indexed by radial bin)
/// @{
extern __device__ double* d_vol_H_lut;         ///< Scale height H(r) [geometric]
extern __device__ double* d_vol_rho_mid_lut;    ///< Midplane density rho_mid(r)
/// @}

/// @name Vertical profile LUTs (2D, indexed [r_bin * n_z + z_bin])
/// @{
extern __device__ double* d_vol_rho_profile_lut;  ///< rho(r,z)/rho_mid(r) [normalized]
extern __device__ double* d_vol_T_profile_lut;    ///< Temperature T(r,z) [K]
/// @}

/// @name Opacity LUTs (3D and 2D)
/// @{
extern __device__ double* d_opacity_kappa_abs_lut;   ///< 3D: kappa_abs(nu, rho, T) [cm^2/g]
extern __device__ double* d_opacity_kappa_es_lut;    ///< 2D: kappa_es(rho, T) [cm^2/g]
extern __device__ double* d_opacity_kappa_ross_lut;  ///< 2D: kappa_ross(rho, T) [cm^2/g]
extern __device__ double* d_opacity_mu_lut;          ///< 2D: mean molecular weight mu(rho, T)
/// @}

// ---------------------------------------------------------------------------
// CGS physical constants (hardcoded to avoid namespace imports on device)
// ---------------------------------------------------------------------------

/// @brief Planck constant [erg s]
__device__ constexpr double VOL_h_planck = 6.626070e-27;
/// @brief Boltzmann constant [erg/K]
__device__ constexpr double VOL_k_B = 1.380649e-16;
/// @brief Speed of light [cm/s]
__device__ constexpr double VOL_c_cgs = 2.997924e10;
/// @brief Thomson cross-section [cm^2]
__device__ constexpr double VOL_sigma_T = 6.652e-25;

// ---------------------------------------------------------------------------
// Forward declaration (vol_interp_2d needs vol_scale_height)
// ---------------------------------------------------------------------------

__device__ inline double vol_scale_height(double r, const RenderParams& params);

// ---------------------------------------------------------------------------
// LUT interpolation helpers
// ---------------------------------------------------------------------------

/// @brief Compute index and fractional offset for log-space interpolation.
///
/// Maps log10(val) into a uniform grid [log_min, log_max] with n bins.
/// Clamps to valid range [0, n-2].
///
/// @param log_val  log10 of the value to look up
/// @param log_min  log10 of the grid minimum
/// @param log_max  log10 of the grid maximum
/// @param n        Number of grid points
/// @param[out] idx   Integer index (0..n-2)
/// @param[out] frac  Fractional offset in [0, 1]
__device__ inline void vol_log_interp(double log_val, double log_min, double log_max,
                                       int n, int& idx, double& frac) {
    double t = (log_val - log_min) / (log_max - log_min) * (n - 1);
    if (t < 0.0) t = 0.0;
    if (t > (double)(n - 2)) t = (double)(n - 2);
    idx = (int)t;
    frac = t - idx;
}

/// @brief Linear interpolation on a 1D radial LUT.
///
/// The LUT covers [vol_r_min, vol_r_max] uniformly with vol_n_r points.
/// Clamps to boundary values outside the range.
///
/// @param lut     Device pointer to the 1D LUT
/// @param r       Radial coordinate
/// @param params  RenderParams containing grid metadata
__device__ inline double vol_interp_radial(const double* lut, double r,
                                            const RenderParams& params) {
    if (r <= params.vol_r_min) return lut[0];
    if (r >= params.vol_r_max) return lut[params.vol_n_r - 1];
    const double frac = (r - params.vol_r_min) / (params.vol_r_max - params.vol_r_min)
                        * (params.vol_n_r - 1);
    int idx = (int)frac;
    if (idx < 0) idx = 0;
    if (idx > params.vol_n_r - 2) idx = params.vol_n_r - 2;
    const double t = frac - idx;
    return lut[idx] * (1.0 - t) + lut[idx + 1] * t;
}

/// @brief Bilinear interpolation on a 2D (r, z) profile LUT.
///
/// The radial axis covers [vol_r_min, vol_r_max] with vol_n_r points.
/// The vertical axis covers [0, 3*H(r)] with vol_n_z points.
/// Returns 0 if z_abs >= 3*H(r).
///
/// @param lut     Device pointer to the 2D LUT (row-major: [r_bin * n_z + z_bin])
/// @param r       Radial coordinate
/// @param z_abs   Absolute value of vertical offset |z|
/// @param params  RenderParams containing grid metadata
__device__ inline double vol_interp_2d(const double* lut, double r, double z_abs,
                                        const RenderParams& params) {
    // Radial index
    double r_frac = (r - params.vol_r_min) / (params.vol_r_max - params.vol_r_min)
                    * (params.vol_n_r - 1);
    if (r_frac < 0.0) r_frac = 0.0;
    if (r_frac > (double)(params.vol_n_r - 1)) r_frac = (double)(params.vol_n_r - 1);
    int ri = (int)r_frac;
    if (ri > params.vol_n_r - 2) ri = params.vol_n_r - 2;
    const double rt = r_frac - ri;

    // Vertical index: z_abs / (3*H(r)) maps to [0, n_z-1]
    const double H = vol_scale_height(r, params);
    const double z_max = 3.0 * H;
    if (z_abs >= z_max) return 0.0;

    double z_frac = z_abs / z_max * (params.vol_n_z - 1);
    if (z_frac < 0.0) z_frac = 0.0;
    if (z_frac > (double)(params.vol_n_z - 1)) z_frac = (double)(params.vol_n_z - 1);
    int zi = (int)z_frac;
    if (zi > params.vol_n_z - 2) zi = params.vol_n_z - 2;
    const double zt = z_frac - zi;

    // Bilinear interpolation
    const int n_z = params.vol_n_z;
    const double v00 = lut[ri * n_z + zi];
    const double v01 = lut[ri * n_z + zi + 1];
    const double v10 = lut[(ri + 1) * n_z + zi];
    const double v11 = lut[(ri + 1) * n_z + zi + 1];
    return (v00 * (1.0 - rt) + v10 * rt) * (1.0 - zt)
         + (v01 * (1.0 - rt) + v11 * rt) * zt;
}

// ---------------------------------------------------------------------------
// Opacity LUT lookups
// ---------------------------------------------------------------------------

/// @brief Trilinear interpolation in log space on the 3D absorption opacity LUT.
///
/// Mirrors CPU OpacityLUTs::lookup_kappa_abs(). Interpolates in log(kappa) space
/// for smoother behavior across decades of opacity variation.
///
/// @param nu       Frequency [Hz]
/// @param rho_cgs  Density [g/cm^3]
/// @param T        Temperature [K]
/// @param params   RenderParams containing LUT grid metadata
__device__ inline double vol_lookup_kappa_abs(double nu, double rho_cgs, double T,
                                               const RenderParams& params) {
    int inu, irho, iT;
    double fnu, frho, fT;
    vol_log_interp(log10(nu), params.opacity_log_nu_min, params.opacity_log_nu_max,
                   params.opacity_n_nu, inu, fnu);
    vol_log_interp(log10(rho_cgs), params.opacity_log_rho_min, params.opacity_log_rho_max,
                   params.opacity_n_rho, irho, frho);
    vol_log_interp(log10(T), params.opacity_log_T_min, params.opacity_log_T_max,
                   params.opacity_n_T, iT, fT);

    const int n_rho = params.opacity_n_rho;
    const int n_T = params.opacity_n_T;

    // 3D index: [nu_bin * n_rho * n_T + rho_bin * n_T + T_bin]
    const double c000 = d_opacity_kappa_abs_lut[inu * n_rho * n_T + irho * n_T + iT];
    const double c001 = d_opacity_kappa_abs_lut[inu * n_rho * n_T + irho * n_T + iT + 1];
    const double c010 = d_opacity_kappa_abs_lut[inu * n_rho * n_T + (irho + 1) * n_T + iT];
    const double c011 = d_opacity_kappa_abs_lut[inu * n_rho * n_T + (irho + 1) * n_T + iT + 1];
    const double c100 = d_opacity_kappa_abs_lut[(inu + 1) * n_rho * n_T + irho * n_T + iT];
    const double c101 = d_opacity_kappa_abs_lut[(inu + 1) * n_rho * n_T + irho * n_T + iT + 1];
    const double c110 = d_opacity_kappa_abs_lut[(inu + 1) * n_rho * n_T + (irho + 1) * n_T + iT];
    const double c111 = d_opacity_kappa_abs_lut[(inu + 1) * n_rho * n_T + (irho + 1) * n_T + iT + 1];

    // Interpolate in log(kappa) for smoother results
    const double floor_val = 1e-100;
    const double l00 = log(fmax(c000, floor_val)) * (1.0 - fT) + log(fmax(c001, floor_val)) * fT;
    const double l01 = log(fmax(c010, floor_val)) * (1.0 - fT) + log(fmax(c011, floor_val)) * fT;
    const double l10 = log(fmax(c100, floor_val)) * (1.0 - fT) + log(fmax(c101, floor_val)) * fT;
    const double l11 = log(fmax(c110, floor_val)) * (1.0 - fT) + log(fmax(c111, floor_val)) * fT;
    const double l0 = l00 * (1.0 - frho) + l01 * frho;
    const double l1 = l10 * (1.0 - frho) + l11 * frho;
    return exp(l0 * (1.0 - fnu) + l1 * fnu);
}

/// @brief Bilinear interpolation on the 2D electron scattering opacity LUT.
///
/// @param rho_cgs  Density [g/cm^3]
/// @param T        Temperature [K]
/// @param params   RenderParams containing LUT grid metadata
__device__ inline double vol_lookup_kappa_es(double rho_cgs, double T,
                                              const RenderParams& params) {
    int irho, iT;
    double frho, fT;
    vol_log_interp(log10(rho_cgs), params.opacity_log_rho_min, params.opacity_log_rho_max,
                   params.opacity_n_rho, irho, frho);
    vol_log_interp(log10(T), params.opacity_log_T_min, params.opacity_log_T_max,
                   params.opacity_n_T, iT, fT);
    const int n_T = params.opacity_n_T;
    const double v00 = d_opacity_kappa_es_lut[irho * n_T + iT];
    const double v01 = d_opacity_kappa_es_lut[irho * n_T + iT + 1];
    const double v10 = d_opacity_kappa_es_lut[(irho + 1) * n_T + iT];
    const double v11 = d_opacity_kappa_es_lut[(irho + 1) * n_T + iT + 1];
    return v00 * (1.0 - frho) * (1.0 - fT) + v01 * (1.0 - frho) * fT
         + v10 * frho * (1.0 - fT) + v11 * frho * fT;
}

/// @brief Bilinear interpolation on the 2D Rosseland mean opacity LUT.
///
/// @param rho_cgs  Density [g/cm^3]
/// @param T        Temperature [K]
/// @param params   RenderParams containing LUT grid metadata
__device__ inline double vol_lookup_kappa_ross(double rho_cgs, double T,
                                                const RenderParams& params) {
    int irho, iT;
    double frho, fT;
    vol_log_interp(log10(rho_cgs), params.opacity_log_rho_min, params.opacity_log_rho_max,
                   params.opacity_n_rho, irho, frho);
    vol_log_interp(log10(T), params.opacity_log_T_min, params.opacity_log_T_max,
                   params.opacity_n_T, iT, fT);
    const int n_T = params.opacity_n_T;
    const double v00 = d_opacity_kappa_ross_lut[irho * n_T + iT];
    const double v01 = d_opacity_kappa_ross_lut[irho * n_T + iT + 1];
    const double v10 = d_opacity_kappa_ross_lut[(irho + 1) * n_T + iT];
    const double v11 = d_opacity_kappa_ross_lut[(irho + 1) * n_T + iT + 1];
    return v00 * (1.0 - frho) * (1.0 - fT) + v01 * (1.0 - frho) * fT
         + v10 * frho * (1.0 - fT) + v11 * frho * fT;
}

/// @brief Bilinear interpolation on the 2D mean molecular weight LUT.
///
/// @param rho_cgs  Density [g/cm^3]
/// @param T        Temperature [K]
/// @param params   RenderParams containing LUT grid metadata
__device__ inline double vol_lookup_mu(double rho_cgs, double T,
                                        const RenderParams& params) {
    int irho, iT;
    double frho, fT;
    vol_log_interp(log10(rho_cgs), params.opacity_log_rho_min, params.opacity_log_rho_max,
                   params.opacity_n_rho, irho, frho);
    vol_log_interp(log10(T), params.opacity_log_T_min, params.opacity_log_T_max,
                   params.opacity_n_T, iT, fT);
    const int n_T = params.opacity_n_T;
    const double v00 = d_opacity_mu_lut[irho * n_T + iT];
    const double v01 = d_opacity_mu_lut[irho * n_T + iT + 1];
    const double v10 = d_opacity_mu_lut[(irho + 1) * n_T + iT];
    const double v11 = d_opacity_mu_lut[(irho + 1) * n_T + iT + 1];
    return v00 * (1.0 - frho) * (1.0 - fT) + v01 * (1.0 - frho) * fT
         + v10 * frho * (1.0 - fT) + v11 * frho * fT;
}

// ---------------------------------------------------------------------------
// Disk geometry and thermodynamics
// ---------------------------------------------------------------------------

/// @brief Scale height H(r) from the precomputed radial LUT.
__device__ inline double vol_scale_height(double r, const RenderParams& params) {
    return vol_interp_radial(d_vol_H_lut, r, params);
}

/// @brief ISCO taper factor: 1 for r >= r_isco, Gaussian decay inside.
///
/// Smoothly reduces density inside the ISCO to model the plunging region
/// where gas falls rapidly with diminishing emission.
__device__ inline double vol_taper(double r, const RenderParams& params) {
    if (r >= params.disk_r_isco) return 1.0;
    const double dr = params.disk_r_isco - r;
    const double w = params.disk_taper_width;
    return exp(-(dr * dr) / (w * w));
}

/// @brief Test whether (r, z) is within the disk volume.
///
/// The disk extends from r_horizon to r_outer radially, and |z| < 3*H(r) vertically.
__device__ inline bool vol_inside(double r, double z, const RenderParams& params) {
    if (r <= params.disk_r_horizon || r > params.vol_r_max) return false;
    const double H = vol_scale_height(r, params);
    return fabs(z) < 3.0 * H;
}

/// @brief Density in CGS [g/cm^3] at (r, z, phi).
///
/// Combines midplane density, vertical profile, normalization scale,
/// ISCO taper, and turbulent noise modulation.
__device__ inline double vol_density_cgs(double r, double z, double phi,
                                          const RenderParams& params) {
    if (!vol_inside(r, z, params)) return 0.0;
    const double z_abs = fabs(z);
    const double rho_mid = vol_interp_radial(d_vol_rho_mid_lut, r, params);
    const double rho_norm = vol_interp_2d(d_vol_rho_profile_lut, r, z_abs, params);
    const double H = vol_scale_height(r, params);
    const double base = rho_mid * rho_norm * params.disk_rho_scale * vol_taper(r, params);

    // Turbulent noise perturbation (Cartesian coords scaled to fixed reference)
    const double ns = params.disk_noise_scale;
    const double nx = r * cos(phi) / ns;
    const double ny = r * sin(phi) / ns;
    const double nz = z / ns;
    const double n = cuda_simplex_noise_turbulent(nx, ny, nz);
    return base * (1.0 + params.disk_turbulence * n);
}

/// @brief Temperature at (r, z) from precomputed vertical profile LUT [K].
__device__ inline double vol_temperature(double r, double z,
                                          const RenderParams& params) {
    if (!vol_inside(r, z, params)) return 0.0;
    return vol_interp_2d(d_vol_T_profile_lut, r, fabs(z), params);
}

// ---------------------------------------------------------------------------
// Kerr orbital mechanics
// ---------------------------------------------------------------------------

/// @brief Kerr prograde orbital frequency: Omega = sqrt(M) / (r^{3/2} + a*sqrt(M)).
__device__ inline double vol_omega_orb(double r, double M, double a) {
    return sqrt(M) / (r * sqrt(r) + a * sqrt(M));
}

/// @brief Circular orbit 4-velocity at r >= r_isco.
///
/// Computes u^t and u^phi from the equatorial Kerr metric.
/// u^r = u^theta = 0 for circular orbits.
__device__ inline void vol_circular_velocity(double r, double M, double a,
                                              double& ut, double& uphi) {
    const double Omg = vol_omega_orb(r, M, a);

    // Equatorial Kerr metric components
    const double g_tt   = -(1.0 - 2.0 * M / r);
    const double g_tphi = -2.0 * M * a / r;
    const double g_phph = r * r + a * a + 2.0 * M * a * a / r;

    const double denom = -(g_tt + 2.0 * g_tphi * Omg + g_phph * Omg * Omg);
    ut = 1.0 / sqrt(fmax(denom, 1e-30));
    uphi = Omg * ut;
}

/// @brief Plunging geodesic 4-velocity at r < r_isco.
///
/// Uses the BPT72 conserved quantities (E_isco, L_isco) to compute
/// the 4-velocity of gas on a free-fall geodesic inside the ISCO.
/// u^r is negative (infall).
__device__ inline void vol_plunging_velocity(double r, double theta,
                                              double M, double a,
                                              double E_isco, double L_isco,
                                              double& ut, double& ur, double& uphi) {
    const double E = E_isco;
    const double L = L_isco;

    const double Delta = r * r - 2.0 * M * r + a * a;
    const double cos_th = cos(theta);
    const double Sigma = r * r + a * a * cos_th * cos_th;

    // Contravariant velocity from inverse metric + conserved quantities
    ut = (E * (r * r + a * a + 2.0 * M * a * a / r) - 2.0 * M * a * L / r) / Delta;
    uphi = (L * (1.0 - 2.0 * M / r) + 2.0 * M * a * E / r) / Delta;

    // Radial potential R(r) from Kerr geodesic equation
    const double Ea_L = E * (r * r + a * a) - a * L;
    const double L_aE = L - a * E;
    const double R = Ea_L * Ea_L - Delta * (r * r + L_aE * L_aE);

    // u^r is negative (infall) and uses full Sigma
    ur = -sqrt(fmax(0.0, R)) / Sigma;
}

// ---------------------------------------------------------------------------
// Planck function
// ---------------------------------------------------------------------------

/// @brief Planck function B_nu in frequency form [erg/(s cm^2 Hz sr)].
///
/// B_nu(nu, T) = (2h*nu^3 / c^2) / (exp(h*nu/(k_B*T)) - 1)
///
/// Returns 0 for unphysical inputs or if h*nu/(k_B*T) > 500 (negligible).
__device__ inline double vol_planck_nu(double nu, double T) {
    if (nu <= 0.0 || T <= 0.0) return 0.0;
    const double x = VOL_h_planck * nu / (VOL_k_B * T);
    if (x > 500.0) return 0.0;
    return (2.0 * VOL_h_planck * nu * nu * nu / (VOL_c_cgs * VOL_c_cgs))
           / (exp(x) - 1.0);
}

// ---------------------------------------------------------------------------
// Volumetric raymarching
// ---------------------------------------------------------------------------

/// @brief Raymarch through the volumetric accretion disk along a photon geodesic.
///
/// Performs per-channel (RGB at 450/550/650 nm) covariant radiative transfer
/// using the Lorentz-invariant formulation: I_nu / nu^3 = const along a ray.
/// At each step, computes local density, temperature, opacity, and emission,
/// then solves the transfer equation incrementally.
///
/// The geodesic is integrated using adaptive RK4 while inside the disk volume.
/// Step size is controlled by both optical depth and geometric considerations.
///
/// @param state   Current geodesic state (modified in place to disk exit point)
/// @param color   Accumulated RGB color (modified in place)
/// @param params  RenderParams with all disk and integrator parameters
__device__ inline void vol_raymarch(GeodesicState& state, Vec3& color,
                                     const RenderParams& params) {
    // RGB observation frequencies at 450nm, 550nm, 650nm
    const double nu_obs[3] = {
        VOL_c_cgs / 450e-7,
        VOL_c_cgs / 550e-7,
        VOL_c_cgs / 650e-7
    };

    // Lorentz-invariant specific intensity J per channel
    double J[3] = {0.0, 0.0, 0.0};

    // Observer p.u for static observer at observer_r
    const double ut_obs = 1.0 / sqrt(1.0 - 2.0 * params.mass / params.observer_r);

    const double M = params.mass;
    const double a = params.spin;

    double r = state.position[1];
    double ds = vol_scale_height(r, params) / 16.0;  // finer initial step
    int step_count = 0;
    constexpr int MAX_STEPS = 4096;
    constexpr double DTAU_TARGET = 0.05;  // target optical depth per step
    double tau_acc[3] = {0.0, 0.0, 0.0};

    while (step_count < MAX_STEPS) {
        // Take one RK4 step
        GeodesicState new_state = rk4_step(params.metric_type, M, a, state, ds);
        step_count++;

        r = new_state.position[1];
        const double theta = new_state.position[2];
        const double phi = new_state.position[3];
        const double z = r * cos(theta);

        // Hard exits
        if (r < params.disk_r_horizon) break;
        if (r > params.disk_r_outer) break;  // left the disk radially
        if (tau_acc[0] > 10.0 && tau_acc[1] > 10.0 && tau_acc[2] > 10.0) break;

        // Outside vertical extent: only exit if well beyond the disk (|z|>6H)
        const double H = vol_scale_height(r, params);
        if (!vol_inside(r, z, params)) {
            if (fabs(z) > 6.0 * H) break;  // truly gone
            ds = fmax(H / 4.0, H / 64.0);
            if (ds > H) ds = H;
            state = new_state;
            continue;
        }

        // Local thermodynamic state
        const double rho_cgs = vol_density_cgs(r, z, phi, params);
        const double T = vol_temperature(r, fabs(z), params);
        if (rho_cgs <= 0.0 || T <= 0.0) {
            state = new_state;
            continue;
        }

        // Compute redshift g = (p.u)_emit / (p.u)_obs
        double ut_emit = 0.0, ur_emit = 0.0, uphi_emit = 0.0;
        if (r >= params.disk_r_isco) {
            vol_circular_velocity(r, M, a, ut_emit, uphi_emit);
        } else {
            vol_plunging_velocity(r, theta, M, a,
                                   params.disk_E_isco, params.disk_L_isco,
                                   ut_emit, ur_emit, uphi_emit);
        }

        // p.u (covariant momentum . contravariant velocity)
        const double p_dot_u_emit = new_state.momentum[0] * ut_emit
                                  + new_state.momentum[1] * ur_emit
                                  + new_state.momentum[3] * uphi_emit;
        const double p_dot_u_obs = new_state.momentum[0] * ut_obs;
        const double g = p_dot_u_emit / p_dot_u_obs;

        // Proper distance along ray
        const double ds_proper = fabs(p_dot_u_emit) * fabs(ds);

        // Per-channel radiative transfer
        for (int ch = 0; ch < 3; ch++) {
            const double nu_emit = fabs(g) * nu_obs[ch];

            const double kabs = vol_lookup_kappa_abs(nu_emit, rho_cgs, T, params);
            const double kes = vol_lookup_kappa_es(rho_cgs, T, params);
            const double ktot = kabs + kes;
            const double epsilon = (ktot > 0.0) ? kabs / ktot : 1.0;

            const double dtau = ktot * rho_cgs * ds_proper;
            tau_acc[ch] += dtau;

            // Invariant source: S = epsilon * B_nu(nu_emit, T) / nu_emit^3
            const double Bnu = vol_planck_nu(nu_emit, T);
            const double S = epsilon * Bnu / (nu_emit * nu_emit * nu_emit);

            const double exp_dtau = exp(-dtau);
            J[ch] = J[ch] * exp_dtau + S * (1.0 - exp_dtau);
        }

        // Smooth adaptive step control: adjust ds so dtau ≈ DTAU_TARGET
        const double alpha_tot = (vol_lookup_kappa_abs(fabs(g) * nu_obs[1], rho_cgs, T, params)
                                + vol_lookup_kappa_es(rho_cgs, T, params)) * rho_cgs;
        double ds_tau = (alpha_tot > 0.0)
                      ? DTAU_TARGET / alpha_tot
                      : ds * 2.0;

        const double ds_geo = 0.1 * fmax(r - params.disk_r_horizon, 0.5);
        ds = fmin(ds_tau, ds_geo);
        // H already declared above in the outside-volume check
        if (ds < H / 64.0) ds = H / 64.0;
        if (ds > H) ds = H;

        state = new_state;
    }

    // Recover observed intensity: I_obs = J * nu_obs^3
    for (int ch = 0; ch < 3; ch++) {
        color[ch] += J[ch] * nu_obs[ch] * nu_obs[ch] * nu_obs[ch];
    }
}

} // namespace cuda

#endif // GRRT_CUDA_VOLUMETRIC_DISK_H
