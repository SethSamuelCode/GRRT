#ifndef GRRT_CUDA_SCENE_H
#define GRRT_CUDA_SCENE_H

/// @file cuda_scene.h
/// @brief CUDA device functions for accretion disk and celestial sphere.
///
/// Mirrors the CPU grrt::AccretionDisk and grrt::CelestialSphere implementations,
/// adapted for GPU execution. Uses enum+switch dispatch and plain device functions
/// (no virtual functions). All physics formulas match the CPU reference exactly.
///
/// Accretion disk: Novikov-Thorne flux profile, Kerr-aware orbital mechanics,
/// gravitational + Doppler redshift, blackbody emission via spectrum LUT.
///
/// Celestial sphere: star catalog stored in constant memory, angular proximity
/// search for escaped rays.
///
/// Units: geometrized (G = c = 1). Coordinates: Boyer-Lindquist (t, r, θ, φ).

#include "cuda_math.h"
#include "cuda_metric.h"
#include "cuda_color.h"
#include "cuda_types.h"

namespace cuda {

// ---------------------------------------------------------------------------
// Device memory declarations — defined in cuda_render.cu (or test file)
// ---------------------------------------------------------------------------

/// @brief Precomputed Novikov-Thorne radiative flux LUT, indexed [0, disk_flux_lut_size).
///
/// Normalized so flux_lut[i] = F(r_i) in geometrized units.
/// Built on host, uploaded once before rendering.
/// Stored in constant memory (fits within 64 KB alongside color LUTs).
extern __constant__ double d_flux_lut[MAX_FLUX_LUT_ENTRIES];

/// @brief Star positions and brightnesses for the celestial sphere.
///
/// Stored in global device memory (120 KB for 5000 stars × 3 doubles,
/// which exceeds the 64 KB constant memory limit).
/// Matches the CPU CelestialSphere star catalog (same random seed, same layout).
extern __device__ Star d_stars[MAX_STARS];

// ---------------------------------------------------------------------------
// Kerr-aware orbital mechanics
// ---------------------------------------------------------------------------

/// @brief Newtonian Keplerian angular velocity sqrt(M / r^3).
///
/// Used as the seed for the Kerr-corrected Omega below.
/// Schwarzschild limit: Omega = omega_kepler exactly.
///
/// @param r  Radial coordinate (Boyer-Lindquist)
/// @param M  Mass
/// @param a  Spin parameter (unused here; kept for interface consistency)
__device__ inline double omega_kepler(double r, double M, double /*a*/) {
    return sqrt(M / (r * r * r));
}

/// @brief Kerr circular orbit angular velocity Ω = ω_K / (1 + a ω_K).
///
/// Reduces to ω_K for a=0 (Schwarzschild).
/// Reference: Bardeen, Press & Teukolsky (1972).
///
/// @param r  Radial coordinate
/// @param M  Mass
/// @param a  Spin parameter
__device__ inline double Omega_kepler(double r, double M, double a) {
    double w = omega_kepler(r, M, a);
    return w / (1.0 + a * w);
}

/// @brief Specific energy of a circular orbit at radius r in Kerr spacetime.
///
/// E = (1 - 2M/r + a*omega_K) / sqrt(1 - 3M/r + 2*a*omega_K)
///
/// Reference: accretion_disk.cpp E_circ().
///
/// @param r  Radial coordinate (must be >= ISCO)
/// @param M  Mass
/// @param a  Spin parameter
__device__ inline double E_circ(double r, double M, double a) {
    double w  = omega_kepler(r, M, a);
    double aw = a * w;
    return (1.0 - 2.0 * M / r + aw) / sqrt(1.0 - 3.0 * M / r + 2.0 * aw);
}

/// @brief Specific angular momentum of a circular orbit at radius r in Kerr spacetime.
///
/// L = sqrt(M*r) * (1 - 2*a*omega_K + a²/r²) / sqrt(1 - 3M/r + 2*a*omega_K)
///
/// Reference: accretion_disk.cpp L_circ().
///
/// @param r  Radial coordinate (must be >= ISCO)
/// @param M  Mass
/// @param a  Spin parameter
__device__ inline double L_circ(double r, double M, double a) {
    double w  = omega_kepler(r, M, a);
    double aw = a * w;
    return sqrt(M * r) * (1.0 - 2.0 * aw + a * a / (r * r))
           / sqrt(1.0 - 3.0 * M / r + 2.0 * aw);
}

// ---------------------------------------------------------------------------
// Flux lookup
// ---------------------------------------------------------------------------

/// @brief Linear interpolation into the precomputed flux LUT.
///
/// Returns normalized flux F(r) / F_max ∈ [0, 1].
/// Returns 0 if r is outside [disk_r_inner, disk_r_outer].
///
/// @param r       Radial coordinate of disk crossing
/// @param params  RenderParams containing LUT metadata and disk bounds
__device__ inline double flux_lookup(double r, const RenderParams& params) {
    if (r <= params.disk_r_inner || r >= params.disk_r_outer) return 0.0;
    if (params.disk_flux_max <= 0.0) return 0.0;

    double frac = (r - params.disk_flux_r_min)
                  / (params.disk_flux_r_max - params.disk_flux_r_min)
                  * (params.disk_flux_lut_size - 1);
    int idx = (int)frac;
    double t = frac - idx;

    if (idx >= params.disk_flux_lut_size - 1) {
        return d_flux_lut[params.disk_flux_lut_size - 1] / params.disk_flux_max;
    }
    double F = d_flux_lut[idx] * (1.0 - t) + d_flux_lut[idx + 1] * t;
    return F / params.disk_flux_max;
}

// ---------------------------------------------------------------------------
// Disk temperature
// ---------------------------------------------------------------------------

/// @brief Effective blackbody temperature at disk radius r.
///
/// T(r) = T_peak * (F(r) / F_max)^{1/4}
///
/// Matches accretion_disk.cpp temperature().
///
/// @param r       Radial coordinate of disk crossing
/// @param params  RenderParams containing peak temperature and flux LUT
__device__ inline double disk_temperature(double r, const RenderParams& params) {
    double F_norm = flux_lookup(r, params);   // already F/F_max
    if (F_norm <= 0.0) return 0.0;
    return params.disk_peak_temperature * pow(F_norm, 0.25);
}

// ---------------------------------------------------------------------------
// Disk redshift
// ---------------------------------------------------------------------------

/// @brief Gravitational + Doppler redshift factor g = ν_obs / ν_emit.
///
/// Matches accretion_disk.cpp redshift() exactly.
///
/// Observer: static at r_obs in Boyer-Lindquist coordinates.
///   u^μ_obs = (u^t, 0, 0, 0)  with  u^t = 1/sqrt(-g_tt)
///   (p_μ u^μ)_obs = p_t * u^t = p_t / sqrt(-g_tt(r_obs))
///
/// Emitter: circular orbit at (r_cross, θ=π/2).
///   u^t_emit = 1 / sqrt(1 - 3M/r + 2*a*omega_K)
///   u^φ_emit = Ω * u^t_emit
///   (p_μ u^μ)_emit = p_t * u^t_emit + p_φ * u^φ_emit
///
/// g = (p_μ u^μ)_emit / (p_μ u^μ)_obs
///
/// @param r_cross    Radial coordinate where ray crossed equatorial plane
/// @param p_cross    Covariant 4-momentum at crossing point
/// @param observer_r Radial coordinate of observer
/// @param params     RenderParams (mass, spin)
__device__ inline double disk_redshift(double r_cross, const Vec4& p_cross,
                                        double observer_r, const RenderParams& params) {
    double M = params.mass;
    double a = params.spin;

    // Observer: static (u^φ = 0)
    // u^t_obs = 1 / sqrt(-(g_tt)) = 1 / sqrt(1 - 2M/r_obs)
    double u_t_obs = 1.0 / sqrt(1.0 - 2.0 * M / observer_r);
    double pu_obs  = p_cross[0] * u_t_obs;

    // Emitter: prograde circular orbit
    double w          = omega_kepler(r_cross, M, a);
    double aw         = a * w;
    double u_t_emit   = 1.0 / sqrt(1.0 - 3.0 * M / r_cross + 2.0 * aw);
    double u_phi_emit = Omega_kepler(r_cross, M, a) * u_t_emit;
    double pu_emit    = p_cross[0] * u_t_emit + p_cross[3] * u_phi_emit;

    if (fabs(pu_obs) < 1e-30) return 1.0;
    return pu_emit / pu_obs;
}

// ---------------------------------------------------------------------------
// Disk emission
// ---------------------------------------------------------------------------

/// @brief Spectral emission from accretion disk at crossing point.
///
/// Matches accretion_disk.cpp emission() exactly:
///   1. Compute local temperature T(r)
///   2. Compute redshift g
///   3. Observed temperature T_obs = g * T
///   4. color = chromaticity(T_obs), luminosity from T_emit (not T_obs)
///   5. Return color * lum * g^4  (flux boosted by Doppler beaming)
///
/// Returns {0,0,0} if disk is disabled, T=0, or T_obs < 100 K.
///
/// @param r_cross    Radial coordinate of disk-ray intersection
/// @param p_cross    Covariant 4-momentum at crossing
/// @param observer_r Observer radial coordinate (for redshift calculation)
/// @param params     RenderParams
__device__ inline Vec3 disk_emission(double r_cross, const Vec4& p_cross,
                                      double observer_r, const RenderParams& params) {
    if (!params.disk_enabled) return {0.0, 0.0, 0.0};

    double T = disk_temperature(r_cross, params);
    if (T <= 0.0) return {0.0, 0.0, 0.0};

    double g     = disk_redshift(r_cross, p_cross, observer_r, params);
    double T_obs = g * T;
    if (T_obs < 100.0) return {0.0, 0.0, 0.0};

    Vec3   chroma = spectrum_chromaticity(T_obs,
                                          params.spectrum_t_min,
                                          params.spectrum_t_max,
                                          params.spectrum_num_entries);
    double lum    = spectrum_luminosity(T, params.spectrum_t_max);
    double g4     = g * g * g * g;

    return chroma * (lum * g4);
}

// ---------------------------------------------------------------------------
// Celestial sphere
// ---------------------------------------------------------------------------

/// @brief Sample the celestial sphere at the position of an escaped ray.
///
/// Normalizes (θ, φ) from the escaping geodesic position vector, then performs
/// a linear scan over d_stars to find the nearest star within angular_tolerance.
/// Returns that star's brightness as a white Vec3, or {0,0,0} for empty sky.
///
/// Angular distance (approximate great-circle):
///   dist² = Δθ² + (Δφ * sin θ)²
///
/// @param position  Boyer-Lindquist position of escaped ray (θ at index 2, φ at 3)
/// @param params    RenderParams (num_stars, star_angular_tolerance)
__device__ inline Vec3 celestial_sphere_sample(const Vec4& position,
                                                const RenderParams& params) {
    // Normalize θ to [0, π], reflecting if needed
    double theta = position[2];
    double phi   = position[3];

    theta = fmod(theta, 2.0 * M_PI);
    if (theta < 0.0) theta += 2.0 * M_PI;
    if (theta > M_PI) {
        theta = 2.0 * M_PI - theta;
        phi  += M_PI;   // reflect hemisphere
    }

    // Normalize φ to [-π, π]
    phi = fmod(phi, 2.0 * M_PI);
    if (phi >  M_PI) phi -= 2.0 * M_PI;
    if (phi < -M_PI) phi += 2.0 * M_PI;

    double tol2   = params.star_angular_tolerance * params.star_angular_tolerance;
    double sin_t  = sin(theta);
    int    n      = params.num_stars;
    if (n > MAX_STARS) n = MAX_STARS;

    for (int i = 0; i < n; ++i) {
        double dtheta = theta - d_stars[i].theta;
        double dphi   = phi   - d_stars[i].phi;

        // Wrap dphi to [-π, π]
        if (dphi >  M_PI) dphi -= 2.0 * M_PI;
        if (dphi < -M_PI) dphi += 2.0 * M_PI;

        double ang_dist2 = dtheta * dtheta + dphi * dphi * sin_t * sin_t;

        if (ang_dist2 < tol2) {
            double b = d_stars[i].brightness;
            return {b, b, b};
        }
    }

    return {0.0, 0.0, 0.0};
}

} // namespace cuda

#endif // GRRT_CUDA_SCENE_H
