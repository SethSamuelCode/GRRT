#include "grrt/scene/volumetric_disk.h"
#include "grrt/math/constants.h"
#include <cmath>
#include <algorithm>
#include <iterator>
#include <numbers>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <limits>

namespace grrt {

// ============================================================================
// Kerr metric helpers (Bardeen-Press-Teukolsky 1972)
// ============================================================================

/// Compute prograde ISCO radius for Kerr metric.
/// Uses the Bardeen formula: r_isco = M * (3 + Z2 - sqrt((3-Z1)(3+Z1+2*Z2)))
static double compute_isco(double M, double a) {
    const double a_star = a / M;
    const double Z1 = 1.0 + std::cbrt(1.0 - a_star * a_star)
                           * (std::cbrt(1.0 + a_star) + std::cbrt(1.0 - a_star));
    const double Z2 = std::sqrt(3.0 * a_star * a_star + Z1 * Z1);
    return M * (3.0 + Z2 - std::sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)));
}

/// Compute outer horizon radius r+ = M + sqrt(M^2 - a^2).
static double compute_horizon(double M, double a) {
    return M + std::sqrt(M * M - a * a);
}

/// Cubic Hermite smoothstep, C¹-continuous interpolation from 0 (at edge0) to 1 (at edge1).
/// Used for the outer-radial taper and elsewhere we need a smooth 0→1 transition.
static double smoothstep(double edge0, double edge1, double x) {
    if (edge1 == edge0) return x < edge0 ? 0.0 : 1.0;
    const double t = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// ============================================================================
// Constructor
// ============================================================================

VolumetricDisk::VolumetricDisk(double mass, double spin, double r_outer,
                               double peak_temperature, const VolumetricParams& params)
    : mass_(mass), spin_(spin), r_outer_(r_outer),
      peak_temperature_(peak_temperature),
      params_(params),
      noise_(params.seed)
{
    r_isco_ = compute_isco(mass_, spin_);
    r_horizon_ = compute_horizon(mass_, spin_);
    r_min_ = r_horizon_ + 0.01 * mass_;

    {
        const double v = std::sqrt(mass_ / r_isco_);
        const double a_star = spin_ / mass_;
        const double v3 = v * v * v;
        const double denom = std::sqrt(1.0 - 3.0 * v * v + 2.0 * a_star * v3);
        E_isco_ = (1.0 - 2.0 * v * v + a_star * v3) / denom;
        L_isco_ = std::sqrt(mass_ * r_isco_)
                  * (1.0 - 2.0 * a_star * v3 + a_star * a_star * v * v * v * v)
                  / denom;
    }

    // Effective physical mass [M_sun]. The VolumetricParams default (10.0) is the
    // single source of truth for "unset"; a non-physical mass_solar <= 0 would
    // zero the length scale and blow up later unit conversions (Ω_cgs = Ω·c/r_g),
    // so warn loudly and recover rather than substitute silently.
    double mass_solar_eff = params_.mass_solar;
    if (mass_solar_eff <= 0.0) {
        emit(WarningSeverity::Promptable, "mass_solar_invalid",
             "mass_solar <= 0 is non-physical; using 10 M_sun for the length scale");
        mass_solar_eff = 10.0;
    }

    // Physical length scale (Approach A): r_g = G·M_phys/c^2 [cm].
    // Geometric lengths (H, z) become cm via × r_g — this is what makes the
    // vertical optical-depth integral dimensionally honest.
    {
        using namespace constants;
        r_g_ = G_cgs * (mass_solar_eff * M_sun) / (c_cgs * c_cgs);
    }

    // Physical accretion rate (Approach A): Mdot = f_Edd·L_Edd/(η c²) [g/s],
    // or a direct override. η = 1 − E_isco (radiative efficiency); L_Edd is the
    // Eddington luminosity 4πG M m_p c/σ_T (the c is in the numerator).
    // Diagnostic now; anchors Σ in the column BVP later. Reuses mass_solar_eff.
    {
        using namespace constants;
        const double eta = 1.0 - E_isco_;
        const double L_Edd = 4.0 * std::numbers::pi * G_cgs * (mass_solar_eff * M_sun)
                           * m_p * c_cgs / sigma_T;
        if (params_.mdot_override > 0.0) {
            mdot_ = params_.mdot_override;
        } else if (eta > 0.0) {
            mdot_ = params_.eddington_fraction * L_Edd / (eta * c_cgs * c_cgs);
        } else {
            mdot_ = 0.0;  // eta <= 0 is unreachable for physical Kerr (E_isco < 1 always)
        }
    }

    std::printf("[VolumetricDisk] Building opacity LUTs...\n");
    opacity_luts_ = build_opacity_luts(1e-18, 1e-6, 3000.0, 1e8,
                                       params_.opacity_nu_min, params_.opacity_nu_max);

    // --- Refinement-driven LUT construction ---
    if (params_.bins_per_gradient > 0) {
        const double sizing_scale = std::max((r_isco_ - r_horizon_) / 3.0, 0.01);
        n_r_ = std::clamp(params_.bins_per_gradient *
                          static_cast<int>(std::ceil((r_outer_ - r_min_) / sizing_scale)),
                          params_.min_n_r, params_.max_n_r);
    } else {
        n_r_ = std::max(params_.min_n_r, 256);
    }
    if (params_.bins_per_h > 0) {
        n_z_ = std::clamp(params_.bins_per_h * 8, params_.min_n_z, params_.max_n_z);
    } else {
        n_z_ = std::max(params_.min_n_z, 64);
    }

    // Initial radial build (used by both manual and auto modes)
    H_lut_.assign(n_r_, 0.0);
    rho_mid_lut_.assign(n_r_, 0.0);
    T_eff_lut_.assign(n_r_, 0.0);
    compute_radial_structure();
    compute_plunging_region_decay();
    apply_outer_radial_taper();

    std::printf("[VolumetricDisk] Refining LUT sizing (n_r=%d, n_z=%d initial)...\n",
                n_r_, n_z_);

    auto [final_n_r, final_n_z] = nested_refine();
    n_r_ = final_n_r;
    n_z_ = final_n_z;

    std::printf("[VolumetricDisk] Refinement done: n_r=%d, n_z=%d\n", n_r_, n_z_);

    // Final vertical-profile build at the converged (n_r_, n_z_)
    compute_vertical_profiles();

    std::printf("[VolumetricDisk] Normalizing density...\n");
    normalize_density();

    compute_sigma_s_phys();
    validate_luts();

    std::printf("[VolumetricDisk] Construction complete. r_isco=%.4f r_horizon=%.4f\n",
                r_isco_, r_horizon_);
}

// ============================================================================
// Kerr orbital mechanics
// ============================================================================

double VolumetricDisk::omega_orb(double r) const {
    // Kerr prograde: Omega = sqrt(M) / (r^{3/2} + a*sqrt(M))
    return std::sqrt(mass_) / (r * std::sqrt(r) + spin_ * std::sqrt(mass_));
}

double VolumetricDisk::omega_z_sq(double r) const {
    // Kerr vertical epicyclic frequency squared
    const double Omg = omega_orb(r);
    const double sqM = std::sqrt(mass_);
    const double r3 = r * r * r;
    return Omg * Omg * (1.0 - 4.0 * spin_ * sqM / std::sqrt(r3)
                        + 3.0 * spin_ * spin_ / (r * r));
}

// ============================================================================
// 4-velocity: circular orbit (r >= r_isco)
// ============================================================================

void VolumetricDisk::circular_velocity(double r, double& ut, double& uphi) const {
    const double M = mass_;
    const double a = spin_;
    const double Omg = omega_orb(r);

    // Equatorial Kerr metric components
    const double g_tt   = -(1.0 - 2.0 * M / r);
    const double g_tphi = -2.0 * M * a / r;
    const double g_phph = r * r + a * a + 2.0 * M * a * a / r;

    const double denom = -(g_tt + 2.0 * g_tphi * Omg + g_phph * Omg * Omg);
    ut = 1.0 / std::sqrt(std::max(denom, 1e-30));
    uphi = Omg * ut;
}

// ============================================================================
// 4-velocity: plunging geodesic (r < r_isco)
// ============================================================================

void VolumetricDisk::plunging_velocity(double r, double theta,
                                       double& ut, double& ur, double& uphi) const {
    const double M = mass_;
    const double a = spin_;
    const double E = E_isco_;
    const double L = L_isco_;

    const double Delta = r * r - 2.0 * M * r + a * a;
    const double Sigma = r * r + a * a * std::cos(theta) * std::cos(theta);

    // Equatorial inverse metric components (Section 4.3 of spec)
    ut = (E * (r * r + a * a + 2.0 * M * a * a / r) - 2.0 * M * a * L / r) / Delta;
    uphi = (L * (1.0 - 2.0 * M / r) + 2.0 * M * a * E / r) / Delta;

    // Radial potential R(r) from Kerr geodesic equation
    const double Ea_L = E * (r * r + a * a) - a * L;
    const double L_aE = L - a * E;
    const double R = Ea_L * Ea_L - Delta * (r * r + L_aE * L_aE);

    // u^r is negative (infall) and uses full Sigma
    ur = -std::sqrt(std::max(0.0, R)) / Sigma;
}

// ============================================================================
// Taper
// ============================================================================

double VolumetricDisk::taper(double r) const {
    if (r >= r_isco_) return 1.0;
    if (r <= r_horizon_) return 0.0;

    // Mass conservation along the BPT72 plunging geodesic:
    //   ρ(r) ∝ 1 / (r · |u^r(r)|)
    // Normalize so taper saturates to 1 at ISCO via a regulator at r_isco·EPS.
    constexpr double EPS = 0.99;
    constexpr double THETA = 1.5707963267948966;  // pi/2, equatorial plane

    double ut, ur_ref, uphi;
    plunging_velocity(r_isco_ * EPS, THETA, ut, ur_ref, uphi);
    const double r_ref = r_isco_ * EPS;
    const double denom_ref = r_ref * std::abs(ur_ref);
    if (denom_ref <= 0.0) return 1.0;

    double ur;
    plunging_velocity(r, THETA, ut, ur, uphi);
    const double denom = r * std::abs(ur);
    if (denom <= 0.0) return 1.0;

    return std::clamp(denom_ref / denom, 0.0, 1.0);
}

// ============================================================================
// Volume bounds
// ============================================================================

bool VolumetricDisk::inside_volume(double r, double z) const {
    if (r <= r_horizon_ || r > r_outer_ + 0.5 * outer_taper_width_) return false;
    const double zm = z_max_at(r);
    const double H  = scale_height(r);
    return std::abs(z) < zm + 0.5 * H;
}

// ============================================================================
// LUT interpolation helpers
// ============================================================================

double VolumetricDisk::interp_radial(const std::vector<double>& lut, double r) const {
    if (r <= r_min_) return lut.front();
    if (r >= r_outer_) return lut.back();
    const double frac = (r - r_min_) / (r_outer_ - r_min_) * (n_r_ - 1);
    const int idx = std::clamp(static_cast<int>(frac), 0, n_r_ - 2);
    const double t = frac - idx;
    return lut[idx] * (1.0 - t) + lut[idx + 1] * t;
}

double VolumetricDisk::interp_2d(const std::vector<double>& lut, double r, double z_abs) const {
    // Radial interpolation
    const double r_frac = std::clamp((r - r_min_) / (r_outer_ - r_min_) * (n_r_ - 1),
                                      0.0, static_cast<double>(n_r_ - 1));
    const int ri = std::min(static_cast<int>(r_frac), n_r_ - 2);
    const double tr = r_frac - ri;

    // Per-column z normalization: each column has its own z_max
    const double zm_lo = z_max_lut_[ri];
    const double zm_hi = z_max_lut_[std::min(ri + 1, n_r_ - 1)];

    // If z_abs is beyond both columns' extent, return 0
    if ((zm_lo <= 0.0 || z_abs >= zm_lo) && (zm_hi <= 0.0 || z_abs >= zm_hi))
        return 0.0;

    // Look up in column ri
    double val_lo = 0.0;
    if (zm_lo > 0.0 && z_abs < zm_lo) {
        const double z_frac_lo = std::clamp(z_abs / zm_lo * (n_z_ - 1), 0.0,
                                             static_cast<double>(n_z_ - 1));
        const int zi_lo = std::min(static_cast<int>(z_frac_lo), n_z_ - 2);
        const double tz_lo = z_frac_lo - zi_lo;
        val_lo = (1.0 - tz_lo) * lut[ri * n_z_ + zi_lo]
               + tz_lo * lut[ri * n_z_ + zi_lo + 1];
    }

    // Look up in column ri+1
    double val_hi = 0.0;
    const int ri1 = std::min(ri + 1, n_r_ - 1);
    if (zm_hi > 0.0 && z_abs < zm_hi) {
        const double z_frac_hi = std::clamp(z_abs / zm_hi * (n_z_ - 1), 0.0,
                                             static_cast<double>(n_z_ - 1));
        const int zi_hi = std::min(static_cast<int>(z_frac_hi), n_z_ - 2);
        const double tz_hi = z_frac_hi - zi_hi;
        val_hi = (1.0 - tz_hi) * lut[ri1 * n_z_ + zi_hi]
               + tz_hi * lut[ri1 * n_z_ + zi_hi + 1];
    }

    return (1.0 - tr) * val_lo + tr * val_hi;
}

// ============================================================================
// Public accessors
// ============================================================================

double VolumetricDisk::scale_height(double r) const {
    return interp_radial(H_lut_, r);
}

double VolumetricDisk::noise_correlation_length(double r) const {
    const double H = scale_height(r);
    const double c_corr = (params_.noise_correlation_length_factor > 0.0)
                        ? params_.noise_correlation_length_factor : 0.5;
    return (params_.noise_scale > 0.0) ? params_.noise_scale * H : c_corr * H;
}

double VolumetricDisk::z_max_at(double r) const {
    return interp_radial(z_max_lut_, r);
}

double VolumetricDisk::density(double r, double z, double phi) const {
    if (r <= r_horizon_ || r > r_outer_ + 0.5 * outer_taper_width_) return 0.0;
    const double z_abs = std::abs(z);
    const double zm = z_max_at(r);
    if (z_abs >= zm) return 0.0;

    const double rho_mid  = interp_radial(rho_mid_lut_, r);
    const double rho_norm = interp_2d(rho_profile_lut_, r, z_abs);
    const double base     = rho_mid * rho_norm * rho_scale_ * taper(r);

    // Single source of truth for the turbulence correlation length (also
    // exposed for the raymarch's fine-step sizing).
    const double L = noise_correlation_length(r);
    if (L <= 0.0) return base;

    const double nx = r * std::cos(phi) / L;
    const double ny = r * std::sin(phi) / L;
    const double nz = z / L;
    const double n  = noise_.evaluate_fbm(nx, ny, nz, params_.noise_octaves);

    double arg = sigma_s_phys_ * params_.turbulence * n;
    arg = std::clamp(arg, -50.0, 50.0);
    return base * std::exp(arg);
}

double VolumetricDisk::density_cgs(double r, double z, double phi) const {
    // The density is already in CGS after rho_scale normalization
    return density(r, z, phi);
}

double VolumetricDisk::temperature(double r, double z) const {
    if (r <= r_horizon_ || r > r_outer_) return 0.0;
    const double z_abs = std::abs(z);
    const double zm = z_max_at(r);
    if (z_abs >= zm) return 0.0;
    return interp_2d(T_profile_lut_, r, z_abs);
}

// ============================================================================
// build_flux_lut() — same Novikov-Thorne flux as AccretionDisk
// ============================================================================

void VolumetricDisk::build_flux_lut(std::vector<double>& flux, double& flux_max) const {
    const int N = n_r_;
    flux.resize(N);
    flux_max = 0.0;

    // Kerr circular orbit helpers (same formulas as accretion_disk.cpp)
    auto omega_kepler = [&](double r) { return std::sqrt(mass_ / (r * r * r)); };
    auto Omega = [&](double r) {
        const double w = omega_kepler(r);
        return w / (1.0 + spin_ * w);
    };
    auto E_circ = [&](double r) {
        const double w = omega_kepler(r);
        const double aw = spin_ * w;
        return (1.0 - 2.0 * mass_ / r + aw) / std::sqrt(1.0 - 3.0 * mass_ / r + 2.0 * aw);
    };
    auto L_circ = [&](double r) {
        const double w = omega_kepler(r);
        const double aw = spin_ * w;
        return std::sqrt(mass_ * r) * (1.0 - 2.0 * aw + spin_ * spin_ / (r * r))
               / std::sqrt(1.0 - 3.0 * mass_ / r + 2.0 * aw);
    };

    const double E_isco = E_circ(r_isco_);
    const double L_isco = L_circ(r_isco_);
    constexpr double fd_eps = 1e-6;

    double I_cumulative = 0.0;
    double prev_integrand = 0.0;

    for (int i = 0; i < N; ++i) {
        const double r = r_isco_ + (r_outer_ - r_isco_) * i / (N - 1);

        if (i == 0) {
            flux[i] = 0.0;
            continue;
        }

        const double E_prime = (E_circ(r + fd_eps) - E_circ(r - fd_eps)) / (2.0 * fd_eps);
        const double L_prime = (L_circ(r + fd_eps) - L_circ(r - fd_eps)) / (2.0 * fd_eps);
        const double integrand = (E_circ(r) - E_isco) * L_prime
                               - (L_circ(r) - L_isco) * E_prime;

        const double dr = (r_outer_ - r_isco_) / (N - 1);
        I_cumulative += 0.5 * (prev_integrand + integrand) * dr;
        prev_integrand = integrand;

        const double Om = Omega(r);
        const double E_r = E_circ(r);
        const double L_r = L_circ(r);
        const double dOmega_dr = (Omega(r + fd_eps) - Omega(r - fd_eps)) / (2.0 * fd_eps);

        const double denominator = E_r - Om * L_r;
        if (std::abs(denominator) < 1e-20) {
            flux[i] = 0.0;
            continue;
        }

        const double F = (3.0 * mass_ / (8.0 * std::numbers::pi * r * r * r))
                         * (1.0 / denominator) * (-dOmega_dr) * I_cumulative;

        flux[i] = std::max(F, 0.0);
        if (flux[i] > flux_max) flux_max = flux[i];
    }
}

// ============================================================================
// compute_radial_structure()
// ============================================================================

void VolumetricDisk::compute_radial_structure() {
    using namespace constants;

    // Build Novikov-Thorne flux LUT (over orbiting region r_isco..r_outer)
    std::vector<double> flux_orb;
    double flux_max_val = 0.0;
    build_flux_lut(flux_orb, flux_max_val);

    // Helper: interpolate flux for r >= r_isco
    auto flux_at = [&](double r) -> double {
        if (r <= r_isco_ || r >= r_outer_ || flux_max_val <= 0.0) return 0.0;
        const double frac = (r - r_isco_) / (r_outer_ - r_isco_) * (static_cast<int>(flux_orb.size()) - 1);
        const int idx = static_cast<int>(frac);
        const double t = frac - idx;
        if (idx >= static_cast<int>(flux_orb.size()) - 1) return flux_orb.back();
        return flux_orb[idx] * (1.0 - t) + flux_orb[idx + 1] * t;
    };

    // Allocate radial LUTs
    H_lut_.resize(n_r_);
    rho_mid_lut_.resize(n_r_);
    T_eff_lut_.resize(n_r_);

    // Find T_plunge: T_eff at first bin just outside ISCO
    const double T_plunge = (flux_max_val > 0.0)
        ? peak_temperature_ * std::pow(flux_at(r_isco_ + 0.01) / flux_max_val, 0.25)
        : 0.0;

    // H(r_isco) will be computed and used for r < r_isco
    double H_isco = 0.0;

    // Index of ISCO in radial grid (for freezing H inside ISCO)
    int isco_idx = -1;

    for (int i = 0; i < n_r_; ++i) {
        const double r = r_min_ + (r_outer_ - r_min_) * i / (n_r_ - 1);

        // --- T_eff(r) ---
        double T_eff = 0.0;
        if (r >= r_isco_) {
            const double F = flux_at(r);
            if (F > 0.0 && flux_max_val > 0.0) {
                T_eff = peak_temperature_ * std::pow(F / flux_max_val, 0.25);
            }
        } else {
            T_eff = T_plunge;
        }
        T_eff_lut_[i] = T_eff;

        // Track the first bin at or past ISCO
        if (r >= r_isco_ && isco_idx < 0) isco_idx = i;

        // --- Scale height H(r) and rho_mid(r) via iterative solve ---
        // Uses gas + radiation pressure (Section 1.2 of spec)

        if (T_eff <= 0.0) {
            H_lut_[i] = (H_isco > 0.0) ? H_isco : 0.01 * mass_;
            rho_mid_lut_[i] = 0.0;
            continue;
        }

        // Omega_z^2 for this radius
        double Omz2 = omega_z_sq(r);

        // For r < r_isco, Omega_z^2 can go to zero/negative.
        // Placeholder; real values set by compute_plunging_region_decay()
        if (r < r_isco_ || Omz2 <= 0.0) {
            H_lut_[i]       = 0.01 * mass_;
            rho_mid_lut_[i] = 0.0;
            continue;
        }

        const double Omz = std::sqrt(Omz2);

        // Scale height from gas pressure only (Section 1.2 of spec).
        // Radiation pressure refinement is deferred: we cannot compute
        // the rho_CGS term (4*sigma*T^4 / (3*rho*c^3)) without knowing
        // rho_scale, which comes from normalize_density() later.
        // The vertical profile solver captures the full pressure anyway.

        // Look up mu from opacity LUT at a reasonable reference density
        // (mu depends weakly on density at high T where gas is fully ionized)
        const double rho_ref_cgs = 1e-10; // typical midplane for ~10 Msun BH
        double mu = opacity_luts_.lookup_mu(
            std::clamp(rho_ref_cgs, 1e-18, 1e-6),
            std::clamp(T_eff, 3000.0, 1e8));
        if (mu <= 0.0 || !std::isfinite(mu)) mu = 0.6;

        // Gas-pressure sound speed in geometric units (v/c)
        const double c_gas2 = k_B * T_eff / (mu * m_p * c_cgs * c_cgs);
        double H = std::sqrt(c_gas2) / Omz;

        // Compute midplane density (proportional, will be normalized later)
        // Exact Kerr shear: dOmega/dr = -(3/2)*sqrt(M)*r^{1/2} / (r^{3/2}+a*sqrt(M))^2
        const double sqM = std::sqrt(mass_);
        const double denom_shear = r * std::sqrt(r) + spin_ * sqM;
        const double dOmega_dr = -1.5 * sqM * std::sqrt(r) / (denom_shear * denom_shear);
        const double shear_sq = r * r * dOmega_dr * dOmega_dr;

        const double c_s_cgs = std::sqrt(c_gas2) * c_cgs;
        const double nu_visc = params_.alpha * c_s_cgs * H;

        const double F = (r >= r_isco_) ? flux_at(r) : flux_at(r_isco_ + 0.01);
        double rho_mid = 1.0;
        if (shear_sq > 0.0 && nu_visc > 0.0 && H > 0.0 && F > 0.0) {
            const double Sigma_prop = F / (nu_visc * shear_sq);
            rho_mid = Sigma_prop / (std::sqrt(2.0 * std::numbers::pi) * H);
        }

        // Clamp H to reasonable range
        H = std::clamp(H, 0.001 * mass_, 5.0 * mass_);
        H_lut_[i] = H;
        rho_mid_lut_[i] = rho_mid;

        // Record H at ISCO for freezing inside ISCO
        if (r >= r_isco_ && (H_isco <= 0.0 || (isco_idx >= 0 && i == isco_idx))) {
            H_isco = H;
        }
    }

}

// ============================================================================
// compute_plunging_region_decay()
// ============================================================================

void VolumetricDisk::compute_plunging_region_decay() {
    int isco_idx = -1;
    for (int i = 0; i < n_r_; ++i) {
        const double r = r_min_ + (r_outer_ - r_min_) * i / (n_r_ - 1);
        if (r >= r_isco_) { isco_idx = i; break; }
    }
    if (isco_idx <= 0) return;

    const double H_isco       = H_lut_[isco_idx];
    const double rho_mid_isco = rho_mid_lut_[isco_idx];
    const double p            = params_.plunging_h_decay_exponent;

    for (int i = 0; i < isco_idx; ++i) {
        const double r = r_min_ + (r_outer_ - r_min_) * i / (n_r_ - 1);
        const double t = taper(r);
        H_lut_[i]       = H_isco * std::pow(std::max(t, 1e-30), p);
        rho_mid_lut_[i] = rho_mid_isco * t;
    }
}

// ============================================================================
// apply_outer_radial_taper()
// ============================================================================

void VolumetricDisk::apply_outer_radial_taper() {
    double width = (params_.outer_taper_width > 0.0)
                 ? params_.outer_taper_width
                 : 2.0 * H_lut_.back();

    const double max_allowed = (r_outer_ - r_min_) - 0.1 * r_outer_;
    if (width > max_allowed && max_allowed > 0.0) {
        emit(WarningSeverity::Warning, "outer_taper_clamped",
             "outer_taper_width clamped to fit disk extent");
        width = max_allowed;
    }
    outer_taper_width_ = width;

    if (width <= 0.0) return;
    const double r_taper_start = r_outer_ - width;

    for (int i = 0; i < n_r_; ++i) {
        const double r = r_min_ + (r_outer_ - r_min_) * i / (n_r_ - 1);
        if (r < r_taper_start) continue;
        const double factor = 1.0 - smoothstep(r_taper_start, r_outer_, r);
        rho_mid_lut_[i] *= factor;
    }
}

// ============================================================================
// Flux-limiter helpers (Levermore & Pomraning 1981)
// ============================================================================

/// Levermore-Pomraning flux limiter λ(R).
static double lp_lambda(double R) {
    return (2.0 + R) / (6.0 + 3.0 * R + R * R);
}

/// Eddington factor f(R) = λ + λ²R² (radiation pressure coefficient).
/// Limits: f → 1/3 (optically thick), f → 1 (optically thin).
static double lp_eddington_factor(double R) {
    const double lam = lp_lambda(R);
    return lam + lam * lam * R * R;
}

// ============================================================================
// solve_column()
// ============================================================================

VolumetricDisk::ColumnSolution VolumetricDisk::solve_column(
    double r, double H, double T_eff,
    double rho_mid_val, int n_z) const
{
    using namespace constants;

    constexpr double Z_MAX_CAP_FACTOR = 30.0;   // (was 20.0)
    constexpr double CONV_FLOOR       = 1e-15;  // (was 1e-10)
    constexpr double RHO_FLOOR        = 1e-18;  // (was 1e-15)
    constexpr int    MAX_OUTER_ITERS  = 8;

    // DEBUG (env+radius gated): log the vertical-ODE RHS terms for one column.
    const bool col_log = (std::getenv("GRRT_COL_LOG") != nullptr)
                       && std::abs(r - 8.26) < 0.01;

    ColumnSolution out;
    out.rho_z.assign(n_z, 1.0);
    out.T_z.assign(n_z, T_eff);

    if (H <= 0.0 || T_eff <= 0.0 || rho_mid_val <= 0.0) {
        out.z_max = 3.0 * H;
        out.rho_z[0] = 1.0;
        for (int zi = 1; zi < n_z; ++zi) out.rho_z[zi] = 0.0;
        return out;
    }

    double Omz2 = omega_z_sq(r);
    if (r < r_isco_ || Omz2 <= 0.0) {
        Omz2 = omega_z_sq(r_isco_);
        if (Omz2 <= 0.0) Omz2 = omega_orb(r_isco_) * omega_orb(r_isco_);
    }

    const double kR_ref = opacity_luts_.lookup_kappa_ross(
        1e-10, std::clamp(T_eff, 3000.0, 1e8));
    const double kE_ref = opacity_luts_.lookup_kappa_es(
        1e-10, std::clamp(T_eff, 3000.0, 1e8));
    const double kappa_ref_total = std::max(kR_ref + kE_ref, 1.0);
    const double rho_cgs_ref = std::clamp(
        params_.tau_mid / (kappa_ref_total * 3.0 * H), 1e-18, 1e-6);

    const double T_mid4 = 0.75 * T_eff * T_eff * T_eff * T_eff
                         * (params_.tau_mid + 2.0/3.0);
    const double T_mid = std::pow(T_mid4, 0.25);

    double z_max = 3.0 * H;

    std::vector<double> tau_z(n_z), E_rad_z(n_z), f_z(n_z), mu_z(n_z);
    std::vector<double> prev_rho_z(n_z, 1.0);

    double last_max_delta = 0.0;

    for (int outer = 0; outer < MAX_OUTER_ITERS; ++outer) {
        const double dz = z_max / (n_z - 1);

        std::fill(out.rho_z.begin(), out.rho_z.end(), 1.0);
        std::fill(out.T_z.begin(), out.T_z.end(), T_mid);
        out.rho_z[0] = 1.0;
        out.T_z[0]   = T_mid;

        // Pass 1: tau(z)
        std::fill(tau_z.begin(), tau_z.end(), 0.0);
        for (int zi = n_z - 2; zi >= 0; --zi) {
            const double rho_h_cgs = out.rho_z[zi]   * rho_cgs_ref;
            const double rho_n_cgs = out.rho_z[zi+1] * rho_cgs_ref;
            const double kR_h = opacity_luts_.lookup_kappa_ross(
                std::clamp(rho_h_cgs, 1e-18, 1e-6),
                std::clamp(out.T_z[zi], 3000.0, 1e8));
            const double kE_h = opacity_luts_.lookup_kappa_es(
                std::clamp(rho_h_cgs, 1e-18, 1e-6),
                std::clamp(out.T_z[zi], 3000.0, 1e8));
            const double kR_n = opacity_luts_.lookup_kappa_ross(
                std::clamp(rho_n_cgs, 1e-18, 1e-6),
                std::clamp(out.T_z[zi+1], 3000.0, 1e8));
            const double kE_n = opacity_luts_.lookup_kappa_es(
                std::clamp(rho_n_cgs, 1e-18, 1e-6),
                std::clamp(out.T_z[zi+1], 3000.0, 1e8));
            const double dtau = 0.5 * ((kR_h + kE_h) * rho_h_cgs
                                      + (kR_n + kE_n) * rho_n_cgs) * dz;
            tau_z[zi] = tau_z[zi+1] + dtau;
        }

        // Pass 2: T(z) from Eddington
        for (int zi = 0; zi < n_z; ++zi) {
            const double T4 = 0.75 * T_eff*T_eff*T_eff*T_eff * (tau_z[zi] + 2.0/3.0);
            out.T_z[zi] = std::pow(std::max(T4, 0.0), 0.25);
        }

        // Pass 3: radiation field and flux limiter
        for (int zi = 0; zi < n_z; ++zi) {
            E_rad_z[zi] = a_rad * std::pow(out.T_z[zi], 4.0);
            const double rho_cgs = out.rho_z[zi] * rho_cgs_ref;
            mu_z[zi] = opacity_luts_.lookup_mu(
                std::clamp(rho_cgs, 1e-18, 1e-6),
                std::clamp(out.T_z[zi], 3000.0, 1e8));
            if (mu_z[zi] <= 0.0 || !std::isfinite(mu_z[zi])) mu_z[zi] = 0.6;
        }

        for (int zi = 0; zi < n_z; ++zi) {
            double dE_dz = 0.0;
            if (zi == 0) dE_dz = 0.0;
            else if (zi == n_z - 1) dE_dz = (E_rad_z[zi] - E_rad_z[zi-1]) / dz;
            else dE_dz = (E_rad_z[zi+1] - E_rad_z[zi-1]) / (2.0 * dz);

            const double rho_cgs = out.rho_z[zi] * rho_cgs_ref;
            const double kR = opacity_luts_.lookup_kappa_ross(
                std::clamp(rho_cgs, 1e-18, 1e-6),
                std::clamp(out.T_z[zi], 3000.0, 1e8));
            const double denom = kR * rho_cgs * E_rad_z[zi];
            const double R_param = (denom < 1e-30) ? 1e30 : std::abs(dE_dz) / denom;

            const double lam = (2.0 + R_param) / (6.0 + 3.0*R_param + R_param*R_param);
            f_z[zi] = lam + lam*lam * R_param*R_param;
        }

        // Pass 4: rho(z) RK4 outward
        std::vector<double> d_cs2_dz(n_z, 0.0), d_fE_dz(n_z, 0.0);
        for (int zi = 0; zi < n_z; ++zi) {
            if (zi == 0) {
                d_cs2_dz[zi] = 0.0;
                d_fE_dz[zi]  = 0.0;
            } else if (zi == n_z - 1) {
                const double cs2_h = k_B * out.T_z[zi]   / (mu_z[zi]   * m_p);
                const double cs2_p = k_B * out.T_z[zi-1] / (mu_z[zi-1] * m_p);
                d_cs2_dz[zi] = (cs2_h - cs2_p) / dz;
                d_fE_dz[zi]  = (f_z[zi]*E_rad_z[zi] - f_z[zi-1]*E_rad_z[zi-1]) / dz;
            } else {
                const double cs2_n = k_B * out.T_z[zi+1] / (mu_z[zi+1] * m_p);
                const double cs2_p = k_B * out.T_z[zi-1] / (mu_z[zi-1] * m_p);
                d_cs2_dz[zi] = (cs2_n - cs2_p) / (2.0 * dz);
                d_fE_dz[zi]  = (f_z[zi+1]*E_rad_z[zi+1] - f_z[zi-1]*E_rad_z[zi-1]) / (2.0 * dz);
            }
        }

        // Define rhs lambda once (used by all DP45 stages)
        auto rhs = [&](double z_eval, double rho_eval) -> double {
            const double z_frac = z_eval / dz;
            const int idx = std::clamp(static_cast<int>(z_frac), 0, n_z - 2);
            const double t = z_frac - idx;
            const double cs2 = k_B * ((1.0-t)*out.T_z[idx] + t*out.T_z[idx+1])
                             / (((1.0-t)*mu_z[idx] + t*mu_z[idx+1]) * m_p);
            const double dcs2 = (1.0-t)*d_cs2_dz[idx] + t*d_cs2_dz[idx+1];
            const double dfE  = (1.0-t)*d_fE_dz[idx]  + t*d_fE_dz[idx+1];
            if (cs2 < 1e-30) return 0.0;
            const double cs2_geom = cs2 / (c_cgs * c_cgs);
            const double dcs2_geom = dcs2 / (c_cgs * c_cgs);
            const double dfE_geom = dfE / (rho_cgs_ref * c_cgs * c_cgs);
            return (-rho_eval * Omz2 * z_eval - rho_eval * dcs2_geom - dfE_geom)
                   / std::max(cs2_geom, 1e-30);
        };

        // Adaptive Dormand-Prince RK4(5) integration of dρ/dz from z=0 to z_max.
        // Variable step size handles the photosphere cliff; result is sampled
        // onto the uniform n_z grid for storage in rho_z[].
        //
        // dp45_tol is the per-step ODE accuracy. It is INTENTIONALLY decoupled
        // from params_.target_lut_eps (which controls refinement convergence —
        // a different question). Coupling the two means a user who relaxes
        // refinement also degrades integrator accuracy, which can quietly
        // corrupt rendered density. Keep these independent; if you need to
        // expose this for tuning, add a separate VolumetricParams field.
        // 1e-6 is tight enough that adjacent columns resolve the photosphere
        // cliff to bin-precision (which is what compare_columns wants) and
        // loose enough that construction stays under a few minutes on default
        // n_r/n_z. Don't fold this back into target_lut_eps.
        const double dp45_tol = 1e-6;
        const double h_floor = z_max * 1e-9;
        constexpr int MAX_DP45_STEPS = 4096;  // safety against pathological cliffs

        std::vector<double> z_samples;
        std::vector<double> rho_samples;
        z_samples.reserve(256);
        rho_samples.reserve(256);
        z_samples.push_back(0.0);
        rho_samples.push_back(1.0);

        double z_cur = 0.0;
        double rho_cur = 1.0;
        double h = z_max / 64.0;  // start with ~64 steps; adapt from there
        int step_count = 0;

        while (z_cur < z_max && step_count < MAX_DP45_STEPS) {
            ++step_count;
            h = std::min(h, z_max - z_cur);

            const double k1 = rhs(z_cur, rho_cur);
            const double k2 = rhs(z_cur + h/5.0,
                                  std::max(rho_cur + h*k1/5.0, RHO_FLOOR));
            const double k3 = rhs(z_cur + 3.0*h/10.0,
                                  std::max(rho_cur + h*(3.0*k1/40.0 + 9.0*k2/40.0), RHO_FLOOR));
            const double k4 = rhs(z_cur + 4.0*h/5.0,
                                  std::max(rho_cur + h*(44.0*k1/45.0 - 56.0*k2/15.0 + 32.0*k3/9.0), RHO_FLOOR));
            const double k5 = rhs(z_cur + 8.0*h/9.0,
                                  std::max(rho_cur + h*(19372.0*k1/6561.0 - 25360.0*k2/2187.0
                                                        + 64448.0*k3/6561.0 - 212.0*k4/729.0), RHO_FLOOR));
            const double k6 = rhs(z_cur + h,
                                  std::max(rho_cur + h*(9017.0*k1/3168.0 - 355.0*k2/33.0
                                                        + 46732.0*k3/5247.0 + 49.0*k4/176.0
                                                        - 5103.0*k5/18656.0), RHO_FLOOR));

            const double rho_next = rho_cur + h*(35.0*k1/384.0 + 500.0*k3/1113.0
                                                 + 125.0*k4/192.0 - 2187.0*k5/6784.0
                                                 + 11.0*k6/84.0);
            const double k7 = rhs(z_cur + h, std::max(rho_next, RHO_FLOOR));

            const double err = h * std::abs(71.0*k1/57600.0 - 71.0*k3/16695.0
                                            + 71.0*k4/1920.0 - 17253.0*k5/339200.0
                                            + 22.0*k6/525.0 - k7/40.0);
            const double scale = std::max(std::abs(rho_cur), RHO_FLOOR);
            const double err_rel = err / scale;

            if (err_rel < dp45_tol || h <= h_floor) {
                // Accept
                z_cur += h;
                rho_cur = std::max(rho_next, RHO_FLOOR);
                z_samples.push_back(z_cur);
                rho_samples.push_back(rho_cur);
                if (col_log) {
                    const double zf = z_cur / dz;
                    const int i = std::clamp(static_cast<int>(zf), 0, n_z - 2);
                    const double tt = zf - i;
                    const double cs2_d  = k_B * ((1.0-tt)*out.T_z[i] + tt*out.T_z[i+1])
                                        / (((1.0-tt)*mu_z[i] + tt*mu_z[i+1]) * m_p);
                    const double dcs2_d = (1.0-tt)*d_cs2_dz[i] + tt*d_cs2_dz[i+1];
                    const double dfE_d  = (1.0-tt)*d_fE_dz[i]  + tt*d_fE_dz[i+1];
                    const double cs2g   = cs2_d / (c_cgs * c_cgs);
                    const double grav = -rho_cur * Omz2 * z_cur;
                    const double gas  = -rho_cur * (dcs2_d / (c_cgs * c_cgs));
                    const double rad  = -(dfE_d / (rho_cgs_ref * c_cgs * c_cgs));
                    std::fprintf(stderr,
                        "[COL] o=%d z=%.5f rho=%.3e grav=%+.3e gas=%+.3e rad=%+.3e "
                        "cs2g=%.3e drho=%+.3e\n",
                        outer, z_cur, rho_cur, grav, gas, rad, cs2g,
                        (grav + gas + rad) / std::max(cs2g, 1e-30));
                }
                const double scale_factor = (err_rel > 1e-30)
                    ? std::clamp(0.9 * std::pow(dp45_tol / err_rel, 0.2), 0.2, 5.0)
                    : 5.0;
                h *= scale_factor;
            } else {
                // Reject — shrink and retry
                h *= std::max(0.2, 0.9 * std::pow(dp45_tol / err_rel, 0.2));
            }
        }

        // Sample rho onto the uniform n_z grid via linear interpolation
        out.rho_z[0] = 1.0;
        for (int zi = 1; zi < n_z; ++zi) {
            const double z_target = zi * dz;
            auto it = std::upper_bound(z_samples.begin(), z_samples.end(), z_target);
            if (it == z_samples.begin()) {
                out.rho_z[zi] = rho_samples[0];
            } else if (it == z_samples.end()) {
                out.rho_z[zi] = rho_samples.back();
            } else {
                const size_t hi = static_cast<size_t>(std::distance(z_samples.begin(), it));
                const size_t lo = hi - 1;
                const double span = z_samples[hi] - z_samples[lo];
                const double t = (span > 0.0) ? (z_target - z_samples[lo]) / span : 0.0;
                out.rho_z[zi] = std::max((1.0 - t) * rho_samples[lo] + t * rho_samples[hi],
                                          RHO_FLOOR);
            }
        }

        // Extend z_max if not yet at convergence floor
        if (out.rho_z[n_z-1] > CONV_FLOOR && z_max < Z_MAX_CAP_FACTOR * H) {
            z_max = std::min(z_max + H, Z_MAX_CAP_FACTOR * H);
            prev_rho_z = out.rho_z;
            continue;
        }

        // Convergence check
        double max_delta = 0.0;
        for (int zi = 0; zi < n_z; ++zi) {
            if (prev_rho_z[zi] > RHO_FLOOR * 10.0) {
                const double d = std::abs(out.rho_z[zi] - prev_rho_z[zi]) / prev_rho_z[zi];
                max_delta = std::max(max_delta, d);
            }
        }
        last_max_delta = max_delta;
        prev_rho_z = out.rho_z;
        if (outer > 0 && max_delta < 0.001) break;
    }

    out.max_delta = last_max_delta;
    out.z_max = z_max;
    return out;
}

// ============================================================================
// compute_vertical_profiles()
// ============================================================================

void VolumetricDisk::compute_vertical_profiles() {
    z_max_lut_.resize(n_r_);
    rho_profile_lut_.resize(n_r_ * n_z_, 0.0);
    T_profile_lut_.resize(n_r_ * n_z_, 0.0);

    for (int ri = 0; ri < n_r_; ++ri) {
        const double r = r_min_ + (r_outer_ - r_min_) * ri / (n_r_ - 1);
        ColumnSolution col = solve_column(r, H_lut_[ri], T_eff_lut_[ri],
                                           rho_mid_lut_[ri], n_z_);

        // Convergence warning (spec requirement): fire if the final
        // iteration-to-iteration relative density delta did not drop below threshold.
        if (col.max_delta >= 0.001) {
            std::fprintf(stderr,
                "[VolumetricDisk] WARNING: vertical profile did not converge at r_idx=%d (max delta=%.2e)\n",
                ri, col.max_delta);
        }

        z_max_lut_[ri] = col.z_max;
        for (int zi = 0; zi < n_z_; ++zi) {
            rho_profile_lut_[ri * n_z_ + zi] = col.rho_z[zi];
            T_profile_lut_[ri * n_z_ + zi]   = col.T_z[zi];
        }
    }

    std::printf("[VolumetricDisk] Vertical profiles computed via solve_column "
                "(n_r=%d, n_z=%d)\n", n_r_, n_z_);
}

// ============================================================================
// normalize_density()
// ============================================================================

void VolumetricDisk::normalize_density() {
    using namespace constants;

    // Find peak-flux radius (maximum rho_mid in orbiting region)
    int peak_idx = 0;
    double peak_rho = 0.0;
    for (int i = 0; i < n_r_; ++i) {
        const double r = r_min_ + (r_outer_ - r_min_) * i / (n_r_ - 1);
        if (r >= r_isco_ && rho_mid_lut_[i] > peak_rho) {
            peak_rho = rho_mid_lut_[i];
            peak_idx = i;
        }
    }

    if (peak_rho <= 0.0) {
        std::printf("[VolumetricDisk] Warning: peak rho_mid is zero, using rho_scale=1\n");
        rho_scale_ = 1.0;
        return;
    }

    // Integrate rho_profile * dz at peak radius to get column integral
    const double z_max = z_max_lut_[peak_idx];
    const double dz = z_max / (n_z_ - 1);
    double col_integral = 0.0;
    for (int zi = 0; zi < n_z_ - 1; ++zi) {
        // Trapezoidal integration over one side, multiply by 2 for both sides
        col_integral += 0.5 * (rho_profile_lut_[peak_idx * n_z_ + zi]
                              + rho_profile_lut_[peak_idx * n_z_ + zi + 1]) * dz;
    }
    col_integral *= 2.0; // Both sides of midplane

    // Reference opacity at peak: use a guess density to look up kappa
    const double T_peak = T_eff_lut_[peak_idx];
    // Initial rho_scale guess
    double rho_guess_cgs = 1e-10;

    // Iterate to self-consistency (Section 0 of spec)
    for (int iter = 0; iter < 3; ++iter) {
        const double kR = opacity_luts_.lookup_kappa_ross(
            std::clamp(rho_guess_cgs, 1e-18, 1e-6),
            std::clamp(T_peak, 3000.0, 1e8));
        const double kE = opacity_luts_.lookup_kappa_es(
            std::clamp(rho_guess_cgs, 1e-18, 1e-6),
            std::clamp(T_peak, 3000.0, 1e8));
        const double kappa_ref = kR + kE;

        if (kappa_ref <= 0.0 || col_integral <= 0.0) {
            rho_scale_ = 1.0;
            return;
        }

        // tau_mid = kappa_ref * rho_scale * peak_rho * col_integral
        // => rho_scale = tau_mid / (kappa_ref * peak_rho * col_integral)
        rho_scale_ = params_.tau_mid / (kappa_ref * peak_rho * col_integral);

        // Update guess for next iteration
        rho_guess_cgs = rho_scale_ * peak_rho;
        rho_guess_cgs = std::clamp(rho_guess_cgs, 1e-18, 1e-6);
    }

    std::printf("[VolumetricDisk] rho_scale = %.4e, midplane rho_cgs ~ %.4e\n",
                rho_scale_, rho_scale_ * peak_rho);
}

// ============================================================================
// compute_sigma_s_phys()
// ============================================================================

void VolumetricDisk::compute_sigma_s_phys() {
    using namespace constants;

    double b = params_.noise_compressive_b;
    double beta = std::numeric_limits<double>::quiet_NaN();
    bool used_default = false;

    if (b <= 0.0) {
        // Find peak-flux radius
        int peak_idx = 0;
        double peak_rho = 0.0;
        for (int i = 0; i < n_r_; ++i) {
            const double r = r_min_ + (r_outer_ - r_min_) * i / (n_r_ - 1);
            if (r >= r_isco_ && rho_mid_lut_[i] > peak_rho) {
                peak_rho = rho_mid_lut_[i];
                peak_idx = i;
            }
        }

        const double T_eff_peak = T_eff_lut_[peak_idx];
        const double T_mid4 = 0.75 * std::pow(T_eff_peak, 4.0)
                            * (params_.tau_mid + 2.0/3.0);
        const double T_mid = std::pow(std::max(T_mid4, 0.0), 0.25);
        double rho_mid_cgs = rho_scale_ * rho_mid_lut_[peak_idx];
        rho_mid_cgs = std::clamp(rho_mid_cgs, 1e-18, 1e-6);

        double mu = opacity_luts_.lookup_mu(rho_mid_cgs, std::clamp(T_mid, 3000.0, 1e8));
        if (mu <= 0.0 || !std::isfinite(mu)) mu = 0.6;

        const double P_gas = rho_mid_cgs * k_B * T_mid / (mu * m_p);
        const double P_rad = (a_rad / 3.0) * std::pow(T_mid, 4.0);
        beta = P_gas / (P_gas + P_rad);

        if (!std::isfinite(beta)) {
            emit(WarningSeverity::Info, "beta_fallback",
                 "pressure regime detection failed; using b=0.5");
            b = 0.5;
            used_default = true;
        } else {
            constexpr double B_GAS = 0.35;
            constexpr double B_RAD = 0.70;
            b = B_GAS + (B_RAD - B_GAS) * (1.0 - beta);
        }
    }

    sigma_s_phys_ = b * std::sqrt(std::log1p(params_.alpha));

    if (sigma_s_phys_ < 0.05 || sigma_s_phys_ > 1.5) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "σ_s_phys=%.3f outside typical [0.05, 1.5]", sigma_s_phys_);
        emit(WarningSeverity::Info, "sigma_s_atypical", buf);
    }

    std::printf("[VolumetricDisk] σ_s_phys = %.4f (b = %.3f, β = %.3f%s)\n",
                sigma_s_phys_, b,
                std::isfinite(beta) ? beta : 0.0,
                used_default ? ", default" : "");
}

bool VolumetricDisk::validate_luts() {
    bool ok = true;
    int severe_cells = 0;

    for (int i = 0; i < n_r_; ++i) {
        if (!std::isfinite(H_lut_[i]) || H_lut_[i] <= 0.0) { ++severe_cells; ok = false; }
        if (!std::isfinite(rho_mid_lut_[i]) || rho_mid_lut_[i] < 0.0) { ++severe_cells; ok = false; }
        if (!std::isfinite(T_eff_lut_[i]) || T_eff_lut_[i] < 0.0) { ++severe_cells; ok = false; }
        if (!std::isfinite(z_max_lut_[i]) || z_max_lut_[i] <= 0.0) { ++severe_cells; ok = false; }
        for (int zi = 0; zi < n_z_; ++zi) {
            const double rho = rho_profile_lut_[i * n_z_ + zi];
            const double T   = T_profile_lut_[i * n_z_ + zi];
            if (!std::isfinite(rho) || rho < 0.0) { ++severe_cells; ok = false; }
            if (!std::isfinite(T)   || T   < 0.0) { ++severe_cells; ok = false; }
        }
    }
    if (severe_cells > 0) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "validate_luts: %d non-finite or negative cells", severe_cells);
        emit(WarningSeverity::Severe, "validate_failed", buf);
    }

    // Smoothness: H jumps
    for (int i = 1; i < n_r_; ++i) {
        if (H_lut_[i] > 0.0 && H_lut_[i-1] > 0.0) {
            const double jump = std::abs(H_lut_[i] - H_lut_[i-1])
                              / std::max(H_lut_[i-1], 1e-30);
            if (jump > 0.5) {
                char buf[256];
                std::snprintf(buf, sizeof(buf),
                    "H jump %.2f at i=%d, smoothness violated", jump, i);
                emit(WarningSeverity::Promptable, "h_jump", buf);
                break;  // one warning per construction
            }
        }
    }

    // Outer-taper monotonicity
    if (outer_taper_width_ > 0.0) {
        const double r_taper_start = r_outer_ - outer_taper_width_;
        bool monotone = true;
        for (int i = 1; i < n_r_; ++i) {
            const double r = r_min_ + (r_outer_ - r_min_) * i / (n_r_ - 1);
            if (r >= r_taper_start && rho_mid_lut_[i] > rho_mid_lut_[i-1] * 1.001) {
                monotone = false; break;
            }
        }
        if (!monotone) {
            emit(WarningSeverity::Warning, "outer_taper_non_monotone",
                 "rho_mid is not monotonic in the outer-taper zone");
        }
    }

    return ok;
}

void VolumetricDisk::emit(WarningSeverity sev, std::string code, std::string message) {
    const char* level = "INFO";
    FILE* sink = stdout;
    switch (sev) {
        case WarningSeverity::Info:       level = "INFO";       sink = stdout; break;
        case WarningSeverity::Warning:    level = "WARNING";    sink = stderr; break;
        case WarningSeverity::Promptable: level = "PROMPTABLE"; sink = stderr; break;
        case WarningSeverity::Severe:     level = "SEVERE";     sink = stderr; break;
    }
    std::fprintf(sink, "[VolumetricDisk] %s [%s]: %s\n",
                 level, code.c_str(), message.c_str());
    warnings_.push_back({sev, std::move(code), std::move(message)});
}

int VolumetricDisk::promptable_count() const {
    int count = 0;
    for (const auto& w : warnings_) {
        if (w.severity >= WarningSeverity::Promptable) ++count;
    }
    return count;
}

// ============================================================================
// compare_columns() — optical-depth-weighted max-envelope error metric
// ============================================================================
// Used by Richardson refinement (Tasks 16-17) to decide whether n_z / n_r
// needs to be increased. Algorithm:
//   1. For each of N_freq sample frequencies, compute the contribution function
//      C(z) = (dτ/dz) · exp(−τ), where τ is integrated from the top down.
//   2. Normalise C(z) per frequency so it sums to 1.
//   3. Take the pointwise max-envelope across all frequencies.
//   4. Normalise the envelope → weights w(z_i) with Σw = 1.
//   5. The error at each z_i is |ρ_lo − ρ_hi| / ρ_lo, weighted by √w(z_i).
//   6. Return max(z_max relative delta, max of weighted density deltas).

double VolumetricDisk::compare_columns(const ColumnSolution& lo,
                                        const ColumnSolution& hi) const {
    const int n_lo = static_cast<int>(lo.rho_z.size());
    const int n_hi = static_cast<int>(hi.rho_z.size());
    if (n_lo < 2 || n_hi < 2 || lo.z_max <= 0.0) return 0.0;

    const int N_freq = std::max(1, params_.refine_num_frequencies);
    const double log_min = std::log10(std::max(params_.opacity_nu_min, 1e-30));
    const double log_max = std::log10(std::max(params_.opacity_nu_max, params_.opacity_nu_min * 10.0));
    const double dz_lo = lo.z_max / (n_lo - 1);

    std::vector<double> C_max(n_lo, 0.0);
    std::vector<double> dtau_local(n_lo);
    std::vector<double> tau(n_lo);
    std::vector<double> C_nu(n_lo);

    for (int k = 0; k < N_freq; ++k) {
        const double frac = (N_freq > 1) ? static_cast<double>(k) / (N_freq - 1) : 0.0;
        const double nu = std::pow(10.0, log_min + frac * (log_max - log_min));

        for (int zi = 0; zi < n_lo; ++zi) {
            const double rho_cgs = std::clamp(lo.rho_z[zi] * rho_scale_ * 1.0, 1e-18, 1e-6);
            const double T_clamped = std::clamp(lo.T_z[zi], 3000.0, 1e8);
            const double k_abs = opacity_luts_.lookup_kappa_abs(nu, rho_cgs, T_clamped);
            const double k_es  = opacity_luts_.lookup_kappa_es(rho_cgs, T_clamped);
            dtau_local[zi] = (k_abs + k_es) * lo.rho_z[zi] * dz_lo;
        }

        tau[n_lo - 1] = 0.0;
        for (int zi = n_lo - 2; zi >= 0; --zi) {
            tau[zi] = tau[zi+1] + 0.5 * (dtau_local[zi] + dtau_local[zi+1]);
        }

        for (int zi = 0; zi < n_lo; ++zi) {
            C_nu[zi] = dtau_local[zi] * std::exp(-tau[zi]);
        }
        double Z = 0.0;
        for (int zi = 0; zi < n_lo; ++zi) Z += C_nu[zi];
        if (Z > 0.0) {
            for (int zi = 0; zi < n_lo; ++zi) C_nu[zi] /= Z;
        }

        for (int zi = 0; zi < n_lo; ++zi) {
            C_max[zi] = std::max(C_max[zi], C_nu[zi]);
        }
    }

    double Z_env = 0.0;
    for (int zi = 0; zi < n_lo; ++zi) Z_env += C_max[zi];
    std::vector<double> w(n_lo);
    if (Z_env > 0.0) {
        for (int zi = 0; zi < n_lo; ++zi) w[zi] = C_max[zi] / Z_env;
    } else {
        for (int zi = 0; zi < n_lo; ++zi) w[zi] = 1.0 / n_lo;
    }

    const double zmax_delta = std::abs(lo.z_max - hi.z_max) / std::max(lo.z_max, 1e-30);

    double max_weighted = 0.0;
    for (int zi = 0; zi < n_lo; ++zi) {
        const double z_norm = static_cast<double>(zi) / (n_lo - 1);
        const double hi_idx = z_norm * (n_hi - 1);
        const int    hi_i   = std::clamp(static_cast<int>(hi_idx), 0, n_hi - 2);
        const double hi_t   = hi_idx - hi_i;
        const double rho_hi_at = (1.0 - hi_t) * hi.rho_z[hi_i] + hi_t * hi.rho_z[hi_i + 1];
        const double denom = std::max(lo.rho_z[zi], 1e-12);
        const double delta = std::abs(lo.rho_z[zi] - rho_hi_at) / denom;
        const double weighted = delta * std::sqrt(std::max(w[zi], 0.0));
        max_weighted = std::max(max_weighted, weighted);
    }
    return std::max(zmax_delta, max_weighted);
}

// ============================================================================
// refine_n_z_globally() — Richardson refinement for vertical LUT resolution
// ============================================================================

int VolumetricDisk::refine_n_z_globally() {
    int n_z = std::max(params_.min_n_z, 32);

    auto build_columns = [&](int nz) {
        std::vector<ColumnSolution> cols;
        cols.reserve(n_r_);
        for (int i = 0; i < n_r_; ++i) {
            const double r = r_min_ + (r_outer_ - r_min_) * i / (n_r_ - 1);
            cols.push_back(solve_column(r, H_lut_[i], T_eff_lut_[i],
                                         rho_mid_lut_[i], nz));
        }
        return cols;
    };

    auto store = [&](const std::vector<ColumnSolution>& cols, int nz) {
        n_z_ = nz;
        z_max_lut_.resize(n_r_);
        rho_profile_lut_.assign(n_r_ * n_z_, 0.0);
        T_profile_lut_.assign(n_r_ * n_z_, 0.0);
        for (int i = 0; i < n_r_; ++i) {
            z_max_lut_[i] = cols[i].z_max;
            for (int zi = 0; zi < n_z_; ++zi) {
                rho_profile_lut_[i * n_z_ + zi] = cols[i].rho_z[zi];
                T_profile_lut_[i * n_z_ + zi]   = cols[i].T_z[zi];
            }
        }
    };

    auto cols_lo = build_columns(n_z);

    while (true) {
        const int n_z_hi = std::min(2 * n_z, params_.max_n_z);
        if (n_z_hi <= n_z) {
            // Already at or above max_n_z — cannot double to check convergence.
            // Emit n_z_cap warning (Promptable) so callers know refinement was skipped.
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "n_z capped at %d; cannot double to verify convergence (max_n_z=%d)",
                n_z, params_.max_n_z);
            emit(WarningSeverity::Promptable, "n_z_cap", buf);
            store(cols_lo, n_z);
            return n_z;
        }
        auto cols_hi = build_columns(n_z_hi);

        double max_delta = 0.0;
        for (int i = 0; i < n_r_; ++i) {
            max_delta = std::max(max_delta, compare_columns(cols_lo[i], cols_hi[i]));
        }

        if (max_delta < params_.target_lut_eps) {
            store(cols_hi, n_z_hi);
            return n_z_hi;
        }
        if (n_z_hi >= params_.max_n_z) {
            const auto sev = (max_delta >= 2.0 * params_.target_lut_eps)
                           ? WarningSeverity::Promptable : WarningSeverity::Warning;
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "n_z capped at %d with delta=%.2e > %.2e",
                params_.max_n_z, max_delta, params_.target_lut_eps);
            emit(sev, "n_z_cap", buf);
            store(cols_hi, n_z_hi);
            return n_z_hi;
        }

        cols_lo = std::move(cols_hi);
        n_z = n_z_hi;
    }
}

// ============================================================================
// refine_n_r() — Richardson refinement for radial LUT resolution
// ============================================================================

int VolumetricDisk::refine_n_r() {
    int n_r = std::max(params_.min_n_r, 32);

    auto build_radial_at = [&](int nr) {
        n_r_ = nr;
        H_lut_.assign(n_r_, 0.0);
        rho_mid_lut_.assign(n_r_, 0.0);
        T_eff_lut_.assign(n_r_, 0.0);
        compute_radial_structure();
        compute_plunging_region_decay();
        apply_outer_radial_taper();
    };

    auto snapshot = [&]() {
        return std::tuple{H_lut_, rho_mid_lut_, T_eff_lut_};
    };

    auto compare_radial = [&](
        const std::tuple<std::vector<double>,std::vector<double>,std::vector<double>>& lo,
        const std::tuple<std::vector<double>,std::vector<double>,std::vector<double>>& hi) -> double {
        const auto& [H_lo, R_lo, T_lo] = lo;
        const auto& [H_hi, R_hi, T_hi] = hi;
        const int n_lo = static_cast<int>(H_lo.size());
        const int n_hi = static_cast<int>(H_hi.size());
        double max_delta = 0.0;
        auto cmp = [&](const std::vector<double>& a_lo, const std::vector<double>& a_hi) {
            for (int i = 0; i < n_lo; ++i) {
                const double t = static_cast<double>(i) / (n_lo - 1);
                const double hi_idx = t * (n_hi - 1);
                const int hi_i = std::clamp(static_cast<int>(hi_idx), 0, n_hi - 2);
                const double hi_t = hi_idx - hi_i;
                const double v_hi = (1.0 - hi_t) * a_hi[hi_i] + hi_t * a_hi[hi_i+1];
                const double denom = std::max(std::abs(a_lo[i]), 1e-30);
                max_delta = std::max(max_delta, std::abs(a_lo[i] - v_hi) / denom);
            }
        };
        cmp(H_lo, H_hi);
        cmp(R_lo, R_hi);
        cmp(T_lo, T_hi);
        return max_delta;
    };

    build_radial_at(n_r);
    auto snap_lo = snapshot();

    while (true) {
        const int n_r_hi = std::min(2 * n_r, params_.max_n_r);
        if (n_r_hi <= n_r) return n_r;

        build_radial_at(n_r_hi);
        auto snap_hi = snapshot();

        const double delta = compare_radial(snap_lo, snap_hi);

        if (delta < params_.target_lut_eps) return n_r_hi;
        if (n_r_hi >= params_.max_n_r) {
            const auto sev = (delta >= 2.0 * params_.target_lut_eps)
                           ? WarningSeverity::Promptable : WarningSeverity::Warning;
            char buf[256];
            std::snprintf(buf, sizeof(buf),
                "n_r capped at %d with delta=%.2e > %.2e",
                params_.max_n_r, delta, params_.target_lut_eps);
            emit(sev, "n_r_cap", buf);
            return n_r_hi;
        }
        snap_lo = std::move(snap_hi);
        n_r = n_r_hi;
    }
}

// ============================================================================
// nested_refine() — alternating Richardson refinement for n_r and n_z
// ============================================================================

std::pair<int, int> VolumetricDisk::nested_refine() {
    constexpr int MAX_NESTED_ITERS = 5;
    int n_r = std::max(params_.min_n_r, 32);
    int n_z = std::max(params_.min_n_z, 32);

    for (int iter = 0; iter < MAX_NESTED_ITERS; ++iter) {
        const int n_z_new = (params_.bins_per_h > 0)
                          ? params_.min_n_z   // forced — refinement skipped
                          : refine_n_z_globally();
        const int n_r_new = (params_.bins_per_gradient > 0)
                          ? n_r              // forced — refinement skipped
                          : refine_n_r();
        if (n_r_new == n_r && n_z_new == n_z) {
            return {n_r_new, n_z_new};
        }
        n_r = n_r_new;
        n_z = n_z_new;
    }
    emit(WarningSeverity::Promptable, "nested_refine_no_fixed_point",
         "nested refinement did not reach fixed point in 5 iterations");
    return {n_r, n_z};
}

} // namespace grrt
