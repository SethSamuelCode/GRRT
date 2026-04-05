#include "grrt/scene/volumetric_disk.h"
#include "grrt/math/constants.h"
#include <cmath>
#include <algorithm>
#include <numbers>
#include <cstdio>

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
    taper_width_ = (r_isco_ - r_horizon_) / 3.0;

    // BPT72 conserved quantities at ISCO
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

    std::printf("[VolumetricDisk] Building opacity LUTs...\n");
    // Build opacity LUTs BEFORE density normalization (Section 0 of spec)
    opacity_luts_ = build_opacity_luts(1e-18, 1e-6, 3000.0, 1e8,
                                       params_.opacity_nu_min, params_.opacity_nu_max);

    std::printf("[VolumetricDisk] Computing radial structure...\n");
    compute_radial_structure();

    std::printf("[VolumetricDisk] Computing vertical profiles...\n");
    compute_vertical_profiles();

    std::printf("[VolumetricDisk] Normalizing density...\n");
    normalize_density();

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
    const double dr = r_isco_ - r;
    return std::exp(-(dr * dr) / (taper_width_ * taper_width_));
}

// ============================================================================
// Volume bounds
// ============================================================================

bool VolumetricDisk::inside_volume(double r, double z) const {
    if (r <= r_horizon_ || r > r_outer_) return false;
    const double zm = z_max_at(r);
    return std::abs(z) < zm;
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

double VolumetricDisk::z_max_at(double r) const {
    return interp_radial(z_max_lut_, r);
}

double VolumetricDisk::density(double r, double z, double phi) const {
    if (!inside_volume(r, z)) return 0.0;
    const double z_abs = std::abs(z);
    const double rho_mid = interp_radial(rho_mid_lut_, r);
    const double rho_norm = interp_2d(rho_profile_lut_, r, z_abs);
    const double H = scale_height(r);
    const double base = rho_mid * rho_norm * rho_scale_ * taper(r);
    // Turbulent noise (Section 1.5)
    // Sample in Cartesian-like coordinates scaled to a fixed reference length
    // to avoid aliasing when r/H is huge (r/H can exceed 1000 in outer disk).
    const double nx = r * std::cos(phi) / noise_scale_;
    const double ny = r * std::sin(phi) / noise_scale_;
    const double nz = z / noise_scale_;
    const double n = noise_.evaluate_fbm(nx, ny, nz, params_.noise_octaves);
    return std::max(0.0, base * (1.0 + params_.turbulence * n));
}

double VolumetricDisk::density_cgs(double r, double z, double phi) const {
    // The density is already in CGS after rho_scale normalization
    return density(r, z, phi);
}

double VolumetricDisk::temperature(double r, double z) const {
    if (!inside_volume(r, z)) return 0.0;
    return interp_2d(T_profile_lut_, r, std::abs(z));
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

        // For r < r_isco, Omega_z^2 can go to zero/negative. Freeze H.
        if (r < r_isco_ || Omz2 <= 0.0) {
            H_lut_[i] = (H_isco > 0.0) ? H_isco : 0.01 * mass_;
            // rho_mid inside ISCO: use same proportionality as ISCO value
            // (will be further modulated by taper)
            if (i > 0 && rho_mid_lut_[isco_idx >= 0 ? isco_idx : i - 1] > 0.0) {
                rho_mid_lut_[i] = rho_mid_lut_[isco_idx >= 0 ? isco_idx : i - 1];
            } else {
                rho_mid_lut_[i] = 1.0; // placeholder, will be normalized
            }
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

    // Backfill H for bins inside ISCO
    if (H_isco > 0.0) {
        for (int i = 0; i < n_r_; ++i) {
            const double r = r_min_ + (r_outer_ - r_min_) * i / (n_r_ - 1);
            if (r < r_isco_) {
                H_lut_[i] = H_isco;
            }
        }
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
// compute_vertical_profiles()
// ============================================================================

void VolumetricDisk::compute_vertical_profiles() {
    using namespace constants;

    // Initial z_max guess: 3H (will be extended if needed)
    z_max_lut_.resize(n_r_);

    // Temporary storage for profiles (will be resized per radius if z_max changes)
    std::vector<double> rho_z(n_z_);
    std::vector<double> T_z(n_z_);
    std::vector<double> tau_z(n_z_);
    std::vector<double> E_rad_z(n_z_);
    std::vector<double> f_z(n_z_);      // Eddington factor at each z
    std::vector<double> mu_z(n_z_);     // mean molecular weight

    // Final 2D LUTs
    rho_profile_lut_.resize(n_r_ * n_z_, 0.0);
    T_profile_lut_.resize(n_r_ * n_z_, 0.0);

    for (int ri = 0; ri < n_r_; ++ri) {
        const double r = r_min_ + (r_outer_ - r_min_) * ri / (n_r_ - 1);
        const double H = H_lut_[ri];
        const double T_eff = T_eff_lut_[ri];
        const double rho_mid_val = rho_mid_lut_[ri];

        if (H <= 0.0 || T_eff <= 0.0 || rho_mid_val <= 0.0) {
            z_max_lut_[ri] = 3.0 * H;
            rho_profile_lut_[ri * n_z_] = 1.0;
            T_profile_lut_[ri * n_z_] = T_eff;
            for (int zi = 1; zi < n_z_; ++zi) {
                rho_profile_lut_[ri * n_z_ + zi] = 0.0;
                T_profile_lut_[ri * n_z_ + zi] = T_eff;
            }
            continue;
        }

        // Omega_z^2
        double Omz2 = omega_z_sq(r);
        if (r < r_isco_ || Omz2 <= 0.0) {
            Omz2 = omega_z_sq(r_isco_);
            if (Omz2 <= 0.0) Omz2 = omega_orb(r_isco_) * omega_orb(r_isco_);
        }

        // Reference CGS midplane density (for opacity lookups)
        const double kR_ref = opacity_luts_.lookup_kappa_ross(
            1e-10, std::clamp(T_eff, 3000.0, 1e8));
        const double kE_ref = opacity_luts_.lookup_kappa_es(
            1e-10, std::clamp(T_eff, 3000.0, 1e8));
        const double kappa_ref_total = std::max(kR_ref + kE_ref, 1.0);
        const double rho_cgs_ref = std::clamp(
            params_.tau_mid / (kappa_ref_total * 3.0 * H), 1e-18, 1e-6);

        // Midplane temperature from Eddington at tau = tau_mid
        const double T_mid4 = 0.75 * T_eff * T_eff * T_eff * T_eff
                             * (params_.tau_mid + 2.0 / 3.0);
        const double T_mid = std::pow(T_mid4, 0.25);

        // Dynamic z_max: start at 3H, extend up to 20H if needed
        double z_max = 3.0 * H;
        constexpr double Z_MAX_CAP = 20.0;  // in units of H

        // Iterate for self-consistency (up to 8 times)
        std::vector<double> prev_rho_z(n_z_, 1.0);
        const double rho_floor = 1e-15;  // floor relative to midplane
        for (int iter = 0; iter < 8; ++iter) {
            const double dz = z_max / (n_z_ - 1);

            // Initialize
            std::fill(rho_z.begin(), rho_z.end(), 1.0);
            std::fill(T_z.begin(), T_z.end(), T_mid);
            rho_z[0] = 1.0;
            T_z[0] = T_mid;

            // --- Pass 1: Compute tau(z) from current rho(z) ---
            std::fill(tau_z.begin(), tau_z.end(), 0.0);
            for (int zi = n_z_ - 2; zi >= 0; --zi) {
                const double rho_here_cgs = rho_z[zi] * rho_cgs_ref;
                const double rho_next_cgs = rho_z[zi + 1] * rho_cgs_ref;
                const double kR_h = opacity_luts_.lookup_kappa_ross(
                    std::clamp(rho_here_cgs, 1e-18, 1e-6),
                    std::clamp(T_z[zi], 3000.0, 1e8));
                const double kE_h = opacity_luts_.lookup_kappa_es(
                    std::clamp(rho_here_cgs, 1e-18, 1e-6),
                    std::clamp(T_z[zi], 3000.0, 1e8));
                const double kR_n = opacity_luts_.lookup_kappa_ross(
                    std::clamp(rho_next_cgs, 1e-18, 1e-6),
                    std::clamp(T_z[zi + 1], 3000.0, 1e8));
                const double kE_n = opacity_luts_.lookup_kappa_es(
                    std::clamp(rho_next_cgs, 1e-18, 1e-6),
                    std::clamp(T_z[zi + 1], 3000.0, 1e8));
                const double dtau = 0.5 * ((kR_h + kE_h) * rho_here_cgs
                                          + (kR_n + kE_n) * rho_next_cgs) * dz;
                tau_z[zi] = tau_z[zi + 1] + dtau;
            }

            // --- Pass 2: Compute T(z) from Eddington relation ---
            for (int zi = 0; zi < n_z_; ++zi) {
                const double T4 = 0.75 * T_eff * T_eff * T_eff * T_eff
                                * (tau_z[zi] + 2.0 / 3.0);
                T_z[zi] = std::pow(std::max(T4, 0.0), 0.25);
            }

            // --- Pass 3: Compute radiation field and flux limiter ---
            for (int zi = 0; zi < n_z_; ++zi) {
                E_rad_z[zi] = a_rad * T_z[zi] * T_z[zi] * T_z[zi] * T_z[zi];
                const double rho_cgs = rho_z[zi] * rho_cgs_ref;
                mu_z[zi] = opacity_luts_.lookup_mu(
                    std::clamp(rho_cgs, 1e-18, 1e-6),
                    std::clamp(T_z[zi], 3000.0, 1e8));
                if (mu_z[zi] <= 0.0 || !std::isfinite(mu_z[zi])) mu_z[zi] = 0.6;
            }

            // Compute Eddington factor f(z) via flux limiter
            for (int zi = 0; zi < n_z_; ++zi) {
                // dE_rad/dz via finite differences
                double dE_dz = 0.0;
                if (zi == 0) {
                    dE_dz = 0.0;  // symmetry at midplane
                } else if (zi == n_z_ - 1) {
                    dE_dz = (E_rad_z[zi] - E_rad_z[zi - 1]) / dz;  // one-sided
                } else {
                    dE_dz = (E_rad_z[zi + 1] - E_rad_z[zi - 1]) / (2.0 * dz);  // central
                }

                const double rho_cgs = rho_z[zi] * rho_cgs_ref;
                const double kR = opacity_luts_.lookup_kappa_ross(
                    std::clamp(rho_cgs, 1e-18, 1e-6),
                    std::clamp(T_z[zi], 3000.0, 1e8));
                const double denom = kR * rho_cgs * E_rad_z[zi];

                double R_param;
                if (denom < 1e-30) {
                    R_param = 1e30;  // free-streaming
                } else {
                    R_param = std::abs(dE_dz) / denom;
                }
                f_z[zi] = lp_eddington_factor(R_param);
            }

            // --- Pass 4: Integrate density via RK4 ---
            // ODE: dρ/dz = F(z, ρ) where
            // F = [-ρ·Ωz²·z - ρ·d(kT/μmp)/dz - d(f·E_rad)/dz] / (kT/μmp)
            // We precompute the d(kT/μmp)/dz and d(f·E_rad)/dz arrays.

            std::vector<double> d_cs2_dz(n_z_, 0.0);   // d(kT/μmp)/dz
            std::vector<double> d_fE_dz(n_z_, 0.0);     // d(f·E_rad)/dz

            for (int zi = 0; zi < n_z_; ++zi) {
                // d(kT/μmp)/dz
                if (zi == 0) {
                    d_cs2_dz[zi] = 0.0;
                } else if (zi == n_z_ - 1) {
                    const double cs2_here = k_B * T_z[zi] / (mu_z[zi] * m_p);
                    const double cs2_prev = k_B * T_z[zi-1] / (mu_z[zi-1] * m_p);
                    d_cs2_dz[zi] = (cs2_here - cs2_prev) / dz;
                } else {
                    const double cs2_next = k_B * T_z[zi+1] / (mu_z[zi+1] * m_p);
                    const double cs2_prev = k_B * T_z[zi-1] / (mu_z[zi-1] * m_p);
                    d_cs2_dz[zi] = (cs2_next - cs2_prev) / (2.0 * dz);
                }

                // d(f·E_rad)/dz
                if (zi == 0) {
                    d_fE_dz[zi] = 0.0;
                } else if (zi == n_z_ - 1) {
                    d_fE_dz[zi] = (f_z[zi]*E_rad_z[zi] - f_z[zi-1]*E_rad_z[zi-1]) / dz;
                } else {
                    d_fE_dz[zi] = (f_z[zi+1]*E_rad_z[zi+1] - f_z[zi-1]*E_rad_z[zi-1])
                                 / (2.0 * dz);
                }
            }

            // RK4 integration from midplane outward
            rho_z[0] = 1.0;
            for (int zi = 0; zi < n_z_ - 1; ++zi) {
                const double z_here = zi * dz;
                const double rho_here = rho_z[zi];

                // RHS function: dρ/dz at a given z and ρ
                // We use precomputed arrays evaluated at grid points,
                // linearly interpolating for fractional positions.
                auto rhs = [&](double z_eval, double rho_eval) -> double {
                    // Find the grid index for z_eval
                    const double z_frac = z_eval / dz;
                    const int idx = std::clamp(static_cast<int>(z_frac), 0, n_z_ - 2);
                    const double t = z_frac - idx;

                    const double cs2 = k_B * ((1.0-t)*T_z[idx] + t*T_z[idx+1])
                                     / (((1.0-t)*mu_z[idx] + t*mu_z[idx+1]) * m_p);
                    const double dcs2 = (1.0-t)*d_cs2_dz[idx] + t*d_cs2_dz[idx+1];
                    const double dfE  = (1.0-t)*d_fE_dz[idx] + t*d_fE_dz[idx+1];
                    const double Omz2_z = Omz2 * z_eval;

                    if (cs2 < 1e-30) return 0.0;

                    // Unit conversion: mixed-unit framework.
                    // z is geometric (units of M), Ωz² is geometric (1/M²).
                    // Thermodynamic quantities are CGS.
                    //
                    // The ODE is: dρ̃/dz = [-ρ̃·Ωz²·z - ρ̃·d(cs²)/dz - (1/ρ_cgs_ref)·d(f·E_rad)/dz] / cs²
                    // where ρ̃ = ρ/ρ_midplane (dimensionless), z is geometric.
                    //
                    // cs² = kT/(μmp) [cm²/s²] → divide by c² to get geometric (dimensionless)
                    // d(f·E_rad)/dz [erg/cm⁴] → divide by (ρ_cgs_ref · c²) to get
                    //   geometric units consistent with (ρ̃ · Ωz² · z) [1/M]
                    const double cs2_geom = cs2 / (c_cgs * c_cgs);
                    const double dcs2_geom = dcs2 / (c_cgs * c_cgs);
                    const double dfE_geom = dfE / (rho_cgs_ref * c_cgs * c_cgs);

                    return (-rho_eval * Omz2_z - rho_eval * dcs2_geom - dfE_geom)
                           / std::max(cs2_geom, 1e-30);
                };

                // RK4 step
                const double k1 = dz * rhs(z_here, rho_here);
                const double k2 = dz * rhs(z_here + 0.5*dz, std::max(rho_here + 0.5*k1, rho_floor));
                const double k3 = dz * rhs(z_here + 0.5*dz, std::max(rho_here + 0.5*k2, rho_floor));
                const double k4 = dz * rhs(z_here + dz, std::max(rho_here + k3, rho_floor));

                rho_z[zi + 1] = std::max(rho_here + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0,
                                          rho_floor);
            }

            // --- Check if z_max needs extending ---
            // rho_z is normalized to midplane = 1.0, so 1e-10 is relative to midplane
            if (rho_z[n_z_ - 1] > 1e-10 && z_max < Z_MAX_CAP * H) {
                z_max = std::min(z_max + H, Z_MAX_CAP * H);
                prev_rho_z = rho_z;  // update before restart to avoid stale comparison
                // This counts as one of the 8 outer iterations (counter increments via for-loop)
                continue;
            }

            // --- Convergence check ---
            double max_delta = 0.0;
            for (int zi = 0; zi < n_z_; ++zi) {
                if (prev_rho_z[zi] > rho_floor * 10.0) {
                    const double delta = std::abs(rho_z[zi] - prev_rho_z[zi]) / prev_rho_z[zi];
                    max_delta = std::max(max_delta, delta);
                }
            }
            prev_rho_z = rho_z;

            if (iter > 0 && max_delta < 0.001) break;  // converged
        }

        // Convergence warning (spec requirement)
        {
            double max_delta = 0.0;
            for (int zi = 0; zi < n_z_; ++zi) {
                if (prev_rho_z[zi] > rho_floor * 10.0) {
                    const double delta = std::abs(rho_z[zi] - prev_rho_z[zi]) / prev_rho_z[zi];
                    max_delta = std::max(max_delta, delta);
                }
            }
            if (max_delta >= 0.001) {
                std::fprintf(stderr,
                    "[VolumetricDisk] WARNING: vertical profile did not converge at r_idx=%d (max delta=%.2e)\n",
                    ri, max_delta);
            }
        }

        z_max_lut_[ri] = z_max;

        // Store profiles
        for (int zi = 0; zi < n_z_; ++zi) {
            rho_profile_lut_[ri * n_z_ + zi] = rho_z[zi];
            T_profile_lut_[ri * n_z_ + zi] = T_z[zi];
        }
    }

    std::printf("[VolumetricDisk] Vertical profiles computed. z_max range: %.2f H to %.2f H\n",
                *std::min_element(z_max_lut_.begin(), z_max_lut_.end()) / H_lut_[0],
                *std::max_element(z_max_lut_.begin(), z_max_lut_.end()) / H_lut_[n_r_/2]);
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

    // Set noise scale to ~2× scale height at peak-flux radius (avoids aliasing)
    noise_scale_ = (params_.noise_scale > 0.0) ? params_.noise_scale : 2.0 * H_lut_[peak_idx];
    if (noise_scale_ < 0.01) noise_scale_ = 0.01; // safety floor

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

} // namespace grrt
