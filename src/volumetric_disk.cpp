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
    opacity_luts_ = build_opacity_luts(1e-18, 1e-6, 3000.0, 1e8);

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
    const double H = scale_height(r);
    return std::abs(z) < 3.0 * H;
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
    // Radial index
    const double r_frac = std::clamp((r - r_min_) / (r_outer_ - r_min_) * (n_r_ - 1),
                                     0.0, static_cast<double>(n_r_ - 1));
    const int ri = std::clamp(static_cast<int>(r_frac), 0, n_r_ - 2);
    const double rt = r_frac - ri;

    // Vertical index: z_abs / (3*H(r)) maps to [0, n_z_-1]
    const double H = scale_height(r);
    const double z_max = 3.0 * H;
    if (z_abs >= z_max) return 0.0;
    const double z_frac = std::clamp(z_abs / z_max * (n_z_ - 1), 0.0, static_cast<double>(n_z_ - 1));
    const int zi = std::clamp(static_cast<int>(z_frac), 0, n_z_ - 2);
    const double zt = z_frac - zi;

    // Bilinear interpolation
    const auto idx = [&](int ir, int iz) { return ir * n_z_ + iz; };
    const double v00 = lut[idx(ri, zi)];
    const double v01 = lut[idx(ri, zi + 1)];
    const double v10 = lut[idx(ri + 1, zi)];
    const double v11 = lut[idx(ri + 1, zi + 1)];
    return (v00 * (1.0 - rt) + v10 * rt) * (1.0 - zt)
         + (v01 * (1.0 - rt) + v11 * rt) * zt;
}

// ============================================================================
// Public accessors
// ============================================================================

double VolumetricDisk::scale_height(double r) const {
    return interp_radial(H_lut_, r);
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
// compute_vertical_profiles()
// ============================================================================

void VolumetricDisk::compute_vertical_profiles() {
    using namespace constants;

    rho_profile_lut_.resize(n_r_ * n_z_, 0.0);
    T_profile_lut_.resize(n_r_ * n_z_, 0.0);

    for (int ri = 0; ri < n_r_; ++ri) {
        const double r = r_min_ + (r_outer_ - r_min_) * ri / (n_r_ - 1);
        const double H = H_lut_[ri];
        const double T_eff = T_eff_lut_[ri];
        const double rho_mid_val = rho_mid_lut_[ri];

        if (H <= 0.0 || T_eff <= 0.0 || rho_mid_val <= 0.0) {
            // Zero profiles: set midplane to identity for rho_profile, T_eff for T
            rho_profile_lut_[ri * n_z_] = 1.0;
            T_profile_lut_[ri * n_z_] = T_eff;
            for (int zi = 1; zi < n_z_; ++zi) {
                rho_profile_lut_[ri * n_z_ + zi] = 0.0;
                T_profile_lut_[ri * n_z_ + zi] = T_eff;
            }
            continue;
        }

        // Omega_z^2 (use ISCO value if inside ISCO)
        double Omz2 = omega_z_sq(r);
        if (r < r_isco_ || Omz2 <= 0.0) {
            Omz2 = omega_z_sq(r_isco_);
            if (Omz2 <= 0.0) Omz2 = omega_orb(r_isco_) * omega_orb(r_isco_);
        }

        const double z_max = 3.0 * H;
        const double dz = z_max / (n_z_ - 1);

        // Hydrostatic equilibrium ODE solve (iterated)
        // We store rho(z)/rho_mid as normalized profile
        std::vector<double> rho_z(n_z_, 1.0);
        std::vector<double> T_z(n_z_, T_eff);

        // Estimate a physically meaningful CGS midplane density from
        // the target optical depth: tau_mid ~ kappa * rho_mid * 3H.
        // Use a reference opacity at this T_eff to invert.
        const double kR_ref = opacity_luts_.lookup_kappa_ross(
            1e-10, std::clamp(T_eff, 3000.0, 1e8));
        const double kE_ref = opacity_luts_.lookup_kappa_es(
            1e-10, std::clamp(T_eff, 3000.0, 1e8));
        const double kappa_ref_total = std::max(kR_ref + kE_ref, 1.0);
        // H is in geometric units; for the column integral we use H directly
        // since rho_cgs * dz (both CGS) ~ rho_cgs * H * L_unit. But we're
        // working in a self-consistent normalized framework, so:
        const double rho_cgs_ref = std::clamp(
            params_.tau_mid / (kappa_ref_total * 3.0 * H),
            1e-18, 1e-6);

        // Midplane temperature from Eddington at tau = tau_mid
        const double T_mid4 = 0.75 * T_eff * T_eff * T_eff * T_eff
                             * (params_.tau_mid + 2.0 / 3.0);
        const double T_mid = std::pow(T_mid4, 0.25);

        // Iterate vertical profile (3-5 iterations for self-consistency)
        for (int iter = 0; iter < 4; ++iter) {
            // Forward pass: integrate from midplane outward
            rho_z[0] = 1.0;
            T_z[0] = T_mid;

            // Accumulate optical depth from surface inward to get tau(z)
            // First pass: estimate tau from current profiles
            std::vector<double> tau_z(n_z_, 0.0);

            // Integrate tau from surface (z_max) inward to each z
            for (int zi = n_z_ - 2; zi >= 0; --zi) {
                const double z_here = zi * dz;
                const double z_next = (zi + 1) * dz;
                const double rho_here_cgs = rho_z[zi] * rho_cgs_ref;
                const double rho_next_cgs = rho_z[zi + 1] * rho_cgs_ref;
                const double T_here = T_z[zi];
                const double T_next = T_z[zi + 1];

                // Total extinction = Rosseland absorption + Thomson scattering
                const double kR_here = opacity_luts_.lookup_kappa_ross(
                    std::clamp(rho_here_cgs, 1e-18, 1e-6),
                    std::clamp(T_here, 3000.0, 1e8));
                const double kE_here = opacity_luts_.lookup_kappa_es(
                    std::clamp(rho_here_cgs, 1e-18, 1e-6),
                    std::clamp(T_here, 3000.0, 1e8));
                const double kR_next = opacity_luts_.lookup_kappa_ross(
                    std::clamp(rho_next_cgs, 1e-18, 1e-6),
                    std::clamp(T_next, 3000.0, 1e8));
                const double kE_next = opacity_luts_.lookup_kappa_es(
                    std::clamp(rho_next_cgs, 1e-18, 1e-6),
                    std::clamp(T_next, 3000.0, 1e8));

                const double dtau = 0.5 * ((kR_here + kE_here) * rho_here_cgs
                                          + (kR_next + kE_next) * rho_next_cgs) * dz;
                tau_z[zi] = tau_z[zi + 1] + dtau;
            }

            // Now compute T(z) from Eddington relation using tau(z)
            // and drho/dz from hydrostatic equilibrium
            rho_z[0] = 1.0;
            for (int zi = 0; zi < n_z_; ++zi) {
                // Eddington T-tau: T^4 = (3/4)*T_eff^4*(tau+2/3)
                const double T4 = 0.75 * T_eff * T_eff * T_eff * T_eff
                                * (tau_z[zi] + 2.0 / 3.0);
                T_z[zi] = std::pow(std::max(T4, 0.0), 0.25);
            }

            // Integrate hydrostatic equilibrium from midplane outward
            // dP/dz = -rho * Omega_z^2 * z
            // P = P_gas + P_rad = rho*kB*T/(mu*mp) + (4*sigma/3c)*T^4
            // dP/dz = (kB*T/(mu*mp) + ...) * drho/dz + rho * kB/(mu*mp) * dT/dz + ...
            // Simplification: assume pressure profile and solve for density
            rho_z[0] = 1.0;
            for (int zi = 1; zi < n_z_; ++zi) {
                const double z_here = zi * dz;
                const double T_here = T_z[zi];
                const double T_prev = T_z[zi - 1];

                // Mean molecular weight
                const double rho_prev_cgs = rho_z[zi - 1] * rho_cgs_ref;
                const double mu = opacity_luts_.lookup_mu(
                    std::clamp(rho_prev_cgs, 1e-18, 1e-6),
                    std::clamp(T_here, 3000.0, 1e8));
                const double mu_safe = (mu > 0.0 && std::isfinite(mu)) ? mu : 0.6;

                // Pressure scale: P_gas/rho = kB*T/(mu*mp)
                const double P_gas_per_rho = k_B * T_here / (mu_safe * m_p);
                // Radiation pressure: P_rad = (4*sigma_SB/(3*c)) * T^4
                const double P_rad = 4.0 * sigma_SB / (3.0 * c_cgs) * T_here * T_here * T_here * T_here;
                const double P_rad_prev = 4.0 * sigma_SB / (3.0 * c_cgs) * T_prev * T_prev * T_prev * T_prev;

                // Total pressure per unit density (rough):
                // P_total = rho * P_gas_per_rho + P_rad
                // dP/dz = -rho * Omega_z^2 * z (in mixed geometric/CGS)
                // The Omega_z^2 is in geometric units (1/M^2), z is geometric.
                // We work with normalized rho (dimensionless), solving:
                // d(rho_norm)/dz = -(Omega_z^2 * z * rho_norm) / c_eff^2
                // where c_eff^2 = P_gas_per_rho + P_rad/(rho*rho_cgs_ref)

                const double rho_here_cgs = rho_z[zi - 1] * rho_cgs_ref;
                const double c_eff2_cgs = P_gas_per_rho + P_rad / std::max(rho_here_cgs, 1e-30);
                // Convert to geometric: c_eff_geom^2 = c_eff_cgs / c^2
                const double c_eff2_geom = c_eff2_cgs / (c_cgs * c_cgs);

                // Simple forward Euler: d(ln rho)/dz = -Omega_z^2 * z / c_eff_geom^2
                const double dlnrho = -Omz2 * z_here / std::max(c_eff2_geom, 1e-30) * dz;
                rho_z[zi] = rho_z[zi - 1] * std::exp(std::clamp(dlnrho, -10.0, 0.0));

                // Floor
                if (rho_z[zi] < 1e-12) rho_z[zi] = 1e-12;
            }
        }

        // Store profiles
        for (int zi = 0; zi < n_z_; ++zi) {
            rho_profile_lut_[ri * n_z_ + zi] = rho_z[zi];
            T_profile_lut_[ri * n_z_ + zi] = T_z[zi];
        }
    }
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
    const double H_peak = H_lut_[peak_idx];
    const double z_max = 3.0 * H_peak;
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
