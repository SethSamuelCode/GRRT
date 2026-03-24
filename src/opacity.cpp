#include "grrt/color/opacity.h"
#include <cmath>
#include <numbers>
#include <algorithm>
#include <numeric>

namespace grrt {

using namespace constants;
using namespace atomic;

static double lambda_dB(double T) {
    return h_planck / std::sqrt(2.0 * std::numbers::pi * m_e * k_B * T);
}

IonizationState solve_saha(double rho_cgs, double T) {
    IonizationState result{};
    if (T < 100.0 || rho_cgs <= 0.0) {
        result.mu = 1.3;
        for (int s = 0; s < NUM_ELEMENTS; s++) {
            double n_total_s = rho_cgs * elements[s].mass_fraction
                             / (elements[s].atomic_mass * m_p);
            result.n_pop[s][0] = n_total_s;
            if (s == 0) result.n_HI = n_total_s;
        }
        return result;
    }

    double Ldb = lambda_dB(T);
    double Ldb3 = Ldb * Ldb * Ldb;
    double saha_prefactor = 2.0 / Ldb3;

    std::array<double, NUM_ELEMENTS> n_total{};
    for (int s = 0; s < NUM_ELEMENTS; s++) {
        n_total[s] = rho_cgs * elements[s].mass_fraction
                   / (elements[s].atomic_mass * m_p);
    }

    double n_e = rho_cgs * (1.0 + X_hydrogen) / (2.0 * m_p);

    for (int iter = 0; iter < 50; iter++) {
        double n_e_new = 0.0;

        for (int s = 0; s < NUM_ELEMENTS; s++) {
            const auto& elem = elements[s];
            int ns = elem.num_stages;

            std::array<double, MAX_ION_STAGES> ratio{};
            for (int i = 0; i < ns - 1; i++) {
                double chi_erg = elem.chi_eV[i] * eV_to_erg;
                ratio[i] = (saha_prefactor / n_e) * (elem.g0[i + 1] / elem.g0[i])
                         * std::exp(-chi_erg / (k_B * T));
            }

            double denom = 1.0;
            double product = 1.0;
            for (int i = 0; i < ns - 1; i++) {
                product *= ratio[i];
                denom += product;
            }

            double f0 = 1.0 / denom;
            result.n_pop[s][0] = f0 * n_total[s];
            product = f0;
            for (int i = 0; i < ns - 1; i++) {
                product *= ratio[i];
                result.n_pop[s][i + 1] = product * n_total[s];
            }

            for (int i = 1; i < ns; i++) {
                n_e_new += i * result.n_pop[s][i];
            }
        }

        double rel_change = std::abs(n_e_new - n_e) / std::max(n_e, 1e-30);
        n_e = n_e_new;
        if (rel_change < 1e-6) break;
    }

    result.n_e = n_e;
    result.n_HI = result.n_pop[0][0];

    result.n_ion_eff = 0.0;
    for (int s = 0; s < NUM_ELEMENTS; s++) {
        for (int i = 1; i < elements[s].num_stages; i++) {
            result.n_ion_eff += static_cast<double>(i * i) * result.n_pop[s][i];
        }
    }

    double n_particles = n_e;
    for (int s = 0; s < NUM_ELEMENTS; s++) {
        for (int i = 0; i < elements[s].num_stages; i++) {
            n_particles += result.n_pop[s][i];
        }
    }
    result.mu = rho_cgs / (n_particles * m_p);

    double hminus_ratio = n_e * Ldb3 * 0.25
                        * std::exp(chi_Hminus_erg / (k_B * T));
    result.n_Hminus = result.n_HI * hminus_ratio;

    return result;
}

// --- Opacity functions (Task 3) ---

/// Thermally-averaged free-free Gaunt factor (Rybicki & Lightman approximation)
static double gaunt_ff(double nu, double T) {
    double arg = 4.0 * k_B * T / (gamma_E * h_planck * nu);
    double gff = (std::sqrt(3.0) / std::numbers::pi) * std::log(std::max(arg, 1.0));
    return std::max(gff, 1.0);
}

/// Free-free (bremsstrahlung) absorption coefficient [cm^{-1}]
/// Rybicki & Lightman eq. 5.18a, includes stimulated emission correction
double alpha_ff(double nu, double T, const IonizationState& ion) {
    if (ion.n_e <= 0.0 || ion.n_ion_eff <= 0.0 || nu <= 0.0 || T <= 0.0) return 0.0;
    double gff = gaunt_ff(nu, T);
    double stim = 1.0 - std::exp(-h_planck * nu / (k_B * T));
    return C_ff * (ion.n_e * ion.n_ion_eff / (nu * nu * nu))
         * gff * std::pow(T, -0.5) * stim;
}

/// H⁻ bound-free cross-section [cm^2] from Wishart 1979 table, linearly interpolated
static double sigma_bf_hminus(double lambda_cm) {
    double lambda_nm = lambda_cm * 1e7;
    if (lambda_nm >= hminus_bf_lambda_nm.back() || lambda_nm <= 0.0) return 0.0;
    if (lambda_nm <= hminus_bf_lambda_nm.front()) return hminus_bf_sigma.front();
    for (int i = 0; i < HMINUS_BF_TABLE_SIZE - 1; i++) {
        if (lambda_nm >= hminus_bf_lambda_nm[i] && lambda_nm <= hminus_bf_lambda_nm[i + 1]) {
            double frac = (lambda_nm - hminus_bf_lambda_nm[i])
                        / (hminus_bf_lambda_nm[i + 1] - hminus_bf_lambda_nm[i]);
            return hminus_bf_sigma[i] + frac * (hminus_bf_sigma[i + 1] - hminus_bf_sigma[i]);
        }
    }
    return 0.0;
}

/// H⁻ free-free cross-section [cm^5] from Bell & Berrington 1987, bilinear interpolation
static double sigma_ff_hminus(double lambda_cm, double T) {
    double lambda_nm = lambda_cm * 1e7;
    double lam_clamped = std::clamp(lambda_nm, hminus_ff_lambdas.front(), hminus_ff_lambdas.back());
    double T_clamped = std::clamp(T, hminus_ff_temps.front(), hminus_ff_temps.back());
    int il = 0;
    for (int i = 0; i < HMINUS_FF_LAMBDA_SIZE - 1; i++) {
        if (lam_clamped >= hminus_ff_lambdas[i]) il = i;
    }
    int il1 = std::min(il + 1, HMINUS_FF_LAMBDA_SIZE - 1);
    double fl = (il == il1) ? 0.0 : (lam_clamped - hminus_ff_lambdas[il]) / (hminus_ff_lambdas[il1] - hminus_ff_lambdas[il]);
    int it = 0;
    for (int i = 0; i < HMINUS_FF_TEMP_SIZE - 1; i++) {
        if (T_clamped >= hminus_ff_temps[i]) it = i;
    }
    int it1 = std::min(it + 1, HMINUS_FF_TEMP_SIZE - 1);
    double ft = (it == it1) ? 0.0 : (T_clamped - hminus_ff_temps[it]) / (hminus_ff_temps[it1] - hminus_ff_temps[it]);
    auto val = [](int ti, int li) -> double {
        return hminus_ff_sigma[ti * HMINUS_FF_LAMBDA_SIZE + li];
    };
    double s00 = val(it, il), s01 = val(it, il1);
    double s10 = val(it1, il), s11 = val(it1, il1);
    return s00*(1-fl)*(1-ft) + s01*fl*(1-ft) + s10*(1-fl)*ft + s11*fl*ft;
}

/// Combined H⁻ bound-free + free-free absorption coefficient [cm^{-1}]
double alpha_hminus(double nu, double T, const IonizationState& ion) {
    if (T < 1500.0 || ion.n_HI <= 0.0) return 0.0;
    double lambda_cm = c_cgs / nu;
    double stim = 1.0 - std::exp(-h_planck * nu / (k_B * T));
    double alpha_bf = ion.n_Hminus * sigma_bf_hminus(lambda_cm) * stim;
    double alpha_free = ion.n_HI * ion.n_e * sigma_ff_hminus(lambda_cm, T);
    return alpha_bf + alpha_free;
}

/// Bound-free absorption from all tracked ions (Kramers cross-section with threshold edges) [cm^{-1}]
double alpha_bf_ion(double nu, double T, const IonizationState& ion) {
    double stim = 1.0 - std::exp(-h_planck * nu / (k_B * T));
    double alpha_total = 0.0;
    for (int s = 0; s < NUM_ELEMENTS; s++) {
        const auto& elem = elements[s];
        for (int i = 0; i < elem.num_stages - 1; i++) {
            if (elem.Z_eff[i] <= 0.0 || elem.n_outer[i] <= 0) continue;
            double chi_erg = elem.chi_eV[i] * eV_to_erg;
            double nu_threshold = chi_erg / h_planck;
            if (nu < nu_threshold) continue;
            double n_qn = static_cast<double>(elem.n_outer[i]);
            double Z = elem.Z_eff[i];
            double sigma_0 = 7.91e-18 * n_qn / (Z * Z);
            double sigma = sigma_0 * std::pow(nu_threshold / nu, 3.0);
            alpha_total += ion.n_pop[s][i] * sigma * stim;
        }
    }
    return alpha_total;
}

/// Thomson electron scattering opacity [cm^2/g]
double kappa_es(double rho_cgs, const IonizationState& ion) {
    if (rho_cgs <= 0.0) return 0.0;
    return sigma_T * ion.n_e / rho_cgs;
}

/// Total absorption opacity (ff + H⁻ + bf) [cm^2/g]
double kappa_abs(double nu, double rho_cgs, double T, const IonizationState& ion) {
    double alpha = alpha_ff(nu, T, ion) + alpha_hminus(nu, T, ion) + alpha_bf_ion(nu, T, ion);
    if (rho_cgs <= 0.0) return 0.0;
    return alpha / rho_cgs;
}

/// Planck function B_nu [erg/(s cm^2 Hz sr)]
double planck_nu(double nu, double T) {
    if (nu <= 0.0 || T <= 0.0) return 0.0;
    double x = h_planck * nu / (k_B * T);
    if (x > 500.0) return 0.0;
    return (2.0 * h_planck * nu * nu * nu / (c_cgs * c_cgs)) / (std::exp(x) - 1.0);
}

// --- LUT helpers ---

static void log_interp(double log_val, double log_min, double log_max, int n,
                        int& idx, double& frac) {
    double t = (log_val - log_min) / (log_max - log_min) * (n - 1);
    t = std::clamp(t, 0.0, static_cast<double>(n - 2));
    idx = static_cast<int>(t);
    frac = t - idx;
}

static double dplanck_nu_dT(double nu, double T) {
    double x = h_planck * nu / (k_B * T);
    if (x > 500.0) return 0.0;
    double ex = std::exp(x);
    double denom = (ex - 1.0) * (ex - 1.0);
    return (2.0 * h_planck * h_planck * nu * nu * nu * nu / (c_cgs * c_cgs * k_B * T * T))
         * ex / denom;
}

// --- LUT construction ---

OpacityLUTs build_opacity_luts(double rho_min, double rho_max,
                                double T_min, double T_max) {
    OpacityLUTs luts;
    luts.n_nu = 20;
    luts.n_rho = 100;
    luts.n_T = 100;
    luts.log_nu_min = std::log10(1e14);
    luts.log_nu_max = std::log10(1e16);
    luts.log_rho_min = std::log10(rho_min);
    luts.log_rho_max = std::log10(rho_max);
    luts.log_T_min = std::log10(T_min);
    luts.log_T_max = std::log10(T_max);

    size_t size_3d = luts.n_nu * luts.n_rho * luts.n_T;
    size_t size_2d = luts.n_rho * luts.n_T;
    luts.kappa_abs_lut.resize(size_3d);
    luts.kappa_es_lut.resize(size_2d);
    luts.kappa_ross_lut.resize(size_2d);
    luts.mu_lut.resize(size_2d);

    constexpr int n_ross_nu = 50;
    double log_nu_ross_min = std::log10(1e13);
    double log_nu_ross_max = std::log10(1e16);

    for (int j = 0; j < luts.n_rho; j++) {
        double log_rho = luts.log_rho_min + j * (luts.log_rho_max - luts.log_rho_min) / (luts.n_rho - 1);
        double rho = std::pow(10.0, log_rho);

        for (int k = 0; k < luts.n_T; k++) {
            double log_T = luts.log_T_min + k * (luts.log_T_max - luts.log_T_min) / (luts.n_T - 1);
            double T = std::pow(10.0, log_T);

            IonizationState ion = solve_saha(rho, T);

            size_t idx_2d = j * luts.n_T + k;
            luts.kappa_es_lut[idx_2d] = kappa_es(rho, ion);
            luts.mu_lut[idx_2d] = ion.mu;

            for (int i = 0; i < luts.n_nu; i++) {
                double log_nu = luts.log_nu_min + i * (luts.log_nu_max - luts.log_nu_min) / (luts.n_nu - 1);
                double nu = std::pow(10.0, log_nu);
                size_t idx_3d = i * luts.n_rho * luts.n_T + idx_2d;
                luts.kappa_abs_lut[idx_3d] = kappa_abs(nu, rho, T, ion);
            }

            // Rosseland mean
            double numerator = 0.0, denominator = 0.0;
            for (int i = 0; i < n_ross_nu; i++) {
                double log_nu = log_nu_ross_min + i * (log_nu_ross_max - log_nu_ross_min) / (n_ross_nu - 1);
                double nu = std::pow(10.0, log_nu);
                double dnu = nu * (log_nu_ross_max - log_nu_ross_min) / (n_ross_nu - 1) * std::log(10.0);
                double dBdT = dplanck_nu_dT(nu, T);
                double ka = kappa_abs(nu, rho, T, ion);
                if (ka > 1e-30 && dBdT > 0.0) {
                    numerator += (1.0 / ka) * dBdT * dnu;
                }
                denominator += dBdT * dnu;
            }
            luts.kappa_ross_lut[idx_2d] = (numerator > 0.0 && denominator > 0.0)
                                         ? denominator / numerator : 0.0;
        }
    }
    return luts;
}

// --- LUT lookups ---

double OpacityLUTs::lookup_kappa_abs(double nu, double rho_cgs, double T) const {
    int inu, irho, iT;
    double fnu, frho, fT;
    log_interp(std::log10(nu), log_nu_min, log_nu_max, n_nu, inu, fnu);
    log_interp(std::log10(rho_cgs), log_rho_min, log_rho_max, n_rho, irho, frho);
    log_interp(std::log10(T), log_T_min, log_T_max, n_T, iT, fT);

    auto idx = [&](int i, int j, int k) -> size_t {
        return i * n_rho * n_T + j * n_T + k;
    };

    double c000 = kappa_abs_lut[idx(inu, irho, iT)];
    double c001 = kappa_abs_lut[idx(inu, irho, iT+1)];
    double c010 = kappa_abs_lut[idx(inu, irho+1, iT)];
    double c011 = kappa_abs_lut[idx(inu, irho+1, iT+1)];
    double c100 = kappa_abs_lut[idx(inu+1, irho, iT)];
    double c101 = kappa_abs_lut[idx(inu+1, irho, iT+1)];
    double c110 = kappa_abs_lut[idx(inu+1, irho+1, iT)];
    double c111 = kappa_abs_lut[idx(inu+1, irho+1, iT+1)];

    auto safe_log = [](double x) { return std::log(std::max(x, 1e-100)); };

    double l00 = safe_log(c000)*(1-fT) + safe_log(c001)*fT;
    double l01 = safe_log(c010)*(1-fT) + safe_log(c011)*fT;
    double l10 = safe_log(c100)*(1-fT) + safe_log(c101)*fT;
    double l11 = safe_log(c110)*(1-fT) + safe_log(c111)*fT;
    double l0 = l00*(1-frho) + l01*frho;
    double l1 = l10*(1-frho) + l11*frho;
    return std::exp(l0*(1-fnu) + l1*fnu);
}

double OpacityLUTs::lookup_kappa_es(double rho_cgs, double T) const {
    int irho, iT; double frho, fT;
    log_interp(std::log10(rho_cgs), log_rho_min, log_rho_max, n_rho, irho, frho);
    log_interp(std::log10(T), log_T_min, log_T_max, n_T, iT, fT);
    double v00 = kappa_es_lut[irho*n_T+iT], v01 = kappa_es_lut[irho*n_T+iT+1];
    double v10 = kappa_es_lut[(irho+1)*n_T+iT], v11 = kappa_es_lut[(irho+1)*n_T+iT+1];
    return v00*(1-frho)*(1-fT) + v01*(1-frho)*fT + v10*frho*(1-fT) + v11*frho*fT;
}

double OpacityLUTs::lookup_kappa_ross(double rho_cgs, double T) const {
    int irho, iT; double frho, fT;
    log_interp(std::log10(rho_cgs), log_rho_min, log_rho_max, n_rho, irho, frho);
    log_interp(std::log10(T), log_T_min, log_T_max, n_T, iT, fT);
    double v00 = kappa_ross_lut[irho*n_T+iT], v01 = kappa_ross_lut[irho*n_T+iT+1];
    double v10 = kappa_ross_lut[(irho+1)*n_T+iT], v11 = kappa_ross_lut[(irho+1)*n_T+iT+1];
    return v00*(1-frho)*(1-fT) + v01*(1-frho)*fT + v10*frho*(1-fT) + v11*frho*fT;
}

double OpacityLUTs::lookup_mu(double rho_cgs, double T) const {
    int irho, iT; double frho, fT;
    log_interp(std::log10(rho_cgs), log_rho_min, log_rho_max, n_rho, irho, frho);
    log_interp(std::log10(T), log_T_min, log_T_max, n_T, iT, fT);
    double v00 = mu_lut[irho*n_T+iT], v01 = mu_lut[irho*n_T+iT+1];
    double v10 = mu_lut[(irho+1)*n_T+iT], v11 = mu_lut[(irho+1)*n_T+iT+1];
    return v00*(1-frho)*(1-fT) + v01*(1-frho)*fT + v10*frho*(1-fT) + v11*frho*fT;
}

} // namespace grrt
