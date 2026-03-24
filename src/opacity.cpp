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

// Stubs for Task 3 — return 0 so the library links
double alpha_ff(double, double, const IonizationState&) { return 0.0; }
double alpha_hminus(double, double, const IonizationState&) { return 0.0; }
double alpha_bf_ion(double, double, const IonizationState&) { return 0.0; }
double kappa_es(double, const IonizationState&) { return 0.0; }
double kappa_abs(double, double, double, const IonizationState&) { return 0.0; }
double planck_nu(double, double) { return 0.0; }

// Stubs for Task 4
double OpacityLUTs::lookup_kappa_abs(double, double, double) const { return 0.0; }
double OpacityLUTs::lookup_kappa_es(double, double) const { return 0.0; }
double OpacityLUTs::lookup_kappa_ross(double, double) const { return 0.0; }
double OpacityLUTs::lookup_mu(double, double) const { return 0.0; }
OpacityLUTs build_opacity_luts(double, double, double, double) { return {}; }

} // namespace grrt
