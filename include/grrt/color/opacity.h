#ifndef GRRT_OPACITY_H
#define GRRT_OPACITY_H

#include "grrt/math/constants.h"
#include "grrt/color/atomic_data.h"
#include "grrt_export.h"
#include <vector>
#include <array>
#include <cstddef>

namespace grrt {

struct IonizationState {
    double n_e;
    double mu;
    double n_ion_eff;
    double n_HI;
    double n_Hminus;
    std::array<std::array<double, atomic::MAX_ION_STAGES>, atomic::NUM_ELEMENTS> n_pop;
};

GRRT_EXPORT IonizationState solve_saha(double rho_cgs, double T);

GRRT_EXPORT double alpha_ff(double nu, double T, const IonizationState& ion);
GRRT_EXPORT double alpha_hminus(double nu, double T, const IonizationState& ion);
GRRT_EXPORT double alpha_bf_ion(double nu, double T, const IonizationState& ion);
GRRT_EXPORT double kappa_es(double rho_cgs, const IonizationState& ion);
GRRT_EXPORT double kappa_abs(double nu, double rho_cgs, double T, const IonizationState& ion);
GRRT_EXPORT double planck_nu(double nu, double T);

struct OpacityLUTs {
    std::vector<double> kappa_abs_lut;
    int n_nu, n_rho, n_T;
    double log_nu_min, log_nu_max;
    double log_rho_min, log_rho_max;
    double log_T_min, log_T_max;

    std::vector<double> kappa_es_lut;
    std::vector<double> kappa_ross_lut;
    std::vector<double> mu_lut;

    GRRT_EXPORT double lookup_kappa_abs(double nu, double rho_cgs, double T) const;
    GRRT_EXPORT double lookup_kappa_es(double rho_cgs, double T) const;
    GRRT_EXPORT double lookup_kappa_ross(double rho_cgs, double T) const;
    GRRT_EXPORT double lookup_mu(double rho_cgs, double T) const;
};

GRRT_EXPORT OpacityLUTs build_opacity_luts(double rho_min, double rho_max,
                                double T_min, double T_max,
                                double nu_min = 1e14, double nu_max = 1e16);

} // namespace grrt

#endif
