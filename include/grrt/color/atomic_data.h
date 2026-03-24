#ifndef GRRT_ATOMIC_DATA_H
#define GRRT_ATOMIC_DATA_H

#include "grrt/math/constants.h"
#include <array>
#include <cmath>

namespace grrt::atomic {

// Maximum ionization stages tracked per element (neutral counts as stage 0)
inline constexpr int MAX_ION_STAGES = 5;

struct Element {
    double mass_fraction;                            // X, Y, or individual metal fraction
    double atomic_mass;                              // in units of m_p
    int num_stages;                                  // number of ionization stages (including neutral)
    std::array<double, MAX_ION_STAGES> chi_eV;       // ionization potential from stage i to i+1 [eV]
    std::array<double, MAX_ION_STAGES> g0;           // ground-state statistical weight per stage
    std::array<double, MAX_ION_STAGES> Z_eff;        // effective nuclear charge for Kramers bf
    std::array<int, MAX_ION_STAGES> n_outer;         // principal quantum number of outermost electron
};

// H: stages H I (neutral), H II (proton)
inline constexpr Element hydrogen = {
    .mass_fraction = 0.70,
    .atomic_mass   = 1.0,
    .num_stages    = 2,
    .chi_eV        = {13.60, 0.0, 0.0, 0.0, 0.0},
    .g0            = {2.0, 1.0, 0.0, 0.0, 0.0},
    .Z_eff         = {1.0, 0.0, 0.0, 0.0, 0.0},
    .n_outer       = {1, 0, 0, 0, 0},
};

// He: stages He I, He II, He III
inline constexpr Element helium = {
    .mass_fraction = 0.28,
    .atomic_mass   = 4.0,
    .num_stages    = 3,
    .chi_eV        = {24.59, 54.42, 0.0, 0.0, 0.0},
    .g0            = {1.0, 2.0, 1.0, 0.0, 0.0},
    .Z_eff         = {1.0, 2.0, 0.0, 0.0, 0.0},
    .n_outer       = {1, 1, 0, 0, 0},
};

// C: stages C I through C V
inline constexpr Element carbon = {
    .mass_fraction = 3.0e-3,
    .atomic_mass   = 12.0,
    .num_stages    = 5,
    .chi_eV        = {11.26, 24.38, 47.89, 64.49, 0.0},
    .g0            = {9.0, 6.0, 1.0, 6.0, 9.0},
    .Z_eff         = {1.0, 1.5, 2.0, 2.5, 0.0},
    .n_outer       = {2, 2, 2, 2, 0},
};

// O: stages O I through O V
inline constexpr Element oxygen = {
    .mass_fraction = 6.6e-3,
    .atomic_mass   = 16.0,
    .num_stages    = 5,
    .chi_eV        = {13.62, 35.12, 54.93, 77.41, 0.0},
    .g0            = {9.0, 4.0, 9.0, 4.0, 1.0},
    .Z_eff         = {1.0, 1.5, 2.0, 2.5, 0.0},
    .n_outer       = {2, 2, 2, 2, 0},
};

// Fe: stages Fe I through Fe V
inline constexpr Element iron = {
    .mass_fraction = 1.2e-3,
    .atomic_mass   = 56.0,
    .num_stages    = 5,
    .chi_eV        = {7.90, 16.19, 30.65, 54.80, 0.0},
    .g0            = {25.0, 30.0, 25.0, 28.0, 25.0},
    .Z_eff         = {1.0, 1.5, 2.0, 2.5, 0.0},
    .n_outer       = {4, 4, 3, 3, 0},
};

inline constexpr std::array<Element, 5> elements = {hydrogen, helium, carbon, oxygen, iron};
inline constexpr int NUM_ELEMENTS = 5;

// H⁻ data
inline constexpr double chi_Hminus_eV = 0.754;       // electron affinity [eV]
inline constexpr double chi_Hminus_erg = 0.754 * 1.602e-12; // [erg]
inline constexpr double g_Hminus = 1.0;               // H⁻ statistical weight

// H⁻ bound-free cross-section (Wishart 1979 Table 2)
// Tabulated as (wavelength_nm, sigma_cm2) pairs
inline constexpr int HMINUS_BF_TABLE_SIZE = 16;
inline constexpr std::array<double, HMINUS_BF_TABLE_SIZE> hminus_bf_lambda_nm = {
    200, 300, 400, 450, 500, 550, 600, 650,
    700, 750, 800, 850, 900, 1000, 1200, 1642
};
inline constexpr std::array<double, HMINUS_BF_TABLE_SIZE> hminus_bf_sigma = {
    0.36e-17, 1.06e-17, 1.63e-17, 2.33e-17, 3.02e-17, 3.58e-17, 3.96e-17, 4.14e-17,
    4.09e-17, 3.98e-17, 3.93e-17, 3.98e-17, 3.85e-17, 3.40e-17, 2.23e-17, 0.0
};

// H⁻ free-free cross-section (Bell & Berrington 1987 Table 1)
// sigma_ff(lambda, T) in cm^5 per (H I)(e⁻) pair
// Temperature grid [K]:
inline constexpr int HMINUS_FF_TEMP_SIZE = 8;
inline constexpr std::array<double, HMINUS_FF_TEMP_SIZE> hminus_ff_temps = {
    2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000
};
// Wavelength grid [nm]:
inline constexpr int HMINUS_FF_LAMBDA_SIZE = 7;
inline constexpr std::array<double, HMINUS_FF_LAMBDA_SIZE> hminus_ff_lambdas = {
    400, 500, 600, 700, 800, 900, 1000
};
// sigma_ff values [cm^5] — 7 wavelengths × 8 temperatures (row-major: lambda varies fastest)
inline constexpr std::array<double, HMINUS_FF_LAMBDA_SIZE * HMINUS_FF_TEMP_SIZE> hminus_ff_sigma = {
    // T=2000K: 400..1000 nm
    2.21e-39, 3.44e-39, 4.91e-39, 6.61e-39, 8.55e-39, 1.07e-38, 1.31e-38,
    // T=3000K
    3.30e-39, 5.15e-39, 7.35e-39, 9.90e-39, 1.28e-38, 1.60e-38, 1.96e-38,
    // T=4000K
    4.39e-39, 6.86e-39, 9.79e-39, 1.32e-38, 1.70e-38, 2.13e-38, 2.61e-38,
    // T=5000K
    5.48e-39, 8.56e-39, 1.22e-38, 1.65e-38, 2.13e-38, 2.67e-38, 3.27e-38,
    // T=6000K
    6.57e-39, 1.03e-38, 1.47e-38, 1.98e-38, 2.56e-38, 3.20e-38, 3.92e-38,
    // T=7000K
    7.66e-39, 1.20e-38, 1.71e-38, 2.30e-38, 2.98e-38, 3.73e-38, 4.57e-38,
    // T=8000K
    8.76e-39, 1.37e-38, 1.95e-38, 2.63e-38, 3.41e-38, 4.27e-38, 5.22e-38,
    // T=10000K
    1.09e-38, 1.71e-38, 2.44e-38, 3.29e-38, 4.26e-38, 5.33e-38, 6.53e-38,
};

} // namespace grrt::atomic

#endif
