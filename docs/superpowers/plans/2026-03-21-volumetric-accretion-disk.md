# Volumetric Accretion Disk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the thin equatorial accretion disk with a physically accurate volumetric disk featuring Shakura-Sunyaev vertical structure, first-principles opacity (Saha + ff + H⁻ + bf), invariant radiative transfer, simplex noise turbulence, and plunging-region dynamics — on both CPU and CUDA backends.

**Architecture:** The volumetric disk is built bottom-up: physical constants → atomic data → Saha solver → opacity functions → LUT builder → vertical structure solver → volumetric disk class → raymarching integration. The CPU path modifies `GeodesicTracer::trace()` to detect disk entry and switch to an adaptive raymarch loop. The CUDA path mirrors this in device functions. All thermodynamic/opacity data is precomputed into LUTs during initialization to avoid per-sample Saha solves.

**Tech Stack:** C++23, CUDA 12.x (device functions, constant/global memory, texture cache), CMake, OpenMP (CPU parallelism), stb_image_write (output)

**Spec:** `docs/superpowers/specs/2026-03-19-volumetric-accretion-disk-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `include/grrt/math/constants.h` | Physical constants (CGS) shared by CPU and CUDA |
| `include/grrt/math/noise.h` | CPU 3D simplex noise declaration |
| `src/noise.cpp` | CPU 3D simplex noise implementation |
| `include/grrt/color/atomic_data.h` | Ionization potentials, partition functions, cross-section tables |
| `include/grrt/color/opacity.h` | Saha solver, opacity functions, LUT builder declaration |
| `src/opacity.cpp` | Saha solver, ff/H⁻/bf/Thomson/Rosseland opacity, LUT construction |
| `include/grrt/scene/volumetric_disk.h` | Volumetric disk class declaration |
| `src/volumetric_disk.cpp` | Vertical structure, density, temperature, plunging velocity |
| `cuda/cuda_volumetric_disk.h` | CUDA device functions for volumetric disk raymarching |
| `cuda/cuda_noise.h` | CUDA 3D simplex noise device functions |
| `tests/test_opacity.cpp` | Opacity model validation (standalone executable) |
| `tests/test_volumetric.cpp` | Volumetric disk structure validation |

### Modified Files

| File | Change |
|------|--------|
| `include/grrt/types.h` | Add volumetric params to GRRTParams |
| `include/grrt/scene/accretion_disk.h` | Add CUDA accessor for ISCO radius |
| `include/grrt/geodesic/geodesic_tracer.h` | Add VolumetricDisk* parameter, raymarch methods |
| `src/geodesic_tracer.cpp` | Disk volume entry detection + raymarch loop |
| `include/grrt/render/renderer.h` | Add VolumetricDisk* member |
| `src/renderer.cpp` | Pass VolumetricDisk through render path |
| `src/api.cpp` | Create VolumetricDisk, wire params |
| `include/grrt/api.h` | (no change needed — uses GRRTParams from types.h) |
| `cli/main.cpp` | Parse new CLI flags |
| `cuda/cuda_types.h` | Add volumetric fields to RenderParams |
| `cuda/cuda_render.cu` | Add constant/global memory for new LUTs, upload wrappers, modify kernel |
| `cuda/cuda_backend.cu` | Build and upload volumetric LUTs |
| `CMakeLists.txt` | Add new source files, test targets |

---

## Task 1: Physical Constants and Atomic Data

**Files:**
- Create: `include/grrt/math/constants.h`
- Create: `include/grrt/color/atomic_data.h`

These are header-only data files with no logic — they provide the foundation for all subsequent physics code.

- [ ] **Step 1: Create `include/grrt/math/constants.h`**

```cpp
#ifndef GRRT_CONSTANTS_H
#define GRRT_CONSTANTS_H

// Physical constants in CGS units
// Used by volumetric disk opacity and thermodynamics.
// The geodesic integrator remains in pure geometric units (G=c=1).

namespace grrt::constants {

// Fundamental
inline constexpr double k_B        = 1.380649e-16;    // erg/K, Boltzmann
inline constexpr double sigma_SB   = 5.670374e-5;     // erg/(cm^2 s K^4), Stefan-Boltzmann
inline constexpr double m_p        = 1.672622e-24;     // g, proton mass
inline constexpr double m_e        = 9.10938e-28;      // g, electron mass
inline constexpr double c_cgs      = 2.997924e10;      // cm/s, speed of light
inline constexpr double h_planck   = 6.626070e-27;     // erg*s, Planck constant
inline constexpr double G_cgs      = 6.674e-8;         // cm^3/(g s^2), gravitational constant
inline constexpr double M_sun      = 1.989e33;         // g, solar mass
inline constexpr double sigma_T    = 6.652e-25;        // cm^2, Thomson cross-section
inline constexpr double eV_to_erg  = 1.602e-12;        // erg/eV
inline constexpr double Ry         = 2.180e-11;        // erg, Rydberg energy (13.6 eV)

// Derived
inline constexpr double gamma_E    = 1.7811;           // exp(Euler-Mascheroni), for Gaunt factor
inline constexpr double C_ff       = 3.69e8;           // CGS, free-free coefficient (R&L 5.18a)

// Composition
inline constexpr double X_hydrogen = 0.70;             // Hydrogen mass fraction
inline constexpr double Y_helium   = 0.28;             // Helium mass fraction
inline constexpr double Z_metal    = 0.02;             // Metal mass fraction (solar)

// Gas
inline constexpr double gamma_gas  = 5.0 / 3.0;        // Adiabatic index (ideal monatomic)
inline constexpr double mu_fully_ionized = 0.6;         // Mean molecular weight reference

} // namespace grrt::constants

#endif
```

- [ ] **Step 2: Create `include/grrt/color/atomic_data.h`**

This header contains all ionization data needed by the Saha solver and bound-free opacity. Each species has its ionization stages, ionization potentials (eV), and ground-state statistical weights.

```cpp
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
```

- [ ] **Step 3: Verify compilation**

Run:
```bash
cd C:/Users/seths/Projects/gr_ray_tracer
cmake -B build -G "Visual Studio 17 2022" && cmake --build build --config Release 2>&1 | tail -20
```

Headers are not yet included anywhere, so this just confirms they parse correctly if included. Add a temporary include in `src/accretion_disk.cpp` at the top:
```cpp
#include "grrt/math/constants.h"
#include "grrt/color/atomic_data.h"
```
Build. Then remove the temporary includes.

- [ ] **Step 4: Commit**

```bash
git add include/grrt/math/constants.h include/grrt/color/atomic_data.h
git commit -m "feat: add physical constants and atomic data headers for volumetric disk"
```

---

## Task 2: Saha Ionization Solver

**Files:**
- Create: `include/grrt/color/opacity.h`
- Create: `src/opacity.cpp`
- Create: `tests/test_opacity.cpp`
- Modify: `CMakeLists.txt`

The Saha solver computes ionization equilibrium (electron density, ion populations, mean molecular weight) for a given (rho, T). This is the foundation for all opacity calculations.

- [ ] **Step 1: Add test target to CMakeLists.txt**

After the `grrt-cli` target (around line 61), add:

```cmake
# Test executables
add_executable(test-opacity tests/test_opacity.cpp)
target_link_libraries(test-opacity PRIVATE grrt)
```

- [ ] **Step 2: Create `include/grrt/color/opacity.h` with Saha solver interface**

```cpp
#ifndef GRRT_OPACITY_H
#define GRRT_OPACITY_H

#include "grrt/math/constants.h"
#include "grrt/color/atomic_data.h"
#include <vector>
#include <array>
#include <cstddef>

namespace grrt {

// Result of Saha ionization equilibrium solve
struct IonizationState {
    double n_e;                    // electron density [cm^{-3}]
    double mu;                     // mean molecular weight [dimensionless]
    double n_ion_eff;              // sum Z_i^2 * n_i for free-free [cm^{-3}]
    double n_HI;                   // neutral hydrogen density [cm^{-3}]
    double n_Hminus;               // H⁻ density [cm^{-3}]
    // Per-element, per-stage populations [cm^{-3}]
    std::array<std::array<double, atomic::MAX_ION_STAGES>, atomic::NUM_ELEMENTS> n_pop;
};

// Solve Saha ionization equilibrium for given density and temperature
// rho_cgs: mass density [g/cm^3]
// T: temperature [K]
IonizationState solve_saha(double rho_cgs, double T);

// --- Opacity functions (computed from IonizationState) ---

// Free-free absorption coefficient [cm^{-1}]
double alpha_ff(double nu, double T, const IonizationState& ion);

// H⁻ total absorption coefficient (bound-free + free-free) [cm^{-1}]
double alpha_hminus(double nu, double T, const IonizationState& ion);

// Ion bound-free absorption coefficient [cm^{-1}]
double alpha_bf_ion(double nu, double T, const IonizationState& ion);

// Thomson scattering opacity [cm^2/g]
double kappa_es(double rho_cgs, const IonizationState& ion);

// Total monochromatic absorption opacity [cm^2/g]
double kappa_abs(double nu, double rho_cgs, double T, const IonizationState& ion);

// --- Precomputed Lookup Tables ---

struct OpacityLUTs {
    // 3D absorption opacity: kappa_abs(nu, rho, T) [cm^2/g]
    std::vector<double> kappa_abs_lut;   // [nu_idx * n_rho * n_T + rho_idx * n_T + T_idx]
    int n_nu, n_rho, n_T;
    double log_nu_min, log_nu_max;
    double log_rho_min, log_rho_max;
    double log_T_min, log_T_max;

    // 2D scattering opacity: kappa_es(rho, T) [cm^2/g]
    std::vector<double> kappa_es_lut;    // [rho_idx * n_T + T_idx]

    // 2D Rosseland mean absorption opacity: kappa_R(rho, T) [cm^2/g]
    std::vector<double> kappa_ross_lut;  // [rho_idx * n_T + T_idx]

    // 2D mean molecular weight: mu(rho, T) [dimensionless]
    std::vector<double> mu_lut;          // [rho_idx * n_T + T_idx]

    // Trilinear interpolation for absorption opacity
    double lookup_kappa_abs(double nu, double rho_cgs, double T) const;

    // Bilinear interpolation for 2D LUTs
    double lookup_kappa_es(double rho_cgs, double T) const;
    double lookup_kappa_ross(double rho_cgs, double T) const;
    double lookup_mu(double rho_cgs, double T) const;
};

// Build all opacity LUTs. Call once during initialization.
// rho_min/rho_max: density range [g/cm^3]
// T_min/T_max: temperature range [K]
OpacityLUTs build_opacity_luts(double rho_min, double rho_max,
                                double T_min, double T_max);

} // namespace grrt

#endif
```

- [ ] **Step 3: Write Saha solver test**

Create `tests/test_opacity.cpp`:

```cpp
#include "grrt/color/opacity.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

int failures = 0;

void check(const char* name, double got, double expected, double rel_tol) {
    double rel_err = std::abs(got - expected) / std::max(std::abs(expected), 1e-30);
    bool pass = rel_err < rel_tol;
    std::printf("  %s: got=%.4e expected=%.4e rel_err=%.2e %s\n",
                name, got, expected, rel_err, pass ? "PASS" : "FAIL");
    if (!pass) failures++;
}

void test_saha_fully_ionized() {
    std::printf("\n=== Saha: fully ionized (T=1e7 K, rho=1e-8 g/cm^3) ===\n");
    auto ion = grrt::solve_saha(1e-8, 1e7);

    // At T=10^7 K, H and He should be fully ionized
    // n_e ≈ rho*(1+X)/(2*m_p) = 1e-8 * 1.7 / (2 * 1.67e-24) ≈ 5.09e15
    double n_e_expected = 1e-8 * (1.0 + 0.70) / (2.0 * 1.672622e-24);
    check("n_e", ion.n_e, n_e_expected, 0.05);

    // mu should be close to 0.6 for fully ionized solar composition
    check("mu", ion.mu, 0.6, 0.05);

    // H⁻ should be negligible at this temperature
    check("n_Hminus ~0", ion.n_Hminus, 0.0, 1.0); // absolute check
}

void test_saha_partially_ionized() {
    std::printf("\n=== Saha: partially ionized (T=6000 K, rho=1e-7) ===\n");
    auto ion = grrt::solve_saha(1e-7, 6000.0);

    // At T=6000 K (solar photosphere-like), H is mostly neutral
    // n_e should be much less than fully ionized value
    double n_e_full = 1e-7 * 1.7 / (2.0 * 1.672622e-24);
    std::printf("  n_e = %.4e (fully ionized would be %.4e)\n", ion.n_e, n_e_full);
    if (ion.n_e >= n_e_full * 0.5) {
        std::printf("  FAIL: n_e should be << fully ionized at T=6000K\n");
        failures++;
    } else {
        std::printf("  PASS: n_e is significantly below fully ionized\n");
    }

    // mu should be > 1.0 for mostly neutral gas
    std::printf("  mu = %.4f\n", ion.mu);
    if (ion.mu < 0.8) {
        std::printf("  FAIL: mu should be > 0.8 for partially ionized gas\n");
        failures++;
    } else {
        std::printf("  PASS: mu > 0.8\n");
    }

    // H⁻ should be present
    std::printf("  n_Hminus = %.4e, n_HI = %.4e\n", ion.n_Hminus, ion.n_HI);
    if (ion.n_Hminus <= 0.0) {
        std::printf("  FAIL: H⁻ should be nonzero at T=6000K\n");
        failures++;
    } else {
        std::printf("  PASS: H⁻ is present\n");
    }
}

void test_saha_neutral() {
    std::printf("\n=== Saha: mostly neutral (T=3000 K, rho=1e-6) ===\n");
    auto ion = grrt::solve_saha(1e-6, 3000.0);

    // mu should be ~1.3 for neutral H+He
    check("mu ~1.3", ion.mu, 1.3, 0.15);

    // n_e should be very small
    double n_e_full = 1e-6 * 1.7 / (2.0 * 1.672622e-24);
    std::printf("  n_e = %.4e (fully ionized: %.4e, ratio: %.4e)\n",
                ion.n_e, n_e_full, ion.n_e / n_e_full);
}

int main() {
    test_saha_fully_ionized();
    test_saha_partially_ionized();
    test_saha_neutral();

    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
```

- [ ] **Step 4: Run test to verify it fails**

```bash
cmake --build build --config Release --target test-opacity 2>&1 | tail -5
```
Expected: Link error — `solve_saha` not defined.

- [ ] **Step 5: Implement Saha solver in `src/opacity.cpp`**

Create `src/opacity.cpp` with the Saha solver implementation:

```cpp
#include "grrt/color/opacity.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace grrt {

using namespace constants;
using namespace atomic;

// Thermal de Broglie wavelength [cm]
static double lambda_dB(double T) {
    return h_planck / std::sqrt(2.0 * M_PI * m_e * k_B * T);
}

IonizationState solve_saha(double rho_cgs, double T) {
    IonizationState result{};
    if (T < 100.0 || rho_cgs <= 0.0) {
        // Cold gas: everything neutral
        result.mu = 1.3; // neutral H+He
        for (int s = 0; s < NUM_ELEMENTS; s++) {
            double n_total_s = rho_cgs * elements[s].mass_fraction
                             / (elements[s].atomic_mass * m_p);
            result.n_pop[s][0] = n_total_s; // all neutral
            if (s == 0) result.n_HI = n_total_s;
        }
        return result;
    }

    double Ldb = lambda_dB(T);
    double Ldb3 = Ldb * Ldb * Ldb;
    double saha_prefactor = 2.0 / Ldb3; // (2/Lambda^3)

    // Total number density per element
    std::array<double, NUM_ELEMENTS> n_total{};
    for (int s = 0; s < NUM_ELEMENTS; s++) {
        n_total[s] = rho_cgs * elements[s].mass_fraction
                   / (elements[s].atomic_mass * m_p);
    }

    // Iterative Saha solve
    // Initial guess: fully ionized
    double n_e = rho_cgs * (1.0 + X_hydrogen) / (2.0 * m_p);

    for (int iter = 0; iter < 50; iter++) {
        double n_e_new = 0.0;

        for (int s = 0; s < NUM_ELEMENTS; s++) {
            const auto& elem = elements[s];
            int ns = elem.num_stages;

            // Compute Saha ratios: R_i = n_{i+1}/n_i
            // R_i = (saha_prefactor / n_e) * (g0[i+1]/g0[i]) * exp(-chi[i]/(k_B*T))
            std::array<double, MAX_ION_STAGES> ratio{};
            for (int i = 0; i < ns - 1; i++) {
                double chi_erg = elem.chi_eV[i] * eV_to_erg;
                ratio[i] = (saha_prefactor / n_e) * (elem.g0[i + 1] / elem.g0[i])
                         * std::exp(-chi_erg / (k_B * T));
            }

            // Compute fractional populations: f_0 = 1/(1 + R_0 + R_0*R_1 + ...)
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

            // Contribution to n_e: each ion at stage i contributes i electrons
            for (int i = 1; i < ns; i++) {
                n_e_new += i * result.n_pop[s][i];
            }
        }

        // Check convergence
        double rel_change = std::abs(n_e_new - n_e) / std::max(n_e, 1e-30);
        n_e = n_e_new;
        if (rel_change < 1e-6) break;
    }

    result.n_e = n_e;
    result.n_HI = result.n_pop[0][0]; // H I = first element, stage 0

    // Effective ion density for free-free: sum Z^2 * n_ion
    result.n_ion_eff = 0.0;
    for (int s = 0; s < NUM_ELEMENTS; s++) {
        for (int i = 1; i < elements[s].num_stages; i++) {
            result.n_ion_eff += static_cast<double>(i * i) * result.n_pop[s][i];
        }
    }

    // Mean molecular weight: mu = rho / (n_total_particles * m_p)
    double n_particles = n_e; // electrons
    for (int s = 0; s < NUM_ELEMENTS; s++) {
        for (int i = 0; i < elements[s].num_stages; i++) {
            n_particles += result.n_pop[s][i]; // ions + neutrals
        }
    }
    result.mu = rho_cgs / (n_particles * m_p);

    // H⁻ from inverted Saha: n(H⁻)/n(HI) = n_e * Ldb^3 * g(H⁻)/(2*g(HI)) * exp(chi_H⁻/(k_B*T))
    // g(H⁻)/(2*g(HI)) = 1/(2*2) = 1/4
    double hminus_ratio = n_e * Ldb3 * 0.25
                        * std::exp(chi_Hminus_erg / (k_B * T));
    result.n_Hminus = result.n_HI * hminus_ratio;

    return result;
}

} // namespace grrt
```

Add `src/opacity.cpp` to the `grrt` library in CMakeLists.txt:

```cmake
# In the add_library(grrt SHARED ...) list, add:
    src/opacity.cpp
```

- [ ] **Step 6: Build and run Saha tests**

```bash
cmake -B build -G "Visual Studio 17 2022" && cmake --build build --config Release --target test-opacity
./build/Release/test-opacity.exe
```
Expected: All tests PASS, 0 failures.

- [ ] **Step 7: Commit**

```bash
git add include/grrt/color/opacity.h src/opacity.cpp tests/test_opacity.cpp CMakeLists.txt
git commit -m "feat: add Saha ionization equilibrium solver with iterative n_e convergence"
```

---

## Task 3: Opacity Functions (ff, H⁻, bf, Thomson)

**Files:**
- Modify: `src/opacity.cpp` — add opacity functions
- Modify: `tests/test_opacity.cpp` — add opacity tests

- [ ] **Step 1: Add opacity tests to `tests/test_opacity.cpp`**

Add the following test functions before `main()`:

```cpp
void test_ff_opacity() {
    std::printf("\n=== Free-free opacity (T=1e6 K, fully ionized) ===\n");
    auto ion = grrt::solve_saha(1e-8, 1e6);
    double nu = 6.0e14; // ~500 nm
    double alpha = grrt::alpha_ff(nu, 1e6, ion);
    std::printf("  alpha_ff = %.4e cm^{-1}\n", alpha);
    // Should be positive and finite
    if (alpha <= 0.0 || !std::isfinite(alpha)) {
        std::printf("  FAIL: alpha_ff should be positive and finite\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }
}

void test_hminus_opacity() {
    std::printf("\n=== H⁻ opacity (T=6000 K, partial ionization) ===\n");
    auto ion = grrt::solve_saha(1e-7, 6000.0);
    double nu = 6.0e14; // ~500 nm
    double alpha = grrt::alpha_hminus(nu, 6000.0, ion);
    std::printf("  alpha_Hminus = %.4e cm^{-1}\n", alpha);
    std::printf("  n_Hminus = %.4e, n_HI = %.4e\n", ion.n_Hminus, ion.n_HI);

    // H⁻ should dominate over ff at this temperature
    double alpha_free = grrt::alpha_ff(nu, 6000.0, ion);
    std::printf("  alpha_ff = %.4e (H⁻ should dominate)\n", alpha_free);
    if (alpha <= alpha_free) {
        std::printf("  FAIL: H⁻ should dominate over ff at T=6000K\n");
        failures++;
    } else {
        std::printf("  PASS: H⁻ dominates\n");
    }
}

void test_bf_opacity() {
    std::printf("\n=== Bound-free ion opacity (T=3e4 K) ===\n");
    auto ion = grrt::solve_saha(1e-8, 3e4);
    double nu = 6.0e14;
    double alpha = grrt::alpha_bf_ion(nu, 3e4, ion);
    std::printf("  alpha_bf = %.4e cm^{-1}\n", alpha);
    if (alpha < 0.0 || !std::isfinite(alpha)) {
        std::printf("  FAIL: alpha_bf should be non-negative and finite\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }
}

void test_thomson() {
    std::printf("\n=== Thomson scattering (fully ionized) ===\n");
    auto ion = grrt::solve_saha(1e-8, 1e7);
    double kes = grrt::kappa_es(1e-8, ion);
    // Fully ionized: kappa_es ≈ 0.34 cm^2/g
    check("kappa_es ~0.34", kes, 0.34, 0.05);
}

void test_total_opacity() {
    std::printf("\n=== Total absorption opacity (T=5e4 K) ===\n");
    auto ion = grrt::solve_saha(1e-9, 5e4);
    double nu = 6.0e14;
    double kabs = grrt::kappa_abs(nu, 1e-9, 5e4, ion);
    std::printf("  kappa_abs(500nm, 1e-9, 5e4K) = %.4e cm^2/g\n", kabs);
    if (kabs <= 0.0 || !std::isfinite(kabs)) {
        std::printf("  FAIL\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }
}
```

Add calls in `main()`:
```cpp
    test_ff_opacity();
    test_hminus_opacity();
    test_bf_opacity();
    test_thomson();
    test_total_opacity();
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cmake --build build --config Release --target test-opacity 2>&1 | tail -5
```
Expected: Link errors for `alpha_ff`, `alpha_hminus`, `alpha_bf_ion`, `kappa_es`, `kappa_abs`.

- [ ] **Step 3: Implement opacity functions in `src/opacity.cpp`**

Add the following implementations after `solve_saha()`:

```cpp
// Gaunt factor for free-free (R&L 5.19b, Karzas & Latter 1961)
static double gaunt_ff(double nu, double T) {
    double arg = 4.0 * k_B * T / (gamma_E * h_planck * nu);
    double gff = (std::sqrt(3.0) / M_PI) * std::log(std::max(arg, 1.0));
    return std::max(gff, 1.0);
}

double alpha_ff(double nu, double T, const IonizationState& ion) {
    if (ion.n_e <= 0.0 || ion.n_ion_eff <= 0.0 || nu <= 0.0 || T <= 0.0) return 0.0;
    double gff = gaunt_ff(nu, T);
    double stim = 1.0 - std::exp(-h_planck * nu / (k_B * T));
    return C_ff * (ion.n_e * ion.n_ion_eff / (nu * nu * nu))
         * gff * std::pow(T, -0.5) * stim;
}

// H⁻ bound-free cross-section interpolation (Wishart 1979)
static double sigma_bf_hminus(double lambda_cm) {
    double lambda_nm = lambda_cm * 1e7; // cm to nm
    if (lambda_nm >= atomic::hminus_bf_lambda_nm.back() || lambda_nm <= 0.0) return 0.0;
    if (lambda_nm <= atomic::hminus_bf_lambda_nm.front()) {
        return atomic::hminus_bf_sigma.front();
    }

    // Linear interpolation
    for (int i = 0; i < atomic::HMINUS_BF_TABLE_SIZE - 1; i++) {
        if (lambda_nm >= atomic::hminus_bf_lambda_nm[i] &&
            lambda_nm <= atomic::hminus_bf_lambda_nm[i + 1]) {
            double frac = (lambda_nm - atomic::hminus_bf_lambda_nm[i])
                        / (atomic::hminus_bf_lambda_nm[i + 1] - atomic::hminus_bf_lambda_nm[i]);
            return atomic::hminus_bf_sigma[i]
                 + frac * (atomic::hminus_bf_sigma[i + 1] - atomic::hminus_bf_sigma[i]);
        }
    }
    return 0.0;
}

// H⁻ free-free cross-section interpolation (Bell & Berrington 1987)
static double sigma_ff_hminus(double lambda_cm, double T) {
    double lambda_nm = lambda_cm * 1e7;

    // Clamp to table bounds
    double lam_clamped = std::clamp(lambda_nm,
        atomic::hminus_ff_lambdas.front(), atomic::hminus_ff_lambdas.back());
    double T_clamped = std::clamp(T,
        atomic::hminus_ff_temps.front(), atomic::hminus_ff_temps.back());

    // Find lambda index
    int il = 0;
    for (int i = 0; i < atomic::HMINUS_FF_LAMBDA_SIZE - 1; i++) {
        if (lam_clamped >= atomic::hminus_ff_lambdas[i]) il = i;
    }
    int il1 = std::min(il + 1, atomic::HMINUS_FF_LAMBDA_SIZE - 1);
    double fl = (il == il1) ? 0.0
              : (lam_clamped - atomic::hminus_ff_lambdas[il])
              / (atomic::hminus_ff_lambdas[il1] - atomic::hminus_ff_lambdas[il]);

    // Find temperature index
    int it = 0;
    for (int i = 0; i < atomic::HMINUS_FF_TEMP_SIZE - 1; i++) {
        if (T_clamped >= atomic::hminus_ff_temps[i]) it = i;
    }
    int it1 = std::min(it + 1, atomic::HMINUS_FF_TEMP_SIZE - 1);
    double ft = (it == it1) ? 0.0
              : (T_clamped - atomic::hminus_ff_temps[it])
              / (atomic::hminus_ff_temps[it1] - atomic::hminus_ff_temps[it]);

    // Bilinear interpolation (table is row-major: T varies slowest)
    auto val = [](int ti, int li) -> double {
        return atomic::hminus_ff_sigma[ti * atomic::HMINUS_FF_LAMBDA_SIZE + li];
    };

    double s00 = val(it, il), s01 = val(it, il1);
    double s10 = val(it1, il), s11 = val(it1, il1);

    return s00 * (1 - fl) * (1 - ft) + s01 * fl * (1 - ft)
         + s10 * (1 - fl) * ft + s11 * fl * ft;
}

double alpha_hminus(double nu, double T, const IonizationState& ion) {
    if (T < 1500.0 || ion.n_HI <= 0.0) return 0.0;

    double lambda_cm = c_cgs / nu;
    double stim = 1.0 - std::exp(-h_planck * nu / (k_B * T));

    // Bound-free
    double alpha_bf = ion.n_Hminus * sigma_bf_hminus(lambda_cm) * stim;

    // Free-free
    double alpha_free = ion.n_HI * ion.n_e * sigma_ff_hminus(lambda_cm, T);

    return alpha_bf + alpha_free;
}

double alpha_bf_ion(double nu, double T, const IonizationState& ion) {
    double stim = 1.0 - std::exp(-h_planck * nu / (k_B * T));
    double alpha_total = 0.0;

    for (int s = 0; s < atomic::NUM_ELEMENTS; s++) {
        const auto& elem = atomic::elements[s];
        for (int i = 0; i < elem.num_stages - 1; i++) {
            // Only if this stage has bound electrons (Z_eff > 0, n_outer > 0)
            if (elem.Z_eff[i] <= 0.0 || elem.n_outer[i] <= 0) continue;

            double chi_erg = elem.chi_eV[i] * eV_to_erg;
            double nu_threshold = chi_erg / h_planck;
            if (nu < nu_threshold) continue; // Below threshold

            // Kramers cross-section: sigma_0 = 7.91e-18 * n / Z_eff^2
            double n_qn = static_cast<double>(elem.n_outer[i]);
            double Z = elem.Z_eff[i];
            double sigma_0 = 7.91e-18 * n_qn / (Z * Z);

            double sigma = sigma_0 * std::pow(nu_threshold / nu, 3.0); // g_bf ≈ 1

            alpha_total += ion.n_pop[s][i] * sigma * stim;
        }
    }

    return alpha_total;
}

double kappa_es(double rho_cgs, const IonizationState& ion) {
    if (rho_cgs <= 0.0) return 0.0;
    return sigma_T * ion.n_e / rho_cgs;
}

double kappa_abs(double nu, double rho_cgs, double T, const IonizationState& ion) {
    double alpha = alpha_ff(nu, T, ion) + alpha_hminus(nu, T, ion) + alpha_bf_ion(nu, T, ion);
    if (rho_cgs <= 0.0) return 0.0;
    return alpha / rho_cgs;
}
```

- [ ] **Step 4: Build and run opacity tests**

```bash
cmake --build build --config Release --target test-opacity && ./build/Release/test-opacity.exe
```
Expected: All tests PASS, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add src/opacity.cpp tests/test_opacity.cpp
git commit -m "feat: add monochromatic opacity functions (ff, H⁻, bf, Thomson)"
```

---

## Task 4: Opacity LUT Builder

**Files:**
- Modify: `src/opacity.cpp` — add LUT construction and interpolation
- Modify: `tests/test_opacity.cpp` — add LUT tests

- [ ] **Step 1: Add LUT tests**

Add to `tests/test_opacity.cpp`:

```cpp
void test_lut_construction() {
    std::printf("\n=== Opacity LUT construction ===\n");
    auto luts = grrt::build_opacity_luts(1e-12, 1e-4, 3000.0, 1e8);

    std::printf("  kappa_abs LUT size: %d x %d x %d = %zu entries\n",
                luts.n_nu, luts.n_rho, luts.n_T, luts.kappa_abs_lut.size());
    std::printf("  kappa_es  LUT size: %d x %d = %zu entries\n",
                luts.n_rho, luts.n_T, luts.kappa_es_lut.size());

    // Test interpolation at known point
    double kabs = luts.lookup_kappa_abs(6e14, 1e-8, 1e6);
    std::printf("  kappa_abs(6e14, 1e-8, 1e6) = %.4e\n", kabs);
    if (kabs <= 0.0 || !std::isfinite(kabs)) {
        std::printf("  FAIL: should be positive\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }

    // Thomson at fully ionized conditions
    double kes = luts.lookup_kappa_es(1e-8, 1e7);
    check("kappa_es LUT ~0.34", kes, 0.34, 0.10);

    // mu at fully ionized
    double mu = luts.lookup_mu(1e-8, 1e7);
    check("mu LUT ~0.6", mu, 0.6, 0.10);

    // Rosseland mean should be positive
    double kr = luts.lookup_kappa_ross(1e-8, 1e6);
    std::printf("  kappa_ross(1e-8, 1e6) = %.4e\n", kr);
    if (kr <= 0.0 || !std::isfinite(kr)) {
        std::printf("  FAIL\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }
}
```

Add call in `main()`:
```cpp
    test_lut_construction();
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cmake --build build --config Release --target test-opacity && ./build/Release/test-opacity.exe 2>&1 | tail -5
```
Expected: Link error for `build_opacity_luts` and lookup methods.

- [ ] **Step 3: Implement LUT builder**

Add to `src/opacity.cpp`:

```cpp
// Planck function B_nu(nu, T) [erg/(cm^2 s Hz sr)]
static double planck_nu(double nu, double T) {
    double x = h_planck * nu / (k_B * T);
    if (x > 500.0) return 0.0; // Prevent overflow
    return (2.0 * h_planck * nu * nu * nu / (c_cgs * c_cgs))
         / (std::exp(x) - 1.0);
}

// dB_nu/dT for Rosseland mean
static double dplanck_nu_dT(double nu, double T) {
    double x = h_planck * nu / (k_B * T);
    if (x > 500.0) return 0.0;
    double ex = std::exp(x);
    double denom = (ex - 1.0) * (ex - 1.0);
    return (2.0 * h_planck * h_planck * nu * nu * nu * nu / (c_cgs * c_cgs * k_B * T * T))
         * ex / denom;
}

OpacityLUTs build_opacity_luts(double rho_min, double rho_max,
                                double T_min, double T_max) {
    OpacityLUTs luts;

    // Grid dimensions
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

    // Frequency grid for Rosseland integral (50 bins for better accuracy)
    constexpr int n_ross_nu = 50;
    double log_nu_ross_min = std::log10(1e13);
    double log_nu_ross_max = std::log10(1e16);

    // Fill LUTs
    for (int j = 0; j < luts.n_rho; j++) {
        double log_rho = luts.log_rho_min + j * (luts.log_rho_max - luts.log_rho_min) / (luts.n_rho - 1);
        double rho = std::pow(10.0, log_rho);

        for (int k = 0; k < luts.n_T; k++) {
            double log_T = luts.log_T_min + k * (luts.log_T_max - luts.log_T_min) / (luts.n_T - 1);
            double T = std::pow(10.0, log_T);

            // Solve Saha once per (rho, T)
            IonizationState ion = solve_saha(rho, T);

            // 2D LUTs
            size_t idx_2d = j * luts.n_T + k;
            luts.kappa_es_lut[idx_2d] = kappa_es(rho, ion);
            luts.mu_lut[idx_2d] = ion.mu;

            // 3D absorption opacity LUT
            for (int i = 0; i < luts.n_nu; i++) {
                double log_nu = luts.log_nu_min + i * (luts.log_nu_max - luts.log_nu_min) / (luts.n_nu - 1);
                double nu = std::pow(10.0, log_nu);

                size_t idx_3d = i * luts.n_rho * luts.n_T + idx_2d;
                luts.kappa_abs_lut[idx_3d] = kappa_abs(nu, rho, T, ion);
            }

            // Rosseland mean opacity via numerical integration
            // 1/kappa_R = integral( (1/kappa_abs) * dB_nu/dT dnu ) / integral( dB_nu/dT dnu )
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

            if (numerator > 0.0 && denominator > 0.0) {
                luts.kappa_ross_lut[idx_2d] = denominator / numerator;
            } else {
                luts.kappa_ross_lut[idx_2d] = 0.0;
            }
        }
    }

    return luts;
}

// --- LUT interpolation ---

// Helper: find lower index and fractional position in log-spaced grid
static void log_interp(double log_val, double log_min, double log_max, int n,
                        int& idx, double& frac) {
    double t = (log_val - log_min) / (log_max - log_min) * (n - 1);
    t = std::clamp(t, 0.0, static_cast<double>(n - 2));
    idx = static_cast<int>(t);
    frac = t - idx;
}

double OpacityLUTs::lookup_kappa_abs(double nu, double rho_cgs, double T) const {
    int inu, irho, iT;
    double fnu, frho, fT;
    log_interp(std::log10(nu), log_nu_min, log_nu_max, n_nu, inu, fnu);
    log_interp(std::log10(rho_cgs), log_rho_min, log_rho_max, n_rho, irho, frho);
    log_interp(std::log10(T), log_T_min, log_T_max, n_T, iT, fT);

    // Trilinear interpolation in log space
    auto idx = [&](int i, int j, int k) -> size_t {
        return i * n_rho * n_T + j * n_T + k;
    };

    double c000 = kappa_abs_lut[idx(inu, irho, iT)];
    double c001 = kappa_abs_lut[idx(inu, irho, iT + 1)];
    double c010 = kappa_abs_lut[idx(inu, irho + 1, iT)];
    double c011 = kappa_abs_lut[idx(inu, irho + 1, iT + 1)];
    double c100 = kappa_abs_lut[idx(inu + 1, irho, iT)];
    double c101 = kappa_abs_lut[idx(inu + 1, irho, iT + 1)];
    double c110 = kappa_abs_lut[idx(inu + 1, irho + 1, iT)];
    double c111 = kappa_abs_lut[idx(inu + 1, irho + 1, iT + 1)];

    // Interpolate in log(kappa) for smoother results
    auto safe_log = [](double x) { return std::log(std::max(x, 1e-100)); };
    auto interp = [](double a, double b, double f) { return a + f * (b - a); };

    double l000 = safe_log(c000), l001 = safe_log(c001);
    double l010 = safe_log(c010), l011 = safe_log(c011);
    double l100 = safe_log(c100), l101 = safe_log(c101);
    double l110 = safe_log(c110), l111 = safe_log(c111);

    double l00 = interp(l000, l001, fT);
    double l01 = interp(l010, l011, fT);
    double l10 = interp(l100, l101, fT);
    double l11 = interp(l110, l111, fT);

    double l0 = interp(l00, l01, frho);
    double l1 = interp(l10, l11, frho);

    return std::exp(interp(l0, l1, fnu));
}

double OpacityLUTs::lookup_kappa_es(double rho_cgs, double T) const {
    int irho, iT;
    double frho, fT;
    log_interp(std::log10(rho_cgs), log_rho_min, log_rho_max, n_rho, irho, frho);
    log_interp(std::log10(T), log_T_min, log_T_max, n_T, iT, fT);

    double v00 = kappa_es_lut[irho * n_T + iT];
    double v01 = kappa_es_lut[irho * n_T + iT + 1];
    double v10 = kappa_es_lut[(irho + 1) * n_T + iT];
    double v11 = kappa_es_lut[(irho + 1) * n_T + iT + 1];

    return v00 * (1 - frho) * (1 - fT) + v01 * (1 - frho) * fT
         + v10 * frho * (1 - fT) + v11 * frho * fT;
}

double OpacityLUTs::lookup_kappa_ross(double rho_cgs, double T) const {
    int irho, iT;
    double frho, fT;
    log_interp(std::log10(rho_cgs), log_rho_min, log_rho_max, n_rho, irho, frho);
    log_interp(std::log10(T), log_T_min, log_T_max, n_T, iT, fT);

    double v00 = kappa_ross_lut[irho * n_T + iT];
    double v01 = kappa_ross_lut[irho * n_T + iT + 1];
    double v10 = kappa_ross_lut[(irho + 1) * n_T + iT];
    double v11 = kappa_ross_lut[(irho + 1) * n_T + iT + 1];

    return v00 * (1 - frho) * (1 - fT) + v01 * (1 - frho) * fT
         + v10 * frho * (1 - fT) + v11 * frho * fT;
}

double OpacityLUTs::lookup_mu(double rho_cgs, double T) const {
    int irho, iT;
    double frho, fT;
    log_interp(std::log10(rho_cgs), log_rho_min, log_rho_max, n_rho, irho, frho);
    log_interp(std::log10(T), log_T_min, log_T_max, n_T, iT, fT);

    double v00 = mu_lut[irho * n_T + iT];
    double v01 = mu_lut[irho * n_T + iT + 1];
    double v10 = mu_lut[(irho + 1) * n_T + iT];
    double v11 = mu_lut[(irho + 1) * n_T + iT + 1];

    return v00 * (1 - frho) * (1 - fT) + v01 * (1 - frho) * fT
         + v10 * frho * (1 - fT) + v11 * frho * fT;
}
```

- [ ] **Step 4: Build and run LUT tests**

```bash
cmake --build build --config Release --target test-opacity && ./build/Release/test-opacity.exe
```
Expected: All tests PASS, 0 failures. LUT construction may take 5-30 seconds depending on CPU.

- [ ] **Step 5: Commit**

```bash
git add src/opacity.cpp tests/test_opacity.cpp
git commit -m "feat: add opacity LUT builder with Rosseland mean, mu, and interpolation"
```

---

## Task 5: 3D Simplex Noise (CPU)

**Files:**
- Create: `include/grrt/math/noise.h`
- Create: `src/noise.cpp`
- Modify: `CMakeLists.txt`

Independent of Tasks 2-4. Can be worked on in parallel.

- [ ] **Step 1: Create `include/grrt/math/noise.h`**

```cpp
#ifndef GRRT_NOISE_H
#define GRRT_NOISE_H

#include <array>
#include <cstdint>

namespace grrt {

class SimplexNoise3D {
public:
    explicit SimplexNoise3D(uint32_t seed = 42);

    // Evaluate noise at (x, y, z), returns value in [-1, 1]
    double evaluate(double x, double y, double z) const;

    // Multi-octave evaluation: 2 octaves at scale 1 and 1/3 with 0.5 amplitude
    double evaluate_turbulent(double x, double y, double z) const;

    // Access permutation table for CUDA upload
    const std::array<int, 512>& permutation_table() const { return perm_; }

private:
    std::array<int, 512> perm_;
};

} // namespace grrt

#endif
```

- [ ] **Step 2: Create `src/noise.cpp`**

Standard 3D simplex noise (Perlin/Gustavson). Key implementation points:
- Uses permutation table shuffled by seed
- 4 gradient vectors at simplex corners
- Skew/unskew factors: F3 = 1/3, G3 = 1/6
- Returns continuous value in [-1, 1]

```cpp
#include "grrt/math/noise.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>

namespace grrt {

// 12 gradient directions for 3D simplex noise
static constexpr double grad3[12][3] = {
    {1,1,0}, {-1,1,0}, {1,-1,0}, {-1,-1,0},
    {1,0,1}, {-1,0,1}, {1,0,-1}, {-1,0,-1},
    {0,1,1}, {0,-1,1}, {0,1,-1}, {0,-1,-1}
};

static double dot3(const double g[3], double x, double y, double z) {
    return g[0]*x + g[1]*y + g[2]*z;
}

SimplexNoise3D::SimplexNoise3D(uint32_t seed) {
    // Initialize permutation table with Fisher-Yates shuffle
    std::array<int, 256> base;
    std::iota(base.begin(), base.end(), 0);

    std::mt19937 rng(seed);
    for (int i = 255; i > 0; i--) {
        std::uniform_int_distribution<int> dist(0, i);
        std::swap(base[i], base[dist(rng)]);
    }

    for (int i = 0; i < 256; i++) {
        perm_[i] = perm_[i + 256] = base[i];
    }
}

double SimplexNoise3D::evaluate(double x, double y, double z) const {
    constexpr double F3 = 1.0 / 3.0;
    constexpr double G3 = 1.0 / 6.0;

    double s = (x + y + z) * F3;
    int i = static_cast<int>(std::floor(x + s));
    int j = static_cast<int>(std::floor(y + s));
    int k = static_cast<int>(std::floor(z + s));

    double t = (i + j + k) * G3;
    double X0 = i - t, Y0 = j - t, Z0 = k - t;
    double x0 = x - X0, y0 = y - Y0, z0 = z - Z0;

    // Determine simplex
    int i1, j1, k1, i2, j2, k2;
    if (x0 >= y0) {
        if (y0 >= z0)      { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
        else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
        else               { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    } else {
        if (y0 < z0)       { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
        else if (x0 < z0)  { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
        else               { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }

    double x1 = x0 - i1 + G3, y1 = y0 - j1 + G3, z1 = z0 - k1 + G3;
    double x2 = x0 - i2 + 2*G3, y2 = y0 - j2 + 2*G3, z2 = z0 - k2 + 2*G3;
    double x3 = x0 - 1 + 3*G3, y3 = y0 - 1 + 3*G3, z3 = z0 - 1 + 3*G3;

    int ii = i & 255, jj = j & 255, kk = k & 255;

    auto contrib = [&](double cx, double cy, double cz, int gi) -> double {
        double t0 = 0.6 - cx*cx - cy*cy - cz*cz;
        if (t0 < 0.0) return 0.0;
        t0 *= t0;
        return t0 * t0 * dot3(grad3[gi % 12], cx, cy, cz);
    };

    double n0 = contrib(x0, y0, z0, perm_[ii + perm_[jj + perm_[kk]]]);
    double n1 = contrib(x1, y1, z1, perm_[ii+i1 + perm_[jj+j1 + perm_[kk+k1]]]);
    double n2 = contrib(x2, y2, z2, perm_[ii+i2 + perm_[jj+j2 + perm_[kk+k2]]]);
    double n3 = contrib(x3, y3, z3, perm_[ii+1 + perm_[jj+1 + perm_[kk+1]]]);

    return 32.0 * (n0 + n1 + n2 + n3);
}

double SimplexNoise3D::evaluate_turbulent(double x, double y, double z) const {
    return evaluate(x, y, z) + 0.5 * evaluate(x * 3.0, y * 3.0, z * 3.0);
}

} // namespace grrt
```

Add `src/noise.cpp` to CMakeLists.txt's `add_library(grrt SHARED ...)` list.

- [ ] **Step 3: Build and verify**

```bash
cmake -B build -G "Visual Studio 17 2022" && cmake --build build --config Release
```
Expected: Clean build.

- [ ] **Step 4: Commit**

```bash
git add include/grrt/math/noise.h src/noise.cpp CMakeLists.txt
git commit -m "feat: add 3D simplex noise for turbulent density structure"
```

---

## Task 6: Volumetric Disk Class (CPU)

**Files:**
- Create: `include/grrt/scene/volumetric_disk.h`
- Create: `src/volumetric_disk.cpp`
- Create: `tests/test_volumetric.cpp`
- Modify: `CMakeLists.txt`

This is the central piece — it owns the vertical structure solver, density/temperature LUTs, opacity LUTs, noise, and scale height computation.

- [ ] **Step 1: Add test target to CMakeLists.txt**

```cmake
add_executable(test-volumetric tests/test_volumetric.cpp)
target_link_libraries(test-volumetric PRIVATE grrt)
```

- [ ] **Step 2: Create `include/grrt/scene/volumetric_disk.h`**

```cpp
#ifndef GRRT_VOLUMETRIC_DISK_H
#define GRRT_VOLUMETRIC_DISK_H

#include "grrt/math/constants.h"
#include "grrt/math/noise.h"
#include "grrt/color/opacity.h"
#include "grrt/spacetime/kerr.h"
#include <vector>
#include <cmath>

namespace grrt {

struct VolumetricParams {
    double alpha      = 0.1;    // Shakura-Sunyaev viscosity
    double turbulence = 0.4;    // Noise amplitude delta
    uint32_t seed     = 42;     // Noise seed
    double tau_mid    = 100.0;  // Midplane optical depth normalization
};

class VolumetricDisk {
public:
    VolumetricDisk(double mass, double spin, double r_outer,
                   double peak_temperature, const VolumetricParams& params = {});

    // Accessors for raymarching
    double scale_height(double r) const;
    double density(double r, double z, double phi) const;        // [geometric units, scaled]
    double density_cgs(double r, double z, double phi) const;    // [g/cm^3]
    double temperature(double r, double z) const;                // [K]
    double taper(double r) const;
    bool inside_volume(double r, double z) const;

    // Kerr orbital mechanics for redshift
    double omega_orb(double r) const;
    double omega_z_sq(double r) const;

    // Plunging 4-velocity components (r < r_isco)
    void plunging_velocity(double r, double theta,
                           double& ut, double& ur, double& uphi) const;
    // Circular orbit 4-velocity (r >= r_isco)
    void circular_velocity(double r, double& ut, double& uphi) const;

    // Opacity LUT access (for raymarching)
    const OpacityLUTs& opacity_luts() const { return opacity_luts_; }

    // CUDA data accessors
    const std::vector<double>& scale_height_lut() const { return H_lut_; }
    const std::vector<double>& rho_mid_lut() const { return rho_mid_lut_; }
    const std::vector<double>& density_profile_lut() const { return rho_profile_lut_; }
    const std::vector<double>& temperature_profile_lut() const { return T_profile_lut_; }
    int radial_bins() const { return n_r_; }
    int vertical_bins() const { return n_z_; }
    double r_min() const { return r_min_; }
    double r_max() const { return r_outer_; }
    double r_isco() const { return r_isco_; }
    double r_horizon() const { return r_horizon_; }
    const SimplexNoise3D& noise() const { return noise_; }

private:
    double mass_, spin_, r_outer_, peak_temperature_;
    double r_isco_, r_horizon_;
    double r_min_;       // Inner bound (slightly outside horizon)
    double taper_width_; // Gaussian taper width inside ISCO
    VolumetricParams params_;
    SimplexNoise3D noise_;

    // BPT72 conserved quantities at ISCO
    double E_isco_, L_isco_;

    // Radial LUTs (n_r_ bins from r_min_ to r_outer_)
    int n_r_ = 500;
    std::vector<double> H_lut_;        // scale height H(r) [geometric]
    std::vector<double> rho_mid_lut_;  // midplane density [geometric, scaled]
    std::vector<double> T_eff_lut_;    // effective temperature T_eff(r) [K]

    // 2D vertical structure LUTs (n_r_ × n_z_)
    int n_z_ = 64;
    std::vector<double> rho_profile_lut_;  // rho(r, z) / rho_mid(r) [normalized 0..1]
    std::vector<double> T_profile_lut_;    // T(r, z) [K]

    // Opacity LUTs
    OpacityLUTs opacity_luts_;

    // Density normalization factor
    double rho_scale_ = 1.0;

    // Construction helpers
    void build_flux_lut(std::vector<double>& flux, double& flux_max) const;
    void compute_radial_structure();
    void compute_vertical_profiles();
    void normalize_density();

    // LUT interpolation helpers
    double interp_radial(const std::vector<double>& lut, double r) const;
    double interp_2d(const std::vector<double>& lut, double r, double z_abs) const;
};

} // namespace grrt

#endif
```

- [ ] **Step 3: Create `tests/test_volumetric.cpp`**

```cpp
#include "grrt/scene/volumetric_disk.h"
#include <cstdio>
#include <cmath>

int failures = 0;

void check(const char* name, double got, double expected, double rel_tol) {
    double rel_err = std::abs(got - expected) / std::max(std::abs(expected), 1e-30);
    bool pass = rel_err < rel_tol;
    std::printf("  %s: got=%.4e expected=%.4e rel_err=%.2e %s\n",
                name, got, expected, rel_err, pass ? "PASS" : "FAIL");
    if (!pass) failures++;
}

void test_construction() {
    std::printf("\n=== VolumetricDisk construction (a=0.998, T_peak=1e7 K) ===\n");
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);
    std::printf("  r_isco = %.4f M\n", disk.r_isco());
    std::printf("  r_horizon = %.4f M\n", disk.r_horizon());

    // Scale height should be positive and finite
    double H = disk.scale_height(10.0);
    std::printf("  H(10M) = %.4f M\n", H);
    if (H <= 0.0 || !std::isfinite(H)) {
        std::printf("  FAIL\n"); failures++;
    } else {
        std::printf("  PASS\n");
    }

    // H should increase inward (radiation pressure puffs inner disk)
    double H_inner = disk.scale_height(3.0);
    double H_outer = disk.scale_height(20.0);
    double ratio = H_inner / 3.0;  // H/r at inner edge
    double ratio_outer = H_outer / 20.0;
    std::printf("  H/r at r=3M: %.4f, H/r at r=20M: %.4f\n", ratio, ratio_outer);
}

void test_density_profile() {
    std::printf("\n=== Density profile ===\n");
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);

    double r = 10.0;
    double H = disk.scale_height(r);
    double rho_mid = disk.density(r, 0.0, 0.0);
    double rho_1H = disk.density(r, H, 0.0);
    double rho_3H = disk.density(r, 3.0 * H, 0.0);

    std::printf("  rho(midplane) = %.4e\n", rho_mid);
    std::printf("  rho(1H) = %.4e (ratio: %.4f)\n", rho_1H, rho_1H / rho_mid);
    std::printf("  rho(3H) = %.4e (ratio: %.4e)\n", rho_3H, rho_3H / rho_mid);

    // Density should decrease with height
    if (rho_1H >= rho_mid || rho_3H >= rho_1H) {
        std::printf("  FAIL: density should decrease with height\n"); failures++;
    } else {
        std::printf("  PASS: density decreases with height\n");
    }
}

void test_temperature_profile() {
    std::printf("\n=== Temperature profile ===\n");
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);

    double r = 10.0;
    double H = disk.scale_height(r);
    double T_mid = disk.temperature(r, 0.0);
    double T_1H = disk.temperature(r, H);
    double T_3H = disk.temperature(r, 3.0 * H);

    std::printf("  T(midplane) = %.2f K\n", T_mid);
    std::printf("  T(1H) = %.2f K\n", T_1H);
    std::printf("  T(3H) = %.2f K\n", T_3H);

    // Temperature should decrease with height (Eddington)
    if (T_1H >= T_mid || T_3H >= T_1H) {
        std::printf("  FAIL: T should decrease with height\n"); failures++;
    } else {
        std::printf("  PASS\n");
    }
}

void test_taper() {
    std::printf("\n=== ISCO taper ===\n");
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);

    double t_isco = disk.taper(disk.r_isco());
    double t_outside = disk.taper(disk.r_isco() + 1.0);
    double t_horizon = disk.taper(disk.r_horizon());

    std::printf("  taper(r_isco) = %.6f\n", t_isco);
    std::printf("  taper(r_isco+1) = %.6f\n", t_outside);
    std::printf("  taper(r_horizon) = %.6e\n", t_horizon);

    // taper at ISCO should be 1, outside should be 1, at horizon should be small
    check("taper(r_isco)", t_isco, 1.0, 0.01);
    check("taper(r_isco+1)", t_outside, 1.0, 0.01);
    if (t_horizon > 0.1) {
        std::printf("  FAIL: taper at horizon should be small\n"); failures++;
    } else {
        std::printf("  PASS: taper at horizon is %.4e\n", t_horizon);
    }
}

void test_volume_bounds() {
    std::printf("\n=== Volume bounds ===\n");
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);

    // Midplane at r=10 should be inside
    if (!disk.inside_volume(10.0, 0.0)) {
        std::printf("  FAIL: midplane at r=10 should be inside\n"); failures++;
    } else {
        std::printf("  PASS: midplane inside\n");
    }

    // Far above disk should be outside
    if (disk.inside_volume(10.0, 100.0)) {
        std::printf("  FAIL: z=100 should be outside\n"); failures++;
    } else {
        std::printf("  PASS: far above is outside\n");
    }

    // Beyond r_outer should be outside
    if (disk.inside_volume(50.0, 0.0)) {
        std::printf("  FAIL: r=50 should be outside (r_outer=30)\n"); failures++;
    } else {
        std::printf("  PASS: beyond r_outer is outside\n");
    }
}

int main() {
    test_construction();
    test_density_profile();
    test_temperature_profile();
    test_taper();
    test_volume_bounds();

    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
```

- [ ] **Step 4: Implement `src/volumetric_disk.cpp`**

This is the largest single implementation file. Key sections:

1. **Constructor** — calls `compute_radial_structure()`, `compute_vertical_profiles()`, `normalize_density()`, and `build_opacity_luts()`
2. **`compute_radial_structure()`** — builds T_eff_lut_, H_lut_, rho_mid_lut_ from Novikov-Thorne flux, iterative H with radiation pressure, Shakura-Sunyaev Sigma
3. **`compute_vertical_profiles()`** — solves hydrostatic equilibrium ODE per radial bin using RK4 in z, with Eddington T-τ, Rosseland mean opacity, iterative tau self-consistency
4. **`normalize_density()`** — sets rho_scale_ so midplane tau = tau_mid at peak-flux radius
5. **Accessors** — `density()`, `temperature()`, `scale_height()` use LUT interpolation
6. **4-velocity helpers** — `circular_velocity()`, `plunging_velocity()` from spec Section 4.3

Key formulas (all from spec):

```cpp
// Kerr orbital frequency
double VolumetricDisk::omega_orb(double r) const {
    return std::sqrt(mass_) / (std::pow(r, 1.5) + spin_ * std::sqrt(mass_));
}

// Vertical epicyclic frequency squared
double VolumetricDisk::omega_z_sq(double r) const {
    double Omg = omega_orb(r);
    double a = spin_;
    double r3 = r * r * r;
    return Omg * Omg * (1.0 - 4.0*a*std::sqrt(mass_/r3) + 3.0*a*a/(r*r));
}

// Scale height (frozen inside ISCO)
double VolumetricDisk::scale_height(double r) const {
    return interp_radial(H_lut_, std::max(r, r_isco_));
}

// Taper with spin-scaled width
double VolumetricDisk::taper(double r) const {
    if (r >= r_isco_) return 1.0;
    double dr = r_isco_ - r;
    return std::exp(-(dr * dr) / (taper_width_ * taper_width_));
}
```

The vertical ODE solver is the most complex part:

```cpp
// Per radial bin, solve dP/dz = -rho * Omega_z^2 * z iteratively
// with T(z) from Eddington: T^4 = (3/4)*T_eff^4*(tau_z + 2/3)
// where tau_z uses Rosseland mean + Thomson opacity
```

Implementation length: ~400-500 lines. The full code is too long to inline here — implement following the spec's Section 1.1 procedure exactly (RK4 in z, iterative tau self-consistency, 3-5 iterations until rho(z) converges to 1e-3 relative change).

Add `src/volumetric_disk.cpp` to CMakeLists.txt's `add_library(grrt SHARED ...)` list.

- [ ] **Step 5: Build and run volumetric tests**

```bash
cmake -B build -G "Visual Studio 17 2022" && cmake --build build --config Release --target test-volumetric
./build/Release/test-volumetric.exe
```
Expected: All tests PASS. Construction may take 30-60 seconds (opacity LUT + vertical profiles).

- [ ] **Step 6: Commit**

```bash
git add include/grrt/scene/volumetric_disk.h src/volumetric_disk.cpp tests/test_volumetric.cpp CMakeLists.txt
git commit -m "feat: add volumetric disk with vertical structure, opacity LUTs, and plunging region"
```

---

## Task 7: CPU Raymarching Integration

**Files:**
- Modify: `include/grrt/geodesic/geodesic_tracer.h`
- Modify: `src/geodesic_tracer.cpp`
- Modify: `include/grrt/render/renderer.h`
- Modify: `src/renderer.cpp`

This modifies the existing geodesic tracer to detect disk volume entry and switch to adaptive raymarching with invariant radiative transfer.

- [ ] **Step 1: Add VolumetricDisk to GeodesicTracer interface**

In `include/grrt/geodesic/geodesic_tracer.h`, add:

```cpp
// Forward declaration (at top, before class)
class VolumetricDisk;

// Add to TraceResult:
//   (no changes needed — accumulated_color already holds HDR linear RGB)

// Add to GeodesicTracer class:
//   Constructor: add const VolumetricDisk* vol_disk parameter (default nullptr)
//   New private method: raymarch_volumetric()
//   New member: const VolumetricDisk* vol_disk_;
```

Specifically, modify the constructor signature:

```cpp
GeodesicTracer(const Metric& metric, const Integrator& integrator,
               double observer_r, int max_steps = 10000,
               double r_escape = 1000.0, double tolerance = 1e-8,
               const VolumetricDisk* vol_disk = nullptr);
```

Add private method:

```cpp
private:
    // Volumetric raymarching through disk volume
    void raymarch_volumetric(GeodesicState& state, Vec3& color,
                             const SpectrumLUT* spectrum) const;
    const VolumetricDisk* vol_disk_ = nullptr;
```

- [ ] **Step 2: Modify `src/geodesic_tracer.cpp` — add volume detection**

In the main `trace()` loop, after each adaptive RK4 step, add volume entry detection **before** the existing thin-disk crossing check:

```cpp
// Check for volumetric disk entry
if (vol_disk_) {
    double z = state.position[1] * std::cos(state.position[2]); // r * cos(theta)
    if (vol_disk_->inside_volume(state.position[1], z)) {
        // Switch to raymarching mode
        raymarch_volumetric(state, color, spectrum);
        // After exiting, continue normal integration
        continue;
    }
}
```

The thin-disk crossing check (`if (disk && spectrum)`) should be wrapped in `else if (!vol_disk_)` to avoid double-counting.

- [ ] **Step 3: Implement `raymarch_volumetric()` method**

This is the core raymarching loop implementing spec Sections 4.1 and 5.2:

```cpp
void GeodesicTracer::raymarch_volumetric(GeodesicState& state, Vec3& color,
                                          const SpectrumLUT* spectrum) const {
    using namespace constants;

    const auto& luts = vol_disk_->opacity_luts();

    // Three-channel wavelengths [cm]: B=450nm, G=550nm, R=650nm
    constexpr double lambda_obs[3] = {450e-7, 550e-7, 650e-7};
    constexpr double nu_obs[3] = {c_cgs / 450e-7, c_cgs / 550e-7, c_cgs / 650e-7};

    // Invariant J per channel (initialized to 0 or carried from previous crossing)
    double J[3] = {0.0, 0.0, 0.0};

    // Observer p·u for redshift denominator
    double g_obs_factor = 1.0 / std::sqrt(1.0 - 2.0 * vol_disk_->r_horizon() /*mass is 1*/
                                           ... ); // Compute properly from observer_r_

    double r = state.position[1];
    double ds = vol_disk_->scale_height(r) / 8.0;
    int step_count = 0;
    constexpr int MAX_STEPS = 4096;
    double tau_accumulated[3] = {0.0, 0.0, 0.0};

    while (step_count < MAX_STEPS) {
        // RK4 step (geodesic follows curved path through disk)
        GeodesicState new_state = integrator_.step(metric_, state, ds);
        step_count++;

        r = new_state.position[1];
        double theta = new_state.position[2];
        double phi = new_state.position[3];
        double z = r * std::cos(theta);

        // Exit conditions
        if (!vol_disk_->inside_volume(r, z)) break;
        if (r < vol_disk_->r_horizon()) break;
        bool all_opaque = (tau_accumulated[0] > 10.0 && tau_accumulated[1] > 10.0
                        && tau_accumulated[2] > 10.0);
        if (all_opaque) break;

        // 1. Look up local thermodynamic state
        double rho_cgs = vol_disk_->density_cgs(r, z, phi);
        double T = vol_disk_->temperature(r, std::abs(z));
        if (rho_cgs <= 0.0 || T <= 0.0) { state = new_state; continue; }

        // Turbulent temperature coupling (Section 2.3)
        // rho_smooth = density without noise, rho_turb = density with noise
        double rho_smooth_cgs = vol_disk_->density_cgs(r, z, phi); // includes noise
        // To get smooth density, evaluate without noise (need to factor out)
        // For now, approximate: T_turb = T * (rho_turb/rho_smooth)^beta
        // where beta = (gamma-1)*t_cool/(t_cool+t_turb)
        // Full implementation:
        double H_local = vol_disk_->scale_height(r);
        double kes_local = luts.lookup_kappa_es(rho_cgs, T);
        double kR_local = luts.lookup_kappa_ross(rho_cgs, T);
        double ktot_local = kR_local + kes_local;
        double tau_local = ktot_local * rho_cgs * H_local; // in proper units
        double c_s_sq = constants::k_B * T / (luts.lookup_mu(rho_cgs, T) * constants::m_p);
        double t_cool = (rho_cgs * c_s_sq) / (constants::sigma_SB * T*T*T*T) * tau_local;
        double Omg = vol_disk_->omega_orb(std::max(r, vol_disk_->r_isco()));
        double t_turb = 1.0 / Omg; // in geometric time units — needs L_unit consistency
        double beta_turb = (constants::gamma_gas - 1.0) * t_cool / (t_cool + t_turb);
        // rho_turb/rho_smooth ratio comes from noise amplitude
        // For the first integration pass, T_turb ≈ T (noise modulates density, not T directly)
        double T_turb = T; // Will be refined: need separate smooth density accessor

        // 2. Compute redshift g = (p·u)_emit / (p·u)_obs
        double ut_emit, uphi_emit, ur_emit = 0.0;
        if (r >= vol_disk_->r_isco()) {
            vol_disk_->circular_velocity(r, ut_emit, uphi_emit);
        } else {
            vol_disk_->plunging_velocity(r, theta, ut_emit, ur_emit, uphi_emit);
        }

        // p·u at emitter (covariant momentum · contravariant velocity)
        double p_dot_u_emit = new_state.momentum[0] * ut_emit
                            + new_state.momentum[1] * ur_emit
                            + new_state.momentum[3] * uphi_emit;

        // p·u at observer (static observer)
        double ut_obs = 1.0 / std::sqrt(1.0 - 2.0 / observer_r_);
        double p_dot_u_obs = new_state.momentum[0] * ut_obs;

        double g = p_dot_u_emit / p_dot_u_obs;

        // Proper distance step
        double ds_proper = std::abs(p_dot_u_emit) * std::abs(ds)
                         * (G_cgs * /* M_BH */ 1.0 / (c_cgs * c_cgs));
        // Note: L_unit conversion depends on M_BH; for normalized rendering
        // the rho_scale absorbs this. ds_proper in the opacity*rho*ds product
        // must be consistent with the rho normalization.

        // Per-channel radiative transfer
        for (int ch = 0; ch < 3; ch++) {
            double nu_emit = g * nu_obs[ch];

            // 3. Look up opacity at emitter-frame frequency
            double kabs = luts.lookup_kappa_abs(nu_emit, rho_cgs, T_turb);
            double kes = luts.lookup_kappa_es(rho_cgs, T_turb);
            double ktot = kabs + kes;
            double epsilon = (ktot > 0.0) ? kabs / ktot : 1.0;

            // 4. Optical depth increment
            double dtau = ktot * rho_cgs * ds_proper;
            tau_accumulated[ch] += dtau;

            // 5. Invariant source function: S = epsilon * B_nu(nu_emit, T) / nu_emit^3
            double Bnu = planck_nu(nu_emit, T_turb); // Need to expose or reimplement
            double S = epsilon * Bnu / (nu_emit * nu_emit * nu_emit);

            // 6. Update invariant J
            double exp_dtau = std::exp(-dtau);
            J[ch] = J[ch] * exp_dtau + S * (1.0 - exp_dtau);
        }

        // Adaptive step control (Section 5.2)
        double nu_G_emit = g * nu_obs[1]; // Green channel
        double kabs_G = luts.lookup_kappa_abs(nu_G_emit, rho_cgs, T_turb);
        double kes_G = luts.lookup_kappa_es(rho_cgs, T_turb);
        double dtau_ref = (kabs_G + kes_G) * rho_cgs * ds_proper;

        double ds_tau = ds;
        if (dtau_ref > 0.1) ds_tau = ds * 0.5;
        if (dtau_ref < 0.01) ds_tau = ds * 2.0;

        double ds_geo = 0.1 * std::max(r - vol_disk_->r_horizon(), 0.5);
        ds = std::min(ds_tau, ds_geo);
        double H = vol_disk_->scale_height(r);
        ds = std::clamp(ds, H / 64.0, H);

        state = new_state;
    }

    // 7. Recover observed intensity: I_obs = J * nu_obs^3
    for (int ch = 0; ch < 3; ch++) {
        double I_obs = J[ch] * nu_obs[ch] * nu_obs[ch] * nu_obs[ch];
        color[ch] += I_obs;
    }
}
```

**Important implementation note:** The `planck_nu` function is currently `static` in `src/opacity.cpp`. Either:
- Move it to the header as an inline function, or
- Add a public `planck_nu()` declaration to `opacity.h`

The second option is cleaner. Add to `opacity.h`:
```cpp
// Planck function B_nu(nu, T) [erg/(cm^2 s Hz sr)]
double planck_nu(double nu, double T);
```

And remove the `static` keyword from its definition in `opacity.cpp`.

- [ ] **Step 4: Modify `src/renderer.cpp` and `include/grrt/render/renderer.h`**

Add `const VolumetricDisk*` to the Renderer constructor and pass it through to GeodesicTracer.

In `renderer.h`, add:
```cpp
class VolumetricDisk; // forward declaration
// Modify constructor to accept const VolumetricDisk* vol_disk = nullptr
```

In `renderer.cpp`, store and pass through.

- [ ] **Step 5: Build and verify**

```bash
cmake -B build -G "Visual Studio 17 2022" && cmake --build build --config Release
```
Expected: Clean build. No runtime test yet — this will be tested end-to-end in Task 9.

- [ ] **Step 6: Commit**

```bash
git add include/grrt/geodesic/geodesic_tracer.h src/geodesic_tracer.cpp \
        include/grrt/render/renderer.h src/renderer.cpp \
        include/grrt/color/opacity.h src/opacity.cpp
git commit -m "feat: add CPU volumetric raymarching with invariant radiative transfer"
```

---

## Task 8: CLI and C API Integration

**Files:**
- Modify: `include/grrt/types.h`
- Modify: `src/api.cpp`
- Modify: `cli/main.cpp`

- [ ] **Step 1: Add volumetric params to GRRTParams**

In `include/grrt/types.h`, add after the `disk_temperature` field (around line 41):

```c
    int disk_volumetric;        /* 0 = thin disk (default), 1 = volumetric */
    double disk_alpha;          /* Shakura-Sunyaev viscosity (default 0.1) */
    double disk_turbulence;     /* Noise amplitude (default 0.4) */
    int disk_seed;              /* Noise seed (default 42) */
```

- [ ] **Step 2: Wire VolumetricDisk into api.cpp**

In `src/api.cpp`:
1. Add `#include "grrt/scene/volumetric_disk.h"` at the top
2. Add `std::unique_ptr<grrt::VolumetricDisk> vol_disk;` to GRRTContext
3. In `grrt_create()`, after creating AccretionDisk, add:

```cpp
if (params->disk_enabled && params->disk_volumetric) {
    VolumetricParams vp;
    vp.alpha = params->disk_alpha;
    vp.turbulence = params->disk_turbulence;
    vp.seed = static_cast<uint32_t>(params->disk_seed);
    ctx->vol_disk = std::make_unique<VolumetricDisk>(
        params->mass, params->spin, params->disk_outer,
        params->disk_temperature, vp);
}
```

4. Pass `ctx->vol_disk.get()` to GeodesicTracer and Renderer constructors.

- [ ] **Step 3: Add CLI flags**

In `cli/main.cpp`, add argument parsing for:
- `--disk-volumetric` (flag, sets `params.disk_volumetric = 1`)
- `--disk-alpha` (float, default 0.1)
- `--disk-turbulence` (float, default 0.4)
- `--disk-seed` (int, default 42)

Follow the existing pattern (around lines 60-135) using string comparison.

Also set defaults in the params initialization:
```cpp
params.disk_volumetric = 0;
params.disk_alpha = 0.1;
params.disk_turbulence = 0.4;
params.disk_seed = 42;
```

- [ ] **Step 4: Build and test**

```bash
cmake -B build -G "Visual Studio 17 2022" && cmake --build build --config Release
# Test thin disk still works (regression):
./build/Release/grrt-cli.exe --metric kerr --spin 0.998 --observer-r 50 --output test_thin.hdr
# Test volumetric disk:
./build/Release/grrt-cli.exe --metric kerr --spin 0.998 --observer-r 50 --disk-volumetric --output test_vol.hdr
```

Expected: Both produce images. Thin disk output should be identical to before. Volumetric output should show visible disk thickness (especially at moderate inclination).

- [ ] **Step 5: Commit**

```bash
git add include/grrt/types.h src/api.cpp cli/main.cpp
git commit -m "feat: wire volumetric disk through C API and CLI"
```

---

## Task 9: CUDA Volumetric Disk Device Functions

**Files:**
- Create: `cuda/cuda_volumetric_disk.h`
- Create: `cuda/cuda_noise.h`
- Modify: `cuda/cuda_types.h`

- [ ] **Step 1: Add volumetric fields to `cuda/cuda_types.h`**

Add to `RenderParams`:

```cpp
    // Volumetric disk
    int disk_volumetric;
    double disk_alpha;
    double disk_turbulence;
    double r_isco;
    double r_horizon;
    double taper_width;
    double E_isco, L_isco; // BPT72 plunging conserved quantities

    // LUT grid parameters
    int vol_n_r, vol_n_z;
    double vol_r_min, vol_r_max;
    int opacity_n_nu, opacity_n_rho, opacity_n_T;
    double opacity_log_nu_min, opacity_log_nu_max;
    double opacity_log_rho_min, opacity_log_rho_max;
    double opacity_log_T_min, opacity_log_T_max;
```

- [ ] **Step 2: Create `cuda/cuda_noise.h`**

Port the CPU simplex noise to CUDA `__device__` functions. Same algorithm, but:
- Permutation table in `__constant__` memory (already allocated in the constant memory budget)
- Gradient table as `__constant__` or hardcoded
- All functions are `__device__ inline`

```cpp
#ifndef CUDA_NOISE_H
#define CUDA_NOISE_H

// Permutation table in constant memory (uploaded from host)
extern __constant__ int d_perm[512];

__device__ inline double simplex_noise_3d(double x, double y, double z) {
    // Same algorithm as CPU SimplexNoise3D::evaluate(), using d_perm
    // [Full implementation mirrors src/noise.cpp]
}

__device__ inline double simplex_noise_turbulent(double x, double y, double z) {
    return simplex_noise_3d(x, y, z)
         + 0.5 * simplex_noise_3d(x * 3.0, y * 3.0, z * 3.0);
}

#endif
```

- [ ] **Step 3: Create `cuda/cuda_volumetric_disk.h`**

Device functions that mirror the CPU VolumetricDisk methods, operating on LUTs stored in global/constant memory:

```cpp
#ifndef CUDA_VOLUMETRIC_DISK_H
#define CUDA_VOLUMETRIC_DISK_H

#include "cuda_types.h"
#include "cuda_math.h"
#include "cuda_noise.h"

// LUTs in global memory (uploaded from host)
extern __device__ double* d_H_lut;           // scale height [vol_n_r]
extern __device__ double* d_rho_mid_lut;     // midplane density [vol_n_r]
extern __device__ double* d_rho_profile_lut; // density profile [vol_n_r * vol_n_z]
extern __device__ double* d_T_profile_lut;   // temperature profile [vol_n_r * vol_n_z]
extern __device__ double* d_kappa_abs_lut;   // 3D absorption opacity
extern __device__ double* d_kappa_es_lut;    // 2D Thomson scattering
extern __device__ double* d_kappa_ross_lut;  // 2D Rosseland mean
extern __device__ double* d_mu_lut;          // 2D mean molecular weight

// Interpolation helpers
__device__ inline double vol_interp_radial(const double* lut, double r,
                                            const RenderParams& p);
__device__ inline double vol_interp_2d(const double* lut, double r, double z_abs,
                                        const RenderParams& p);
__device__ inline double vol_lookup_kappa_abs(double nu, double rho, double T,
                                               const RenderParams& p);
__device__ inline double vol_lookup_kappa_es(double rho, double T,
                                              const RenderParams& p);

// Disk geometry
__device__ inline double vol_scale_height(double r, const RenderParams& p);
__device__ inline double vol_taper(double r, const RenderParams& p);
__device__ inline bool vol_inside(double r, double z, const RenderParams& p);
__device__ inline double vol_density_cgs(double r, double z, double phi,
                                          const RenderParams& p);
__device__ inline double vol_temperature(double r, double z, const RenderParams& p);

// 4-velocity for redshift
__device__ inline void vol_circular_velocity(double r, const RenderParams& p,
                                              double& ut, double& uphi);
__device__ inline void vol_plunging_velocity(double r, double theta,
                                              const RenderParams& p,
                                              double& ut, double& ur, double& uphi);

// Planck function B_nu(nu, T)
__device__ inline double vol_planck_nu(double nu, double T);

// Full raymarching loop (called from render kernel when ray enters disk volume)
__device__ inline Vec3 vol_raymarch(/* geodesic state, params, observer info */);

#endif
```

The `vol_raymarch()` function contains the complete raymarching loop (same logic as CPU Task 7, step 3), but using device functions for all LUT lookups and noise evaluation.

- [ ] **Step 4: Build (CUDA)**

```bash
cmake -B build -G "Visual Studio 17 2022" -DGRRT_ENABLE_CUDA=ON
cmake --build build --config Release
```
Expected: Clean build (device functions are header-only, compiled when included by cuda_render.cu).

- [ ] **Step 5: Commit**

```bash
git add cuda/cuda_volumetric_disk.h cuda/cuda_noise.h cuda/cuda_types.h
git commit -m "feat: add CUDA device functions for volumetric disk raymarching"
```

---

## Task 10: CUDA LUT Upload and Kernel Integration

**Files:**
- Modify: `cuda/cuda_render.cu`
- Modify: `cuda/cuda_backend.cu`

- [ ] **Step 1: Add global memory allocations in `cuda_render.cu`**

Add device pointer declarations and upload functions:

```cpp
// Volumetric disk LUTs (global memory)
static double* d_H_lut_ptr = nullptr;
static double* d_rho_mid_lut_ptr = nullptr;
static double* d_rho_profile_ptr = nullptr;
static double* d_T_profile_ptr = nullptr;
static double* d_kappa_abs_ptr = nullptr;
static double* d_kappa_es_ptr = nullptr;
static double* d_kappa_ross_ptr = nullptr;
static double* d_mu_ptr = nullptr;

// Upload functions
void cuda_upload_volumetric_luts(const VolumetricDisk& disk);
void cuda_free_volumetric_luts();
```

- [ ] **Step 2: Implement LUT upload in `cuda_backend.cu`**

In `cuda_render()`, after uploading existing LUTs and before launching the kernel:

```cpp
if (params->disk_volumetric) {
    // Upload permutation table to constant memory
    cudaMemcpyToSymbol(d_perm, vol_disk->noise().permutation_table().data(),
                       512 * sizeof(int));

    // Upload radial LUTs
    cuda_upload_volumetric_luts(*vol_disk);

    // Fill RenderParams volumetric fields
    render_params.disk_volumetric = 1;
    render_params.r_isco = vol_disk->r_isco();
    render_params.r_horizon = vol_disk->r_horizon();
    // ... (all LUT grid parameters)
}
```

- [ ] **Step 3: Modify render kernel in `cuda_render.cu`**

In the main geodesic integration loop (after each adaptive RK4 step), add volume detection and raymarching — same pattern as CPU:

```cpp
if (d_params.disk_volumetric) {
    double z = state.position[1] * cos(state.position[2]);
    if (vol_inside(state.position[1], z, d_params)) {
        Vec3 vol_color = vol_raymarch(state, prev_state, d_params);
        accumulated_color += vol_color;
        // Continue normal integration after exit
        continue;
    }
}
```

The existing thin-disk crossing code is wrapped in `else if (!d_params.disk_volumetric)`.

- [ ] **Step 4: Build and test**

```bash
cmake --build build --config Release
# Test CUDA volumetric:
./build/Release/grrt-cli.exe --metric kerr --spin 0.998 --observer-r 50 \
    --disk-volumetric --backend cuda --output test_vol_cuda.hdr
```

- [ ] **Step 5: Commit**

```bash
git add cuda/cuda_render.cu cuda/cuda_backend.cu
git commit -m "feat: integrate volumetric disk LUT upload and raymarching into CUDA kernel"
```

---

## Task 11: End-to-End Validation

**Files:**
- No new files — uses existing `--validate` CLI mode and visual inspection

- [ ] **Step 1: Thin disk regression test**

Verify that renders without `--disk-volumetric` produce identical output to before:

```bash
./build/Release/grrt-cli.exe --metric kerr --spin 0.998 --observer-r 50 --output regression.hdr
```

Compare pixel-for-pixel with a reference render saved before the volumetric changes (save one now if not already saved).

- [ ] **Step 2: CPU/CUDA validation**

```bash
./build/Release/grrt-cli.exe --metric kerr --spin 0.998 --observer-r 50 \
    --disk-volumetric --validate --output validate_vol.hdr
```

Expected: CPU and CUDA outputs agree within ~1-2% relative error. The noise evaluation may cause slightly higher divergence than thin disk due to FP differences, hence the relaxed tolerance.

- [ ] **Step 3: Visual inspection — face-on vs edge-on**

```bash
# Face-on (theta close to 0)
./build/Release/grrt-cli.exe --metric kerr --spin 0.998 --observer-r 50 \
    --observer-theta 15 --disk-volumetric --output vol_face_on.hdr

# Edge-on
./build/Release/grrt-cli.exe --metric kerr --spin 0.998 --observer-r 50 \
    --observer-theta 80 --disk-volumetric --output vol_edge_on.hdr
```

Expected:
- Face-on: should look similar to thin disk but with slight color variation from turbulence
- Edge-on: should show visible vertical extent, limb effects, self-occlusion

- [ ] **Step 4: Spin sweep**

```bash
for spin in 0.0 0.3 0.6 0.9 0.998; do
    ./build/Release/grrt-cli.exe --metric kerr --spin $spin --observer-r 50 \
        --disk-volumetric --output vol_spin_${spin}.hdr
done
```

Expected: Inner disk visibly puffs up more at higher spin (radiation pressure from hotter inner edge).

- [ ] **Step 5: Hamiltonian conservation check**

During raymarching, the Hamiltonian `H = ½ g^{ab} p_a p_b` should remain below 1e-10. Add a diagnostic print in the raymarching loop (debug build only):

```cpp
#ifndef NDEBUG
    double H_check = 0.5 * metric_.g_upper(state.position).contract(state.momentum)
                         .data[0]; // simplified — compute full contraction
    if (std::abs(H_check) > 1e-10) {
        std::fprintf(stderr, "WARNING: H=%.4e at r=%.4f during raymarch\n",
                     H_check, r);
    }
#endif
```

- [ ] **Step 6: Commit validation scripts/results**

```bash
git add scripts/
git commit -m "test: add volumetric disk validation renders and spin sweep"
```

---

## Task Dependency Graph

```
Task 1 (constants, atomic data)
  └─→ Task 2 (Saha solver)
       └─→ Task 3 (opacity functions)
            └─→ Task 4 (opacity LUT builder)
                 └─→ Task 6 (volumetric disk class)
                      ├─→ Task 7 (CPU raymarching)
                      │    └─→ Task 8 (CLI/API integration)
                      │         └─→ Task 11 (validation)
                      └─→ Task 9 (CUDA device functions)
                           └─→ Task 10 (CUDA upload/kernel)
                                └─→ Task 11 (validation)

Task 5 (simplex noise) ──→ Task 6, Task 9
```

Tasks 5 is independent and can be done in parallel with Tasks 2-4.
Tasks 9-10 (CUDA) can be started after Task 6 completes, in parallel with Tasks 7-8 (CPU).
