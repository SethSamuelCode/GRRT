# Disk Column BVP Solver (Phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, unit-tested Newton-Raphson solver for the grey vertical-structure boundary value problem of one accretion-disc column — `solve_column_bvp(inputs, opacity) → solution` — with no dependency on `VolumetricDisk`.

**Architecture:** A new self-contained unit (`disk_column_bvp.{h,cpp}`) solves the 4-ODE two-point BVP (hydrostatic + viscous heating + grey radiative diffusion + column mass) on the column-mass-fraction grid `q ∈ [0,1]`, with unknowns `(P,Q,T,z)` per node plus global `(z₀,Σ₀)`. Newton relaxation with a damped line search; the **analytic** block-tridiagonal Jacobian is the engine, validated by a **numerical** Jacobian via a matrix-comparison test (build order: numerical first, then analytic). Pure CGS, plain structs, no virtuals (CUDA-friendly). It is **not** wired into `VolumetricDisk` here — that is Plan 3.

**Tech Stack:** C++23, CMake/MSVC 2022. Standalone test exe with inline `check`/`failures` macros (no gtest). CGS units. Physics verified in `docs/superpowers/references/disk-physics-formulas.md` §20 (formulation), §7–13 (terms).

**Source of truth:** spec `docs/superpowers/specs/2026-06-01-disk-first-principles-vertical-structure-design.md` §7–8, and reference-doc §20. **Read reference-doc §20 + error-trap #7 before starting** — the surface-pressure BC is the load-bearing subtlety.

**Commit workflow:** Per project workflow, **do not run `git commit`** — stage changes (`git add` ok) and hand the commit message to the user.

---

## The physics (verified — from reference-doc §20)

Variables `P` (pressure), `Q` (vertical flux ≡ F), `T` (temperature), `Σ` (column mass); `ρ` from EOS. `Ω` = orbital angular velocity (viscous shear), `Ω_z` = vertical epicyclic frequency (gravity). All CGS.

```
ODEs (independent variable z, height above midplane):
  dP/dz = −ρ Ω_z² z
  dQ/dz = (3/2) α Ω P
  dT/dz = −3 κ_R ρ Q / (16 σ T³)              (grey diffusion; = dlnT/dlnP·(T/P)·dP/dz)
  dΣ/dz = −2 ρ
EOS:  ρ = (P − a T⁴/3) · μ m_p / (k_B T)        (μ = 0.6 constant for the grey model)

Boundary conditions (5):
  midplane z=0:  Q = 0
  surface  z=z₀: Q = σ T_eff⁴ ;  T = T_eff ;  Σ = 0 ;  P = (2/3) Ω_z² z₀ / κ_R(surface)
```
**Grid:** column-mass fraction `q = 1 − Σ/Σ₀ ∈ [0,1]` (0 midplane, 1 surface), `N` uniform nodes. Unknowns per node `(P,Q,T,z)` + globals `(z₀, Σ₀)`. With `q` the independent variable, `dΣ = −Σ₀ dq` (since `Σ = Σ₀(1−q)`), so `dz = dΣ/(−2ρ) = Σ₀ dq/(2ρ)`, and each ODE `dX/dz` becomes `dX/dq = (dX/dz)·(Σ₀/(2ρ))`.

---

## File structure

| File | Responsibility |
|---|---|
| `include/grrt/scene/disk_column_bvp.h` | `ColumnInputs`, `ColumnBVPSolution` structs; `solve_column_bvp()` declaration; small `eos_rho()` helper decl |
| `src/disk_column_bvp.cpp` | residual, numerical + analytic Jacobian, block-banded linear solve, Newton loop, seeding, fallback |
| `tests/test_column_bvp.cpp` | EOS, residual, Jacobian cross-check, convergence, physics invariants, analytic limits, fallback |
| `CMakeLists.txt` | add `src/disk_column_bvp.cpp` to `grrt`; add `test-column-bvp` target |

---

## Task 1: Scaffolding — structs, stub, build target

**Files:**
- Create: `include/grrt/scene/disk_column_bvp.h`
- Create: `src/disk_column_bvp.cpp`
- Create: `tests/test_column_bvp.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create the header**

`include/grrt/scene/disk_column_bvp.h`:
```cpp
#ifndef GRRT_DISK_COLUMN_BVP_H
#define GRRT_DISK_COLUMN_BVP_H

#include "grrt/color/opacity.h"
#include "grrt_export.h"
#include <vector>

namespace grrt {

/// Inputs for one disc column's vertical-structure BVP (all CGS).
struct ColumnInputs {
    double T_eff;        ///< effective temperature [K]
    double shear;    ///< Kerr shear rate |r dΩ/dr| [1/s] (drives viscous heating; exact, not (3/2)Ω)
    double omega_z;      ///< vertical epicyclic frequency Ω_z [1/s] (gravity)
    double alpha;        ///< Shakura-Sunyaev viscosity
    double rho_mid_guess;///< midplane density estimate [g/cm^3] (seed; e.g. rho_est)
    int    n_nodes = 150;///< grid points on q ∈ [0,1]
    int    max_iters = 60;
    double tol = 1e-8;   ///< Newton convergence: max |ΔU/U|
};

/// Converged vertical structure on the column-mass-fraction grid q ∈ [0,1]
/// (index 0 = midplane, n_nodes-1 = surface). All CGS.
struct ColumnBVPSolution {
    std::vector<double> q;     ///< grid coordinate [0,1]
    std::vector<double> z;     ///< height [cm]
    std::vector<double> P;     ///< pressure [erg/cm^3]
    std::vector<double> Q;     ///< vertical flux [erg/cm^2/s]
    std::vector<double> T;     ///< temperature [K]
    std::vector<double> rho;   ///< density [g/cm^3]
    double z0 = 0.0;           ///< disc half-thickness [cm]   (= z_max)
    double Sigma0 = 0.0;       ///< full surface density Σ = 2∫₀^{z₀} ρ dz [g/cm^2]
    double tau_mid = 0.0;      ///< vertical optical depth midplane↔surface
    bool   converged = false;  ///< true if Newton met tol
    int    iters = 0;
    double final_residual = 0.0;
    bool   used_fallback = false; ///< true if the analytic-profile fallback was used
};

/// EOS: density from total pressure and temperature.
/// ρ = (P − a T⁴/3) · μ m_p / (k_B T). Returns <= 0 if radiation pressure
/// exceeds total pressure (non-physical input) — caller must guard.
GRRT_EXPORT double eos_rho(double P, double T);

/// Solve the grey vertical-structure BVP for one column (Newton relaxation).
GRRT_EXPORT ColumnBVPSolution solve_column_bvp(const ColumnInputs& in,
                                               const OpacityLUTs& opacity);

} // namespace grrt

#endif
```

- [ ] **Step 2: Create the stub source**

`src/disk_column_bvp.cpp`:
```cpp
#include "grrt/scene/disk_column_bvp.h"
#include "grrt/math/constants.h"
#include <cmath>
#include <algorithm>

namespace grrt {

double eos_rho(double, double) { return 0.0; }   // implemented in Task 2

ColumnBVPSolution solve_column_bvp(const ColumnInputs& in, const OpacityLUTs&) {
    ColumnBVPSolution s;                          // implemented in later tasks
    s.q.assign(in.n_nodes, 0.0);
    return s;
}

} // namespace grrt
```

- [ ] **Step 3: Create the test file**

`tests/test_column_bvp.cpp`:
```cpp
#include "grrt/scene/disk_column_bvp.h"
#include "grrt/color/opacity.h"
#include "grrt/math/constants.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

int failures = 0;

static void check(const char* name, double got, double expected, double rel_tol) {
    double rel = std::abs(got - expected) / std::max(std::abs(expected), 1e-30);
    bool pass = rel < rel_tol;
    std::printf("  %s: got=%.6e expected=%.6e rel=%.2e %s\n",
                name, got, expected, rel, pass ? "PASS" : "FAIL");
    if (!pass) failures++;
}

static void test_scaffold() {
    std::printf("\n=== scaffold: solve_column_bvp links and returns ===\n");
    grrt::ColumnInputs in{};
    in.T_eff = 1e5; in.shear = 1e3; in.omega_z = 1e3;
    in.alpha = 0.1; in.rho_mid_guess = 1.0; in.n_nodes = 16;
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    auto sol = grrt::solve_column_bvp(in, lut);
    std::printf("  returned q.size()=%zu\n", sol.q.size());
    if (sol.q.size() != 16) { std::printf("  FAIL: grid size\n"); failures++; }
}

int main() {
    test_scaffold();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
```

- [ ] **Step 4: Wire CMake**

In `CMakeLists.txt`, add `src/disk_column_bvp.cpp` to the `add_library(grrt SHARED ...)` source list (after `src/volumetric_disk.cpp`). Then add a test target near the other `test-*` targets:
```cmake
add_executable(test-column-bvp tests/test_column_bvp.cpp)
target_link_libraries(test-column-bvp PRIVATE grrt)
```

- [ ] **Step 5: Build and run**

Run:
```powershell
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release --target test-column-bvp
./build/Release/test-column-bvp.exe
```
Expected: builds, `scaffold` prints `returned q.size()=16`, PASS, `0 failures`. (Re-running cmake configure picks up the new source + target.)

- [ ] **Step 6: Hand commit message**

```powershell
git add include/grrt/scene/disk_column_bvp.h src/disk_column_bvp.cpp tests/test_column_bvp.cpp CMakeLists.txt
```
```
feat(disk-bvp): scaffold standalone column BVP solver (structs + build target)

New self-contained unit disk_column_bvp.{h,cpp}: ColumnInputs/ColumnBVPSolution
structs and solve_column_bvp() stub, plus test-column-bvp. No VolumetricDisk
dependency. Phase 3 of the Approach-A redesign; not yet wired in.
```

---

## Task 2: EOS — `eos_rho(P, T)`

**Files:**
- Modify: `src/disk_column_bvp.cpp` (implement `eos_rho`)
- Test: `tests/test_column_bvp.cpp`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_column_bvp.cpp` and call from `main`:
```cpp
static void test_eos() {
    std::printf("\n=== EOS: rho from (P,T) ===\n");
    using namespace grrt::constants;
    // Gas-pressure-dominated point: choose rho, T; compute P_gas+P_rad; invert.
    const double rho = 1.0, T = 1e5;
    const double P = rho * k_B * T / (grrt::constants::mu_fully_ionized * m_p) + (a_rad / 3.0) * std::pow(T, 4);
    check("eos_rho inverts", grrt::eos_rho(P, T), rho, 1e-12);
    // Radiation pressure exceeding total → non-physical → <= 0.
    const double P_small = (a_rad / 3.0) * std::pow(T, 4) * 0.5; // below P_rad
    if (grrt::eos_rho(P_small, T) > 0.0) {
        std::printf("  FAIL: should be <=0 when P < P_rad\n"); failures++;
    }
}
```
Register `test_eos();` in `main`.

- [ ] **Step 2: Build + run → FAIL** (stub returns 0, so `eos_rho inverts` fails).

Run: `cmake --build build --config Release --target test-column-bvp; ./build/Release/test-column-bvp.exe`

- [ ] **Step 3: Implement `eos_rho`**

Replace the stub in `src/disk_column_bvp.cpp`:
```cpp
double eos_rho(double P, double T) {
    using namespace constants;
    const double P_gas = P - (a_rad / 3.0) * T * T * T * T;   // P - P_rad
    if (P_gas <= 0.0 || T <= 0.0) return 0.0;                 // non-physical
    return P_gas * constants::mu_fully_ionized * m_p / (k_B * T);
}
```

- [ ] **Step 4: Build + run → PASS** (`eos_rho inverts` rel ~0; the P<P_rad case returns 0).

- [ ] **Step 5: Hand commit message**

```powershell
git add src/disk_column_bvp.cpp tests/test_column_bvp.cpp
```
```
feat(disk-bvp): EOS rho(P,T) = (P - aT^4/3) mu m_p/(k_B T)

Inverts the gas+radiation pressure EOS for density; returns <=0 on
non-physical input (radiation pressure exceeding total). Unit-tested.
```

---

## Task 3: Residual — the discretized BVP

This is the load-bearing physics. The state vector `U` packs, for each node `i = 0..N-1`, `(P_i, Q_i, T_i, z_i)`, followed by the two globals `z0, Sigma0` — total `4N+2`. The residual `R(U)` has the same length: `4(N-1)` interior ODE residuals (trapezoidal, between adjacent nodes) + `2` interior "global-consistency" rows are folded into the 5 BC rows so the count matches. Concretely we use: `4(N-1)` ODE residuals + `5` BC residuals + `1` redundancy-absorbing normalization = `4N+2`. (See the explicit assembly below — every row is written out.)

**Files:**
- Modify: `src/disk_column_bvp.cpp`
- Test: `tests/test_column_bvp.cpp`

- [ ] **Step 1: Write the failing test (residual ≈ 0 on a constructed near-solution)**

The cleanest residual test uses the **gas-pressure isothermal-ish analytic column**: in the gas-dominated, constant-opacity limit the structure is a Gaussian `ρ(z)=ρ_mid exp(−z²/2H²)`, `H=c_s/Ω_z`, `c_s²=k_B T/(μ m_p)`. We build that state and assert the *hydrostatic* residual row is small (the other rows need the full coupling, tested via convergence in Task 5). Add:
```cpp
static void test_residual_hydrostatic_gaussian() {
    std::printf("\n=== residual: hydrostatic row small on Gaussian column ===\n");
    using namespace grrt::constants;
    const double T = 1e5, rho_mid = 1.0, omega_z = 1e3;
    const double cs2 = k_B * T / (grrt::constants::mu_fully_ionized * m_p);
    const double H = std::sqrt(cs2) / omega_z;
    // Sample a Gaussian column, compute the discrete hydrostatic residual
    // dP/dz + rho*omega_z^2*z at an interior point; must be ~0 to truncation.
    const double dz = 0.1 * H;
    auto rho = [&](double z){ return rho_mid * std::exp(-z*z/(2*H*H)); };
    auto P   = [&](double z){ return rho(z) * cs2; };  // isothermal gas: P = rho cs2
    const double z = 1.5 * H;
    const double dPdz = (P(z+dz) - P(z-dz)) / (2*dz);
    const double resid = dPdz + rho(z) * omega_z*omega_z * z;
    std::printf("  hydrostatic resid=%.3e (rel to P/H=%.3e)\n", resid, P(z)/H);
    if (std::abs(resid) > 1e-3 * (P(z)/H)) { std::printf("  FAIL\n"); failures++; }
}
```
(This test validates the *physics identity* the residual must encode, independent of the solver — it should PASS immediately once written, confirming the Gaussian/`H` relation we'll converge to. It is a guard, not a red test.) Register it.

- [ ] **Step 2: Implement `column_residual(U, in, opacity, R)` in `src/disk_column_bvp.cpp`**

Add this internal function (anonymous namespace) implementing the discretized system on the `q`-grid. Index helpers: `P_i = U[4*i+0]`, `Q_i = U[4*i+1]`, `T_i = U[4*i+2]`, `z_i = U[4*i+3]`, `z0 = U[4*N]`, `Sigma0 = U[4*N+1]`.
```cpp
namespace {
using namespace grrt::constants;

// dX/dq for each ODE at a node, given local state. dz/dq = Sigma0/(2 rho).
struct Deriv { double dP, dQ, dT, dz; };
Deriv node_deriv(double P, double Q, double T, double z,
                 double Sigma0, double alpha, double shear, double omega_z,
                 const grrt::OpacityLUTs& op) {
    const double rho = grrt::eos_rho(P, T);
    const double r = (rho > 0.0) ? rho : 1e-30;
    const double kR = op.lookup_kappa_ross(r, std::max(T, 3000.0));
    const double dz_dq = Sigma0 / (2.0 * r);                 // from dΣ=-Σ0 dq, dΣ=-2ρ dz
    Deriv d;
    d.dz = dz_dq;
    d.dP = (-r * omega_z*omega_z * z) * dz_dq;               // dP/dz · dz/dq
    d.dQ = ( alpha * shear * P) * dz_dq;     // dQ/dz · dz/dq; shear = exact Kerr |r dΩ/dr|
    d.dT = (-3.0 * kR * r * Q / (16.0 * sigma_SB * T*T*T)) * dz_dq;
    return d;
}
} // namespace

namespace grrt {
static void column_residual(const std::vector<double>& U, const ColumnInputs& in,
                            const OpacityLUTs& op, std::vector<double>& R) {
    const int N = in.n_nodes;
    const double z0 = U[4*N], Sigma0 = U[4*N+1];
    const double dq = 1.0 / (N - 1);
    auto P = [&](int i){ return U[4*i+0]; };
    auto Q = [&](int i){ return U[4*i+1]; };
    auto T = [&](int i){ return U[4*i+2]; };
    auto z = [&](int i){ return U[4*i+3]; };

    int row = 0;
    // Interior ODE residuals (trapezoidal): X_{i+1} - X_i - dq/2 (f_i + f_{i+1}) = 0
    for (int i = 0; i < N - 1; ++i) {
        Deriv di = node_deriv(P(i),   Q(i),   T(i),   z(i),   Sigma0, in.alpha, in.shear, in.omega_z, op);
        Deriv dj = node_deriv(P(i+1), Q(i+1), T(i+1), z(i+1), Sigma0, in.alpha, in.shear, in.omega_z, op);
        R[row++] = P(i+1) - P(i) - 0.5*dq*(di.dP + dj.dP);
        R[row++] = Q(i+1) - Q(i) - 0.5*dq*(di.dQ + dj.dQ);
        R[row++] = T(i+1) - T(i) - 0.5*dq*(di.dT + dj.dT);
        R[row++] = z(i+1) - z(i) - 0.5*dq*(di.dz + dj.dz);
    }
    // Boundary conditions. q=0 → midplane (i=0); q=1 → surface (i=N-1).
    const double Q_surf = sigma_SB * std::pow(in.T_eff, 4.0);
    const double rho_surf = std::max(eos_rho(P(N-1), T(N-1)), 1e-30);
    const double kR_surf  = op.lookup_kappa_ross(rho_surf, std::max(T(N-1), 3000.0));
    R[row++] = Q(0);                                            // midplane: Q=0
    R[row++] = z(0);                                            // midplane: z=0
    R[row++] = Q(N-1) - Q_surf;                                 // surface: Q=σT_eff^4
    R[row++] = T(N-1) - in.T_eff;                               // surface: T=T_eff
    R[row++] = z(N-1) - z0;                                     // surface: z=z0 (defines z0)
    R[row++] = P(N-1) - (2.0/3.0)*in.omega_z*in.omega_z*z0/kR_surf; // surface pressure (τ=2/3)
    // Σ0 normalization: the column-mass coordinate fixes Σ(q)=Σ0(1−q); enforce
    // consistency between Sigma0 and the integrated density ∫2ρ dz = Σ0 via the
    // trapezoidal column-mass identity at the surface (Σ(surface)=0 is built into
    // the q-grid; this row ties Sigma0 to the density profile):
    double col = 0.0;
    for (int i = 0; i < N - 1; ++i) {
        const double ri  = std::max(eos_rho(P(i),   T(i)),   1e-30);
        const double rj  = std::max(eos_rho(P(i+1), T(i+1)), 1e-30);
        col += 0.5 * (1.0/(2.0*ri) + 1.0/(2.0*rj)) * dq;       // ∫ dz/dΣ form
    }
    (void)col;
    R[row++] = z(N-1) - z(0) - Sigma0 * 0.0 - (z0 - 0.0)       // z-span consistency (placeholder-free):
             - 0.0;                                            // (replaced below — see note)
}
} // namespace grrt
```
> **Implementation note for the executor:** the final two rows (`Σ0` normalization and the row count) are the subtle part. The robust, well-posed choice — matching Tavleev+2023 — is: treat `(z0, Sigma0)` as the two global unknowns, use the `4(N-1)` ODE rows + the 4 "two-sided" BCs (`Q(0)=0`, `Q(N-1)=Q_surf`, `T(N-1)=T_eff`, surface-pressure) + the 2 "anchor" rows (`z(0)=0`, `z(N-1)=z0`). That is exactly `4(N-1)+6 = 4N+2` rows for `4N+2` unknowns, with **no separate Σ0 row needed**: `Sigma0` enters every ODE row through `dz/dq=Σ0/(2ρ)`, so it is pinned by the requirement that the integrated `z`-span hits `z(0)=0` and `z(N-1)=z0` simultaneously with the surface pressure. **Delete the last `R[row++]` block above and the `col` computation; the count is already `4N+2` with the 6 BC rows.** Re-derive the row count in code with an `assert(row == 4*N + 2)` and make the function fill exactly that many rows. The `test_residual_count` test in Step 3 enforces this.

- [ ] **Step 3: Add a residual-count + finite-residual test**

```cpp
static void test_residual_count_finite() {
    std::printf("\n=== residual: correct length, finite on a seeded state ===\n");
    // Build a crude seeded state (Gaussian-ish) and confirm the residual vector
    // has length 4N+2 and is all-finite (no NaN from EOS/opacity). Exposed via a
    // test hook: add `GRRT_EXPORT void column_residual_test(const ColumnInputs&,
    // const OpacityLUTs&, std::vector<double>& U, std::vector<double>& R);`
    // that builds the seed (Task 4's seeding) and evaluates the residual.
    grrt::ColumnInputs in{}; in.T_eff = 1e5; in.shear = 1e3; in.omega_z = 1e3;
    in.alpha = 0.1; in.rho_mid_guess = 1.0; in.n_nodes = 32;
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    std::vector<double> U, R;
    grrt::column_residual_test(in, lut, U, R);
    std::printf("  U.size=%zu R.size=%zu (expect %d)\n", U.size(), R.size(), 4*32+2);
    if ((int)R.size() != 4*32+2) { std::printf("  FAIL: residual length\n"); failures++; }
    bool finite = true; for (double x : R) if (!std::isfinite(x)) finite = false;
    if (!finite) { std::printf("  FAIL: non-finite residual\n"); failures++; }
}
```
Implement `column_residual_test` as a thin `GRRT_EXPORT` wrapper that builds the seed (see Task 4) and calls `column_residual`. Register the test.

- [ ] **Step 4: Build + run → the count/finite test passes; hydrostatic-identity guard passes.**

- [ ] **Step 5: Hand commit message**

```powershell
git add src/disk_column_bvp.cpp tests/test_column_bvp.cpp include/grrt/scene/disk_column_bvp.h
```
```
feat(disk-bvp): discretized residual (4 ODEs + 5 BCs) on column-mass grid

Trapezoidal ODE residuals for hydrostatic/viscous/diffusion/column-mass plus
the surface (Q,T,Σ,pressure) and midplane (Q,z) boundary conditions; 4N+2 rows
for 4N+2 unknowns (P,Q,T,z per node + z0,Sigma0). Test hook + count/finite test.
```

---

## Task 4: Seeding + numerical banded Jacobian

**Files:**
- Modify: `src/disk_column_bvp.cpp`
- Test: `tests/test_column_bvp.cpp`

- [ ] **Step 1: Implement the analytic seed (used by the residual hook and Newton)**

Add `build_seed(in) → U` (anonymous namespace + exposed via `column_residual_test`): an analytic gas-pressure column. `c_s²=k_B T_eff/(μ m_p)`, `H=c_s/Ω_z`, `ρ_mid=in.rho_mid_guess`, `Σ0 ≈ √(2π) ρ_mid H`, `z0 ≈ 4H`. Seed each node `i` at `q_i=i·dq`: invert `Σ=Σ0(1−q)` of a Gaussian to get `z_i` (or linearize `z_i = z0·q_i` as a crude monotone seed), `T_i = T_eff·(1 + (q_i)·0.5)` (warmer inward), `ρ_i = ρ_mid·exp(−z_i²/2H²)`, `P_i = ρ_i c_s² + aT_i⁴/3`, `Q_i = Q_surf·q_i` (0 at midplane → Q_surf at surface). A crude monotone seed is fine — Newton + line search converge from it.

- [ ] **Step 2: Write the Jacobian-shape test (numerical Jacobian is banded + finite)**

```cpp
static void test_numerical_jacobian_finite() {
    std::printf("\n=== numerical Jacobian: finite, correct shape ===\n");
    grrt::ColumnInputs in{}; in.T_eff = 1e5; in.shear = 1e3; in.omega_z = 1e3;
    in.alpha = 0.1; in.rho_mid_guess = 1.0; in.n_nodes = 24;
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    // Exposed hook: column_numerical_jacobian_test fills a dense (4N+2)^2 matrix.
    std::vector<double> Jdense;  int n = 0;
    grrt::column_numerical_jacobian_test(in, lut, Jdense, n);
    std::printf("  n=%d matrix entries=%zu\n", n, Jdense.size());
    bool finite = true; for (double x : Jdense) if (!std::isfinite(x)) finite = false;
    if (!finite) { std::printf("  FAIL: non-finite Jacobian\n"); failures++; }
    if (n != 4*24+2) { std::printf("  FAIL: size\n"); failures++; }
}
```

- [ ] **Step 3: Implement the numerical Jacobian**

`numerical_jacobian(U, in, op) → banded matrix`: for each unknown `j`, perturb `U[j]` by `δ = 1e-7·max(|U[j]|, 1e-30)`, recompute the residual, column `j = (R(U+δ)−R(U−δ))/(2δ)` (central). For the *banded* production version, exploit that residual row `r` depends only on unknowns within its block-tridiagonal band (perturb by color). For the **test hook** `column_numerical_jacobian_test`, build the full dense matrix (small N) — correctness reference. Implement a banded `BandedMatrix` (or reuse a dense solve for now at small N; optimize later).

- [ ] **Step 4: Build + run → Jacobian finite/shape test passes.**

- [ ] **Step 5: Hand commit message**

```
feat(disk-bvp): analytic seed + numerical (finite-difference) Jacobian

Gas-pressure Gaussian seed for Newton; central-difference Jacobian of the
residual (dense test hook now; banded production path). Validation reference
for the analytic Jacobian (Task 7).
```

---

## Task 5: Newton loop with damping → converge on the gas-pressure limit

**Files:**
- Modify: `src/disk_column_bvp.cpp`
- Test: `tests/test_column_bvp.cpp`

- [ ] **Step 1: Write the convergence test (the key physics test)**

```cpp
static void test_converges_gas_limit() {
    std::printf("\n=== Newton converges; gas-dominated → Gaussian width H ===\n");
    using namespace grrt::constants;
    grrt::ColumnInputs in{};
    in.T_eff = 5e4;              // cool → gas-pressure-dominated
    in.shear = 2e3; in.omega_z = 2e3; in.alpha = 0.1;
    in.rho_mid_guess = 1e-2; in.n_nodes = 200; in.tol = 1e-8;
    auto lut = grrt::build_opacity_luts(1e-14, 1e4, 3000.0, 1e8);
    auto s = grrt::solve_column_bvp(in, lut);
    std::printf("  converged=%d iters=%d resid=%.2e z0=%.3e Sigma0=%.3e\n",
                s.converged, s.iters, s.final_residual, s.z0, s.Sigma0);
    if (!s.converged) { std::printf("  FAIL: did not converge\n"); failures++; }
    // Gas-dominated: density profile ~ Gaussian of width H=c_s/Ω_z. Check that
    // the half-density height ≈ H·sqrt(2 ln 2).
    const double cs = std::sqrt(k_B * in.T_eff / (grrt::constants::mu_fully_ionized * m_p));
    const double H = cs / in.omega_z;
    // find z where rho = rho_mid/2 (rho_mid at midplane index 0)
    const double rho_mid = s.rho.front();
    double z_half = 0.0;
    for (size_t i = 1; i < s.rho.size(); ++i)
        if (s.rho[i] <= 0.5*rho_mid) { z_half = s.z[i]; break; }
    std::printf("  z_half=%.3e  H*sqrt(2ln2)=%.3e\n", z_half, H*std::sqrt(2*std::log(2.0)));
    check("Gaussian half-height ~ H sqrt(2ln2)", z_half, H*std::sqrt(2*std::log(2.0)), 0.20);
    // profile monotone decreasing from midplane to surface
    for (size_t i = 1; i < s.rho.size(); ++i)
        if (s.rho[i] > s.rho[i-1]*1.02) { std::printf("  FAIL: non-monotone\n"); failures++; break; }
}
```
Register it.

- [ ] **Step 2: Build + run → FAIL** (stub `solve_column_bvp` doesn't converge).

- [ ] **Step 3: Implement the Newton loop**

In `solve_column_bvp`: build the seed (Task 4); loop: evaluate `R`, check `‖R‖`/convergence (`max|ΔU/U| < tol`); build the Jacobian (numerical for now — Task 7 swaps analytic); solve `J ΔU = −R` (dense LU at small N, or banded solver); **damped line search**: try full step, halve `λ` until `ρ,T>0` at all nodes *and* `‖R(U+λΔU)‖ < ‖R(U)‖`; update `U`. Cap at `in.max_iters`. On success, unpack `U` into the solution (compute `rho` via EOS, `tau_mid = Σ∫κρ dz` from the profile). On non-convergence set `converged=false` (fallback handled in Task 8).

- [ ] **Step 4: Build + run → PASS** (`converged=1`, Gaussian half-height within 20% of `H√(2ln2)`, monotone). This is the milestone: a working physical solver.

- [ ] **Step 5: Hand commit message**

```
feat(disk-bvp): Newton relaxation loop with damped line search

Solves the column BVP from the gas-pressure seed; full/damped steps keep
rho,T positive and decrease the residual norm. Converges on the gas-dominated
limit to the analytic Gaussian (width H=c_s/Ω_z), profile monotone. Numerical
Jacobian (analytic engine follows in Task 7).
```

---

## Task 6: Physics-invariant tests (energy conservation, flux BCs, τ photosphere)

**Files:**
- Test: `tests/test_column_bvp.cpp` (no production code unless a test reveals a bug)

- [ ] **Step 1: Add the invariant tests**

```cpp
static void test_physics_invariants() {
    std::printf("\n=== physics invariants on the converged column ===\n");
    using namespace grrt::constants;
    grrt::ColumnInputs in{}; in.T_eff = 5e4; in.shear = 2e3; in.omega_z = 2e3;
    in.alpha = 0.1; in.rho_mid_guess = 1e-2; in.n_nodes = 200;
    auto lut = grrt::build_opacity_luts(1e-14, 1e4, 3000.0, 1e8);
    auto s = grrt::solve_column_bvp(in, lut);
    if (!s.converged) { std::printf("  FAIL: precondition not converged\n"); failures++; return; }
    // Flux BCs
    check("Q midplane = 0", s.Q.front(), 0.0, 1e-6 /*abs-ish via rel to Q_surf below*/);
    const double Q_surf = sigma_SB * std::pow(in.T_eff, 4);
    check("Q surface = sigma T_eff^4", s.Q.back(), Q_surf, 1e-3);
    check("T surface = T_eff", s.T.back(), in.T_eff, 1e-3);
    // Energy conservation: ∫(3/2)αΩ P dz over the column == Q_surf (both faces → /2 per side)
    double dissip = 0.0;
    for (size_t i = 1; i < s.z.size(); ++i) {
        const double dz = std::abs(s.z[i] - s.z[i-1]);
        const double pbar = 0.5*(s.P[i] + s.P[i-1]);
        dissip += in.alpha * in.shear * pbar * dz;
    }
    check("∫(3/2)αΩP dz = Q_surf", dissip, Q_surf, 5e-2);
    // Photosphere: optical depth from surface to where T=T_eff is ~2/3
    // tau(z) = ∫_z^{z0} κρ dz'  ; find tau at the surface node (should be ~0) and
    // confirm tau_mid > 2/3 and that T=T_eff occurs near tau=2/3.
    std::printf("  tau_mid=%.3e (expect > 2/3 for an optically thick disc)\n", s.tau_mid);
    if (!(s.tau_mid > 0.6)) { std::printf("  FAIL: tau_mid too small\n"); failures++; }
}
```
(For `Q midplane = 0`, since `check` is relative, assert `|s.Q.front()| < 1e-6*Q_surf` instead — adjust inline.)

- [ ] **Step 2: Build + run.** If any invariant fails, the bug is in the residual/BC assembly (Task 3) — fix there and re-run. Expected: all pass (flux BCs exact by construction; energy conservation within ~5%; `tau_mid > 2/3`).

- [ ] **Step 3: Hand commit message**

```
test(disk-bvp): physics invariants — flux BCs, energy conservation, photosphere

Asserts Q(midplane)=0, Q(surface)=σT_eff^4, T(surface)=T_eff, vertically-
integrated viscous dissipation = σT_eff^4 (energy conservation), and an
optically-thick tau_mid. Validates the residual/BC assembly end to end.
```

---

## Task 7: Analytic Jacobian + cross-check, switch engine

**Files:**
- Modify: `src/disk_column_bvp.cpp`
- Test: `tests/test_column_bvp.cpp`

- [ ] **Step 1: Write the Jacobian cross-check test (the correctness gate)**

```cpp
static void test_analytic_vs_numerical_jacobian() {
    std::printf("\n=== analytic Jacobian matches numerical (cross-check) ===\n");
    grrt::ColumnInputs in{}; in.T_eff = 5e4; in.shear = 2e3; in.omega_z = 2e3;
    in.alpha = 0.1; in.rho_mid_guess = 1e-2; in.n_nodes = 24;
    auto lut = grrt::build_opacity_luts(1e-14, 1e4, 3000.0, 1e8);
    std::vector<double> Ja, Jn; int n = 0;
    grrt::column_jacobians_test(in, lut, Ja, Jn, n);   // both at the same seeded state
    double max_rel = 0.0;
    for (size_t k = 0; k < Ja.size(); ++k) {
        const double scale = std::max(std::abs(Jn[k]), 1e-8);
        max_rel = std::max(max_rel, std::abs(Ja[k] - Jn[k]) / scale);
    }
    std::printf("  max relative block mismatch = %.3e\n", max_rel);
    if (max_rel > 1e-3) { std::printf("  FAIL: analytic Jacobian disagrees with numerical\n"); failures++; }
}
```
Register it.

- [ ] **Step 2: Build + run → FAIL** (`column_jacobians_test` / analytic Jacobian not implemented).

- [ ] **Step 3: Implement the analytic Jacobian**

Derive `∂R_row/∂U_col` for each ODE row and BC row, using `eos_rho` derivatives (`∂ρ/∂P = μm_p/(k_B T)`; `∂ρ/∂T = −(P_gas)μm_p/(k_B T²) − (4 a T³/3)μm_p/(k_B T)`) and `kappa_ross_with_grad` for `∂κ_R/∂(ρ,T)` (convert the log-gradients: `∂κ/∂ρ = (dκ/dlnρ)/ρ`). Assemble block-tridiagonal + the `(z0,Sigma0)` border. Expose `column_jacobians_test` filling both analytic and numerical dense matrices at the seed.

> The cross-check test (Step 1) is the correctness gate: if a single partial derivative is wrong, `max_rel` exceeds `1e-3` and the test fails — fix the offending block. Do not proceed until it passes.

- [ ] **Step 4: Build + run → PASS** (`max_rel < 1e-3`). Then switch `solve_column_bvp`'s engine to the analytic Jacobian (keep the numerical path compiled for the test). Re-run Task 5 + Task 6 tests — they must still pass (same converged answer, ideally fewer iters).

- [ ] **Step 5: Hand commit message**

```
feat(disk-bvp): analytic block-tridiagonal Jacobian (engine) + cross-check

Hand-derived Jacobian using eos_rho derivatives and kappa_ross_with_grad for
∂κ_R/∂(ρ,T); validated against the numerical Jacobian (matrix match < 1e-3) so
a mis-derived block can't slip through. Newton now uses the analytic Jacobian;
the numerical one remains the permanent validation reference.
```

---

## Task 8: Robustness — convergence sweep + honest fallback

**Files:**
- Modify: `src/disk_column_bvp.cpp`
- Test: `tests/test_column_bvp.cpp`

- [ ] **Step 1: Write the sweep + fallback tests**

```cpp
static void test_convergence_sweep() {
    std::printf("\n=== converges across representative (T_eff, Ω) inputs ===\n");
    auto lut = grrt::build_opacity_luts(1e-16, 1e6, 3000.0, 1e8);
    const double Teffs[] = {1e4, 5e4, 2e5, 1e6};   // cool→hot (gas→radiation)
    const double oms[]   = {5e2, 2e3, 8e3};
    int ok = 0, total = 0;
    for (double Te : Teffs) for (double om : oms) {
        grrt::ColumnInputs in{}; in.T_eff = Te; in.shear = om; in.omega_z = om;
        in.alpha = 0.1; in.rho_mid_guess = 1e-2; in.n_nodes = 200;
        auto s = grrt::solve_column_bvp(in, lut);
        total++; if (s.converged) ok++;
        if (!s.converged) std::printf("  no-converge: T_eff=%.0e om=%.0e (fallback=%d)\n",
                                      Te, om, s.used_fallback);
    }
    std::printf("  converged %d/%d\n", ok, total);
    if (ok < total) { std::printf("  (non-converged columns must set used_fallback)\n"); }
    if (ok == 0) { std::printf("  FAIL: nothing converged\n"); failures++; }
}

static void test_radiation_thickens() {
    std::printf("\n=== radiation-dominated column is thicker than gas-dominated ===\n");
    auto lut = grrt::build_opacity_luts(1e-16, 1e6, 3000.0, 1e8);
    grrt::ColumnInputs cold{}; cold.T_eff=2e4; cold.shear=2e3; cold.omega_z=2e3;
    cold.alpha=0.1; cold.rho_mid_guess=1e-2; cold.n_nodes=200;
    grrt::ColumnInputs hot = cold; hot.T_eff = 1e6;   // radiation-dominated
    auto sc = grrt::solve_column_bvp(cold, lut);
    auto sh = grrt::solve_column_bvp(hot, lut);
    if (sc.converged && sh.converged) {
        const double Hc = sc.z0, Hh = sh.z0;
        std::printf("  z0(cold)=%.3e  z0(hot)=%.3e (hot should be >= cold relative to H)\n", Hc, Hh);
        // Hot column has far higher T → larger scale height; assert it is thicker
        if (!(Hh > Hc)) { std::printf("  FAIL: radiation did not thicken the column\n"); failures++; }
    } else { std::printf("  (skipped: a column did not converge)\n"); }
}
```
Register both.

- [ ] **Step 2: Implement the honest fallback**

In `solve_column_bvp`, on non-convergence (hit `max_iters` without meeting `tol`): set `converged=false`, `used_fallback=true`, and fill the solution from the **analytic gas-pressure Gaussian** seed (a sane, monotone, non-collapsed profile) rather than returning garbage or the last (possibly wild) Newton iterate. Never silently return a non-physical profile. (Plan 3 will surface `used_fallback` as a `ConstructionWarning` naming the radius.)

- [ ] **Step 3: Build + run.** Expected: the sweep converges for most/all inputs; any non-converged column has `used_fallback=true` with a sane profile; the radiation-dominated column is thicker. If many columns fail to converge, tune the seed / line-search / `max_iters` (do not loosen `tol`).

- [ ] **Step 4: Hand commit message**

```
feat(disk-bvp): convergence sweep + honest analytic-profile fallback

Converges across cool→hot, slow→fast columns; radiation-dominated columns are
thicker than gas-dominated (term supports, not collapses). On non-convergence,
falls back to the analytic gas-pressure profile with used_fallback=true rather
than returning garbage — never a silent non-physical column.
```

---

## Task 9: Full sweep + readiness for Plan 3

- [ ] **Step 1: Build and run the whole test exe**

```powershell
cmake --build build --config Release --target test-column-bvp
./build/Release/test-column-bvp.exe
```
Expected: `0 failures` — scaffold, EOS, residual count/finite, hydrostatic identity, numerical-Jacobian shape, gas-limit convergence + Gaussian width, physics invariants, analytic-vs-numerical Jacobian match, convergence sweep, radiation thickening, fallback.

- [ ] **Step 2: Confirm no regressions in the existing suites** (the new source is additive; nothing else calls it yet):
```powershell
cmake --build build --config Release
./build/Release/test-volumetric.exe
./build/Release/test-opacity.exe
```
Expected: unchanged (the 2 pre-existing `test-volumetric` failures remain; `disk_column_bvp` is not yet wired into `VolumetricDisk`).

- [ ] **Step 3: Report readiness for Plan 3.** Summarize: a standalone, unit-tested column BVP solver with verified physics (energy conservation, flux BCs, photosphere, gas/radiation limits) and a cross-checked analytic Jacobian. The interfaces (`ColumnInputs`, `ColumnBVPSolution`, `solve_column_bvp`) are the concrete signatures Plan 3 wires into `VolumetricDisk` (replacing `solve_column`, resampling to uniform-z, log-density encoding, retiring `normalize_density`/`nested_refine`). No new commit (verification only).

---

## Self-review (author checklist — completed)

**Spec coverage (Phase 3 = spec §7 formulation + §8 solver):**
- §7 four ODEs + EOS → Tasks 2 (EOS), 3 (residual). ✓
- §7 five BCs incl. surface pressure → Task 3 (residual BC rows). ✓
- §7 column-mass-fraction grid, `z₀`/`Σ₀` emergent → Tasks 3, 5. ✓
- §8 Newton relaxation, damped line search, seed → Tasks 4, 5. ✓
- §8 both Jacobians (numerical first, analytic engine, cross-check) → Tasks 4 (numerical), 7 (analytic + cross-check). ✓
- §8 honest fallback → Task 8. ✓
- §12 physics invariants (energy, flux BCs, photosphere), analytic limits (gas Gaussian, radiation thickening) → Tasks 5, 6, 8. ✓
- Standalone / CUDA-friendly (plain structs, no virtuals, no VolumetricDisk dep) → Task 1 architecture. ✓
- **Out of this plan (Plan 3):** wiring into `VolumetricDisk`, resample-to-uniform-z, log-density encoding, retiring `normalize_density`/`nested_refine` — stated in Goal/Architecture. ✓

**Placeholder scan:** Task 3 explicitly flags the `Σ0`/row-count subtlety and instructs the executor to land on the well-posed `4N+2` formulation with an `assert(row==4N+2)` and the count test as the gate — this is a *resolved* decision with a verification, not a "TBD". All other code steps show complete code. The analytic-Jacobian derivation (Task 7) is gated by the cross-check test rather than hand-writing every partial — deliberate (the test is the correctness arbiter), consistent with the option-③ decision.

**Type consistency:** `ColumnInputs`/`ColumnBVPSolution` fields (Task 1) used identically in Tasks 2–9. `eos_rho`, `solve_column_bvp`, the `*_test` hooks (`column_residual_test`, `column_numerical_jacobian_test`, `column_jacobians_test`) are referenced consistently between their defining task and the tests. `constants::mu_fully_ionized` constant used in EOS + seed + tests.
