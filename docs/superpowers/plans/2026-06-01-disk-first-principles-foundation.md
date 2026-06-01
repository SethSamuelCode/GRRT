# Disk First-Principles Vertical Structure — Foundation (Phases 1–2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the physical length scale `r_g = GM/c²` and accretion anchor into the volumetric disk, and widen the opacity foundation (table range + a derivative supplier), so the Newton-Raphson column BVP (a later plan) has honest cgs lengths and the opacity inputs it needs.

**Architecture:** Two foundation phases of the Approach-A redesign. Phase 1 threads `mass_solar`/`eddington_fraction` (currently discarded in `api.cpp` after deriving `T_peak`) into the disk, and computes/exposes the gravitational radius `r_g` (cm) and the physical accretion rate `Ṁ` (g/s). Phase 2 widens the opacity table to the disk's true density range and adds a swappable opacity-derivative supplier (`κ_R` + log-gradients) for the future Newton Jacobian. These are purely additive — the current (still-collapsed) disk behavior is essentially unchanged; the payoff lands when the BVP plan replaces `solve_column`.

**Tech Stack:** C++23, CMake/MSVC 2022, OpenMP. Standalone test executables with inline `check()`/`failures` macros (no gtest). Geometrized units (G=c=1, M sets the length scale) for geometry; CGS for opacity/thermodynamics (`include/grrt/math/constants.h`).

**Source of truth:** `docs/superpowers/specs/2026-06-01-disk-first-principles-vertical-structure-design.md` (§5 length scale & anchor, §10 opacity table, §8 derivative supplier, §11 interface, §15 phasing).

**Commit workflow:** Per the project workflow (`feedback_review_workflow` memory), **do not run `git commit`.** Each task's final step stages the changes (`git add` is fine) and hands the commit message to the user, who composes the commit.

---

## File structure map

| File | Responsibility | Tasks |
|---|---|---|
| `include/grrt/scene/volumetric_disk.h` | `VolumetricParams` physical-anchor fields; `r_g_`/`mdot_` members + `r_g()`/`mdot()` accessors | 1, 2, 3 |
| `src/volumetric_disk.cpp` | Compute `r_g_`, `mdot_` in the constructor; widen the opacity-build call site | 2, 3, 4 |
| `src/api.cpp` | Thread `mass_solar`/`eddington_fraction` into `VolumetricParams` | 1 |
| `src/opacity.cpp` | `n_rho` resolution; `kappa_ross_with_grad` derivative supplier | 4, 5 |
| `include/grrt/color/opacity.h` | Declare `kappa_ross_with_grad` | 5 |
| `tests/test_volumetric.cpp` | `r_g`, `Ṁ`, opacity-range coverage tests | 2, 3, 4 |
| `tests/test_opacity.cpp` | Derivative-supplier unit test | 5 |

---

## Phase 1 — Length scale & physical anchor

### Task 1: Thread the physical anchor into VolumetricParams

**Files:**
- Modify: `include/grrt/scene/volumetric_disk.h:15-46` (VolumetricParams)
- Modify: `src/api.cpp:104-142` (vp construction)

- [ ] **Step 1: Add the anchor fields to `VolumetricParams`**

`include/grrt/scene/volumetric_disk.h` — in `struct VolumetricParams`, after the `// --- Physical (unchanged) ---` block (after line 22, the `noise_octaves` line), add:

```cpp
    // --- Physical anchor (NEW — Approach A first-principles) ---
    double mass_solar         = 10.0;  ///< Black hole mass [M_sun]; sets r_g = GM/c^2.
    double eddington_fraction = 0.1;   ///< f_Edd; sets Mdot = f_Edd·L_Edd/(η c²).
    double mdot_override      = 0.0;   ///< Direct Mdot [g/s]; 0 = derive from f_Edd.
```

- [ ] **Step 2: Thread them from `api.cpp` (stop discarding `mass_solar`)**

`src/api.cpp` — in the `if (params->disk_volumetric)` block, after the existing conditional `vp.*` assignments (after line 142, the `refine_num_frequencies` block), add:

```cpp
            // Physical anchor (Approach A): pass through so the disk can
            // compute r_g and Mdot. Previously mass_solar was used only for
            // the T_peak derivation above and then discarded.
            if (params->mass_solar > 0.0)
                vp.mass_solar = params->mass_solar;
            if (params->eddington_fraction > 0.0)
                vp.eddington_fraction = params->eddington_fraction;
```

(The conditional assignment preserves the `VolumetricParams` defaults of `10.0`/`0.1` when the caller leaves them unset, exactly like the surrounding `vp.*` pattern.)

- [ ] **Step 3: Build to verify it compiles**

Run:
```powershell
cmake --build build --config Release --target grrt
```
Expected: builds clean. No behavior change yet (the new fields are unused until Task 2/3).

- [ ] **Step 4: Hand the commit message to the user**

```powershell
git add include/grrt/scene/volumetric_disk.h src/api.cpp
```
Commit message for the user:
```
feat(disk): thread physical anchor (mass_solar, f_Edd) into VolumetricParams

mass_solar entered api.cpp only to derive T_peak, then was discarded, so the
disk never learned the physical mass and could not form a length scale.
Add mass_solar/eddington_fraction/mdot_override to VolumetricParams and pass
them through from the C API. Foundation for r_g = GM/c^2 (Approach A).
```

---

### Task 2: Compute and expose the gravitational radius `r_g`

**Files:**
- Modify: `include/grrt/scene/volumetric_disk.h:152-174` (accessors), `:184-192` (members)
- Modify: `src/volumetric_disk.cpp:65-66` (constructor, after `E_isco_` block)
- Test: `tests/test_volumetric.cpp`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_volumetric.cpp` (near the other `test_*` functions, before `main`):

```cpp
void test_gravitational_radius() {
    std::printf("\n=== Length scale r_g = GM/c^2 ===\n");
    const auto& d = shared_disk_default();   // default mass_solar = 10
    // r_g for 10 M_sun = G·(10·M_sun)/c^2
    using namespace grrt::constants;
    const double expected = G_cgs * (10.0 * M_sun) / (c_cgs * c_cgs);
    std::printf("  r_g = %.4e cm (expected %.4e, ~14.8 km)\n", d.r_g(), expected);
    check("r_g", d.r_g(), expected, 1e-9);
    if (d.r_g() < 1e6 || d.r_g() > 2e6) {
        std::printf("  FAIL: r_g out of plausible range for 10 M_sun\n");
        failures++;
    }
}
```

Register it in `main`:
```cpp
    test_gravitational_radius();
```

- [ ] **Step 2: Build and run to verify it fails**

Run:
```powershell
cmake --build build --config Release --target test-volumetric
./build/Release/test-volumetric.exe
```
Expected: compile error — `d.r_g()` does not exist yet (no `r_g()` member).

- [ ] **Step 3: Add the `r_g_` member and accessor**

`include/grrt/scene/volumetric_disk.h:184` — change:
```cpp
    double mass_, spin_, r_outer_, peak_temperature_;
```
to:
```cpp
    double mass_, spin_, r_outer_, peak_temperature_;
    double r_g_ = 0.0;     ///< Gravitational radius GM/c^2 [cm] — geometric→cm length scale
```

And add a public accessor next to `r_isco()`/`r_horizon()` (`include/grrt/scene/volumetric_disk.h:148`, after the `r_horizon()` line):
```cpp
    double r_horizon() const { return r_horizon_; }
```
add directly below:
```cpp
    /// Gravitational radius r_g = G·(mass_solar·M_sun)/c^2 [cm].
    /// Converts a geometric length (units of M) to cm: L_cm = L_geom · r_g.
    double r_g() const { return r_g_; }
```

- [ ] **Step 4: Compute `r_g_` in the constructor**

`src/volumetric_disk.cpp` — immediately after the `E_isco_`/`L_isco_` block closes (after line 65, the `}` that ends the `{ const double v = ... }` scope), add:

```cpp
    // Physical length scale (Approach A): r_g = G·M_phys/c^2 [cm].
    // This is the conversion that makes optical depth honest — geometric
    // lengths (H, z) become cm via × r_g. mass_solar defaults to 10 if unset.
    {
        using namespace constants;
        const double m_solar = (params_.mass_solar > 0.0) ? params_.mass_solar : 10.0;
        r_g_ = G_cgs * (m_solar * M_sun) / (c_cgs * c_cgs);
    }
```

(Confirm `#include "grrt/math/constants.h"` is present in `volumetric_disk.cpp`; the file already uses `constants::` so it is.)

- [ ] **Step 5: Build and run to verify it passes**

Run:
```powershell
cmake --build build --config Release --target test-volumetric
./build/Release/test-volumetric.exe
```
Expected: `test_gravitational_radius` prints `r_g = 1.4771e+06 cm` and PASSes.

- [ ] **Step 6: Hand the commit message to the user**

```powershell
git add include/grrt/scene/volumetric_disk.h src/volumetric_disk.cpp tests/test_volumetric.cpp
```
Commit message for the user:
```
feat(disk): compute and expose gravitational radius r_g = GM/c^2

The keystone of Approach A: the disk now derives r_g [cm] from mass_solar,
giving it the cm length scale it never had. Geometric lengths convert to cm
via × r_g, which is what makes the vertical optical-depth integral
dimensionally honest (used by the upcoming column BVP). Exposed via r_g().
```

---

### Task 3: Compute and expose the physical accretion rate `Ṁ`

**Files:**
- Modify: `include/grrt/scene/volumetric_disk.h:184-192` (member), `:148` (accessor)
- Modify: `src/volumetric_disk.cpp` (constructor, after `r_g_` block)
- Test: `tests/test_volumetric.cpp`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_volumetric.cpp`:

```cpp
void test_accretion_rate() {
    std::printf("\n=== Accretion rate Mdot from f_Edd ===\n");
    const auto& d = shared_disk_default();   // mass_solar=10, f_Edd=0.1 (defaults)
    using namespace grrt::constants;
    // Reference from the same relation, using the disk's own E_isco:
    //   η = 1 − E_isco ;  L_Edd = 4πG M m_p c / σ_T ;  Mdot = f_Edd L_Edd/(η c²)
    const double eta   = 1.0 - d.E_isco();
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * (10.0 * M_sun)
                       * m_p * c_cgs / sigma_T;
    const double mdot_ref = 0.1 * L_Edd / (eta * c_cgs * c_cgs);
    std::printf("  Mdot = %.4e g/s (ref %.4e, eta=%.4f)\n", d.mdot(), mdot_ref, eta);
    check("Mdot vs relation", d.mdot(), mdot_ref, 1e-6);
    // Independent order/value sanity (a=0.998, 10 M_sun, f_Edd=0.1 → ~4.4e17 g/s):
    if (d.mdot() < 3.5e17 || d.mdot() > 5.5e17) {
        std::printf("  FAIL: Mdot outside expected order of magnitude\n");
        failures++;
    }
}
```

Register in `main`:
```cpp
    test_accretion_rate();
```
Ensure `#include <numbers>` is present at the top of `tests/test_volumetric.cpp`; if not, add it.

- [ ] **Step 2: Build and run to verify it fails**

Run:
```powershell
cmake --build build --config Release --target test-volumetric
./build/Release/test-volumetric.exe
```
Expected: compile error — `d.mdot()` does not exist yet.

- [ ] **Step 3: Add the `mdot_` member and accessor**

`include/grrt/scene/volumetric_disk.h` — change the `r_g_` member line (added in Task 2):
```cpp
    double r_g_ = 0.0;     ///< Gravitational radius GM/c^2 [cm] — geometric→cm length scale
```
to:
```cpp
    double r_g_ = 0.0;     ///< Gravitational radius GM/c^2 [cm] — geometric→cm length scale
    double mdot_ = 0.0;    ///< Physical accretion rate [g/s] — from f_Edd or mdot_override
```

And add a public accessor below the `r_g()` accessor added in Task 2:
```cpp
    double r_g() const { return r_g_; }
```
add directly below:
```cpp
    /// Physical accretion rate Mdot [g/s]: f_Edd·L_Edd/(η c²), or mdot_override
    /// if set. η = 1 − E_isco; L_Edd = 4πG M_phys m_p c/σ_T.
    double mdot() const { return mdot_; }
```

- [ ] **Step 4: Compute `mdot_` in the constructor**

`src/volumetric_disk.cpp` — immediately after the `r_g_` block added in Task 2, add:

```cpp
    // Physical accretion rate (Approach A): Mdot = f_Edd·L_Edd/(η c²) [g/s],
    // or a direct override. η = 1 − E_isco (radiative efficiency); L_Edd is the
    // Eddington luminosity. Diagnostic now; anchors Σ in the column BVP later.
    {
        using namespace constants;
        const double m_solar = (params_.mass_solar > 0.0) ? params_.mass_solar : 10.0;
        const double eta = 1.0 - E_isco_;
        const double L_Edd = 4.0 * std::numbers::pi * G_cgs * (m_solar * M_sun)
                           * m_p * c_cgs / sigma_T;
        if (params_.mdot_override > 0.0) {
            mdot_ = params_.mdot_override;
        } else if (eta > 0.0) {
            mdot_ = params_.eddington_fraction * L_Edd / (eta * c_cgs * c_cgs);
        } else {
            mdot_ = 0.0;
        }
    }
```

(Confirm `#include <numbers>` is present in `volumetric_disk.cpp`; the file uses `std::numbers::pi` elsewhere — e.g. line ~510 — so it is.)

- [ ] **Step 5: Build and run to verify it passes**

Run:
```powershell
cmake --build build --config Release --target test-volumetric
./build/Release/test-volumetric.exe
```
Expected: `test_accretion_rate` prints `Mdot = 4.3...e+17 g/s` and PASSes both checks.

- [ ] **Step 6: Hand the commit message to the user**

```powershell
git add include/grrt/scene/volumetric_disk.h src/volumetric_disk.cpp tests/test_volumetric.cpp
```
Commit message for the user:
```
feat(disk): derive and expose physical accretion rate Mdot from f_Edd

Mdot = f_Edd·L_Edd/(η c²) with η = 1 − E_isco and L_Edd = 4πG M m_p c/σ_T,
or a direct mdot_override. Completes the physical accretion anchor; diagnostic
now, will anchor the surface density Σ in the column BVP. Exposed via mdot().
```

---

## Phase 2 — Opacity foundation

### Task 4: Widen the opacity LUT to the disk's real density range

**Files:**
- Modify: `src/opacity.cpp:244` (n_rho)
- Modify: `src/volumetric_disk.cpp:68` (build call site)
- Test: `tests/test_volumetric.cpp`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_volumetric.cpp`:

```cpp
void test_opacity_lut_covers_disk_density() {
    std::printf("\n=== Opacity LUT spans the disk's real density range ===\n");
    const auto& d = shared_disk_default();
    const auto& lut = d.opacity_luts();
    const double rho_max_table = std::pow(10.0, lut.log_rho_max);
    std::printf("  table rho_max = %.3e g/cm^3 (need >= 1e8)\n", rho_max_table);
    if (rho_max_table < 1e8) {
        std::printf("  FAIL: table ceiling below disk density\n");
        failures++;
    }
    // A lookup at a true midplane density (~1e6), fully ionized, must give
    // an electron-scattering-floored Rosseland opacity in a physical range.
    const double kR = lut.lookup_kappa_ross(1e6, 3e6);
    std::printf("  kappa_ross(1e6, 3e6 K) = %.4f cm^2/g (expect finite, > 0)\n", kR);
    if (!(kR > 0.0) || !std::isfinite(kR)) {
        std::printf("  FAIL: kappa_ross at true density non-physical\n");
        failures++;
    }
}
```

Register in `main`:
```cpp
    test_opacity_lut_covers_disk_density();
```

- [ ] **Step 2: Build and run to verify it fails**

Run:
```powershell
cmake --build build --config Release --target test-volumetric
./build/Release/test-volumetric.exe
```
Expected: `test_opacity_lut_covers_disk_density` prints `table rho_max = 1.000e-06` and `FAIL: table ceiling below disk density`.

- [ ] **Step 3: Bump the table resolution in `opacity.cpp`**

`src/opacity.cpp:244` — change:
```cpp
    luts.n_rho = 100;
```
to:
```cpp
    luts.n_rho = 220;   // ~8 bins/decade across the widened [1e-18,1e9] range
```

- [ ] **Step 4: Widen the table range at the disk's call site**

`src/volumetric_disk.cpp:68` — change:
```cpp
    opacity_luts_ = build_opacity_luts(1e-18, 1e-6, 3000.0, 1e8,
                                       params_.opacity_nu_min, params_.opacity_nu_max);
```
to:
```cpp
    opacity_luts_ = build_opacity_luts(1e-18, 1e9, 3000.0, 1e8,
                                       params_.opacity_nu_min, params_.opacity_nu_max);
```

- [ ] **Step 5: Rebuild and run to verify it passes**

Run:
```powershell
cmake --build build --config Release --target test-volumetric
./build/Release/test-volumetric.exe
```
Expected: `test_opacity_lut_covers_disk_density` prints `table rho_max = 1.000e+09`, no FAIL. Note: the current `solve_column` still clamps lookups to `1e-6`, so the rendered disk is essentially unchanged (the widened table only matters once the BVP plan removes those clamps).

- [ ] **Step 6: Hand the commit message to the user**

```powershell
git add src/opacity.cpp src/volumetric_disk.cpp tests/test_volumetric.cpp
```
Commit message for the user:
```
feat(opacity): widen disk opacity LUT to the real density range [1e-18,1e9]

The table spanned only [1e-18,1e-6] g/cm^3, far below the disk's true ~1e6
density. Widen to 1e9 and bump n_rho 100->220 (~8 bins/decade). Additive
foundation for the column BVP; current solve_column still clamps, so rendered
output is unchanged.
```

---

### Task 5: Opacity-derivative supplier (`κ_R` + log-gradients)

The Newton BVP (a later plan) needs `∂κ_R/∂ρ` and `∂κ_R/∂T` for its Jacobian. Provide them now as central finite differences on the log-spaced table, behind one method that the BVP will call — swappable later for a bicubic supplier.

**Files:**
- Modify: `include/grrt/color/opacity.h:42-45` (declare)
- Modify: `src/opacity.cpp:363` (define, after `lookup_mu`)
- Test: `tests/test_opacity.cpp`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_opacity.cpp` (inside its test flow — follow the file's existing structure; it uses `std::printf` and returns a failure count via `main`). Add a function and call it:

```cpp
static int test_kappa_ross_gradients() {
    std::printf("\n=== kappa_ross_with_grad: log-derivatives ===\n");
    int fails = 0;
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    const double rho = 1e-3, T = 1e5;   // a smooth interior point

    double kR, dlnrho, dlnT;
    lut.kappa_ross_with_grad(rho, T, kR, dlnrho, dlnT);

    // kR must match the plain lookup at the same point.
    const double kR_plain = lut.lookup_kappa_ross(rho, T);
    std::printf("  kR=%.4e (plain %.4e)\n", kR, kR_plain);
    if (std::abs(kR - kR_plain) > 1e-9 * std::max(kR_plain, 1e-30)) {
        std::printf("  FAIL: kR disagrees with lookup_kappa_ross\n"); fails++;
    }

    // Independent central difference (different step) must match the supplier's
    // gradients — catches swapped variables / wrong denominator.
    const double h = 0.02;
    const double ref_dlnrho =
        (lut.lookup_kappa_ross(rho * std::exp(h), T)
       - lut.lookup_kappa_ross(rho * std::exp(-h), T)) / (2.0 * h);
    const double ref_dlnT =
        (lut.lookup_kappa_ross(rho, T * std::exp(h))
       - lut.lookup_kappa_ross(rho, T * std::exp(-h))) / (2.0 * h);
    std::printf("  d/dlnrho=%.4e (ref %.4e)  d/dlnT=%.4e (ref %.4e)\n",
                dlnrho, ref_dlnrho, dlnT, ref_dlnT);
    if (std::abs(dlnrho - ref_dlnrho) > 0.10 * std::max(std::abs(ref_dlnrho), 1e-12)
        && std::abs(dlnrho - ref_dlnrho) > 1e-6) {
        std::printf("  FAIL: d/dlnrho mismatch\n"); fails++;
    }
    if (std::abs(dlnT - ref_dlnT) > 0.10 * std::max(std::abs(ref_dlnT), 1e-12)
        && std::abs(dlnT - ref_dlnT) > 1e-6) {
        std::printf("  FAIL: d/dlnT mismatch\n"); fails++;
    }
    if (!std::isfinite(dlnrho) || !std::isfinite(dlnT)) {
        std::printf("  FAIL: non-finite gradient\n"); fails++;
    }
    std::printf("  %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails;
}
```

Wire it into `test_opacity.cpp`'s `main` so its failures count toward the exit code (follow the file's existing aggregation — e.g. add `total_failures += test_kappa_ross_gradients();` alongside the other test calls, or call it and accumulate into whatever variable `main` returns). Ensure `<cmath>` is included (it is, for `std::abs`/`std::isfinite`/`std::exp`).

- [ ] **Step 2: Build and run to verify it fails**

Run:
```powershell
cmake --build build --config Release --target test-opacity
./build/Release/test-opacity.exe
```
Expected: compile error — `kappa_ross_with_grad` is not a member of `OpacityLUTs`.

- [ ] **Step 3: Declare the supplier in the header**

`include/grrt/color/opacity.h` — in `struct OpacityLUTs`, after the `lookup_mu` declaration (line 45):
```cpp
    GRRT_EXPORT double lookup_mu(double rho_cgs, double T) const;
```
add:
```cpp
    /// Rosseland opacity and its logarithmic gradients at (rho_cgs, T), via
    /// central finite differences on the log-spaced table. d/dlnrho and d/dlnT
    /// are dimensionless; convert to dκ/dρ = (d/dlnrho)/rho when needed.
    /// Default opacity-derivative supplier for the column-BVP Newton Jacobian.
    GRRT_EXPORT void kappa_ross_with_grad(double rho_cgs, double T,
        double& kR, double& dkR_dlnrho, double& dkR_dlnT) const;
```

- [ ] **Step 4: Define the supplier in `opacity.cpp`**

`src/opacity.cpp` — after the `lookup_mu` definition (after line 363, the closing `}` of `lookup_mu`), add:

```cpp
void OpacityLUTs::kappa_ross_with_grad(double rho_cgs, double T,
        double& kR, double& dkR_dlnrho, double& dkR_dlnT) const {
    kR = lookup_kappa_ross(rho_cgs, T);
    // Central difference in log-space (the table axes are log-spaced, so a
    // fixed log step gives uniform table resolution). h ≈ 1% in rho/T.
    constexpr double h = 0.01;
    const double kR_rp = lookup_kappa_ross(rho_cgs * std::exp(h),  T);
    const double kR_rm = lookup_kappa_ross(rho_cgs * std::exp(-h), T);
    const double kR_Tp = lookup_kappa_ross(rho_cgs, T * std::exp(h));
    const double kR_Tm = lookup_kappa_ross(rho_cgs, T * std::exp(-h));
    dkR_dlnrho = (kR_rp - kR_rm) / (2.0 * h);
    dkR_dlnT   = (kR_Tp - kR_Tm) / (2.0 * h);
}
```

(Confirm `<cmath>` is included in `opacity.cpp`; the file already uses `std::log10`/`std::pow`, so it is.)

- [ ] **Step 5: Rebuild and run to verify it passes**

Run:
```powershell
cmake --build build --config Release --target test-opacity
./build/Release/test-opacity.exe
```
Expected: `test_kappa_ross_gradients` prints matching `d/dlnrho` and `d/dlnT` vs the references and `PASS`.

- [ ] **Step 6: Hand the commit message to the user**

```powershell
git add include/grrt/color/opacity.h src/opacity.cpp tests/test_opacity.cpp
```
Commit message for the user:
```
feat(opacity): add kappa_ross_with_grad derivative supplier

Provides Rosseland opacity plus its log-gradients (d/dlnrho, d/dlnT) via
central differences on the log-spaced table — the default opacity-derivative
supplier the column-BVP Newton Jacobian will call. Swappable later for a
bicubic supplier without touching the solver.
```

---

## Phase 1–2 wrap: full test sweep

### Task 6: Verify the foundation end-to-end

- [ ] **Step 1: Build everything and run the affected suites**

Run:
```powershell
cmake --build build --config Release
./build/Release/test-volumetric.exe
./build/Release/test-opacity.exe
```
Expected: the four new tests (`test_gravitational_radius`, `test_accretion_rate`, `test_opacity_lut_covers_disk_density`, `test_kappa_ross_gradients`) pass. Pre-existing failures (`test_tau_midplane_near_target`) are unchanged — they are fixed by the later BVP plan, not this foundation. Confirm no *new* failures and no new construction warnings.

- [ ] **Step 2: Confirm rendered output is essentially unchanged (sanity)**

The foundation is additive (the widened table is still clamped by the current `solve_column`). Optionally render a quick frame and confirm the disk looks the same as before this plan:
```powershell
./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 --disk-volumetric --samples 4 --width 128 --height 128 --output foundation_check.png --force
```
Expected: constructs and renders without error; visually unchanged from before (the real change lands with the BVP plan).

- [ ] **Step 3: Report readiness for the BVP plan**

Summarize to the user: `r_g`/`Ṁ` exposed and tested; opacity table widened; derivative supplier in place. The foundation interfaces (`r_g()`, `mdot()`, `kappa_ross_with_grad`) are the concrete signatures the next plan (column BVP) will build against. No new commit (verification only).

---

## Self-review (author checklist — completed)

**Spec coverage (foundation scope = spec §5, §10, §8-supplier, §11, §15 phases 1–2):**
- §5 length scale `r_g=GM/c²` → Task 2. ✓
- §5 `f_Edd`→`Ṁ` anchor (η=1−E_isco, L_Edd) → Task 3. ✓
- §5/§11 `VolumetricParams` anchor fields + `api.cpp` plumbing, default `mass_solar` → Task 1 (+ Task 2/3 default guard). ✓
- §10 opacity range `[1e-18,1e9]` + `n_rho=220` → Task 4. ✓
- §8/§10 swappable opacity-derivative supplier (`∂κ_R/∂ρ,∂κ_R/∂T`) → Task 5. ✓
- §15 phase boundary: BVP (phases 3–4), `normalize_density`/`nested_refine` retirement (phase 5), integration (phase 6) are **out of this plan** by design — next plan. ✓ (Stated in Goal/Architecture.)

**Placeholder scan:** no TBD/TODO; every code step shows the exact edit or full function. The two "follow the file's existing aggregation" notes in Task 5 Step 1 reference a concrete, inspectable mechanism (`test_opacity.cpp`'s `main`), not a vague placeholder. ✓

**Type consistency:** `params_.mass_solar`/`eddington_fraction`/`mdot_override` (Task 1) ↔ used in Task 2/3 constructor. `r_g_`+`r_g()` (Task 2) ↔ test (Task 2). `mdot_`+`mdot()` (Task 3) ↔ test (Task 3). `kappa_ross_with_grad(rho,T,kR,dkR_dlnrho,dkR_dlnT)` signature identical in header (Task 5 S3), definition (S4), and test (S1). ✓
