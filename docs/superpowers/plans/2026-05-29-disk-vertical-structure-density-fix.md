# Disk Vertical-Structure Density Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un-collapse the volumetric disk's vertical density profile (currently a delta-spike at z=0 in 4056/4096 columns) by making the column solver and the global normalization agree on one physically-derived absolute density scale, eliminating the edge-on banding.

**Architecture:** The collapse is caused by a non-physical density clamp. `solve_column` divides its radiation-pressure term by a reference density `rho_cgs_ref` that is clamped to a `1e-6 g/cm³` ceiling — ~12 orders below the disk's true ~`1e6` midplane density — inflating the radiation term ~`1e12×` so the hydrostatic ODE collapses every column. The clamp exists because the opacity LUT only spans `[1e-18, 1e-6]`. The fix: (1) widen the opacity LUT to the disk's real density range; (2) solve `rho_cgs_ref` self-consistently with no clamps; (3) drop the matching clamps in `normalize_density` and `compute_sigma_s_phys` so they use the true density; (4) verify the radiation term's sign/formulation. The committed raymarch fixes are **not** touched.

**Tech Stack:** C++23, CMake/MSVC 2022, OpenMP. Standalone test executables with inline `check`/`failures` macros (no gtest). Geometrized units (G=c=1, M sets length scale); opacity/temperature in CGS.

**Source of truth:** `docs/superpowers/specs/2026-05-29-disk-vertical-structure-density-fix-design.md` (Approach B). Approach A (full coupled-ODE BVP atmosphere) is a documented follow-up, out of scope here.

---

## Design decisions locked for this plan

The spec (§4.2) deferred three choices to plan time. They are resolved here; do not re-litigate during execution:

1. **Opacity LUT range → `[1e-18, 1e9] g/cm³`.** The disk's true midplane density is ~`1e6`–`1e7`; `1e9` brackets it with ~2 decades of margin. `rho_min = 1e-18` (photosphere floor) is unchanged.
2. **Resolution → `n_rho = 220`.** Widening from 12 to 27 decades at the old `n_rho = 100` drops to ~3.7 bins/decade; `220` restores ~8 bins/decade (the original density). `n_T`/`n_nu` unchanged.
3. **`rho_cgs_ref` solve → dimensionless τ_mid proxy with fixed-point κ, no clamps.** Keep the existing `rho_ref = τ_mid / (κ · 3H)` *form* (it does not require the not-yet-known global `rho_scale_`), but evaluate κ at the actual `rho_ref` (fixed point, ~5 iterations) at `T_mid`, drop the `max(κ,1.0)` floor and the `[1e-18,1e-6]` ceiling. The `3H` factor is the column half-depth proxy (`C·H`, `C=3`, comparable to the Gaussian integral `√(2π)·H ≈ 2.5H`). We do **not** introduce a geometric→cm length conversion; the absolute calibration is carried by the global `rho_scale_` in `normalize_density`, and the two scales are required only to agree to **order of magnitude** (the original bug was 12 orders apart). This keeps the change surgical and avoids a units-system rewrite (that is Approach A territory).

**Why order-of-magnitude agreement, not exact:** `solve_column` uses a `3H` depth proxy; `normalize_density` uses the true `∫ρ_norm dz` integral. They differ by the shape factor by construction. Forcing exact equality would either break τ_mid enforcement or require the two-pass self-consistent loop deferred to Approach A. Same-order agreement (within ~10×) proves the 12-order bug is gone, which is the actual requirement.

---

## File-by-file change map

| File | Change | Task |
|---|---|---|
| `src/opacity.cpp:244` | `luts.n_rho = 100;` → `220;` | 1 |
| `src/volumetric_disk.cpp:68` | `build_opacity_luts(1e-18, 1e-6, ...)` → `(1e-18, 1e9, ...)` | 1 |
| `include/grrt/scene/volumetric_disk.h:217-222` | add `double rho_cgs_ref = 0.0;` to `ColumnSolution` | 2 |
| `include/grrt/scene/volumetric_disk.h:204` | add member `std::vector<double> rho_cgs_ref_lut_;` | 2 |
| `include/grrt/scene/volumetric_disk.h:~174` | add accessor `reference_density_lut()` | 2 |
| `src/volumetric_disk.cpp:630-640` | replace clamped `rho_cgs_ref` with self-consistent fixed-point; move `T_mid` up | 2 |
| `src/volumetric_disk.cpp:663-703` | drop internal `std::clamp(rho_*,1e-18,1e-6)` in tau/radiation passes | 2 |
| `src/volumetric_disk.cpp:908-913` | store `col.rho_cgs_ref` into `rho_cgs_ref_lut_` | 2 |
| `src/volumetric_disk.cpp:957-981` | seed from column ref; drop `[1e-18,1e-6]` clamps | 3 |
| `src/volumetric_disk.cpp:1015` | drop `rho_mid_cgs` clamp (β/σ_s now sees true density) | 3 |
| `tests/test_volumetric.cpp` | new vertical-structure tests | 2, 3 |
| `tools/dump_disk_lut.cpp` | unchanged (diagnostic; used for verification) | 5 |

**Not touched:** `src/geodesic_tracer.cpp` (committed uniform-fine raymarch + parked `GRRT_RM_LOG`); the spectral path clamp at `volumetric_disk.cpp:1164` (spec §10, out of scope); the `GRRT_COL_LOG` instrumentation at `solve_column` (kept env-gated for Task 4 verification).

---

## Commit workflow

Per the project workflow (`feedback_review_workflow` memory): **do not run `git commit`.** Each task's final step prepares the staged changes and the commit message, then hands the message to the user, who composes the commit. The `git add` is fine to run; the commit text is for the user.

---

### Task 1: Widen the opacity LUT to the disk's real density range

**Files:**
- Modify: `src/opacity.cpp:244`
- Modify: `src/volumetric_disk.cpp:68`
- Test: `tests/test_volumetric.cpp` (new `test_opacity_lut_covers_disk_density`)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_volumetric.cpp` (near the other `test_*` functions, before `main`):

```cpp
void test_opacity_lut_covers_disk_density() {
    std::printf("\n=== Opacity LUT spans the disk's real density range ===\n");
    const auto& d = shared_disk_no_noise();
    const auto& lut = d.opacity_luts();
    const double rho_max_table = std::pow(10.0, lut.log_rho_max);
    std::printf("  table rho_max = %.3e g/cm^3 (need >= 1e8)\n", rho_max_table);
    if (rho_max_table < 1e8) {
        std::printf("  FAIL: table ceiling below disk density\n");
        failures++;
    }
    // A lookup at a true midplane density (~1e6) must NOT clamp to a wrong
    // edge: kappa_es at 1e6 g/cm^3, fully ionized, ~ electron scattering 0.2-0.4.
    const double kes = lut.lookup_kappa_es(1e6, 3e6);
    std::printf("  kappa_es(1e6, 3e6 K) = %.4f cm^2/g (expect ~0.2-0.4)\n", kes);
    if (!(kes > 0.05 && kes < 1.0)) {
        std::printf("  FAIL: kappa_es at true density out of physical range\n");
        failures++;
    }
}
```

Register it in `main` alongside the other calls:

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

- [ ] **Step 3: Widen the table resolution in `opacity.cpp`**

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

- [ ] **Step 5: Rebuild and run to verify the test passes**

Run:
```powershell
cmake --build build --config Release --target test-volumetric
./build/Release/test-volumetric.exe
```
Expected: `test_opacity_lut_covers_disk_density` prints `table rho_max = 1.000e+09` and the `kappa_es` line in `~0.2-0.4`, no FAIL. (The pre-existing `test_no_horizontal_bands` and `test_tau_midplane_near_target` may still fail — they are fixed by later tasks.)

- [ ] **Step 6: Hand the commit message to the user**

```powershell
git add src/opacity.cpp src/volumetric_disk.cpp tests/test_volumetric.cpp
```
Commit message for the user:
```
fix(disk): widen opacity LUT to the disk's real density range

The opacity table only spanned [1e-18, 1e-6] g/cm^3, forcing every
lookup at the disk's true ~1e6 density to clamp to the table edge.
Widen to [1e-18, 1e9] and bump n_rho 100->220 to keep ~8 bins/decade.
Prerequisite for removing the non-physical density clamps that collapse
the vertical profile.
```

---

### Task 2: Self-consistent per-column reference density in `solve_column`

This is the core fix: it un-collapses the vertical profile. After Task 1 the table can represent the true density; now make `solve_column` actually use it.

**Files:**
- Modify: `include/grrt/scene/volumetric_disk.h` (ColumnSolution field, member LUT, accessor)
- Modify: `src/volumetric_disk.cpp` (reference-density solve; drop internal clamps; store ref into LUT)
- Test: `tests/test_volumetric.cpp` (new `test_vertical_profile_not_collapsed`, `test_vertical_profile_monotone`)

- [ ] **Step 1: Write the failing tests**

Add a small helper and two tests to `tests/test_volumetric.cpp`:

```cpp
// Nearest radial-LUT index to a target radius r.
static int ri_at(const grrt::VolumetricDisk& d, double r) {
    const int n = d.radial_bins();
    const double r0 = d.r_min(), r1 = d.r_max();
    int ri = static_cast<int>(std::lround((r - r0) / (r1 - r0) * (n - 1)));
    return std::clamp(ri, 0, n - 1);
}

void test_vertical_profile_not_collapsed() {
    std::printf("\n=== Vertical profile is not a delta spike ===\n");
    const auto& d = shared_disk_no_noise();
    const int nz = d.vertical_bins();
    const int ri = ri_at(d, 8.0);
    const auto& prof = d.density_profile_lut();   // [ri*nz + zi], rho_z[0]=1
    const int zi_10 = std::clamp(static_cast<int>(std::lround(0.10 * (nz - 1))), 1, nz - 1);
    const int zi_50 = std::clamp(static_cast<int>(std::lround(0.50 * (nz - 1))), 1, nz - 1);
    const double p_mid = prof[ri * nz + 0];
    const double p_10  = prof[ri * nz + zi_10];
    const double p_50  = prof[ri * nz + zi_50];
    std::printf("  r=8: mid=%.3e  0.1*zmax=%.3e  0.5*zmax=%.3e\n", p_mid, p_10, p_50);
    if (p_mid < 0.99)   { std::printf("  FAIL: midplane not normalized to 1\n"); failures++; }
    if (p_10 <= 1e-3)   { std::printf("  FAIL: collapsed at 0.1*zmax\n");        failures++; }
    if (p_50 <= 1e-6)   { std::printf("  FAIL: collapsed at 0.5*zmax\n");        failures++; }
}

void test_vertical_profile_monotone() {
    std::printf("\n=== Vertical profile is monotone non-increasing in |z| ===\n");
    const auto& d = shared_disk_no_noise();
    const int nz = d.vertical_bins();
    const int ri = ri_at(d, 8.0);
    const auto& prof = d.density_profile_lut();
    bool ok = true;
    for (int zi = 1; zi < nz; ++zi) {
        const double a = prof[ri * nz + zi - 1];
        const double b = prof[ri * nz + zi];
        if (b > a * 1.05) {   // allow 5% for interpolation noise
            std::printf("  FAIL: non-monotone at zi=%d: %.3e -> %.3e\n", zi, a, b);
            failures++; ok = false; break;
        }
    }
    if (ok) std::printf("  monotone over %d vertical bins: PASS\n", nz);
}
```

Register both in `main`:
```cpp
    test_vertical_profile_not_collapsed();
    test_vertical_profile_monotone();
```

- [ ] **Step 2: Build and run to verify they fail**

Run:
```powershell
cmake --build build --config Release --target test-volumetric
./build/Release/test-volumetric.exe
```
Expected: `test_vertical_profile_not_collapsed` prints `0.1*zmax` and `0.5*zmax` ≈ `1.000e-18` and FAILs both collapse checks. (`monotone` may pass trivially — a spike-then-floor is technically non-increasing — so it is a guard, not the red bar.)

- [ ] **Step 3: Add the `rho_cgs_ref` field to `ColumnSolution`**

`include/grrt/scene/volumetric_disk.h:217-222` — change:
```cpp
    struct ColumnSolution {
        double z_max = 0.0;
        std::vector<double> rho_z;   // size n_z, normalized so rho_z[0] = 1
        std::vector<double> T_z;     // size n_z, in Kelvin
        double max_delta = 0.0;  ///< Final iteration-to-iteration relative density delta
    };
```
to:
```cpp
    struct ColumnSolution {
        double z_max = 0.0;
        std::vector<double> rho_z;   // size n_z, normalized so rho_z[0] = 1
        std::vector<double> T_z;     // size n_z, in Kelvin
        double max_delta = 0.0;  ///< Final iteration-to-iteration relative density delta
        double rho_cgs_ref = 0.0;  ///< Self-consistent absolute reference density [g/cm^3]
    };
```

- [ ] **Step 4: Add the member LUT and accessor**

`include/grrt/scene/volumetric_disk.h:204` — after the `z_max_lut_` line:
```cpp
    std::vector<double> z_max_lut_;       ///< atmosphere extent z_max(r) [geometric]
```
add:
```cpp
    std::vector<double> rho_cgs_ref_lut_; ///< per-column reference density [g/cm^3]
```

And add a public accessor near the existing `z_max_lut()` accessor (`include/grrt/scene/volumetric_disk.h:174`):
```cpp
    const std::vector<double>& z_max_lut() const { return z_max_lut_; }
```
add directly below:
```cpp
    /// Self-consistent absolute reference density per radial column [g/cm^3].
    const std::vector<double>& reference_density_lut() const { return rho_cgs_ref_lut_; }
```

- [ ] **Step 5: Replace the clamped `rho_cgs_ref` with a self-consistent fixed point**

`src/volumetric_disk.cpp:630-640` — replace this block:
```cpp
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
```
with:
```cpp
    // Midplane temperature from the grey Eddington relation (needed for the
    // reference-density solve below).
    const double T_mid4 = 0.75 * T_eff * T_eff * T_eff * T_eff
                         * (params_.tau_mid + 2.0/3.0);
    const double T_mid = std::pow(T_mid4, 0.25);
    const double T_mid_k = std::clamp(T_mid, 3000.0, 1e8);

    // Self-consistent reference (absolute) density for this column.
    // Solve  tau_mid = kappa(rho_ref, T_mid) * rho_ref * (3H)  for rho_ref,
    // where 3H is the column half-depth proxy (~ the Gaussian integral
    // sqrt(2*pi)*H ~ 2.5H). kappa depends on rho_ref, so iterate to a fixed
    // point. NO non-physical floor or ceiling: kappa's physical floor is
    // electron scattering (in the opacity table) and the table now spans the
    // disk's true density range, so lookup_kappa_* clamp only at the (wide)
    // table edges. This replaces code that pinned rho_ref to a 1e-6 ceiling
    // ~12 orders below reality, inflating the radiation-pressure term ~1e12x
    // and collapsing the column to a delta spike.
    double rho_cgs_ref = params_.tau_mid / (0.34 * 3.0 * H);   // seed: kappa ~ kappa_es
    for (int it = 0; it < 5; ++it) {
        const double kR = opacity_luts_.lookup_kappa_ross(rho_cgs_ref, T_mid_k);
        const double kE = opacity_luts_.lookup_kappa_es(rho_cgs_ref, T_mid_k);
        const double kappa = kR + kE;
        if (kappa <= 0.0 || !std::isfinite(kappa)) break;
        const double next = params_.tau_mid / (kappa * 3.0 * H);
        const bool converged = std::abs(next - rho_cgs_ref) <= 1e-3 * rho_cgs_ref;
        rho_cgs_ref = next;
        if (converged) break;
    }
    out.rho_cgs_ref = rho_cgs_ref;
```

> Note: `rho_cgs_ref` is now a non-`const` local but is effectively constant after the loop; the `rhs` lambda's `[&]` capture (line ~743) is unaffected.

- [ ] **Step 6: Drop the internal density clamps in the tau and radiation passes**

These lookups pin density to the old `1e-6` ceiling; `lookup_kappa_*`/`lookup_mu` already clamp to the (now wide) table edges internally (`opacity.cpp:221`).

`src/volumetric_disk.cpp:662-673` (Pass 1, tau) — change each `std::clamp(rho_*_cgs, 1e-18, 1e-6)` to the bare variable:
```cpp
            const double kR_h = opacity_luts_.lookup_kappa_ross(
                rho_h_cgs, std::clamp(out.T_z[zi], 3000.0, 1e8));
            const double kE_h = opacity_luts_.lookup_kappa_es(
                rho_h_cgs, std::clamp(out.T_z[zi], 3000.0, 1e8));
            const double kR_n = opacity_luts_.lookup_kappa_ross(
                rho_n_cgs, std::clamp(out.T_z[zi+1], 3000.0, 1e8));
            const double kE_n = opacity_luts_.lookup_kappa_es(
                rho_n_cgs, std::clamp(out.T_z[zi+1], 3000.0, 1e8));
```

`src/volumetric_disk.cpp:688-691` (Pass 3, mu) — change:
```cpp
            const double rho_cgs = out.rho_z[zi] * rho_cgs_ref;
            mu_z[zi] = opacity_luts_.lookup_mu(
                rho_cgs, std::clamp(out.T_z[zi], 3000.0, 1e8));
```

`src/volumetric_disk.cpp:701-704` (Pass 3, kR for flux limiter) — change:
```cpp
            const double rho_cgs = out.rho_z[zi] * rho_cgs_ref;
            const double kR = opacity_luts_.lookup_kappa_ross(
                rho_cgs, std::clamp(out.T_z[zi], 3000.0, 1e8));
```

(Leave all `std::clamp(out.T_z[...], 3000.0, 1e8)` temperature clamps — those bound to the legitimate T-table range.)

- [ ] **Step 7: Store the per-column reference density into the LUT**

`src/volumetric_disk.cpp:890-892` (top of `compute_vertical_profiles`) — change:
```cpp
void VolumetricDisk::compute_vertical_profiles() {
    z_max_lut_.resize(n_r_);
    rho_profile_lut_.resize(n_r_ * n_z_, 0.0);
    T_profile_lut_.resize(n_r_ * n_z_, 0.0);
```
to:
```cpp
void VolumetricDisk::compute_vertical_profiles() {
    z_max_lut_.resize(n_r_);
    rho_cgs_ref_lut_.assign(n_r_, 0.0);
    rho_profile_lut_.resize(n_r_ * n_z_, 0.0);
    T_profile_lut_.resize(n_r_ * n_z_, 0.0);
```

And `src/volumetric_disk.cpp:908` — change:
```cpp
        z_max_lut_[ri] = col.z_max;
```
to:
```cpp
        z_max_lut_[ri] = col.z_max;
        rho_cgs_ref_lut_[ri] = col.rho_cgs_ref;
```

- [ ] **Step 8: Rebuild and run to verify the tests pass**

Run:
```powershell
cmake --build build --config Release --target test-volumetric
./build/Release/test-volumetric.exe
```
Expected: `test_vertical_profile_not_collapsed` prints `0.1*zmax` ≈ O(0.5–0.9) and `0.5*zmax` ≈ O(0.1–0.5) (a real Gaussian-like profile), no FAIL; `test_vertical_profile_monotone` PASS. Construction log should still complete in a few minutes.

- [ ] **Step 9: Hand the commit message to the user**

```powershell
git add include/grrt/scene/volumetric_disk.h src/volumetric_disk.cpp tests/test_volumetric.cpp
```
Commit message for the user:
```
fix(disk): self-consistent reference density un-collapses vertical profile

solve_column divided its radiation-pressure term by rho_cgs_ref, which
was clamped to a 1e-6 g/cm^3 ceiling ~12 orders below the disk's true
~1e6 density, inflating the term ~1e12x so every column collapsed to a
delta spike at z=0 (4056/4096 columns floored). Solve rho_cgs_ref by
fixed-point on tau_mid = kappa(rho,T)*rho*3H with no floor/ceiling, and
drop the matching 1e-6 density clamps in the tau and radiation passes.
The opacity table (widened in the prior commit) now represents the true
density. Expose the per-column reference density for normalization
reconciliation. Vertical profile is now a physical gas-pressure profile.
```

---

### Task 3: Reconcile `normalize_density` and `compute_sigma_s_phys` with the true density

With the profile un-collapsed, the global normalization's `col_integral` is now physical, but its lookups still clamp to `1e-6`. Drop those clamps so the global scale and the β/σ_s pressure-regime detection see the true density, and assert the two density scales now agree to order of magnitude.

**Files:**
- Modify: `src/volumetric_disk.cpp` (`normalize_density`, `compute_sigma_s_phys`)
- Test: `tests/test_volumetric.cpp` (new `test_reference_density_agreement`, `test_tau_mid_recovered`)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_volumetric.cpp`:

```cpp
// Peak-flux radial index in the orbiting region (mirrors normalize_density).
static int peak_ri(const grrt::VolumetricDisk& d) {
    const int n = d.radial_bins();
    const double r0 = d.r_min(), r1 = d.r_max();
    const auto& rho_mid = d.rho_mid_lut();
    int idx = 0; double best = 0.0;
    for (int i = 0; i < n; ++i) {
        const double r = r0 + (r1 - r0) * i / (n - 1);
        if (r >= d.r_isco() && rho_mid[i] > best) { best = rho_mid[i]; idx = i; }
    }
    return idx;
}

void test_reference_density_agreement() {
    std::printf("\n=== Column ref density agrees with global scale (same order) ===\n");
    const auto& d = shared_disk_no_noise();
    const int pk = peak_ri(d);
    const double global = d.rho_scale() * d.rho_mid_lut()[pk];   // absolute g/cm^3
    const double col    = d.reference_density_lut()[pk];         // absolute g/cm^3
    const double ratio  = (col > 0.0) ? global / col : 0.0;
    std::printf("  rho_scale*peak=%.3e  col_ref=%.3e  ratio=%.2f\n", global, col, ratio);
    // The two use different depth factors (3H proxy vs the true integral), so
    // they need only agree to order of magnitude. The original bug was 12
    // orders apart; same-order proves it is gone.
    if (!(ratio > 0.1 && ratio < 10.0)) {
        std::printf("  FAIL: density scales differ by more than 10x\n");
        failures++;
    }
}

void test_tau_mid_recovered() {
    std::printf("\n=== Vertical optical depth at peak ~ tau_mid ===\n");
    const auto& d = shared_disk_tau_test();   // tau_mid=100, turbulence=0
    const int nz = d.vertical_bins();
    const int pk = peak_ri(d);
    const auto& prof = d.density_profile_lut();
    const auto& zmax = d.z_max_lut();
    const auto& lut  = d.opacity_luts();
    const double rho_scale = d.rho_scale();
    const double rho_mid_pk = d.rho_mid_lut()[pk];
    const double dz = zmax[pk] / (nz - 1);
    const double T_mid_k = std::clamp(d.temperature(
        d.r_min() + (d.r_max() - d.r_min()) * pk / (d.radial_bins() - 1), 0.0),
        3000.0, 1e8);
    double tau = 0.0;
    for (int zi = 0; zi < nz - 1; ++zi) {
        const double rho_a = prof[pk * nz + zi]     * rho_scale * rho_mid_pk;
        const double rho_b = prof[pk * nz + zi + 1] * rho_scale * rho_mid_pk;
        const double ka = lut.lookup_kappa_ross(rho_a, T_mid_k) + lut.lookup_kappa_es(rho_a, T_mid_k);
        const double kb = lut.lookup_kappa_ross(rho_b, T_mid_k) + lut.lookup_kappa_es(rho_b, T_mid_k);
        tau += 0.5 * (ka * rho_a + kb * rho_b) * dz;
    }
    tau *= 2.0;   // both sides of the midplane
    std::printf("  integral tau = %.2f  (tau_mid target = 100)\n", tau);
    // z-resolved kappa differs from normalize_density's single kappa_ref, so
    // allow a factor ~3. This is a sanity bound, not exact enforcement.
    if (!(tau > 33.0 && tau < 300.0)) {
        std::printf("  FAIL: recovered tau far from tau_mid\n");
        failures++;
    }
}
```

Register both in `main`:
```cpp
    test_reference_density_agreement();
    test_tau_mid_recovered();
```

- [ ] **Step 2: Build and run to verify status**

Run:
```powershell
cmake --build build --config Release --target test-volumetric
./build/Release/test-volumetric.exe
```
Expected: `test_reference_density_agreement` may already pass or fail depending on the residual clamp in `normalize_density` (its lookups still clamp to `1e-6`, pulling the global scale off). Record the printed `ratio`. `test_tau_mid_recovered` records the integral. These tests pin the behavior the next steps must satisfy.

- [ ] **Step 3: Drop the clamps in `normalize_density` and seed from the column ref**

`src/volumetric_disk.cpp:956-981` — replace:
```cpp
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
```
with:
```cpp
    // Initial guess: the column solver's self-consistent reference density at
    // the peak column (already in g/cm^3), falling back to a midplane estimate.
    double rho_guess_cgs = (peak_idx < static_cast<int>(rho_cgs_ref_lut_.size())
                            && rho_cgs_ref_lut_[peak_idx] > 0.0)
                         ? rho_cgs_ref_lut_[peak_idx]
                         : 1e-10;

    // Iterate to self-consistency. No density clamps: lookup_kappa_* clamp at
    // the (wide) opacity-table edges, so the true ~1e6 density is represented.
    for (int iter = 0; iter < 5; ++iter) {
        const double kR = opacity_luts_.lookup_kappa_ross(
            rho_guess_cgs, std::clamp(T_peak, 3000.0, 1e8));
        const double kE = opacity_luts_.lookup_kappa_es(
            rho_guess_cgs, std::clamp(T_peak, 3000.0, 1e8));
        const double kappa_ref = kR + kE;

        if (kappa_ref <= 0.0 || col_integral <= 0.0) {
            rho_scale_ = 1.0;
            return;
        }

        // tau_mid = kappa_ref * rho_scale * peak_rho * col_integral
        // => rho_scale = tau_mid / (kappa_ref * peak_rho * col_integral)
        rho_scale_ = params_.tau_mid / (kappa_ref * peak_rho * col_integral);

        // Update guess for next iteration (no clamp).
        rho_guess_cgs = rho_scale_ * peak_rho;
    }
```

- [ ] **Step 4: Drop the clamp in `compute_sigma_s_phys`**

The β (pressure-regime) detection currently clamps the true midplane density to `1e-6`, which forces `β≈0` (false radiation-dominated) and `b=0.70`. The true ~`1e6` density is strongly gas-dominated (`β→1`, `b≈0.35`).

`src/volumetric_disk.cpp:1014-1015` — change:
```cpp
        double rho_mid_cgs = rho_scale_ * rho_mid_lut_[peak_idx];
        rho_mid_cgs = std::clamp(rho_mid_cgs, 1e-18, 1e-6);
```
to:
```cpp
        // True absolute midplane density (no clamp): gas-vs-radiation pressure
        // regime detection must see the real ~1e6 g/cm^3 scale, not a 1e-6 floor.
        const double rho_mid_cgs = rho_scale_ * rho_mid_lut_[peak_idx];
```

- [ ] **Step 5: Rebuild and run to verify the tests pass**

Run:
```powershell
cmake --build build --config Release --target test-volumetric
./build/Release/test-volumetric.exe
```
Expected: `test_reference_density_agreement` prints `ratio` within `[0.1, 10]`, no FAIL; `test_tau_mid_recovered` prints `tau` in `[33, 300]`, no FAIL. The construction log line should now read `b ≈ 0.35, β ≈ 1.0` (was `b = 0.700, β = 0.000`) — a real, expected change in noise amplitude (σ_s_phys ~0.108 vs ~0.216). Note the pre-existing `test_tau_midplane_near_target` may now pass or move closer; record its value.

- [ ] **Step 6: Hand the commit message to the user**

```powershell
git add src/volumetric_disk.cpp tests/test_volumetric.cpp
```
Commit message for the user:
```
fix(disk): reconcile global density scale and pressure regime with true density

normalize_density and compute_sigma_s_phys still clamped density lookups
to the old 1e-6 ceiling, so the global rho_scale and the gas/radiation
pressure-regime detection (beta) used a density 12 orders too low.
Drop those clamps and seed normalize_density from the column solver's
self-consistent reference density. The two density scales now agree to
order of magnitude, the vertical optical depth recovers ~tau_mid, and
the disk is correctly detected as gas-pressure-dominated (beta~1, b~0.35).
```

---

### Task 4: Verify the radiation hydrostatic term's sign and formulation (spec §4.4)

With `rho_cgs_ref` corrected, confirm the radiation term is now subdominant in the gas-dominated region **and** that it supports (thickens), not collapses, when radiation genuinely dominates — i.e. the term's sign/formulation is physically correct independent of the density scale. This is a verification task; it adds code only if a defect is found.

**Files:**
- Verify only: `src/volumetric_disk.cpp` (`GRRT_COL_LOG` instrumentation, already present at `solve_column:609-611, 814-831`)
- Possible fix: `src/volumetric_disk.cpp:744` (the `rhs` return) if mis-signed

- [ ] **Step 1: Confirm the radiation term is subdominant in the gas-dominated region**

Run the construction with column logging (the probe is gated to `r ≈ 8.26`):
```powershell
$env:GRRT_COL_LOG = "1"
./build/Release/test-volumetric.exe 2>&1 | Select-String "\[COL\]" | Select-Object -First 5
Remove-Item Env:\GRRT_COL_LOG
```
Expected: the `rad=` term is now `≪` the `grav=` term (was `rad≈-3.83` dominating `grav≈-4e-5`). The column no longer crashes to `rho=1e-18` in one step; `rho` decreases smoothly with `z`.

- [ ] **Step 2: Confirm the radiation term supports (does not collapse) when radiation dominates**

Reason about the sign at a hot inner column from the LUT dump rather than building a second disk (keeps construction cost down). Build and run the dumper (Task 5 builds it; if not yet built, build `dump-disk-lut` now):
```powershell
cmake --build build --config Release --target dump-disk-lut
./build/Release/dump-disk-lut.exe
```
Inspect `disk_lut_dump.csv` at the **innermost orbiting** radii (hottest, highest radiation pressure): the `prof@0.1zmax`/`prof@0.5zmax` columns there must be `> 0` (a real profile), and the inner `z_max`/`H` ratio should be **≥** that of the outer disk if radiation thickens. If inner columns are *thinner* or collapsed while outer ones are fine, the radiation term is mis-signed.

- [ ] **Step 3: Decision gate — only if Step 2 shows inner collapse**

If and only if Step 2 reveals a mis-signed radiation term, the hydrostatic balance `dρ/dz = [−ρΩz²z − ρ·d(c_s²)/dz − d(fE)/dz / ρ_ref] / c_s²` (`src/volumetric_disk.cpp:744`) has the radiation gradient with the wrong sign for support. Radiation pressure gradient should *oppose* gravity (push outward). Verify against the spec's hydrostatic form and flip the `dfE_geom` sign if needed, then re-run Steps 1–2. **If Step 2 passes, make no code change** — record the confirmation and move on.

- [ ] **Step 4: Record the verification outcome**

No commit unless Step 3 changed code. If it did:
```powershell
git add src/volumetric_disk.cpp
```
Commit message for the user (only if changed):
```
fix(disk): correct radiation-pressure gradient sign in vertical hydrostatic ODE

With rho_cgs_ref corrected, the radiation term was found to <collapse/
thicken> hot inner columns, indicating a sign error in d(fE)/dz / rho.
Flip to oppose gravity so radiation pressure supports the column.
```

---

### Task 5: Integration verification — LUT dump, banding, edge-on render

Prove the end-to-end fix with the diagnostic tool and a real render, and surface the banding-metric before/after to the user (do not silently recalibrate the threshold).

**Files:**
- Use: `tools/dump_disk_lut.cpp` (unchanged), `tests/test_volumetric.cpp::test_no_horizontal_bands`, the CLI render

- [ ] **Step 1: Build the dumper and count collapsed columns**

```powershell
cmake --build build --config Release --target dump-disk-lut
./build/Release/dump-disk-lut.exe
```
Then count columns whose profile is still collapsed at `0.1·z_max` (the CSV column is `prof_0p1zmax`):
```powershell
$rows = Import-Csv disk_lut_dump.csv
$collapsed = ($rows | Where-Object { [double]$_.prof_0p1zmax -lt 0.01 }).Count
"collapsed columns: $collapsed / $($rows.Count)"
```
Expected: `collapsed` drops from ~4056 to ~0 (a handful at the inner plunging edge is acceptable — see Task 6 / spec §7). Also spot-check `rho_mid_cgs` is smooth across radius.

- [ ] **Step 2: Run the banding regression and record before/after**

```powershell
cmake --build build --config Release --target test-volumetric
./build/Release/test-volumetric.exe 2>&1 | Select-String "band"
```
Record the printed banding metric. Expected: below the `0.25` threshold (target ≤ ~0.21, the pre-regression baseline). **Do not edit the threshold.** If the metric is between `0.21` and `0.25`, that is a pass; if above `0.25`, capture the value and proceed to the visual check before concluding — the metric is a known-flawed proxy (it penalizes faithfully-resolved turbulence).

- [ ] **Step 3: Edge-on visual render**

```powershell
./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 `
  --observer-theta 80 --fov 30 --background black `
  --disk-volumetric --disk-turbulence 0.4 --samples 30 `
  --output edge_on_after.png
```
Expected: a smooth disk with resolved turbulence and **no concentric bright/dark banding**. (Confirm the exact CLI flags against `cli/main.cpp` — flag names may differ; use the same invocation the user used pre-fix for an apples-to-apples comparison.)

- [ ] **Step 4: Surface results to the user**

Present to the user, side by side:
- collapsed-column count before (4056/4096) and after,
- banding metric before (0.570) and after,
- the `edge_on_after.png` render,
- the σ_s_phys / b / β change from Task 3 (noise amplitude halved — expected, flag it).

Per the project review workflow, **wait for the user's assessment** of the visual before declaring the banding resolved. Do not recalibrate the banding threshold without the user's call.

- [ ] **Step 5: No commit** (verification artifacts only; do not commit PNGs or CSVs).

---

### Task 6: Final test sweep, edge cases, and cleanup

**Files:**
- Verify: full test suite
- Decide: `GRRT_COL_LOG` instrumentation retention

- [ ] **Step 1: Run the full volumetric + opacity + raymarch suites**

```powershell
cmake --build build --config Release
./build/Release/test-volumetric.exe
./build/Release/test-opacity.exe
./build/Release/test-raymarch-step-control.exe
```
Expected: the new vertical-structure tests pass; the committed raymarch tests still pass (unchanged); note the status of the pre-existing `test_tau_midplane_near_target` (spec §6 expects it to pass or move closer — record, do not chase if it is a separate issue).

- [ ] **Step 2: Check the inner-edge convergence warning (spec §7)**

Watch the construction log for `WARNING: vertical profile did not converge at r_idx=...`. The `r_idx≈32` inner-edge (plunging-region) warning is a **pre-existing, separate** issue. If it is gone as a side effect, note it. If it remains, leave it — out of scope unless it falls out trivially. Confirm no *new* non-convergence warnings appeared in the orbiting region.

- [ ] **Step 3: Decide on `GRRT_COL_LOG` retention**

Keep the `GRRT_COL_LOG` instrumentation (`solve_column:609-611, 814-831`) **as-is, env-gated** — it is zero-cost when off and is the verification path for Task 4 and future Approach A work. No code change; this step is a decision record.

- [ ] **Step 4: Hand the final state to the user**

Summarize the full change set (Tasks 1–4 commits already handed off) and confirm:
- success criteria from spec §8 (profile non-collapsed, collapsed-count ≈0, banding < 0.25 + clean visual, scale agreement, τ_mid recovered, raymarch untouched, construction time OK),
- any deferred items (inner-edge warning, Approach A follow-up).

No new commit unless Step 1–2 surfaced a regression to fix.

---

## Self-review (author checklist — completed)

**Spec coverage:**
- §4.1 opacity LUT range → Task 1. ✓
- §4.2 self-consistent `rho_cgs_ref` → Task 2 (Steps 5–7). ✓
- §4.3 reconcile `normalize_density` → Task 3 (Steps 3, drops clamps lines 962/965/980). ✓ Plus §2-defect-3's σ_s clamp → Task 3 Step 4. ✓
- §4.4 radiation term verification → Task 4. ✓
- §6 testing: profile-not-collapsed → T2; monotone → T2; normalization agreement → T3; τ_mid recovered → T3; LUT dump → T5; banding → T5; edge-on render → T5; construction warnings → T6. ✓
- §7 edge cases: inner disk / plunging → T6 Step 2; radiation-dominated → T4 Step 2; opacity-table edge → T1 (range `1e9` brackets true density); iteration non-convergence → T2 fixed-point has bounded `it<5` + break. ✓
- §8 success criteria → T6 Step 4. ✓

**Placeholder scan:** no TBD/TODO/"handle appropriately" — every code step shows full code or exact edit. ✓

**Type consistency:** `ColumnSolution::rho_cgs_ref` (T2 S3) ↔ `out.rho_cgs_ref` (T2 S5) ↔ `rho_cgs_ref_lut_` member (T2 S4) ↔ `col.rho_cgs_ref` store (T2 S7) ↔ `reference_density_lut()` accessor (T2 S4) ↔ used in T3 tests and T3 S3 seed. Consistent. ✓
