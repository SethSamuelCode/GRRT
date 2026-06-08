# Disk BVP Wiring (Approach A, Plan 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the converged column BVP solver (`solve_column_bvp`) into `VolumetricDisk` so the disk's vertical structure, optical depth, and densities are honest absolute-CGS physics — eliminating the collapsed-density/banding bug at its root.

**Architecture:** Replace the old `solve_column` (which aliased on the photosphere cliff and was propped up by `normalize_density`/`nested_refine`) with a per-radius Newton-Raphson BVP solve. Columns are solved in radial order, **warm-started from the converged neighbour** (numerical continuation from the easy peak-flux column outward in both directions), so every rendered column converges. The solve emits absolute CGS `ρ_mid(r), Σ(r), τ_mid(r), z₀(r)`; these are resampled onto the uniform-z LUT with **log-density** encoding. The consumer read path (`density_cgs`, both raymarch integrands) becomes dimensionally honest by carrying the `r_g` length scale into the optical-depth integral. No analytic fallback: a column converges or the construction fails loudly (Promptable truncation at the model-validity edge, Severe for an interior hole).

**Tech Stack:** C++23, CMake/MSVC, OpenMP. Solver in `src/disk_column_bvp.cpp`; disk in `src/volumetric_disk.cpp`; consumers in `src/geodesic_tracer.cpp` (+ `src/romberg_step.cpp` sampler). Tests are standalone exes (`test-column-bvp`, `test-volumetric`, `test-opacity`) built by `CMakeLists.txt`.

---

## Context the implementer must know

- **Read first:** `docs/superpowers/specs/2026-06-01-disk-first-principles-vertical-structure-design.md` (§6, §7, §9, §11, §13, §15 phases 4–5) and `docs/superpowers/references/disk-physics-formulas.md` (§20, the verified BVP + the error-trap checklist — **check formulas there, never re-derive**).
- **Plan 1 (committed):** added `r_g_` (`= G·mass_solar·M_sun/c²` [cm]), `mdot_`, `rho_mid_est_`, accessors `r_g()/mdot()/rho_mid_estimate()`, and the mass-adaptive opacity table. **Do not touch.**
- **Plan 2 (committed):** the standalone solver `solve_column_bvp(in, op)` in `src/disk_column_bvp.cpp` / `include/grrt/scene/disk_column_bvp.h`, tested in `tests/test_column_bvp.cpp`. Newton relaxation, analytic block Jacobian (cross-checked to 2.5e-9), damped line search. Returns `ColumnBVPSolution{ q,z[cm],P,Q,T[K],rho[g/cm³] (index 0=midplane, back=surface); z0[cm], Sigma0[g/cm²], tau_mid, converged, iters, final_residual }`.
- **Units (the keystone):** geometric→CGS is `length_cm = length_geom · r_g_`; `Ω_cgs = Ω_geom · c_cgs / r_g_`. `omega_z_sq(r)` and `omega_orb(r)` return geometric values. The exact Kerr shear `|r dΩ/dr|` (geometric) is computed in `compute_radial_structure` (`src/volumetric_disk.cpp:563-567`).
- **Viscous heating uses the orbital shear `|r dΩ/dr|`, NOT `Ω_z`** — they differ for Kerr. `ColumnInputs.shear` is the shear; `ColumnInputs.omega_z` is the vertical gravity.
- **The two raymarch integrands** that accumulate optical depth are: the romberg sampler `sample_integrand` (`src/geodesic_tracer.cpp:73`, drives the RGB path via `rs.dtau`), and the spectral raymarch (`src/geodesic_tracer.cpp:797,808`). Both currently omit `r_g`.
- **Workflow constraints (carry forward, non-negotiable):** NEVER run `git commit` — hand the commit message to the human. Subagents: **sonnet or opus only, never haiku.** Present every reviewer recommendation with a take and WAIT for the human's call before fixing.

## File structure (what each task touches)

- `include/grrt/scene/disk_column_bvp.h` — solver interface: add warm-start param (Task 1); remove `used_fallback` (Task 2).
- `src/disk_column_bvp.cpp` — warm-start branch (Task 1); honest non-convergence (Task 2).
- `tests/test_column_bvp.cpp` — warm-start test (Task 1); repurpose fallback tests (Task 2).
- `include/grrt/scene/volumetric_disk.h` — new private members/helpers; retire dead declarations (Tasks 3,5,6).
- `src/volumetric_disk.cpp` — the bulk: BVP march (Task 3), honest-CGS+log switch (Task 4), retire refinement (Task 5), failure policy + Toomre + range guard (Task 6), σ_s wiring (Task 7).
- `src/geodesic_tracer.cpp` — `r_g` factor in both integrands (Task 4).
- `tests/test_volumetric.cpp` — collapse/smoothness/march tests (Task 3), τ honesty + render (Tasks 4,8), retired-machinery tests (Task 5), policy tests (Task 6), σ_s (Task 7), integration (Task 9).
- `tools/dump_disk_lut.cpp` — extend the dump for the not-collapsed integration check (Task 9).

## Constant for the whole plan

All columns use a fixed node count so neighbour warm-starts share the same `q`-grid and `U` layout. Add near the top of `src/volumetric_disk.cpp` (after the `using namespace` / anonymous helpers, file scope inside `namespace grrt`):

```cpp
// Fixed BVP vertical resolution. All columns share this N so a converged
// neighbour's state vector (length 4N+2) is a valid warm start for the next
// column (numerical continuation). Richardson auto-N is deferred (refinement 1).
static constexpr int kColumnNodes = 200;
```

---

### Task 1: Solver — full-U warm-start parameter

**Files:**
- Modify: `include/grrt/scene/disk_column_bvp.h:45-47`
- Modify: `src/disk_column_bvp.cpp:440-491`
- Test: `tests/test_column_bvp.cpp`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_column_bvp.cpp` (before `int main()`), and call it from `main()` after `test_radiation_thickens();`:

```cpp
static void test_warm_start_converges_fast() {
    std::printf("\n=== warm start from a converged neighbour converges fast ===\n");
    auto lut = grrt::build_opacity_luts(1e-14, 1e4, 3000.0, 1e8);
    grrt::ColumnInputs a{}; a.T_eff = 5e4; a.shear = 3e3; a.omega_z = 2e3;
    a.alpha = 0.1; a.rho_mid_guess = 1e-2; a.n_nodes = 120; a.max_iters = 80; a.tol = 1e-8;
    auto sa = grrt::solve_column_bvp(a, lut);            // cold solve of the "neighbour"
    if (!sa.converged) { std::printf("  FAIL: neighbour did not converge\n"); failures++; return; }

    // Pack neighbour's converged state into a length-(4N+2) warm vector.
    const int N = a.n_nodes;
    std::vector<double> warm((size_t)4*N + 2, 0.0);
    for (int i = 0; i < N; ++i) {
        warm[4*i+0]=sa.P[i]; warm[4*i+1]=sa.Q[i]; warm[4*i+2]=sa.T[i]; warm[4*i+3]=sa.z[i];
    }
    warm[4*N]=sa.z0; warm[4*N+1]=sa.Sigma0;

    // A nearby column (slightly hotter). Cold vs warm iteration counts.
    grrt::ColumnInputs b = a; b.T_eff = 5.2e4;
    auto cold = grrt::solve_column_bvp(b, lut);                 // no warm start
    auto warmed = grrt::solve_column_bvp(b, lut, &warm);        // warm start
    std::printf("  cold: conv=%d iters=%d ; warm: conv=%d iters=%d ; z0 cold=%.3e warm=%.3e\n",
                cold.converged, cold.iters, warmed.converged, warmed.iters, cold.z0, warmed.z0);
    if (!warmed.converged) { std::printf("  FAIL: warm start did not converge\n"); failures++; return; }
    if (!(warmed.iters < cold.iters)) { std::printf("  FAIL: warm start not faster\n"); failures++; }
    // Both must reach the same physical solution.
    if (cold.converged) check("warm z0 == cold z0", warmed.z0, cold.z0, 1e-4);
}
```

- [ ] **Step 2: Run to verify it fails to compile**

Run: `cmake --build build --config Release --target test-column-bvp`
Expected: FAIL — `solve_column_bvp` takes 2 args, the 3-arg call (`&warm`) does not compile.

- [ ] **Step 3: Add the warm-start parameter to the interface**

In `include/grrt/scene/disk_column_bvp.h`, replace the declaration (lines 45-47):

```cpp
/// Solve the grey vertical-structure BVP for one column (Newton relaxation).
/// @param warm_start optional initial state U (length 4·n_nodes+2) from a
///        converged neighbouring column (numerical continuation). When null or
///        wrong-sized, the solver builds its own flux-balanced analytic seed.
GRRT_EXPORT ColumnBVPSolution solve_column_bvp(const ColumnInputs& in,
                                               const OpacityLUTs& opacity,
                                               const std::vector<double>* warm_start = nullptr);
```

- [ ] **Step 4: Branch the seed construction on the warm start**

In `src/disk_column_bvp.cpp`, change the signature at line 440 to match the header, then replace the seed setup (the `std::vector<double> U = build_seed(in);` at line 445 **and** the entire flux-balance rescale block lines 448-490) with:

```cpp
    std::vector<double> U;
    std::vector<double> R, J, Jcopy, rhs, Utry, Rtry;

    if (warm_start && (int)warm_start->size() == n) {
        // Numerical continuation: start Newton from the converged neighbour.
        // It is already flux-balanced, so skip the analytic seed + rescale.
        U = *warm_start;
    } else {
        U = build_seed(in);
        // Flux-balance seed rescale (cold start only).
        // [KEEP the existing block verbatim here — the comment and code from the
        //  current lines 448-490, ending with `U[4*N+1] *= scale;`]
        {
            using namespace constants;
            const double cs2_gas_r = k_B * in.T_eff / (mu_fully_ionized * m_p);
            const double rho_mid_seed = in.rho_mid_guess;
            const double P_rad_seed   = (a_rad / 3.0) * in.T_eff * in.T_eff * in.T_eff * in.T_eff;
            const double H_r = std::sqrt(cs2_gas_r + P_rad_seed / rho_mid_seed) / in.omega_z;
            const double P_gas_mid_seed = rho_mid_seed * cs2_gas_r;
            const double heat_seed    = in.alpha * in.shear * P_gas_mid_seed * H_r * std::sqrt(std::numbers::pi / 2.0);
            const double flux_target  = surface_flux(in.T_eff);
            double scale = (heat_seed > 0.0) ? flux_target / heat_seed : 1.0;
            for (int i = 0; i < N; ++i) {
                const double T_i = U[4*i+2];
                const double rho_old = std::max(eos_rho(U[4*i+0], T_i), 0.0);
                const double rho_new = rho_old * scale;
                U[4*i+0] = rho_new * cs2_gas_r + (a_rad/3.0)*T_i*T_i*T_i*T_i;
            }
            U[4*N+1] *= scale;
        }
    }
```

(Remove the now-duplicated `std::vector<double> R, J, Jcopy, rhs, Utry, Rtry;` declaration at the old line 446 — it now lives in the block above.)

- [ ] **Step 5: Build and run; verify pass**

Run: `cmake --build build --config Release --target test-column-bvp && ./build/Release/test-column-bvp`
Expected: `=== 0 failures ===`, and the new test prints `warm: iters` strictly less than `cold: iters`.

- [ ] **Step 6: Commit** (hand this message to the human — do NOT run git commit)

```
feat(bvp): full-U warm-start for column solver (numerical continuation)

solve_column_bvp() gains an optional initial-state arg; a converged
neighbour's U (4N+2) seeds Newton directly, skipping the cold analytic
seed + flux-balance rescale. Enables radial continuation in Plan 3.
```

---

### Task 2: Solver — remove the analytic fallback; honest non-convergence

**Files:**
- Modify: `include/grrt/scene/disk_column_bvp.h:37`
- Modify: `src/disk_column_bvp.cpp:562-594`
- Test: `tests/test_column_bvp.cpp:192-237`

- [ ] **Step 1: Rewrite the two fallback-dependent tests (they become the spec)**

In `tests/test_column_bvp.cpp`, replace `test_convergence_sweep` (lines 192-217) with:

```cpp
static void test_convergence_sweep() {
    std::printf("\n=== converges across (T_eff, shear); non-converged return empty, never a fake profile ===\n");
    auto lut = grrt::build_opacity_luts(1e-16, 1e6, 3000.0, 1e8);
    const double Teffs[] = {1e4, 5e4, 2e5, 1e6};
    const double oms[]   = {5e2, 2e3, 8e3};
    int ok = 0, total = 0;
    for (double Te : Teffs) for (double om : oms) {
        grrt::ColumnInputs in{}; in.T_eff = Te; in.shear = 1.5*om; in.omega_z = om;
        in.alpha = 0.1; in.rho_mid_guess = 1e-2; in.n_nodes = 120; in.max_iters = 80; in.tol = 1e-8;
        auto s = grrt::solve_column_bvp(in, lut);
        total++;
        if (s.converged) {
            ok++;
            // converged solutions carry a full, sane profile
            if ((int)s.rho.size() != in.n_nodes) { std::printf("  FAIL: converged but profile size wrong\n"); failures++; }
        } else {
            // NO fallback: a non-converged solve returns EMPTY vectors, never a fabricated profile.
            if (!s.q.empty() || !s.rho.empty() || !s.T.empty()) {
                std::printf("  FAIL: non-converged solve returned a non-empty profile (T_eff=%.0e om=%.0e)\n", Te, om);
                failures++;
            }
        }
    }
    std::printf("  converged %d/%d (standalone cold-start; Plan 3 warm-starts these)\n", ok, total);
    if (ok == 0) { std::printf("  FAIL: nothing converged\n"); failures++; }
}
```

Replace `test_radiation_thickens` (lines 219-237) with (drops `used_fallback`):

```cpp
static void test_radiation_thickens() {
    std::printf("\n=== radiation-dominated column thicker than gas-dominated ===\n");
    auto lut = grrt::build_opacity_luts(1e-16, 1e6, 3000.0, 1e8);
    grrt::ColumnInputs cold{}; cold.T_eff=2e4; cold.shear=3e3; cold.omega_z=2e3;
    cold.alpha=0.1; cold.rho_mid_guess=1e-2; cold.n_nodes=160; cold.max_iters=80; cold.tol=1e-8;
    grrt::ColumnInputs hot = cold; hot.T_eff = 1e6;
    auto sc = grrt::solve_column_bvp(cold, lut);
    auto sh = grrt::solve_column_bvp(hot, lut);
    std::printf("  cold converged=%d z0=%.3e ; hot converged=%d z0=%.3e\n",
                sc.converged, sc.z0, sh.converged, sh.z0);
    if (sc.converged && sh.converged) {
        if (!(sh.z0 > sc.z0)) { std::printf("  FAIL: radiation did not thicken the column\n"); failures++; }
        else { std::printf("  PASS: hot column thicker (z0 %.3e > %.3e)\n", sh.z0, sc.z0); }
    } else {
        std::printf("  (skipped thickness compare: a column did not converge from cold start)\n");
    }
}
```

- [ ] **Step 2: Run to verify it fails to compile**

Run: `cmake --build build --config Release --target test-column-bvp`
Expected: FAIL — `ColumnBVPSolution` still has `used_fallback` referenced nowhere now, but the solver still SETS it; compile is fine, but the sweep test will FAIL at runtime because the solver currently returns a non-empty fallback profile on non-convergence. Run `./build/Release/test-column-bvp` → `FAIL: non-converged solve returned a non-empty profile`.

- [ ] **Step 3: Remove `used_fallback` from the struct**

In `include/grrt/scene/disk_column_bvp.h`, delete line 37:

```cpp
    bool   used_fallback = false; ///< true if the analytic-profile fallback was used
```

- [ ] **Step 4: Make non-convergence return empty (no fabricated profile)**

In `src/disk_column_bvp.cpp`, replace the fallback + unpack block (lines 562-593, from the `// On non-convergence:` comment through `s.tau_mid = tau;` and `return s;`) with:

```cpp
    // No fallback (Approach A: fail or succeed, never a fabricated profile).
    // On non-convergence return EMPTY profile vectors; the caller MUST check
    // `converged` before reading the solution.
    if (!s.converged) {
        s.q.clear(); s.z.clear(); s.P.clear(); s.Q.clear(); s.T.clear(); s.rho.clear();
        s.z0 = 0.0; s.Sigma0 = 0.0; s.tau_mid = 0.0;
        return s;
    }

    // Unpack the converged state into the solution.
    s.q.resize(N); s.z.resize(N); s.P.resize(N); s.Q.resize(N); s.T.resize(N); s.rho.resize(N);
    for (int i = 0; i < N; ++i) {
        const double Pi = U[4*i+0], Qi = U[4*i+1], Ti = U[4*i+2], zi = U[4*i+3];
        s.q[i] = (double)i / (N - 1);
        s.P[i] = Pi; s.Q[i] = Qi; s.T[i] = Ti; s.z[i] = zi;
        s.rho[i] = std::max(eos_rho(Pi, Ti), 0.0);
    }
    s.z0 = U[4*N];
    s.Sigma0 = U[4*N+1];

    double tau = 0.0;
    for (int i = 0; i + 1 < N; ++i) {
        const double kRi = kappa_total(op, std::max(s.rho[i],   RHO_GHOST_FLOOR), s.T[i]);
        const double kRj = kappa_total(op, std::max(s.rho[i+1], RHO_GHOST_FLOOR), s.T[i+1]);
        const double dz = std::abs(s.z[i+1] - s.z[i]);
        tau += 0.5 * (kRi*s.rho[i] + kRj*s.rho[i+1]) * dz;
    }
    s.tau_mid = tau;
    return s;
```

- [ ] **Step 5: Build and run; verify pass**

Run: `cmake --build build --config Release --target test-column-bvp && ./build/Release/test-column-bvp`
Expected: `=== 0 failures ===`. The sweep still converges ~6/12 from cold start (expected — Plan 3 warm-starts the rest).

- [ ] **Step 6: Commit** (hand to human)

```
refactor(bvp): remove analytic fallback; non-convergence returns empty

Approach A "fail or succeed": a non-converged column no longer fabricates
an analytic profile (used_fallback removed). It returns converged=false
with empty vectors; the caller must gate on `converged`. Plan 3's
warm-start continuation is what makes the hard columns converge.
```

---

### Task 3: Wiring — BVP column march (warm-started, bidirectional from peak flux)

**Files:**
- Modify: `include/grrt/scene/volumetric_disk.h:236-247` (add helper decls), keep `ColumnSolution` for now (deleted in Task 5)
- Modify: `src/volumetric_disk.cpp` (add `kColumnNodes`; rewrite `compute_vertical_profiles`; add helpers; change constructor pipeline)
- Test: `tests/test_volumetric.cpp`

**Scene-setting:** This task makes the disk's structure come from the BVP, but keeps `normalize_density()` running (so `density_cgs` stays in the OLD calibrated scale and every render/banding test stays green). The honest-CGS switch is the next task. After this task `rho_mid_lut_` holds **absolute cgs** midplane density and `rho_profile_lut_` holds the **linear** normalized profile `ρ(z)/ρ_mid`.

- [ ] **Step 1: Add helper declarations**

In `include/grrt/scene/volumetric_disk.h`, in the private section after the `ColumnSolution solve_column(...)` declaration (line 247), add:

```cpp
    /// Build the CGS BVP inputs for radial bin ri from the radial LUTs.
    /// Converts geometric Ω_z and the exact Kerr shear |r dΩ/dr| to 1/s via
    /// r_g_, and seeds rho_mid_guess from rho_mid_est_ (overridden by the
    /// warm-start march). r is the bin's radius.
    ColumnInputs make_column_inputs(int ri, double r) const;

    /// Resample a converged column (q-grid, CGS) onto the uniform-z LUT for bin
    /// ri: stores z_max_lut_[ri] (geometric = z0/r_g_), rho_mid_lut_[ri]
    /// (absolute cgs = rho.front()), rho_profile_lut_[ri,·] (normalized rho/rho_mid),
    /// T_profile_lut_[ri,·]. `store_log` selects log vs linear density encoding.
    void store_column(int ri, const ColumnBVPSolution& sol, bool store_log);
```

Add `#include "grrt/scene/disk_column_bvp.h"` to the top of `include/grrt/scene/volumetric_disk.h` (after the existing includes).

- [ ] **Step 2: Write the failing test**

In `tests/test_volumetric.cpp`, add (call from `main()` after `test_density_strictly_positive_inside_volume();`):

```cpp
void test_bvp_profile_not_collapsed() {
    std::printf("\n=== BVP vertical profile is resolved (not collapsed to 1-2 bins) ===\n");
    const auto& disk = shared_disk_no_noise();
    // At a representative inner radius, the normalized profile must fall smoothly,
    // not drop to ~0 within the first bin (the original collapse bug).
    const double r = 8.0;
    const double zm = disk.z_max_at(r);
    if (!(zm > 0.0)) { std::printf("  FAIL: z_max <= 0\n"); failures++; return; }
    // Sample the normalized profile at 10%, 50%, 90% of z_max via temperature/density.
    const double rho_mid = disk.density_cgs(r, 0.0, 0.0);
    const double rho_half = disk.density_cgs(r, 0.5 * zm, 0.0);
    std::printf("  rho_mid=%.3e rho(0.5 z_max)=%.3e ratio=%.3e\n",
                rho_mid, rho_half, rho_half / std::max(rho_mid, 1e-300));
    // A resolved Gaussian-ish column keeps a meaningful fraction at half-height.
    if (!(rho_half / std::max(rho_mid, 1e-300) > 1e-3)) {
        std::printf("  FAIL: density collapses before half z_max (the original bug)\n"); failures++;
    } else { std::printf("  PASS\n"); }
}

void test_bvp_radial_smoothness() {
    std::printf("\n=== rho_mid(r), z_max(r) smooth across radius (no holes) ===\n");
    const auto& disk = shared_disk_no_noise();
    int jumps = 0;
    double r_prev = disk.r_isco() + 0.1;
    double rho_prev = disk.density_cgs(r_prev, 0.0, 0.0);
    for (int i = 1; i < 100; ++i) {
        const double r = disk.r_isco() + 0.1 + (20.0 - disk.r_isco()) * i / 100.0;
        const double rho = disk.density_cgs(r, 0.0, 0.0);
        if (rho_prev > 0.0 && rho > 0.0) {
            const double jump = std::abs(std::log(rho) - std::log(rho_prev));
            if (jump > 1.0) { jumps++; }  // > e-fold step between adjacent samples = a hole/discontinuity
        }
        rho_prev = rho; r_prev = r;
    }
    std::printf("  adjacent-sample e-fold jumps: %d\n", jumps);
    if (jumps > 2) { std::printf("  FAIL: rho_mid(r) not smooth (holes)\n"); failures++; }
    else { std::printf("  PASS\n"); }
}
```

- [ ] **Step 3: Run to verify the test builds against the current disk (baseline)**

Run: `cmake --build build --config Release --target test-volumetric && ./build/Release/test-volumetric`
Expected: builds; the two new tests may PASS or FAIL on the *old* `solve_column` output — record the baseline. (They become the regression guard for the BVP output.)

- [ ] **Step 4: Add `kColumnNodes` and the two helpers**

In `src/volumetric_disk.cpp`, add the `kColumnNodes` constant (shown in "Constant for the whole plan" above) at file scope inside `namespace grrt`. Then add the helper implementations near `compute_vertical_profiles`:

```cpp
ColumnInputs VolumetricDisk::make_column_inputs(int ri, double r) const {
    using namespace constants;
    ColumnInputs in{};
    in.T_eff = T_eff_lut_[ri];
    const double oz_geom = std::sqrt(std::max(omega_z_sq(r), 0.0));   // 1/M
    in.omega_z = oz_geom * c_cgs / r_g_;                              // 1/s
    // Exact Kerr shear |r dΩ/dr| (geometric), same formula as compute_radial_structure.
    const double sqM = std::sqrt(mass_);
    const double denom = r * std::sqrt(r) + spin_ * sqM;
    const double dOmega_dr = -1.5 * sqM * std::sqrt(r) / (denom * denom);
    const double shear_geom = std::abs(r * dOmega_dr);               // 1/M
    in.shear = shear_geom * c_cgs / r_g_;                            // 1/s
    in.alpha = (params_.alpha > 0.0) ? params_.alpha : 0.1;
    in.rho_mid_guess = (rho_mid_est_ > 0.0) ? rho_mid_est_ : 1e-8;   // cold-seed; warm march overrides
    in.n_nodes = kColumnNodes;
    in.max_iters = 80;
    in.tol = 1e-8;
    return in;
}

void VolumetricDisk::store_column(int ri, const ColumnBVPSolution& sol, bool store_log) {
    // z_max in geometric units for the LUT; rho_mid absolute cgs.
    const double z0_geom = sol.z0 / r_g_;
    z_max_lut_[ri] = z0_geom;
    const double rho_mid = sol.rho.front();
    rho_mid_lut_[ri] = rho_mid;

    // Resample the q-grid profile (sol.z monotone 0..z0 cm) onto uniform-z.
    const int N = (int)sol.z.size();
    for (int zi = 0; zi < n_z_; ++zi) {
        const double frac = (n_z_ > 1) ? (double)zi / (n_z_ - 1) : 0.0;
        const double z_cm = frac * sol.z0;
        // Locate bracketing nodes in sol.z (increasing).
        int lo = 0;
        while (lo + 1 < N && sol.z[lo + 1] < z_cm) ++lo;
        const int hi = std::min(lo + 1, N - 1);
        const double span = sol.z[hi] - sol.z[lo];
        const double t = (span > 0.0) ? (z_cm - sol.z[lo]) / span : 0.0;
        // Density: log-interp (matches the LUT log encoding); normalize by rho_mid.
        const double rlo = std::max(sol.rho[lo], 1e-300);
        const double rhi = std::max(sol.rho[hi], 1e-300);
        const double rho_z = std::exp((1.0 - t) * std::log(rlo) + t * std::log(rhi));
        const double rho_norm = std::max(rho_z / rho_mid, 1e-300);
        rho_profile_lut_[ri * n_z_ + zi] = store_log ? std::log(rho_norm) : rho_norm;
        // Temperature: linear.
        T_profile_lut_[ri * n_z_ + zi] = (1.0 - t) * sol.T[lo] + t * sol.T[hi];
    }
}
```

- [ ] **Step 5: Rewrite `compute_vertical_profiles` (the bidirectional warm-start march)**

In `src/volumetric_disk.cpp`, replace the entire body of `compute_vertical_profiles()` (lines 956-983) with:

```cpp
void VolumetricDisk::compute_vertical_profiles() {
    z_max_lut_.assign(n_r_, 0.0);
    rho_profile_lut_.assign(n_r_ * n_z_, 0.0);
    T_profile_lut_.assign(n_r_ * n_z_, 0.0);

    auto r_at = [&](int i){ return r_min_ + (r_outer_ - r_min_) * i / (n_r_ - 1); };

    // First orbiting bin and the peak-flux (hottest T_eff) orbiting bin.
    int isco_idx = -1, peak_idx = -1;
    double peak_T = 0.0;
    for (int i = 0; i < n_r_; ++i) {
        const double r = r_at(i);
        if (r >= r_isco_ && T_eff_lut_[i] > 0.0) {
            if (isco_idx < 0) isco_idx = i;
            if (T_eff_lut_[i] > peak_T) { peak_T = T_eff_lut_[i]; peak_idx = i; }
        }
    }
    if (peak_idx < 0) {
        // No orbiting columns with positive flux — nothing to solve.
        std::fprintf(stderr, "[VolumetricDisk] WARNING: no orbiting columns with T_eff>0\n");
        return;
    }

    // Per-bin converged state vectors, kept so each neighbour can warm-start the next.
    std::vector<std::vector<double>> Uconv(n_r_);
    auto pack = [&](const ColumnBVPSolution& s) {
        const int N = (int)s.z.size();
        std::vector<double> U((size_t)4*N + 2, 0.0);
        for (int i = 0; i < N; ++i) { U[4*i+0]=s.P[i]; U[4*i+1]=s.Q[i]; U[4*i+2]=s.T[i]; U[4*i+3]=s.z[i]; }
        U[4*N]=s.z0; U[4*N+1]=s.Sigma0;
        return U;
    };

    // Solve one bin; warm is the neighbour's packed U (or nullptr to cold-start).
    // Returns the converged solution (caller decides what to do if !converged).
    auto solve_bin = [&](int ri, const std::vector<double>* warm) {
        ColumnInputs in = make_column_inputs(ri, r_at(ri));
        // Warm-start density hint also tracks the neighbour, improving the cold path.
        if (warm) { const int N=in.n_nodes; in.rho_mid_guess = std::max(eos_rho((*warm)[0], (*warm)[2]), in.rho_mid_guess); }
        return solve_column_bvp(in, opacity_luts_, warm);
    };

    // Anchor: cold-solve the peak-flux column (the easiest to converge).
    {
        auto s = solve_bin(peak_idx, nullptr);
        if (!s.converged) {
            // The easiest column failed — a real problem. Task 6 escalates this to Severe;
            // for now, leave it unstored (z_max stays 0 → no disk there) and warn.
            std::fprintf(stderr, "[VolumetricDisk] WARNING: peak-flux column ri=%d did not converge\n", peak_idx);
        } else {
            store_column(peak_idx, s, /*store_log=*/false);
            Uconv[peak_idx] = pack(s);
        }
    }

    // March OUTWARD toward r_outer, warm-starting from the inner neighbour.
    for (int ri = peak_idx + 1; ri < n_r_; ++ri) {
        const std::vector<double>* warm = Uconv[ri-1].empty() ? nullptr : &Uconv[ri-1];
        auto s = solve_bin(ri, warm);
        if (s.converged) { store_column(ri, s, false); Uconv[ri] = pack(s); }
        // (non-converged handling deferred to Task 6; leaves z_max=0 → truncated here)
    }
    // March INWARD toward the ISCO, warm-starting from the outer neighbour.
    for (int ri = peak_idx - 1; ri >= isco_idx; --ri) {
        const std::vector<double>* warm = Uconv[ri+1].empty() ? nullptr : &Uconv[ri+1];
        auto s = solve_bin(ri, warm);
        if (s.converged) { store_column(ri, s, false); Uconv[ri] = pack(s); }
    }

    // Plunging region (r < r_isco) and any zero-flux inner bins: no BVP. Carry the
    // ISCO column's converged shape, scaled by the frozen/decayed H(r) and the
    // BPT72 taper (spec §13: free-fall retains its last equilibrium structure).
    if (isco_idx > 0 && !Uconv[isco_idx].empty()) {
        const double z_max_isco = z_max_lut_[isco_idx];
        const double H_isco = H_lut_[isco_idx];
        const double rho_mid_isco = rho_mid_lut_[isco_idx];
        for (int ri = 0; ri < isco_idx; ++ri) {
            const double r = r_at(ri);
            const double Hr = H_lut_[ri];
            z_max_lut_[ri] = (H_isco > 0.0) ? z_max_isco * (Hr / H_isco) : z_max_isco;
            rho_mid_lut_[ri] = rho_mid_isco * taper(r);
            for (int zi = 0; zi < n_z_; ++zi) {
                rho_profile_lut_[ri * n_z_ + zi] = rho_profile_lut_[isco_idx * n_z_ + zi];
                T_profile_lut_[ri * n_z_ + zi]   = T_profile_lut_[isco_idx * n_z_ + zi];
            }
        }
    }

    std::printf("[VolumetricDisk] Vertical profiles via column BVP "
                "(n_r=%d, n_z=%d, peak ri=%d, isco ri=%d)\n", n_r_, n_z_, peak_idx, isco_idx);
}
```

- [ ] **Step 6: Replace the constructor's refinement call with a fixed build**

In `src/volumetric_disk.cpp` constructor, replace lines 160-170 (the `Refining LUT sizing` printf, the `nested_refine()` call, the `Refinement done` printf, and the `compute_vertical_profiles();` call) with:

```cpp
    // Fixed LUT resolution (Approach A retires Richardson refinement — the BVP
    // solves on a proper q-grid and does not alias like the old solve_column).
    n_z_ = kColumnNodes;
    z_max_lut_.assign(n_r_, 0.0);
    rho_profile_lut_.assign((size_t)n_r_ * n_z_, 0.0);
    T_profile_lut_.assign((size_t)n_r_ * n_z_, 0.0);
    std::printf("[VolumetricDisk] Solving column BVPs (n_r=%d, n_z=%d)...\n", n_r_, n_z_);
    compute_vertical_profiles();
```

(`normalize_density()` at line 173 stays for now — Task 4 removes it.)

- [ ] **Step 7: Build and run**

Run: `cmake --build build --config Release --target test-volumetric && ./build/Release/test-volumetric`
Expected: `test_bvp_profile_not_collapsed` and `test_bvp_radial_smoothness` PASS; construction has no Severe warnings; existing banding/τ tests still pass (still in the old normalized scale). Note the construction time printed.

- [ ] **Step 8: Commit** (hand to human)

```
feat(disk): solve vertical structure via column BVP (warm-start march)

compute_vertical_profiles() now solves each orbiting column with
solve_column_bvp, marching radially from the peak-flux column outward in
both directions and warm-starting each from its converged neighbour
(numerical continuation). Plunging columns carry the ISCO shape. rho_mid
is now absolute cgs; normalize_density still runs (honest-CGS switch next).
```

---

### Task 4: Honest CGS — absolute density, log encoding, and the `r_g` optical-depth factor

**Files:**
- Modify: `src/volumetric_disk.cpp` (`density()`, `interp_2d`, retire `normalize_density`, constructor)
- Modify: `src/geodesic_tracer.cpp:73` and `:797,808,825` (both integrands)
- Modify: `src/romberg_step.cpp` — none (it multiplies the sampler integrand by `ds`; the `r_g` lives in the integrand)
- Test: `tests/test_volumetric.cpp`

**This task is atomic** — the unit system cannot be half-switched. Absolute-cgs `ρ` and the `r_g` integrand factor land together or every `τ` is wrong by ~`r_g`. One commit.

- [ ] **Step 1: Write the failing test — in-render τ_mid ≈ BVP τ_mid**

In `tests/test_volumetric.cpp` add (call after `test_tau_midplane_near_target();`):

```cpp
void test_tau_matches_bvp() {
    std::printf("\n=== in-LUT vertical tau at peak ≈ BVP-emergent tau_mid (honest r_g) ===\n");
    const auto& disk = shared_disk_no_noise();   // mass_solar=10 default
    using namespace grrt::constants;
    // Peak-flux radius by midplane density.
    double best_r = disk.r_isco() + 0.5, best_rho = 0.0;
    for (int i = 0; i < 80; ++i) {
        const double r = disk.r_isco() + (20.0 - disk.r_isco()) * i / 79.0;
        const double rho = disk.density_cgs(r, 0.0, 0.0);
        if (rho > best_rho) { best_rho = rho; best_r = r; }
    }
    const double r = best_r;
    const double zm_geom = disk.z_max_at(r);
    const double rg = disk.r_g();
    const auto& opa = disk.opacity_luts();
    // Integrate kappa_total * rho_cgs * dz_cm (dz_cm = dz_geom * r_g) midplane->surface.
    const int Nz = 400;
    const double dz_geom = zm_geom / (Nz - 1);
    double tau = 0.0;
    for (int i = 0; i + 1 < Nz; ++i) {
        const double za = i * dz_geom, zb = (i+1) * dz_geom;
        const double ra = disk.density_cgs(r, za, 0.0), rb = disk.density_cgs(r, zb, 0.0);
        const double Ta = std::max(disk.temperature(r, za), 3000.0);
        const double Tb = std::max(disk.temperature(r, zb), 3000.0);
        const double ka = opa.lookup_kappa_ross(ra, Ta) + opa.lookup_kappa_es(ra, Ta);
        const double kb = opa.lookup_kappa_ross(rb, Tb) + opa.lookup_kappa_es(rb, Tb);
        tau += 0.5 * (ka*ra + kb*rb) * (dz_geom * rg);
    }
    std::printf("  peak r=%.2f tau(midplane->surface)=%.3f (physical, emergent)\n", r, tau);
    // The emergent optical depth is set by physics, not a knob. It must be a sane,
    // O(1-100) optically-thick value — NOT ~1e6 (missing r_g) and NOT ~1e-6 (double r_g).
    if (!(tau > 0.5 && tau < 1e4)) { std::printf("  FAIL: tau not a physical O(1-1e3) value\n"); failures++; }
    else { std::printf("  PASS\n"); }
}
```

- [ ] **Step 2: Run — verify it fails (τ off by ~r_g because density is still old-scaled)**

Run: `cmake --build build --config Release --target test-volumetric && ./build/Release/test-volumetric`
Expected: the new test prints τ wildly off (old normalized scale × r_g), FAIL.

- [ ] **Step 3: Retire `normalize_density`; set `rho_scale_ ≡ 1`; store log profile**

In `src/volumetric_disk.cpp` constructor, delete the `normalize_density();` call and its printf (lines 172-173). `rho_scale_` is already default 1.0; leave the member but it is now always 1. In `compute_vertical_profiles`, change the two `store_column(..., /*store_log=*/false)` and the peak `store_column(..., false)` calls to `true`, and in the plunging-copy loop the profile is already log (copied verbatim from the ISCO column) — no change needed there.

- [ ] **Step 4: Switch the density read path to log + absolute cgs**

In `src/volumetric_disk.cpp`, replace `density()` (lines 358-381) with:

```cpp
double VolumetricDisk::density(double r, double z, double phi) const {
    if (r <= r_horizon_ || r > r_outer_ + 0.5 * outer_taper_width_) return 0.0;
    const double z_abs = std::abs(z);
    const double zm = z_max_at(r);
    if (z_abs >= zm) return 0.0;

    const double rho_mid  = interp_radial(rho_mid_lut_, r);   // absolute cgs
    const double log_norm = interp_2d(rho_profile_lut_, r, z_abs);  // log(rho/rho_mid)
    const double base     = rho_mid * std::exp(log_norm) * taper(r);

    const double L = noise_correlation_length(r);
    if (L <= 0.0) return base;

    const double nx = r * std::cos(phi) / L;
    const double ny = r * std::sin(phi) / L;
    const double nz = z / L;
    const double n  = noise_.evaluate_fbm(nx, ny, nz, params_.noise_octaves);

    double arg = sigma_s_phys_ * params_.turbulence * n;
    arg = std::clamp(arg, -50.0, 50.0);
    return base * std::exp(arg);
}
```

(`interp_2d` for density now interpolates the **log** profile linearly — correct, since the stored values are already `log(ρ/ρ_mid)`. `interp_2d` is shared with temperature, which stores **linear** T; that is still correct because T's stored values are linear. No change to `interp_2d` itself — it linearly blends whatever is stored.)

`density_cgs()` (lines 383-386) already returns `density(...)`; it now returns absolute cgs. Leave it.

- [ ] **Step 5: Add the `r_g` factor to the RGB sampler integrand**

In `src/geodesic_tracer.cpp`, in the anonymous-namespace `sample_integrand`, after `const double rho_cgs = disk->density_cgs(r, z, phi);` (line 46) add nothing, but change the integrand line (73) to:

```cpp
        const double rg = disk->r_g();   // geometric ds → cm: dz_cm = ds_geom * r_g
        for (size_t i = 0; i < channels_nu_obs.size(); ++i) {
            const double nu_emit = std::abs(g_factor) * channels_nu_obs[i];
            const double kabs    = luts.lookup_kappa_abs(nu_emit, rho_cgs, T_local);
            const double kes     = luts.lookup_kappa_es(rho_cgs, T_local);
            integrand[i] = (kabs + kes) * rho_cgs * abs_pue * rg;
        }
```

- [ ] **Step 6: Add the `r_g` factor to the spectral raymarch**

In `src/geodesic_tracer.cpp` `raymarch_volumetric_spectral`, change line 797 to carry `r_g` into the proper path length, and line 825 so the step control targets the same physical `dτ`:

```cpp
        const double ds_proper = std::abs(p_dot_u_emit) * std::abs(ds) * vol_disk_->r_g();
```
and
```cpp
        const double alpha_tot = (luts.lookup_kappa_abs(nu_med_emit, rho_cgs, T_turb)
                                + luts.lookup_kappa_es(rho_cgs, T_turb)) * rho_cgs * vol_disk_->r_g();
```

- [ ] **Step 7: Build and run the full volumetric suite**

Run: `cmake --build build --config Release --target test-volumetric && ./build/Release/test-volumetric`
Expected: `test_tau_matches_bvp` PASS (τ now O(1–100)); `test_no_horizontal_bands` still PASS (<0.25); `test_bvp_profile_not_collapsed`/`test_bvp_radial_smoothness` PASS. `test_tau_midplane_near_target` will now FAIL (it asserts the old τ≈100 calibration against the fake-cgs scale) — that is expected and fixed in Task 8; leave it failing for now and note it.

- [ ] **Step 8: Commit** (hand to human)

```
feat(disk): honest absolute-CGS density + log encoding + r_g optical depth

Retire normalize_density/rho_scale (densities are now absolute cgs from the
BVP). Store log(rho/rho_mid); density() interpolates in log and multiplies
by absolute rho_mid. Carry r_g into BOTH raymarch integrands so the optical
depth integral uses cm path length (dz_cm = ds_geom * r_g) — without this
absolute-cgs density would make every tau wrong by ~1e6. Emergent in-render
tau_mid now matches the BVP. (test_tau_midplane_near_target rewritten in a
later task.)
```

---

### Task 5: Retire `nested_refine` / `compare_columns` / `solve_column`

**Files:**
- Modify: `include/grrt/scene/volumetric_disk.h:236-267` (delete dead decls)
- Modify: `src/volumetric_disk.cpp` (delete dead defs)
- Test: `tests/test_volumetric.cpp:413-441`

- [ ] **Step 1: Rewrite the two tests that assert the retired machinery**

In `tests/test_volumetric.cpp`, replace `test_compare_columns_compiles` (lines 413-416) and `test_refine_n_z_caps_with_warning` (lines 418-441) with a single test (update the calls in `main()` accordingly — remove both old calls, add the new one):

```cpp
void test_refinement_machinery_retired() {
    std::printf("\n=== Richardson refinement retired: no n_z_cap/n_r_cap/nested warnings ===\n");
    const auto& disk = shared_disk_default();
    int refinement_warnings = 0;
    for (const auto& w : disk.warnings()) {
        if (w.code == "n_z_cap" || w.code == "n_r_cap" || w.code == "nested_refine_no_fixed_point")
            ++refinement_warnings;
    }
    if (refinement_warnings > 0) {
        std::printf("  FAIL: %d retired-refinement warnings present\n", refinement_warnings); failures++;
    } else { std::printf("  PASS: no refinement-cap warnings\n"); }
}
```

- [ ] **Step 2: Run to verify it fails to build**

Run: `cmake --build build --config Release --target test-volumetric`
Expected: still builds (the helpers still exist). The new test PASSES already (no refinement runs). This step confirms the test is wired; the deletion below is the real work.

- [ ] **Step 3: Delete the dead declarations**

In `include/grrt/scene/volumetric_disk.h`, delete: `struct ColumnSolution {...}` (lines 236-241), `solve_column(...)` (lines 243-246), `compare_columns(...)` (lines 249-250), `refine_n_z_globally()` (lines 252-256), `refine_n_r()` (lines 258-262), `nested_refine()` (lines 264-267), and the `normalize_density();` declaration in the construction-helpers block (line 275).

- [ ] **Step 4: Delete the dead definitions**

In `src/volumetric_disk.cpp`, delete the bodies of: `solve_column` (lines 660-950), `normalize_density` (lines 985-1051), `compare_columns` (lines 1196-1280), `refine_n_z_globally` (lines 1286-1355), `refine_n_r` (lines 1361-1429), `nested_refine` (lines 1435-1456). Also delete the now-unused flux-limiter helpers `lp_lambda`/`lp_eddington_factor` (lines 644-658) if no remaining caller references them (grep first: `grep -n lp_eddington_factor src/volumetric_disk.cpp`).

- [ ] **Step 5: Build the whole library and all tests**

Run: `cmake --build build --config Release`
Expected: clean build, no "unused" or "undefined" errors. Run `./build/Release/test-volumetric` → `test_refinement_machinery_retired` PASS, no regressions.

- [ ] **Step 6: Commit** (hand to human)

```
refactor(disk): delete retired solve_column/normalize_density/refinement

Remove the old hydrostatic solve_column, normalize_density/rho_scale
calibration, and the nested_refine/refine_n_*/compare_columns Richardson
apparatus (known-issue item 3 band-aid). The BVP march on a fixed grid
replaces all of it. ColumnSolution and the LP flux-limiter helpers go too.
```

---

### Task 6: Failure policy — Promptable truncation, Severe interior hole, Toomre Q, range guard

**Files:**
- Modify: `src/volumetric_disk.cpp` (`compute_vertical_profiles` non-converged handling; new validation in `validate_luts` or a new method)
- Test: `tests/test_volumetric.cpp`

**Policy (locked with the user):** `T_eff` ≥ 3000 K floor & BVP non-converges at the **outer edge** → Promptable truncation (no disk beyond). `T_eff` < 3000 K (model-validity floor) → Promptable truncation. Non-converged **interior** column (converged neighbours both sides) → **Severe**, and the constructor must surface it as a hard failure. No fabricated profile, ever.

- [ ] **Step 1: Write the failing tests**

In `tests/test_volumetric.cpp` add (call from `main()`):

```cpp
void test_validity_floor_truncates_promptable() {
    std::printf("\n=== cold outer edge below 3000K truncates with Promptable warning ===\n");
    // A supermassive BH with a large outer radius drives T_eff below the 3000 K
    // opacity-validity floor at the edge → Promptable truncation, not Severe abort.
    grrt::VolumetricParams p; p.mass_solar = 1e8; p.turbulence = 0.0;
    grrt::VolumetricDisk smbh(1.0, 0.998, 200.0, 1e6, p);   // large r_outer, cool edge
    bool promptable_trunc = false, severe = false;
    for (const auto& w : smbh.warnings()) {
        if (w.severity >= grrt::WarningSeverity::Promptable && w.code == "disk_truncated") promptable_trunc = true;
        if (w.severity == grrt::WarningSeverity::Severe) severe = true;
    }
    std::printf("  promptable_truncation=%d severe=%d\n", promptable_trunc, severe);
    // The cool edge must truncate (z_max=0 at the outer bin) and must NOT be Severe.
    if (severe) { std::printf("  FAIL: cold edge escalated to Severe\n"); failures++; }
    if (smbh.z_max_at(199.0) > 0.0 && smbh.temperature(199.0, 0.0) > 0.0
        && smbh.temperature(199.0,0.0) < 3000.0) {
        std::printf("  FAIL: sub-3000K column not truncated\n"); failures++;
    }
}

void test_toomre_q_diagnostic_absent_for_normal_disk() {
    std::printf("\n=== Toomre Q >> 1 for the rendered inner disk (no self-gravity warning) ===\n");
    const auto& disk = shared_disk_default();
    bool toomre_warn = false;
    for (const auto& w : disk.warnings()) if (w.code == "toomre_q_low") toomre_warn = true;
    if (toomre_warn) { std::printf("  FAIL: spurious Toomre-Q warning on a normal inner disk\n"); failures++; }
    else { std::printf("  PASS\n"); }
}
```

- [ ] **Step 2: Run to verify the truncation test fails**

Run: `cmake --build build --config Release --target test-volumetric && ./build/Release/test-volumetric`
Expected: `test_validity_floor_truncates_promptable` FAIL — no `disk_truncated` warning yet; sub-3000 K columns may currently still be solved/stored.

- [ ] **Step 3: Add the validity floor + truncation + interior-hole policy to the march**

In `src/volumetric_disk.cpp` `compute_vertical_profiles`, define a validity floor near the top of the function:

```cpp
    constexpr double T_EFF_FLOOR = 3000.0;   // opacity-model validity edge (atomic only)
```

In `make_column_inputs`, no change. In the march, replace the two outward/inward loop bodies and the post-loop with policy-aware handling. Replace the OUTWARD loop with:

```cpp
    bool truncated_outer = false;
    for (int ri = peak_idx + 1; ri < n_r_; ++ri) {
        if (truncated_outer) { z_max_lut_[ri] = 0.0; continue; }   // disk has ended; leave empty
        if (T_eff_lut_[ri] < T_EFF_FLOOR) {                        // model-validity edge
            emit(WarningSeverity::Promptable, "disk_truncated",
                 "T_eff below 3000K opacity-validity floor at r=" + std::to_string(r_at(ri)) +
                 "; disk truncated (continue = accept the smaller disk)");
            z_max_lut_[ri] = 0.0; truncated_outer = true; continue;
        }
        const std::vector<double>* warm = Uconv[ri-1].empty() ? nullptr : &Uconv[ri-1];
        auto s = solve_bin(ri, warm);
        if (s.converged) { store_column(ri, s, true); Uconv[ri] = pack(s); }
        else {
            emit(WarningSeverity::Promptable, "disk_truncated",
                 "BVP non-convergence at outer r=" + std::to_string(r_at(ri)) +
                 "; disk truncated (continue = accept the smaller disk)");
            z_max_lut_[ri] = 0.0; truncated_outer = true;
        }
    }
```

Replace the INWARD loop with (an interior non-convergence between two converged neighbours is a Severe hole):

```cpp
    for (int ri = peak_idx - 1; ri >= isco_idx; --ri) {
        const std::vector<double>* warm = Uconv[ri+1].empty() ? nullptr : &Uconv[ri+1];
        auto s = solve_bin(ri, warm);
        if (s.converged) { store_column(ri, s, true); Uconv[ri] = pack(s); }
        else {
            // Interior column failed with a converged outer neighbour — a hole, not an edge.
            emit(WarningSeverity::Severe, "bvp_interior_hole",
                 "BVP failed to converge at interior r=" + std::to_string(r_at(ri)) +
                 " (converged neighbour outside); cannot truncate a hole");
            z_max_lut_[ri] = 0.0;
        }
    }
```

(Also change the peak-anchor `store_column(peak_idx, s, false)` and the plunging copy to use `true` — done in Task 4; if Task 6 lands after Task 4 they are already `true`. Ensure both outward/inward `store_column` use `store_log=true`.)

Add `#include <string>` to `src/volumetric_disk.cpp` if not present (it is, via the header).

- [ ] **Step 4: Make a Severe `bvp_interior_hole` a hard construction failure**

In the constructor, after `validate_luts();` (line 176), add:

```cpp
    // Approach A "fail or succeed": an interior BVP hole is unrenderable. Abort
    // loudly (the C API translates the exception to an error code).
    for (const auto& w : warnings_) {
        if (w.severity == WarningSeverity::Severe && w.code == "bvp_interior_hole") {
            throw std::runtime_error("VolumetricDisk: " + w.message);
        }
    }
```

Add `#include <stdexcept>` to `src/volumetric_disk.cpp`.

- [ ] **Step 5: Add the Toomre Q diagnostic**

Add a method and call it from the constructor (after `validate_luts();`, before the Severe check). Declaration in `volumetric_disk.h` private helpers: `void validate_toomre_q();`. Definition in `src/volumetric_disk.cpp`:

```cpp
void VolumetricDisk::validate_toomre_q() {
    using namespace constants;
    // Toomre Q = Omega * c_s / (pi G Sigma). Q >> 1 in the rendered inner disk.
    // Sigma ~ 2 * rho_mid * z_max (cgs). Warn once if any orbiting column has Q < 2.
    auto r_at = [&](int i){ return r_min_ + (r_outer_ - r_min_) * i / (n_r_ - 1); };
    for (int i = 0; i < n_r_; ++i) {
        const double r = r_at(i);
        if (r < r_isco_ || z_max_lut_[i] <= 0.0 || rho_mid_lut_[i] <= 0.0) continue;
        const double T_mid = T_profile_lut_[i * n_z_ + 0];
        if (T_mid <= 0.0) continue;
        const double cs = std::sqrt(k_B * T_mid / (mu_fully_ionized * m_p));   // cm/s
        const double Omega_cgs = omega_orb(r) * c_cgs / r_g_;                  // 1/s
        const double z_max_cm = z_max_lut_[i] * r_g_;
        const double Sigma = 2.0 * rho_mid_lut_[i] * z_max_cm;                 // g/cm^2
        if (Sigma <= 0.0) continue;
        const double Q = Omega_cgs * cs / (M_PI_GRRT * G_cgs * Sigma);
        if (Q < 2.0) {
            emit(WarningSeverity::Warning, "toomre_q_low",
                 "Toomre Q=" + std::to_string(Q) + " < 2 at r=" + std::to_string(r) +
                 " (thin-disk/self-gravity assumption marginal)");
            return;  // one warning per construction
        }
    }
}
```

Use `std::numbers::pi` (already included via `<numbers>`): replace `M_PI_GRRT` with `std::numbers::pi`.

- [ ] **Step 6: Build and run**

Run: `cmake --build build --config Release && ./build/Release/test-volumetric`
Expected: `test_validity_floor_truncates_promptable` PASS (Promptable `disk_truncated`, no Severe); `test_toomre_q_diagnostic_absent_for_normal_disk` PASS; existing tests pass. The SMBH smoke construction must not throw.

- [ ] **Step 7: Commit** (hand to human)

```
feat(disk): BVP failure policy — truncate at edge, Severe interior hole

No analytic fallback. Outer-edge non-convergence and sub-3000K columns emit
a Promptable "disk_truncated" (continue = accept a smaller disk). An interior
non-convergence between converged neighbours is a Severe "bvp_interior_hole"
that aborts construction (C API returns an error). Add a Toomre-Q diagnostic
(Warning if Q<2 in the orbiting region).
```

---

### Task 7: Wire `compute_sigma_s_phys` to real density; drop distorting clamps

**Files:**
- Modify: `src/volumetric_disk.cpp:1057-1115` (`compute_sigma_s_phys`)
- Test: `tests/test_volumetric.cpp` (existing `test_sigma_s_phys_in_range` at line 312 must still pass)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_volumetric.cpp` (call from `main()`):

```cpp
void test_sigma_s_uses_real_density() {
    std::printf("\n=== beta/sigma_s computed from absolute rho_mid (no fake-cgs clamp) ===\n");
    const auto& disk = shared_disk_default();
    // sigma_s_phys must be finite, in the physical band, and reflect a real beta.
    const double s = disk.sigma_s_phys();
    std::printf("  sigma_s_phys=%.4f\n", s);
    if (!std::isfinite(s) || s <= 0.0) { std::printf("  FAIL: non-physical sigma_s\n"); failures++; }
    if (s < 0.30 || s > 0.75) { std::printf("  FAIL: sigma_s outside b-range [0.35,0.70]+slack\n"); failures++; }
    else { std::printf("  PASS\n"); }
}
```

- [ ] **Step 2: Run; record whether it passes on current code**

Run: `cmake --build build --config Release --target test-volumetric && ./build/Release/test-volumetric`
Expected: may pass or fail depending on the leftover `[1e-18,1e-6]` clamp distorting `β`. Record.

- [ ] **Step 3: Remove the fake-cgs clamps from `compute_sigma_s_phys`**

In `src/volumetric_disk.cpp` `compute_sigma_s_phys`, replace lines 1080-1081:

```cpp
        double rho_mid_cgs = rho_scale_ * rho_mid_lut_[peak_idx];
        rho_mid_cgs = std::clamp(rho_mid_cgs, 1e-18, 1e-6);
```

with (densities are absolute now; `rho_scale_≡1`; the only legitimate guard is positivity):

```cpp
        const double rho_mid_cgs = std::max(rho_mid_lut_[peak_idx], 1e-300);
```

Replace the `lookup_mu` clamp on line 1083 (`opacity_luts_.lookup_mu(rho_mid_cgs, ...)`) — keep the T clamp (LUT edge, legitimate) but drop any ρ clamp; pass `rho_mid_cgs` directly (the LUT's own `log_interp` clamps to table edges). Use `T_mid` instead of the proportional `T_mid` derived from `tau_mid` — the real midplane temperature is the stored profile value:

```cpp
        const double T_mid = T_profile_lut_[peak_idx * n_z_ + 0];
        double mu = opacity_luts_.lookup_mu(rho_mid_cgs, std::clamp(T_mid, 3000.0, 1e8));
```

(Delete the now-unused `T_mid4`/`T_mid` block at lines 1077-1079 that derived T_mid from `tau_mid` — the BVP gives the real midplane T.)

- [ ] **Step 4: Build and run**

Run: `cmake --build build --config Release --target test-volumetric && ./build/Release/test-volumetric`
Expected: `test_sigma_s_uses_real_density` and `test_sigma_s_phys_in_range` PASS; `test_density_lognormal_mean` PASS (depends on σ_s).

- [ ] **Step 5: Commit** (hand to human)

```
refactor(disk): compute beta/sigma_s from absolute midplane rho and T

compute_sigma_s_phys reads the BVP's real rho_mid (absolute cgs) and the
stored midplane T instead of the rho_scale*proportional value clamped into
[1e-18,1e-6]. The pressure-regime beta is now physically correct.
```

---

### Task 8: Known-issue item 2 — make `test_tau_midplane_near_target` honest

**Files:**
- Modify: `tests/test_volumetric.cpp:507-557`

The old test asserts τ≈100 against the fake-cgs `tau_mid` knob. With Approach A, τ_mid is **emergent**, not a target. Rewrite the test to assert τ is well-defined, dimensionally consistent (uses `r_g`), and matches the BVP — i.e., fold it into the honest check.

- [ ] **Step 1: Replace the test body**

In `tests/test_volumetric.cpp`, replace `test_tau_midplane_near_target` (lines 507-557) with:

```cpp
void test_tau_midplane_well_defined() {
    std::printf("\n=== item 2 retired: midplane tau is well-defined (emergent, honest r_g) ===\n");
    const auto& disk = shared_disk_tau_test();
    // Peak-flux radius by midplane density.
    double best_r = disk.r_isco() + 0.5, best_rho = 0.0;
    for (int i = 0; i < 80; ++i) {
        const double r = disk.r_isco() + (30.0 - disk.r_isco()) * i / 79.0;
        const double rho = disk.density_cgs(r, 0.0, 0.0);
        if (rho > best_rho) { best_rho = rho; best_r = r; }
    }
    const double r = best_r, zm = disk.z_max_at(r), rg = disk.r_g();
    const auto& opa = disk.opacity_luts();
    const int N = 400; const double dz_geom = zm / (N - 1);
    double tau = 0.0;
    for (int i = 0; i + 1 < N; ++i) {
        const double za=i*dz_geom, zb=(i+1)*dz_geom;
        const double ra=disk.density_cgs(r,za,0.0), rb=disk.density_cgs(r,zb,0.0);
        const double Ta=std::max(disk.temperature(r,za),3000.0), Tb=std::max(disk.temperature(r,zb),3000.0);
        const double ka=opa.lookup_kappa_ross(ra,Ta)+opa.lookup_kappa_es(ra,Ta);
        const double kb=opa.lookup_kappa_ross(rb,Tb)+opa.lookup_kappa_es(rb,Tb);
        tau += 0.5*(ka*ra + kb*rb)*(dz_geom*rg);
    }
    std::printf("  emergent midplane->surface tau at r=%.2f: %.3f\n", r, tau);
    // Item 2 was a "convention mismatch" (tau=403 vs 100). The fix is not a number
    // but well-definedness: a finite, physical O(1-1e3) optical depth.
    if (!std::isfinite(tau) || tau <= 0.0 || tau > 1e4) {
        std::printf("  FAIL: tau not well-defined/physical\n"); failures++;
    } else { std::printf("  PASS\n"); }
}
```

Update the call in `main()` from `test_tau_midplane_near_target();` to `test_tau_midplane_well_defined();`.

- [ ] **Step 2: Build and run**

Run: `cmake --build build --config Release --target test-volumetric && ./build/Release/test-volumetric`
Expected: PASS. The fake-cgs comment block (old lines 533-536) is gone.

- [ ] **Step 3: Commit** (hand to human)

```
test(disk): retire known-issue item 2 — tau is emergent and well-defined

Replace test_tau_midplane_near_target (asserted the fake-cgs tau_mid=100
knob) with test_tau_midplane_well_defined: the midplane optical depth is
now a finite, physical, r_g-consistent emergent value, not a target.
```

---

### Task 9: Integration sweep — dump, edge-on render, budget; surface before/after

**Files:**
- Modify: `tools/dump_disk_lut.cpp` (extend the "hole" check)
- Test: `tests/test_volumetric.cpp` (construction-time guard)

- [ ] **Step 1: Extend the LUT dump's hole detector**

In `tools/dump_disk_lut.cpp`, after writing the CSV (after the loop ending near line 65), add a collapse/hole summary to stdout:

```cpp
    // Approach A health summary: count radial holes (z_max<=0 inside the orbiting
    // region) and "collapsed" columns (profile drops below 1e-3 within 2 bins).
    int holes = 0, collapsed = 0, orbiting = 0;
    for (int ri = 0; ri < n_r; ++ri) {
        const double r = disk.r_min() + (disk.r_max() - disk.r_min()) * ri / (n_r - 1);
        if (r < disk.r_isco()) continue;
        orbiting++;
        if (zmax_lut[ri] <= 0.0) { holes++; continue; }
        // prof row is normalized; bin 2 / bin 0 ratio
        const double p0 = prof[ri * disk.vertical_bins() + 0];
        const double p2 = prof[ri * disk.vertical_bins() + std::min(2, disk.vertical_bins()-1)];
        // stored as log(rho/rho_mid): p0~0; collapsed if p2 << 0 within 2 bins
        if (std::exp(p2 - p0) < 1e-3) collapsed++;
    }
    std::printf("HEALTH: orbiting=%d holes=%d collapsed=%d\n", orbiting, holes, collapsed);
    if (holes > 0 || collapsed > orbiting / 10) {
        std::printf("HEALTH: WARNING — holes or widespread collapse detected\n");
    }
```

- [ ] **Step 2: Write a construction-time budget test**

In `tests/test_volumetric.cpp` add (call from `main()`):

```cpp
void test_construction_time_budget() {
    std::printf("\n=== construction stays within a few minutes (BVP march) ===\n");
    // shared_disk_default() is already built once; this just asserts the smoke
    // sweep's stellar-mass canonical case constructs without Severe and is usable.
    grrt::VolumetricParams vp; vp.turbulence = 0.0;
    const auto t0 = std::clock();
    grrt::VolumetricDisk d(1.0, 0.998, 30.0, 1e7, vp);
    const double secs = double(std::clock() - t0) / CLOCKS_PER_SEC;
    std::printf("  build time ~%.1f s (n_r=%d)\n", secs, d.radial_bins());
    // Generous ceiling — the spec budget is "a few minutes"; flag a gross regression.
    if (secs > 300.0) { std::printf("  FAIL: construction exceeded 5 minutes\n"); failures++; }
    else { std::printf("  PASS\n"); }
}
```

Add `#include <ctime>` to `tests/test_volumetric.cpp` if not present.

- [ ] **Step 3: Build, run the dump and the suite**

Run:
```
cmake --build build --config Release
./build/Release/test-volumetric
./build/Release/dump-disk-lut disk_lut_dump.csv
```
Expected: `test-volumetric` 0 failures; dump prints `HEALTH: orbiting=… holes=0 collapsed=0`.

- [ ] **Step 4: Edge-on visual render (manual, surface before/after to the human)**

Run the spec's edge-on scene and inspect for banding:
```
./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 --observer-theta 80 \
  --fov 30 --background black --disk-volumetric 1 --disk-turbulence 0.4 --samples 30 \
  --disk-temperature 1e7 --disk-outer 20 --width 512 --height 512 --output edge_on_bvp.png
```
Capture the image. **Surface the before/after and the banding metric to the human** (do not silently recalibrate the 0.25 threshold). This step's "pass" is the human's visual confirmation.

- [ ] **Step 5: Commit** (hand to human)

```
test(disk): integration sweep — LUT health dump + construction budget

dump-disk-lut now reports holes/collapsed columns (HEALTH line). Add a
construction-time budget guard. Edge-on render captured for human review of
banding (metric stays < 0.25).
```

---

## After all tasks

- [ ] **Final review:** dispatch a code-quality reviewer subagent (sonnet/opus) over the whole branch diff against the spec's success criteria (§14): r_g honest throughout; BVP converges across the orbiting region with no fallback; profile non-collapsed/smooth; `normalize_density`/`nested_refine`/`compare_columns`/`[1e-18,1e-6]` clamps gone; raymarch fixes untouched; construction within budget; CUDA deferral still documented.
- [ ] **CUDA note:** confirm `cuda/cuda_vol_host_data.cpp` is untouched and the `CUDA==CPU` invariant remains documented as suspended (spec §11/§16) — out of scope here.
- [ ] **Update the refinements doc** if the march left any columns relying on the cold-start fallback path (refinement 1b) — record the real convergence rate observed.
- [ ] **Finish the branch** via `superpowers:finishing-a-development-branch` once the human approves the edge-on render.

## Self-review notes (author)

- **Spec coverage:** §15 phase 4 → Tasks 3,4 (resample + log + read path + retire normalize). Phase 5 → Tasks 5,6,7 (retire refinement, σ_s real density, Toomre Q). Phase 6 → Tasks 8,9 (item 2, integration). The `r_g` optical-depth factor (not separately phased in the spec but implied by "honest cgs / τ dimensionally consistent", §5/§12/§14.1) → Task 4.
- **Atomicity:** Task 4 must be one commit (unit system can't be half-switched). Task 3 deliberately keeps `normalize_density` so every prior commit stays green.
- **Failure policy** (user-locked): no fallback; Promptable truncation at edge/validity-floor; Severe interior hole (throws). Tested in Task 6.
- **Type consistency:** `make_column_inputs(int,double)`, `store_column(int,const ColumnBVPSolution&,bool)`, `kColumnNodes`, `validate_toomre_q()` are used consistently across Tasks 3/4/6. `solve_column_bvp(in,op,warm)` 3-arg form from Task 1 used by the march in Task 3.
