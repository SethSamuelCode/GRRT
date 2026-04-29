# BPT72 Taper + Adaptive DPRK45 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the heuristic Gaussian plunging-region `taper(r)` with BPT72 mass-conservation, replace fixed-step RK4 in `solve_column` with adaptive Dormand-Prince RK4(5), and fix the tau-midplane test's units mismatch — making all four currently-failing/regressing tests pass and retiring all five Promptable warnings on healthy construction.

**Architecture:** Three independent surgical changes, ordered by risk:
1. BPT72 taper (small, isolated — replaces ~5 lines of `taper()` body, removes `taper_width_` member, updates one test threshold).
2. Adaptive DPRK45 inside `solve_column`'s vertical rho integration (medium — replaces ~25 lines of fixed-step RK4 with ~70 lines of adaptive DP45, no extraction into a shared utility because the existing DP45 in `rk4.h` is GeodesicState-coupled).
3. Tau test units fix (trivial — `density()` → `density_cgs()` plus clamp-range update).

**Tech Stack:** C++23, CMake, OpenMP, plain C++ test executables (no external test framework), Windows VS2022 + Linux GCC/Clang.

**Reference spec:** `docs/superpowers/specs/2026-04-29-bpt72-taper-and-dprk45.md` — read before starting.

**Build commands** used throughout:
- Build: `cmake --build build --config Release`
- Run tests (Windows): `./build/Release/test-volumetric`
- Run tests (Linux): substitute `./build/test-volumetric`

The plan below uses the Windows form; substitute as needed for Linux.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `include/grrt/scene/volumetric_disk.h` | Type definitions; `taper_width_` member removed; `taper_width()` accessor preserved as compute-on-demand legacy stub for CUDA host-data layout |
| `src/volumetric_disk.cpp` | `taper()` body replaced (BPT72 mass conservation); inner rho-RK4 in `solve_column` replaced (adaptive DP45); `taper_width_ = ...` constructor line removed; one `bins_per_gradient` heuristic line updated |
| `tests/test_volumetric.cpp` | `test_taper` horizon threshold updated; `test_tau_midplane_near_target` units fixed |

No new files. The Dormand-Prince RK4(5) Butcher tableau is inlined into `solve_column` directly — the existing one in `include/grrt/geodesic/rk4.h` is for `GeodesicState` (8-component) and isn't generic enough to share without a substantial refactor that's out of scope.

---

## Task 1: Replace Gaussian taper with BPT72 mass-conservation taper

**Goal:** The plunging-region taper becomes a physically-derived mass-conservation profile along the BPT72 plunging geodesic. The `h_jump` Promptable warning retires. The Gaussian width parameter goes away entirely.

**Files:**
- Modify: `include/grrt/scene/volumetric_disk.h` (remove `taper_width_` member; preserve `taper_width()` accessor as legacy stub)
- Modify: `src/volumetric_disk.cpp` (replace `taper()` body; remove `taper_width_` init line; update `bins_per_gradient` line)
- Modify: `tests/test_volumetric.cpp` (loosen-and-tighten `test_taper` horizon threshold)

- [ ] **Step 1: Confirm baseline test failures and run state**

Run: `./build/Release/test-volumetric 2>&1 | grep -E "FAIL|=== [0-9]+ failures"`

Expected: 3 failures — `test_density_profile`, `test_density_smooth_across_zmax`, `test_tau_midplane_near_target`. `test_taper` should currently PASS (the regression we saw earlier was from the now-reverted Task 1 of the previous plan).

If `test_taper` is NOT in the currently-passing set, halt and report — the working tree is in an unexpected state.

- [ ] **Step 2: Update `test_taper` to expect the BPT72-derived shape**

In `tests/test_volumetric.cpp`, find `test_taper()` (around line 57). Change the horizon assertion threshold from `> 0.1` to `> 0.05`:

```cpp
void test_taper() {
    std::printf("\n=== ISCO taper ===\n");
    grrt::VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);
    check("taper(r_isco)", disk.taper(disk.r_isco()), 1.0, 0.01);
    check("taper(r_isco+1)", disk.taper(disk.r_isco()+1.0), 1.0, 0.01);
    double t_hor = disk.taper(disk.r_horizon());
    if (t_hor > 0.05) { std::printf("  FAIL: BPT72 taper at horizon should be near zero (got %.4e)\n", t_hor); failures++; }
    else { std::printf("  PASS: taper(horizon)=%.4e\n", t_hor); }
}
```

The two `check()` calls at ISCO are unchanged. The horizon threshold tightens from `0.1` to `0.05` (the BPT72 shape goes to 0 at horizon by construction; 0.05 is a numerical-floor allowance).

- [ ] **Step 3: Run test_taper to verify it now FAILS**

Run: `cmake --build build --config Release && ./build/Release/test-volumetric 2>&1 | grep -A1 "ISCO taper"`

Expected: FAIL — the current Gaussian with width `(r_isco-r_horizon)/3` gives `taper(r_horizon) ≈ 1.2e-4`, which would actually pass `> 0.05`. So this might already PASS. If it passes, that's also fine — proceed; the test will continue to pass with the new BPT72 shape.

- [ ] **Step 4: Replace `taper()` body with BPT72 mass conservation**

In `src/volumetric_disk.cpp`, find `VolumetricDisk::taper()` (around line 182). Replace the entire function body with:

```cpp
double VolumetricDisk::taper(double r) const {
    if (r >= r_isco_) return 1.0;
    if (r <= r_horizon_) return 0.0;

    // Mass conservation along the BPT72 plunging geodesic:
    //   ρ(r) ∝ 1 / (r · |u^r(r)|)
    // Normalize so taper saturates to 1 at ISCO via a regulator at r_isco·EPS.
    constexpr double EPS = 0.99;
    constexpr double THETA = 1.5707963267948966;  // pi/2, equatorial plane

    double ut, ur_ref, uphi;
    plunging_velocity(r_isco_ * EPS, THETA, ut, ur_ref, uphi);
    const double r_ref = r_isco_ * EPS;
    const double denom_ref = r_ref * std::abs(ur_ref);
    if (denom_ref <= 0.0) return 1.0;

    double ur;
    plunging_velocity(r, THETA, ut, ur, uphi);
    const double denom = r * std::abs(ur);
    if (denom <= 0.0) return 1.0;

    return std::clamp(denom_ref / denom, 0.0, 1.0);
}
```

The function uses the existing private method `plunging_velocity(r, theta, ut, ur, uphi)` (already at volumetric_disk.cpp:155). It writes `ut`, `ur` (negative for infall), `uphi`. We use only the magnitude of `ur`.

- [ ] **Step 5: Remove `taper_width_` initialization in the constructor**

In `src/volumetric_disk.cpp`, in the `VolumetricDisk` constructor, find the line:
```cpp
    taper_width_ = (r_isco_ - r_horizon_) / 3.0;
```
(currently at volumetric_disk.cpp:53)

Delete it. The new `taper()` doesn't read `taper_width_`.

- [ ] **Step 6: Update the `bins_per_gradient` sizing heuristic**

In `src/volumetric_disk.cpp`, in the constructor, find the line that uses `taper_width_` for radial-bin sizing (currently around line 73):
```cpp
        n_r_ = std::clamp(params_.bins_per_gradient *
                          static_cast<int>(std::ceil((r_outer_ - r_min_) / std::max(taper_width_, 0.01))),
                          params_.min_n_r, params_.max_n_r);
```

Replace with a direct expression (since `taper_width_` no longer exists as a member):
```cpp
        const double sizing_scale = std::max((r_isco_ - r_horizon_) / 3.0, 0.01);
        n_r_ = std::clamp(params_.bins_per_gradient *
                          static_cast<int>(std::ceil((r_outer_ - r_min_) / sizing_scale)),
                          params_.min_n_r, params_.max_n_r);
```

This preserves the previous sizing behavior — the `(r_isco - r_horizon)/3` was always just a heuristic length scale for "how many bins per radial gradient", unrelated to the physical taper shape.

- [ ] **Step 7: Remove `taper_width_` member declaration and convert accessor to legacy stub**

In `include/grrt/scene/volumetric_disk.h`, find the private member declaration:
```cpp
    double taper_width_;        ///< Gaussian taper width inside ISCO
```
(currently at line 175)

Delete this line.

Find the public accessor:
```cpp
    double taper_width() const { return taper_width_; }
```
(currently at line 145)

Replace it with a compute-on-demand legacy stub (keeps `cuda/cuda_vol_host_data.cpp:27` compiling without changes):
```cpp
    /// Legacy accessor — preserved for CUDA host-data layout compatibility.
    /// Returns the heuristic length scale that was used by the old Gaussian
    /// taper. New code should not depend on this value; the BPT72 taper
    /// has no width parameter.
    double taper_width() const { return (r_isco_ - r_horizon_) / 3.0; }
```

- [ ] **Step 8: Update the docstring on `taper()` in the header**

In `include/grrt/scene/volumetric_disk.h`, find the current docstring above the `double taper(double r) const;` declaration (around line 101):
```cpp
    /// ISCO taper factor: 1 for r >= r_isco, Gaussian decay inside.
    double taper(double r) const;
```

Replace with:
```cpp
    /// ISCO taper factor: 1 for r >= r_isco, decays via BPT72 mass
    /// conservation along the plunging geodesic for r < r_isco.
    /// Reaches 0 at r_horizon by construction.
    double taper(double r) const;
```

- [ ] **Step 9: Build**

Run: `cmake --build build --config Release`

Expected: clean build. Files recompiled: `volumetric_disk.cpp`, possibly `cuda/cuda_vol_host_data.cpp` if CUDA is enabled.

If the build fails because some other site references `taper_width_` directly (without going through the accessor), search and fix:
```bash
grep -rn "taper_width_" src/ include/ cuda/
```
Expected: matches only inside `volumetric_disk.h` (the now-deleted member declaration in your working tree before the build) and the accessor body. If you find an external reference, route it through the public `taper_width()` accessor.

- [ ] **Step 10: Run tests**

Run: `./build/Release/test-volumetric 2>&1 | grep -E "ISCO taper|h_jump|FAIL|=== [0-9]+ failures"`

Expected:
- `ISCO taper`: PASS, with `taper(horizon)=` printed as a very small number (likely < 1e-4).
- `h_jump` Promptable: should NO LONGER fire on construction. (Search for `h_jump` in the full test output; expect no matches.)
- Total failures: still 3 (`test_density_profile`, `test_density_smooth_across_zmax`, `test_tau_midplane_near_target` remain — those are addressed in Tasks 2 and 3).

- [ ] **Step 11: Commit**

```bash
git add include/grrt/scene/volumetric_disk.h src/volumetric_disk.cpp tests/test_volumetric.cpp
git commit -m "feat(volumetric): BPT72 mass-conservation taper replaces Gaussian"
```

---

## Task 2: Replace fixed-step RK4 with adaptive DP45 in `solve_column`'s rho integration

**Goal:** The vertical density ODE inside `solve_column` is integrated with adaptive Dormand-Prince RK4(5), so adjacent radial columns converge on the same cliff position to machine precision. This eliminates the cross-column blend artifact in `interp_2d`, making the bulk-disk density tests pass.

**Files:**
- Modify: `src/volumetric_disk.cpp` (`solve_column` — replace ~25 lines of fixed-step RK4 with ~80 lines of adaptive DP45)

- [ ] **Step 1: Confirm the bulk-disk tests are still failing**

Run: `./build/Release/test-volumetric 2>&1 | grep -E "Density profile|Density smooth"`

Expected:
- `Density profile (no noise)`: FAIL with `density should decrease with height`.
- `Density smooth across z_max`: FAIL with `rho_below z_max too large`.

These are the tests that DP45 is supposed to fix. Recording baseline diagnostics:
```bash
./build/Release/test-volumetric 2>&1 | grep -E "rho\(mid\)|rho_below"
```
Record the exact values printed. They become the before/after comparison.

- [ ] **Step 2: Locate the fixed-step RK4 block to replace**

In `src/volumetric_disk.cpp`, inside `VolumetricDisk::solve_column()`, find the comment `// Pass 4: rho(z) RK4 outward`. Below it is the fixed-step RK4 block, currently approximately:

```cpp
        // Pass 4: rho(z) RK4 outward
        std::vector<double> d_cs2_dz(n_z, 0.0), d_fE_dz(n_z, 0.0);
        for (int zi = 0; zi < n_z; ++zi) {
            // ... central differences for d_cs2_dz, d_fE_dz ...
        }

        out.rho_z[0] = 1.0;
        for (int zi = 0; zi < n_z - 1; ++zi) {
            const double z_here = zi * dz;
            const double rho_here = out.rho_z[zi];

            auto rhs = [&](double z_eval, double rho_eval) -> double {
                // ... existing rhs body ...
            };

            const double k1 = dz * rhs(z_here, rho_here);
            const double k2 = dz * rhs(z_here + 0.5*dz, std::max(rho_here + 0.5*k1, RHO_FLOOR));
            const double k3 = dz * rhs(z_here + 0.5*dz, std::max(rho_here + 0.5*k2, RHO_FLOOR));
            const double k4 = dz * rhs(z_here + dz,     std::max(rho_here + k3,     RHO_FLOOR));
            out.rho_z[zi+1] = std::max(rho_here + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0,
                                        RHO_FLOOR);
        }
```

The d_cs2_dz / d_fE_dz precomputation block stays. The `rhs` lambda body stays. Only the fixed-step RK4 marching loop (the second `for` loop) is replaced.

- [ ] **Step 3: Replace the fixed-step RK4 marching loop with adaptive DP45**

In `src/volumetric_disk.cpp`, inside `solve_column`, replace the marching loop (the `for (int zi = 0; zi < n_z - 1; ++zi)` block including its lambda) with:

```cpp
        // Define rhs lambda once (used by all DP45 stages)
        auto rhs = [&](double z_eval, double rho_eval) -> double {
            const double z_frac = z_eval / dz;
            const int idx = std::clamp(static_cast<int>(z_frac), 0, n_z - 2);
            const double t = z_frac - idx;
            const double cs2 = k_B * ((1.0-t)*out.T_z[idx] + t*out.T_z[idx+1])
                             / (((1.0-t)*mu_z[idx] + t*mu_z[idx+1]) * m_p);
            const double dcs2 = (1.0-t)*d_cs2_dz[idx] + t*d_cs2_dz[idx+1];
            const double dfE  = (1.0-t)*d_fE_dz[idx]  + t*d_fE_dz[idx+1];
            if (cs2 < 1e-30) return 0.0;
            const double cs2_geom = cs2 / (c_cgs * c_cgs);
            const double dcs2_geom = dcs2 / (c_cgs * c_cgs);
            const double dfE_geom = dfE / (rho_cgs_ref * c_cgs * c_cgs);
            return (-rho_eval * Omz2 * z_eval - rho_eval * dcs2_geom - dfE_geom)
                   / std::max(cs2_geom, 1e-30);
        };

        // Adaptive Dormand-Prince RK4(5) integration of dρ/dz from z=0 to z_max.
        // Variable step size handles the photosphere cliff; result is sampled
        // onto the uniform n_z grid for storage in rho_z[].
        const double dp45_tol = std::max(params_.target_lut_eps, 1e-8);
        const double h_floor = z_max * 1e-12;

        std::vector<double> z_samples;
        std::vector<double> rho_samples;
        z_samples.reserve(static_cast<size_t>(n_z) * 4);
        rho_samples.reserve(static_cast<size_t>(n_z) * 4);
        z_samples.push_back(0.0);
        rho_samples.push_back(1.0);

        double z_cur = 0.0;
        double rho_cur = 1.0;
        double h = dz;  // initial suggestion = uniform-grid spacing

        while (z_cur < z_max) {
            h = std::min(h, z_max - z_cur);

            const double k1 = rhs(z_cur, rho_cur);
            const double k2 = rhs(z_cur + h/5.0,
                                  std::max(rho_cur + h*k1/5.0, RHO_FLOOR));
            const double k3 = rhs(z_cur + 3.0*h/10.0,
                                  std::max(rho_cur + h*(3.0*k1/40.0 + 9.0*k2/40.0), RHO_FLOOR));
            const double k4 = rhs(z_cur + 4.0*h/5.0,
                                  std::max(rho_cur + h*(44.0*k1/45.0 - 56.0*k2/15.0 + 32.0*k3/9.0), RHO_FLOOR));
            const double k5 = rhs(z_cur + 8.0*h/9.0,
                                  std::max(rho_cur + h*(19372.0*k1/6561.0 - 25360.0*k2/2187.0
                                                        + 64448.0*k3/6561.0 - 212.0*k4/729.0), RHO_FLOOR));
            const double k6 = rhs(z_cur + h,
                                  std::max(rho_cur + h*(9017.0*k1/3168.0 - 355.0*k2/33.0
                                                        + 46732.0*k3/5247.0 + 49.0*k4/176.0
                                                        - 5103.0*k5/18656.0), RHO_FLOOR));

            const double rho_next = rho_cur + h*(35.0*k1/384.0 + 500.0*k3/1113.0
                                                 + 125.0*k4/192.0 - 2187.0*k5/6784.0
                                                 + 11.0*k6/84.0);
            const double k7 = rhs(z_cur + h, std::max(rho_next, RHO_FLOOR));

            const double err = h * std::abs(71.0*k1/57600.0 - 71.0*k3/16695.0
                                            + 71.0*k4/1920.0 - 17253.0*k5/339200.0
                                            + 22.0*k6/525.0 - k7/40.0);
            const double scale = std::max(std::abs(rho_cur), RHO_FLOOR);
            const double err_rel = err / scale;

            if (err_rel < dp45_tol || h <= h_floor) {
                // Accept
                z_cur += h;
                rho_cur = std::max(rho_next, RHO_FLOOR);
                z_samples.push_back(z_cur);
                rho_samples.push_back(rho_cur);
                const double scale_factor = (err_rel > 1e-30)
                    ? std::clamp(0.9 * std::pow(dp45_tol / err_rel, 0.2), 0.2, 5.0)
                    : 5.0;
                h *= scale_factor;
            } else {
                // Reject — shrink and retry
                h *= std::max(0.2, 0.9 * std::pow(dp45_tol / err_rel, 0.2));
            }
        }

        // Sample rho onto the uniform n_z grid via linear interpolation
        out.rho_z[0] = 1.0;
        for (int zi = 1; zi < n_z; ++zi) {
            const double z_target = zi * dz;
            auto it = std::upper_bound(z_samples.begin(), z_samples.end(), z_target);
            if (it == z_samples.begin()) {
                out.rho_z[zi] = rho_samples[0];
            } else if (it == z_samples.end()) {
                out.rho_z[zi] = rho_samples.back();
            } else {
                const size_t hi = static_cast<size_t>(std::distance(z_samples.begin(), it));
                const size_t lo = hi - 1;
                const double span = z_samples[hi] - z_samples[lo];
                const double t = (span > 0.0) ? (z_target - z_samples[lo]) / span : 0.0;
                out.rho_z[zi] = std::max((1.0 - t) * rho_samples[lo] + t * rho_samples[hi],
                                          RHO_FLOOR);
            }
        }
```

The `dz` variable referenced in `rhs` is the uniform-grid spacing computed earlier in the outer iteration (`const double dz = z_max / (n_z - 1);`). The interpolation in `rhs` uses `dz` to index into the auxiliary arrays `out.T_z`, `mu_z`, `d_cs2_dz`, `d_fE_dz`, which are stored on the same uniform grid. The DP45 integration evaluates `rhs` at arbitrary `z_eval` values and the existing `rhs` lambda already handles arbitrary z via index-clamping and linear interpolation — no modification needed.

`std::upper_bound`, `std::distance` come from `<algorithm>` and `<iterator>` respectively. `<algorithm>` is already included. Add `#include <iterator>` if it's not already in the includes block at the top of `src/volumetric_disk.cpp`.

- [ ] **Step 4: Build**

Run: `cmake --build build --config Release`

Expected: clean build, only the pre-existing `D9025` warning.

If the build fails due to a missing include (`std::distance`, `std::upper_bound`), add the missing include and rebuild.

- [ ] **Step 5: Run all tests**

Run: `./build/Release/test-volumetric 2>&1 | grep -E "rho\(mid\)|rho_below|FAIL|=== [0-9]+ failures"`

Expected:
- `Density profile (no noise)`: PASS. The printed `rho(mid), rho(1H), rho(3H)` should now be monotonically decreasing.
- `Density smooth across z_max`: PASS. The printed `rho_below / rho_mid` should be much smaller than `1e-10`.
- `test_tau_midplane_near_target`: still FAIL (fixed in Task 3).
- Total failures: down from 3 to 1.

If `Density profile` still fails: capture the printed rho values and the construction log's `Refinement done: n_r=X, n_z=Y` line. Most likely cause: the DP45 tolerance is too loose. Try tightening `dp45_tol` from `1e-8` floor to `1e-10` in Step 3's code and rebuilding.

- [ ] **Step 6: Verify refinement converges below the cap**

Run a render and check the construction log:
```bash
./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 --observer-theta 80 --disk-volumetric --mass-solar 10 --eddington-fraction 0.1 --output post_task2 --width 64 --height 64 --force 2>&1 | grep -E "Refinement done|n_z_cap|n_r_cap|nested_refine"
```

Expected: `Refinement done: n_r=X, n_z=Y` with `X << 4096` and `Y << 1024`. NO `n_z_cap`, `n_r_cap`, or `nested_refine_no_fixed_point` lines.

If caps still bind, log the actual `compare_columns` deltas at the cap-bind point and report — there may be a residual issue (e.g., the Eddington T-tau outer iteration's convergence test).

- [ ] **Step 7: Commit**

```bash
git add src/volumetric_disk.cpp
git commit -m "feat(volumetric): adaptive DPRK45 in solve_column rho integration"
```

---

## Task 3: Fix `test_tau_midplane_near_target` units mismatch

**Goal:** The tau-midplane test integrates `kappa·rho·dz` and compares to `tau_mid=100`. It currently uses `density()` (geometric-scaled units) where it should use `density_cgs()` (g/cm³).

**Files:**
- Modify: `tests/test_volumetric.cpp` (`test_tau_midplane_near_target` only)

- [ ] **Step 1: Confirm the test currently fails**

Run: `./build/Release/test-volumetric 2>&1 | grep -A1 "tau_mid at peak-flux"`

Expected: `τ(z=0..z_max) at r=X.XX: 0.00 (target 100.00)` followed by `FAIL`.

- [ ] **Step 2: Replace `density()` calls with `density_cgs()` and widen clamps**

In `tests/test_volumetric.cpp`, find `test_tau_midplane_near_target()`. Locate the three `disk.density(...)` calls:

1. The peak-radius scan (around line 458-463), currently:
```cpp
        const double rho = disk.density(r, 0.0, 0.0);
```

Replace with:
```cpp
        const double rho = disk.density_cgs(r, 0.0, 0.0);
```

2. The two integration-loop calls, currently:
```cpp
        const double rho_a = std::clamp(disk.density(r, z_a, 0.0), 1e-30, 1e-3);
        const double rho_b = std::clamp(disk.density(r, z_b, 0.0), 1e-30, 1e-3);
```

Replace with:
```cpp
        const double rho_a = std::clamp(disk.density_cgs(r, z_a, 0.0), 1e-18, 1e-6);
        const double rho_b = std::clamp(disk.density_cgs(r, z_b, 0.0), 1e-18, 1e-6);
```

The clamp range `[1e-18, 1e-6]` matches the bounds the opacity LUTs were built with.

The rest of the function body — kappa lookups, integration formula, comparison to `vp.tau_mid` — stays identical.

- [ ] **Step 3: Build**

Run: `cmake --build build --config Release`

Expected: only `test_volumetric.cpp` recompiles, clean build.

- [ ] **Step 4: Run the test**

Run: `./build/Release/test-volumetric 2>&1 | grep -A1 "tau_mid at peak-flux"`

Expected: PASS. The printed τ should be within 30% of `tau_mid=100`, i.e., somewhere in `[70, 130]`.

If the test still fails:
- τ in `[70, 130]`: PASS, proceed.
- τ in `[10, 70]` or `[130, 1000]`: wrong-magnitude bug in `normalize_density`'s peak-radius detection. Capture the printed `τ` and `r` values, halt, and report — out of this plan's scope.
- τ near 0 or NaN: something else broke. Halt and report.

- [ ] **Step 5: Run full test suite to confirm zero failures**

Run: `./build/Release/test-volumetric 2>&1 | tail -3`

Expected: `=== 0 failures ===`.

- [ ] **Step 6: Commit**

```bash
git add tests/test_volumetric.cpp
git commit -m "test(volumetric): use density_cgs in tau-midplane integration"
```

---

## Task 4: Final integration sanity & smoke render

**Goal:** Verify all four currently-relevant tests pass, no Promptable warnings fire on construction, refinement converges below the cap, and a 1024² render still produces output.

**Files:** None modified. Verification only.

- [ ] **Step 1: Run all three test executables**

Run:
```bash
cmake --build build --config Release
./build/Release/test-opacity
./build/Release/test-spectral
./build/Release/test-volumetric 2>&1 | tail -3
```

Expected: all three exit 0 with `=== 0 failures ===` (test-volumetric) and PASS lines (test-opacity, test-spectral).

- [ ] **Step 2: Re-enable the smoke parameter sweep**

In `tests/test_volumetric.cpp`'s `main()`, find the line (currently around line 522):
```cpp
    // test_smoke_parameter_sweep();  // ~3-4 min — uncomment for full sweep
```

Replace with the active call:
```cpp
    test_smoke_parameter_sweep();  // ~1-2 min after refinement converges below cap
```

The previous comment said 3-4 min; with refinement no longer cap-binding it should run in 1-2 minutes.

- [ ] **Step 3: Build and run the full sweep**

Run:
```bash
cmake --build build --config Release && ./build/Release/test-volumetric 2>&1 | tail -3
```

Expected: `=== 0 failures ===`. Total runtime well under 5 minutes.

If any of the 7 sweep cases now fails (was passing before): halt, capture the failing case's `mass`, `spin`, `T_peak`, and the error message. Report — likely DP45 needs tolerance tuning for that mass scale.

- [ ] **Step 4: Smoke render — canonical Kerr volumetric**

Run:
```bash
./build/Release/grrt-cli --metric kerr --spin 0.998 --observer-r 50 --observer-theta 80 \
    --disk-volumetric --mass-solar 10 --eddington-fraction 0.1 \
    --output final_smoke --width 1024 --height 1024 --force 2>&1 | grep -E "Refinement done|σ_s_phys|Construction complete|prompt_count|Promptable|Saved"
```

Expected:
- `Refinement done: n_r=X, n_z=Y` with X < 4096 and Y < 1024.
- No `Promptable` cap warnings.
- `Saved final_smoke.png` (and `.hdr` files).

The `--force` flag is included for safety; if no Promptable warnings fire, it's a no-op.

- [ ] **Step 5: Visual comparison**

Open `final_smoke.png` in an image viewer. Compare against the previously-rendered `visual_stellar.png` from before the BPT72 + DP45 changes. Expected differences:

- Inner disk near ISCO: slightly different brightness pattern (BPT72 taper has a different shape than the old Gaussian).
- Photosphere top: cleaner, no flat plate (unchanged from prior work, but verify no regression).
- Outer edge: smooth fade (unchanged from prior work, but verify).
- Bulk disk: density should be smoother in z due to the cliff position now being numerically stable across columns.

If the visual is dramatically different (entire color palette shifted, disk missing, etc.): something is wrong. Halt and report.

- [ ] **Step 6: Commit the test-suite re-enablement**

```bash
git add tests/test_volumetric.cpp
git commit -m "test(volumetric): re-enable parameter sweep now that refinement converges"
```

- [ ] **Step 7: Document the result in the spec**

Append a `## Implementation Results` section to `docs/superpowers/specs/2026-04-29-bpt72-taper-and-dprk45.md` recording:

- Final refinement sizes from Step 4: `n_r=X, n_z=Y`.
- Promptable warning count on healthy construction (should be 0).
- Total `test-volumetric` runtime with sweep enabled.
- Any visual differences observed in Step 5.

Then commit:
```bash
git add docs/superpowers/specs/2026-04-29-bpt72-taper-and-dprk45.md
git commit -m "docs(spec): record BPT72 + DPRK45 implementation results"
```

---

## Self-Review Notes (for the plan author)

**Spec coverage:**
- Spec §Components.1 (BPT72 taper) → Task 1 Steps 4, 5, 6, 7, 8 ✓
- Spec §Components.2 (DPRK45) → Task 2 Step 3 ✓
- Spec §Components.3 (tau test units fix) → Task 3 Step 2 ✓
- Spec §Components.4 (test_taper update) → Task 1 Step 2 ✓
- Spec §Validation criteria → Task 1 Step 10 (test_taper, h_jump retire), Task 2 Step 5 (density tests pass), Task 3 Step 4 (tau test passes), Task 4 Step 4 (no Promptables, refinement below cap) ✓
- Spec §Migration notes (CUDA compat for taper_width()) → Task 1 Step 7's legacy-stub instructions ✓

**Placeholder scan:** Every step has actual code or actual commands. No TBDs, TODOs, "implement later", or vague hand-waves. The DP45 Butcher tableau is fully written out in Step 3.

**Type consistency:** `taper_width_` is consistently a removed member; `taper_width()` is consistently the legacy accessor returning `(r_isco - r_horizon) / 3.0`. `dp45_tol`, `RHO_FLOOR`, `h_floor` are local variables in Task 2 only. `density_cgs` is the public accessor consistently used in Task 3.

**Estimated scope:** ~120-150 lines of changed/new code across 3 files, distributed:
- Task 1: ~25 lines (taper body, header member removal, header docstring, accessor stub, constructor cleanup, test threshold)
- Task 2: ~85 lines (replaces ~25 lines of fixed-step RK4 with ~85 lines of adaptive DP45)
- Task 3: ~3 lines
- Task 4: documentation only
