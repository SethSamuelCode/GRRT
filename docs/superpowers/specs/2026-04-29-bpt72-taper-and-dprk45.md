# BPT72 Plunging-Region Taper + Adaptive DPRK45 in solve_column

**Date:** 2026-04-29
**Branch:** `fix/volumetric-ring`
**Supersedes:** `2026-04-28-volumetric-disk-numerics-fix-design.md`

## Summary

Replace the heuristic Gaussian plunging-region `taper(r)` with a physically-derived shape from BPT72 mass conservation along the plunging geodesic, and replace the fixed-step RK4 in `solve_column` with adaptive Dormand-Prince RK4(5) so adjacent radial columns converge to truly self-similar profiles. Together these fix the three remaining `test-volumetric` failures and retire all five Promptable warnings emitted on every render.

## Background

The 2026-04-28 spec proposed widening the Gaussian taper as a way to fix the `test_density_profile` and `test_density_smooth_across_zmax` failures. Implementing it in Task 1 of that plan revealed two problems:

1. **The taper widening doesn't reach the failing tests.** The taper function returns `1.0` exactly for `r >= r_isco` (volumetric_disk.cpp:183). Both failing tests sample at `r=10M` and `r=6M` — well outside ISCO. The taper has zero effect there. The widening helps only the inner edge near horizon, where it retires the `h_jump` Promptable warning but doesn't touch the failing tests.
2. **The bulk-disk artifact has a different cause.** At `r=10M`, density drops 4.4×10⁴ in just one scale height because the radiation-pressure cliff in the hydrostatic ODE produces a near-discontinuous photosphere boundary. Adjacent columns at `r=10.0M` and `r=10.1M` resolve this cliff with fixed-step RK4 and end up with cliff positions that differ by ~0.5% of `z_max`. The per-column-z-normalized `interp_2d` blends two non-self-similar profiles across radius, producing non-monotonic density in z. This is a real numerical artifact in the LUT that no LUT-side workaround can fully fix.

Diagnostic visualization at `interp_2d_artifact_viz.html` shows the geometry of the artifact in detail.

The right fix is to attack both root causes:

- **The taper itself was always heuristic.** A Gaussian with arbitrary width factor had no physical justification. The BPT72 plunging-region geodesic gives an exact mass-conservation profile. With this change, `taper_width_` and `params_.plunging_taper_width_factor` go away — the taper has zero free parameters, set entirely by mass and spin.
- **The cliff position needs a stable numerical solution.** Adaptive DPRK45 takes small steps where the ODE is stiff (right at the cliff) and larger steps where it's smooth (deep midplane, deep tail). Adjacent columns then converge to the same cliff position to machine precision, eliminating the cross-column blend artifact in `interp_2d`.

The codebase already has `VolumetricDisk::plunging_velocity(r, theta, ut, ur, uphi)` (volumetric_disk.cpp:155) which computes the BPT72 4-velocity from precomputed `E_isco_`, `L_isco_`. The new taper reuses it directly. The codebase also has Dormand-Prince RK4(5) in `include/grrt/geodesic/rk4.h` for geodesic integration; the implementation pattern is already established.

## Goals

- All three currently-failing tests pass: `test_density_profile`, `test_density_smooth_across_zmax`, `test_tau_midplane_near_target`.
- `test_taper` updated to reflect the new physically-derived shape (taper saturates to 1 outside ISCO, decays toward 0 at horizon — same qualitative shape, but the precise values come from physics not a Gaussian).
- All Promptable warnings retire on healthy construction: `h_jump` (from smooth taper), `n_z_cap`, `n_r_cap`, `nested_refine_no_fixed_point` (all from refinement converging at modest n_z/n_r once cliff position is stable).
- The `taper_width_` private member, the `taper_width_factor` parameter idea (never landed), and the Gaussian shape are all removed.

## Non-goals

- Changing `interp_2d`'s per-column z-normalization scheme. With DPRK45, adjacent columns produce sufficiently self-similar profiles that the existing scheme works.
- Modifying `compute_plunging_region_decay`'s vertical structure inside ISCO. That keeps using `H_isco · taper(r)^p`, just with the new physical taper.
- Investigating `normalize_density`'s peak-radius detection. If `test_tau_midplane_near_target` still fails after the units fix and DPRK45, that's a separate bug for a follow-up.
- Replacing the per-iteration Eddington T-tau loop in `solve_column`. DPRK45 only replaces the inner RK4 step; the outer iteration over T(τ) and ρ(z) stays.

## Approach

Three changes:

1. **BPT72 mass-conservation taper.** Replace the body of `taper(r)` with `r_isco · |u^r(r_isco·0.99)| / (r · |u^r(r)|)`, where `u^r` comes from `plunging_velocity`. Drop the `taper_width_` member entirely. The shape saturates to 1 at ISCO by construction (because the regulator radius is `r_isco·0.99` ≈ ISCO) and goes to 0 at the horizon (where `|u^r|` → ∞ in the local frame, so the ratio → 0).
2. **Adaptive DPRK45 in `solve_column`.** Replace the fixed-step RK4 inside the per-column outer iteration with Dormand-Prince RK4(5) using a tolerance that tracks `params_.target_lut_eps`. Step size adapts: small near the photosphere cliff, large in the smooth midplane/tail. The number of stored z-bins (`n_z`) is decoupled from the integrator step count — z-bins are sampled by interpolation onto a uniform z grid spanning `[0, z_max]`.
3. **Tau test units fix** (carryover from 2026-04-28 spec). Replace `density()` with `density_cgs()` in `test_tau_midplane_near_target`, widen clamps from `[1e-30, 1e-3]` to `[1e-18, 1e-6]`.

## Components

### 1. BPT72 mass-conservation taper

In `src/volumetric_disk.cpp`, replace the current `taper(r)` body:

```cpp
double VolumetricDisk::taper(double r) const {
    if (r >= r_isco_) return 1.0;
    if (r <= r_horizon_) return 0.0;

    // Mass conservation along plunging geodesic:
    // ρ ∝ 1/(r · |u^r|). Normalize so taper(r_isco·0.99) ≈ 1.
    constexpr double EPS = 0.99;        // regulate the v^r=0 singularity at ISCO
    constexpr double THETA = M_PI_2;    // equatorial plane

    double ut, ur_ref, uphi;
    plunging_velocity(r_isco_ * EPS, THETA, ut, ur_ref, uphi);
    const double r_ref = r_isco_ * EPS;
    const double denom_ref = r_ref * std::abs(ur_ref);
    if (denom_ref <= 0.0) return 1.0;   // pathological — no plunging motion

    double ur;
    plunging_velocity(r, THETA, ut, ur, uphi);
    const double denom = r * std::abs(ur);
    if (denom <= 0.0) return 1.0;       // numerical floor near ISCO

    return std::clamp(denom_ref / denom, 0.0, 1.0);
}
```

Remove the `taper_width_` private member from `volumetric_disk.h`. Remove the line `taper_width_ = (r_isco_ - r_horizon_) / 3.0;` from the constructor.

The `taper_width()` public accessor (line 145) is referenced by `cuda/cuda_vol_host_data.cpp:27`. The CUDA path is out of scope (see 2026-04-27 spec migration notes — `--validate` disabled for volumetric until a follow-up CUDA spec) but the source must still compile when CUDA is enabled. Replace the accessor body with a compute-on-demand legacy stub:

```cpp
/// Legacy accessor — preserved for CUDA host-data layout compatibility.
/// Returns the heuristic Gaussian taper width that was used before
/// the BPT72 mass-conservation taper replaced it. New code should
/// not depend on this value.
double taper_width() const { return (r_isco_ - r_horizon_) / 3.0; }
```

Update the constructor's `bins_per_gradient` heuristic (volumetric_disk.cpp:73) to use `(r_isco_ - r_horizon_) / 3.0` directly, since that line was just a sizing heuristic that happened to use `taper_width_`.

Update `test_taper` to match the new behavior:
- `taper(r_isco) ≈ 1.0` ✓ unchanged
- `taper(r_isco + 1.0) ≈ 1.0` ✓ unchanged
- `taper(r_horizon) ≈ 0.0` (within e.g. 0.05) — was `< 0.1`, the new threshold can be tighter because the BPT72 shape genuinely goes to 0.

### 2. Adaptive DPRK45 in `solve_column`

The current `solve_column` (volumetric_disk.cpp:514+) uses fixed-step RK4 over `n_z` evenly-spaced z bins. The cliff at the photosphere boundary forces RK4 into oscillation when the step size doesn't match the local stiffness.

Replace the inner RK4 step with adaptive Dormand-Prince RK4(5):

- Step size starts at `dz = z_max / (n_z - 1)` and adapts based on the embedded error estimate.
- The integrator advances z continuously from 0 to `z_max`, producing an arbitrary number of internal steps.
- After each successful step, the integrator state is sampled. At the end, results are interpolated onto the uniform z grid `[0, z_max]` with `n_z` bins for storage in `rho_z[]` and `T_z[]`.
- Tolerance: `local_tol = params_.target_lut_eps · max(rho, RHO_FLOOR)`. Acceptance: `|err| < local_tol`. Step adjustment: standard P-controller `h_new = h · 0.9 · (tol/|err|)^(1/5)`, clamped to `[h/5, 5h]`.
- The Dormand-Prince Butcher tableau and step adjustment can be lifted from `include/grrt/geodesic/rk4.h` (the `Adaptive Dormand-Prince 4(5)` block) into a free-function utility `adaptive_dp45_step` that both call sites use. Generic over a callable for the RHS.

Why this works:
- At the photosphere cliff (rho dropping by orders of magnitude over a small dz), the embedded error estimate jumps; step size shrinks; the cliff is resolved with sub-dz resolution.
- In the smooth midplane and deep tail, error stays small; step size grows; we don't waste effort.
- Adjacent columns with similar (T_eff, H, mass, spin) end up resolving the same cliff position to within machine precision because the adaptive integrator converges on the same numerical solution regardless of the initial step.

The outer iteration over `T(τ)` and the photosphere convergence test (`rho_z[n_z-1] > CONV_FLOOR` → extend `z_max`) stay unchanged.

### 3. Tau test units fix

Carryover from 2026-04-28 spec. In `test_tau_midplane_near_target`:
- Replace three calls to `disk.density(r, z, 0.0)` with `disk.density_cgs(r, z, 0.0)` (peak-radius scan, two integration-loop calls).
- Change clamps from `[1e-30, 1e-3]` to `[1e-18, 1e-6]` to match opacity LUT bounds.

### 4. test_taper update

In `tests/test_volumetric.cpp::test_taper`, change the horizon assertion from:
```cpp
if (t_hor > 0.1) { ... FAIL ... }
```
to:
```cpp
if (t_hor > 0.05) { ... FAIL: BPT72 taper at horizon should be near zero ... }
```

The two `check(...)` lines at ISCO and ISCO+1 stay unchanged (they still pass with the new physical taper).

## Files modified

| File | Change |
|------|--------|
| `include/grrt/scene/volumetric_disk.h` | Remove `taper_width_` member; convert `taper_width()` accessor to compute-on-demand legacy stub for CUDA compat |
| `src/volumetric_disk.cpp` | Replace `taper(r)` body, remove `taper_width_` init line, update `bins_per_gradient` sizing line, replace inner RK4 in `solve_column` with adaptive DP45, factor out shared `adaptive_dp45_step` utility |
| `include/grrt/geodesic/rk4.h` (or new `include/grrt/math/adaptive_rk.h`) | Extract reusable Dormand-Prince utility |
| `tests/test_volumetric.cpp` | Update `test_taper` horizon threshold; fix `test_tau_midplane_near_target` units |

Estimated scope: ~120-150 lines of new/changed code (most in the DPRK45 utility extraction and `solve_column` rewrite). One commit per component is fine; together it's a single coherent change so a single combined commit is also defensible.

## Validation

After landing, the following invariants should hold on a healthy stellar-mass disk (`mass=1, spin=0.998, r_outer=30, T_peak=1e7`):

- `taper(r_horizon) < 0.05` — passes the updated `test_taper`.
- `taper(r_isco) ≈ 1.0` — unchanged.
- At `r=10M, z=H, 3H`: `density(r, 1H, 0) > density(r, 3H, 0)` — passes `test_density_profile`. The cliff position is stable across columns.
- At `r=6M, z=0.99·z_max`: `density / density_mid < 1e-10` — passes `test_density_smooth_across_zmax`. The deep tail of each column reaches the convergence floor cleanly.
- `compute τ at peak-flux radius using density_cgs` ≈ `tau_mid` (within 30%) — passes `test_tau_midplane_near_target`.
- No Promptable warnings on construction. `validate_luts`'s `h_jump` retires (smooth taper). `n_z_cap`, `n_r_cap`, `nested_refine_no_fixed_point` retire (refinement converges at modest sizes once cliff is stable).
- Construction time on a stellar-mass disk drops materially (refinement no longer cap-binds, so fewer doublings). Expected n_r ~ 512, n_z ~ 128 for typical disks.

## Error handling

- `taper(r)` returns 1.0 if `denom <= 0.0` at ISCO (numerical pathology near the regulator point) — the disk model degrades gracefully to "no plunging-region taper, full disk values" rather than producing NaN.
- `taper(r)` returns 0.0 for `r <= r_horizon`, matching the existing convention in `density()` which also bails at horizon.
- `adaptive_dp45_step` returns the previous state with a flagged error if the step size shrinks below a hard floor (e.g., `1e-12 · z_max`). The caller (`solve_column`) treats this as "cliff is too steep to resolve at this tolerance" and accepts the current state, advancing in next iteration. This matches the existing behavior of fixed-step RK4 hitting numerical underflow.

## Testing

No new test files. The four existing tests are the validation gate:

| Test | Currently | After |
|------|-----------|-------|
| `test_density_profile` | FAIL | PASS |
| `test_density_smooth_across_zmax` | FAIL | PASS |
| `test_tau_midplane_near_target` | FAIL | PASS |
| `test_taper` | PASS | PASS (with updated threshold) |

`test_smoke_parameter_sweep` (currently commented out for runtime) should be re-enabled and verified after the change. The 7-case sweep should run faster (smaller refined n_r/n_z) and still pass.

## Future work

- **Vertical hydrostatic structure inside ISCO.** Right now `compute_plunging_region_decay` uses `H_lut_[i] = H_isco · taper(r)^p` — a heuristic that mostly recovers the right qualitative shape. The genuinely physical model would compute the vertical scale height from the plunging-region geodesics with conserved p_θ at ISCO. Out of scope; the current heuristic is acceptable now that the radial taper is physical.
- **Krolik 1999 / Agol-Krolik 2000 magnetic stresses inside ISCO.** Would let material radiate inside ISCO rather than just falling in. Significant model extension; not blocking any current test.
- **Mummery 2024 intra-ISCO emission profile.** Recent analytical work that goes beyond pure freefall. Could replace `compute_plunging_region_decay`'s heuristic if more inner-region accuracy is needed.

## References

- **Bardeen, Press & Teukolsky 1972** ApJ 178, 347 — the BPT72 paper for plunging-region geodesics. Source of the conserved (E_isco, L_isco) constants and v^r(r) formula used in `plunging_velocity`.
- **Novikov & Thorne 1973** in *Black Holes* (Les Houches) — classical thin-disk model with hard truncation at ISCO. The starting point our taper modifies.
- **Page & Thorne 1974** ApJ 191, 499 — Page-Thorne radial flux profile.
- **Reynolds & Begelman 1997** ApJ 488, 109 — first argued for measurable column density inside ISCO.
- **Krolik 1999** ApJ 515, L73 — magnetic stresses inside ISCO; density set by radial velocity profile.
- **Agol & Krolik 2000** ApJ 528, 161 — disk emission with inside-ISCO contribution.
- **Hawley & Krolik 2001** ApJ 548, 348; **Beckwith, Hawley & Krolik 2008** ApJ 678, 1180; **Penna et al. 2010** MNRAS 408, 752 — GRMHD simulations confirming finite plunging-region density.
- **Mummery & Balbus 2023, Mummery 2024** — recent analytic intra-ISCO emission models.
- **Hairer, Nørsett & Wanner 1993** *Solving Ordinary Differential Equations I* — the standard reference for Dormand-Prince RK4(5) with embedded error estimation.

## Migration notes

The new BPT72 taper produces a different shape than the old Gaussian (factor=1/3). Existing renders will look slightly different in the inner-disk region near ISCO and into the plunging zone:

- The taper saturates to 1 at ISCO instead of `exp(-((r_isco-r_horizon)/(r_isco-r_horizon)/3)^2) ≈ 1.0` (effectively the same).
- Inside ISCO, the new taper decays as `1/v^r(r)·1/r` (power-law-ish set by Kerr metric), versus the old `exp(-((r_isco-r)/w)^2)` (Gaussian).
- At the horizon, the new taper exactly → 0 (because v^r → ∞), versus the old `exp(-9) ≈ 1.2e-4` (basically zero, but not exactly).

The visual difference is mostly in the inner-edge brightness pattern. The high-spin (a=0.998) ISCO is at ~1.24M, very close to the horizon at 1.06M, so the plunging region is narrow and the visual effect is subtle. Reference images of pre-change renders may need updating.

The C-API ABI is unchanged — no `GRRTParams` field added or removed by this spec. The intermediate `disk_plunging_taper_width_factor` field that was being added in the 2026-04-28 plan was never committed, so no compatibility concern there.
