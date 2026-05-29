# Uniform Fine Raymarch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the volumetric-disk banding/fireflies by replacing the placement-dependent adaptive step policy inside the disk with a deterministic two-regime stepper — coarse adaptive outside the envelope, snap to `z_max`, then fixed uniform fine steps through the dense region — while preserving the transversal-transit emission fix and resolving turbulence faithfully.

**Architecture:** `raymarch_volumetric` chooses its step from the photon's position relative to the disk vertical envelope `z_max(r)`: outside (ρ=0) it steps coarse/adaptive; a boundary-crossing step is bisected to land exactly on `z_max`; inside it steps at a fixed `ds_fine = min(H, L)/k` (L = turbulence correlation length, k=4) with midpoint-source radiative transfer. The z-resolution gate and the H/4 growth cap are removed; `mid_state` and the direction-aware outer exit are kept.

**Tech Stack:** C++23, MSVC 2022 / CMake, OpenMP (existing). Tests are standalone `.exe`s with inline `check`/`EXPECT_*` helpers.

**Spec:** `docs/superpowers/specs/2026-05-29-uniform-fine-raymarch-design.md`

**Commit policy:** The user composes and runs all commits. Every "Commit" step below is informational — provide the message text and stop; do NOT run `git commit`.

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `include/grrt/scene/volumetric_disk.h` | Modify | Declare `noise_correlation_length(double r)` |
| `src/volumetric_disk.cpp` | Modify | Define it (mirror the formula already in `density()`) |
| `tests/test_volumetric.cpp` | Modify | Unit-test the accessor; later record the new banding calibration |
| `src/geodesic_tracer.cpp` | Modify (`raymarch_volumetric`, L247-405) | Two-regime stepper + boundary snap; remove z-gate + H/4 cap |
| `include/grrt/geodesic/raymarch_step_control.h` | Modify | Remove `step_needs_z_refinement` (dead); keep `raymarch_exits_outer` |
| `tests/test_raymarch_step_control.cpp` | Modify | Remove the `step_needs_z_refinement` cases; keep `raymarch_exits_outer` |

---

## Task 1: `VolumetricDisk::noise_correlation_length(r)` accessor + unit test

**Files:**
- Modify: `include/grrt/scene/volumetric_disk.h`
- Modify: `src/volumetric_disk.cpp`
- Modify: `tests/test_volumetric.cpp`

- [ ] **Step 1: Write the failing test in `tests/test_volumetric.cpp`**

Add this function (the file already defines `check(const char* name, double got, double expected, double rel_tol)` and `shared_disk_default()`):

```cpp
void test_noise_correlation_length() {
    std::printf("\n=== noise_correlation_length ===\n");
    const auto& disk = shared_disk_default();   // turbulence=1.0, noise_scale=0, c_corr=0.5
    const double r = 10.0;
    const double H = disk.scale_height(r);
    const double L = disk.noise_correlation_length(r);
    // Defaults: noise_scale=0 -> L = c_corr * H = 0.5 * H.
    check("L == 0.5*H", L, 0.5 * H, 1e-9);
}
```

Register it: add `test_noise_correlation_length();` in `main()` next to the other `test_*();` calls (e.g. right after `test_construction();`).

- [ ] **Step 2: Build — must fail to compile (method doesn't exist)**

Run:
```
cmake --build build --config Release
```
Expected: COMPILE ERROR — `'noise_correlation_length' is not a member of 'grrt::VolumetricDisk'`.

- [ ] **Step 3: Declare the accessor in `include/grrt/scene/volumetric_disk.h`**

In the `// --- Accessors for raymarching ---` section, immediately after the `scale_height` declaration (line 90), add:

```cpp
    /// Turbulence correlation length L(r) [geometric units] — the spatial scale
    /// of the fractal density noise (= c_corr·H(r), or noise_scale·H(r) when
    /// noise_scale > 0). The raymarch sizes its fine step against this so the
    /// turbulence is resolved. Independent of the `turbulence` amplitude.
    double noise_correlation_length(double r) const;
```

- [ ] **Step 4: Define it in `src/volumetric_disk.cpp`**

Add the definition next to `z_max_at` / `scale_height` (mirrors the `L` computation inside `density()` at lines 295-299):

```cpp
double VolumetricDisk::noise_correlation_length(double r) const {
    const double H = scale_height(r);
    const double c_corr = (params_.noise_correlation_length_factor > 0.0)
                        ? params_.noise_correlation_length_factor : 0.5;
    return (params_.noise_scale > 0.0) ? params_.noise_scale * H : c_corr * H;
}
```

- [ ] **Step 5: Build and run — test must pass**

Run:
```
cmake --build build --config Release
.\build\Release\test-volumetric.exe 2>&1 | findstr /C:"noise_correlation_length" /C:"L == 0.5*H"
```
Expected: `L == 0.5*H: got=... expected=... PASS`. (The full `test-volumetric` run also exercises the slow banding render — that's fine; you only need the accessor line to PASS here. The pre-existing `test_tau_midplane_near_target` may report FAIL; that is a known pre-existing issue unrelated to this change.)

- [ ] **Step 6: Commit (DO NOT run — hand the message to the user)**

```
feat(disk): expose noise_correlation_length(r) for fine raymarch stepping

Surfaces the turbulence correlation length L(r) (= c_corr·H, or
noise_scale·H) already computed inside density(), so the volumetric
raymarch can size its fine step ds_fine = min(H,L)/k against it.
Independent of the turbulence amplitude.
```

---

## Task 2: Two-regime stepper in `raymarch_volumetric`

**Files:**
- Modify: `src/geodesic_tracer.cpp` (`raymarch_volumetric`, currently lines 247-405)

`raymarch_volumetric` is a private method needing a disk + sampler, so it is **not** unit-testable in isolation. The gate for this task is: **build clean + all existing unit suites stay green.** Full behavioral validation is Task 4.

- [ ] **Step 1: Replace the entire `raymarch_volumetric` function body**

Replace the whole function (lines 247-405, from `void GeodesicTracer::raymarch_volumetric(...)` through its closing `}`) with:

```cpp
void GeodesicTracer::raymarch_volumetric(GeodesicState& state, Vec3& /*color*/,
                                          double J_rgb[3], double T_rgb[3]) const {
    using namespace constants;
    const auto& luts = vol_disk_->opacity_luts();

    constexpr std::array<double, 3> nu_obs_arr = {
        c_cgs / 450e-7, c_cgs / 550e-7, c_cgs / 650e-7
    };
    std::span<const double> ch_span{nu_obs_arr.data(), 3};

    double J[3] = {J_rgb[0], J_rgb[1], J_rgb[2]};
    double T[3] = {T_rgb[0], T_rgb[1], T_rgb[2]};

    VolumetricDiskSampler sampler(vol_disk_, observer_r_);
    const double ut_obs = sampler.ut_obs;

    // Fine-sampling quality knob: inside the disk envelope we step UNIFORMLY at
    // ds_fine = min(H, L) / FINE_SAMPLES_PER_CORR, where L is the turbulence
    // correlation length — ~4 samples per correlation length, ~8 across the base
    // peak. Uniform spacing (no placement-dependent step decisions) is what
    // removes the banding; tying ds to physics resolves turbulence faithfully.
    constexpr int FINE_SAMPLES_PER_CORR = 4;

    // True when the photon state is inside the disk's vertical envelope (the
    // emitting region: density is 0 for |z| >= z_max(r)).
    auto inside_envelope = [&](const GeodesicState& s) {
        const double rr = s.position[1];
        const double zz = rr * std::cos(s.position[2]);
        return std::abs(zz) <= vol_disk_->z_max_at(rr);
    };

    double r = state.position[1];
    const double z_start = r * std::cos(state.position[2]);
    const double H_start = vol_disk_->scale_height(r);
    // Coarse step proposal — used only OUTSIDE the envelope (rho = 0 there).
    double ds_coarse = std::min(std::abs(z_start) / 8.0, H_start * 2.0);
    if (ds_coarse <= 0.0) ds_coarse = H_start;

    int step_count = 0;
    constexpr int MAX_STEPS = 16384;   // headroom for fine transit stepping

    while (step_count < MAX_STEPS) {
        // Hard exits.
        if (r < vol_disk_->r_horizon())  break;
        // Direction-aware outer-radius exit (side-impact fix): only bail if the
        // photon is genuinely leaving (moving outward). Inward rays beyond r_max
        // are entering from outside the rim — keep marching.
        if (r > vol_disk_->r_max()) {
            const double dr_dl = RK4::derivatives_kerr(metric_, state).position[1];
            if (raymarch_exits_outer(r, vol_disk_->r_max(), dr_dl)) break;
        }
        if (T[0] < 1e-6 && T[1] < 1e-6 && T[2] < 1e-6)  break;

        const bool inside = inside_envelope(state);

        // Step size: fixed uniform fine inside the envelope (tied to local H, L);
        // coarse adaptive outside.
        double ds;
        if (inside) {
            const double H_loc = vol_disk_->scale_height(r);
            const double L_loc = vol_disk_->noise_correlation_length(r);
            ds = std::min(H_loc, L_loc) / static_cast<double>(FINE_SAMPLES_PER_CORR);
        } else {
            ds = ds_coarse;
        }

        RombergStep rs = romberg_step(state, ds, ch_span, sampler, metric_, integrator_);

        // Boundary snap: if this step crosses the envelope (inside<->outside),
        // bisect to land exactly on |z| = z_max, then re-take it. This makes fine
        // sampling begin at the envelope boundary — no coarse overshoot into the
        // disk, so no placement-dependent skipped emission, so no banding.
        if (inside_envelope(rs.end_state) != inside) {
            double lo = 0.0, hi = ds;
            for (int it = 0; it < 16; ++it) {
                const double m = 0.5 * (lo + hi);
                if (inside_envelope(integrator_.step_kerr(metric_, state, m)) == inside)
                    lo = m;
                else
                    hi = m;
            }
            ds = hi;
            rs = romberg_step(state, ds, ch_span, sampler, metric_, integrator_);
        }

        // Per-channel radiative transfer, source sampled at the step MIDPOINT.
        // Outside the envelope rho = 0, so this block contributes nothing.
        const GeodesicState& mid = rs.mid_state;
        const double r_mid       = mid.position[1];
        const double theta_mid   = mid.position[2];
        const double phi_mid     = mid.position[3];
        const double z_mid       = r_mid * std::cos(theta_mid);

        const double rho_cgs = vol_disk_->density_cgs(r_mid, z_mid, phi_mid);
        const double T_local = vol_disk_->temperature(r_mid, std::abs(z_mid));
        if (rho_cgs > 0.0 && T_local > 0.0) {
            double ut_emit = 0.0, ur_emit = 0.0, uphi_emit = 0.0;
            if (r_mid >= vol_disk_->r_isco()) {
                vol_disk_->circular_velocity(r_mid, ut_emit, uphi_emit);
            } else {
                vol_disk_->plunging_velocity(r_mid, theta_mid, ut_emit, ur_emit, uphi_emit);
            }
            const double p_dot_u_emit = mid.momentum[0] * ut_emit
                                       + mid.momentum[1] * ur_emit
                                       + mid.momentum[3] * uphi_emit;
            const double p_dot_u_obs  = mid.momentum[0] * ut_obs;
            const double g_factor     = p_dot_u_emit / p_dot_u_obs;

            for (int ch = 0; ch < 3; ++ch) {
                const double nu_emit = std::abs(g_factor) * nu_obs_arr[ch];
                const double kabs    = luts.lookup_kappa_abs(nu_emit, rho_cgs, T_local);
                const double kes     = luts.lookup_kappa_es(rho_cgs, T_local);
                const double ktot    = kabs + kes;
                const double epsilon = (ktot > 0.0) ? kabs / ktot : 1.0;

                const double Bnu     = planck_nu(nu_emit, T_local);
                const double S       = epsilon * Bnu / (nu_emit * nu_emit * nu_emit);

                const double dtau    = rs.dtau[ch];
                const double exp_dtau = std::exp(-dtau);
                J[ch] += T[ch] * S * (1.0 - exp_dtau);
                T[ch] *= exp_dtau;
            }
        }

        state = rs.end_state;
        r = state.position[1];

        // Coarse-step adaptive sizing — OUTSIDE the envelope only (fine steps are
        // fixed). Grow when well under tolerance (capped at H), shrink on error.
        if (!inside) {
            const double H_now = vol_disk_->scale_height(r);
            if (rs.max_err < raymarch_tol_ / 8.0)
                ds_coarse = std::min(ds_coarse * 2.0, H_now);
            else if (rs.max_err > raymarch_tol_)
                ds_coarse = std::max(ds_coarse * 0.5, H_now / 256.0);
        }
        step_count++;
    }

    // Persist for caller.
    J_rgb[0] = J[0]; J_rgb[1] = J[1]; J_rgb[2] = J[2];
    T_rgb[0] = T[0]; T_rgb[1] = T[1]; T_rgb[2] = T[2];
}
```

Notes for the implementer:
- This removes the `step_needs_z_refinement` call and the envelope-aware H/4 growth cap. `step_needs_z_refinement` is still *defined* in the header after this task (just no longer called) — it is deleted in Task 3.
- `raymarch_exits_outer` and `RombergStep::mid_state` are retained exactly.
- The `#include "grrt/geodesic/raymarch_step_control.h"` at the top of the file stays (still used by `raymarch_exits_outer`).
- `integrator_.step_kerr(metric_, state, m)` is the same call `romberg_step` uses internally; `RK4::derivatives_kerr` is already used in this file.

- [ ] **Step 2: Build clean**

Run:
```
cmake --build build --config Release
```
Expected: builds with no new warnings/errors (the pre-existing `/Ob2`→`/Ob3` cl warning is unrelated).

- [ ] **Step 3: Run the unit suites — all green**

Run:
```
.\build\Release\test-romberg-step.exe
.\build\Release\test-raymarch-step-control.exe
.\build\Release\test-disk-step-entry.exe
.\build\Release\test-spectral.exe
```
Expected: each prints `=== 0 failures ===` (or `ALL PASSED`). (`test-volumetric` is deferred to Task 4 — it includes the banding render that this change is meant to move.)

- [ ] **Step 4: Commit (DO NOT run — hand the message to the user)**

```
fix(raymarch): uniform fine stepping through the disk envelope

Replace the placement-dependent adaptive step policy inside the disk with
a two-regime stepper: coarse adaptive outside the envelope, a boundary
step bisected to land exactly on z_max, then fixed uniform fine steps
ds = min(H, L)/4 through the dense region (midpoint-source RT unchanged).

The banding was a systematic aliasing artifact: the old z-gate + H/4 cap
made step placement depend on z, aliasing the sharp emission peak and the
turbulence against the step grid (spp-immune). Uniform fine spacing tied
to the physical scales (base peak ~H, turbulence corr ~0.5H) removes it
while resolving turbulence faithfully and preserving the transit emission.

Removes the z-resolution gate call and the H/4 growth cap; keeps mid_state
and the direction-aware outer exit (raymarch_exits_outer). See spec
2026-05-29-uniform-fine-raymarch-design.md.
```

---

## Task 3: Remove the dead `step_needs_z_refinement` gate + its tests

**Files:**
- Modify: `include/grrt/geodesic/raymarch_step_control.h`
- Modify: `tests/test_raymarch_step_control.cpp`

After Task 2 nothing calls `step_needs_z_refinement`; remove it and its unit cases (YAGNI / no dead code). `raymarch_exits_outer` stays.

- [ ] **Step 1: Remove `step_needs_z_refinement` from the header**

In `include/grrt/geodesic/raymarch_step_control.h`, delete the doc comment + function for `step_needs_z_refinement` (currently lines 9-28, the block from `/// Returns true if a raymarch step ...` through the closing `}` of `step_needs_z_refinement`). Leave the `#include`s, the `namespace grrt {`, the `raymarch_exits_outer` function (lines 30-45), and the closing `} // namespace grrt` / `#endif` intact.

- [ ] **Step 2: Remove the `step_needs_z_refinement` cases from the test**

In `tests/test_raymarch_step_control.cpp`:
1. Update the file header comment (lines 1-4) to describe `raymarch_exits_outer` instead of `step_needs_z_refinement`:
```cpp
// tests/test_raymarch_step_control.cpp
//
// Unit tests for raymarch_exits_outer — the direction-aware outer-radius
// exit predicate. Pure scalar logic; no disk, metric, or integrator needed.
```
2. Delete the seven `step_needs_z_refinement` blocks and the now-unused `qH`/`env` constants — i.e. remove everything from `constexpr double qH  = 0.005;` (line 25) through the `fine_far_from_disk` check (the `check("fine_far_from_disk", ...)` call ending at line 56). Keep `main()`'s opening `std::printf("Running test_raymarch_step_control...\n");`, then the `raymarch_exits_outer` section, then the trailing `=== failures ===` print.

The resulting `main()` body is exactly:
```cpp
int main() {
    std::printf("Running test_raymarch_step_control...\n");

    // --- raymarch_exits_outer: direction-aware outer-radius exit ---
    constexpr double rmax = 20.0;

    // Inside the disk radius: never exits, regardless of radial direction.
    check("inside_moving_in",  raymarch_exits_outer(10.0, rmax, -1.0), false);
    check("inside_moving_out", raymarch_exits_outer(10.0, rmax, +1.0), false);

    // Outside the rim, moving OUTWARD (genuinely leaving): exit.
    check("outside_moving_out", raymarch_exits_outer(21.0, rmax, +0.5), true);

    // Outside the rim, moving INWARD (entering from outside): do NOT exit —
    // this is the side-impact case the fix exists for. Must keep marching.
    check("outside_moving_in",  raymarch_exits_outer(21.0, rmax, -0.5), false);

    // Exactly at the rim: still in range, do not exit (strict r > r_max).
    check("at_boundary",        raymarch_exits_outer(20.0, rmax, +1.0), false);

    // Outside, radially stationary (dr=0): not inbound, so exiting is correct —
    // a photon at a radial turning point above the rim is not entering the disk.
    check("outside_stationary", raymarch_exits_outer(20.5, rmax, 0.0), true);

    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
```

- [ ] **Step 3: Build and run — test must pass**

Run:
```
cmake --build build --config Release
.\build\Release\test-raymarch-step-control.exe
```
Expected: `=== 0 failures ===`, exit 0 (the six `raymarch_exits_outer` cases). The build also confirms nothing else referenced `step_needs_z_refinement`.

- [ ] **Step 4: Commit (DO NOT run — hand the message to the user)**

```
refactor(raymarch): remove dead step_needs_z_refinement gate

The z-resolution gate is superseded by the uniform fine stepper and is no
longer called. Remove the predicate and its unit cases. raymarch_exits_outer
(the side-impact exit) is retained and still tested.
```

---

## Task 4: Integration verification

**Files:** none modified, except `tests/test_volumetric.cpp` only if the banding calibration comment/threshold needs updating (Step 2 — a judgment step).

- [ ] **Step 1: Banding regression**

Run:
```
cmake --build build --config Release
.\build\Release\test-volumetric.exe 2>&1 | findstr /C:"banding metric" /C:"Banding regression" /C:"rows with disk"
```
Expected: the `banding metric (avg|drow|/<row>)` is now **below 0.25** (target ≤ ~0.21; was 0.369). Record the value.

- [ ] **Step 2: Update the banding calibration comment (and threshold only if needed — JUDGMENT)**

In `tests/test_volumetric.cpp`, add a calibration-history line to the comment block (after line 725, `//   - Buggy build (H_max=H): rel = 0.281 (real banding)`):
```cpp
    //   - Uniform fine stepper (this work):       rel = 0.XXX (placement-free)
```
(fill `0.XXX` with the Step-1 value).

If Step 1 is **below 0.25**: leave `THRESHOLD = 0.25` unchanged — the test now passes; no further action.

If Step 1 is unexpectedly **above 0.25** despite a visually clean render (Step 3): this is a JUDGMENT call — surface the before/after metric and the render to the user with a recommendation; do **not** loosen the threshold silently.

- [ ] **Step 3: Debug-pixel + visual render**

Run the previously-failing transversal-transit pixel and a turbulent render:
```
.\build\Release\grrt-cli.exe --disk-volumetric --samples 1 --width 256 --height 256 --fov 30 --debug-pixel 150 180 --output dbg.png --force 2>&1 | findstr "ENTRY RAYMARCH ESCAPED HORIZON"
.\build\Release\grrt-cli.exe --disk-volumetric --samples 100 --width 256 --height 256 --fov 30 --output uniform_fine.png --force
```
Expected: the (150,180) RAYMARCH still returns non-zero emission (bottom disk preserved). The render `uniform_fine.png.png` is left on disk for inspection — the executor can read the PNG: confirm the disk shows turbulence clumps/holes with **no horizontal banding stripes** and no scattered fireflies.

- [ ] **Step 4: MAX_STEPS sanity**

Confirm rays are not routinely hitting the 16384 cap (which would silently truncate the optical-depth column). The simplest check: the render in Step 3 completes and looks correct. If a deeper check is wanted, temporarily add an exit-reason counter to `raymarch_volumetric` (file-static atomic + atexit print, as in the prior side-impact investigation), run a small render, confirm `maxsteps` ≈ 0, then revert the instrumentation before any commit.

- [ ] **Step 5: Report to the user**

Summarize:
- Banding metric before (0.369) / after (recorded), pass/fail vs 0.25, whether the comment was updated.
- Debug-pixel (150,180) RAYMARCH emission (non-zero?).
- Visual: bottom disk renders, turbulence clumps/holes present, no stripes/fireflies.
- All unit suites green (modulo the pre-existing `test_tau_midplane_near_target`).
- Render wall-clock time (expected slower than before — quality-first).
- Request visual inspection of `uniform_fine.png.png`.

- [ ] **Step 6: Commit any calibration-comment change (DO NOT run — hand the message to the user)**

Only if Step 2 changed `tests/test_volumetric.cpp`:
```
test(volumetric): record uniform-fine-stepper banding calibration

Banding metric drops 0.369 -> 0.XXX with the uniform fine stepper;
add the calibration-history line. Threshold 0.25 unchanged.
```

---

## Plan self-review

Spec coverage scan against `2026-05-29-uniform-fine-raymarch-design.md`:

- §4.1 coarse adaptive outside + revert H/4 cap to plain H → Task 2 Step 1 (the `if (!inside)` growth block, capped at `H_now`).
- §4.1 delivery may take one/several coarse steps → emergent from the loop (no special code); covered by the coarse regime.
- §4.2 entry boundary snap (REQUIRED) → Task 2 Step 1 (the bisection when `inside_envelope(rs.end_state) != inside`, with `inside == false`).
- §4.3 fixed uniform fine `ds = min(H,L)/k`, k=4, midpoint-S, no rejection/gate/cap → Task 2 Step 1 (the `inside` branch + unchanged RT block).
- §4.4 exit boundary snap (symmetric) → Task 2 Step 1 (same bisection with `inside == true`).
- §4.5 remove z-gate + H/4 cap; keep mid_state + raymarch_exits_outer → Task 2 (call removed; cap removed) + Task 3 (function removed).
- §5 `noise_correlation_length(r)` → Task 1.
- §6 file-by-file → Tasks 1-4 match the table.
- §7 edge cases: multi-crossing/photon ring (per-iteration regime test — Task 2); thin disk (ds scales with H — Task 2); turbulence pushed hard (k constant — Task 2 `FINE_SAMPLES_PER_CORR`); snap failure (bounded 16 iters — Task 2); r>r_max (direction-aware exit retained — Task 2).
- §8 testing: unit (Task 1 accessor; Task 3 keeps `raymarch_exits_outer`), integration banding/debug-pixel/visual/MAX_STEPS (Task 4).
- §9 `k = FINE_SAMPLES_PER_CORR` constexpr default 4 → Task 2 Step 1.

Placeholder scan: no "TBD"/"implement later". All code blocks are complete. Task 4 Step 2 contains a deliberate `0.XXX` to be filled with a *measured* value (not a placeholder for unwritten logic) and an explicit judgment gate.

Type consistency: `noise_correlation_length(double r) const` declared (Task 1 Step 3) and called as `vol_disk_->noise_correlation_length(r)` (Task 2 Step 1) — match. `inside_envelope` lambda used consistently. `raymarch_exits_outer(r, r_max, dr_dlambda)` signature unchanged and still called identically. `RombergStep::mid_state`/`dtau`/`end_state`/`max_err` all used as defined.
