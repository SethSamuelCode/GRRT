# Raymarch transit sampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make transversal disk transits (far-side light bent around the BH) render correctly by sampling the radiative-transfer source function at each Romberg step's midpoint and forcing fine steps while crossing the disk envelope.

**Architecture:** Surface `romberg_step`'s already-computed midpoint as `RombergStep::mid_state`. In `raymarch_volumetric`, sample the source function at that midpoint instead of the step end, and reject+shrink any step whose signed z-interval overlaps the disk envelope while `|Δz| > H/4`. A small pure predicate `step_needs_z_refinement` encodes the signed-interval gate and is unit-tested directly.

**Tech Stack:** C++23, MSVC 2022 / CMake, OpenMP (existing). Tests are standalone `.exe`s with an inline `EXPECT_*` macro.

**Spec:** `docs/superpowers/specs/2026-05-15-raymarch-transit-sampling-design.md`

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `include/grrt/geodesic/romberg_step.h` | Modify | Add `mid_state` field to `RombergStep` |
| `src/romberg_step.cpp` | Modify | Populate `mid_state` (main path + empty-channel path) |
| `include/grrt/geodesic/raymarch_step_control.h` | Create | Header-only inline `step_needs_z_refinement` predicate |
| `src/geodesic_tracer.cpp` | Modify (`raymarch_volumetric`, ~L246-351) | Midpoint-S sampling, z-resolution rejection, growth cap, MAX_STEPS bump |
| `tests/test_romberg_step.cpp` | Modify | Assert `mid_state` is the half-step junction |
| `tests/test_raymarch_step_control.cpp` | Create | Unit-test `step_needs_z_refinement` |
| `CMakeLists.txt` | Modify | Add `test-raymarch-step-control` target |

---

## Task 1: `RombergStep::mid_state` field + populate + test

**Files:**
- Modify: `include/grrt/geodesic/romberg_step.h`
- Modify: `src/romberg_step.cpp`
- Modify: `tests/test_romberg_step.cpp`

- [ ] **Step 1: Add the failing test to `tests/test_romberg_step.cpp`**

Add this function before `int main()`:

```cpp
// Test 4: mid_state must equal the geodesic state at the half-step junction,
// i.e. integrator.step_kerr(metric, start, ds/2). This is the point
// raymarch_volumetric samples the source function at (midpoint-S sampling).
static void test_mid_state_is_half_step() {
    ConstantSampler sampler{1.0};
    Kerr metric(1.0, 0.0);
    RK4 integrator;

    GeodesicState start;
    start.position = {0.0, 10.0, 1.5707963267948966, 0.0};
    start.momentum = {-0.8, 1.0, 0.0, 0.0};  // null radial geodesic at r=10, M=1

    constexpr double channels[] = {550e-7};
    const double ds = 0.1;

    RombergStep r = romberg_step(start, ds,
                                  std::span<const double>{channels, 1},
                                  sampler, metric, integrator);

    const GeodesicState expected_mid =
        integrator.step_kerr(metric, start, 0.5 * ds);

    EXPECT_NEAR(r.mid_state.position[1], expected_mid.position[1], 1e-12);
    EXPECT_NEAR(r.mid_state.position[2], expected_mid.position[2], 1e-12);
    EXPECT_NEAR(r.mid_state.momentum[1], expected_mid.momentum[1], 1e-12);
}

// Test 5: the empty-channel path must still set a valid (finite) mid_state.
static void test_mid_state_empty_channels() {
    ConstantSampler sampler{1.0};
    Kerr metric(1.0, 0.0);
    RK4 integrator;

    GeodesicState start;
    start.position = {0.0, 10.0, 1.5707963267948966, 0.0};
    start.momentum = {-0.8, 1.0, 0.0, 0.0};

    const double ds = 0.1;
    RombergStep r = romberg_step(start, ds,
                                  std::span<const double>{}, // zero channels
                                  sampler, metric, integrator);

    // mid_state.position[1] (r) must be finite and between start and end r.
    const double r_mid = r.mid_state.position[1];
    if (!std::isfinite(r_mid)) {
        std::printf("FAIL: empty-channel mid_state.r not finite (%.6e)\n", r_mid);
        failures++;
    }
}
```

Add both calls to `main()` after `test_romberg_order_convergence();`:

```cpp
    test_mid_state_is_half_step();
    test_mid_state_empty_channels();
```

- [ ] **Step 2: Build and run — test must fail to compile (mid_state doesn't exist yet)**

Run:
```
cmake --build build --config Release
```
Expected: COMPILE ERROR — `'mid_state' is not a member of 'grrt::RombergStep'`.

- [ ] **Step 3: Add `mid_state` to the `RombergStep` struct**

In `include/grrt/geodesic/romberg_step.h`, change the struct (currently lines 22-28):

```cpp
/// Result of one Romberg-controlled raymarch step.
struct RombergStep {
    GeodesicState end_state;                           ///< Geodesic state at end of accepted half-step path.
    GeodesicState mid_state;                           ///< Geodesic state at the step midpoint (junction of the two half-steps).
    std::array<double, MAX_ROMBERG_CHANNELS> dtau;     ///< Per-channel Δτ from the half-step pass (more accurate).
    double max_err;                                    ///< Max over channels of |Δτ_full − Δτ_half|.
    double ds_taken;                                   ///< = ds_proposed (helper does not shrink; caller does).
    int n_channels;                                    ///< Count of valid entries in dtau[].
};
```

- [ ] **Step 4: Populate `mid_state` in `src/romberg_step.cpp`**

In the main path, after the half-step `mid` is computed (currently line 71-72) and before/at the `out.end_state = end_half;` assignment (line 87), add `out.mid_state = mid;`. The block becomes:

```cpp
    // --- Half-step pass (two steps of ds/2) ---
    const double half = 0.5 * ds_proposed;
    const GeodesicState mid      = integrator.step_kerr(metric, start_state, half);
    const GeodesicState end_half = integrator.step_kerr(metric, mid,         half);
    sampler.sample_integrand(mid,      channels_nu_obs, span_mid);
    sampler.sample_integrand(end_half, channels_nu_obs, span_end_h);
    out.mid_state = mid;
```

In the empty-channel early-return path (currently lines 36-41), set `mid_state` too:

```cpp
    if (out.n_channels <= 0) {
        // Empty channel list: nothing to integrate. Still advance state.
        out.mid_state = integrator.step_kerr(metric, start_state, 0.5 * ds_proposed);
        out.end_state = integrator.step_kerr(metric, start_state, ds_proposed);
        out.max_err = 0.0;
        return out;
    }
```

- [ ] **Step 5: Build and run — tests must pass**

Run:
```
cmake --build build --config Release
.\build\Release\test-romberg-step.exe
```
Expected: `=== 0 failures ===`, exit 0. All 5 tests pass.

- [ ] **Step 6: Commit (DO NOT run this — hand the message to the user)**

The user composes commits. Provide this message text and wait for confirmation:
```
feat(romberg): surface step midpoint as RombergStep::mid_state

Exposes the half-step junction romberg_step already computes, so
raymarch_volumetric can sample the source function at the step midpoint
(needed for correct emission on transversal disk transits). Zero new
integration on the hot path. Empty-channel path also populates mid_state.
```

---

## Task 2: `step_needs_z_refinement` predicate + unit test

**Files:**
- Create: `include/grrt/geodesic/raymarch_step_control.h`
- Create: `tests/test_raymarch_step_control.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create the header `include/grrt/geodesic/raymarch_step_control.h`**

```cpp
#ifndef GRRT_GEODESIC_RAYMARCH_STEP_CONTROL_H
#define GRRT_GEODESIC_RAYMARCH_STEP_CONTROL_H

#include <algorithm>
#include <cmath>

namespace grrt {

/// Returns true if a raymarch step whose Cartesian-z endpoints are z0 and z1
/// should be refined (shrunk) for z-resolution.
///
/// The step needs refinement when its signed z-interval
/// [min(z0,z1), max(z0,z1)] overlaps the disk's vertical extent [-env, +env]
/// AND its vertical excursion |z1 - z0| exceeds quarter_H (a quarter scale
/// height). The signed-interval overlap test — NOT endpoint membership — is
/// what catches transversal transits, where both endpoints lie outside the
/// envelope (|z| > env) but the path crosses z = 0 through dense disk material.
///
/// @param z0       Cartesian z (= r*cos(theta)) at the step start.
/// @param z1       Cartesian z at the step end.
/// @param quarter_H  H/4 at a representative radius; the max allowed |Δz|.
/// @param env      Disk vertical envelope z_max(r) + H(r) at that radius.
inline bool step_needs_z_refinement(double z0, double z1,
                                    double quarter_H, double env) {
    const double dz = std::abs(z1 - z0);
    const bool crosses = (std::min(z0, z1) < env) && (std::max(z0, z1) > -env);
    return crosses && dz > quarter_H;
}

} // namespace grrt

#endif
```

- [ ] **Step 2: Create the failing test `tests/test_raymarch_step_control.cpp`**

```cpp
// tests/test_raymarch_step_control.cpp
//
// Unit tests for step_needs_z_refinement — the signed-interval z-resolution
// gate. Pure scalar logic; no disk, metric, or integrator needed.

#include "grrt/geodesic/raymarch_step_control.h"

#include <cstdio>

using namespace grrt;

int failures = 0;

static void check(const char* name, bool got, bool expected) {
    if (got != expected) {
        std::printf("FAIL %s: got=%d expected=%d\n", name, got ? 1 : 0,
                    expected ? 1 : 0);
        failures++;
    }
}

int main() {
    std::printf("Running test_raymarch_step_control...\n");

    constexpr double qH  = 0.005;   // quarter scale height
    constexpr double env = 0.076;   // disk envelope

    // Transversal transit: both endpoints outside env, path crosses z=0.
    check("transversal_transit",
          step_needs_z_refinement(-0.27, 0.12, qH, env), true);

    // Entirely below disk: no envelope overlap (max=-0.20 < -env).
    check("entirely_below",
          step_needs_z_refinement(-0.30, -0.20, qH, env), false);

    // Skim far above disk: no envelope overlap (min=+0.49 > +env).
    check("skim_far_above",
          step_needs_z_refinement(0.50, 0.49, qH, env), false);

    // Already-fine step near the disk top: overlaps env but dz < quarter_H.
    check("already_fine_near_top",
          step_needs_z_refinement(0.072, 0.070, qH, env), false);

    // Coarse step straddling the disk top: overlaps env and dz > quarter_H.
    check("coarse_at_disk_top",
          step_needs_z_refinement(0.08, 0.07, qH, env), true);

    // Coarse step fully inside the disk crossing midplane.
    check("inside_disk_coarse",
          step_needs_z_refinement(-0.03, 0.03, qH, env), true);

    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
```

- [ ] **Step 3: Add the test target to `CMakeLists.txt`**

Find the block of `add_executable(test-... )` lines (search for `test-romberg-step`). Add, mirroring the pattern:

```cmake
add_executable(test-raymarch-step-control tests/test_raymarch_step_control.cpp)
target_link_libraries(test-raymarch-step-control PRIVATE grrt)
```

(The test only needs the header, which is header-only, but linking `grrt` matches the existing convention and is harmless.)

- [ ] **Step 4: Build and run — test must pass**

Run:
```
cmake --build build --config Release
.\build\Release\test-raymarch-step-control.exe
```
Expected: `=== 0 failures ===`, exit 0. All 6 cases pass. (This is a header-only pure function; the test exercises it directly, no failing-first phase needed beyond confirming it compiles and the logic is right.)

- [ ] **Step 5: Commit (DO NOT run — hand the message to the user)**

```
feat(raymarch): add step_needs_z_refinement z-resolution gate

Pure predicate: a raymarch step needs finer steps when its signed
z-interval overlaps the disk envelope [-env,+env] AND |Δz| > H/4. The
signed-interval test (not endpoint membership) is what catches
transversal disk transits — both endpoints outside the envelope but the
path crosses z=0. Header-only inline; unit-tested across 6 cases.
```

---

## Task 3: Wire midpoint-S + z-resolution control into `raymarch_volumetric`

**Files:**
- Modify: `src/geodesic_tracer.cpp` (`raymarch_volumetric`, ~L246-351)

This task bundles four coordinated changes to one function. `raymarch_volumetric`
is not unit-testable in isolation (private method, needs disk/sampler), so it is
verified by integration in Task 4. Build-clean + existing tests staying green is
the gate for this task.

- [ ] **Step 1: Add the include**

At the top of `src/geodesic_tracer.cpp`, after the existing
`#include "grrt/geodesic/romberg_step.h"`, add:

```cpp
#include "grrt/geodesic/raymarch_step_control.h"
```

- [ ] **Step 2: Bump MAX_STEPS**

In `raymarch_volumetric`, change (currently line 271):

```cpp
    constexpr int MAX_STEPS = 4096;
```
to:
```cpp
    constexpr int MAX_STEPS = 16384;   // headroom for fine transit stepping (thin disks)
```

- [ ] **Step 3: Add the z-resolution rejection after the max_err rejection**

The current accept/refine block reads (around lines 283-295):

```cpp
        // Reject if error exceeds tolerance.
        if (rs.max_err > raymarch_tol_) {
            const double H_local = vol_disk_->scale_height(state.position[1]);
            const double ds_floor = H_local / 256.0;
            if (ds_proposed <= ds_floor) {
                // Already at floor — accept anyway (LUT discontinuity is the cause,
                // not the integrator).
            } else {
                ds_proposed = std::max(ds_proposed * 0.5, ds_floor);
                continue;
            }
        }
        step_count++;
```

Insert the z-resolution rejection between the `max_err` block and `step_count++`:

```cpp
        // Reject if error exceeds tolerance.
        if (rs.max_err > raymarch_tol_) {
            const double H_local = vol_disk_->scale_height(state.position[1]);
            const double ds_floor = H_local / 256.0;
            if (ds_proposed <= ds_floor) {
                // Already at floor — accept anyway (LUT discontinuity is the cause,
                // not the integrator).
            } else {
                ds_proposed = std::max(ds_proposed * 0.5, ds_floor);
                continue;
            }
        }

        // z-resolution control: a step whose signed z-interval overlaps the
        // disk envelope must not jump more than H/4 in z, or its midpoint
        // (where we sample the source function) won't reliably land inside the
        // disk. Reject and halve ds, reusing the shrink-and-retry loop. Gated
        // on envelope overlap so empty-space steps stay coarse.
        {
            const double z0 = state.position[1]
                            * std::cos(state.position[2]);
            const double z1 = rs.end_state.position[1]
                            * std::cos(rs.end_state.position[2]);
            const double r_for_H = rs.mid_state.position[1];
            const double H_z   = vol_disk_->scale_height(r_for_H);
            const double env_z = vol_disk_->z_max_at(r_for_H) + H_z;
            const double ds_floor_z = H_z / 256.0;
            if (step_needs_z_refinement(z0, z1, 0.25 * H_z, env_z)
                && ds_proposed > ds_floor_z) {
                ds_proposed = std::max(ds_proposed * 0.5, ds_floor_z);
                continue;
            }
        }
        step_count++;
```

- [ ] **Step 4: Move source-function sampling from end_state to mid_state**

The source-function block currently samples at `rs.end_state` (around lines 298-336):

```cpp
        // Accepted: per-channel radiative transfer using rs.dtau.
        // Compute source function at the END of the half-step path.
        const GeodesicState& end = rs.end_state;
        const double r_end       = end.position[1];
        const double theta_end   = end.position[2];
        const double phi_end     = end.position[3];
        const double z_end       = r_end * std::cos(theta_end);

        const double rho_cgs = vol_disk_->density_cgs(r_end, z_end, phi_end);
        const double T_local = vol_disk_->temperature(r_end, std::abs(z_end));
        if (rho_cgs > 0.0 && T_local > 0.0) {
            // Redshift factor at the end-state.
            double ut_emit = 0.0, ur_emit = 0.0, uphi_emit = 0.0;
            if (r_end >= vol_disk_->r_isco()) {
                vol_disk_->circular_velocity(r_end, ut_emit, uphi_emit);
            } else {
                vol_disk_->plunging_velocity(r_end, theta_end, ut_emit, ur_emit, uphi_emit);
            }
            const double p_dot_u_emit = end.momentum[0] * ut_emit
                                       + end.momentum[1] * ur_emit
                                       + end.momentum[3] * uphi_emit;
            const double p_dot_u_obs  = end.momentum[0] * ut_obs;
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
```

Replace `end`/`_end` with `mid`/`_mid` throughout this block (the `dtau` and the
J/T accumulation stay identical — only the source sampling point moves):

```cpp
        // Accepted: per-channel radiative transfer using rs.dtau.
        // Sample the source function at the step MIDPOINT (not the end). For a
        // transversal disk transit the end can lie outside the disk (density 0)
        // even though the path crossed dense material; the midpoint lands at
        // representative density. dtau (the optical depth over the step) is
        // unchanged — only the source sampling point moves. See spec §5.3.
        const GeodesicState& mid = rs.mid_state;
        const double r_mid       = mid.position[1];
        const double theta_mid   = mid.position[2];
        const double phi_mid     = mid.position[3];
        const double z_mid       = r_mid * std::cos(theta_mid);

        const double rho_cgs = vol_disk_->density_cgs(r_mid, z_mid, phi_mid);
        const double T_local = vol_disk_->temperature(r_mid, std::abs(z_mid));
        if (rho_cgs > 0.0 && T_local > 0.0) {
            // Redshift factor at the mid-state.
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
```

- [ ] **Step 5: Cap step-growth inside the disk envelope**

The current growth block reads (around lines 341-345):

```cpp
        // Step-size growth: well under tolerance → grow, capped at 1·H.
        if (rs.max_err < raymarch_tol_ / 8.0) {
            const double H_now = vol_disk_->scale_height(r);
            ds_proposed = std::min(ds_proposed * 2.0, H_now);
        }
```

Replace with an envelope-aware cap:

```cpp
        // Step-size growth: well under tolerance → grow. Cap at H/4 while the
        // ray is inside the disk envelope (avoids grow-then-reject thrashing
        // against the z-resolution gate), else cap at H. Point test here, not
        // the interval test used for rejection: a ray currently inside keeps
        // fine steps; the single boundary step on entry is caught by the gate.
        if (rs.max_err < raymarch_tol_ / 8.0) {
            const double z_now   = r * std::cos(state.position[2]);
            const double H_now   = vol_disk_->scale_height(r);
            const double env_now = vol_disk_->z_max_at(r) + H_now;
            const double grow_cap = (std::abs(z_now) < env_now)
                                  ? (0.25 * H_now)
                                  : H_now;
            ds_proposed = std::min(ds_proposed * 2.0, grow_cap);
        }
```

- [ ] **Step 6: Build and run the full unit-test suite**

Run:
```
cmake --build build --config Release
.\build\Release\test-romberg-step.exe
.\build\Release\test-raymarch-step-control.exe
.\build\Release\test-disk-step-entry.exe
.\build\Release\test-spectral.exe
```
Expected: all `=== 0 failures ===`. (test-volumetric is run in Task 4 because it
includes the banding regression that may need recalibration.)

- [ ] **Step 7: Commit (DO NOT run — hand the message to the user)**

```
fix(raymarch): midpoint source sampling + z-resolution step control

raymarch_volumetric sampled the radiative-transfer source function S at
each Romberg step's END, which lands at zero density for transversal disk
transits (far-side light bent around the BH crossing the midplane). Result:
the bottom half of the disk rendered black.

Fix (spec 2026-05-15):
- Sample S, density, temperature, and redshift at the step MIDPOINT
  (rs.mid_state) instead of the end. 2nd-order; catches the emission peak.
- Reject + halve ds when a step's signed z-interval overlaps the disk
  envelope and |Δz| > H/4 (step_needs_z_refinement), so the midpoint lands
  at representative density.
- Cap step-growth at H/4 inside the envelope to avoid grow-then-reject
  thrashing.
- Bump raymarch MAX_STEPS 4096 -> 16384 for thin-disk transit headroom.

dtau (optical depth over the step) is unchanged. No change to the helper,
disk model, or trace-loop call sites.
```

---

## Task 4: Integration verification — debug-pixel, visual smoke, banding

**Files:** none modified (verification only; may modify `tests/test_volumetric.cpp` only if banding needs recalibration — see Step 4).

- [ ] **Step 1: Debug-pixel at the previously-failing position**

Run:
```
.\build\Release\grrt-cli.exe --disk-volumetric --samples 1 --width 256 --height 256 --fov 30 --debug-pixel 150 180 --output /tmp/dbg.png --force 2>&1 | grep -E "ENTRY|RAYMARCH|ESCAPED|HORIZON|MAX" | head -40
```
Expected: the ENTRY events now show `RAYMARCH exit color=(...)` with values
meaningfully non-zero (not ~1e-19). Trace terminates `ESCAPED` or `HORIZON`,
not `MAX STEPS`. Report the color values.

- [ ] **Step 2: Banding regression test**

Run:
```
.\build\Release\test-volumetric.exe 2>&1 | grep -A 4 "Banding"
```
Expected: `Banding regression test (256x256 spp=30)` then `PASS` or `FAIL` with
the metric value. Record the metric.

- [ ] **Step 3: Visual smoke render (success criterion)**

Run:
```
.\build\Release\grrt-cli.exe --disk-volumetric --samples 100 --width 256 --height 256 --output transit_fix.png --force --fov 30
```
Report wall-clock time and file size. Do NOT open the image (the executor can't
see it). Leave `transit_fix.png.png` on disk for the user to inspect: the bottom
half of the disk should now render rather than being black.

- [ ] **Step 4: If banding regressed, recalibrate (only if Step 2 FAILED)**

If `test_no_horizontal_bands` failed but Step 1's debug-pixel shows correct
non-zero emission and Step 3's render looks visually correct, the fix changed
the (previously-buggy) banding baseline. Recalibrate following the documented
protocol in `tests/test_volumetric.cpp` (the comment block in
`test_no_horizontal_bands` that records prior calibration values):
1. Record the new metric value.
2. Add a calibration-history line to the comment (e.g.
   `// - Midpoint-S + z-resolution (current): rel = 0.XXX`).
3. Set the threshold with headroom above the new baseline, matching the prior
   convention.
This is a JUDGMENT step — surface the before/after metric to the user with a
recommendation and wait for their call before changing the threshold. Do NOT
loosen the threshold silently.

- [ ] **Step 5: Report back to the user**

Summarize for the user (this is the end of the plan — hand off for visual
inspection and commit):
- Debug-pixel (150,180) RAYMARCH color values (non-zero?)
- Banding metric value + pass/fail (recalibrated?)
- Visual render path + wall-clock time
- All unit tests green (modulo pre-existing `test_tau_midplane_near_target`)
- Request visual inspection of `transit_fix.png.png`: does the bottom-half disk render?

---

## Plan self-review

Spec coverage scan against `2026-05-15-raymarch-transit-sampling-design.md`:

- §5.1 `RombergStep::mid_state` field → Task 1 Step 3.
- §5.2 `romberg_step` populates `mid_state` (both paths) → Task 1 Step 4.
- §5.3 Midpoint source sampling → Task 3 Step 4.
- §5.4 `step_needs_z_refinement` testable free function → Task 2 Step 1 (header), Step 2 (test).
- §5.5 z-resolution control + growth cap → Task 3 Steps 3 and 5.
- §5.6 MAX_STEPS headroom → Task 3 Step 2.
- §6.1 shrink-loop floor → Task 3 Step 3 (the `ds_proposed > ds_floor_z` guard).
- §6.2 very thin disk → MAX_STEPS bump (Task 3 Step 2) + graceful exit (existing loop behavior).
- §6.3 mid_state empty-channel path → Task 1 Step 4 + Task 1 Step 1 test `test_mid_state_empty_channels`.
- §6.4 shallow grazes → covered by the predicate logic (Task 2 test `already_fine_near_top`).
- §6.5 no-op-detection interaction → unchanged code; no task needed (verified by debug-pixel still terminating cleanly, Task 4 Step 1).
- §6.6 performance → acknowledged; Task 4 Step 3 reports wall-clock.
- §7.1 unit tests → Task 1 (mid_state), Task 2 (predicate).
- §7.2 integration verification → Task 4 Steps 1.
- §7.3 visual smoke + banding → Task 4 Steps 2-4.

Placeholder scan: no "TBD"/"TODO"/"implement later". All code blocks are complete. Task 4 Step 4 is conditional-and-judgment (recalibration) but spells out the exact protocol and the wait-for-user gate, not a placeholder.

Type consistency: `RombergStep::mid_state` (Task 1) is read in Task 3 as `rs.mid_state`. `step_needs_z_refinement(z0, z1, quarter_H, env)` (Task 2) is called in Task 3 Step 3 as `step_needs_z_refinement(z0, z1, 0.25 * H_z, env_z)` — signature matches. `integrator.step_kerr(metric, start, 0.5*ds)` used in Task 1 test matches the real `RK4::step_kerr` signature.
