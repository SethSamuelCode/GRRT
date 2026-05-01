# Cliff-Aware Raymarch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed `H_max=H` clamp in `raymarch_volumetric` with a Romberg-controlled adaptive step that bounds per-step τ error to a user-tunable tolerance, eliminating the photosphere-cliff banding artifact.

**Architecture:** Introduce a `romberg_step` helper in `src/romberg_step.cpp` (declared in `include/grrt/geodesic/romberg_step.h`) that takes a `StepSampler` interface and returns a `RombergStep` struct with per-channel `dtau[]` and `max_err`. The RGB raymarcher uses the helper through a `VolumetricDiskSampler` adaptor. The user controls fidelity via `--raymarch-tol`.

**Tech Stack:** C++23, CMake/MSBuild, `std::span`, `std::array`. Tests use the existing assertion macros in `tests/test_volumetric.cpp` style.

**Spec:** `docs/superpowers/specs/2026-05-01-cliff-aware-raymarch-design.md`

**Notes for the implementer:**
- The spec accidentally calls the params struct `GRRTContextParams`. The actual struct in this codebase is `GRRTParams`, defined in `include/grrt/types.h`. Use `GRRTParams`.
- The spec says headers go in `include/grrt/render/`. The actual codebase puts `geodesic_tracer.h` in `include/grrt/geodesic/`. Place the helper header alongside it: `include/grrt/geodesic/romberg_step.h`.
- The `H/8` debug edit in `src/geodesic_tracer.cpp` (line ~329) is currently committed. Task 9 reverts it as part of switching to the helper.
- Build with `cmake --build build --config Release` on Windows (Visual Studio 2022 generator).
- Run a test target with `build/Release/test-<name>.exe`.

---

## File structure

**New:**
- `include/grrt/geodesic/romberg_step.h` — `MAX_CH`, `RombergStep`, `StepSampler` (virtual), `romberg_step()` declaration.
- `src/romberg_step.cpp` — `romberg_step()` implementation.
- `tests/test_romberg_step.cpp` — unit tests with a synthetic `StepSampler` having a closed-form `ρ(z)` profile.

**Modified:**
- `include/grrt/types.h` — add `double raymarch_tol` to `GRRTParams`.
- `src/api.cpp` — pass `params->raymarch_tol` to `GeodesicTracer`.
- `cli/main.cpp` — `--raymarch-tol` flag and help line; default `1e-2`.
- `include/grrt/geodesic/geodesic_tracer.h` — add `double raymarch_tol_` member; new constructor parameter.
- `src/geodesic_tracer.cpp` — `raymarch_volumetric` rewired to use helper; revert H/8 hack; define a private `VolumetricDiskSampler` struct adaptor.
- `tests/test_volumetric.cpp` — tolerance-convergence test, banding-regression test.
- `CMakeLists.txt` (root) — add `test-romberg-step` target.

---

## Task 1: Skeleton header for `romberg_step`

**Files:**
- Create: `include/grrt/geodesic/romberg_step.h`

- [ ] **Step 1: Write the header skeleton**

```cpp
#ifndef GRRT_GEODESIC_ROMBERG_STEP_H
#define GRRT_GEODESIC_ROMBERG_STEP_H

#include "grrt/geodesic/integrator.h"
#include "grrt_export.h"

#include <array>
#include <span>

namespace grrt {

class Kerr;
class RK4;

/// Maximum number of control channels carried through a single helper call.
/// Covers RGB (3) and modest spectral outputs without heap allocation.
/// Spectral callers wanting more bins must either raise this constant
/// or split their bins across multiple helper calls.
constexpr int MAX_ROMBERG_CHANNELS = 32;

/// Result of one Romberg-controlled raymarch step.
struct RombergStep {
    GeodesicState end_state;                           ///< Geodesic state at end of accepted half-step path.
    std::array<double, MAX_ROMBERG_CHANNELS> dtau;     ///< Per-channel Δτ from the half-step pass (more accurate).
    double max_err;                                    ///< Max over channels of |Δτ_full − Δτ_half|.
    double ds_taken;                                   ///< = ds_proposed (helper does not shrink; caller does).
    int n_channels;                                    ///< Count of valid entries in dtau[].
};

/// Sampler interface: callers provide one of these to romberg_step()
/// so the helper can query the integrand κρ|p·u_emit| at any state.
/// VolumetricDiskSampler in geodesic_tracer.cpp wraps the production
/// VolumetricDisk; tests provide synthetic implementations.
struct GRRT_EXPORT StepSampler {
    virtual ~StepSampler() = default;

    /// Sample the per-channel integrand at a geodesic state.
    /// integrand[ch] = κ_total(ν_emit, ρ, T) · ρ · |p·u_emit|
    /// where ν_emit = |g| · channels_nu_obs[ch] and g is the redshift factor.
    /// If the state is outside the optically active region, the sampler
    /// must zero the integrand[] entries and return false.
    /// Returns true when the integrand was sampled.
    virtual bool sample_integrand(const GeodesicState& state,
                                  std::span<const double> channels_nu_obs,
                                  std::span<double> integrand) const = 0;
};

/// Take one Romberg-controlled raymarch step.
/// Caller manages step proposal/growth/shrinkage between calls.
/// Helper does ONE geodesic full step + TWO half-steps to estimate error.
GRRT_EXPORT RombergStep romberg_step(
    const GeodesicState& start_state,
    double ds_proposed,
    std::span<const double> channels_nu_obs,
    const StepSampler& sampler,
    const Kerr& metric,
    const RK4& integrator);

} // namespace grrt

#endif
```

- [ ] **Step 2: Verify compilation (header-only at this point)**

Run: `cmake --build build --config Release --target grrt`
Expected: Build succeeds. The header compiles even though no `.cpp` exists yet — nothing references `romberg_step()` so the linker doesn't complain.

- [ ] **Step 3: Commit**

```bash
git add include/grrt/geodesic/romberg_step.h
git commit -m "feat(romberg): add helper header (skeleton)"
```

---

## Task 2: Test scaffolding + first failing test

**Files:**
- Create: `tests/test_romberg_step.cpp`
- Modify: `CMakeLists.txt` (root)

- [ ] **Step 1: Add test target to root `CMakeLists.txt`**

Find the block of `add_executable(test-volumetric ...)` lines near the end (existing tests) and add right after:

```cmake
add_executable(test-romberg-step tests/test_romberg_step.cpp)
target_link_libraries(test-romberg-step PRIVATE grrt)
```

- [ ] **Step 2: Write the test file with one failing test**

```cpp
// tests/test_romberg_step.cpp
//
// Unit tests for romberg_step using synthetic StepSampler implementations.
// We test against analytic dτ values for closed-form integrands.

#include "grrt/geodesic/romberg_step.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/spacetime/kerr.h"
#include "grrt/math/vec3.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace grrt;

#define EXPECT_NEAR(actual, expected, tol)                                    \
    do {                                                                      \
        double a = (actual), e = (expected), t = (tol);                       \
        if (std::abs(a - e) > t) {                                            \
            std::printf("FAIL %s:%d: %s ≈ %s (got %.6e, expected %.6e, tol %.2e)\n", \
                        __FILE__, __LINE__, #actual, #expected, a, e, t);     \
            std::exit(1);                                                     \
        }                                                                     \
    } while (0)

// --- Synthetic samplers ---

// Constant integrand: κρ|p·u_emit| = K everywhere. Then dτ = K·Δλ exactly.
struct ConstantSampler : StepSampler {
    double K;
    explicit ConstantSampler(double k) : K(k) {}
    bool sample_integrand(const GeodesicState& /*state*/,
                          std::span<const double> channels,
                          std::span<double> integrand) const override {
        for (size_t i = 0; i < channels.size(); ++i) integrand[i] = K;
        return true;
    }
};

// Test 1: a constant integrand should give dτ = K·ds exactly,
// and Romberg's full and half estimates should agree (max_err == 0).
static void test_constant_integrand() {
    ConstantSampler sampler{2.5};
    Kerr metric(1.0, 0.0);  // Schwarzschild as a Kerr-with-spin-0
    RK4 integrator;

    // A simple radial null geodesic at r=10, theta=pi/2.
    GeodesicState start;
    start.position = {0.0, 10.0, 1.5707963267948966, 0.0};
    start.momentum = {-1.0, 0.0, 0.0, 0.0};  // purely temporal — won't move spatially much
    // Note: the test only requires the integrand machinery, not realistic geodesic motion.

    constexpr double channels[] = {550e-7};  // one channel, value irrelevant for ConstantSampler
    const double ds = 0.1;

    RombergStep r = romberg_step(start, ds,
                                  std::span<const double>{channels, 1},
                                  sampler, metric, integrator);

    EXPECT_NEAR(r.dtau[0], 2.5 * 0.1, 1e-12);
    EXPECT_NEAR(r.max_err, 0.0,        1e-12);
    EXPECT_NEAR(r.ds_taken, 0.1,       1e-12);
    if (r.n_channels != 1) {
        std::printf("FAIL: n_channels=%d, expected 1\n", r.n_channels);
        std::exit(1);
    }
}

int main() {
    std::printf("Running test_romberg_step...\n");
    test_constant_integrand();
    std::printf("All tests passed.\n");
    return 0;
}
```

- [ ] **Step 3: Run cmake to refresh build files, then build the test**

Run:
```bash
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release --target test-romberg-step
```

Expected: Build **fails** with a linker error along the lines of "unresolved external symbol grrt::romberg_step" — the function is declared but not defined.

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt tests/test_romberg_step.cpp
git commit -m "test(romberg): scaffold test target with constant-integrand failing test"
```

---

## Task 3: Implement `romberg_step` body — single-channel constant integrand passes

**Files:**
- Create: `src/romberg_step.cpp`

- [ ] **Step 1: Write the implementation**

```cpp
// src/romberg_step.cpp

#include "grrt/geodesic/romberg_step.h"
#include "grrt/geodesic/rk4.h"
#include "grrt/spacetime/kerr.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace grrt {

RombergStep romberg_step(
    const GeodesicState& start_state,
    double ds_proposed,
    std::span<const double> channels_nu_obs,
    const StepSampler& sampler,
    const Kerr& metric,
    const RK4& integrator)
{
    RombergStep out{};
    out.ds_taken = ds_proposed;
    out.n_channels = static_cast<int>(channels_nu_obs.size());

    if (out.n_channels <= 0) {
        // Empty channel list: nothing to integrate. Still advance state.
        out.end_state = integrator.step_kerr(metric, start_state, ds_proposed);
        out.max_err = 0.0;
        return out;
    }

    // Per-sample integrand storage (sized to MAX_ROMBERG_CHANNELS).
    std::array<double, MAX_ROMBERG_CHANNELS> i_start{};
    std::array<double, MAX_ROMBERG_CHANNELS> i_mid{};
    std::array<double, MAX_ROMBERG_CHANNELS> i_end_full{};
    std::array<double, MAX_ROMBERG_CHANNELS> i_end_half{};

    // Spans bounded to the actual channel count.
    std::span<double> span_start{i_start.data(),    static_cast<size_t>(out.n_channels)};
    std::span<double> span_mid  {i_mid.data(),      static_cast<size_t>(out.n_channels)};
    std::span<double> span_end_f{i_end_full.data(), static_cast<size_t>(out.n_channels)};
    std::span<double> span_end_h{i_end_half.data(), static_cast<size_t>(out.n_channels)};

    // Sample at start (shared between full and half passes).
    sampler.sample_integrand(start_state, channels_nu_obs, span_start);

    // --- Full step pass ---
    GeodesicState end_full = integrator.step_kerr(metric, start_state, ds_proposed);
    sampler.sample_integrand(end_full, channels_nu_obs, span_end_f);

    // Trapezoidal Δτ_full[ch] = 0.5 · (i_start + i_end_full) · ds
    std::array<double, MAX_ROMBERG_CHANNELS> dtau_full{};
    for (int ch = 0; ch < out.n_channels; ++ch) {
        dtau_full[ch] = 0.5 * (i_start[ch] + i_end_full[ch]) * ds_proposed;
    }

    // --- Half-step pass (two steps of ds/2) ---
    const double half = 0.5 * ds_proposed;
    GeodesicState mid     = integrator.step_kerr(metric, start_state, half);
    GeodesicState end_half = integrator.step_kerr(metric, mid,         half);
    sampler.sample_integrand(mid,      channels_nu_obs, span_mid);
    sampler.sample_integrand(end_half, channels_nu_obs, span_end_h);

    // Composite trapezoid Δτ_half[ch] = 0.5·(i_start + 2·i_mid + i_end_half) · half
    for (int ch = 0; ch < out.n_channels; ++ch) {
        out.dtau[ch] = 0.5 * (i_start[ch] + 2.0 * i_mid[ch] + i_end_half[ch]) * half;
    }

    // --- Error estimate ---
    double err = 0.0;
    for (int ch = 0; ch < out.n_channels; ++ch) {
        err = std::max(err, std::abs(dtau_full[ch] - out.dtau[ch]));
    }
    out.max_err = err;
    out.end_state = end_half;  // half-step path is more accurate

    return out;
}

} // namespace grrt
```

- [ ] **Step 2: Add the .cpp to the library target**

Open root `CMakeLists.txt`, find the list of source files for the `grrt` library (look for `add_library(grrt`), and add `src/romberg_step.cpp` to that list. Build:

```bash
cmake --build build --config Release --target grrt
```

Expected: Build succeeds.

- [ ] **Step 3: Run the test**

```bash
cmake --build build --config Release --target test-romberg-step
build/Release/test-romberg-step.exe
```

Expected output:
```
Running test_romberg_step...
All tests passed.
```

- [ ] **Step 4: Commit**

```bash
git add src/romberg_step.cpp CMakeLists.txt
git commit -m "feat(romberg): implement single-channel trapezoidal step"
```

---

## Task 4: Multi-channel batching test + verification

**Files:**
- Modify: `tests/test_romberg_step.cpp`

- [ ] **Step 1: Add a multi-channel test**

In `tests/test_romberg_step.cpp`, add this sampler before the existing `ConstantSampler`:

```cpp
// Per-channel constant integrand: integrand[ch] = ks[ch].
struct PerChannelConstantSampler : StepSampler {
    std::vector<double> ks;
    explicit PerChannelConstantSampler(std::vector<double> values)
        : ks(std::move(values)) {}
    bool sample_integrand(const GeodesicState& /*state*/,
                          std::span<const double> channels,
                          std::span<double> integrand) const override {
        for (size_t i = 0; i < channels.size(); ++i) {
            integrand[i] = (i < ks.size()) ? ks[i] : 0.0;
        }
        return true;
    }
};
```

And add this test function before `int main()`:

```cpp
static void test_multi_channel_batching() {
    PerChannelConstantSampler sampler({1.0, 2.5, 7.0});  // 3 channels
    Kerr metric(1.0, 0.0);
    RK4 integrator;

    GeodesicState start;
    start.position = {0.0, 10.0, 1.5707963267948966, 0.0};
    start.momentum = {-1.0, 0.0, 0.0, 0.0};

    constexpr double channels[] = {450e-7, 550e-7, 650e-7};
    const double ds = 0.1;

    RombergStep r = romberg_step(start, ds,
                                  std::span<const double>{channels, 3},
                                  sampler, metric, integrator);

    EXPECT_NEAR(r.dtau[0], 1.0 * 0.1, 1e-12);
    EXPECT_NEAR(r.dtau[1], 2.5 * 0.1, 1e-12);
    EXPECT_NEAR(r.dtau[2], 7.0 * 0.1, 1e-12);
    EXPECT_NEAR(r.max_err, 0.0,       1e-12);

    if (r.n_channels != 3) {
        std::printf("FAIL: n_channels=%d, expected 3\n", r.n_channels);
        std::exit(1);
    }
}
```

Also call it from main:

```cpp
int main() {
    std::printf("Running test_romberg_step...\n");
    test_constant_integrand();
    test_multi_channel_batching();
    std::printf("All tests passed.\n");
    return 0;
}
```

(Don't forget to add `#include <vector>` and `#include <utility>` near the top.)

- [ ] **Step 2: Build and run**

```bash
cmake --build build --config Release --target test-romberg-step
build/Release/test-romberg-step.exe
```

Expected: All tests pass. (The implementation already handles multi-channel from Task 3 — this test confirms it.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_romberg_step.cpp
git commit -m "test(romberg): verify multi-channel batching"
```

---

## Task 5: Romberg-order convergence test

The Romberg estimator's error should fall as `O(ds²)` for trapezoid-vs-composite-trapezoid (the textbook order is 4 for full Romberg extrapolation, but our setup compares step-doubled trapezoid which is order 2). We test this empirically by halving `ds` and checking the error drops by ~4×.

**Files:**
- Modify: `tests/test_romberg_step.cpp`

- [ ] **Step 1: Add the order test**

Add this sampler:

```cpp
// Smooth nonlinear integrand for order-of-convergence checks.
// integrand = sin(r)  → analytic dτ over a path that doesn't move r much
// is approximately sin(r₀)·ds plus small corrections. We use the magnitude
// of (full - half) to verify the estimator's order.
struct SmoothNonlinearSampler : StepSampler {
    bool sample_integrand(const GeodesicState& state,
                          std::span<const double> channels,
                          std::span<double> integrand) const override {
        const double r = state.position[1];
        for (size_t i = 0; i < channels.size(); ++i) integrand[i] = std::sin(r);
        return true;
    }
};
```

And the test:

```cpp
static void test_romberg_order_convergence() {
    SmoothNonlinearSampler sampler;
    Kerr metric(1.0, 0.0);
    RK4 integrator;

    GeodesicState start;
    start.position = {0.0, 10.0, 1.5707963267948966, 0.0};
    // Give it a non-degenerate radial momentum so the geodesic moves.
    start.momentum = {-1.0, 0.5, 0.0, 0.0};

    constexpr double channels[] = {550e-7};

    auto run = [&](double ds) {
        return romberg_step(start, ds, std::span<const double>{channels, 1},
                            sampler, metric, integrator).max_err;
    };

    const double err_h    = run(0.1);
    const double err_h2   = run(0.05);
    const double err_h4   = run(0.025);

    // Trapezoid-vs-step-doubled-trapezoid is order 2 in ds.
    // err(h/2) ≈ err(h) / 4. We allow 2× slack to absorb geodesic
    // curvature and the cubic term in the residual.
    const double ratio_1 = err_h    / err_h2;
    const double ratio_2 = err_h2   / err_h4;

    if (!(ratio_1 > 2.0 && ratio_1 < 8.0)) {
        std::printf("FAIL: order ratio_1=%.3f outside [2,8]\n", ratio_1);
        std::exit(1);
    }
    if (!(ratio_2 > 2.0 && ratio_2 < 8.0)) {
        std::printf("FAIL: order ratio_2=%.3f outside [2,8]\n", ratio_2);
        std::exit(1);
    }
}
```

Wire it into `main()` after `test_multi_channel_batching();`.

- [ ] **Step 2: Build and run**

```bash
cmake --build build --config Release --target test-romberg-step
build/Release/test-romberg-step.exe
```

Expected: All tests pass. If ratio falls outside [2,8], the helper has an order-of-convergence bug.

- [ ] **Step 3: Commit**

```bash
git add tests/test_romberg_step.cpp
git commit -m "test(romberg): verify O(ds^2) convergence of error estimator"
```

---

## Task 6: Wire `--raymarch-tol` through `GRRTParams` and CLI

**Files:**
- Modify: `include/grrt/types.h`
- Modify: `cli/main.cpp`

- [ ] **Step 1: Add field to `GRRTParams`**

Open `include/grrt/types.h`. Find the `GRRTParams` struct (line ~20). Add the new field right before the closing `}`:

```c
    /* Per-step optical-depth tolerance for the volumetric raymarcher.
     * Smaller = higher fidelity, slower. Default 1e-2 (set by the CLI/api). */
    double raymarch_tol;
} GRRTParams;
```

- [ ] **Step 2: Add CLI flag handling in `cli/main.cpp`**

Find the help-text block in `cli/main.cpp` (around line 48). Add this line after the `--samples` help line:

```cpp
    std::println("  --raymarch-tol T      Volumetric raymarch τ-tolerance (default: 1e-2)");
```

Find the parameter-default block (around line 86). Add:

```cpp
    params.raymarch_tol = 1e-2;
```

near the other defaults like `params.disk_turbulence = 0.4;`.

Find the flag-parsing block (around line 170, after `--samples`). Add:

```cpp
        } else if (arg("--raymarch-tol")) {
            if (auto v = next()) {
                params.raymarch_tol = std::atof(v);
                if (params.raymarch_tol <= 0.0) {
                    std::println(stderr, "Error: --raymarch-tol must be > 0");
                    return 1;
                }
            }
```

- [ ] **Step 3: Build, run a quick smoke test**

```bash
cmake --build build --config Release --target grrt-cli
build/Release/grrt-cli.exe --help | grep raymarch-tol
```

Expected: Help text appears.

- [ ] **Step 4: Commit**

```bash
git add include/grrt/types.h cli/main.cpp
git commit -m "feat(api): add --raymarch-tol flag (plumbing only)"
```

---

## Task 7: Wire `raymarch_tol` into `GeodesicTracer` + `api.cpp`

**Files:**
- Modify: `include/grrt/geodesic/geodesic_tracer.h`
- Modify: `src/api.cpp`

- [ ] **Step 1: Add member and constructor parameter to `GeodesicTracer`**

Open `include/grrt/geodesic/geodesic_tracer.h`. Find the constructor declaration (line ~40). Replace the constructor with:

```cpp
    GeodesicTracer(const Kerr& metric, const RK4& integrator,
                   double observer_r, int max_steps = 10000, double r_escape = 1000.0,
                   double tolerance = 1e-8,
                   const VolumetricDisk* vol_disk = nullptr,
                   double raymarch_tol = 1e-2);
```

In the private section (around line 64), add the member:

```cpp
    double raymarch_tol_ = 1e-2;
```

- [ ] **Step 2: Update the constructor body in `src/geodesic_tracer.cpp`**

Find the constructor body (search for `GeodesicTracer::GeodesicTracer`). Update its initializer list to capture and clamp the new parameter. Append `raymarch_tol_(raymarch_tol > 0.0 ? raymarch_tol : 1e-12)` to the initializer list. The clamp matches the spec's defensive-second-layer requirement.

- [ ] **Step 3: Pass through in `src/api.cpp`**

Find where `GeodesicTracer` is instantiated in `src/api.cpp` (search `make_unique<grrt::GeodesicTracer>` or `new grrt::GeodesicTracer`). Update the call to pass `params->raymarch_tol`:

```cpp
ctx->tracer = std::make_unique<grrt::GeodesicTracer>(
    *ctx->metric, *ctx->integrator,
    params->observer_r, max_steps, r_escape, integrator_tolerance,
    ctx->vol_disk.get(),
    params->raymarch_tol);
```

(Match whatever existing parameter style is used; this is the same constructor with one new trailing argument.)

- [ ] **Step 4: Build**

```bash
cmake --build build --config Release
```

Expected: Build succeeds. No behavioral change yet — we wired the value through but haven't used it.

- [ ] **Step 5: Commit**

```bash
git add include/grrt/geodesic/geodesic_tracer.h src/geodesic_tracer.cpp src/api.cpp
git commit -m "feat(tracer): plumb raymarch_tol through constructor"
```

---

## Task 8: Switch `raymarch_volumetric` to use the helper; revert `H/8` hack

**Files:**
- Modify: `src/geodesic_tracer.cpp`

This is the largest task. We replace the existing step-control machinery with a Romberg-driven control loop.

- [ ] **Step 1: Add a `VolumetricDiskSampler` adaptor at the top of `src/geodesic_tracer.cpp`**

Just below the includes, add (inside `namespace grrt {` if there's one, otherwise as an anonymous-namespace utility):

```cpp
#include "grrt/geodesic/romberg_step.h"

namespace {

// Adaptor: bridges StepSampler interface to the production VolumetricDisk + opacity LUTs.
struct VolumetricDiskSampler : grrt::StepSampler {
    const grrt::VolumetricDisk* disk;
    double observer_r;
    double ut_obs;  // observer's 4-velocity time component (cached)

    VolumetricDiskSampler(const grrt::VolumetricDisk* d, double obs_r)
        : disk(d), observer_r(obs_r),
          ut_obs(1.0 / std::sqrt(1.0 - 2.0 / obs_r)) {}

    bool sample_integrand(const grrt::GeodesicState& state,
                          std::span<const double> channels_nu_obs,
                          std::span<double> integrand) const override {
        const double r     = state.position[1];
        const double theta = state.position[2];
        const double phi   = state.position[3];
        const double z     = r * std::cos(theta);

        // Volume / r-extent guards
        if (r <= disk->r_horizon() || r > disk->r_max()) {
            for (size_t i = 0; i < channels_nu_obs.size(); ++i) integrand[i] = 0.0;
            return false;
        }

        const double rho_cgs = disk->density_cgs(r, z, phi);
        const double T_local = disk->temperature(r, std::abs(z));
        if (rho_cgs <= 0.0 || T_local <= 0.0) {
            for (size_t i = 0; i < channels_nu_obs.size(); ++i) integrand[i] = 0.0;
            return false;
        }

        // Redshift factor g = (p·u)_emit / (p·u)_obs.
        double ut_emit = 0.0, ur_emit = 0.0, uphi_emit = 0.0;
        if (r >= disk->r_isco()) {
            disk->circular_velocity(r, ut_emit, uphi_emit);
        } else {
            disk->plunging_velocity(r, theta, ut_emit, ur_emit, uphi_emit);
        }
        const double p_dot_u_emit = state.momentum[0] * ut_emit
                                   + state.momentum[1] * ur_emit
                                   + state.momentum[3] * uphi_emit;
        const double p_dot_u_obs  = state.momentum[0] * ut_obs;
        const double g_factor     = p_dot_u_emit / p_dot_u_obs;

        const auto& luts = disk->opacity_luts();
        const double abs_pue = std::abs(p_dot_u_emit);

        for (size_t i = 0; i < channels_nu_obs.size(); ++i) {
            const double nu_emit = std::abs(g_factor) * channels_nu_obs[i];
            const double kabs    = luts.lookup_kappa_abs(nu_emit, rho_cgs, T_local);
            const double kes     = luts.lookup_kappa_es(rho_cgs, T_local);
            integrand[i] = (kabs + kes) * rho_cgs * abs_pue;
        }
        return true;
    }
};

} // anonymous namespace
```

- [ ] **Step 2: Rewrite `raymarch_volumetric()` body**

Replace the entire body of `void GeodesicTracer::raymarch_volumetric(...)` with:

```cpp
void GeodesicTracer::raymarch_volumetric(GeodesicState& state, Vec3& /*color*/,
                                          double J_rgb[3], double T_rgb[3]) const {
    using namespace constants;
    const auto& luts = vol_disk_->opacity_luts();

    constexpr std::array<double, 3> nu_obs = {
        c_cgs / 450e-7, c_cgs / 550e-7, c_cgs / 650e-7
    };
    std::span<const double> ch_span{nu_obs.data(), 3};

    double J[3] = {J_rgb[0], J_rgb[1], J_rgb[2]};
    double T[3] = {T_rgb[0], T_rgb[1], T_rgb[2]};

    VolumetricDiskSampler sampler(vol_disk_, observer_r_);
    const double ut_obs = sampler.ut_obs;

    // Initial step proposal — same heuristics as before.
    double r = state.position[1];
    const double z_start = r * std::cos(state.position[2]);
    const double H_start = vol_disk_->scale_height(r);
    double ds_proposed = vol_disk_->inside_volume(r, z_start)
                       ? H_start / 16.0
                       : std::min(std::abs(z_start) / 8.0, H_start * 2.0);

    int step_count = 0;
    constexpr int MAX_STEPS = 4096;
    bool been_inside = vol_disk_->inside_volume(r, z_start);

    while (step_count < MAX_STEPS) {
        // Hard exits — match prior logic.
        if (r < vol_disk_->r_horizon())                        break;
        if (r > vol_disk_->r_max())                            break;
        if (T[0] < 1e-6 && T[1] < 1e-6 && T[2] < 1e-6)         break;

        // Romberg-controlled step.
        RombergStep rs = romberg_step(state, ds_proposed, ch_span,
                                       sampler, metric_, integrator_);

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
                const double nu_emit = std::abs(g_factor) * nu_obs[ch];
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
        been_inside = been_inside || vol_disk_->inside_volume(r, r * std::cos(state.position[2]));

        // Step-size growth: well under tolerance → grow, capped at 1·H.
        if (rs.max_err < raymarch_tol_ / 8.0) {
            const double H_now = vol_disk_->scale_height(r);
            ds_proposed = std::min(ds_proposed * 2.0, H_now);
        }
    }

    // Persist for caller.
    J_rgb[0] = J[0]; J_rgb[1] = J[1]; J_rgb[2] = J[2];
    T_rgb[0] = T[0]; T_rgb[1] = T[1]; T_rgb[2] = T[2];
}
```

- [ ] **Step 3: Delete now-unused machinery**

Inside the file, delete these (now obsolete):
- The line `constexpr double DTAU_TARGET = 0.05;` (was at the top of the old function body)
- The H/8 debug clamp `ds = std::clamp(ds, H / 64.0, H / 8.0);` — this whole block is replaced by Romberg control above.
- Any stray `H_start`, `inside_volume(r, z_start)` initialisations from the old loop that aren't referenced by the new body.

(In practice these are inside the function body that's already replaced wholesale; this step is a sanity check to confirm nothing leaked.)

- [ ] **Step 4: Build**

```bash
cmake --build build --config Release
```

Expected: Build succeeds.

- [ ] **Step 5: Smoke render — confirm bands are gone**

```bash
build/Release/grrt-cli.exe --disk-volumetric --samples 30 --width 256 --height 256 --output bug_fixed --force
```

Expected: A 256×256 image with a smooth disk (no horizontal banding). Compare visually to `bug_repro.png` from the brainstorm session.

- [ ] **Step 6: Run the existing volumetric test suite**

```bash
cmake --build build --config Release --target test-volumetric
build/Release/test-volumetric.exe
```

Expected: Same pass/fail status as the existing branch. Should not regress; the tau-test factor-of-4 known issue remains.

- [ ] **Step 7: Commit**

```bash
git add src/geodesic_tracer.cpp
git commit -m "feat(tracer): switch raymarch_volumetric to Romberg-controlled stepping"
```

---

## Task 9: Tolerance-convergence integration test

**Files:**
- Modify: `tests/test_volumetric.cpp`

- [ ] **Step 1: Add a tolerance-convergence test**

In `tests/test_volumetric.cpp`, find the place where the existing tests are listed (near the bottom, in `main()` or similar). Add this test:

```cpp
// Render a small image at three tolerances and verify default vs tight converges.
static void test_tolerance_convergence() {
    auto render_at_tol = [](double tol) -> std::vector<float> {
        grrt::GRRTParams params{};
        // Minimal scene matching the user's bug repro
        params.width = 64;
        params.height = 64;
        params.metric_type = GRRT_METRIC_KERR;
        params.mass = 1.0;
        params.spin = 0.998;
        params.observer_r = 50.0;
        params.observer_theta = 80.0;
        params.fov = 90.0;
        params.disk_enabled = 1;
        params.disk_volumetric = 1;
        params.disk_temperature = 1e7;
        params.disk_outer = 20.0;
        params.disk_alpha = 0.1;
        params.disk_turbulence = 0.0;  // disable noise for reproducibility
        params.disk_seed = 42;
        params.disk_force = 1;
        params.background_type = GRRT_BG_BLACK;
        params.integrator_max_steps = 10000;
        params.integrator_tolerance = 1e-8;
        params.samples_per_pixel = 4;
        params.thread_count = 0;
        params.backend = GRRT_BACKEND_CPU;
        params.raymarch_tol = tol;

        GRRTContext* ctx = grrt_create(&params);
        if (!ctx) {
            std::printf("FAIL: grrt_create returned null at tol=%g\n", tol);
            std::exit(1);
        }
        std::vector<float> fb(params.width * params.height * 4, 0.0f);
        grrt_render(ctx, fb.data());
        grrt_destroy(ctx);
        return fb;
    };

    auto ref   = render_at_tol(1e-3);
    auto def   = render_at_tol(1e-2);
    auto loose = render_at_tol(1e-1);

    auto max_diff = [&](const std::vector<float>& a, const std::vector<float>& b) {
        float m = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            // Compare RGB only, skip alpha (every 4th).
            if (i % 4 == 3) continue;
            m = std::max(m, std::abs(a[i] - b[i]));
        }
        return m;
    };

    const float diff_def   = max_diff(def,   ref);
    const float diff_loose = max_diff(loose, ref);

    std::printf("test_tolerance_convergence: max|def-ref|=%.4e, max|loose-ref|=%.4e\n",
                diff_def, diff_loose);

    // Default (1e-2) must be within 1% of reference (1e-3) on linear-light.
    if (diff_def > 0.01f) {
        std::printf("FAIL: default tol does not converge to reference (diff=%.4e)\n", diff_def);
        std::exit(1);
    }
    // Loose (1e-1) is allowed up to 5%.
    if (diff_loose > 0.05f) {
        std::printf("FAIL: loose tol diverges too far from reference (diff=%.4e)\n", diff_loose);
        std::exit(1);
    }
}
```

Wire it into the existing `main()` test list.

- [ ] **Step 2: Build and run**

```bash
cmake --build build --config Release --target test-volumetric
build/Release/test-volumetric.exe
```

Expected: Test prints both diffs and reports PASS. (Render is small: 64² × 4 spp, three tolerances → completes in ~1–2 minutes.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_volumetric.cpp
git commit -m "test(volumetric): tolerance-convergence test for raymarch_tol"
```

---

## Task 10: Banding-regression test

**Files:**
- Modify: `tests/test_volumetric.cpp`

- [ ] **Step 1: Add the regression test**

Add this function to `tests/test_volumetric.cpp`:

```cpp
// Reproduces the user's --samples 30 bug scenario and asserts that horizontal
// scanlines through the disk have low intensity variance (no bands).
//
// Calibration (recorded 2026-05-01, branch fix/volumetric-ring):
//   - Buggy build (H_max=H):  σ/mean ≈ 0.7-0.9 on disk-crossing rows
//   - H/8 debug build:        σ/mean ≈ 0.15
//   - Romberg (this work):    σ/mean expected ≪ 0.30
// Threshold of 0.30 fails any future regression to the buggy regime.
static void test_no_horizontal_bands() {
    grrt::GRRTParams params{};
    params.width = 256;
    params.height = 256;
    params.metric_type = GRRT_METRIC_KERR;
    params.mass = 1.0;
    params.spin = 0.998;
    params.observer_r = 50.0;
    params.observer_theta = 80.0;
    params.fov = 90.0;
    params.disk_enabled = 1;
    params.disk_volumetric = 1;
    params.disk_temperature = 1e7;
    params.disk_outer = 20.0;
    params.disk_alpha = 0.1;
    params.disk_turbulence = 0.0;
    params.disk_seed = 42;
    params.disk_force = 1;
    params.background_type = GRRT_BG_BLACK;
    params.integrator_max_steps = 10000;
    params.integrator_tolerance = 1e-8;
    params.samples_per_pixel = 30;  // user's bug repro setting
    params.thread_count = 0;
    params.backend = GRRT_BACKEND_CPU;
    params.raymarch_tol = 1e-2;

    GRRTContext* ctx = grrt_create(&params);
    if (!ctx) { std::printf("FAIL: grrt_create returned null\n"); std::exit(1); }
    std::vector<float> fb(params.width * params.height * 4, 0.0f);
    grrt_render(ctx, fb.data());
    grrt_destroy(ctx);

    // For each row in image-y centerline ± 5, compute σ/mean of luminance.
    auto luminance = [&](int x, int y) -> float {
        size_t i = static_cast<size_t>((y * params.width + x) * 4);
        // Rec. 709 luminance approximation
        return 0.2126f * fb[i] + 0.7152f * fb[i+1] + 0.0722f * fb[i+2];
    };

    int rows_checked = 0;
    int rows_failed  = 0;
    const int center_y = params.height / 2;
    for (int y = center_y - 5; y <= center_y + 5; ++y) {
        double sum = 0.0, sum_sq = 0.0;
        int n = 0;
        for (int x = 0; x < params.width; ++x) {
            float L = luminance(x, y);
            // Only count pixels with non-trivial intensity (skip background).
            if (L > 1e-6f) {
                sum += L; sum_sq += L * L; n++;
            }
        }
        if (n < 20) continue;  // not enough disk on this row to test
        const double mean = sum / n;
        const double var  = std::max(0.0, sum_sq / n - mean * mean);
        const double sigma = std::sqrt(var);
        const double rel = (mean > 0.0) ? sigma / mean : 0.0;

        rows_checked++;
        if (rel > 0.30) {
            std::printf("test_no_horizontal_bands: row y=%d σ/mean=%.3f (FAIL)\n", y, rel);
            rows_failed++;
        }
    }

    std::printf("test_no_horizontal_bands: %d/%d rows passed\n",
                rows_checked - rows_failed, rows_checked);

    if (rows_checked == 0) {
        std::printf("FAIL: no rows had enough disk pixels — scene setup broken\n");
        std::exit(1);
    }
    if (rows_failed > 0) {
        std::printf("FAIL: %d rows exceeded σ/mean threshold of 0.30\n", rows_failed);
        std::exit(1);
    }
}
```

Wire it into `main()`.

- [ ] **Step 2: Build and run**

```bash
cmake --build build --config Release --target test-volumetric
build/Release/test-volumetric.exe
```

Expected: All rows pass; σ/mean printed below 0.30 for centerline disk rows. (256×256 × 30 spp render takes ~30 s on CPU.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_volumetric.cpp
git commit -m "test(volumetric): regression guard for raymarch banding artifact"
```

---

## Self-review

After all 10 tasks above are complete:

- [ ] **Step A: Confirm spec coverage**

Open `docs/superpowers/specs/2026-05-01-cliff-aware-raymarch-design.md` and walk down each section:
- "The `romberg_step` helper" → covered by Tasks 1–5.
- "Caller — `raymarch_volumetric`" → covered by Task 8.
- "Tolerance — what it bounds" → covered by tests in Tasks 9–10 (tolerance-convergence + banding regression).
- "CLI flag" → covered by Task 6.
- "Validation & testing 6a/6b/6c" → covered by Tasks 4–5/9/10.
- "Error handling & edge cases (negative tol, empty channels, ds floor)" → tol clamp in Task 7; empty-channel handling in Task 3 (`if n_channels <= 0` early return); ds floor in Task 8.

- [ ] **Step B: Final smoke render**

```bash
build/Release/grrt-cli.exe --disk-volumetric --samples 30 --width 1024 --height 1024 --output final --force
```

Visually compare `final.png` to `bug_repro.png` in the working tree. The new render should be free of horizontal bands.

- [ ] **Step C: Use the `superpowers:finishing-a-development-branch` skill** to wrap up the branch.
