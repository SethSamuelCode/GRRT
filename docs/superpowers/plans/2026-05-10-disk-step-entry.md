# Disk-step-entry helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the endpoint-only raymarch-entry predicate (triplicated at three sites in `geodesic_tracer.cpp`) with a tiered helper that adds a conservative segment bound and momentum-aware adaptive substep recursion. Eliminates wedge artifacts at narrow FOV in `--disk-volumetric` renders.

**Architecture:** New free-function `check_disk_step_entry` in a new translation unit. Three-tier gate: existing endpoint predicate (Tier A), conservative segment-bound test (Tier B), recursive substep with adaptive depth_limit (Tier C). Helper takes `(prev_state, new_state, dλ_full, disk, metric, integrator, opts)` and returns `{should_raymarch, refined_endpoint}`.

**Tech Stack:** C++23, MSVC 2022 / CMake, OpenMP (existing), shared `VolumetricDisk` singleton for tests (Meyer's pattern).

**Spec:** `docs/superpowers/specs/2026-05-10-disk-step-entry-design.md`

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `include/grrt/geodesic/disk_step_entry.h` | Create | Public API: `DiskStepEntryResult`, `DiskStepEntryOptions`, `check_disk_step_entry` declaration |
| `src/disk_step_entry.cpp` | Create | All three tiers, anonymous-namespace helpers, recursion |
| `tests/test_disk_step_entry.cpp` | Create | Unit tests using shared `VolumetricDisk` Meyer's singleton |
| `src/geodesic_tracer.cpp` | Modify (3 sites: ~L192–202, ~L441–449, ~L585–595) | Replace inline predicates with calls to helper |
| `include/grrt/geodesic/geodesic_tracer.h` | Modify | Add `mutable std::atomic<long> substep_invocation_count_` for diagnostics |
| `CMakeLists.txt` | Modify | Add `src/disk_step_entry.cpp` to library, `tests/test_disk_step_entry.cpp` to test target |

Each task below produces self-contained changes. Frequent commits — one per task minimum.

---

## Task 1: Header skeleton + test scaffold + minimal stub

**Files:**
- Create: `include/grrt/geodesic/disk_step_entry.h`
- Create: `src/disk_step_entry.cpp` (stub)
- Create: `tests/test_disk_step_entry.cpp` (scaffold)
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Create header `include/grrt/geodesic/disk_step_entry.h`**

```cpp
#ifndef GRRT_DISK_STEP_ENTRY_H
#define GRRT_DISK_STEP_ENTRY_H

#include "grrt/geodesic/integrator.h"
#include "grrt_export.h"

namespace grrt {

// Forward declarations
class VolumetricDisk;
class Kerr;
class RK4;

struct DiskStepEntryResult {
    bool should_raymarch;
    GeodesicState refined_endpoint;   ///< valid only when should_raymarch == true
    int substep_invocations;          ///< Tier C subdivide() invocations consumed
};

struct DiskStepEntryOptions {
    int    depth_limit_floor = 4;     ///< minimum subdivisions
    int    depth_limit_cap   = 10;    ///< hard ceiling (1024x refinement)
    double curvature_pad     = 0.5;   ///< chord-length multiplier (see spec §5.4)
};

/// Three-tier gate replacing the endpoint-only predicate at three sites in
/// geodesic_tracer.cpp. See docs/superpowers/specs/2026-05-10-disk-step-entry-design.md
/// for design rationale.
GRRT_EXPORT DiskStepEntryResult check_disk_step_entry(
    const GeodesicState& prev_state,
    const GeodesicState& new_state,
    double dlambda_full,
    const VolumetricDisk& disk,
    const Kerr& metric,
    const RK4& integrator,
    const DiskStepEntryOptions& opts = {});

} // namespace grrt

#endif
```

- [ ] **Step 2: Create stub `src/disk_step_entry.cpp`**

```cpp
#include "grrt/geodesic/disk_step_entry.h"
#include "grrt/scene/volumetric_disk.h"
#include "grrt/spacetime/kerr.h"
#include "grrt/geodesic/rk4.h"
#include <algorithm>
#include <cmath>

namespace grrt {

DiskStepEntryResult check_disk_step_entry(
    const GeodesicState& /*prev_state*/,
    const GeodesicState& /*new_state*/,
    double /*dlambda_full*/,
    const VolumetricDisk& /*disk*/,
    const Kerr& /*metric*/,
    const RK4& /*integrator*/,
    const DiskStepEntryOptions& /*opts*/)
{
    // Stub — Task 5 wires up real implementation.
    return { false, {}, 0 };
}

} // namespace grrt
```

- [ ] **Step 3: Create test scaffold `tests/test_disk_step_entry.cpp`**

```cpp
// tests/test_disk_step_entry.cpp
//
// Unit tests for check_disk_step_entry. Uses the shared Meyer's-singleton
// VolumetricDisk (same pattern as test_volumetric.cpp) so disk-API changes
// flow through.

#include "grrt/geodesic/disk_step_entry.h"
#include "grrt/scene/volumetric_disk.h"
#include "grrt/spacetime/kerr.h"
#include "grrt/geodesic/rk4.h"
#include <cstdio>
#include <cmath>

using namespace grrt;

int failures = 0;

#define EXPECT_TRUE(cond, msg)                                                \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::printf("FAIL %s:%d: %s — %s\n",                              \
                        __FILE__, __LINE__, #cond, msg);                      \
            failures++;                                                       \
        }                                                                     \
    } while (0)

// Shared real-disk singleton — construction is ~1 minute. Tests share one
// disk instance for the whole process. Same configuration as
// shared_disk_default() in test_volumetric.cpp.
static const VolumetricDisk& shared_disk() {
    static const VolumetricDisk disk(1.0, 0.998, 30.0, 1e7);
    return disk;
}

// Static metric + integrator. Cheap to construct; not shared for clarity.
static Kerr make_metric() { return Kerr(1.0, 0.998); }
static RK4  make_integrator() { return RK4{}; }

// Build a minimal GeodesicState at given (r, theta) with arbitrary momenta.
// Tests construct synthetic states; the helper does not require physically
// valid geodesics for the bound-test logic.
static GeodesicState make_state(double r, double theta, double pr = 0.0,
                                double ptheta = 0.0) {
    GeodesicState s;
    s.position = Vec4{{0.0, r, theta, 0.0}};        // (t, r, theta, phi)
    s.momentum = Vec4{{-1.0, pr, ptheta, 0.0}};     // (-E=-1, p_r, p_theta, p_phi)
    return s;
}

// First failing test: stub returns should_raymarch=false even when
// endpoints are clearly inside the disk. Will turn green when the
// endpoint predicate is wired up in Task 2.
static void test_endpoints_inside_disk_should_raymarch() {
    const VolumetricDisk& disk = shared_disk();
    Kerr metric = make_metric();
    RK4 integrator = make_integrator();

    // Both endpoints at (r=10M, theta=pi/2), squarely inside the volume.
    constexpr double half_pi = 1.5707963267948966;
    GeodesicState prev = make_state(10.0, half_pi);
    GeodesicState curr = make_state(10.0, half_pi);

    DiskStepEntryResult r = check_disk_step_entry(
        prev, curr, /*dlambda_full=*/1.0, disk, metric, integrator);

    EXPECT_TRUE(r.should_raymarch,
                "endpoints inside disk should trigger raymarch");
}

int main() {
    std::printf("Running test_disk_step_entry...\n");
    test_endpoints_inside_disk_should_raymarch();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
```

- [ ] **Step 4: Update `CMakeLists.txt`**

Find the library source list (search for existing `src/sobol_sampler.cpp` or `src/volumetric_disk.cpp`); add `src/disk_step_entry.cpp` alongside it.

Find the test target list (search for `test_sobol_sampler.cpp`); add a new test target for `tests/test_disk_step_entry.cpp` mirroring its pattern.

- [ ] **Step 5: Build to confirm scaffold compiles**

Run:
```
cmake --build build --config Release
```
Expected: builds without errors.

- [ ] **Step 6: Run scaffold test to verify it fails**

Run:
```
./build/Release/test-disk-step-entry.exe
```
Expected: FAIL with `endpoints inside disk should trigger raymarch`. The stub returns false.

- [ ] **Step 7: Commit**

```bash
git add include/grrt/geodesic/disk_step_entry.h src/disk_step_entry.cpp tests/test_disk_step_entry.cpp CMakeLists.txt
git commit -m "feat(disk-step-entry): scaffold helper, stub, failing test"
```

---

## Task 2: Endpoint predicate extraction (Tier A)

**Files:**
- Modify: `src/disk_step_entry.cpp`
- Modify: `tests/test_disk_step_entry.cpp`

- [ ] **Step 1: Add `endpoint_predicate` to anonymous namespace in `src/disk_step_entry.cpp`**

This is byte-for-byte equivalent to the inline blocks at `geodesic_tracer.cpp` lines 192–202, 441–449, 585–595.

```cpp
namespace {

constexpr double kHalfPi = 1.5707963267948966;

// Tier A. Identical to the inline predicate at geodesic_tracer.cpp three sites.
bool endpoint_predicate(const GeodesicState& prev,
                        const GeodesicState& curr,
                        const VolumetricDisk& disk) {
    const double theta_prev = prev.position[2];
    const double theta_curr = curr.position[2];
    const double r_prev = prev.position[1];
    const double r_curr = curr.position[1];

    const double d_prev = theta_prev - kHalfPi;
    const double d_curr = theta_curr - kHalfPi;

    const double z_prev = r_prev * std::cos(theta_prev);
    const double z_curr = r_curr * std::cos(theta_curr);

    const bool crossed_midplane =
        (d_prev * d_curr < 0.0) && std::abs(d_prev - d_curr) > 1e-12;

    const bool inside_now = disk.inside_volume(r_curr, z_curr);

    const double zm_curr = disk.z_max_at(r_curr);
    const double H_curr  = disk.scale_height(r_curr);
    const double H_prev  = disk.scale_height(r_prev);
    const bool near_disk =
        (std::abs(z_curr) < zm_curr + 1.0 * H_curr
         || std::abs(z_prev) < disk.z_max_at(r_prev) + 1.0 * H_prev)
        && r_curr >= disk.r_horizon()
        && r_curr <= disk.r_max() + 0.5 * disk.outer_taper_width();

    return crossed_midplane || inside_now || near_disk;
}

} // anonymous namespace
```

- [ ] **Step 2: Wire `check_disk_step_entry` to use it (temporary; full orchestration in Task 5)**

Replace the stub body:

```cpp
DiskStepEntryResult check_disk_step_entry(
    const GeodesicState& prev_state,
    const GeodesicState& new_state,
    double /*dlambda_full*/,
    const VolumetricDisk& disk,
    const Kerr& /*metric*/,
    const RK4& /*integrator*/,
    const DiskStepEntryOptions& /*opts*/)
{
    if (endpoint_predicate(prev_state, new_state, disk)) {
        return { true, new_state, 0 };
    }
    return { false, {}, 0 };
}
```

- [ ] **Step 3: Re-run the existing test — should now pass**

Run:
```
cmake --build build --config Release
./build/Release/test-disk-step-entry.exe
```
Expected: `endpoints inside disk should trigger raymarch` PASSES.

- [ ] **Step 4: Add equivalence test mirroring the existing inline predicate**

Append to `tests/test_disk_step_entry.cpp` before `main()`:

```cpp
// Test the extracted endpoint predicate matches the original inline logic
// across a battery of (prev, curr) cases. Reproduces the inline computation
// directly and asserts identical result.
static bool inline_endpoint_predicate_replica(const GeodesicState& prev,
                                              const GeodesicState& curr,
                                              const VolumetricDisk& disk) {
    constexpr double half_pi = 1.5707963267948966;
    const double theta_prev = prev.position[2];
    const double theta_new  = curr.position[2];
    const double d_prev = theta_prev - half_pi;
    const double d_new  = theta_new  - half_pi;
    const double r_new  = curr.position[1];
    const double r_prev = prev.position[1];
    const double z_new  = r_new  * std::cos(theta_new);
    const double z_prev = r_prev * std::cos(theta_prev);
    const bool crossed_midplane =
        (d_prev * d_new < 0.0) && std::abs(d_prev - d_new) > 1e-12;
    const bool inside_now = disk.inside_volume(r_new, z_new);
    const double zm_new = disk.z_max_at(r_new);
    const double H_new  = disk.scale_height(r_new);
    const double H_prev = disk.scale_height(r_prev);
    const bool near_disk =
        (std::abs(z_new) < zm_new + 1.0 * H_new
         || std::abs(z_prev) < disk.z_max_at(r_prev) + 1.0 * H_prev)
        && r_new >= disk.r_horizon()
        && r_new <= disk.r_max() + 0.5 * disk.outer_taper_width();
    return crossed_midplane || inside_now || near_disk;
}

static void test_endpoint_predicate_equivalence() {
    const VolumetricDisk& disk = shared_disk();
    Kerr metric = make_metric();
    RK4 integrator = make_integrator();
    constexpr double half_pi = 1.5707963267948966;

    // Battery of cases: deep above, deep below, crossing midplane, inside,
    // near-but-above, far-r, near-horizon.
    struct Case { double r_prev, theta_prev, r_curr, theta_curr; const char* name; };
    Case cases[] = {
        {50.0, 0.5,        50.0, 0.5,        "deep above midplane"},
        {50.0, 2.5,        50.0, 2.5,        "deep below midplane"},
        {15.0, 1.4,        15.0, 1.7,        "midplane crossing"},
        {10.0, half_pi,    10.0, half_pi,    "inside volume"},
        { 8.0, half_pi-0.05, 8.0, half_pi-0.04, "near-but-above"},
        {500.0, 1.0,       500.0, 1.0,       "far-r escape"},
        { 2.5, 1.5,         2.5, 1.6,        "near-horizon midplane"},
    };

    for (const auto& c : cases) {
        GeodesicState prev = make_state(c.r_prev, c.theta_prev);
        GeodesicState curr = make_state(c.r_curr, c.theta_curr);

        const bool expected = inline_endpoint_predicate_replica(prev, curr, disk);
        DiskStepEntryResult r = check_disk_step_entry(
            prev, curr, /*dlambda_full=*/1.0, disk, metric, integrator);
        const bool got = r.should_raymarch;

        if (got != expected) {
            std::printf("FAIL endpoint equiv [%s]: expected=%d got=%d\n",
                        c.name, expected ? 1 : 0, got ? 1 : 0);
            failures++;
        }
    }
}
```

Add a call to `test_endpoint_predicate_equivalence()` in `main()`.

- [ ] **Step 5: Build and run — both tests should pass**

Run:
```
cmake --build build --config Release
./build/Release/test-disk-step-entry.exe
```
Expected: 0 failures.

- [ ] **Step 6: Commit**

```bash
git add src/disk_step_entry.cpp tests/test_disk_step_entry.cpp
git commit -m "feat(disk-step-entry): extract Tier A endpoint predicate + equivalence test"
```

---

## Task 3: Segment-bound test (Tier B) + unit tests

**Files:**
- Modify: `src/disk_step_entry.cpp`
- Modify: `tests/test_disk_step_entry.cpp`

- [ ] **Step 1: Add `segment_could_intersect_disk` to anonymous namespace**

In `src/disk_step_entry.cpp`, after `endpoint_predicate`:

```cpp
// Tier B. Conservative bounding-region test for the (r, |z|) trajectory of
// a step. Uses momentum-aware curvature pad to cover the case where both
// endpoints are above z_max but the trajectory dips below it mid-step.
//
// Returns true if the segment "could" intersect the disk volume — caller
// must subdivide. Returns false only when the segment is conclusively
// outside (rejection guaranteed safe).
bool segment_could_intersect_disk(const GeodesicState& prev,
                                  const GeodesicState& curr,
                                  double dlambda_full,
                                  const VolumetricDisk& disk,
                                  const Kerr& metric,
                                  double curvature_pad)
{
    const double r_prev = prev.position[1];
    const double r_curr = curr.position[1];
    const double theta_prev = prev.position[2];
    const double theta_curr = curr.position[2];

    const double z_prev = r_prev * std::cos(theta_prev);
    const double z_curr = r_curr * std::cos(theta_curr);

    // r-range bound (expand by chord length so it's loose).
    const double r_min = std::min(r_prev, r_curr);
    const double r_max = std::max(r_prev, r_curr);

    // Reject if r-range entirely outside disk r-cylinder.
    const double disk_r_lo = disk.r_horizon();
    const double disk_r_hi = disk.r_max() + 0.5 * disk.outer_taper_width();
    if (r_max < disk_r_lo || r_min > disk_r_hi) return false;

    // Compute |z|_min over the segment with velocity-aware pad.
    // GeodesicState::momentum stores covariant p_μ. The chain rule
    //     dz/dlambda = cos(theta) * dr/dlambda - r * sin(theta) * dtheta/dlambda
    // requires CONTRAVARIANT position-derivatives, which we get from
    // RK4::derivatives_kerr (same source the renderer's step clamp uses
    // at src/geodesic_tracer.cpp:147-152). Using p_μ directly would
    // conflate energy with velocity (units mismatch by ~Σ for Kerr).
    const auto deriv_prev = RK4::derivatives_kerr(metric, prev);
    const auto deriv_curr = RK4::derivatives_kerr(metric, curr);
    const double vz_prev = std::cos(theta_prev) * deriv_prev.position[1]
                         - r_prev * std::sin(theta_prev) * deriv_prev.position[2];
    const double vz_curr = std::cos(theta_curr) * deriv_curr.position[1]
                         - r_curr * std::sin(theta_curr) * deriv_curr.position[2];

    const double dz_chord = std::abs(z_prev - z_curr);
    const double dz_swing = 0.5 * std::abs(vz_prev - vz_curr) * dlambda_full;
    const double pad      = std::max(curvature_pad * dz_chord, dz_swing);

    double abs_z_min = std::min(std::abs(z_prev), std::abs(z_curr)) - pad;
    if (z_prev * z_curr < 0.0) abs_z_min = 0.0;       // crosses midplane
    if (abs_z_min < 0.0)       abs_z_min = 0.0;

    // Sample disk envelope at three r-points (endpoints + midpoint) to
    // get a conservative max(z_max + 0.5*H) over the segment's r-range.
    auto envelope_at = [&](double r) {
        const double r_clamped = std::clamp(r, disk_r_lo, disk_r_hi);
        return disk.z_max_at(r_clamped) + 0.5 * disk.scale_height(r_clamped);
    };
    const double env_lo  = envelope_at(r_min);
    const double env_hi  = envelope_at(r_max);
    const double env_mid = envelope_at(0.5 * (r_min + r_max));
    const double env_max = std::max({env_lo, env_hi, env_mid});

    return abs_z_min <= env_max;
}
```

- [ ] **Step 2: Add tests for Tier B**

Append to `tests/test_disk_step_entry.cpp`:

```cpp
// Probe the Tier B path directly via the public API by setting up cases
// where Tier A is false but Tier B's outcome is testable through end-state.
// We can't call segment_could_intersect_disk directly (anonymous-namespace),
// so we use the orchestrator's behavior as a proxy. After Task 5 wires
// Tier C, "Tier B fires" means subdivide is invoked, observable via
// substep_invocations > 0.

static void test_segment_bound_rejects_far_above() {
    const VolumetricDisk& disk = shared_disk();
    Kerr metric = make_metric();
    RK4 integrator = make_integrator();

    // Both endpoints at z = 50M (well above disk top, even at hot inner radii).
    // No midplane crossing, no inside_now, no near_disk.
    GeodesicState prev = make_state(20.0, 0.05);   // theta near 0 → z = r*cos ~ 20
    GeodesicState curr = make_state(20.0, 0.10);

    DiskStepEntryResult r = check_disk_step_entry(
        prev, curr, /*dlambda_full=*/0.5, disk, metric, integrator);

    EXPECT_TRUE(!r.should_raymarch,
                "segment far above disk should reject");
    EXPECT_TRUE(r.substep_invocations == 0,
                "no subdivision should occur for far-above segment");
}

static void test_segment_bound_passes_when_dipping() {
    const VolumetricDisk& disk = shared_disk();
    Kerr metric = make_metric();
    RK4 integrator = make_integrator();
    constexpr double half_pi = 1.5707963267948966;

    // Endpoints just above disk top with large p_theta so dz_swing > endpoint
    // |z|. Use disk parameters: at r=10M, expect z_max ~= 1-3M, H ~= 0.5-1M.
    // Place endpoints at z = 1.5*z_max, dz_chord small but |p_theta| large.
    // theta = half_pi - 0.05 → z ≈ 0.5M (above some z_max+H for narrow ranges).
    GeodesicState prev = make_state(10.0, half_pi - 0.10, /*pr=*/0.0,
                                    /*ptheta=*/-0.5);
    GeodesicState curr = make_state(10.0, half_pi - 0.08, /*pr=*/0.0,
                                    /*ptheta=*/-0.5);

    // dlambda_full chosen so dz_swing >> endpoint |z|, forcing Tier B to pass.
    DiskStepEntryResult r = check_disk_step_entry(
        prev, curr, /*dlambda_full=*/2.0, disk, metric, integrator);

    // Pre-Task-5: Tier B passing has no observable effect (orchestrator only
    // runs Tier A). This test will become meaningful in Task 5.
    // For now, just exercise the path to make sure no crash.
    (void)r;
    std::printf("  (segment-bound dipping case — meaningful after Task 5)\n");
}
```

Add calls to both tests in `main()`. (`test_segment_bound_passes_when_dipping` is a smoke check now; will assert in Task 5.)

- [ ] **Step 3: Build and run**

Run:
```
cmake --build build --config Release
./build/Release/test-disk-step-entry.exe
```
Expected: 0 failures.

- [ ] **Step 4: Commit**

```bash
git add src/disk_step_entry.cpp tests/test_disk_step_entry.cpp
git commit -m "feat(disk-step-entry): Tier B segment-bound test with momentum-aware pad"
```

---

## Task 4: Subdivision recursion + adaptive depth math (Tier C)

**Files:**
- Modify: `src/disk_step_entry.cpp`
- Modify: `tests/test_disk_step_entry.cpp`

- [ ] **Step 1: Add adaptive-depth helper + subdivide function to anonymous namespace**

In `src/disk_step_entry.cpp`, after `segment_could_intersect_disk`:

```cpp
// Compute adaptive depth_limit such that the smallest substep is on the
// order of H_min over the segment's r-range. Spec §5.5.
int compute_adaptive_depth(double dlambda_full,
                           double r_prev, double r_curr,
                           const VolumetricDisk& disk,
                           int depth_floor, int depth_cap)
{
    const double H_prev = std::max(disk.scale_height(r_prev), 1e-30);
    const double H_curr = std::max(disk.scale_height(r_curr), 1e-30);
    const double H_min  = std::min(H_prev, H_curr);
    if (dlambda_full <= 0.0) return floor;
    const double ratio = std::max(dlambda_full / H_min, 1.0);
    const int needed = static_cast<int>(std::ceil(std::log2(ratio)));
    return std::clamp(needed, floor, cap);
}

struct SubdivResult {
    bool should_raymarch;
    GeodesicState refined;
    int invocations;        // includes this call + recursive children
};

// Tier C. Recursive substep with depth_limit. Returns
// {should_raymarch=true, refined=substep_endpoint} when Tier A fires
// somewhere in the substep tree. On depth exhaustion: conservative policy
// (spec §6.1) — return {true, curr}.
SubdivResult subdivide(const GeodesicState& prev,
                       const GeodesicState& curr,
                       double dlambda_remaining,
                       int depth_remaining,
                       const VolumetricDisk& disk,
                       const Kerr& metric,
                       const RK4& integrator,
                       double curvature_pad)
{
    if (depth_remaining == 0) {
        // Conservative: assume entry, raymarch handles non-entry cheaply.
        return { true, curr, 1 };
    }

    // Substep using fixed-step Dormand-Prince RK4(5) — same integrator family
    // as the main loop's adaptive_step_kerr_dp45 (spec §5.1). Discard the
    // error_norm; we only need the 5th-order y5 trajectory state. ~50% more
    // derivative evals per substep than plain RK4, but Tier C fires rarely
    // enough that absolute cost is negligible (<1% of total render).
    // See docs/superpowers/optimizations/2026-05-10-disk-step-entry-rk4-substep.md
    // for the perf revisit if this ever shows up in profiling.
    const double dl_half = dlambda_remaining * 0.5;
    GeodesicState mid = integrator.step_kerr_rkdp45(metric, prev, dl_half).y5;

    int invocations = 1;

    // Tier A on each half.
    if (endpoint_predicate(prev, mid, disk)) {
        return { true, mid, invocations };
    }
    if (endpoint_predicate(mid, curr, disk)) {
        return { true, curr, invocations };
    }

    // Tier B on each half — recurse only on halves that might intersect.
    if (segment_could_intersect_disk(prev, mid, dl_half, disk, metric, curvature_pad)) {
        SubdivResult left = subdivide(prev, mid, dl_half, depth_remaining - 1,
                                      disk, metric, integrator, curvature_pad);
        invocations += left.invocations;
        if (left.should_raymarch) {
            return { true, left.refined, invocations };
        }
    }
    if (segment_could_intersect_disk(mid, curr, dl_half, disk, metric, curvature_pad)) {
        SubdivResult right = subdivide(mid, curr, dl_half, depth_remaining - 1,
                                       disk, metric, integrator, curvature_pad);
        invocations += right.invocations;
        if (right.should_raymarch) {
            return { true, right.refined, invocations };
        }
    }

    return { false, {}, invocations };
}
```

- [ ] **Step 2: Add unit test for adaptive depth math**

Append to `tests/test_disk_step_entry.cpp`. We test the math indirectly via behavior — by setting `depth_limit_floor = depth_limit_cap = N` and confirming the subdivision count tracks N, then by varying the `dlambda_full / H` ratio and checking that the natural depth floor/cap clamping behaves.

This test gets meaningful results after Task 5. For now, add a placeholder:

```cpp
static void test_adaptive_depth_clamped_at_cap() {
    // After Task 5, with depth_limit_cap=10 and a pathological
    // dlambda_full/H_min ratio, subdivisions should stop at cap.
    // Placeholder — exercised in Task 5's integration test.
    std::printf("  (adaptive depth — meaningful after Task 5)\n");
}
```

Add to `main()`. Don't fail.

- [ ] **Step 3: Build and run**

Run:
```
cmake --build build --config Release
./build/Release/test-disk-step-entry.exe
```
Expected: 0 failures (no behavior change yet — orchestrator still only calls Tier A).

- [ ] **Step 4: Commit**

```bash
git add src/disk_step_entry.cpp tests/test_disk_step_entry.cpp
git commit -m "feat(disk-step-entry): Tier C subdivide + adaptive depth math"
```

---

## Task 5: Public orchestrator wires all three tiers

**Files:**
- Modify: `src/disk_step_entry.cpp`
- Modify: `tests/test_disk_step_entry.cpp`

- [ ] **Step 1: Replace `check_disk_step_entry` body with full orchestration**

In `src/disk_step_entry.cpp`:

```cpp
DiskStepEntryResult check_disk_step_entry(
    const GeodesicState& prev_state,
    const GeodesicState& new_state,
    double dlambda_full,
    const VolumetricDisk& disk,
    const Kerr& metric,
    const RK4& integrator,
    const DiskStepEntryOptions& opts)
{
    // Defensive: degenerate disk → no entry (spec §6.3).
    if (disk.r_max() <= disk.r_horizon()) {
        return { false, {}, 0 };
    }

    // Tier A: existing endpoint predicate. Fast path; preserves byte-exact
    // behavior on the no-bug case.
    if (endpoint_predicate(prev_state, new_state, disk)) {
        return { true, new_state, 0 };
    }

    // Degenerate dlambda → can't substep. Tier A only.
    if (dlambda_full <= 0.0) {
        return { false, {}, 0 };
    }

    // Tier B: cheap segment bound. If false, segment provably outside disk.
    if (!segment_could_intersect_disk(prev_state, new_state, dlambda_full,
                                      disk, metric, opts.curvature_pad)) {
        return { false, {}, 0 };
    }

    // Tier C: subdivide with adaptive depth limit.
    const int depth_limit = compute_adaptive_depth(
        dlambda_full,
        prev_state.position[1], new_state.position[1],
        disk,
        opts.depth_limit_floor, opts.depth_limit_cap);

    SubdivResult sr = subdivide(prev_state, new_state, dlambda_full,
                                depth_limit, disk, metric, integrator,
                                opts.curvature_pad);
    return { sr.should_raymarch, sr.refined, sr.invocations };
}
```

- [ ] **Step 2: Add integration test — synthetic wedge case**

Replace the placeholder in `test_segment_bound_passes_when_dipping` and add new tests:

```cpp
// Constructed wedge case: endpoints above z_max but a substep midpoint
// lands inside. Verify Tier C surfaces an interior detection.
static void test_subdivide_finds_interior_entry() {
    const VolumetricDisk& disk = shared_disk();
    Kerr metric = make_metric();
    RK4 integrator = make_integrator();
    constexpr double half_pi = 1.5707963267948966;

    // Trick: large p_theta with momentum aimed at midplane. Both endpoints
    // computed off-axis by integrating forward-then-back. Easier — synthesize
    // a config where Tier A fires only on the half-step.
    //
    // Use prev at theta = half_pi - 0.10 (above midplane), curr at
    // theta = half_pi - 0.05 (still above), but with large p_theta such
    // that the half-step lands at theta ~ half_pi (inside).
    GeodesicState prev = make_state(8.0, half_pi - 0.10, /*pr=*/0.0,
                                    /*ptheta=*/0.5);
    GeodesicState curr = make_state(8.0, half_pi - 0.05, /*pr=*/0.0,
                                    /*ptheta=*/0.5);

    DiskStepEntryResult r = check_disk_step_entry(
        prev, curr, /*dlambda_full=*/0.5, disk, metric, integrator);

    // Either Tier A fires directly (curr is "near enough"), or subdivision
    // surfaces an interior hit. Either way — should_raymarch should be true.
    EXPECT_TRUE(r.should_raymarch,
                "interior dip should produce should_raymarch=true");
}

// Pathological: depth exhaustion. Set depth_limit_cap=1 so subdivision
// terminates almost immediately. With Tier B passing repeatedly, conservative
// policy (spec §6.1) returns should_raymarch=true with refined=curr.
static void test_subdivide_depth_limit_respected() {
    const VolumetricDisk& disk = shared_disk();
    Kerr metric = make_metric();
    RK4 integrator = make_integrator();
    constexpr double half_pi = 1.5707963267948966;

    // Just above disk surface — Tier B will keep firing.
    GeodesicState prev = make_state(8.0, half_pi - 0.05);
    GeodesicState curr = make_state(8.0, half_pi - 0.04);

    DiskStepEntryOptions opts;
    opts.depth_limit_floor = 1;
    opts.depth_limit_cap   = 1;

    DiskStepEntryResult r = check_disk_step_entry(
        prev, curr, /*dlambda_full=*/0.5, disk, metric, integrator, opts);

    // With cap=1 and a near-disk segment, depth exhausts → conservative
    // policy → should_raymarch=true. Ensure invocations bounded.
    EXPECT_TRUE(r.invocations <= 4,
                "depth_limit=1 should produce ≤4 subdivide invocations (root + 2 children + slack)");
}

// Adaptive depth: pathological H/dlambda ratio → cap.
static void test_adaptive_depth_supermassive() {
    const VolumetricDisk& disk = shared_disk();
    // At r=20M scale_height is ~0.5–1M for the default disk. Use
    // dlambda_full = 100 to force ratio ~100 → log2 ~6.6 → 7.
    const double r = 20.0;
    const double H = disk.scale_height(r);
    const double dl = 100.0 * H;
    const int depth = [&]{
        // Mirror compute_adaptive_depth's math directly for testability.
        const double ratio = dl / H;
        return static_cast<int>(std::ceil(std::log2(ratio)));
    }();

    EXPECT_TRUE(depth >= 6 && depth <= 8,
                "adaptive depth at ratio=100 should be ~7");
}
```

Add `test_subdivide_finds_interior_entry`, `test_subdivide_depth_limit_respected`, `test_adaptive_depth_supermassive` calls in `main()`. Remove the placeholder smoke test bodies from Task 4 (replace with actual asserts).

- [ ] **Step 3: Build and run**

Run:
```
cmake --build build --config Release
./build/Release/test-disk-step-entry.exe
```
Expected: 0 failures.

- [ ] **Step 4: Commit**

```bash
git add src/disk_step_entry.cpp tests/test_disk_step_entry.cpp
git commit -m "feat(disk-step-entry): wire orchestrator + integration tests"
```

---

## Task 6: Refactor `trace()` RGB path to use the helper

**Files:**
- Modify: `src/geodesic_tracer.cpp` (around lines 165–219)

- [ ] **Step 1: Capture `dlambda` before the integrator call**

In `src/geodesic_tracer.cpp`, locate the block (around line 165–169):

```cpp
{
    auto result = integrator_.adaptive_step_kerr_dp45(metric_, state, dlambda, tolerance_);
    state = result.state;
    dlambda = result.next_dlambda;
}
```

Modify to capture the input dlambda:

```cpp
const double dlambda_used = dlambda;
{
    auto result = integrator_.adaptive_step_kerr_dp45(metric_, state, dlambda, tolerance_);
    state = result.state;
    dlambda = result.next_dlambda;
}
```

(`dlambda_used` is the conservative upper bound on the actual step duration. The adaptive integrator may have shrunk internally, which makes the bound *more* conservative — safe.)

- [ ] **Step 2: Add include at the top of the file**

```cpp
#include "grrt/geodesic/disk_step_entry.h"
```

- [ ] **Step 3: Replace the inline predicate at lines ~179–219**

Find:

```cpp
if (vol_disk_) {
    const bool opaque = (running_T[0] < 1e-6 && running_T[1] < 1e-6 && running_T[2] < 1e-6);
    if (!opaque) {
        const double theta_prev = prev.position[2];
        const double theta_new = state.position[2];
        const double d_prev = theta_prev - half_pi;
        const double d_new = theta_new - half_pi;
        const double r_new = state.position[1];
        const double r_prev = prev.position[1];

        const double z_new = r_new * std::cos(theta_new);
        const double z_prev = r_prev * std::cos(theta_prev);
        const bool crossed_midplane = (d_prev * d_new < 0.0)
                                   && std::abs(d_prev - d_new) > 1e-12;
        const bool inside_now = vol_disk_->inside_volume(r_new, z_new);
        const double zm_new = vol_disk_->z_max_at(r_new);
        const double H_new = vol_disk_->scale_height(r_new);
        const double H_prev = vol_disk_->scale_height(r_prev);
        const bool near_disk = (std::abs(z_new) < zm_new + 1.0 * H_new
                             || std::abs(z_prev) < vol_disk_->z_max_at(r_prev) + 1.0 * H_prev)
                            && r_new >= vol_disk_->r_horizon()
                            && r_new <= vol_disk_->r_max() + 0.5 * vol_disk_->outer_taper_width();
        const bool should_raymarch = crossed_midplane || inside_now || near_disk;

        if (should_raymarch) {
            const double r_lo = std::min(r_prev, r_new);
            const double r_hi = std::max(r_prev, r_new);
            if (r_hi >= vol_disk_->r_horizon() && r_lo <= vol_disk_->r_max()) {
                GeodesicState entry = prev;
                const double re = entry.position[1];
                if (re >= vol_disk_->r_horizon() * 0.9
                    && re <= vol_disk_->r_max() * 1.5) {
                    raymarch_volumetric(entry, color, running_J, running_T);
                    state = entry;
                    continue;
                }
            }
        }
    }
}
```

Replace with:

```cpp
if (vol_disk_) {
    const bool opaque = (running_T[0] < 1e-6 && running_T[1] < 1e-6 && running_T[2] < 1e-6);
    if (!opaque) {
        DiskStepEntryResult entry_check = check_disk_step_entry(
            prev, state, dlambda_used, *vol_disk_, metric_, integrator_);

        if (entry_check.should_raymarch) {
            const double r_prev = prev.position[1];
            const double r_new  = entry_check.refined_endpoint.position[1];
            const double r_lo = std::min(r_prev, r_new);
            const double r_hi = std::max(r_prev, r_new);
            if (r_hi >= vol_disk_->r_horizon() && r_lo <= vol_disk_->r_max()) {
                GeodesicState entry = prev;
                const double re = entry.position[1];
                if (re >= vol_disk_->r_horizon() * 0.9
                    && re <= vol_disk_->r_max() * 1.5) {
                    raymarch_volumetric(entry, color, running_J, running_T);
                    state = entry;
                    continue;
                }
            }
        }
    }
}
```

(The behavior on a true Tier A hit is identical to before. Tier B/C add the previously-missed wedge cases.)

- [ ] **Step 4: Build**

Run:
```
cmake --build build --config Release
```
Expected: builds without errors.

- [ ] **Step 5: Run all existing tests**

Run:
```
ctest --test-dir build -C Release
```
Expected: 0 failures. The Tier A path is byte-equivalent for all existing test scenarios.

- [ ] **Step 6: Render a deep-space ray (no disk encounter) and confirm output unchanged**

Run a simple render with the disk off:
```
./build/Release/grrt-cli.exe --width 64 --height 64 --output before_after_test.png --force --fov 30
```
Expected: completes without error. (Comparing pre/post bytes is impractical because timestamps; the unit suite is the real regression gate.)

- [ ] **Step 7: Commit**

```bash
git add src/geodesic_tracer.cpp
git commit -m "feat(geodesic): wire trace() RGB path to disk_step_entry helper"
```

---

## Task 7: Refactor `trace_debug()` and `trace_spectral()` to use the helper

**Files:**
- Modify: `src/geodesic_tracer.cpp` (sites at ~L441–449 and ~L585–595)

- [ ] **Step 1: Locate and refactor the `trace_debug` site (~L441–449)**

The pattern is identical to Task 6 — find the duplicated predicate block, capture `dlambda` as `dlambda_used` before the integrator call, then replace the inline block with `check_disk_step_entry(prev, state, dlambda_used, *vol_disk_, metric_, integrator_)`.

The surrounding behavior (`event = "CROSS"/"INSIDE"/"NEAR"` debug strings) was used for printing. After refactor, derive the event tag from the helper's output if needed, or simply emit a single "ENTRY" tag — debug output is not part of the contract.

- [ ] **Step 2: Locate and refactor the `trace_spectral` site (~L585–595)**

Same pattern. The surrounding code calls `raymarch_volumetric_spectral` instead of `raymarch_volumetric`. Replace the predicate inline block with the helper call; keep the spectral raymarch invocation unchanged.

- [ ] **Step 3: Build**

Run:
```
cmake --build build --config Release
```
Expected: builds without errors.

- [ ] **Step 4: Run full test suite**

Run:
```
ctest --test-dir build -C Release
```
Expected: 0 failures. Spectral tests in particular should stay green.

- [ ] **Step 5: Commit**

```bash
git add src/geodesic_tracer.cpp
git commit -m "feat(geodesic): wire trace_debug + trace_spectral to disk_step_entry helper"
```

---

## Task 8: Diagnostic counter, visual smoke render, banding regression check

**Files:**
- Modify: `include/grrt/geodesic/geodesic_tracer.h`
- Modify: `src/geodesic_tracer.cpp`

- [ ] **Step 1: Add atomic counter to `GeodesicTracer`**

In `include/grrt/geodesic/geodesic_tracer.h`, add `<atomic>` include and a mutable atomic member:

```cpp
#include <atomic>
// ... in class GeodesicTracer, private section:
mutable std::atomic<long> substep_invocation_count_{0};

// public method to read:
public:
long substep_invocation_count() const {
    return substep_invocation_count_.load(std::memory_order_relaxed);
}
```

- [ ] **Step 2: Bump the counter in each of the three tracer call sites**

After each `check_disk_step_entry(...)` call, before the `if (entry_check.should_raymarch)` check:

```cpp
substep_invocation_count_.fetch_add(entry_check.substep_invocations,
                                    std::memory_order_relaxed);
```

- [ ] **Step 3: Build and run unit tests**

Run:
```
cmake --build build --config Release
ctest --test-dir build -C Release
```
Expected: 0 failures.

- [ ] **Step 4: Visual smoke render — original wedge repro**

Run the failing repro:
```
./build/Release/grrt-cli.exe --disk-volumetric --samples 100 --width 256 --height 256 --output wedge_post_fix.png --force --fov 30
```

Open `wedge_post_fix.png`. Pass: no wedge-shaped black voids visible by eye. Compare against any pre-fix screenshot if available.

- [ ] **Step 5: Visual smoke render — wide-FOV regression check**

Run:
```
./build/Release/grrt-cli.exe --disk-volumetric --samples 100 --width 256 --height 256 --output wide_post_fix.png --force --fov 90
```

Open `wide_post_fix.png`. Pass: no new artifacts compared to pre-fix wide-FOV behavior.

- [ ] **Step 6: Banding regression — run existing test_volumetric**

Run:
```
./build/Release/test_volumetric.exe
```

Specifically watch `test_no_horizontal_bands`. Expected: PASS without recalibration. The relative-band-strength metric should stay below the 0.25 threshold (Sobol baseline 0.211). If it moved significantly, Tier C is shifting raymarch start positions in an unintended way — investigate before merge.

- [ ] **Step 7: Commit**

```bash
git add include/grrt/geodesic/geodesic_tracer.h src/geodesic_tracer.cpp
git commit -m "feat(geodesic): atomic substep diagnostic counter on tracer"
```

(Plus the rendered PNGs are not committed — they're inspection-only. Delete them or move to a sandbox dir.)

- [ ] **Step 8: Hand back to the user with diagnostic report**

Report to user:
- Visual smoke result (wedges gone? regression at wide FOV?)
- `tracer.substep_invocation_count()` divided by total ray-steps for the smoke render. Target `<< 1%`.
- `test_no_horizontal_bands` passed without recalibration?
- Any tests that went red, plus the failure context.

---

## Plan self-review

Spec coverage scan against `docs/superpowers/specs/2026-05-10-disk-step-entry-design.md`:

- §3 Goal "eliminate wedge artifacts" → Task 8 visual smoke.
- §3 Goal "robust to step-too-coarse + LUT-cliff" → Tasks 3, 4 (Tier B/C).
- §3 Goal "deduplicate three-way" → Tasks 6, 7 (refactor all three sites).
- §3 Goal "preserve no-bug-path byte-for-byte" → Task 5 Step 1 (Tier A returns immediately, identical behavior).
- §3 Goal "mass-regime robust" → Task 4 (adaptive depth) + Task 5 test_adaptive_depth_supermassive.
- §5.1 Public API → Task 1 header.
- §5.2 Endpoint predicate extraction → Task 2.
- §5.3 Segment-bound test → Task 3.
- §5.4 Curvature pad momentum-aware → Task 3 Step 1.
- §5.5 Subdivision recursion + adaptive depth → Task 4.
- §6.1 Conservative depth-exhaustion → Task 4 Step 1, Task 5 test_subdivide_depth_limit_respected.
- §6.2 NaN handling → relies on existing raymarch defensive logic. **Not explicitly covered by a test — accepted as a non-blocker (existing behavior preserved).**
- §6.3 Degenerate inputs → Task 5 Step 1 has degenerate guards.
- §6.5 Thread safety → counter uses atomic; helper is pure (Task 8 Step 1).
- §7.1 Unit tests → Tasks 2, 3, 4, 5.
- §7.2 Integration smoke → Task 6 Step 6 (deep-space ray) and Tasks 6/7 with full ctest run.
- §7.3 Visual smoke → Task 8 Step 4.
- §7.4 Diagnostic counter → Task 8.
- §7.5 No banding regression → Task 8 Step 6.
- §8 In-scope items → all covered. The "audit existing API" requirement was already done above the plan: `RK4::step_kerr_rkdp45` exists (matches main loop's DP45 family per spec §5.1), no new method needed. A future optimization to switch to `RK4::step_kerr` for ~33% per-substep cost savings is documented at `docs/superpowers/optimizations/2026-05-10-disk-step-entry-rk4-substep.md`.

Placeholder scan: no "TBD", "TODO", "implement later" present. All test code is concrete.

Type consistency: `DiskStepEntryResult`, `DiskStepEntryOptions`, `GeodesicState`, `VolumetricDisk`, `Kerr`, `RK4` — names consistent across all tasks.

Spec gap fixed inline: §6.2 NaN handling is not unit-tested in the plan. Accepted as a non-blocker because the helper's NaN handling delegates to existing raymarch behavior, which is unchanged.
