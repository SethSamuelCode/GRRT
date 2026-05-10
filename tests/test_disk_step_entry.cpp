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
#include <numbers>

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
    s.position = Vec4{{0.0, r, theta, 0.0}};          // (t, r, theta, phi)
    s.momentum = Vec4{{-1.0, pr, ptheta, 0.0}};       // (-E=-1, p_r, p_theta, p_phi)
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
    constexpr double half_pi = std::numbers::pi / 2.0;
    GeodesicState prev = make_state(10.0, half_pi);
    GeodesicState curr = make_state(10.0, half_pi);

    DiskStepEntryResult r = check_disk_step_entry(
        prev, curr, /*dlambda_full=*/1.0, disk, metric, integrator);

    EXPECT_TRUE(r.should_raymarch,
                "endpoints inside disk should trigger raymarch");
}

// Test the extracted endpoint predicate matches the original inline logic
// across a battery of (prev, curr) cases. Reproduces the inline computation
// directly and asserts identical result.
static bool inline_endpoint_predicate_replica(const GeodesicState& prev,
                                              const GeodesicState& curr,
                                              const VolumetricDisk& disk) {
    constexpr double half_pi = std::numbers::pi / 2.0;
    const double theta_prev = prev.position[2];
    const double theta_curr = curr.position[2];
    const double d_prev = theta_prev - half_pi;
    const double d_curr = theta_curr - half_pi;
    const double r_curr = curr.position[1];
    const double r_prev = prev.position[1];
    const double z_curr = r_curr * std::cos(theta_curr);
    const double z_prev = r_prev * std::cos(theta_prev);
    const bool crossed_midplane =
        (d_prev * d_curr < 0.0) && std::abs(d_prev - d_curr) > 1e-12;
    const bool inside_now = disk.inside_volume(r_curr, z_curr);
    const double zm_curr = disk.z_max_at(r_curr);
    const double H_curr  = disk.scale_height(r_curr);
    const double H_prev = disk.scale_height(r_prev);
    const bool near_disk =
        (std::abs(z_curr) < zm_curr + 1.0 * H_curr
         || std::abs(z_prev) < disk.z_max_at(r_prev) + 1.0 * H_prev)
        && r_curr >= disk.r_horizon()
        && r_curr <= disk.r_max() + 0.5 * disk.outer_taper_width();
    return crossed_midplane || inside_now || near_disk;
}

static void test_endpoint_predicate_equivalence() {
    const VolumetricDisk& disk = shared_disk();
    Kerr metric = make_metric();
    RK4 integrator = make_integrator();
    constexpr double half_pi = std::numbers::pi / 2.0;

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

// Probe the Tier B path indirectly via the public API. After Task 5 wires
// Tier C, "Tier B fires" means subdivide is invoked, observable via
// substep_invocations > 0. Pre-Task-5 these tests just exercise the path
// to ensure no crash and that Tier-A-still-applies semantics are intact.

static void test_segment_bound_rejects_far_above() {
    const VolumetricDisk& disk = shared_disk();
    Kerr metric = make_metric();
    RK4 integrator = make_integrator();

    // Both endpoints with |z| ≈ 20 (well above disk top, even at hot inner radii).
    // No midplane crossing, no inside_now, no near_disk → Tier A returns false.
    // Pre-Task-5: orchestrator returns {false, {}, 0}. After Task 5: still
    // false (Tier B rejects), invocations=0.
    GeodesicState prev = make_state(20.0, 0.05);   // theta near 0 → z ≈ r*cos ≈ 20
    GeodesicState curr = make_state(20.0, 0.10);

    DiskStepEntryResult r = check_disk_step_entry(
        prev, curr, /*dlambda_full=*/0.5, disk, metric, integrator);

    EXPECT_TRUE(!r.should_raymarch,
                "segment far above disk should not raymarch");
    EXPECT_TRUE(r.substep_invocations == 0,
                "no subdivision counter increments for far-above segment");
}

static void test_segment_bound_passes_when_dipping() {
    const VolumetricDisk& disk = shared_disk();
    Kerr metric = make_metric();
    RK4 integrator = make_integrator();
    constexpr double half_pi = std::numbers::pi / 2.0;

    // Endpoints just above disk top; large p_theta makes dz_swing dominate the
    // pad, which (after Task 5) will trigger Tier B → Tier C subdivision.
    // Pre-Task-5: orchestrator only runs Tier A. Tier A's `near_disk` may or
    // may not fire depending on disk parameters at r=10M; this is a smoke
    // test that just confirms no crash. Real assertions land in Task 5.
    GeodesicState prev = make_state(10.0, half_pi - 0.10, /*pr=*/0.0,
                                    /*ptheta=*/-0.5);
    GeodesicState curr = make_state(10.0, half_pi - 0.08, /*pr=*/0.0,
                                    /*ptheta=*/-0.5);

    DiskStepEntryResult r = check_disk_step_entry(
        prev, curr, /*dlambda_full=*/2.0, disk, metric, integrator);

    // Pre-Task-5 smoke check: no crash, fields are well-formed.
    (void)r;
    std::printf("  test_segment_bound_passes_when_dipping smoke: should_raymarch=%d invocations=%d\n",
                r.should_raymarch ? 1 : 0, r.substep_invocations);
}

// Note on synthetic states: make_state above does NOT satisfy the null
// geodesic Hamiltonian constraint g^{μν} p_μ p_ν = 0. That is fine for
// Tier A (purely positional) and Tier B (positional + chain-rule velocity)
// tests. Tier C substeps integrate these states, producing non-physical
// trajectories — Task 5 introduces physically valid states for end-to-end
// Tier C assertions. Pre-Task-5 Tier C tests are placeholders only.

static void test_adaptive_depth_math() {
    // Mirror compute_adaptive_depth's math directly (it lives in anonymous
    // namespace, so we reproduce the formula here for testability):
    //   needed = ceil(log2(dlambda / H_min))
    //   depth  = clamp(needed, floor, cap)
    //
    // We can't observe compute_adaptive_depth() through the public API
    // until Task 5. This test verifies the formula's expected output for
    // representative ratios so the implementation can be validated against
    // it after Task 5 wiring.

    auto expected_depth = [](double dl_over_h, int floor, int cap) {
        if (dl_over_h <= 1.0) return floor;
        const int needed = static_cast<int>(std::ceil(std::log2(dl_over_h)));
        return std::clamp(needed, floor, cap);
    };

    EXPECT_TRUE(expected_depth(0.5, 4, 10) == 4,
                "ratio<1 returns floor");
    EXPECT_TRUE(expected_depth(2.0, 4, 10) == 4,
                "ratio=2 (log2=1) clamped to floor=4");
    EXPECT_TRUE(expected_depth(100.0, 4, 10) == 7,
                "ratio=100 → ceil(log2)=7 (supermassive AGN regime)");
    EXPECT_TRUE(expected_depth(10000.0, 4, 10) == 10,
                "ratio=10000 → ceil(log2)=14 → clamped to cap=10");
}

int main() {
    std::printf("Running test_disk_step_entry...\n");
    test_endpoints_inside_disk_should_raymarch();
    test_endpoint_predicate_equivalence();
    test_segment_bound_rejects_far_above();
    test_segment_bound_passes_when_dipping();
    test_adaptive_depth_math();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
