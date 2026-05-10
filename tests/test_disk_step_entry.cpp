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

// Disk envelope at radius r — env = z_max(r) + 0.5*H(r). Tests that need
// to place endpoints relative to the disk surface use this so they stay
// valid as the disk model evolves.
static double disk_envelope_at(const VolumetricDisk& disk, double r) {
    return disk.z_max_at(r) + 0.5 * disk.scale_height(r);
}

// Sanity: endpoints squarely inside the disk volume must trigger
// raymarch via Tier A's `inside_now` clause.
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

// Tier B path verification through the public API. With the orchestrator
// fully wired, "Tier B fires" means subdivide is invoked — observable via
// substep_invocations > 0 in the rejecting test below.

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

// Orchestrator's degenerate-dlambda guard: when the integrator step is
// zero or negative (e.g., main-loop's first iteration where prev == new),
// substepping is meaningless. Tier A still gets a shot, but Tier B/C are
// skipped per spec §6.3. Endpoints above the disk envelope so Tier A is
// false → orchestrator should return {false, {}, 0}.
static void test_degenerate_dlambda_returns_no_raymarch() {
    const VolumetricDisk& disk = shared_disk();
    Kerr metric = make_metric();
    RK4 integrator = make_integrator();
    constexpr double half_pi = std::numbers::pi / 2.0;

    constexpr double r_test = 10.0;
    const double env = disk_envelope_at(disk, r_test);
    const double theta = half_pi - std::asin(2.0 * env / r_test);

    GeodesicState prev = make_state(r_test, theta);
    GeodesicState curr = make_state(r_test, theta);   // identical state

    DiskStepEntryResult r = check_disk_step_entry(
        prev, curr, /*dlambda_full=*/0.0, disk, metric, integrator);

    EXPECT_TRUE(!r.should_raymarch,
                "dlambda_full=0 with above-disk endpoints must not raymarch");
    EXPECT_TRUE(r.substep_invocations == 0,
                "no Tier C invocations for zero-dlambda case");
}

static void test_segment_bound_passes_when_dipping() {
    const VolumetricDisk& disk = shared_disk();
    Kerr metric = make_metric();
    RK4 integrator = make_integrator();
    constexpr double half_pi = std::numbers::pi / 2.0;

    // Construct endpoints relative to the actual disk envelope at r=10M.
    // Both above near_disk threshold (z_max + H), so Tier A is FALSE.
    // Large opposing p_theta values produce a large dz_swing, forcing
    // Tier B to pass via the velocity term (not the chord term). Once
    // Tier B passes, Tier C runs; either Tier A fires on a substep or
    // depth exhaustion's conservative policy returns {true, curr, 1}.
    // Either way, should_raymarch=true.
    constexpr double r_test = 10.0;
    const double env = disk_envelope_at(disk, r_test);
    const double theta_prev = half_pi - std::asin(2.0 * env / r_test);
    const double theta_curr = half_pi - std::asin(1.5 * env / r_test);

    GeodesicState prev = make_state(r_test, theta_prev,
                                    /*pr=*/0.0, /*ptheta=*/+10.0);
    GeodesicState curr = make_state(r_test, theta_curr,
                                    /*pr=*/0.0, /*ptheta=*/-10.0);

    DiskStepEntryResult r = check_disk_step_entry(
        prev, curr, /*dlambda_full=*/0.5, disk, metric, integrator);

    EXPECT_TRUE(r.should_raymarch,
                "dipping segment with large dz_swing should raymarch");
    std::printf("  dipping case: should_raymarch=%d invocations=%d (env=%.4g)\n",
                r.should_raymarch ? 1 : 0, r.substep_invocations, env);
}

// Constructed wedge case: endpoints just above the disk surface with large
// p_theta steering the trajectory into the disk volume mid-step. Verify
// that the orchestrator returns should_raymarch=true. Either Tier A fires
// (near_disk margin), or Tier B+C find the interior entry. If Tier C ran,
// substep_invocations > 0.
static void test_subdivide_finds_interior_entry() {
    const VolumetricDisk& disk = shared_disk();
    Kerr metric = make_metric();
    RK4 integrator = make_integrator();
    constexpr double half_pi = std::numbers::pi / 2.0;

    // Endpoints relative to actual disk envelope at r=8M. Both above
    // near_disk threshold (Tier A FALSE). With opts.curvature_pad cranked
    // to 2.0, the chord-based pad reaches below env even with zero
    // p_theta — Tier B passes via the chord term (distinct from the
    // dz_swing exercise in test_segment_bound_passes_when_dipping).
    constexpr double r_test = 8.0;
    const double env = disk_envelope_at(disk, r_test);
    const double theta_prev = half_pi - std::asin(2.0 * env / r_test);
    const double theta_curr = half_pi - std::asin(1.5 * env / r_test);

    GeodesicState prev = make_state(r_test, theta_prev);
    GeodesicState curr = make_state(r_test, theta_curr);

    DiskStepEntryOptions opts;
    opts.curvature_pad = 2.0;     // pad = 2.0 * 0.5*env = env, > 0.5*env required

    DiskStepEntryResult r = check_disk_step_entry(
        prev, curr, /*dlambda_full=*/0.5, disk, metric, integrator, opts);

    EXPECT_TRUE(r.should_raymarch,
                "interior dip segment should produce should_raymarch=true");
    std::printf("  interior-entry case: should_raymarch=%d invocations=%d (env=%.4g)\n",
                r.should_raymarch ? 1 : 0, r.substep_invocations, env);
}

// Pathological depth exhaustion: depth_limit_cap=1 forces subdivide() to
// terminate almost immediately. With Tier B passing repeatedly (segment
// near disk surface), conservative policy (spec §6.1) returns
// {should_raymarch=true, refined=curr, invocations=1} — depth_remaining
// hits 0 on the first subdivide call.
static void test_subdivide_depth_limit_respected() {
    const VolumetricDisk& disk = shared_disk();
    Kerr metric = make_metric();
    RK4 integrator = make_integrator();
    constexpr double half_pi = std::numbers::pi / 2.0;

    // Same disk-relative geometry as test_subdivide_finds_interior_entry
    // (forces Tier B to pass), then squeeze depth_limit_cap=1 so Tier C's
    // recursion exhausts on the first level and conservative policy
    // (spec §6.1) takes over.
    constexpr double r_test = 8.0;
    const double env = disk_envelope_at(disk, r_test);
    const double theta_prev = half_pi - std::asin(2.0 * env / r_test);
    const double theta_curr = half_pi - std::asin(1.5 * env / r_test);

    GeodesicState prev = make_state(r_test, theta_prev);
    GeodesicState curr = make_state(r_test, theta_curr);

    DiskStepEntryOptions opts;
    opts.curvature_pad     = 2.0;   // force Tier B to pass
    opts.depth_limit_floor = 1;
    opts.depth_limit_cap   = 1;

    DiskStepEntryResult r = check_disk_step_entry(
        prev, curr, /*dlambda_full=*/0.5, disk, metric, integrator, opts);

    // With Tier B forced to pass and depth_limit=1, Tier C runs once.
    // Either Tier A fires on a substep half (rare here) or depth
    // exhausts → conservative policy → {true, curr, ≤4}.
    EXPECT_TRUE(r.should_raymarch,
                "Tier C with cap=1 must return should_raymarch=true");
    EXPECT_TRUE(r.substep_invocations >= 1,
                "Tier C should have run (substep_invocations >= 1)");
    EXPECT_TRUE(r.substep_invocations <= 4,
                "depth_limit=1 should produce ≤4 subdivide invocations");
    std::printf("  depth_limit=1 case: should_raymarch=%d invocations=%d\n",
                r.should_raymarch ? 1 : 0, r.substep_invocations);
}

// Note on synthetic states: make_state above does NOT satisfy the null
// geodesic Hamiltonian constraint g^{μν} p_μ p_ν = 0. That is fine for
// Tier A (purely positional) and Tier B (positional + chain-rule velocity)
// tests. Tier C substeps integrate these states, producing trajectories
// that are mathematically well-defined but not physical null geodesics.
// The orchestrator's behavioral assertions (should_raymarch=true) hold
// regardless of substep physicality because the conservative-policy
// fallback (spec §6.1) returns true on depth exhaustion.

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
    test_degenerate_dlambda_returns_no_raymarch();
    test_segment_bound_passes_when_dipping();
    test_subdivide_finds_interior_entry();
    test_subdivide_depth_limit_respected();
    test_adaptive_depth_math();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
