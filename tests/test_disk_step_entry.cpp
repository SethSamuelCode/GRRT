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

int main() {
    std::printf("Running test_disk_step_entry...\n");
    test_endpoints_inside_disk_should_raymarch();
    test_endpoint_predicate_equivalence();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
