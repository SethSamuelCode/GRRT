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
