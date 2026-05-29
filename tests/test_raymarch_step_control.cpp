// tests/test_raymarch_step_control.cpp
//
// Unit tests for raymarch_exits_outer — the direction-aware outer-radius
// exit predicate. Pure scalar logic; no disk, metric, or integrator needed.

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
