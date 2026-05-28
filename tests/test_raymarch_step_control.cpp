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

    // Fine step far from the disk: crosses=false AND dz < quarter_H. Completes
    // the (crosses x dz>qH) truth table — the coarseness gate is not consulted
    // when the interval is remote, so this returns false regardless.
    check("fine_far_from_disk",
          step_needs_z_refinement(0.50, 0.5005, qH, env), false);

    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
