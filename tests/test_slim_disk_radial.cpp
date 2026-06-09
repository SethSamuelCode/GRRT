#include "grrt/scene/slim_disk_radial.h"
#include "grrt/color/opacity.h"
#include <cstdio>
#include <cmath>

int failures = 0;
static void check(const char* name, double got, double expected, double rel_tol) {
    double rel = std::abs(got - expected) / std::max(std::abs(expected), 1e-30);
    bool pass = rel < rel_tol;
    std::printf("  %s: got=%.6e expected=%.6e rel=%.2e %s\n",
                name, got, expected, rel, pass ? "PASS" : "FAIL");
    if (!pass) failures++;
}

static void test_links_and_returns() {
    std::printf("\n=== scaffold: solve_slim_disk_radial links and returns ===\n");
    grrt::SlimDiskInputs in{};
    in.mass = 1.0; in.spin = 0.9; in.mdot = 1e17; in.alpha = 0.1;
    in.r_g = 1.48e6; in.r_in = 1.5; in.r_out = 50.0; in.n_nodes = 64;
    auto lut = grrt::build_opacity_luts(1e-14, 1e4, 3000.0, 1e8);
    auto sol = grrt::solve_slim_disk_radial(in, lut);
    std::printf("  converged=%d (stub returns false; later tasks implement)\n", sol.converged);
    // Scaffold-level contract: the call links, returns, and the stub is honestly non-converged.
    if (sol.converged) { std::printf("  FAIL: stub should not claim convergence\n"); failures++; }
}

int main() {
    test_links_and_returns();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
