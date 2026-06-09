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

static void test_kerr_factors() {
    std::printf("\n=== Kerr relativistic factors (verified vs §22) ===\n");
    using namespace grrt::slim_detail;
    const double M = 1.0;
    // --- Schwarzschild (a=0), r=10 ---
    check("omega_k Schw r=10", omega_k(M,0.0,10.0), std::pow(10.0,-1.5), 1e-12);
    check("calC Schw r=10",    calC(M,0.0,10.0),    0.7, 1e-12);
    check("calD Schw r=10",    calD(M,0.0,10.0),    0.8, 1e-12);
    check("calH Schw r=10",    calH(M,0.0,10.0),    1.0, 1e-12);
    // Schwarzschild: vertical epicyclic EQUALS orbital -> omega_perp2 = omega_k^2 = M/r^3 = 1e-3.
    check("omega_perp2 Schw r=10 (= Omega_K^2)", omega_perp2(M,0.0,10.0), 1e-3, 1e-12);
    check("kerr_delta Schw r=10", kerr_delta(M,0.0,10.0), 80.0, 1e-12);
    check("kerr_A Schw r=10",     kerr_A(M,0.0,10.0),     10000.0, 1e-12);
    // --- Kerr a=0.9, r=10 (independently computed reference values) ---
    check("omega_k Kerr a0.9 r10",    omega_k(M,0.9,10.0),    0.03074768, 1e-6);
    check("calC Kerr a0.9 r10",       calC(M,0.9,10.0),       0.75692097, 1e-6);
    check("calH Kerr a0.9 r10",       calH(M,0.9,10.0),       0.91045797, 1e-6);
    check("omega_perp2 Kerr a0.9 r10",omega_perp2(M,0.9,10.0),8.607652e-4, 1e-5);
    check("kerr_delta Kerr a0.9 r10", kerr_delta(M,0.9,10.0), 80.81, 1e-9);
    check("kerr_A Kerr a0.9 r10",     kerr_A(M,0.9,10.0),     10097.2, 1e-9);
    // Cross-check the identity that ties us to the rest of GRRT: omega_perp2 == omega_k^2 * calH,
    // which is exactly the formula in VolumetricDisk::omega_z_sq (Omega^2*(1-4a√M/√(r^3)+3a²/r²)).
    const double a=0.7, r=7.5;
    const double ref = omega_k(M,a,r)*omega_k(M,a,r)*calH(M,a,r);
    check("omega_perp2 identity == omega_z_sq form", omega_perp2(M,a,r), ref, 1e-14);
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
    test_kerr_factors();
    test_links_and_returns();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
