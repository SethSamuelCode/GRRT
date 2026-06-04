#include "grrt/scene/disk_column_bvp.h"
#include "grrt/color/opacity.h"
#include "grrt/math/constants.h"
#include <cstdio>
#include <cmath>
#include <algorithm>

int failures = 0;

static void check(const char* name, double got, double expected, double rel_tol) {
    double rel = std::abs(got - expected) / std::max(std::abs(expected), 1e-30);
    bool pass = rel < rel_tol;
    std::printf("  %s: got=%.6e expected=%.6e rel=%.2e %s\n",
                name, got, expected, rel, pass ? "PASS" : "FAIL");
    if (!pass) failures++;
}

static void test_eos() {
    std::printf("\n=== EOS: rho from (P,T) ===\n");
    using namespace grrt::constants;
    // Gas-pressure-dominated point: choose rho, T; compute P_gas+P_rad; invert.
    const double rho = 1.0, T = 1e5;
    const double P = rho * k_B * T / (mu_fully_ionized * m_p) + (a_rad / 3.0) * std::pow(T, 4);
    check("eos_rho inverts", grrt::eos_rho(P, T), rho, 1e-12);
    // Radiation pressure exceeding total → non-physical → <= 0.
    const double P_small = (a_rad / 3.0) * std::pow(T, 4) * 0.5; // below P_rad
    if (grrt::eos_rho(P_small, T) > 0.0) {
        std::printf("  FAIL: should be <=0 when P < P_rad\n"); failures++;
    }
}

static void test_scaffold() {
    std::printf("\n=== scaffold: solve_column_bvp links and returns ===\n");
    grrt::ColumnInputs in{};
    in.T_eff = 1e5; in.omega_orb = 1e3; in.omega_z = 1e3;
    in.alpha = 0.1; in.rho_mid_guess = 1.0; in.n_nodes = 16;
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    auto sol = grrt::solve_column_bvp(in, lut);
    std::printf("  returned q.size()=%zu\n", sol.q.size());
    if (sol.q.size() != 16) { std::printf("  FAIL: grid size\n"); failures++; }
}

int main() {
    test_eos();
    test_scaffold();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
