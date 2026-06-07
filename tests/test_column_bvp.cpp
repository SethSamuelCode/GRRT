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
    in.T_eff = 1e5; in.shear = 1e3; in.omega_z = 1e3;
    in.alpha = 0.1; in.rho_mid_guess = 1.0; in.n_nodes = 16;
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    auto sol = grrt::solve_column_bvp(in, lut);
    std::printf("  returned q.size()=%zu\n", sol.q.size());
    if (sol.q.size() != 16) { std::printf("  FAIL: grid size\n"); failures++; }
}

static void test_residual_hydrostatic_identity() {
    std::printf("\n=== residual: hydrostatic identity on a Gaussian column ===\n");
    using namespace grrt::constants;
    const double T = 1e5, rho_mid = 1.0, omega_z = 1e3;
    const double cs2 = k_B * T / (mu_fully_ionized * m_p);
    const double H = std::sqrt(cs2) / omega_z;
    const double dz = 0.02 * H;   // finer FD step: O(dz^2) truncation well under the 1e-3 tolerance
    auto rho = [&](double z){ return rho_mid * std::exp(-z*z/(2*H*H)); };
    auto P   = [&](double z){ return rho(z) * cs2; };   // isothermal gas
    const double z = 1.5 * H;
    const double dPdz = (P(z+dz) - P(z-dz)) / (2*dz);
    const double resid = dPdz + rho(z) * omega_z*omega_z * z;
    std::printf("  hydrostatic resid=%.3e (rel %.3e)\n", resid, P(z)/H);
    if (std::abs(resid) > 1e-3 * (P(z)/H)) { std::printf("  FAIL\n"); failures++; }
}

static void test_residual_count_finite() {
    std::printf("\n=== residual: length 4N+2, all finite ===\n");
    grrt::ColumnInputs in{}; in.T_eff = 1e5; in.shear = 1e3; in.omega_z = 1e3;
    in.alpha = 0.1; in.rho_mid_guess = 1.0; in.n_nodes = 32;
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    std::vector<double> U, R;
    grrt::column_residual_test(in, lut, U, R);
    std::printf("  U.size=%zu R.size=%zu (expect %d)\n", U.size(), R.size(), 4*32+2);
    if ((int)R.size() != 4*32+2) { std::printf("  FAIL: residual length\n"); failures++; }
    if ((int)U.size() != 4*32+2) { std::printf("  FAIL: state length\n"); failures++; }
    bool finite = true; for (double x : R) if (!std::isfinite(x)) finite = false;
    if (!finite) { std::printf("  FAIL: non-finite residual\n"); failures++; }
}

static void test_numerical_jacobian_finite() {
    std::printf("\n=== numerical Jacobian: finite, correct shape ===\n");
    grrt::ColumnInputs in{}; in.T_eff = 1e5; in.shear = 1e3; in.omega_z = 1e3;
    in.alpha = 0.1; in.rho_mid_guess = 1.0; in.n_nodes = 24;
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    std::vector<double> Jdense; int n = 0;
    grrt::column_numerical_jacobian_test(in, lut, Jdense, n);
    std::printf("  n=%d entries=%zu (expect %d, %d)\n", n, Jdense.size(), 4*24+2, (4*24+2)*(4*24+2));
    if (n != 4*24+2) { std::printf("  FAIL: size n\n"); failures++; }
    if ((int)Jdense.size() != (4*24+2)*(4*24+2)) { std::printf("  FAIL: matrix size\n"); failures++; }
    bool finite = true; for (double x : Jdense) if (!std::isfinite(x)) finite = false;
    if (!finite) { std::printf("  FAIL: non-finite Jacobian entry\n"); failures++; }
    // sanity: the Jacobian must be non-trivial (not all zeros)
    double maxabs = 0.0; for (double x : Jdense) maxabs = std::max(maxabs, std::abs(x));
    if (maxabs <= 0.0) { std::printf("  FAIL: all-zero Jacobian\n"); failures++; }
}

int main() {
    test_eos();
    test_scaffold();
    test_residual_hydrostatic_identity();
    test_residual_count_finite();
    test_numerical_jacobian_finite();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
