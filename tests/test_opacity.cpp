#include "grrt/color/opacity.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

int failures = 0;

void check(const char* name, double got, double expected, double rel_tol) {
    double rel_err = std::abs(got - expected) / std::max(std::abs(expected), 1e-30);
    bool pass = rel_err < rel_tol;
    std::printf("  %s: got=%.4e expected=%.4e rel_err=%.2e %s\n",
                name, got, expected, rel_err, pass ? "PASS" : "FAIL");
    if (!pass) failures++;
}

void test_saha_fully_ionized() {
    std::printf("\n=== Saha: fully ionized (T=1e7 K, rho=1e-8 g/cm^3) ===\n");
    auto ion = grrt::solve_saha(1e-8, 1e7);
    double n_e_expected = 1e-8 * (1.0 + 0.70) / (2.0 * 1.672622e-24);
    check("n_e", ion.n_e, n_e_expected, 0.05);
    check("mu", ion.mu, 0.6, 0.05);
    // At T=1e7K, H- should be negligible compared to n_e
    if (ion.n_Hminus / ion.n_e < 1e-10) {
        std::printf("  n_Hminus ~0: got=%.4e (ratio to n_e=%.2e) PASS\n",
                    ion.n_Hminus, ion.n_Hminus / ion.n_e);
    } else {
        std::printf("  n_Hminus ~0: FAIL (n_Hminus=%.4e too large vs n_e=%.4e)\n",
                    ion.n_Hminus, ion.n_e);
        failures++;
    }
}

void test_saha_partially_ionized() {
    std::printf("\n=== Saha: partially ionized (T=6000 K, rho=1e-7) ===\n");
    auto ion = grrt::solve_saha(1e-7, 6000.0);
    double n_e_full = 1e-7 * 1.7 / (2.0 * 1.672622e-24);
    std::printf("  n_e = %.4e (fully ionized would be %.4e)\n", ion.n_e, n_e_full);
    if (ion.n_e >= n_e_full * 0.5) {
        std::printf("  FAIL: n_e should be << fully ionized at T=6000K\n");
        failures++;
    } else {
        std::printf("  PASS: n_e is significantly below fully ionized\n");
    }
    std::printf("  mu = %.4f\n", ion.mu);
    if (ion.mu < 0.8) {
        std::printf("  FAIL: mu should be > 0.8 for partially ionized gas\n");
        failures++;
    } else {
        std::printf("  PASS: mu > 0.8\n");
    }
    std::printf("  n_Hminus = %.4e, n_HI = %.4e\n", ion.n_Hminus, ion.n_HI);
    if (ion.n_Hminus <= 0.0) {
        std::printf("  FAIL: H- should be nonzero at T=6000K\n");
        failures++;
    } else {
        std::printf("  PASS: H- is present\n");
    }
}

void test_saha_neutral() {
    std::printf("\n=== Saha: mostly neutral (T=3000 K, rho=1e-6) ===\n");
    auto ion = grrt::solve_saha(1e-6, 3000.0);
    check("mu ~1.3", ion.mu, 1.3, 0.15);
    double n_e_full = 1e-6 * 1.7 / (2.0 * 1.672622e-24);
    std::printf("  n_e = %.4e (fully ionized: %.4e, ratio: %.4e)\n",
                ion.n_e, n_e_full, ion.n_e / n_e_full);
}

void test_ff_opacity() {
    std::printf("\n=== Free-free opacity (T=1e6 K, fully ionized) ===\n");
    auto ion = grrt::solve_saha(1e-8, 1e6);
    double nu = 6.0e14; // ~500 nm
    double alpha = grrt::alpha_ff(nu, 1e6, ion);
    std::printf("  alpha_ff = %.4e cm^{-1}\n", alpha);
    if (alpha <= 0.0 || !std::isfinite(alpha)) {
        std::printf("  FAIL: alpha_ff should be positive and finite\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }
}

void test_hminus_opacity() {
    std::printf("\n=== H- opacity (T=6000 K, partial ionization) ===\n");
    auto ion = grrt::solve_saha(1e-7, 6000.0);
    double nu = 6.0e14; // ~500 nm
    double alpha_hm = grrt::alpha_hminus(nu, 6000.0, ion);
    std::printf("  alpha_Hminus = %.4e cm^{-1}\n", alpha_hm);
    double alpha_free = grrt::alpha_ff(nu, 6000.0, ion);
    std::printf("  alpha_ff = %.4e (H- should dominate)\n", alpha_free);
    if (alpha_hm <= alpha_free) {
        std::printf("  FAIL: H- should dominate over ff at T=6000K\n");
        failures++;
    } else {
        std::printf("  PASS: H- dominates\n");
    }
}

void test_bf_opacity() {
    std::printf("\n=== Bound-free ion opacity (T=3e4 K) ===\n");
    auto ion = grrt::solve_saha(1e-8, 3e4);
    double nu = 6.0e14;
    double alpha = grrt::alpha_bf_ion(nu, 3e4, ion);
    std::printf("  alpha_bf = %.4e cm^{-1}\n", alpha);
    if (alpha < 0.0 || !std::isfinite(alpha)) {
        std::printf("  FAIL: alpha_bf should be non-negative and finite\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }
}

void test_thomson() {
    std::printf("\n=== Thomson scattering (fully ionized) ===\n");
    auto ion = grrt::solve_saha(1e-8, 1e7);
    double kes = grrt::kappa_es(1e-8, ion);
    check("kappa_es ~0.34", kes, 0.34, 0.05);
}

void test_total_opacity() {
    std::printf("\n=== Total absorption opacity (T=5e4 K) ===\n");
    auto ion = grrt::solve_saha(1e-9, 5e4);
    double nu = 6.0e14;
    double kabs = grrt::kappa_abs(nu, 1e-9, 5e4, ion);
    std::printf("  kappa_abs(500nm, 1e-9, 5e4K) = %.4e cm^2/g\n", kabs);
    if (kabs <= 0.0 || !std::isfinite(kabs)) {
        std::printf("  FAIL\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }
}

void test_planck_nu() {
    std::printf("\n=== Planck function B_nu ===\n");
    double nu = 6.0e14;
    double T = 6000.0;
    double B = grrt::planck_nu(nu, T);
    std::printf("  B_nu(6e14 Hz, 6000K) = %.4e\n", B);
    if (B <= 0.0 || !std::isfinite(B)) {
        std::printf("  FAIL\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }
}

int main() {
    test_saha_fully_ionized();
    test_saha_partially_ionized();
    test_saha_neutral();
    test_ff_opacity();
    test_hminus_opacity();
    test_bf_opacity();
    test_thomson();
    test_total_opacity();
    test_planck_nu();

    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
