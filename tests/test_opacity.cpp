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
    check("n_Hminus ~0", ion.n_Hminus, 0.0, 1.0);
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

int main() {
    test_saha_fully_ionized();
    test_saha_partially_ionized();
    test_saha_neutral();

    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
