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

void test_lut_construction() {
    std::printf("\n=== Opacity LUT construction ===\n");
    auto luts = grrt::build_opacity_luts(1e-12, 1e-4, 3000.0, 1e8);

    std::printf("  kappa_abs LUT size: %d x %d x %d = %zu entries\n",
                luts.n_nu, luts.n_rho, luts.n_T, luts.kappa_abs_lut.size());

    // Test interpolation at known point
    double kabs = luts.lookup_kappa_abs(6e14, 1e-8, 1e6);
    std::printf("  kappa_abs(6e14, 1e-8, 1e6) = %.4e\n", kabs);
    if (kabs <= 0.0 || !std::isfinite(kabs)) {
        std::printf("  FAIL: should be positive\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }

    double kes = luts.lookup_kappa_es(1e-8, 1e7);
    check("kappa_es LUT ~0.34", kes, 0.34, 0.10);

    double mu = luts.lookup_mu(1e-8, 1e7);
    check("mu LUT ~0.6", mu, 0.6, 0.10);

    double kr = luts.lookup_kappa_ross(1e-8, 1e6);
    std::printf("  kappa_ross(1e-8, 1e6) = %.4e\n", kr);
    if (kr <= 0.0 || !std::isfinite(kr)) {
        std::printf("  FAIL\n");
        failures++;
    } else {
        std::printf("  PASS\n");
    }
}

static int test_kappa_ross_gradients() {
    std::printf("\n=== kappa_ross_with_grad: log-derivatives ===\n");
    int fails = 0;
    auto lut = grrt::build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    const double rho = 1e-3, T = 1e5;   // a smooth interior point

    double kR, dlnrho, dlnT;
    lut.kappa_ross_with_grad(rho, T, kR, dlnrho, dlnT);

    // kR must match the plain lookup at the same point.
    const double kR_plain = lut.lookup_kappa_ross(rho, T);
    std::printf("  kR=%.4e (plain %.4e)\n", kR, kR_plain);
    if (std::abs(kR - kR_plain) > 1e-9 * std::max(kR_plain, 1e-30)) {
        std::printf("  FAIL: kR disagrees with lookup_kappa_ross\n"); fails++;
    }

    // Independent central difference (different step) must match the supplier's
    // gradients — catches swapped variables / wrong denominator.
    const double h = 0.02;
    const double ref_dlnrho =
        (lut.lookup_kappa_ross(rho * std::exp(h), T)
       - lut.lookup_kappa_ross(rho * std::exp(-h), T)) / (2.0 * h);
    const double ref_dlnT =
        (lut.lookup_kappa_ross(rho, T * std::exp(h))
       - lut.lookup_kappa_ross(rho, T * std::exp(-h))) / (2.0 * h);
    std::printf("  d/dlnrho=%.4e (ref %.4e)  d/dlnT=%.4e (ref %.4e)\n",
                dlnrho, ref_dlnrho, dlnT, ref_dlnT);
    if (std::abs(dlnrho - ref_dlnrho) > 0.10 * std::max(std::abs(ref_dlnrho), 1e-12)
        && std::abs(dlnrho - ref_dlnrho) > 1e-6) {
        std::printf("  FAIL: d/dlnrho mismatch\n"); fails++;
    }
    if (std::abs(dlnT - ref_dlnT) > 0.10 * std::max(std::abs(ref_dlnT), 1e-12)
        && std::abs(dlnT - ref_dlnT) > 1e-6) {
        std::printf("  FAIL: d/dlnT mismatch\n"); fails++;
    }
    if (!std::isfinite(dlnrho) || !std::isfinite(dlnT)) {
        std::printf("  FAIL: non-finite gradient\n"); fails++;
    }
    // Edge robustness: at the table's upper rho edge the one-sided stencil must
    // still return a finite slope (the old centered difference would straddle a
    // clamped lookup here).
    const double rho_hi = std::pow(10.0, lut.log_rho_max);
    double kEr, dErho, dErT;
    lut.kappa_ross_with_grad(rho_hi, T, kEr, dErho, dErT);
    std::printf("  edge rho=%.3e: d/dlnrho=%.4e d/dlnT=%.4e\n", rho_hi, dErho, dErT);
    if (!std::isfinite(dErho) || !std::isfinite(dErT) || !std::isfinite(kEr)) {
        std::printf("  FAIL: non-finite gradient at table edge\n"); fails++;
    }
    std::printf("  %s\n", fails == 0 ? "PASS" : "FAIL");
    return fails;
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
    test_lut_construction();
    failures += test_kappa_ross_gradients();

    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
