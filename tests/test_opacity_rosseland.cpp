// Regression test for the temperature-adaptive Rosseland-mean opacity window.
//
// Background: build_opacity_luts() previously integrated the dB_nu/dT-weighted
// harmonic (Rosseland) mean over a HARDCODED [1e13, 1e16] Hz window that did not
// scale with temperature. The Rosseland weight peaks at nu_peak ~ 8e10*T Hz, so
// for hot disk midplane gas (T ~ 5e6-1e7 K, nu_peak ~ 4e17-8e17 Hz) the fixed
// window sat entirely on the Rayleigh-Jeans tail where free-free/bound-free
// kappa_nu ~ nu^-3 is LARGEST. The harmonic mean is then dominated by the most
// OPAQUE part of the spectrum, over-estimating kappa_R by ~7-8 orders of
// magnitude (measured ~3.8e9 cm^2/g at rho~12, T~5e6 vs physical ~few x 1e2).
//
// The fix integrates over the dimensionless x = h*nu/(k_B*T) on a fixed range,
// so the frequency window automatically tracks the Planck peak at every T and
// captures the transparent high-nu continuum that should dominate the mean.
//
// These assertions are physical-bound / scaling-trend checks (order of
// magnitude), NOT exact value matches.

#include "grrt/color/opacity.h"
#include <cstdio>
#include <cmath>

static int failures = 0;

static void expect(const char* name, bool pass, const char* detail) {
    std::printf("  %s: %s  [%s]\n", name, pass ? "PASS" : "FAIL", detail);
    if (!pass) failures++;
}

// Analytic Kramers bound-free opacity (cm^2/g), order-of-magnitude reference for
// the free-free/bound-free regime: kappa_bf ~ 4e24 * rho * T^-3.5 (Z=0.02-ish,
// Gaunt ~ unity). Used only for self-documenting comparison and trend checks.
static double kramers_bf(double rho, double T) {
    return 4.0e24 * rho * std::pow(T, -3.5);
}

int main() {
    std::printf("=== Rosseland-mean T-adaptive window regression ===\n");

    // Build LUTs over the disk range the slim-disk test uses.
    auto luts = grrt::build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    // --- (1) High-T physical bound (the bug) ----------------------------------
    std::printf("\n-- (1) High-T physical bound --\n");
    {
        double kR = luts.lookup_kappa_ross(1.0, 1e7);
        double ref = kramers_bf(1.0, 1e7);   // ~1.3 cm^2/g
        char d[160];
        std::snprintf(d, sizeof d,
            "kappa_R(rho=1, T=1e7)=%.4e cm^2/g, Kramers ref=%.4e, was ~1e9 broken",
            kR, ref);
        expect("kappa_R(1, 1e7) < 100 (finite, positive)",
               std::isfinite(kR) && kR > 0.0 && kR < 100.0, d);
    }
    {
        double kR = luts.lookup_kappa_ross(12.0, 5e6);
        double ref = kramers_bf(12.0, 5e6);  // few x 1e2 cm^2/g
        char d[160];
        std::snprintf(d, sizeof d,
            "kappa_R(rho=12, T=5e6)=%.4e cm^2/g, Kramers ref=%.4e, was ~3.8e9 broken",
            kR, ref);
        expect("kappa_R(12, 5e6) < 1e4 (finite, positive)",
               std::isfinite(kR) && kR > 0.0 && kR < 1e4, d);
    }

    // --- (2) Kramers scaling trend kappa_R ~ T^-3.5 at fixed rho --------------
    std::printf("\n-- (2) Kramers T^-3.5 scaling trend --\n");
    {
        const double rho = 1.0;
        const double T1 = 3e6, T2 = 1e7;
        double k1 = luts.lookup_kappa_ross(rho, T1);
        double k2 = luts.lookup_kappa_ross(rho, T2);
        double ratio = k1 / k2;
        double expected = std::pow(T1 / T2, -3.5);  // (0.3)^-3.5 ~ 67.6
        double fold = (ratio > expected) ? ratio / expected : expected / ratio;
        char d[200];
        std::snprintf(d, sizeof d,
            "kappa_R(%.0e)=%.4e kappa_R(%.0e)=%.4e ratio=%.3e expected T^-3.5=%.3e fold=%.2f",
            T1, k1, T2, k2, ratio, expected, fold);
        // Generous 3x tolerance: bound-free edges + Gaunt detail are not pure
        // power law. This is a trend/order-of-magnitude check.
        expect("ratio matches (T1/T2)^-3.5 within 3x",
               std::isfinite(ratio) && ratio > 0.0 && fold < 3.0, d);
    }

    // --- (3) No regression at cool T (old window was already correct here) ----
    std::printf("\n-- (3) Cool-T no-regression --\n");
    {
        double kR = luts.lookup_kappa_ross(1e-8, 3e4);
        double ref = kramers_bf(1e-8, 3e4);  // ~6.8e-2 cm^2/g (bf only)
        char d[180];
        std::snprintf(d, sizeof d,
            "kappa_R(rho=1e-8, T=3e4)=%.4e cm^2/g, Kramers bf ref=%.4e (H- also active)",
            kR, ref);
        expect("kappa_R(1e-8, 3e4) finite, 1e-6 < kR < 1e6",
               std::isfinite(kR) && kR > 1e-6 && kR < 1e6, d);
    }

    // --- (4) The slim-disk unlock value ---------------------------------------
    std::printf("\n-- (4) Slim-disk unlock probe --\n");
    {
        double kR = luts.lookup_kappa_ross(12.0, 5e6);
        std::printf("  kappa_R(12, 5e6) = %.4e cm^2/g "
                    "(was ~3.8e9 blocking slim disk; now should be ~1e2)\n", kR);
    }

    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
