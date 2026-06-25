// ===========================================================================
// SLIM η₄ MOMENT PROBE  (gate for the S11 density 2nd moment — DELETABLE)
// ---------------------------------------------------------------------------
// QUESTION: column_moments() must compute η₄ ≡ (1/Σ)∫ρz²dz, the density second
// moment about the midplane (S11). Over the STORED HALF-PROFILE (z ∈ [0,z0],
// midplane→photosphere) this is convention-free:
//     η₄ = (∫₀^h ρ z² dz) / (∫₀^h ρ dz)        (= density-weighted ⟨z²⟩)
// the both-faces factor in the numerator and the Σ=2∫ρdz factor in the
// denominator cancel, so no Σ0 convention is needed — just a ratio of two
// half-integrals. η₄ carries NO Ω_⊥² (that multiplies η₄ later, Task 8).
//
// One-zone reductions (the GATE):
//   * Uniform ρ over [0,h]:                    η₄ = h²/3
//       (∫ρz² = ρh³/3, ∫ρ = ρh ⇒ ratio = h²/3).
//   * Gaussian ρ ∝ exp(−z²/(2H²)) over [0,∞):  η₄ = H².
//
// TEST: build analytic synthetic columns, compute η₄ via column_moments, and
// check it against (a) an INLINE reference trapezoidal quadrature (proves the
// function computes exactly what the probe computes, ~1e-12) and (b) the closed
// forms above (the physics, looser trapezoidal tol on the z²-weighted endpoint).
//
// Build: cmake --build build --config Release --target slim-eta4-moment-probe
// Run:   build/Release/slim-eta4-moment-probe.exe
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"
#include "../src/disk_column_coupled.cpp"

#include <cstdio>
#include <cmath>
#include <vector>

using namespace grrt;

int failures = 0;

// Inline independent reference: η₄_ref = (Σ ρ_i z_i² Δz)/(Σ ρ_i Δz), trapezoidal
// on the SAME half-grid column_moments integrates over. Used to prove the function
// computes what the probe computes (to ~1e-12) independent of the closed form.
static double eta4_reference(const ColumnBVPSolution& s) {
    double m2 = 0.0, m0 = 0.0;
    for (size_t i = 0; i + 1 < s.z.size(); ++i) {
        const double dz = s.z[i+1] - s.z[i];
        m2 += 0.5 * (s.rho[i]*s.z[i]*s.z[i] + s.rho[i+1]*s.z[i+1]*s.z[i+1]) * dz;
        m0 += 0.5 * (s.rho[i] + s.rho[i+1]) * dz;
    }
    return (m0 > 0.0) ? (m2 / m0) : 0.0;
}

// Fill a synthetic half-profile with the given z-grid and density; P/P_gas/T are
// set to harmless positive constants (η₄ ignores them).
static ColumnBVPSolution make_column(const std::vector<double>& z,
                                     const std::vector<double>& rho) {
    ColumnBVPSolution s;
    const size_t N = z.size();
    s.z = z;
    s.rho = rho;
    s.P.assign(N, 1.0);
    s.P_gas.assign(N, 1.0);
    s.T.assign(N, 1.0);
    return s;
}

// Run one case: compute η₄ via column_moments, assert it matches the inline
// reference (~1e-12) and the closed form (phys_tol).
static void run_case(const char* name, const ColumnBVPSolution& s,
                     double closed_form, double phys_tol) {
    double eta3 = 0.0, eta4 = 0.0;
    column_moments(s, eta3, eta4);

    const double ref = eta4_reference(s);
    const double rel_fn  = std::abs(eta4 - ref) / std::max(std::abs(ref), 1e-300);
    const double rel_cf  = std::abs(eta4 - closed_form) / std::max(std::abs(closed_form), 1e-300);

    const bool ok_fn = (rel_fn <= 1e-12);
    const bool ok_cf = (rel_cf <= phys_tol);

    std::printf("---- %s ----\n", name);
    std::printf("  column_moments eta4 = %.12e\n", eta4);
    std::printf("  inline reference    = %.12e   (rel diff = %.3e, tol 1e-12)\n", ref, rel_fn);
    std::printf("  closed form         = %.12e   (rel diff = %.3e, tol %.1e)\n",
                closed_form, rel_cf, phys_tol);
    std::printf("  function-vs-reference: %s\n", ok_fn ? "PASS" : "FAIL");
    std::printf("  reference-vs-closed-form: %s\n\n", ok_cf ? "PASS" : "FAIL");

    if (!ok_fn) failures++;
    if (!ok_cf) failures++;
}

int main() {
    std::printf("# slim-eta4-moment-probe : gate eta4 = (1/Sigma)INT rho z^2 dz\n");
    std::printf("# uniform -> h^2/3 ; Gaussian -> H^2 ; function == inline reference\n\n");

    // ---- 1. Uniform column: rho const over [0,h], η₄ = h²/3 ------------------
    {
        const int N = 512;
        const double h = 1.0;
        std::vector<double> z(N), rho(N);
        for (int i = 0; i < N; ++i) {
            z[i]   = i * h / (N - 1);
            rho[i] = 1.0;                       // constant density
        }
        ColumnBVPSolution s = make_column(z, rho);
        run_case("uniform column (rho=const)", s, h*h/3.0, 1e-3);
    }

    // ---- 2. Gaussian column: rho ∝ exp(−z²/(2H²)) over [0,6H], η₄ = H² -------
    {
        const int N = 512;
        const double H = 1.0;
        const double zmax = 6.0 * H;
        std::vector<double> z(N), rho(N);
        for (int i = 0; i < N; ++i) {
            z[i]   = i * zmax / (N - 1);
            rho[i] = std::exp(-z[i]*z[i] / (2.0 * H*H));
        }
        ColumnBVPSolution s = make_column(z, rho);
        run_case("gaussian column (rho ~ exp(-z^2/2H^2))", s, H*H, 1e-3);
    }

    std::printf("## %d failure(s) ##\n", failures);
    return failures ? 1 : 0;
}
