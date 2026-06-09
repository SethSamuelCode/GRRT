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

static void test_one_zone_closure() {
    std::printf("\n=== one-zone vertical closure ===\n");
    using namespace grrt::constants;
    grrt::SlimDiskInputs in{};
    in.mass=1.0; in.spin=0.9; in.alpha=0.1; in.r_g=1.48e6;
    auto lut = grrt::build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    const double r=10.0;
    // Pick a gas-dominated point: high Sigma, moderate T_c so p_rad << p_gas.
    const double Sigma=1e5, Tc=3e4;
    auto s = grrt::slim_detail::one_zone_closure(Sigma, Tc, r, in, lut);
    std::printf("  H=%.3e rho_mid=%.3e c_s=%.3e p_mid=%.3e P=%.3e S=%.3e mu=%.3f p_rad/p_gas=%.2e\n",
                s.H, s.rho_mid, s.c_s, s.p_mid, s.P, s.S, s.mu, s.p_rad/s.p_gas);
    // Structural identities (hold in ANY regime, by construction):
    const double Omega_perp = std::sqrt(grrt::slim_detail::omega_perp2(1.0,0.9,r))*c_cgs/in.r_g;
    check("H = c_s/Omega_perp",     s.H,       s.c_s/Omega_perp, 1e-9);
    check("rho_mid = Sigma/(2H)",   s.rho_mid, Sigma/(2.0*s.H),  1e-12);
    check("p_mid = rho_mid c_s^2",  s.p_mid,   s.rho_mid*s.c_s*s.c_s, 1e-9);
    check("P = 2 p_mid H",          s.P,       2.0*s.p_mid*s.H,  1e-12);
    check("p_mid = p_gas + p_rad",  s.p_mid,   s.p_gas+s.p_rad,  1e-12);
    // Gas-dominated regime check: radiation subdominant, H ~ c_s_gas/Omega_perp.
    if (!(s.p_rad/s.p_gas < 0.05)) { std::printf("  FAIL: chosen point not gas-dominated (raise Sigma / lower Tc)\n"); failures++; }
    const double cs_gas = std::sqrt(k_B*Tc/(s.mu*m_p));
    check("gas-limit H ~ c_s_gas/Omega_perp", s.H, cs_gas/Omega_perp, 0.05);  // ~5% (radiation slightly thickens)
}

static void test_one_zone_radiation_dominated() {
    std::printf("\n=== one-zone closure: radiation-dominated (exercises the b-term) ===\n");
    using namespace grrt::constants;
    grrt::SlimDiskInputs in{};
    in.mass=1.0; in.spin=0.9; in.alpha=0.1; in.r_g=1.48e6;
    auto lut = grrt::build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    const double r=10.0, Sigma=1e2, Tc=1e7;   // hot + low Sigma -> radiation-dominated
    auto s = grrt::slim_detail::one_zone_closure(Sigma, Tc, r, in, lut);
    const double Omega_perp = std::sqrt(grrt::slim_detail::omega_perp2(1.0,0.9,r))*c_cgs/in.r_g;
    const double cs_gas = std::sqrt(k_B*Tc/(s.mu*m_p));
    const double H_gas_only = cs_gas/Omega_perp;
    std::printf("  H=%.3e H_gas_only=%.3e ratio=%.2f p_rad/p_gas=%.2e\n",
                s.H, H_gas_only, s.H/H_gas_only, s.p_rad/s.p_gas);
    // Structural identities still hold (regime-independent):
    check("H = c_s/Omega_perp",    s.H,     s.c_s/Omega_perp,      1e-9);
    check("p_mid = rho_mid c_s^2", s.p_mid, s.rho_mid*s.c_s*s.c_s, 1e-9);
    check("P = 2 p_mid H",         s.P,     2.0*s.p_mid*s.H,       1e-12);
    // Radiation-dominated: p_rad >> p_gas, and radiation THICKENS the column.
    if (!(s.p_rad/s.p_gas > 10.0)) { std::printf("  FAIL: not radiation-dominated (p_rad/p_gas=%.2e)\n", s.p_rad/s.p_gas); failures++; }
    if (!(s.H > 1.5*H_gas_only))   { std::printf("  FAIL: radiation did not thicken (ratio=%.2f)\n", s.H/H_gas_only); failures++; }
    else { std::printf("  PASS: radiation thickens (H/H_gas_only=%.2f)\n", s.H/H_gas_only); }
}

static void test_radial_residual_finite() {
    std::printf("\n=== transonic radial residual: length 4N+2 and all finite ===\n");
    grrt::SlimDiskInputs in{};
    const int N = 64;
    in.mass = 1.0;
    in.spin = 0.998;            // near-extremal render spin
    in.alpha = 0.1;
    in.r_g = 1.48e6;            // ~10 M_sun
    in.r_in = 1.3;             // inside the ISCO (transonic; horizon r+ ≈ 1.063)
    in.r_out = 50.0;
    in.n_nodes = N;
    // Near-Eddington Mdot: Ṁ_Edd ≈ 10 L_Edd/c² ~ 1.4e18 g/s for 10 M_sun; use ~Eddington.
    in.mdot = 1.0e18;

    auto lut = grrt::build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    std::vector<double> U = grrt::build_thin_disk_seed(in, lut);
    std::printf("  seed length: %zu (expected %d)\n", U.size(), 4 * N + 2);
    if (U.size() != (size_t)(4 * N + 2)) { std::printf("  FAIL: seed length\n"); failures++; }

    std::vector<double> R;
    grrt::slim_radial_residual(U, in, lut, R);
    std::printf("  R.size()=%zu (expected %d)\n", R.size(), 4 * N + 2);
    if (R.size() != (size_t)(4 * N + 2)) { std::printf("  FAIL: R.size != 4N+2\n"); failures++; }

    int nonfinite = 0;
    for (size_t i = 0; i < R.size(); ++i) {
        if (!std::isfinite(R[i])) {
            if (nonfinite < 8) std::printf("  non-finite R[%zu]=%g\n", i, R[i]);
            nonfinite++;
        }
    }
    std::printf("  non-finite entries: %d\n", nonfinite);
    if (nonfinite > 0) { std::printf("  FAIL: residual has non-finite entries\n"); failures++; }
    else std::printf("  PASS: all %zu residual entries finite\n", R.size());
}

int main() {
    test_kerr_factors();
    test_links_and_returns();
    test_one_zone_closure();
    test_one_zone_radiation_dominated();
    test_radial_residual_finite();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
