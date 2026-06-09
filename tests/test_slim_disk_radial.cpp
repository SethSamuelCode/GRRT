#include "grrt/scene/slim_disk_radial.h"
#include "grrt/color/opacity.h"
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
    auto lut = grrt::build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    auto sol = grrt::solve_slim_disk_radial(in, lut);
    std::printf("  converged=%d iters=%d final_residual=%.3e\n",
                sol.converged, sol.iters, sol.final_residual);
    // Contract: the call links, returns, and honours the no-fabrication invariant —
    // a converged result must carry a full profile; a non-converged one must be empty.
    if (sol.converged) {
        if (sol.r.size() != (size_t)in.n_nodes) {
            std::printf("  FAIL: converged but profile not unpacked\n"); failures++;
        }
    } else {
        if (!sol.r.empty()) { std::printf("  FAIL: non-converged but profile non-empty\n"); failures++; }
    }
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

// Prograde Kerr ISCO (Bardeen, Press & Teukolsky 1972), M=1.
static double isco_prograde(double a) {
    const double Z1 = 1.0 + std::cbrt(1.0 - a*a) * (std::cbrt(1.0 + a) + std::cbrt(1.0 - a));
    const double Z2 = std::sqrt(3.0*a*a + Z1*Z1);
    return 3.0 + Z2 - std::sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0*Z2));
}
// Kerr outer horizon r+ = M + sqrt(M^2 - a^2), M=1.
static double horizon_plus(double a) { return 1.0 + std::sqrt(std::max(1.0 - a*a, 0.0)); }

// Mid-Ṁ transonic relaxation at a given spin: solve, assert convergence + sanity.
static void solve_one_spin(double a) {
    std::printf("\n--- a=%.3f ---\n", a);
    grrt::SlimDiskInputs in{};
    in.mass = 1.0;
    in.spin = a;
    in.alpha = 0.1;
    in.r_g = 1.48e6;                 // ~10 M_sun
    // r_in just OUTSIDE the prograde photon orbit r_ph (the inner limit of circular
    // timelike orbits — inside it ℓ_K diverges).  r_ph = 2M[1+cos((2/3)arccos(-a))];
    // a=0.9 -> 1.558, a=0.998 -> 1.074. The transonic flow's sonic point still sits
    // between r_in and the ISCO. (horizon r+ for a=0.998 ~1.063, a=0.9 ~1.436.)
    const double r_ph = 2.0 * (1.0 + std::cos((2.0/3.0) * std::acos(-a)));
    in.r_in = r_ph + 0.02;
    in.r_out = 50.0;
    in.n_nodes = 150;
    in.max_iters = 100;
    in.tol = 1e-6;
    // Mid Ṁ: f_Edd ≈ 0.3.  L_Edd ≈ 1.26e39 erg/s for 10 M_sun; with η≈0.1,
    // Ṁ_Edd = L_Edd/(η c²) ≈ 1.4e18 g/s, so 0.3 Ṁ_Edd ≈ 4e17 g/s.
    in.mdot = 4.0e17;

    auto lut = grrt::build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);
    auto s = grrt::solve_slim_disk_radial(in, lut);

    const double r_isco = isco_prograde(a);
    std::printf("  horizon=%.4f r_in=%.4f r_isco=%.4f | converged=%d iters=%d final_residual=%.3e r_sonic=%.4f\n",
                horizon_plus(a), in.r_in, r_isco, s.converged, s.iters, s.final_residual, s.r_sonic);

    if (!s.converged) {
        std::printf("  FAIL: a=%.3f did not converge\n", a);
        failures++;
        return;
    }
    if (!(s.final_residual < in.tol)) {
        std::printf("  FAIL: final_residual %.3e !< tol %.3e\n", s.final_residual, in.tol);
        failures++;
    }
    // Sanity: V<0 (inflow) everywhere, Σ>0, sonic point inside the ISCO.
    bool all_inflow = true, all_pos = true;
    double f_adv_lo = 1e300, f_adv_hi = -1e300;
    for (size_t i = 0; i < s.r.size(); ++i) {
        if (!(s.V[i] < 0.0)) all_inflow = false;
        if (!(s.Sigma[i] > 0.0)) all_pos = false;
        f_adv_lo = std::min(f_adv_lo, s.f_adv[i]);
        f_adv_hi = std::max(f_adv_hi, s.f_adv[i]);
    }
    std::printf("  V<0 everywhere=%d  Sigma>0 everywhere=%d  f_adv in [%.3e, %.3e]\n",
                all_inflow, all_pos, f_adv_lo, f_adv_hi);
    if (!all_inflow) { std::printf("  FAIL: V not <0 everywhere\n"); failures++; }
    if (!all_pos)    { std::printf("  FAIL: Sigma not >0 everywhere\n"); failures++; }
    if (!(s.r_sonic < r_isco)) {
        std::printf("  FAIL: r_sonic %.4f not inside ISCO %.4f\n", s.r_sonic, r_isco);
        failures++;
    }
    // Mid-Ṁ: advection is present but sub-dominant -> small-but-positive f_adv.
    if (!(f_adv_hi > 0.0)) {
        std::printf("  FAIL: f_adv never positive (expected small-but-positive at mid Ṁ)\n");
        failures++;
    }
}

static void test_converges_midmdot() {
    std::printf("\n=== transonic relaxation converges at mid-Ṁ (a=0.9 and a=0.998) ===\n");
    solve_one_spin(0.9);
    solve_one_spin(0.998);
}

int main() {
    test_kerr_factors();
    test_links_and_returns();
    test_one_zone_closure();
    test_one_zone_radiation_dominated();
    test_radial_residual_finite();
    test_converges_midmdot();
    std::printf("\n=== %d failures ===\n", failures);
    return failures > 0 ? 1 : 0;
}
