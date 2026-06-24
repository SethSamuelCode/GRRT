#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1
#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"
#include "../src/disk_column_coupled.cpp"
#include <cstdio>
#include <cmath>
using namespace grrt;
int failures = 0;

static void test_coupled_repose_roundtrip() {
    std::printf("\n=== C1: Sigma+Tc-driven column recovers Teff-driven root ===\n");
    auto lut = build_opacity_luts(1e-12, 1e6, 3000.0, 1e8);
    ColumnInputs ref{}; ref.T_eff=3e5; ref.shear=2e3; ref.omega_z=2e3;
    ref.alpha=0.1; ref.rho_mid_guess=1.0; ref.n_nodes=96; ref.max_iters=300; ref.tol=1e-8;
    auto s = solve_column_bvp(ref, lut);
    if (!s.converged) { std::printf("  FAIL: reference column did not converge\n"); failures++; return; }
    const double Sigma_target = s.Sigma0, Tc_mid = s.T.front();
    ColumnCoupledInputs ci{}; ci.Sigma_target=Sigma_target; ci.Tc=Tc_mid; ci.f_adv=0.0;
    ci.shear=2e3; ci.omega_z=2e3; ci.alpha=0.1; ci.rho_mid_guess=1.0;
    ci.n_nodes=96; ci.max_iters=300; ci.tol=1e-8;
    ColumnClosure c = solve_column_coupled(ci, lut, nullptr);
    const double F_expect = grrt::constants::sigma_SB*std::pow(ref.T_eff,4);
    const double relF = std::abs(c.F - F_expect)/F_expect;
    const double relz = std::abs(c.z0 - s.z0)/s.z0;
    std::printf("  conv=%d F=%.4e (expect %.4e rel=%.2e)  z0=%.4e (ref %.4e rel=%.2e) Teff=%.4e (ref 3e5)\n",
                c.converged, c.F, F_expect, relF, c.z0, s.z0, relz, c.T_eff);
    if (!c.converged || relF>1e-3 || relz>1e-3) { std::printf("  FAIL\n"); failures++; }
}
int main(){ test_coupled_repose_roundtrip(); std::printf("\n## %d failure(s) ##\n", failures); return failures?1:0; }
