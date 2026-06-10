// TEMPORARY smoke check for the safety-budget addition:
//   (1) the a=0 f_Edd≈0.02 corner (the PROVEN-CONVERGENT anchor) still converges
//       with the budget installed — the budget must not break the working path;
//   (2) the budget actually TRIPS and returns the honest empty fallback when given
//       an absurdly small cap (proves the abort path + message + no fabrication).
// Safe to delete; registered in CMakeLists as slim-budget-smoke.
#include "grrt/scene/slim_disk_radial.h"
#include "grrt/color/opacity.h"
#include <cstdio>
#include <cmath>

int main() {
    using namespace grrt;
    using namespace grrt::constants;
    auto lut = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    // a=0, f_Edd≈0.02 anchor at small N (the proven corner; no spin walk since a<0.05).
    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = 0.0; in.alpha = 0.1;
    in.r_g = 1.48e6; in.r_in = 2.1; in.r_out = 50.0;
    in.n_nodes = 48; in.max_iters = 100; in.tol = 1e-6;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double L_Edd = 4.0 * 3.14159265358979323846 * G_cgs * M_cgs * c_cgs / 0.34;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    in.mdot = 0.02 * Mdot_Edd;

    std::printf("[A] a=0 f_Edd=0.02 anchor (budget installed, generous default)\n");
    auto s = solve_slim_disk_radial(in, lut);
    std::printf("[A] converged=%d r_sonic=%.4f final_residual=%.3e\n",
                s.converged, s.r_sonic, s.final_residual);
    const bool A_ok = s.converged && !s.r.empty();
    std::printf("[A] -> %s\n", A_ok ? "PASS (still converges)" : "FAIL");

    // Same corner, but with an absurd 5-inner-iter cap: must trip -> honest empty.
    SlimDiskInputs in2 = in;
    in2.budget_inner_iter_cap = 5;
    std::printf("\n[B] same corner, inner_iter_cap=5 (must trip honestly)\n");
    auto s2 = solve_slim_disk_radial(in2, lut);
    std::printf("[B] converged=%d r.size=%zu\n", s2.converged, s2.r.size());
    const bool B_ok = (!s2.converged) && s2.r.empty();   // honest empty fallback
    std::printf("[B] -> %s\n", B_ok ? "PASS (honest fallback, no fabrication)" : "FAIL");

    return (A_ok && B_ok) ? 0 : 1;
}
