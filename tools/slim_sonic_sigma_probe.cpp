// ===========================================================================
// SLIM SONIC-Σ vs COLUMN-CAPACITY  (rigorous convection-gate — DIAGNOSTIC, DELETABLE)
// ---------------------------------------------------------------------------
// Confirms (or refutes) that the pure-radiative column genuinely cannot hold the
// f_Edd=0.9 disk — using the column's OWN converged sound speed, not the raw seed.
//
// On the subsonic branch [r_s, r_out], |V| < c_s, so mass conservation Σ|V|=const
// forces Σ ≥ Σ_sonic (its MINIMUM, at the sonic point where |V|=c_s). Compute the
// self-consistent sonic point at r_s: sweep T_eff, solve the pure-radiative column,
// take its sonic sound speed c_s² = Γ̃₁·P/Σ (the EXACT radial-residual closure), and
// the mass-conservation Σ_req(c_s) = Ṁ/(2π r_s √Δ r_g c · (c_s/c)). The self-consistent
// sonic Σ is where Σ0_column(T_eff) = Σ_req. Compare to the column Σ0 capacity.
//
//   Σ_sonic  >  Σ_cap   -> even the THINNEST disk point needs more Σ than any
//                          pure-radiative column holds -> CONVECTION (#13) required.
//   Σ_sonic  <= Σ_cap   -> a pure-radiative sonic point exists; re-examine the seed.
//
// Build:  cmake --build build --config Release --target slim-sonic-sigma-probe
// Run:    build/Release/slim-sonic-sigma-probe.exe            (a=0.9 f_Edd=0.9 nz=96)
// REUSE: include-the-.cpp (opacity + disk_column_bvp + disk_column_coupled +
//        slim_disk_radial + slim_disk_coupled), mirrors the hard probe.
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"
#include "../src/disk_column_coupled.cpp"
#include "../src/slim_disk_radial.cpp"
#include "../src/slim_disk_coupled.cpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <numbers>
#include <algorithm>

using namespace grrt;
using namespace grrt::slim_coupled_detail;

int main(int argc, char** argv) {
    std::setbuf(stdout, nullptr);
    using namespace constants;
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    const double a     = (argc > 1) ? std::atof(argv[1]) : 0.9;
    const double f_Edd = (argc > 2) ? std::atof(argv[2]) : 0.9;
    const int    nz    = (argc > 3) ? std::atoi(argv[3]) : 96;

    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = a; in.alpha = 0.1; in.r_g = 1.48e6;
    in.r_out = 50.0; in.n_nodes = 18; in.tol = 1e-8;
    in.r_in = 0.5 * slim_detail::isco_prograde(in.mass, in.spin);

    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / 0.34;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    in.mdot = f_Edd * Mdot_Edd;

    // r_s + the inner node geometry from the slim seed.
    std::vector<double> U = build_slim_disk_seed(in, op);
    const int N = std::max(in.n_nodes, 4);
    const double r_s = U[4 * N + 1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);

    std::printf("# =====================================================================\n");
    std::printf("# slim-sonic-sigma-probe : Σ_sonic (column's own c_s) vs column capacity\n");
    std::printf("#   a=%.3f f_Edd=%.4g  mdot=%.4e g/s  r_s=%.4f  nz=%d\n",
                a, f_Edd, in.mdot, r_s, nz);
    std::printf("# =====================================================================\n\n");

    // Evaluate at r_s and 2 nodes just outside (the inner disk, where Σ is smallest).
    const double rtest[3] = { r_s, std::exp(lr0 + (lr1-lr0)*(1.0/(N-1))),
                                   std::exp(lr0 + (lr1-lr0)*(2.0/(N-1))) };
    for (int ridx = 0; ridx < 3; ++ridx) {
        const double r = rtest[ridx];
        const double ellK = slim_detail::ell_kepler(in.mass, in.spin, r);
        const double Om   = slim_detail::omega_from_ell(in.mass, in.spin, r, ellK);
        const int j = 1;  // neighbour for shear (use a small outward step)
        const double r_j = std::exp(std::log(r) + 0.05);
        const double Om_j = slim_detail::omega_from_ell(in.mass, in.spin, r_j,
                             slim_detail::ell_kepler(in.mass, in.spin, r_j)); (void)j;
        const double sh  = shear_cgs(in, r, Om, r_j, Om_j);
        const double ozf = omega_perp_cgs(in, r);
        const double sqrtD = std::sqrt(std::max(kerr_delta(in.mass, in.spin, r), 0.0));

        const double K = in.mdot / (2.0 * std::numbers::pi * sqrtD * in.r_g * c_cgs); // = Σ·|V|·Γ  [g/cm²]

        std::printf("### r = %.4f M   shear=%.3e  omega_z=%.3e  sqrtDelta=%.4f ###\n", r, sh, ozf, sqrtD);
        std::printf("    %-11s %-12s %-9s %-11s %-12s %-9s\n",
                    "T_eff[K]", "Sig0_col", "Tc_mid", "cs/c", "Sig_req", "col/req");
        double Sig0_cap = -1.0, sonic_Sig = -1.0, best_ratio = -1.0;
        for (int it = 0; it <= 18; ++it) {
            const double Te = 3e5 * std::pow(1e8/3e5, it/18.0);   // 3e5 .. 1e8
            ColumnInputs b{}; b.T_eff=Te; b.shear=std::max(sh,1e-300);
            b.omega_z=std::max(ozf,1e-300); b.alpha=in.alpha; b.f_adv=0.0;
            // rho_mid guess from one-zone at a nominal (Σ,Tc); refined internally.
            b.rho_mid_guess=1e-6; b.n_nodes=nz; b.max_iters=400; b.tol=1e-8;  // convective-column-friendly seed
            ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
            if (!s.converged) { std::printf("    %-11.3e %-12s\n", Te, "(fail)"); continue; }
            const double Sig0 = s.Sigma0;
            const double Tc   = s.T.empty() ? Te : s.T[0];   // midplane T (node 0)
            // Sonic sound speed via the EXACT radial closure: cs² = Γ̃₁·P/Σ.
            const slim_detail::OneZoneState oz = slim_detail::one_zone_closure(Sig0, Tc, r, in, op);
            const double beta = (oz.p_mid>0) ? oz.p_gas/oz.p_mid : 1.0;
            const double eta3 = 3.0 - 1.5*std::clamp(beta,0.0,1.0);
            const double gt1  = 1.0 + 1.0/eta3;
            const double PoverSig = (oz.P>0 && Sig0>0) ? oz.P/Sig0 : 0.0;     // [cm²/s²]
            const double cs = std::sqrt(std::max(gt1*PoverSig, 0.0));          // [cm/s]
            const double Vc = cs / c_cgs;                                      // |V|=c_s/c at sonic
            const double Sig_req = (Vc>0) ? K * std::sqrt(1.0 - Vc*Vc) / Vc : 0.0;
            const double ratio = (Sig_req>0) ? Sig0/Sig_req : 0.0;
            if (Sig0 > Sig0_cap) Sig0_cap = Sig0;
            if (ratio > best_ratio) best_ratio = ratio;
            // self-consistent sonic Σ ~ where ratio crosses 1
            if (sonic_Sig < 0 && ratio >= 1.0) sonic_Sig = Sig_req;
            std::printf("    %-11.3e %-12.4e %-9.3e %-11.4e %-12.4e %-9.3f\n",
                        Te, Sig0, Tc, Vc, Sig_req, ratio);
        }
        std::printf("    -> max column Σ0 capacity ≈ %.3e ; max(col/req)=%.3f\n", Sig0_cap, best_ratio);
        if (best_ratio >= 1.0)
            std::printf("    -> a pure-radiative sonic point EXISTS here (Σ_sonic ≈ %.3e ≤ capacity)\n", sonic_Sig);
        else
            std::printf("    -> NO pure-radiative sonic point: column Σ0 < Σ_req at ALL T_eff "
                        "(deficit %.1f×) -> CONVECTION (#13) required\n", 1.0/std::max(best_ratio,1e-30));
        std::printf("\n");
    }
    std::printf("DONE\n");
    return 0;
}
