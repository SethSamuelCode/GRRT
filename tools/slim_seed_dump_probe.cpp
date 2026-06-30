// ===========================================================================
// SLIM GLOBAL-SEED DUMP + COLUMN-FEASIBILITY  (Task 1/2-gate — DIAGNOSTIC, DELETABLE)
// ---------------------------------------------------------------------------
// (1) Dumps the PRODUCTION global Sądowski slim seed build_slim_disk_seed(a,f_Edd)
//     RAW (no relax), to judge slim-physical vs torus.
// (2) THE GATE for the seed-pivot design: per node, runs solve_column_coupled (the
//     RELAX's actual column solver) on the slim seed's (Σ, T_c, shear) at the target
//     f_Edd, + the column Σ0 capacity ceiling at that node's geometry. Answers: can
//     relax_coupled START from this slim seed (columns feasible at the seed Σ)?
//
//     solve_ok per node  -> relax can start from the slim seed (wire Task 2)
//     widespread fail     -> seed Σ too high for the column even at f_Edd-0.9 capacity
//                            -> refine the seed's (Σ,V) onto the transonic branch first
//
// Build:  cmake --build build --config Release --target slim-seed-dump-probe
// Run:    build/Release/slim-seed-dump-probe.exe            (a=0.9 f_Edd=0.9 N=18 nz=24)
//         build/Release/slim-seed-dump-probe.exe 0.9 0.9 18 96
// REUSE: include-the-.cpp (opacity + disk_column_bvp + disk_column_coupled +
//        slim_disk_radial + slim_disk_coupled) — mirrors slim-coupled-hard-probe.
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
    const int    Nreq  = (argc > 3) ? std::atoi(argv[3]) : 18;
    const int    nz    = (argc > 4) ? std::atoi(argv[4]) : 24;

    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = a; in.alpha = 0.1; in.r_g = 1.48e6;
    in.r_out = 50.0; in.n_nodes = Nreq; in.tol = 1e-8;
    in.r_in = 0.5 * slim_detail::isco_prograde(in.mass, in.spin);

    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / 0.34;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    in.mdot = f_Edd * Mdot_Edd;

    const double isco = slim_detail::isco_prograde(in.mass, in.spin);
    const double ellK_isco = slim_detail::ell_kepler(in.mass, in.spin, isco);

    std::printf("# =====================================================================\n");
    std::printf("# slim-seed-dump-probe : RAW build_slim_disk_seed + column feasibility\n");
    std::printf("#   a=%.3f f_Edd=%.4g  mdot=%.4e g/s  N=%d  nz=%d  ISCO=%.4f\n",
                a, f_Edd, in.mdot, std::max(Nreq,4), nz, isco);
    std::printf("# =====================================================================\n\n");

    std::vector<double> U = build_slim_disk_seed(in, op);
    const int N = std::max(in.n_nodes, 4);
    const double ell_in = U[4 * N + 0];
    const double r_s    = U[4 * N + 1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);

    std::printf("    r_s=%.5f M (%.4f ISCO; inside=%s)  ell_in=%.6f (%.4f ellK_isco)\n\n",
                r_s, r_s/isco, (r_s<isco?"YES":"NO"), ell_in, ell_in/ellK_isco);

    std::vector<double> r(N), ell(N), Om(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N==1)?0.0:double(i)/double(N-1);
        r[i]   = std::exp(lr0 + (lr1-lr0)*t);
        ell[i] = U[4*i+2];
        Om[i]  = slim_detail::omega_from_ell(in.mass, in.spin, r[i], ell[i]);
    }

    // column Σ0 capacity ceiling at a node geometry (max over a T_eff×f_adv grid).
    auto sigma0_ceiling = [&](double sh, double oz_, double rmg) -> double {
        double best = -1.0;
        for (int it = 0; it <= 12; ++it) {
            const double Te = 1e5 * std::pow(1e3, it/12.0);   // 1e5..1e8
            for (double fa : {2.0, 0.0, -0.5, -0.9}) {
                ColumnInputs b{}; b.T_eff=Te; b.shear=std::max(sh,1e-300);
                b.omega_z=std::max(oz_,1e-300); b.alpha=in.alpha; b.f_adv=fa;
                b.rho_mid_guess=rmg; b.n_nodes=nz; b.max_iters=300; b.tol=1e-8;
                ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
                if (s.converged && s.Sigma0 > best) best = s.Sigma0;
            }
        }
        return best;
    };

    std::printf("    %-3s %-9s %-11s %-9s %-11s %-12s %-9s %-9s\n",
                "i", "r[M]", "Sig_seed", "Tc_seed", "Sig0_ceil", "Sg/ceil", "solve_ok", "f_adv");
    int n_feas=0, n_over=0;
    for (int i = 0; i < N; ++i) {
        const int j = (i+1<N)?i+1:i-1;
        const double sh  = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
        const double ozf = omega_perp_cgs(in, r[i]);
        const double Sg  = std::max(U[4*i+0], 1e2);
        const double Tc  = std::max(U[4*i+3], 1.0);
        const slim_detail::OneZoneState oz = slim_detail::one_zone_closure(Sg, Tc, r[i], in, op);
        const double rmg = std::max(oz.rho_mid, 1e-30);

        const double ceil = sigma0_ceiling(sh, ozf, rmg);
        if (ceil < Sg) ++n_over;

        ColumnCoupledInputs ci{};
        ci.Sigma_target=Sg; ci.Tc=Tc; ci.shear=std::max(sh,1e-300);
        ci.omega_z=std::max(ozf,1e-300); ci.alpha=in.alpha; ci.rho_mid_guess=rmg;
        ci.n_nodes=nz; ci.max_iters=300; ci.tol=1e-8; ci.Teff_guess=0.0;
        const ColumnClosure c = solve_column_coupled(ci, op, nullptr);
        if (c.converged) ++n_feas;

        std::printf("    %-3d %-9.4f %-11.4e %-9.3e %-11.4e %-12.2f %-9s %-+9.3e\n",
                    i, r[i], Sg, Tc, ceil, Sg/std::max(ceil,1.0),
                    c.converged?"Y":"N", c.converged?c.f_adv:std::nan(""));
    }

    std::printf("\n");
    std::printf("    => columns FEASIBLE at the slim seed (solve_column_coupled): %d/%d\n", n_feas, N);
    std::printf("       nodes with Σ_seed ABOVE the n_z=%d ceiling: %d/%d\n", nz, n_over, N);
    std::printf("    (n_z=%d undercounts Σ0 ~3× vs converged; capacity is a LOWER bound)\n", nz);
    std::printf("    VERDICT: %s\n",
        (n_feas >= N-1) ? "slim seed is COLUMN-FEASIBLE -> wire relax_coupled to seed from it (Task 2)"
        : (n_feas == 0) ? "slim seed columns INFEASIBLE -> refine seed (Σ,V) onto transonic branch first"
                        : "PARTIAL feasibility -> inspect which nodes fail (inner vs outer) before wiring");
    std::printf("DONE\n");
    return 0;
}
