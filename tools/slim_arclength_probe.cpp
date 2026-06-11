// ===========================================================================
// PSEUDO-ARCLENGTH CONTINUATION PROBE  (Task 4 driver + Task 2 tangent check)
// ---------------------------------------------------------------------------
// Drives solve_slim_disk_arclength to trace the slim-disk branch THROUGH the
// f_Edd≈0.11 turning point that simple Ṁ-marching cannot cross.  Reports:
//   • the Task-2 tangent at a CONVERGED sub-fold anchor (a=0.9, f_Edd=0.10):
//     ‖J·U̇+R_Mdot·Ṁ̇‖≈0 and the Ṁ̇ component;
//   • the full traced (Ṁ,Σ) branch — every accepted point, with the Ṁ̇ sign so the
//     FOLD (sign flip) is visible;
//   • the GATE: does the trace cross f_Edd=0.11? how high does it reach?
//   • the physics at the highest f_Edd (H/r, β, f_adv);
//   • then the same at a=0.998.
//
// #includes slim_disk_radial.cpp + opacity.cpp directly (the probe/test pattern)
// so it reaches the internal helpers (slim_arclength_tangent, etc.).
//
// Build:
//   cmake --build build --config Release --target slim-arclength-probe
//   build/Release/slim-arclength-probe.exe
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/slim_disk_radial.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

using namespace grrt;
using namespace grrt::slim_detail;

static SlimDiskInputs make_inputs(double a, double f_Edd, int N, double wall_s) {
    using namespace constants;
    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = a; in.alpha = 0.1; in.r_g = 1.48e6;
    in.r_out = 50.0; in.n_nodes = N; in.max_iters = 800; in.tol = 1e-6;
    in.budget_wall_seconds = wall_s;
    const double r_ph = 2.0 * (1.0 + std::cos((2.0/3.0) * std::acos(-a)));
    in.r_in = r_ph + 0.02;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    in.mdot = f_Edd * Mdot_Edd;
    return in;
}

// Task-2 tangent check at a CONVERGED anchor: converge (a,0.10) the DIRECT way
// (cold-seed + solve_single_am — the same in-basin path the driver uses at a<=0.9),
// then build the tangent and report the null residual + Ṁ̇ at the converged point.
static void tangent_check_at_anchor(const OpacityLUTs& op, double a) {
    using namespace constants;
    const int N = 48;
    SlimDiskInputs in = make_inputs(a, 0.10, N, 600.0);
    in.max_iters = 800;
    const int n = 4*N+2, m = n+1;
    std::printf("\n=== Task-2 tangent at converged anchor (a=%.3f, f_Edd=0.10) ===\n", a);
    std::vector<double> U = build_thin_disk_seed(in, op);
    SolveBudget budget; budget.wall_cap_s = 600.0; g_budget = &budget;
    const bool conv = solve_single_am(in, op, U, /*require_N1=*/false);
    g_budget = nullptr;
    if (!conv) { std::printf("  anchor did not converge -> skip\n"); return; }
    std::vector<double> t_prev(m,0.0); t_prev[n]=1.0;
    std::vector<double> t;
    const bool ok = slim_arclength_tangent(U, in, op, t_prev, t);
    // null residual (row-scaled).
    std::vector<double> J, Rmd; slim_analytic_jacobian(U,in,op,J); slim_R_Mdot_column(U,in,op,Rmd);
    const GroupScales gs = slim_group_scales(U, in);
    auto rsinv=[&](int r)->double{
        if (r<N) return 1.0/std::max(gs.mass,1e-300);
        if (r<2*N) return 1.0/std::max(gs.ang,1e-300);
        if (r<3*N-1) return 1.0/std::max(gs.rad,1e-300);
        if (r<4*N-2) return 1.0/std::max(gs.ene,1e-300);
        if (r<4*N-1) return 1.0/std::max(gs.bc_ell,1e-300);
        if (r<4*N) return 1.0/std::max(gs.ene,1e-300);
        if (r<4*N+1) return 1.0/std::max(gs.reg_D0,1e-300);
        return 1.0/std::max(gs.reg_N1,1e-300);
    };
    double nr=0, rf=0;
    for (int row=0; row<n; ++row) {
        double v=0, ref=0;
        for (int c=0;c<n;++c){ const double tm=J[(size_t)row*n+c]*t[c]; v+=tm; ref+=std::abs(tm); }
        v+=Rmd[row]*t[n]; ref+=std::abs(Rmd[row]*t[n]);
        v*=rsinv(row); ref*=rsinv(row); nr+=v*v; rf+=ref*ref;
    }
    std::printf("  ok=%d  Mdot_dot=%+.5e  null_resid_rel=%.3e  r_sonic=%.4f ell_in=%.5f\n",
                (int)ok, t[n], std::sqrt(nr)/(std::sqrt(rf)+1e-300), U[4*N+1], U[4*N+0]);
}

static void run_branch(const OpacityLUTs& op, double a) {
    std::printf("\n############################################################\n");
    std::printf("#  ARCLENGTH CONTINUATION  a=%.3f\n", a);
    std::printf("############################################################\n");
    const char* nenv = std::getenv("ARC_N");
    const int N = nenv ? std::atoi(nenv) : 48;
    // target f_Edd encoded in in.mdot only sets Ṁ_Edd scale; the driver anchors at 0.10.
    SlimDiskInputs in = make_inputs(a, 0.10, N, 30.0*60.0);   // 30-min total budget
    auto t0 = std::chrono::steady_clock::now();
    SlimArclengthResult r = solve_slim_disk_arclength(in, op);
    auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1-t0).count();

    std::printf("\n# ---- traced branch (a=%.3f): %zu points, wall=%.1fs ----\n",
                a, r.branch.size(), wall);
    std::printf("# %-8s %-12s %-8s %-8s %-8s %-11s %-11s %-11s %-6s\n",
                "f_Edd","Mdot[g/s]","r_sonic","H/r","ell_in","peakSigma","beta_min","fadv_max","Mddot");
    for (const auto& p : r.branch) {
        std::printf("  %-8.4f %-12.4e %-8.4f %-8.4f %-8.5f %-11.4e %-11.4e %-11.4e %+d\n",
                    p.f_Edd, p.mdot, p.r_sonic, p.max_Hr, p.ell_in, p.peak_Sigma,
                    p.beta_min, p.fadv_max, p.Mdot_dot_sign);
    }
    std::printf("\n# GATE: crossed f_Edd=0.11 = %s ;  max_f_Edd reached = %.4f ;  fold detected = %s\n",
                r.crossed_011 ? "YES" : "NO", r.max_f_Edd, r.crossed_fold ? "YES" : "NO");
    if (r.ok && r.top.converged) {
        // physics at the top point.
        double maxHr=0, bmin=1e300, bmax=0, fmin=1e300, fmax=-1e300, pksig=0, pkT=0;
        for (size_t i=0;i<r.top.r.size();++i){
            const double Hr=r.top.H[i]/(r.top.r[i]*in.r_g); maxHr=std::max(maxHr,Hr);
            pksig=std::max(pksig,r.top.Sigma[i]); pkT=std::max(pkT,r.top.Tc[i]);
            const OneZoneState oz=one_zone_closure(std::max(r.top.Sigma[i],kSigmaFloor),
                                                   std::max(r.top.Tc[i],kTFloor), r.top.r[i], in, op);
            const double beta=oz.p_gas/std::max(oz.p_mid,1e-300);
            bmin=std::min(bmin,beta); bmax=std::max(bmax,beta);
            fmin=std::min(fmin,r.top.f_adv[i]); fmax=std::max(fmax,r.top.f_adv[i]);
        }
        std::printf("# TOP physics @ f_Edd=%.4f: max H/r=%.4f  beta=[%.3e,%.3e]  f_adv=[%.3e,%.3e]"
                    "  peakSigma=%.4e  peakTc=%.4e  r_sonic=%.4f  ell_in=%.5f  merit=%.3e\n",
                    r.max_f_Edd, maxHr, bmin, bmax, fmin, fmax, pksig, pkT,
                    r.top.r_sonic, r.top.ell_in, r.top.final_residual);
    } else {
        std::printf("# (no point past the anchor was accepted)\n");
    }
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    using namespace grrt;
    (void)argc; (void)argv;
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    // Task-2 tangent at the converged anchor (a=0.9) — gated by ARC_TANGENT_CHECK=1
    // (it re-converges the anchor via the full bracket, so skip it by default to spend
    // the wall budget on the continuation; the driver also reports the initial tangent).
    if (std::getenv("ARC_TANGENT_CHECK")) tangent_check_at_anchor(op, 0.9);

    // Task-4: drive the branch across the fold at a=0.9, then (gated) a=0.998.
    run_branch(op, 0.9);
    if (std::getenv("ARC_RUN_998")) run_branch(op, 0.998);

    std::printf("\n[probe] done.\n");
    return 0;
}
