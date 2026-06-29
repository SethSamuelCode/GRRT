// ===========================================================================
// SLIM COUPLED-COLUMN HARD-NODE FOCUSED PROBE  (DIAGNOSTIC — DELETABLE)
// ---------------------------------------------------------------------------
// FOCUSED reproduction + diagnosis of the solve_column_coupled HIGH-Σ failure.
//
// Takes ONE representative high-Σ node from the failing regime (a=0.9,
// f_Edd=0.01; thin-disk seed; node with Σ~5e4) and:
//   (M) builds the f_adv=0 MANIFOLD column via build_coupled_seed (the base
//       T_eff-driven solver, which is robust). This GIVES us Tc_manifold AND the
//       full converged column state — the EXACT solution the augmented Newton
//       should be able to polish from.
//   (1) calls solve_column_coupled at (Σ, Tc_manifold) — the failing path — and
//       observes what its internal seed lands on + whether the Newton polishes.
//   (2) THE DECISIVE EXPERIMENT: packs the MANIFOLD column (from M) directly into
//       the augmented 4N+4 state and runs affine_invariant_newton from THAT seed.
//       If it converges (it should, in ~1 step — the manifold is an EXACT root of
//       the f_adv=0 sub-problem, so the only nonzero residual is the T(0)-Tc pin
//       which is ZERO when Tc=Tc_manifold), the bad SEED is the root cause and the
//       fix is to seed from the manifold column.
//   (3) inspects the SEED the cold path builds internally (build_coupled_seed_2d
//       then build_coupled_seed) at Tc_manifold and prints its midplane T(0) — to
//       show it lands on the WRONG branch (T(0) ≪ Tc_manifold).
//
// Enable the augmented-Newton trajectory with AIN_DBG=1.
//
// Build:  cmake --build build --config Release --target slim-coupled-hard-probe
// Run:    AIN_DBG=1 build/Release/slim-coupled-hard-probe.exe
// REUSE: include-the-.cpp — opacity + disk_column_bvp + disk_column_coupled +
//        slim_disk_radial + slim_disk_coupled, in that order (mirrors the walk
//        probe), so the file-static seed builders / affine_invariant_newton /
//        coupled_column_residual / coupled_residual_norm are in scope, AND the
//        radial helpers (build_thin_disk_seed, shear_cgs, omega_perp_cgs,
//        one_zone_closure) are reachable.
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/disk_column_bvp.cpp"
#include "../src/disk_column_coupled.cpp"
#include "../src/slim_disk_radial.cpp"
#include "../src/slim_disk_coupled.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <numbers>
#include <algorithm>

using namespace grrt;
using namespace grrt::slim_coupled_detail;

static constexpr double R_G_10MSUN = 1.48e6;  // cm (GM/c² for ~10 M_sun)

static double mdot_from_fEdd(const SlimDiskInputs& in, double f_Edd) {
    using namespace constants;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    return f_Edd * Mdot_Edd;
}

// Pack a converged base ColumnBVPSolution (f_adv=0) into the augmented 4N+4 state,
// EXACTLY as build_coupled_seed does: profile + (z0, Σ0=Σ_target, T_eff, f_adv=0).
static std::vector<double> pack_manifold(const ColumnBVPSolution& s, double Sigma_target,
                                         double T_eff, int N) {
    std::vector<double> U(4 * N + 4, 0.0);
    for (int i = 0; i < N; ++i) {
        U[4 * i + 0] = s.P_gas[i]; U[4 * i + 1] = s.Q[i];
        U[4 * i + 2] = s.T[i];     U[4 * i + 3] = s.z[i];
    }
    U[4 * N]     = s.z0;
    U[4 * N + 1] = Sigma_target;
    U[4 * N + 2] = T_eff;
    U[4 * N + 3] = 0.0;
    return U;
}

int main() {
    std::setbuf(stdout, nullptr);
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    // EXACT base operating point of the walk probe.
    SlimDiskInputs in_base{};
    in_base.mass = 1.0; in_base.spin = 0.9; in_base.alpha = 0.1; in_base.r_g = R_G_10MSUN;
    in_base.r_out = 50.0; in_base.n_nodes = 18; in_base.tol = 1e-8;
    in_base.r_in = 0.5 * slim_detail::isco_prograde(in_base.mass, in_base.spin);

    ColumnOpts copt;   // n_z=24, 300 iters, tol 1e-8
    double f_Edd = 0.01;
    if (const char* e = std::getenv("F_EDD")) f_Edd = std::atof(e);
    SlimDiskInputs in = in_base;
    in.mdot = mdot_from_fEdd(in, f_Edd);

    std::printf("# =====================================================================\n");
    std::printf("# slim-coupled-hard-probe : focused HIGH-Σ solve_column_coupled failure\n");
    std::printf("#   a=%.3f alpha=%.2f f_Edd=%.4g mdot=%.4e g/s  N=%d  n_z=%d\n",
                in.spin, in.alpha, f_Edd, in.mdot, in_base.n_nodes, copt.n_z);
    std::printf("# =====================================================================\n\n");

    // Thin-disk seed + node grid (mirrors the walk/audit probes).
    std::vector<double> Uthin = build_thin_disk_seed(in, op);
    const int N = std::max(in.n_nodes, 4);
    const double r_s = Uthin[4 * N + 1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N), Om(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i]  = std::exp(lr0 + (lr1 - lr0) * t);
        Om[i] = slim_detail::omega_from_ell(in.mass, in.spin, r[i], Uthin[4 * i + 2]);
    }

    // Helper: the Σ0 ceiling at a node's geometry (max over a dense (T_eff,f_adv) grid).
    auto sigma0_ceiling = [&](double sh, double oz_, double rmg) -> double {
        double best = -1.0;
        for (int it = 0; it <= 24; ++it) {
            const double Te = 1e5 * std::pow(1e3, it / 24.0);   // 1e5 .. 1e8
            for (double fa : {10.0, 2.0, 0.0, -0.5, -0.9}) {
                ColumnInputs b{}; b.T_eff = Te; b.shear = std::max(sh,1e-300);
                b.omega_z = std::max(oz_,1e-300); b.alpha = in.alpha; b.f_adv = fa;
                b.rho_mid_guess = rmg; b.n_nodes = copt.n_z; b.max_iters = copt.max_iter; b.tol = copt.tol;
                ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
                if (s.converged && s.Sigma0 > best) best = s.Sigma0;
            }
        }
        return best;
    };

    // SWEEP MODE (HARD_NODE=-1): per-node feasibility table — Σ_target vs the Σ0 ceiling,
    // build_coupled_seed's achieved Σ0 (Σ-match quality), and whether solve_column_coupled
    // converges. PROVES feasibility is set by Σ_target ≶ ceiling (a column-capacity frontier),
    // not by Newton robustness.
    if (const char* es = std::getenv("HARD_NODE"); es && std::atoi(es) < 0) {
        std::printf("### PER-NODE FEASIBILITY TABLE (f_Edd=%.4g) ###\n", f_Edd);
        std::printf("    %-3s %-9s %-12s %-12s %-13s %-9s %-9s %-9s\n",
                    "i", "r[M]", "Sigma_tgt", "Sig0_ceil", "seed_Sig0", "match%", "seed_ok", "solve_ok");
        int n_feas = 0, n_above_ceil = 0;
        for (int k = 0; k < N; ++k) {
            const int jj = (k + 1 < N) ? k + 1 : k - 1;
            const double sh  = shear_cgs(in, r[k], Om[k], r[jj], Om[jj]);
            const double ozf = omega_perp_cgs(in, r[k]);
            const double Sg  = std::max(Uthin[4 * k + 0], 1e2);
            const double Tcs = std::max(Uthin[4 * k + 3], 1.0);
            const slim_detail::OneZoneState ozk = slim_detail::one_zone_closure(Sg, Tcs, r[k], in, op);
            const double rmg = std::max(ozk.rho_mid, 1e-30);
            const double ceil = sigma0_ceiling(sh, ozf, rmg);

            ColumnCoupledInputs ci{};
            ci.Sigma_target = Sg; ci.Tc = Tcs; ci.shear = std::max(sh,1e-300);
            ci.omega_z = std::max(ozf,1e-300); ci.alpha = in.alpha; ci.rho_mid_guess = rmg;
            ci.n_nodes = copt.n_z; ci.max_iters = copt.max_iter; ci.tol = copt.tol; ci.Teff_guess = 0.0;
            std::vector<double> Uc;
            const bool seed_ok = build_coupled_seed(ci, op, Uc);
            // Independent secant-achieved Σ0 (re-derive via a fresh base solve at the seed T_eff).
            double seed_sig0 = std::nan(""), matchpct = std::nan("");
            if (seed_ok) {
                ColumnInputs b{}; b.T_eff = Uc[4*copt.n_z+2]; b.shear = std::max(sh,1e-300);
                b.omega_z = std::max(ozf,1e-300); b.alpha = in.alpha; b.f_adv = 0.0;
                b.rho_mid_guess = rmg; b.n_nodes = copt.n_z; b.max_iters = copt.max_iter; b.tol = copt.tol;
                ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
                if (s.converged) { seed_sig0 = s.Sigma0; matchpct = 100.0*(s.Sigma0/Sg - 1.0); }
            }
            // The relax's actual feasibility test: solve_column_coupled at the calibrated Tc
            // (= manifold T(0) if seed_ok, else the grey Tc).
            ColumnCoupledInputs cv = ci;
            if (seed_ok) cv.Tc = Uc[2];
            const ColumnClosure c = solve_column_coupled(cv, op, nullptr);
            if (c.converged) ++n_feas;
            if (ceil < Sg) ++n_above_ceil;
            std::printf("    %-3d %-9.4f %-12.4e %-12.4e %-13.4e %-+9.1f %-9s %-9s\n",
                        k, r[k], Sg, ceil, seed_sig0, matchpct,
                        seed_ok ? "Y" : "N", c.converged ? "Y" : "N");
        }
        std::printf("    => feasible(solve_ok) %d/%d ; nodes with Σ_target ABOVE ceiling %d/%d"
                    " (the infeasible set)\n", n_feas, N, n_above_ceil, N);
        std::printf("DONE (sweep)\n");
        return 0;
    }

    // Representative HIGH-Σ node: node 1 (Σ~5e4) is the canonical failing case in the
    // audit. Allow override via env HARD_NODE.
    int node = 1;
    if (const char* e = std::getenv("HARD_NODE")) { node = std::atoi(e); node = std::clamp(node, 0, N - 1); }
    const int i = node;
    const int j = (i + 1 < N) ? i + 1 : i - 1;

    const double shear_i  = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
    const double omegaz_i = omega_perp_cgs(in, r[i]);
    const double Sigma_i  = std::max(Uthin[4 * i + 0], 1e2);
    const double Tc_seed  = std::max(Uthin[4 * i + 3], 1.0);   // grey thin-disk Tc

    // rho_mid_guess CONSISTENT with the relax (eval_node_coupled uses the one-zone rho_mid).
    const slim_detail::OneZoneState oz =
        slim_detail::one_zone_closure(Sigma_i, Tc_seed, r[i], in, op);
    const double rho_mid_guess = std::max(oz.rho_mid, 1e-30);

    std::printf("### REPRESENTATIVE HIGH-Σ NODE %d ###\n", i);
    std::printf("    r=%.5f M  Sigma=%.6e  Tc_seed(grey)=%.6e\n", r[i], Sigma_i, Tc_seed);
    std::printf("    shear=%.6e  omega_z=%.6e  rho_mid_guess(one-zone)=%.6e\n\n",
                shear_i, omegaz_i, rho_mid_guess);

    // Common ColumnCoupledInputs (matches eval_node_coupled / the relax).
    auto make_ci = [&](double Tc) {
        ColumnCoupledInputs ci{};
        ci.Sigma_target  = Sigma_i;
        ci.Tc            = Tc;
        ci.shear         = std::max(shear_i, 1e-300);
        ci.omega_z       = std::max(omegaz_i, 1e-300);
        ci.alpha         = in.alpha;
        ci.rho_mid_guess = rho_mid_guess;
        ci.n_nodes       = copt.n_z;
        ci.max_iters     = copt.max_iter;
        ci.tol           = copt.tol;
        ci.Teff_guess    = 0.0;
        return ci;
    };

    // ======================================================================
    // (M) MANIFOLD column via the base T_eff-driven solver (build_coupled_seed at
    //     the GREY Tc — this returns the f_adv=0 manifold column whose midplane T(0)
    //     IS Tc_manifold). This is the KNOWN-GOOD root.
    // ======================================================================
    std::printf("### (M) manifold f_adv=0 column via build_coupled_seed(grey Tc) ###\n");
    ColumnCoupledInputs ci_grey = make_ci(Tc_seed);
    std::vector<double> Uman;
    const bool man_ok = build_coupled_seed(ci_grey, op, Uman);
    if (!man_ok) {
        std::printf("    build_coupled_seed FAILED at the grey Tc — cannot locate the manifold.\n");
        std::printf("DONE (manifold-unreachable)\n");
        return 1;
    }
    const int nz = copt.n_z;
    const double Tc_manifold = Uman[2];                 // midplane T(0) of the manifold column
    const double Teff_manifold = Uman[4 * nz + 2];
    const double fadv_manifold = Uman[4 * nz + 3];
    std::printf("    Tc_manifold = T(0) = %.6e   T_eff = %.6e   f_adv = %+.3e\n",
                Tc_manifold, Teff_manifold, fadv_manifold);
    // Sanity: the manifold packed state should have ~zero coupled residual at Tc=Tc_manifold.
    {
        ColumnCoupledInputs ci_m = make_ci(Tc_manifold);
        std::vector<double> R;
        coupled_column_residual(Uman, ci_m, op, R);
        const double merit = coupled_residual_norm(Uman, R, ci_m);
        std::printf("    coupled residual merit of the MANIFOLD state at Tc=Tc_manifold = %.3e"
                    "  (expect ~0)\n", merit);
        // Per-row breakdown: which rows carry the residual? Interior rows 0..4N-5; the 8
        // BC/pin rows are the LAST 8. Report the largest |R| ABSOLUTE entries + the 8 BCs.
        const int nrows = (int)R.size();
        int amax = 0; for (int k = 1; k < nrows; ++k) if (std::abs(R[k]) > std::abs(R[amax])) amax = k;
        std::printf("    largest |R| row = %d  value=%.4e  (interior block ends at row %d)\n",
                    amax, R[amax], 4 * nz - 1);
        const char* bcn[8] = {"Q(0)","z(0)","Q(srf)-sTeff^4","T(srf)-Teff","z(srf)-z0",
                              "P(srf)","T(0)-Tc","Sigma0-Sig"};
        for (int b = 0; b < 8; ++b)
            std::printf("      BC[%d] %-16s R=%.6e\n", b, bcn[b], R[4 * nz + b]);

        // CROSS-CHECK: evaluate the BASE column residual (4N+2) on the SAME profile +
        // (z0,Σ0), at the manifold T_eff & f_adv=0. If the base residual is ALSO huge at
        // row 1, build_coupled_seed accepted a NON-converged column (the secant's sbest).
        ColumnInputs b{}; b.T_eff = Teff_manifold; b.shear = ci_m.shear; b.omega_z = ci_m.omega_z;
        b.alpha = ci_m.alpha; b.f_adv = 0.0; b.rho_mid_guess = ci_m.rho_mid_guess;
        b.n_nodes = nz; b.max_iters = ci_m.max_iters; b.tol = ci_m.tol;
        std::vector<double> Ubase(4 * nz + 2, 0.0);
        for (int k = 0; k < nz; ++k) { Ubase[4*k+0]=Uman[4*k+0]; Ubase[4*k+1]=Uman[4*k+1];
                                       Ubase[4*k+2]=Uman[4*k+2]; Ubase[4*k+3]=Uman[4*k+3]; }
        Ubase[4*nz] = Uman[4*nz]; Ubase[4*nz+1] = Uman[4*nz+1];
        std::vector<double> Rb;
        column_residual(Ubase, b, op, Rb);
        int bmax = 0; for (int k = 1; k < (int)Rb.size(); ++k) if (std::abs(Rb[k]) > std::abs(Rb[bmax])) bmax = k;
        std::printf("    [xcheck] BASE residual on same state: largest |R| row=%d value=%.4e\n",
                    bmax, Rb[bmax]);
        // Re-solve the base column FRESH at (Teff_manifold, f_adv=0) and report its merit +
        // whether its converged profile matches Uman (i.e. is Uman really the base root?).
        ColumnBVPSolution sfresh = solve_column_bvp(b, op, nullptr);
        std::printf("    [xcheck] fresh solve_column_bvp(Teff=%.4e,f_adv=0): converged=%d final_resid=%.3e"
                    "  Sigma0=%.6e (Uman Sigma0=%.6e, target=%.6e)\n",
                    Teff_manifold, (int)sfresh.converged, sfresh.final_residual,
                    sfresh.converged ? sfresh.Sigma0 : -1.0, Uman[4*nz+1], Sigma_i);
        std::printf("\n");
    }

    // ======================================================================
    // (M2) Does an f_adv=0 column with Σ0 = Σ_target EVEN EXIST? Sweep T_eff and
    //      print Σ0(T_eff) at f_adv=0. build_coupled_seed's secant assumes Σ0 is
    //      monotone-reachable; if Σ0 SATURATES below Σ_target the f_adv=0 manifold
    //      never reaches this Σ and the secant returns a bogus best-so-far.
    // ======================================================================
    std::printf("### (M2) Σ0(T_eff) sweep at f_adv=0 (does Σ0=Σ_target exist?) ###\n");
    {
        std::printf("    target Σ = %.4e\n", Sigma_i);
        std::printf("    %-12s %-14s %-8s\n", "T_eff[K]", "Sigma0[g/cm2]", "conv?");
        double sig_max = -1.0; double sig_max_Te = 0.0;
        for (double Te : {1e5, 2e5, 5e5, 7.74e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7}) {
            ColumnInputs b{}; b.T_eff = Te; b.shear = std::max(shear_i,1e-300);
            b.omega_z = std::max(omegaz_i,1e-300); b.alpha = in.alpha; b.f_adv = 0.0;
            b.rho_mid_guess = rho_mid_guess; b.n_nodes = nz; b.max_iters = copt.max_iter; b.tol = copt.tol;
            ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
            std::printf("    %-12.3e %-14.6e %-8d\n", Te, s.converged ? s.Sigma0 : -1.0, (int)s.converged);
            if (s.converged && s.Sigma0 > sig_max) { sig_max = s.Sigma0; sig_max_Te = Te; }
        }
        std::printf("    => max reachable Σ0 at f_adv=0 ≈ %.4e (at T_eff≈%.3e); target=%.4e -> %s\n\n",
                    sig_max, sig_max_Te, Sigma_i,
                    (sig_max < Sigma_i) ? "*** f_adv=0 manifold CANNOT reach Σ_target (needs f_adv<0) ***"
                                        : "f_adv=0 manifold reaches Σ_target");
    }

    // ======================================================================
    // (M3) Can a NEGATIVE f_adv reach Σ_target? Sweep f_adv at a few T_eff and print
    //      Σ0. If Σ0 grows large as f_adv -> -1, the real root has f_adv<0 and the
    //      f_adv=0-seed strategy is fundamentally wrong for this Σ.
    // ======================================================================
    std::printf("### (M3) Σ0(f_adv) sweep (does a NEGATIVE f_adv reach Σ_target=%.4e?) ###\n", Sigma_i);
    {
        std::printf("    %-12s", "f_adv\\T_eff");
        const double Tes[] = {2e6, 1e7, 3e7};
        for (double Te : Tes) std::printf("  Te=%-10.2e", Te);
        std::printf("\n");
        for (double fa : {5.0, 2.0, 1.0, 0.5, 0.0, -0.3, -0.6, -0.8, -0.9, -0.95, -0.99}) {
            std::printf("    %-12.3f", fa);
            for (double Te : Tes) {
                ColumnInputs b{}; b.T_eff = Te; b.shear = std::max(shear_i,1e-300);
                b.omega_z = std::max(omegaz_i,1e-300); b.alpha = in.alpha; b.f_adv = fa;
                b.rho_mid_guess = rho_mid_guess; b.n_nodes = nz; b.max_iters = copt.max_iter; b.tol = copt.tol;
                ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
                if (s.converged) std::printf("  %-12.4e", s.Sigma0);
                else             std::printf("  %-12s", "(fail)");
            }
            std::printf("\n");
        }
        std::printf("    (if Σ0 crosses %.4e as f_adv->-1, the TRUE root needs f_adv<0)\n\n", Sigma_i);
    }

    // ======================================================================
    // (M4) Is Σ0 MULTIVALUED in the seed? The column ODE may have a low-density and a
    //      high-density branch; the secant only sees the branch its rho_mid_guess seeds.
    //      Sweep rho_mid_guess at a fixed (T_eff=2e6, f_adv=0) and print Σ0 — if a much
    //      larger seed finds Σ0≈Σ_target, the manifold EXISTS but the SEED is the bug.
    // ======================================================================
    std::printf("### (M4) Σ0(rho_mid_guess) at T_eff=2e6, f_adv=0 (is there a high-Σ branch?) ###\n");
    {
        std::printf("    %-14s %-14s %-8s\n", "rho_mid_guess", "Sigma0", "conv?");
        for (double rg : {1e-2, 1e-1, 1.0, 2.37, 10.0, 1e2, 1e3, 1e4}) {
            ColumnInputs b{}; b.T_eff = 2e6; b.shear = std::max(shear_i,1e-300);
            b.omega_z = std::max(omegaz_i,1e-300); b.alpha = in.alpha; b.f_adv = 0.0;
            b.rho_mid_guess = rg; b.n_nodes = nz; b.max_iters = copt.max_iter; b.tol = copt.tol;
            ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
            std::printf("    %-14.3e %-14.6e %-8d\n", rg, s.converged ? s.Sigma0 : -1.0, (int)s.converged);
        }
        std::printf("    (if Σ0 is the SAME for all seeds, the column is single-valued ⇒ genuinely infeasible)\n\n");
    }

    // ======================================================================
    // (M5) The Σ0 CEILING: max Σ0 over a dense (T_eff, f_adv) grid. States the
    //      feasibility ceiling definitively for this node's geometry.
    // ======================================================================
    std::printf("### (M5) Σ0 ceiling over a dense (T_eff,f_adv) grid ###\n");
    {
        double best = -1.0, bestTe = 0, bestFa = 0;
        for (int it = 0; it <= 40; ++it) {
            const double Te = 1e5 * std::pow(1e3, it / 40.0);   // 1e5 .. 1e8
            for (double fa : {10.0, 5.0, 2.0, 1.0, 0.5, 0.0, -0.5, -0.9}) {
                ColumnInputs b{}; b.T_eff = Te; b.shear = std::max(shear_i,1e-300);
                b.omega_z = std::max(omegaz_i,1e-300); b.alpha = in.alpha; b.f_adv = fa;
                b.rho_mid_guess = rho_mid_guess; b.n_nodes = nz; b.max_iters = copt.max_iter; b.tol = copt.tol;
                ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
                if (s.converged && s.Sigma0 > best) { best = s.Sigma0; bestTe = Te; bestFa = fa; }
            }
        }
        std::printf("    Σ0_ceiling ≈ %.4e (at T_eff≈%.3e, f_adv=%.2f).  Σ_target=%.4e  ->  %s\n\n",
                    best, bestTe, bestFa, Sigma_i,
                    (best < Sigma_i) ? "*** Σ_target ABOVE the ceiling: NO column root exists (genuine infeasibility) ***"
                                     : "Σ_target below ceiling: a root should exist");
    }

    // ======================================================================
    // (1) THE FAILING PATH: solve_column_coupled at (Σ, Tc_manifold).
    // ======================================================================
    std::printf("### (1) FAILING PATH: solve_column_coupled(Σ, Tc_manifold) ###\n");
    {
        ColumnCoupledInputs ci_m = make_ci(Tc_manifold);
        ColumnClosure c = solve_column_coupled(ci_m, op, nullptr);
        std::printf("    => converged=%d", (int)c.converged);
        if (c.converged)
            std::printf("  f_adv=%+.4e  F=%.4e  z0=%.4e  T_eff=%.4e\n", c.f_adv, c.F, c.z0, c.T_eff);
        else
            std::printf("  (FAILED — the bug)\n");
        std::printf("\n");
    }

    // ======================================================================
    // (3) Inspect the SEED the cold path builds internally at Tc_manifold.
    //     build_coupled_seed_2d is tried FIRST, else build_coupled_seed.
    // ======================================================================
    std::printf("### (3) what SEED does the cold path build at Tc_manifold? ###\n");
    {
        ColumnCoupledInputs ci_m = make_ci(Tc_manifold);
        std::vector<double> U2d, U1d;
        const bool ok2d = build_coupled_seed_2d(ci_m, op, U2d);
        const bool ok1d = build_coupled_seed(ci_m, op, U1d);
        const std::vector<double>& Useed = ok2d ? U2d : (ok1d ? U1d : std::vector<double>{});
        const char* which = ok2d ? "2-D (build_coupled_seed_2d)"
                                  : (ok1d ? "1-D (build_coupled_seed)" : "NEITHER");
        std::printf("    cold-path seed builder that runs: %s\n", which);
        if (!Useed.empty()) {
            const double seedT0   = Useed[2];
            const double seedTeff = Useed[4 * nz + 2];
            const double seedfadv = Useed[4 * nz + 3];
            ColumnCoupledInputs ci_mm = make_ci(Tc_manifold);
            std::vector<double> R;
            coupled_column_residual(Useed, ci_mm, op, R);
            const double merit = coupled_residual_norm(Useed, R, ci_mm);
            std::printf("    seed midplane T(0) = %.6e   (Tc_manifold=%.6e, ratio=%.4f)\n",
                        seedT0, Tc_manifold, seedT0 / Tc_manifold);
            std::printf("    seed T_eff=%.4e  f_adv=%+.4e   coupled residual merit at seed = %.3e\n",
                        seedTeff, seedfadv, merit);
            std::printf("    => %s\n",
                        (std::abs(seedT0 / Tc_manifold - 1.0) < 0.05)
                            ? "seed IS on the manifold branch (good)"
                            : "*** seed is on the WRONG branch (T(0) != Tc_manifold) ***");
        }
        std::printf("\n");
    }

    // ======================================================================
    // (2) THE DECISIVE EXPERIMENT: polish the MANIFOLD column with the augmented
    //     Newton. Pack (M) into the 4N+4 state at f_adv=0 and run
    //     affine_invariant_newton at Tc=Tc_manifold (so the T(0)-Tc pin residual = 0).
    // ======================================================================
    std::printf("### (2) DECISIVE: affine_invariant_newton from the MANIFOLD seed ###\n");
    {
        ColumnCoupledInputs ci_m = make_ci(Tc_manifold);
        std::vector<double> U = pack_manifold(
            // rebuild a ColumnBVPSolution view from Uman's profile:
            [&]{ ColumnBVPSolution s; s.P_gas.resize(nz); s.Q.resize(nz); s.T.resize(nz); s.z.resize(nz);
                 for (int k = 0; k < nz; ++k) { s.P_gas[k]=Uman[4*k+0]; s.Q[k]=Uman[4*k+1];
                                                s.T[k]=Uman[4*k+2]; s.z[k]=Uman[4*k+3]; }
                 s.z0 = Uman[4*nz]; return s; }(),
            Sigma_i, Teff_manifold, nz);
        int it = 0;
        const bool ok = affine_invariant_newton(U, ci_m, op, &it);
        std::printf("    => affine_invariant_newton converged=%d in %d iters\n", (int)ok, it);
        if (ok) {
            std::printf("    f_adv=%+.4e  T_eff=%.4e  z0=%.4e  T(0)=%.6e\n",
                        U[4*nz+3], U[4*nz+2], U[4*nz], U[2]);
            std::printf("    => PROVES: seeding the augmented Newton from the manifold column WORKS.\n");
        } else {
            std::printf("    => manifold seed ALSO fails — diagnosis is NOT just the seed branch.\n");
        }
        std::printf("\n");
    }

    // ======================================================================
    // (2b) Polish the manifold seed at an OFF-manifold Tc (the relax's actual ask).
    //      Use Tc = grey seed Tc (the value the relax starts each node at). The
    //      manifold column is the closest feasible f_adv=0 start; the Newton must
    //      back-solve f_adv to satisfy T(0)=Tc_grey.
    // ======================================================================
    std::printf("### (2b) manifold seed -> polish at the OFF-manifold grey Tc=%.4e ###\n", Tc_seed);
    {
        ColumnCoupledInputs ci_g = make_ci(Tc_seed);
        std::vector<double> U = pack_manifold(
            [&]{ ColumnBVPSolution s; s.P_gas.resize(nz); s.Q.resize(nz); s.T.resize(nz); s.z.resize(nz);
                 for (int k = 0; k < nz; ++k) { s.P_gas[k]=Uman[4*k+0]; s.Q[k]=Uman[4*k+1];
                                                s.T[k]=Uman[4*k+2]; s.z[k]=Uman[4*k+3]; }
                 s.z0 = Uman[4*nz]; return s; }(),
            Sigma_i, Teff_manifold, nz);
        int it = 0;
        const bool ok = affine_invariant_newton(U, ci_g, op, &it);
        std::printf("    => converged=%d in %d iters", (int)ok, it);
        if (ok) std::printf("  f_adv=%+.4e  T(0)=%.6e (target %.6e)\n", U[4*nz+3], U[2], Tc_seed);
        else    std::printf("  (off-manifold polish from the manifold seed failed)\n");
        std::printf("\n");
    }

    std::printf("DONE\n");
    return 0;
}
