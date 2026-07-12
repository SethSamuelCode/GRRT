// ===========================================================================
// SLIM BASE-INFEASIBLE PER-NODE CLASSIFIER PROBE  (DIAGNOSTIC — DELETABLE)
// ---------------------------------------------------------------------------
// THE QUESTION.  At f_Edd=0.001, a=0.9 the COUPLED slim-disk relax fails to start
// ("BASE-INFEASIBLE"): only ~6-7 of 18 radial nodes' vertical columns solve at the
// (manifold-calibrated) seed.  For EACH failing node, which of two causes dominates?
//   (H1) SEED-PLACEMENT : (Σ_target, T_c) has NO feasible column near it (Σ above the
//        column's Σ0 capacity) — it would fail with ANY solver.
//   (H2) SOLVER-ROBUSTNESS : a valid column EXISTS at/near the target, but the column's
//        own Newton cannot REACH it from its starting guess (e.g. chatter across the
//        radiative↔convective Schwarzschild switch).
//
// METHOD (per node i=0..17):
//   1. r, Σ_seed, T_c_seed, β (one-zone) from the EXACT seed the relax starts from
//      (build_thin_disk_seed + calibrate_seed_to_manifold — the walk-probe Phase-1 seed).
//   2. Direct column solve at the coupled target (Σ_target,T_c,shear,Ω_z) via the SAME
//      solve_column_coupled the relax calls — converged? f_adv?  (Its OWN internal
//      [coupled] prints carry the failure mode: Newton iters, Σ-continuation sub-solves.)
//   3. Convective content: # of n_z column nodes with is_conv (detail_bvp::convective_
//      gradient) in the best converged column; + a T_c-landscape chatter proxy (does the
//      convective-node count OSCILLATE across small T_c steps at fixed Σ?).
//   4. DISCRIMINATOR: column Σ0-capacity ceiling (base solve over T_eff×f_adv) → ratio
//      Σ_target/Σ0_ceil; + a FINE continuation from a known-feasible anchor (reduced Σ,
//      manifold T_c) toward (Σ_target, T_c_seed) with geometric step-halving.  REACHED ⇒
//      solution exists near target (H2).  STALLED short in Σ ⇒ no reachable column (H1).
//   5. Per-node verdict: H1 / H2 / BOTH / FEASIBLE + the SUMMARY counts.
//
// Honest outcome is the deliverable — no forced conclusion.
//
// Build:  cmake --build build --config Release --target slim-base-classify-probe
// Run:    build/Release/slim-base-classify-probe.exe
// REUSE: include-the-.cpp (opacity + disk_column_bvp + disk_column_coupled +
//        slim_disk_radial + slim_disk_coupled) in THAT order — mirrors the coupled
//        probes so all TU-local helpers (build_coupled_seed, eval_node_coupled,
//        shear_cgs/omega_perp_cgs, one_zone_closure, node_deriv, kappa_total) are in
//        scope.  Does NOT link grrt (avoids duplicate symbols).  Delete with the .cpp.
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
#include <io.h>
#include <fcntl.h>

using namespace grrt;
using namespace grrt::slim_coupled_detail;

static constexpr double R_G_10MSUN = 1.48e6;  // cm (GM/c² for ~10 M_sun)

// --- stdout muting (Windows), to keep the noisy internal [coupled] prints out of the
//     discriminator sub-solves while KEEPING them for the primary per-node solve. ---
static int g_saved_fd = -1;
static void mute_stdout() {
    std::fflush(stdout);
    g_saved_fd = _dup(_fileno(stdout));
    int nul = _open("NUL", _O_WRONLY);
    _dup2(nul, _fileno(stdout));
    _close(nul);
}
static void unmute_stdout() {
    std::fflush(stdout);
    if (g_saved_fd >= 0) { _dup2(g_saved_fd, _fileno(stdout)); _close(g_saved_fd); g_saved_fd = -1; }
}

// f_Edd -> Mdot [g/s], SAME textbook convention as the coupled probes.
static double mdot_from_fEdd(const SlimDiskInputs& in, double f_Edd) {
    using namespace constants;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    return f_Edd * Mdot_Edd;
}

// FAITHFUL copy of the walk probe's calibrate_seed_to_manifold (recalibrate each node's
// seed T_c to the pure-radiative f_adv≈0 manifold; leave T_c grey where build_coupled_seed
// cannot Σ-match — those are the nodes that then fail).  Prints the same calib line.
static int calibrate_seed_to_manifold(std::vector<double>& U, const SlimDiskInputs& in,
                                      const OpacityLUTs& op, const ColumnOpts& copt) {
    using namespace grrt::slim_detail;
    const int N = std::max(in.n_nodes, 4);
    const double r_s = U[4 * N + 1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N), Om(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i]  = std::exp(lr0 + (lr1 - lr0) * t);
        Om[i] = omega_from_ell(in.mass, in.spin, r[i], U[4 * i + 2]);
    }
    int n_ok = 0, n_feasible = 0;
    for (int i = 0; i < N; ++i) {
        const int j = (i + 1 < N) ? i + 1 : i - 1;
        const double shear_i  = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
        const double omegaz_i = omega_perp_cgs(in, r[i]);
        const double Sig     = std::max(U[4 * i + 0], 1e2);
        const double Tc_seed = std::max(U[4 * i + 3], 1.0);
        const OneZoneState oz = one_zone_closure(Sig, Tc_seed, r[i], in, op);
        ColumnCoupledInputs ci{};
        ci.Sigma_target  = Sig;
        ci.Tc            = Tc_seed;
        ci.shear         = std::max(shear_i, 1e-300);
        ci.omega_z       = std::max(omegaz_i, 1e-300);
        ci.alpha         = in.alpha;
        ci.rho_mid_guess = std::max(oz.rho_mid, 1e-30);
        ci.n_nodes       = copt.n_z;
        ci.max_iters     = copt.max_iter;
        ci.tol           = copt.tol;
        ci.Teff_guess    = 0.0;
        std::vector<double> Uc;
        if (build_coupled_seed(ci, op, Uc)) {
            U[4 * i + 3] = Uc[2];   // manifold midplane T_c (f_adv≈0 root)
            ++n_ok;
            ColumnCoupledInputs cv = ci; cv.Tc = U[4 * i + 3];
            const ColumnClosure c = solve_column_coupled(cv, op, nullptr);
            if (c.converged) ++n_feasible;
        }
    }
    std::printf("    [calib] manifold-set %d/%d, solve_column_coupled-feasible %d/%d\n",
                n_ok, N, n_feasible, N);
    return n_ok;
}

// Build the ColumnCoupledInputs eval_node_coupled would build at (Σ,T_c) for node i's
// geometry — the EXACT inputs the relax's column solve sees.
static ColumnCoupledInputs make_ci(const SlimDiskInputs& in, const OpacityLUTs& op,
                                   const ColumnOpts& copt, double r, double Sigma, double Tc,
                                   double shear, double omega_z) {
    using namespace grrt::slim_detail;
    const OneZoneState oz = one_zone_closure(std::max(Sigma, 1e-30), std::max(Tc, 1.0), r, in, op);
    ColumnCoupledInputs ci{};
    ci.Sigma_target  = std::max(Sigma, 1e-30);
    ci.Tc            = std::max(Tc, 1.0);
    ci.shear         = std::max(shear, 1e-300);
    ci.omega_z       = std::max(omega_z, 1e-300);
    ci.alpha         = in.alpha;
    ci.rho_mid_guess = std::max(oz.rho_mid, 1e-30);
    ci.n_nodes       = copt.n_z;
    ci.max_iters     = copt.max_iter;
    ci.tol           = copt.tol;
    ci.Teff_guess    = 0.0;
    return ci;
}

// Pack a converged ColumnClosure into an augmented warm-start state (length 4n_z+4),
// same layout eval_node_coupled uses.
static std::vector<double> pack_warm(const ColumnClosure& c, int nz) {
    std::vector<double> U(4 * nz + 4, 0.0);
    for (int k = 0; k < nz; ++k) {
        U[4*k+0] = c.sol.P_gas[k]; U[4*k+1] = c.sol.Q[k];
        U[4*k+2] = c.sol.T[k];     U[4*k+3] = c.sol.z[k];
    }
    U[4*nz+0] = c.sol.z0; U[4*nz+1] = c.sol.Sigma0; U[4*nz+2] = c.T_eff; U[4*nz+3] = c.f_adv;
    return U;
}

// Count convective nodes in a converged column profile via detail_bvp::convective_gradient.
static void count_convective(const ColumnBVPSolution& s, double omega_z,
                             const OpacityLUTs& op, int& n_conv, int& n_tot) {
    n_conv = 0; n_tot = (int)s.T.size();
    for (int k = 0; k < n_tot; ++k) {
        const double rho = std::max(s.rho[k], RHO_GHOST_FLOOR);
        const double kR  = kappa_total(op, rho, s.T[k]);
        double nabla; bool is_conv;
        grrt::detail_bvp::convective_gradient(rho, s.T[k], s.P[k], s.Q[k], kR, s.z[k],
                                              omega_z, nabla, is_conv);
        if (is_conv) ++n_conv;
    }
}

// Column Σ0 capacity ceiling at a node geometry (max Σ0 over a T_eff×f_adv grid — same
// LOWER-bound estimate the seed-dump probe uses).  Silent (base solver does not print).
static double sigma0_ceiling(const SlimDiskInputs& in, const OpacityLUTs& op, int nz,
                             double shear, double omega_z, double rho_mid_guess) {
    double best = -1.0;
    for (int it = 0; it <= 12; ++it) {
        const double Te = 1e5 * std::pow(1e3, it / 12.0);   // 1e5..1e8
        for (double fa : {2.0, 0.0, -0.5, -0.9}) {
            ColumnInputs b{};
            b.T_eff = Te; b.shear = std::max(shear, 1e-300); b.omega_z = std::max(omega_z, 1e-300);
            b.alpha = in.alpha; b.f_adv = fa; b.rho_mid_guess = rho_mid_guess;
            b.n_nodes = nz; b.max_iters = 300; b.tol = 1e-8;
            ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
            if (s.converged && s.Sigma0 > best) best = s.Sigma0;
        }
    }
    return best;
}

struct NodeResult {
    int i = 0; double r = 0, Sigma = 0, Tc = 0, beta = 0;
    bool direct_ok = false; double f_adv = 0;
    int n_conv = 0, n_tot = 0; bool chatter = false; std::vector<int> conv_landscape;
    double ceil = 0, ratio = 0;
    bool cont_reached = false; double reached_frac = 0; double finest_fail_step = 0;
    double Sig_anchor_frac = 0;
    const char* verdict = "?";
};

int main() {
    std::setbuf(stdout, nullptr);

    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = 0.9; in.alpha = 0.1; in.r_g = R_G_10MSUN;
    in.r_out = 50.0; in.n_nodes = 18; in.tol = 1e-8;
    in.r_in = 0.5 * slim_detail::isco_prograde(in.mass, in.spin);
    const double f_Edd = 1e-3;
    in.mdot = mdot_from_fEdd(in, f_Edd);

    ColumnOpts copt;   // n_z=24, 300 iters, tol 1e-8
    const int nz = copt.n_z;
    const double isco = slim_detail::isco_prograde(in.mass, in.spin);

    std::printf("# =====================================================================\n");
    std::printf("# slim-base-classify-probe : per-node H1(seed-placement) vs H2(solver) at\n");
    std::printf("#   the BASE-INFEASIBLE wall.  a=%.3f  f_Edd=%.4g  mdot=%.4e g/s\n",
                in.spin, f_Edd, in.mdot);
    std::printf("#   alpha=%.2f  r_g=%.3e cm  N=%d  n_z=%d  r_in=%.4f  ISCO=%.4f  r_out=%.1f\n",
                in.alpha, in.r_g, in.n_nodes, nz, in.r_in, isco, in.r_out);
    std::printf("# =====================================================================\n\n");

    // ---- EXACT seed the relax starts from: thin-disk seed + manifold T_c calibration ----
    std::printf("### SEED (build_thin_disk_seed + calibrate_seed_to_manifold) ###\n");
    std::vector<double> U = build_thin_disk_seed(in, op);
    mute_stdout();
    const int n_cal = calibrate_seed_to_manifold(U, in, op, copt);  // its print is muted here...
    unmute_stdout();
    // ...re-run the calib print visibly (cheap: it re-solves, but we want the number on stdout).
    {
        // recompute just the two counts by muting the inner solves and printing the summary.
        mute_stdout();
        std::vector<double> Utmp = build_thin_disk_seed(in, op);
        const int n_ok = calibrate_seed_to_manifold(Utmp, in, op, copt);
        unmute_stdout();
        std::printf("    [calib] manifold-set %d/%d nodes (T_c set to f_adv=0 manifold; rest left grey)\n",
                    n_ok, std::max(in.n_nodes, 4));
        (void)n_cal;
    }

    const int N = std::max(in.n_nodes, 4);
    const double r_s = U[4 * N + 1];
    const double lr0 = std::log(r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N), Om(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i]  = std::exp(lr0 + (lr1 - lr0) * t);
        Om[i] = slim_detail::omega_from_ell(in.mass, in.spin, r[i], U[4 * i + 2]);
    }
    std::printf("    r_s=%.5f M (%.4f ISCO)\n\n", r_s, r_s / isco);

    std::vector<NodeResult> results(N);

    for (int i = 0; i < N; ++i) {
        NodeResult& nr = results[i];
        nr.i = i; nr.r = r[i];
        const int j = (i + 1 < N) ? i + 1 : i - 1;
        const double shear_i  = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
        const double omegaz_i = omega_perp_cgs(in, r[i]);
        const double Sigma = std::max(U[4 * i + 0], 1e2);
        const double Tc    = std::max(U[4 * i + 3], 1.0);
        nr.Sigma = Sigma; nr.Tc = Tc;
        const slim_detail::OneZoneState oz =
            slim_detail::one_zone_closure(Sigma, Tc, r[i], in, op);
        nr.beta = oz.p_gas / std::max(oz.p_mid, 1e-300);
        const double rho_mid_guess = std::max(oz.rho_mid, 1e-30);

        std::printf("========================================================================\n");
        std::printf("NODE %2d  r=%.4f M  Sigma_seed=%.4e  T_c_seed=%.4e K  beta=%.4e\n",
                    i, r[i], Sigma, Tc, nr.beta);
        std::printf("         shear=%.4e /s  omega_z=%.4e /s  rho_mid_guess=%.4e\n",
                    shear_i, omegaz_i, rho_mid_guess);

        // (2) DIRECT coupled solve at the target — internal [coupled] prints show the
        //     failure mode (Newton iters, Σ-continuation sub-solves).  UNMUTED.
        ColumnCoupledInputs ci = make_ci(in, op, copt, r[i], Sigma, Tc, shear_i, omegaz_i);
        std::printf("  -- direct solve_column_coupled at (Sigma_target, T_c) [internal prints]:\n");
        ColumnClosure c_direct = solve_column_coupled(ci, op, nullptr);
        nr.direct_ok = c_direct.converged;
        nr.f_adv = c_direct.converged ? c_direct.f_adv : std::nan("");
        std::printf("  -- DIRECT RESULT: converged=%s  f_adv=%.4e\n",
                    nr.direct_ok ? "YES" : "NO", nr.f_adv);

        // (4a) Σ0 capacity ceiling + ratio (silent).
        mute_stdout();
        nr.ceil = sigma0_ceiling(in, op, nz, shear_i, omegaz_i, rho_mid_guess);
        unmute_stdout();
        nr.ratio = Sigma / std::max(nr.ceil, 1.0);
        std::printf("  -- Sigma0 ceiling (n_z=%d LOWER bound) = %.4e  =>  Sigma_target/ceil = %.4e\n",
                    nz, nr.ceil, nr.ratio);

        // (3) Convective content of the best converged column + T_c chatter landscape.
        ColumnBVPSolution conv_sol; bool have_conv_sol = false;
        if (c_direct.converged) { conv_sol = c_direct.sol; have_conv_sol = true; }
        // T_c landscape: solve at Tc×{0.94,0.97,1.00,1.03,1.06} at fixed Σ (muted); record
        // the convective-node count at each — an OSCILLATION is the chatter signature.
        const double tc_mult[5] = {0.94, 0.97, 1.00, 1.03, 1.06};
        nr.conv_landscape.assign(5, -1);
        mute_stdout();
        for (int m = 0; m < 5; ++m) {
            ColumnCoupledInputs cm = make_ci(in, op, copt, r[i], Sigma, Tc * tc_mult[m],
                                             shear_i, omegaz_i);
            ColumnClosure cc = solve_column_coupled(cm, op, nullptr);
            if (cc.converged) {
                int ncv, ntt; count_convective(cc.sol, cm.omega_z, op, ncv, ntt);
                nr.conv_landscape[m] = ncv; nr.n_tot = ntt;
                if (!have_conv_sol && tc_mult[m] == 1.00) { conv_sol = cc.sol; have_conv_sol = true; }
                if (!have_conv_sol) { conv_sol = cc.sol; have_conv_sol = true; }
            }
        }
        unmute_stdout();
        if (have_conv_sol) {
            int ncv, ntt; count_convective(conv_sol, ci.omega_z, op, ncv, ntt);
            nr.n_conv = ncv; nr.n_tot = ntt;
        }
        // Chatter = the convective-count landscape is non-monotone (a local up-down or
        // down-up in the count across the small T_c steps), among the converged samples.
        {
            std::vector<int> v;
            for (int m = 0; m < 5; ++m) if (nr.conv_landscape[m] >= 0) v.push_back(nr.conv_landscape[m]);
            for (int k = 1; k + 1 < (int)v.size(); ++k)
                if ((v[k] - v[k-1]) * (v[k+1] - v[k]) < 0) nr.chatter = true;
        }
        std::printf("  -- convective nodes (best converged col): %d/%d   "
                    "T_c landscape counts [x0.94..x1.06]: ", nr.n_conv, nr.n_tot);
        for (int m = 0; m < 5; ++m) {
            if (nr.conv_landscape[m] >= 0) std::printf("%d ", nr.conv_landscape[m]);
            else                           std::printf("- ");
        }
        std::printf("  chatter=%s\n", nr.chatter ? "YES" : "no");

        // (4b) DISCRIMINATOR: fine continuation from a known-feasible anchor toward the
        //      target.  Anchor = largest Σ fraction whose f_adv=0 manifold column converges.
        if (c_direct.converged) {
            nr.cont_reached = true; nr.reached_frac = 1.0; nr.Sig_anchor_frac = 1.0;
            std::printf("  -- discriminator: direct solve already FEASIBLE (no continuation needed)\n");
        } else {
            mute_stdout();
            // find feasible anchor
            double anchor_frac = -1.0, anchor_Tc = Tc; ColumnClosure anchor_c;
            for (double f : {1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.10, 0.05, 0.02}) {
                ColumnCoupledInputs ca = make_ci(in, op, copt, r[i], Sigma * f, Tc, shear_i, omegaz_i);
                std::vector<double> Uc;
                if (build_coupled_seed(ca, op, Uc)) {
                    ca.Tc = Uc[2];                                // manifold T_c at this Σ
                    ColumnClosure cchk = solve_column_coupled(ca, op, nullptr);
                    if (cchk.converged) { anchor_frac = f; anchor_Tc = ca.Tc; anchor_c = cchk; break; }
                }
            }
            if (anchor_frac < 0.0) {
                unmute_stdout();
                nr.cont_reached = false; nr.reached_frac = 0.0; nr.Sig_anchor_frac = 0.0;
                std::printf("  -- discriminator: NO feasible anchor even at Sigma=0.02*target"
                            "  => no reachable column (strong H1)\n");
            } else {
                nr.Sig_anchor_frac = anchor_frac;
                // Homotopy from (Sigma*anchor_frac, anchor_Tc) -> (Sigma, Tc_seed).
                const double Sa = Sigma * anchor_frac, Ta = anchor_Tc;
                double t = 0.0, step = 0.25, finest_fail = 1.0;
                std::vector<double> warm = pack_warm(anchor_c, nz);
                bool have_warm = true;
                while (t < 1.0 - 1e-9) {
                    const double t_try = std::min(1.0, t + step);
                    const double Sig_t = Sa + (Sigma - Sa) * t_try;
                    const double Tc_t  = Ta + (Tc - Ta) * t_try;
                    ColumnCoupledInputs ct = make_ci(in, op, copt, r[i], Sig_t, Tc_t, shear_i, omegaz_i);
                    ColumnClosure cs = solve_column_coupled(ct, op, have_warm ? &warm : nullptr);
                    if (cs.converged) {
                        t = t_try; warm = pack_warm(cs, nz); have_warm = true;
                        step = std::min(0.25, step * 1.5);
                    } else {
                        finest_fail = std::min(finest_fail, step);
                        step *= 0.5;
                        if (step < 1e-4) break;
                    }
                }
                unmute_stdout();
                nr.reached_frac = t;
                nr.finest_fail_step = (t >= 1.0 - 1e-9) ? 0.0 : finest_fail;
                nr.cont_reached = (t >= 1.0 - 1e-9);
                const double Sig_reached = Sa + (Sigma - Sa) * t;
                std::printf("  -- discriminator: feasible anchor at Sigma=%.3f*target (Tc_manifold=%.4e)\n",
                            anchor_frac, anchor_Tc);
                std::printf("     fine continuation -> reached t=%.4f (Sigma_reached=%.4e = %.4f*target)"
                            "  finest_failing_step=%.2e\n",
                            t, Sig_reached, Sig_reached / Sigma, nr.finest_fail_step);
            }
        }

        // (5) Per-node verdict.
        if (nr.direct_ok) {
            nr.verdict = "FEASIBLE";
        } else if (nr.cont_reached) {
            // a column exists at the target and continuation reached it, yet the direct
            // solve (from its own seed) failed => solver-robustness.
            nr.verdict = "H2";
        } else if (nr.ratio > 1.15 && nr.reached_frac < 0.98) {
            // Σ_target clearly above the column capacity AND continuation stalled short.
            nr.verdict = "H1";
        } else if (nr.reached_frac >= 0.98) {
            // got essentially to the target Σ but not all the way => borderline / mostly solver.
            nr.verdict = "H2";
        } else {
            nr.verdict = "BOTH";
        }
        std::printf("  ==> NODE %2d VERDICT: %s   (ratio=%.3e, reached=%.3f, direct=%s)\n",
                    i, nr.verdict, nr.ratio, nr.reached_frac, nr.direct_ok ? "Y" : "N");
    }

    // ---------------------------- SUMMARY ----------------------------
    std::printf("\n########################################################################\n");
    std::printf("SUMMARY  (a=%.3f  f_Edd=%.4g  N=%d  n_z=%d)\n", in.spin, f_Edd, N, nz);
    std::printf("%-4s %-9s %-11s %-11s %-10s %-8s %-9s %-8s %-8s %-8s %-7s\n",
                "i", "r[M]", "Sigma", "T_c", "beta", "direct", "ratio", "reached", "conv", "chatter", "VERDICT");
    int nH1 = 0, nH2 = 0, nBOTH = 0, nFEAS = 0;
    for (int i = 0; i < N; ++i) {
        const NodeResult& nr = results[i];
        std::printf("%-4d %-9.4f %-11.4e %-11.4e %-10.3e %-8s %-9.3e %-8.3f %-8s %-8s %-7s\n",
                    i, nr.r, nr.Sigma, nr.Tc, nr.beta, nr.direct_ok ? "OK" : "FAIL",
                    nr.ratio, nr.reached_frac,
                    (std::to_string(nr.n_conv) + "/" + std::to_string(nr.n_tot)).c_str(),
                    nr.chatter ? "YES" : "no", nr.verdict);
        if      (std::string(nr.verdict) == "H1")   ++nH1;
        else if (std::string(nr.verdict) == "H2")   ++nH2;
        else if (std::string(nr.verdict) == "BOTH") ++nBOTH;
        else                                        ++nFEAS;
    }
    std::printf("------------------------------------------------------------------------\n");
    std::printf("COUNTS:  FEASIBLE=%d   H1(seed-placement)=%d   H2(solver-robustness)=%d   BOTH=%d   (of %d)\n",
                nFEAS, nH1, nH2, nBOTH, N);
    const int nfail = N - nFEAS;
    const char* overall;
    if (nfail == 0)                       overall = "no failing nodes (base feasible at this seed?)";
    else if (nH1 + nBOTH > nH2)           overall = "BASE WALL DOMINATED BY SEED-PLACEMENT (H1): Sigma above column capacity";
    else if (nH2 > nH1 + nBOTH)           overall = "BASE WALL DOMINATED BY SOLVER-ROBUSTNESS (H2): feasible columns exist but the direct solve cannot reach them";
    else                                  overall = "MIXED: seed-placement and solver-robustness both material";
    std::printf("OVERALL CLASSIFICATION: %s\n", overall);
    std::printf("########################################################################\n");
    std::printf("DONE\n");
    return 0;
}
