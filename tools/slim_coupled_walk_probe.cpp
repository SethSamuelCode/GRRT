// ===========================================================================
// SLIM COUPLED-COLUMN f_Edd CONTINUATION WALK PROBE  (Task 12 — DELETABLE)
// ---------------------------------------------------------------------------
// THE DECISIVE EXPERIMENT.  Can the coupled (column-driven) slim disk be WALKED
// via f_Edd continuation from a feasible low-f_Edd start up to the POC target
// f_Edd≈0.9 (a=0.9), landing a PHYSICAL disk?  This is EXPLORATORY — the OUTCOME
// is the deliverable (honest-outcome discipline), not a forced pass.
//
//   Phase 1 — find the BASE of the ladder: the highest f_Edd at which the coupled
//             solve converges from a COLD thin-disk seed.
//   Phase 2 — WALK f_Edd up toward 0.9, warm-starting each rung from the previous
//             converged radial state U; multiplicative step ×1.3 with adaptive
//             halving of the log-step on a rung failure (floor → give up the rung).
//   Phase 3 — report ONE of: REACHED-0.9 (physical? the H/r, β, f_adv profiles) /
//             STALLED-at-f_Edd_max (obstruction: column / radial / fold, evidence) /
//             BASE-INFEASIBLE.
//
// NEVER fakes convergence or loosens a validity gate.  A stall WITH a clear
// diagnosis is a SUCCESS of this experiment.
//
// Build:  cmake --build build --config Release --target slim-coupled-walk-probe
// Run:    build/Release/slim-coupled-walk-probe.exe
// REUSE: include-the-.cpp — opacity + disk_column_bvp + disk_column_coupled +
//        slim_disk_radial + slim_disk_coupled, in that order (mirrors
//        slim_coupled_smoke_probe), so slim_disk_coupled.cpp's TU-local helpers
//        (relax_coupled, eval_node_coupled, ColumnCache, ColumnOpts, the radial
//        solver's anonymous-namespace machinery) are all in scope here.
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
#include <chrono>
#include <algorithm>
#include <string>

using namespace grrt;
using namespace grrt::slim_coupled_detail;

// r_g for a 10 M_sun black hole (matches the slim-disk tests/probes).
static constexpr double R_G_10MSUN = 1.48e6;  // cm (GM/c² for ~10 M_sun)

// f_Edd -> Mdot [g/s], SAME textbook convention as the other coupled probes
// (mdot = f_Edd · 10·L_Edd/c²,  L_Edd = 4πGMc/κ_es,  κ_es=0.34).
static double mdot_from_fEdd(const SlimDiskInputs& in, double f_Edd) {
    using namespace constants;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    return f_Edd * Mdot_Edd;
}

// ---------------------------------------------------------------------------
// Per-node coupled diagnostics, re-derived from a converged radial state U by
// re-solving each node's column at its (Σ,T_c) with the SAME node geometry the
// coupled residual uses (shear_cgs / omega_perp_cgs).  This mirrors the grid +
// loop in slim_coupled_residual exactly (free-inner-node log grid r[0]=r_s),
// so the reported H/r=z0/r, β, f_adv, V are the coupled solution's own values.
// ---------------------------------------------------------------------------
struct NodeDiag {
    double r = 0.0, Sigma = 0.0, V = 0.0, ell = 0.0, Tc = 0.0;
    bool   col_ok = false;
    double z0 = 0.0, Hr = 0.0, F = 0.0;
    double beta = 0.0;     // p_gas/p_mid (one-zone diagnostic at (Σ,T_c))
    double f_adv = 0.0;
    double tau_mid = 0.0;
    double eta3 = 0.0, eta4 = 0.0;
};
struct WalkDiag {
    bool   all_cols_ok = true;
    int    first_bad_node = -1;
    double r_s = 0.0, ell_in = 0.0;
    double r_isco = 0.0;
    bool   sonic_inside_isco = false;
    bool   V_all_neg = true;
    double Hr_max = 0.0;
    double beta_min = 1e300, beta_max = -1e300;
    double fadv_min = 1e300, fadv_max = -1e300;
    std::vector<NodeDiag> nodes;
};

static WalkDiag diagnose(const SlimDiskInputs& in, const OpacityLUTs& op,
                         const ColumnOpts& copt, const std::vector<double>& U) {
    using namespace constants;
    using namespace grrt::slim_detail;
    WalkDiag d;
    const int N = std::max(in.n_nodes, 4);
    d.r_s    = U[4 * N + 1];
    d.ell_in = U[4 * N + 0];
    d.r_isco = isco_prograde(in.mass, in.spin);
    d.sonic_inside_isco = (d.r_s < d.r_isco);

    const double lr0 = std::log(d.r_s), lr1 = std::log(in.r_out);
    std::vector<double> r(N);
    for (int i = 0; i < N; ++i) {
        const double t = (N == 1) ? 0.0 : double(i) / double(N - 1);
        r[i] = std::exp(lr0 + (lr1 - lr0) * t);
    }
    std::vector<double> Om(N);
    for (int i = 0; i < N; ++i)
        Om[i] = omega_from_ell(in.mass, in.spin, r[i], U[4 * i + 2]);

    ColumnCache cache; cache.resize(N, copt.n_z);
    d.nodes.resize(N);
    for (int i = 0; i < N; ++i) {
        NodeDiag nd;
        nd.r     = r[i];
        nd.Sigma = U[4 * i + 0];
        nd.V     = U[4 * i + 1];
        nd.ell   = U[4 * i + 2];
        nd.Tc    = U[4 * i + 3];
        if (!(nd.V < 0.0)) d.V_all_neg = false;

        const int j = (i + 1 < N) ? i + 1 : i - 1;
        const double shear_i  = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
        const double omegaz_i = omega_perp_cgs(in, r[i]);
        const CoupledNode e = eval_node_coupled(in, op, copt, cache, i,
                                                r[i], nd.Sigma, nd.V, nd.ell, nd.Tc,
                                                shear_i, omegaz_i);
        nd.col_ok = e.ok;
        if (e.ok) {
            nd.z0    = e.z0;
            nd.Hr    = (r[i] > 0.0) ? e.z0 / (r[i] * in.r_g) : 0.0;
            nd.F     = e.F;
            nd.f_adv = e.f_adv;
            nd.eta3  = e.eta3;
            nd.eta4  = e.eta4;
            // β from the one-zone closure at (Σ,T_c) (diagnostic only; same as the
            // smoke probe). β→1 gas-dominated, β→0 radiation-dominated.
            const OneZoneState oz =
                one_zone_closure(std::max(nd.Sigma, 1e-30), std::max(nd.Tc, 1.0), r[i], in, op);
            nd.beta = oz.p_gas / std::max(oz.p_mid, 1e-300);
            d.Hr_max   = std::max(d.Hr_max, nd.Hr);
            d.beta_min = std::min(d.beta_min, nd.beta);
            d.beta_max = std::max(d.beta_max, nd.beta);
            d.fadv_min = std::min(d.fadv_min, nd.f_adv);
            d.fadv_max = std::max(d.fadv_max, nd.f_adv);
        } else {
            d.all_cols_ok = false;
            if (d.first_bad_node < 0) d.first_bad_node = i;
        }
        d.nodes[i] = nd;
    }
    return d;
}

// Recalibrate each node's seed T_c to the column's pure-radiative f_adv≈0 MANIFOLD.
// The thin-disk seed sets T_c from the GREY (convective-closure) one-zone, which is ~2×
// too cold for the PURE-RADIATIVE column — so the columns are infeasible at that (Σ,T_c)
// pair and the relax cannot even start (the BASE-INFEASIBLE finding, 2026-06-29).
// build_coupled_seed[_2d] pins Σ at f_adv≈0 and returns the column whose midplane T(0) IS
// the manifold T_c; we overwrite U's T_c with it. Returns the count of calibrated nodes.
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
        // rho_mid_guess MUST match the relax (eval_node_coupled uses the one-zone rho_mid, NOT
        // a hard-coded 1e-3 — the plumbing audit found that mismatch lands the manifold solve
        // on a different branch ⇒ a Tc the relax cannot then converge at).
        const OneZoneState oz = one_zone_closure(Sig, Tc_seed, r[i], in, op);
        ColumnCoupledInputs ci{};
        ci.Sigma_target  = Sig;
        ci.Tc            = Tc_seed;                       // seeds the T_eff guess only
        ci.shear         = std::max(shear_i, 1e-300);
        ci.omega_z       = std::max(omegaz_i, 1e-300);
        ci.alpha         = in.alpha;
        ci.rho_mid_guess = std::max(oz.rho_mid, 1e-30);   // CONSISTENT with the relax
        ci.n_nodes       = copt.n_z;
        ci.max_iters     = copt.max_iter;
        ci.tol           = copt.tol;
        ci.Teff_guess    = 0.0;
        // 1-D builder ONLY (f_adv=0 manifold). The 2-D builder pins Tc to the INPUT (back-
        // solving f_adv), so it returns the GREY Tc, not the manifold — a calibration no-op
        // (audit Bug #1).
        std::vector<double> Uc;
        if (build_coupled_seed(ci, op, Uc) || build_coupled_seed_advective(ci, op, Uc)) {
            U[4 * i + 3] = Uc[2];   // manifold midplane T_c (f_adv≈0 or advective root)
            ++n_ok;
            // Verify the RELAX's own function converges at the calibrated Tc (same rho_mid).
            ColumnCoupledInputs cv = ci; cv.Tc = U[4 * i + 3];
            const ColumnClosure c = solve_column_coupled(cv, op, nullptr);
            if (c.converged) ++n_feasible;
        }
    }
    std::printf("    [calib] manifold-set %d/%d, solve_column_coupled-feasible %d/%d\n",
                n_ok, N, n_feasible, N);
    return n_ok;
}

// Compact one-line summary of a converged rung's disk diagnostics.
static void print_summary(const WalkDiag& d) {
    std::printf("    H/r_max=%.4f  beta[%.4f,%.4f]  f_adv[%+.4f,%+.4f]  "
                "r_s=%.4f (isco=%.4f, inside=%s)  V<0=%s  cols_ok=%s\n",
                d.Hr_max, d.beta_min, d.beta_max, d.fadv_min, d.fadv_max,
                d.r_s, d.r_isco, d.sonic_inside_isco ? "Y" : "N",
                d.V_all_neg ? "Y" : "N", d.all_cols_ok ? "Y" : "N");
}

// Full per-node profile print (H/r, β, f_adv, V, F) — for the Phase-3 outcome.
static void print_profile(const WalkDiag& d) {
    std::printf("    %-4s %-9s %-11s %-11s %-11s %-9s %-9s %-9s %-9s %-7s\n",
                "i", "r[M]", "Sigma", "Tc[K]", "V", "z0[cm]", "H/r", "beta", "f_adv", "col?");
    const int N = (int)d.nodes.size();
    const int step = std::max(N / 16, 1);
    for (int i = 0; i < N; i += step) {
        const NodeDiag& n = d.nodes[i];
        std::printf("    %-4d %-9.4f %-11.4e %-11.4e %-11.3e %-9.3e %-9.4f %-9.4f %+9.4f %-7s\n",
                    i, n.r, n.Sigma, n.Tc, n.V, n.z0, n.Hr, n.beta, n.f_adv,
                    n.col_ok ? "Y" : "N");
    }
    // Always include the last node (outer edge) if not already sampled.
    if ((N - 1) % step != 0) {
        const NodeDiag& n = d.nodes[N - 1];
        std::printf("    %-4d %-9.4f %-11.4e %-11.4e %-11.3e %-9.3e %-9.4f %-9.4f %+9.4f %-7s\n",
                    N - 1, n.r, n.Sigma, n.Tc, n.V, n.z0, n.Hr, n.beta, n.f_adv,
                    n.col_ok ? "Y" : "N");
    }
}

// One coupled attempt at a given f_Edd from a SUPPLIED seed U (modified in place
// on success). Installs a per-attempt wall/iter budget (g_budget) so a stuck
// relax aborts honestly instead of grinding. Returns converged; on return,
// out_secs/out_iters carry the wall time and the budget's inner-iter count for
// THIS attempt (cumulative-inner-iter delta).
static bool attempt(const SlimDiskInputs& in_base, const OpacityLUTs& op,
                    const ColumnOpts& copt, double f_Edd,
                    std::vector<double>& U, int max_iters,
                    double wall_budget_s, long long iter_budget,
                    double& out_secs, long long& out_iters) {
    SlimDiskInputs in = in_base;
    in.mdot = mdot_from_fEdd(in, f_Edd);

    // Install a fresh budget for this attempt (RAII-cleared). relax_coupled checks
    // g_budget every inner iteration; a null pointer means "no budget", so we set
    // one explicitly here (we call relax_coupled DIRECTLY, not via the public
    // solve_slim_disk_coupled which would install its own).
    SolveBudget budget;
    budget.wall_cap_s    = wall_budget_s;
    budget.inner_iter_cap = iter_budget;
    struct Guard { ~Guard() { g_budget = nullptr; } } guard;
    g_budget = &budget;

    const auto t0 = std::chrono::steady_clock::now();
    const bool ok = relax_coupled(in, op, copt, U, max_iters);
    out_secs  = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    out_iters = budget.inner_iters;
    if (budget.tripped) {
        std::printf("    [budget tripped: %s  (%.1fs, %lld inner-iters)]\n",
                    budget.what ? budget.what : "?", out_secs, out_iters);
        return false;
    }
    return ok;
}

int main() {
    std::setbuf(stdout, nullptr);
    const auto t_start = std::chrono::steady_clock::now();

    // Opacity LUTs (same range the coupled probes use).
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    // Base operating point: a=0.9, MODEST grid (per the brief: N≈16-20, n_z≈24).
    SlimDiskInputs in_base{};
    in_base.mass = 1.0; in_base.spin = 0.9; in_base.alpha = 0.1; in_base.r_g = R_G_10MSUN;
    in_base.r_out = 50.0; in_base.n_nodes = 18; in_base.tol = 1e-8;
    in_base.r_in = 0.5 * slim_detail::isco_prograde(in_base.mass, in_base.spin);
    // max_iters per coupled relax (the relax loop's own iteration cap).
    const int kMaxIters = 160;

    ColumnOpts copt;             // bring-up defaults (300 column iters, tol 1e-8)
    copt.n_z = 256;              // reliable 18/18-feasible seed (n_z=96 only 15/18; stretched grid shelved 2026-07-24)
    const double r_isco = slim_detail::isco_prograde(in_base.mass, in_base.spin);

    std::printf("# =====================================================================\n");
    std::printf("# slim-coupled-walk-probe : f_Edd continuation of the COUPLED slim disk\n");
    std::printf("#   a=%.3f  alpha=%.2f  r_g=%.3e cm  N=%d  n_z=%d  r_in=%.4f  r_isco=%.4f  r_out=%.1f\n",
                in_base.spin, in_base.alpha, in_base.r_g, in_base.n_nodes, copt.n_z,
                in_base.r_in, r_isco, in_base.r_out);
    std::printf("#   TARGET f_Edd = 0.9 (POC).  Outcome = deliverable (honest).\n");
    std::printf("# =====================================================================\n\n");

    // Per-attempt safety budgets. Cap each attempt so a STUCK one aborts, not hangs.
    //
    // SIZING (2026-07-26): the budget is checked at the TOP of each outer Newton
    // iteration (slim_disk_coupled.cpp ~L983), so a value BELOW one iteration's cost
    // silently permits exactly ONE iteration and aborts at it=1. The old 420 s was
    // sized for the comment's "cold base solve at modest N is seconds-to-minutes"
    // regime and was never revisited when the coupled path went to n_z=256, where a
    // single outer iteration measured 9668 s. Consequence: the base rung had NEVER
    // been observed past it=0 — every "is it converging?" reading came from one point.
    // 12 h admits ~13 iterations at the current post-speedup cost (~50 min/iter);
    // higher rungs still fail fast at the SEED, so they cannot consume this.
    const double kWallPerAttempt = 43200.0;   // 12 h / attempt
    const long long kIterPerAttempt = 60000;  // cumulative inner-iter cap / attempt

    // =======================================================================
    // PHASE 1 — find the base of the ladder (highest cold-seed-converging f_Edd).
    // =======================================================================
    std::printf("### PHASE 1 — base of the ladder (cold thin-disk seed) ###\n");
    const double base_fEdd[] = {1e-3, 3e-3, 1e-2, 3e-2, 1e-1};

    double best_base_fEdd = -1.0;
    std::vector<double> best_base_U;
    WalkDiag best_base_diag;

    for (double f : base_fEdd) {
        SlimDiskInputs in = in_base;
        in.mdot = mdot_from_fEdd(in, f);
        std::vector<double> U = build_thin_disk_seed(in, op);
        // MANIFOLD-CONSISTENT SEED: recalibrate T_c to the pure-radiative f_adv≈0 manifold
        // (the grey thin-disk T_c is ~2× too cold for the column → BASE-INFEASIBLE otherwise).
        const int n_cal = calibrate_seed_to_manifold(U, in, op, copt);
        double secs = 0.0; long long iters = 0;
        const bool ok = attempt(in_base, op, copt, f, U, kMaxIters,
                                kWallPerAttempt, kIterPerAttempt, secs, iters);
        std::printf("  f_Edd=%.4g  mdot=%.4e g/s  [seed T_c calib %d/%d nodes] -> %s  (%.1fs, %lld inner-iters)\n",
                    f, in.mdot, n_cal, std::max(in.n_nodes,4), ok ? "CONVERGED" : "failed", secs, iters);
        if (ok) {
            WalkDiag d = diagnose(in, op, copt, U);
            print_summary(d);
            best_base_fEdd = f;
            best_base_U    = U;
            best_base_diag = d;
        }
    }

    if (best_base_fEdd < 0.0) {
        std::printf("\n### OUTCOME: BASE-INFEASIBLE (even with manifold-calibrated seed) ###\n");
        std::printf("No tried base f_Edd converged even after recalibrating the seed T_c to the\n");
        std::printf("column's f_adv≈0 manifold. The columns are feasible at the seed (calib counts\n");
        std::printf("above), so the obstruction is now the RADIAL relax itself (energy/mass/sonic\n");
        std::printf("balance with the coupled columns), NOT seed feasibility — the next frontier.\n");
        const double total = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        std::printf("\n[total wall %.1fs]\nDONE (BASE-INFEASIBLE)\n", total);
        return 0;
    }

    std::printf("\n  => walk STARTS at f_Edd=%.4g (highest cold-seed-converging base).\n\n",
                best_base_fEdd);

    // =======================================================================
    // PHASE 2 — walk f_Edd up toward 0.9, warm-starting U across rungs.
    // =======================================================================
    std::printf("### PHASE 2 — walk f_Edd up toward 0.9 (warm-started) ###\n");
    std::printf("  %-10s %-10s %-9s %-10s %-9s %-9s %-9s\n",
                "f_Edd", "result", "wall[s]", "in-iters", "H/r_max", "beta_rng", "fadv_rng");

    const double kTarget   = 0.9;
    const double kStep0    = 1.3;     // multiplicative step (f_Edd ×= step)
    const double kStepFloor = 1.0 + 1e-3;  // give up a rung once step shrinks below this
    const double kReachTol = 1e-3;    // treat f_Edd within this of 0.9 as "reached"

    // The last KNOWN-GOOD state (always retry a failed rung from a fresh copy of this).
    double      cur_fEdd = best_base_fEdd;
    std::vector<double> U_good = best_base_U;
    WalkDiag    diag_good = best_base_diag;

    double step = kStep0;
    bool reached = false;
    // Stall bookkeeping for the Phase-3 diagnosis.
    double f_attempt_failed = -1.0;     // the f_Edd value the walk could not pass
    WalkDiag last_good_diag = best_base_diag;
    double   f_max_converged = best_base_fEdd;

    int rung_guard = 0;
    const int kMaxRungs = 200;         // hard cap on continuation rungs (safety)

    while (cur_fEdd < kTarget * (1.0 - kReachTol) && rung_guard++ < kMaxRungs) {
        // Propose the next f_Edd (cap exactly at the target).
        double f_next = std::min(cur_fEdd * step, kTarget);

        std::vector<double> U = U_good;   // warm start from the last converged state
        double secs = 0.0; long long iters = 0;
        const bool ok = attempt(in_base, op, copt, f_next, U, kMaxIters,
                                kWallPerAttempt, kIterPerAttempt, secs, iters);

        if (ok) {
            WalkDiag d = diagnose(in_base, op, copt, U);
            std::printf("  %-10.5g %-10s %-9.1f %-10lld %-9.4f [%.3f,%.3f] [%+.3f,%+.3f]\n",
                        f_next, "CONVERGED", secs, iters, d.Hr_max,
                        d.beta_min, d.beta_max, d.fadv_min, d.fadv_max);
            // Accept the rung.
            cur_fEdd        = f_next;
            U_good          = U;
            diag_good       = d;
            last_good_diag  = d;
            f_max_converged = f_next;
            // After a success, gently grow the step back toward the nominal (but not above).
            step = std::min(kStep0, step * 1.15);
            if (cur_fEdd >= kTarget * (1.0 - kReachTol)) { reached = true; break; }
        } else {
            std::printf("  %-10.5g %-10s %-9.1f %-10lld  (shrinking step)\n",
                        f_next, "failed", secs, iters);
            f_attempt_failed = f_next;
            // Adaptive control: shrink the multiplicative increment (closer to 1).
            const double incr = step - 1.0;
            step = 1.0 + 0.5 * incr;
            if (step < kStepFloor) {
                std::printf("  step floor reached (increment < %.1e) at f_Edd=%.5g -> STALL\n",
                            kStepFloor - 1.0, cur_fEdd);
                break;
            }
        }
    }
    if (cur_fEdd >= kTarget * (1.0 - kReachTol)) reached = true;

    std::printf("\n  => walk reached f_Edd_max = %.5g  (target %.2f).  steps used = %d\n\n",
                f_max_converged, kTarget, rung_guard);

    // =======================================================================
    // PHASE 3 — report the outcome.
    // =======================================================================
    std::printf("### PHASE 3 — OUTCOME ###\n");

    auto physical_check = [&](const WalkDiag& d) {
        // POC physical criteria (design §8): not a one-zone torus, gas-dominated
        // outward, modest advection, inflow everywhere, sonic inside ISCO, columns ok.
        const bool hr_ok   = d.Hr_max < 0.5;           // NOT the torus (H/r→4)
        const bool beta_out= d.beta_max > 0.5;          // gas-dominated somewhere outward
        const bool fadv_ok = (d.fadv_max < 0.6) && (d.fadv_min > -0.6);
        const bool inflow  = d.V_all_neg;
        const bool sonic   = d.sonic_inside_isco;
        const bool cols    = d.all_cols_ok;
        std::printf("    PHYSICAL CHECK:  H/r_max<0.5: %s (%.4f) | beta_max>0.5: %s (%.4f) | "
                    "f_adv in[-0.6,0.6]: %s ([%+.3f,%+.3f]) | V<0: %s | sonic<isco: %s | cols_ok: %s\n",
                    hr_ok?"Y":"N", d.Hr_max, beta_out?"Y":"N", d.beta_max,
                    fadv_ok?"Y":"N", d.fadv_min, d.fadv_max,
                    inflow?"Y":"N", sonic?"Y":"N", cols?"Y":"N");
        return hr_ok && beta_out && fadv_ok && inflow && sonic && cols;
    };

    if (reached) {
        std::printf("OUTCOME: REACHED-0.9  -- the coupled solve lands a converged disk at f_Edd=%.5g.\n",
                    f_max_converged);
        const bool phys = physical_check(diag_good);
        std::printf("    => %s\n", phys ? "PHYSICAL (POC SUCCESS)"
                                        : "converged but NOT fully physical (see flags above)");
        std::printf("  Final coupled profile @ f_Edd=%.5g:\n", f_max_converged);
        print_profile(diag_good);
    } else {
        // STALLED — diagnose the obstruction.
        std::printf("OUTCOME: STALLED-at-f_Edd_max=%.5g  (target %.2f not reached).\n",
                    f_max_converged, kTarget);
        std::printf("  Last CONVERGED rung diagnostics:\n");
        print_summary(last_good_diag);
        std::printf("  Last converged coupled profile:\n");
        print_profile(last_good_diag);

        // Probe the failing step ONCE more with diagnostics ON, to classify the
        // obstruction: (a) COLUMN — a node's column won't converge even warm-started;
        // (b) RADIAL NEWTON — columns converge but the radial relax stalls/diverges;
        // (c) FOLD — step shrank to zero at a turning point.
        std::printf("\n  --- OBSTRUCTION DIAGNOSIS (failing step from the last good state) ---\n");
        const double f_fail = (f_attempt_failed > 0.0)
                            ? f_attempt_failed
                            : std::min(f_max_converged * kStep0, kTarget);
        std::printf("  Re-probing the failing rung f_Edd=%.5g (warm-started from f_Edd=%.5g):\n",
                    f_fail, f_max_converged);

        // (1) Are the COLUMNS feasible at the last-good state pushed to the failing mdot?
        // Re-diagnose the columns at the last good U but with the failing f_Edd's mdot —
        // this tells us whether the column coverage is the wall (the columns themselves
        // fail to close) vs the radial Newton (columns close, the system won't relax).
        {
            SlimDiskInputs in_fail = in_base;
            in_fail.mdot = mdot_from_fEdd(in_fail, f_fail);
            WalkDiag dc = diagnose(in_fail, op, copt, U_good);
            if (!dc.all_cols_ok) {
                std::printf("  (a) COLUMN obstruction: at the failing mdot, node %d's column does NOT\n",
                            dc.first_bad_node);
                if (dc.first_bad_node >= 0 && dc.first_bad_node < (int)dc.nodes.size()) {
                    const NodeDiag& bn = dc.nodes[dc.first_bad_node];
                    std::printf("      converge even warm-started:  r=%.4f  Sigma=%.4e  Tc=%.4e  V=%.3e\n",
                                bn.r, bn.Sigma, bn.Tc, bn.V);
                }
                std::printf("      => the column-coverage wall is the obstruction (COLUMN).\n");
            } else {
                std::printf("  (a) Columns at the last-good state DO all still close at the failing mdot\n");
                std::printf("      (max H/r=%.4f, f_adv[%+.3f,%+.3f]); so the column itself is NOT the\n",
                            dc.Hr_max, dc.fadv_min, dc.fadv_max);
                std::printf("      immediate wall at the start of the step. Likely RADIAL-NEWTON or FOLD:\n");
                std::printf("      the radial relax cannot move the warm state to the new-mdot root.\n");
                std::printf("  (b/c) RADIAL/FOLD: run with SLIM_DIAG=1 to see the per-iter merit/maxrel\n");
                std::printf("      trajectory of the failing relax (printed below if env is set).\n");
                // Run the failing relax once more (diag may be enabled via env) to expose
                // WHERE it stalls (merit floor vs step floor vs gain-ratio stall).
                std::vector<double> U2 = U_good;
                double s2 = 0.0; long long it2 = 0;
                const bool ok2 = attempt(in_base, op, copt, f_fail, U2, kMaxIters,
                                         kWallPerAttempt, kIterPerAttempt, s2, it2);
                std::printf("      (re-run converged=%d, %.1fs, %lld inner-iters)\n",
                            (int)ok2, s2, it2);
            }
        }
        std::printf("\n  NOTE: a stall WITH this diagnosis is a valid, valuable result (design §8:\n");
        std::printf("  'instability, not closure' — the f_Edd≈0.9 coupled root may be unstable or\n");
        std::printf("  unreachable by THIS continuation, OR the column coverage is the wall).\n");
    }

    const double total = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();
    std::printf("\n[total wall %.1fs]\nDONE (%s)\n",
                total, reached ? "REACHED-0.9" : "STALLED");
    return 0;
}
