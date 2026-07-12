// ===========================================================================
// SLIM n_z-REFINEMENT + MULTI-START DISAMBIGUATOR PROBE  (DIAGNOSTIC — DELETABLE)
// ---------------------------------------------------------------------------
// THE QUESTION (three-way ambiguity).  At f_Edd=0.001, a=0.9 the coupled slim-disk
// relax can't START: ~11/18 radial nodes' vertical columns "fail" because the seed
// demands Σ that is 1.2–2.8× above the column's Σ0 capacity — AS MEASURED AT n_z=24.
// A prior probe (slim-base-classify) called this H1 (seed-placement) and recommended
// reseating Σ lower (lever C).  BUT the capacity ceiling AND the fine-continuation
// discriminator BOTH ran at n_z=24 and could share the same discretization limit.
// Three hypotheses:
//   (H1)       PHYSICAL Σ-capacity wall: column cannot hold demanded Σ at ANY n_z
//              -> fix = reseat Σ lower (lever C).
//   (ARTIFACT) n_z=24 RESOLUTION artifact: at higher n_z the column DOES hold the
//              demanded Σ -> fix = raise n_z in the relax.
//   (H2-basin) SOLVER-BASIN failure: a column exists at demanded Σ at n_z=24 but the
//              single-anchor continuation can't reach it -> fix = harden / multi-start.
//
// STUDY (three representative nodes 4/8/16 from the prior run; Part 1 does all 18):
//   PART 1 — FREE re-ratio: recompute the Σ0-capacity ratio with a HIGHER-n_z ceiling
//            (n_z=96) instead of the n_z=24 lower bound.  Ratios dropping below 1 alone
//            implicate ARTIFACT.  Report old ratio (n_z=24) vs new ratio (n_z=96).
//   PART 2 — n_z REFINEMENT (decisive): at nodes 4/8/16, n_z ∈ {24,48,96}, (a) Σ0_max
//            capacity (max converged Σ0 over a T_eff×f_adv envelope) and (b) a DIRECT
//            solve_column_coupled at the node's demanded target Σ.  Richardson read:
//            Σ0_max CONVERGES below target ⇒ H1; GROWS toward/past target ⇒ ARTIFACT.
//   PART 3 — MULTI-START at target Σ (disambiguates H2-basin): at nodes 4/8/16
//            (n_z=24 and 96), attempt solve_column_coupled at the EXACT demanded Σ from
//            a SPREAD of T_c / rho_mid_guess / seed-mode starts.  Because Σ_target is
//            PINNED, ANY convergence PROVES a column holds target Σ ⇒ not a physical Σ
//            wall (H2-basin, not H1).  Report #converged and the best column found.
//
// Honest per-node + overall verdict is the deliverable — no forced conclusion.
//
// Build:  cmake --build build --config Release --target slim-nz-refine-probe
// Run:    build/Release/slim-nz-refine-probe.exe
// REUSE: SAME include-the-.cpp order as slim-base-classify-probe (opacity +
//        disk_column_bvp + disk_column_coupled + slim_disk_radial + slim_disk_coupled)
//        so all TU-local helpers (build_thin_disk_seed, build_coupled_seed,
//        shear_cgs, omega_perp_cgs, one_zone_closure, solve_column_bvp/coupled) are in
//        scope.  Does NOT link grrt.  Delete with the .cpp.
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
#include <chrono>
#include <string>
#include <io.h>
#include <fcntl.h>

using namespace grrt;
using namespace grrt::slim_coupled_detail;

static constexpr double R_G_10MSUN = 1.48e6;  // cm (GM/c² for ~10 M_sun)

// --- stdout muting (Windows): keep the noisy internal [coupled] prints out of the
//     sweeps while KEEPING them for the primary per-node diagnostic solves. ---
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

// FAITHFUL copy of slim-base-classify-probe's calibrate_seed_to_manifold (recalibrate
// each node's seed T_c to the pure-radiative f_adv≈0 manifold; leave T_c grey where
// build_coupled_seed cannot Σ-match).  This reproduces the EXACT demanded (Σ, T_c) the
// prior run classified — so node numbering / ratios line up.
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
    int n_ok = 0;
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
        }
    }
    return n_ok;
}

// Build the ColumnCoupledInputs eval_node_coupled would build at (Σ,T_c) for a node's
// geometry, at a given n_z.  (rho_mid_guess derived from one_zone_closure.)
static ColumnCoupledInputs make_ci(const SlimDiskInputs& in, const OpacityLUTs& op,
                                   int nz, double r, double Sigma, double Tc,
                                   double shear, double omega_z, double tol, int max_iter) {
    using namespace grrt::slim_detail;
    const OneZoneState oz = one_zone_closure(std::max(Sigma, 1e-30), std::max(Tc, 1.0), r, in, op);
    ColumnCoupledInputs ci{};
    ci.Sigma_target  = std::max(Sigma, 1e-30);
    ci.Tc            = std::max(Tc, 1.0);
    ci.shear         = std::max(shear, 1e-300);
    ci.omega_z       = std::max(omega_z, 1e-300);
    ci.alpha         = in.alpha;
    ci.rho_mid_guess = std::max(oz.rho_mid, 1e-30);
    ci.n_nodes       = nz;
    ci.max_iters     = max_iter;
    ci.tol           = tol;
    ci.Teff_guess    = 0.0;
    return ci;
}

// Column Σ0 capacity ceiling at a node geometry: MAX converged Σ0 over a T_eff×f_adv
// envelope, at resolution nz.  (Base solver takes T_eff, not Σ; Σ0 is an OUTPUT, so the
// max over T_eff is the robust capacity.)  NT T_eff points over [Te_lo,Te_hi], f_adv set.
static double sigma0_ceiling(const SlimDiskInputs& in, const OpacityLUTs& op, int nz,
                             double shear, double omega_z, double rho_mid_guess,
                             int NT, double Te_lo, double Te_hi,
                             const std::vector<double>& fadv, double& Tepk, int& nconv) {
    double best = -1.0; Tepk = 0.0; nconv = 0;
    const int denom = std::max(NT - 1, 1);
    for (int it = 0; it < NT; ++it) {
        const double Te = Te_lo * std::pow(Te_hi / Te_lo, double(it) / double(denom));
        for (double fa : fadv) {
            ColumnInputs b{};
            b.T_eff = Te; b.shear = std::max(shear, 1e-300); b.omega_z = std::max(omega_z, 1e-300);
            b.alpha = in.alpha; b.f_adv = fa; b.rho_mid_guess = rho_mid_guess;
            b.n_nodes = nz; b.max_iters = 300; b.tol = 1e-8;
            ColumnBVPSolution s = solve_column_bvp(b, op, nullptr);
            if (s.converged) { ++nconv; if (s.Sigma0 > best) { best = s.Sigma0; Tepk = Te; } }
        }
    }
    return best;
}

struct NodeGeom { int i; double r, shear, omega_z, Sigma, Tc, rho_mid_guess; };

int main() {
    std::setbuf(stdout, nullptr);
    const auto t_start = std::chrono::steady_clock::now();

    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = 0.9; in.alpha = 0.1; in.r_g = R_G_10MSUN;
    in.r_out = 50.0; in.n_nodes = 18; in.tol = 1e-8;
    in.r_in = 0.5 * slim_detail::isco_prograde(in.mass, in.spin);
    const double f_Edd = 1e-3;
    in.mdot = mdot_from_fEdd(in, f_Edd);

    ColumnOpts copt;   // relax bring-up default: n_z=24, 300 iters, tol 1e-8
    const int nz_base = copt.n_z;

    // env knob: include n_z=96 in the (slow) Part-3 multi-start.  Default on.
    auto env_on = [](const char* k, bool def) {
        const char* e = std::getenv(k); return e ? (std::atoi(e) != 0) : def;
    };
    const bool p3_do96 = env_on("P3_96", true);   // include n_z=96 in Part 3 (slow)
    const double isco = slim_detail::isco_prograde(in.mass, in.spin);

    std::printf("# =====================================================================\n");
    std::printf("# slim-nz-refine-probe : H1(Σ-wall) vs ARTIFACT(n_z) vs H2-basin(solver)\n");
    std::printf("#   a=%.3f  f_Edd=%.4g  mdot=%.4e g/s  alpha=%.2f\n",
                in.spin, f_Edd, in.mdot, in.alpha);
    std::printf("#   r_g=%.3e cm  N=%d  n_z_base=%d  r_in=%.4f  ISCO=%.4f  r_out=%.1f\n",
                in.r_g, in.n_nodes, nz_base, in.r_in, isco, in.r_out);
    std::printf("#   capacity ceiling = max converged Σ0 over T_eff×f_adv envelope\n");
    std::printf("#   OLD ratio uses n_z=24 ceiling ; NEW ratio uses n_z=96 ceiling\n");
    std::printf("# =====================================================================\n\n");

    // ---- EXACT seed the relax starts from: thin-disk seed + manifold T_c calibration ----
    std::vector<double> U = build_thin_disk_seed(in, op);
    mute_stdout();
    const int n_cal = calibrate_seed_to_manifold(U, in, op, copt);
    unmute_stdout();
    std::printf("### SEED: build_thin_disk_seed + calibrate_seed_to_manifold : manifold-set %d/%d ###\n",
                n_cal, std::max(in.n_nodes, 4));

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

    // Per-node geometry + demanded (Σ, T_c).
    std::vector<NodeGeom> geom(N);
    for (int i = 0; i < N; ++i) {
        const int j = (i + 1 < N) ? i + 1 : i - 1;
        NodeGeom& g = geom[i];
        g.i = i; g.r = r[i];
        g.shear   = shear_cgs(in, r[i], Om[i], r[j], Om[j]);
        g.omega_z = omega_perp_cgs(in, r[i]);
        g.Sigma   = std::max(U[4 * i + 0], 1e2);
        g.Tc      = std::max(U[4 * i + 3], 1.0);
        const slim_detail::OneZoneState oz = slim_detail::one_zone_closure(g.Sigma, g.Tc, r[i], in, op);
        g.rho_mid_guess = std::max(oz.rho_mid, 1e-30);
    }

    // ==================================================================
    // PART 1 — FREE RE-RATIO (all 18 nodes).  old ratio (n_z=24 ceiling)
    //          vs new ratio (n_z=96 ceiling), SAME T_eff×f_adv envelope.
    // ==================================================================
    std::printf("########################################################################\n");
    std::printf("PART 1 — FREE RE-RATIO  (Σ_target / ceiling)   n_z=24 (old) vs n_z=96 (new)\n");
    std::printf("  ceiling envelope: T_eff∈[1e5,1e8] (13 pts) × f_adv{2,0,-0.5,-0.9}\n");
    std::printf("%-4s %-9s %-12s %-12s %-12s %-10s %-10s %-9s\n",
                "i", "r[M]", "Sigma_tgt", "ceil24", "ceil96", "old_rat", "new_rat", "flip<1?");
    const std::vector<double> fadv_p1 = {2.0, 0.0, -0.5, -0.9};
    std::vector<double> old_ratio(N, 0.0), new_ratio(N, 0.0);
    for (int i = 0; i < N; ++i) {
        const NodeGeom& g = geom[i];
        mute_stdout();
        double Tepk; int nc;
        const double c24 = sigma0_ceiling(in, op, 24, g.shear, g.omega_z, g.rho_mid_guess,
                                          13, 1e5, 1e8, fadv_p1, Tepk, nc);
        const double c96 = sigma0_ceiling(in, op, 96, g.shear, g.omega_z, g.rho_mid_guess,
                                          13, 1e5, 1e8, fadv_p1, Tepk, nc);
        unmute_stdout();
        old_ratio[i] = g.Sigma / std::max(c24, 1.0);
        new_ratio[i] = g.Sigma / std::max(c96, 1.0);
        const bool flips = (old_ratio[i] > 1.0 && new_ratio[i] < 1.0);
        std::printf("%-4d %-9.4f %-12.4e %-12.4e %-12.4e %-10.4e %-10.4e %-9s\n",
                    i, g.r, g.Sigma, c24, c96, old_ratio[i], new_ratio[i],
                    flips ? "YES" : (new_ratio[i] < 1.0 ? "(≤1)" : "no"));
    }
    std::printf("\n");

    // Representative nodes for Parts 2 & 3.
    const std::vector<int> study = {4, 8, 16};
    const std::vector<int> nz_ladder = {24, 48, 96};

    // ==================================================================
    // PART 2 — n_z REFINEMENT (decisive).  Σ0_max(n_z) + direct target-Σ solve.
    // ==================================================================
    std::printf("########################################################################\n");
    std::printf("PART 2 — n_z REFINEMENT   nodes {4,8,16} × n_z {24,48,96}\n");
    std::printf("  Σ0_max = max converged Σ0 over T_eff∈[1e5,5e7] (20 pts) × f_adv{2,0,-0.5,-0.9}\n");
    std::printf("  target-Σ solve = solve_column_coupled at (Σ_target, manifold T_c)\n");
    std::printf("%-4s %-9s %-12s %-6s %-13s %-11s %-12s %-9s %-10s\n",
                "i", "r[M]", "Sigma_tgt", "n_z", "Sigma0_max", "S0max/tgt", "reaches?",
                "tgt_ok", "tgt_fadv");
    const std::vector<double> fadv_p2 = {2.0, 0.0, -0.5, -0.9};
    // Track per-node: Σ0_max at each n_z, and target-solve success at each n_z.
    std::vector<std::vector<double>> s0max(study.size(), std::vector<double>(nz_ladder.size(), -1.0));
    std::vector<std::vector<bool>>   tgt_ok(study.size(), std::vector<bool>(nz_ladder.size(), false));
    for (size_t si = 0; si < study.size(); ++si) {
        const int i = study[si];
        const NodeGeom& g = geom[i];
        for (size_t ni = 0; ni < nz_ladder.size(); ++ni) {
            const int nz = nz_ladder[ni];
            mute_stdout();
            double Tepk; int nc;
            const double cap = sigma0_ceiling(in, op, nz, g.shear, g.omega_z, g.rho_mid_guess,
                                              20, 1e5, 5e7, fadv_p2, Tepk, nc);
            // Direct coupled solve at demanded target Σ (manifold T_c).
            ColumnCoupledInputs ci = make_ci(in, op, nz, g.r, g.Sigma, g.Tc,
                                             g.shear, g.omega_z, 1e-8, 300);
            const ColumnClosure c = solve_column_coupled(ci, op, nullptr);
            unmute_stdout();
            s0max[si][ni] = cap; tgt_ok[si][ni] = c.converged;
            const double ratio = cap / std::max(g.Sigma, 1.0);
            std::printf("%-4d %-9.4f %-12.4e %-6d %-13.4e %-11.4e %-12s %-9s %-+10.3e\n",
                        i, g.r, g.Sigma, nz, cap, ratio,
                        (cap >= g.Sigma) ? "YES" : "no",
                        c.converged ? "OK" : "FAIL",
                        c.converged ? c.f_adv : std::nan(""));
        }
    }
    std::printf("\n");

    // ==================================================================
    // PART 3 — MULTI-START at target Σ.  Any convergence PROVES a column holds
    //          target Σ (Σ_target pinned) ⇒ not a physical Σ wall.
    // ==================================================================
    std::printf("########################################################################\n");
    std::printf("PART 3 — MULTI-START at demanded Σ   nodes {4,8,16} × n_z {24,48%s}\n",
                p3_do96 ? ",96" : "");
    std::printf("  spread: T_c ∈ {manifold, reduced-Σ manifold} × {0.75,1.0,1.5} × {default,naive} seed\n");
    std::printf("  ANY convergence at pinned Σ_target ⇒ a column DOES hold target Σ.\n");
    std::printf("%-4s %-6s %-12s %-16s %-14s %-12s %-12s %-8s\n",
                "i", "n_z", "Sigma_tgt", "converged/total", "best_Tc", "best_Teff", "best_fadv", "sec");
    std::vector<int> nz_ms = {24, 48};
    if (p3_do96) nz_ms.push_back(96);
    // Record whether any multi-start converged, per node, per n_z.
    std::vector<std::vector<bool>> ms_hit(study.size(), std::vector<bool>(nz_ms.size(), false));
    for (size_t si = 0; si < study.size(); ++si) {
        const int i = study[si];
        const NodeGeom& g = geom[i];
        // Build the T_c spread. Center on manifold T_c (g.Tc) AND a grey one-zone T_c.
        // grey T_c proxy: reuse the seed's pre-calibration grey via a modest offset set.
        std::vector<double> tc_centers = {g.Tc};
        // A second center: manifold T_c at a REDUCED Σ (build_coupled_seed there) — a
        // plausible alternate branch. Fall back to g.Tc if it won't build.
        {
            mute_stdout();
            ColumnCoupledInputs cr = make_ci(in, op, 24, g.r, 0.4 * g.Sigma, g.Tc,
                                             g.shear, g.omega_z, 1e-8, 300);
            std::vector<double> Uc;
            if (build_coupled_seed(cr, op, Uc)) tc_centers.push_back(std::max(Uc[2], 1.0));
            unmute_stdout();
        }
        const std::vector<double> mult = {0.75, 1.0, 1.5};
        for (size_t ni = 0; ni < nz_ms.size(); ++ni) {
            const int nz = nz_ms[ni];
            const auto t_ms = std::chrono::steady_clock::now();
            int nconv = 0, ntot = 0;
            double best_Tc = 0, best_Teff = 0, best_fadv = 0; bool have_best = false;
            mute_stdout();
            for (double tc0 : tc_centers) {
                for (double m : mult) {
                    const double Tc_try = std::max(tc0 * m, 1.0);
                    // Two seed modes per (Tc): the default secant bring-up and naive_seed.
                    for (int mode = 0; mode < 2; ++mode) {
                        ColumnCoupledInputs ci = make_ci(in, op, nz, g.r, g.Sigma, Tc_try,
                                                         g.shear, g.omega_z, 1e-8, 300);
                        ci.naive_seed = (mode == 1);
                        ++ntot;
                        const ColumnClosure c = solve_column_coupled(ci, op, nullptr);
                        if (c.converged) {
                            ++nconv;
                            if (!have_best) {
                                have_best = true;
                                best_Tc = Tc_try; best_Teff = c.T_eff; best_fadv = c.f_adv;
                            }
                        }
                    }
                }
            }
            unmute_stdout();
            ms_hit[si][ni] = (nconv > 0);
            const double sec = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_ms).count();
            char tcbuf[32] = "-", tebuf[32] = "-", fabuf[32] = "-";
            if (have_best) {
                std::snprintf(tcbuf, sizeof tcbuf, "%.4e", best_Tc);
                std::snprintf(tebuf, sizeof tebuf, "%.4e", best_Teff);
                std::snprintf(fabuf, sizeof fabuf, "%+.3e", best_fadv);
            }
            std::printf("%-4d %-6d %-12.4e %-16s %-14s %-14s %-12s %-8.1f\n",
                        i, nz, g.Sigma,
                        (std::to_string(nconv) + "/" + std::to_string(ntot)).c_str(),
                        tcbuf, tebuf, fabuf, sec);
        }
    }
    std::printf("\n");

    // ==================================================================
    // SUMMARY + PER-NODE VERDICT.
    // ==================================================================
    std::printf("########################################################################\n");
    std::printf("SUMMARY  (a=%.3f  f_Edd=%.4g  N=%d)\n", in.spin, f_Edd, N);
    std::printf("%-4s %-9s %-12s %-10s %-10s %-14s %-14s %-14s %-12s\n",
                "i", "r[M]", "Sigma_tgt", "old_rat", "new_rat", "S0max_24->96",
                "tgt_ok_24->96", "multistart", "VERDICT");
    int nH1 = 0, nART = 0, nH2 = 0, nINC = 0;
    std::vector<std::string> verdicts(study.size());
    for (size_t si = 0; si < study.size(); ++si) {
        const int i = study[si];
        const NodeGeom& g = geom[i];
        const double s24 = s0max[si][0], s96 = s0max[si].back();
        const bool tgt24 = tgt_ok[si][0];
        bool tgt_highnz = false;
        for (size_t ni = 1; ni < nz_ladder.size(); ++ni) if (tgt_ok[si][ni]) tgt_highnz = true;
        const bool cap_crosses = (s96 >= g.Sigma) || (new_ratio[i] < 1.0);
        const bool cap_flat_below = (s24 > 0 && s96 > 0 && s96 < g.Sigma
                                     && (s96 / std::max(s24, 1.0)) < 1.15);
        bool multistart_hit = false;
        for (size_t ni = 0; ni < nz_ms.size(); ++ni) if (ms_hit[si][ni]) multistart_hit = true;

        std::string v;
        if (tgt_highnz || cap_crosses) {
            v = "ARTIFACT";           // target solves / capacity crosses only at higher n_z
        } else if (multistart_hit) {
            v = "H2-basin";           // a column holds target Σ at fixed n_z, single anchor missed it
        } else if (cap_flat_below) {
            v = "H1";                 // capacity n_z-converged BELOW target, no start reaches target
        } else {
            v = "INCONCLUSIVE";
        }
        verdicts[si] = v;
        if      (v == "H1")       ++nH1;
        else if (v == "ARTIFACT") ++nART;
        else if (v == "H2-basin") ++nH2;
        else                      ++nINC;

        char s0str[64]; std::snprintf(s0str, sizeof s0str, "%.2e->%.2e", s24, s96);
        char okstr[32]; std::snprintf(okstr, sizeof okstr, "%s->%s",
                                      tgt24 ? "OK" : "F", tgt_highnz ? "OK" : "F");
        std::printf("%-4d %-9.4f %-12.4e %-10.3e %-10.3e %-14s %-14s %-14s %-12s\n",
                    i, g.r, g.Sigma, old_ratio[i], new_ratio[i], s0str, okstr,
                    multistart_hit ? "HIT" : "miss", v.c_str());
    }
    std::printf("------------------------------------------------------------------------\n");
    std::printf("COUNTS (of %d studied): H1=%d  ARTIFACT=%d  H2-basin=%d  INCONCLUSIVE=%d\n",
                (int)study.size(), nH1, nART, nH2, nINC);

    const char* overall; const char* fix;
    if (nART >= nH1 && nART >= nH2 && nART > 0) {
        overall = "DOMINATED BY n_z RESOLUTION ARTIFACT: columns hold demanded Σ at higher n_z";
        fix = "RAISE n_z in the relax";
    } else if (nH2 >= nH1 && nH2 > 0) {
        overall = "DOMINATED BY SOLVER-BASIN (H2): columns exist at demanded Σ, single anchor misses them";
        fix = "HARDEN SOLVER / multi-start";
    } else if (nH1 > 0) {
        overall = "DOMINATED BY PHYSICAL Σ-WALL (H1): capacity n_z-converged below demanded Σ";
        fix = "LEVER C reseat-Σ lower";
    } else {
        overall = "INCONCLUSIVE across the studied nodes (see per-node rows)";
        fix = "inspect per-node rows; no single lever indicated";
    }
    std::printf("OVERALL CLASSIFICATION: %s\n", overall);
    std::printf("RECOMMENDED FIX: %s\n", fix);

    const auto t_end = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t_end - t_start).count();
    std::printf("wall-clock: %.1f s\n", wall);
    std::printf("########################################################################\n");
    std::printf("DONE\n");
    return 0;
}
