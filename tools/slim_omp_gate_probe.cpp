// ===========================================================================
// OpenMP parallel==serial CORRECTNESS GATE  (coupled per-node column solves)
// ---------------------------------------------------------------------------
// THE MANDATORY GATE for the OpenMP-parallelization of the coupled relax's
// per-node column loops (slim_coupled_residual's node loop + the reduced
// Jacobian's base loop).  Those per-node eval_node_coupled / solve_column_coupled
// calls are INDEPENDENT (node i touches only its own ColumnCache slot i), so the
// parallel result MUST equal the serial result to floating-point tolerance and
// MUST be deterministic (two parallel builds agree).
//
// This probe, at a feasible (manifold-calibrated) base point:
//   (1) prints omp_get_max_threads() to PROVE /openmp took effect on this TU;
//   (2) builds the coupled residual R serial (1 thread) vs parallel (N threads);
//   (3) builds the reduced Jacobian J serial vs parallel;
//   (4) builds J parallel TWICE (determinism);
//   (5) times one J build serial vs parallel and reports the speedup.
// The assembly downstream of the parallel node loops is a pure per-node scatter
// (no cross-node FP reduction), so we require BIT-IDENTICAL agreement; the report
// still prints the measured max |Δ| so any surprise is visible.
//
// Build:  cmake --build build --config Release --target slim-omp-gate-probe
// Run:    build/Release/slim-omp-gate-probe.exe
// REUSE: include-the-.cpp — opacity + disk_column_bvp + disk_column_coupled +
//        slim_disk_radial + slim_disk_coupled, in that order (mirrors the walk
//        probe), so slim_disk_coupled.cpp's TU-local helpers are in scope here.
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

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace grrt;
using namespace grrt::slim_coupled_detail;

static constexpr double R_G_10MSUN = 1.48e6;  // cm (GM/c² for ~10 M_sun)

// f_Edd -> Mdot [g/s] (same textbook convention as the coupled probes).
static double mdot_from_fEdd(const SlimDiskInputs& in, double f_Edd) {
    using namespace constants;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    return f_Edd * Mdot_Edd;
}

// Recalibrate each node's seed T_c to the column's f_adv≈0 manifold (verbatim from the
// walk probe) so the base point is column-FEASIBLE and the reduced Jacobian actually
// builds J (an infeasible base returns false / empty J — nothing to compare).
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
        ci.Sigma_target  = Sig;   ci.Tc = Tc_seed;
        ci.shear = std::max(shear_i, 1e-300);  ci.omega_z = std::max(omegaz_i, 1e-300);
        ci.alpha = in.alpha;      ci.rho_mid_guess = std::max(oz.rho_mid, 1e-30);
        ci.n_nodes = copt.n_z;    ci.max_iters = copt.max_iter;  ci.tol = copt.tol;
        ci.Teff_guess = 0.0;
        std::vector<double> Uc;
        if (build_coupled_seed(ci, op, Uc) || build_coupled_seed_advective(ci, op, Uc)) {
            U[4 * i + 3] = Uc[2];   // manifold midplane T_c
            ++n_ok;
        }
    }
    return n_ok;
}

// Max absolute + max relative difference between two equal-length vectors.
static void vec_diff(const std::vector<double>& a, const std::vector<double>& b,
                     double& max_abs, double& max_rel, int& n_mismatch) {
    max_abs = 0.0; max_rel = 0.0; n_mismatch = 0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t k = 0; k < n; ++k) {
        const double d = std::abs(a[k] - b[k]);
        if (d != 0.0) ++n_mismatch;
        max_abs = std::max(max_abs, d);
        const double scale = std::max(std::abs(a[k]), std::abs(b[k]));
        if (scale > 0.0) max_rel = std::max(max_rel, d / scale);
    }
}

int main() {
    std::setbuf(stdout, nullptr);

    // Gate results go to STDERR: solve_column_coupled / column_sensitivity print MANY
    // unconditional diagnostic lines to STDOUT, so keeping the gate verdict on stderr keeps
    // it readable (run:  slim-omp-gate-probe.exe 2>gate.txt 1>NUL).
    auto G = [](const char* fmt, auto... a) { std::fprintf(stderr, fmt, a...); std::fflush(stderr); };

#ifndef _OPENMP
    G("FATAL: _OPENMP is NOT defined for this TU — /openmp did not take effect.\n");
    G("       The parallel pragmas are being SILENTLY IGNORED. Gate cannot run.\n");
    return 1;
#else
    const int max_threads = omp_get_max_threads();
    G("# ============================================================\n");
    G("# slim-omp-gate-probe : parallel==serial coupled-column gate\n");
    G("#   _OPENMP defined; omp_get_max_threads() = %d\n", max_threads);
    G("# ============================================================\n\n");

    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    // A feasible base point (walk-probe operating point, reduced grid for a quick gate).
    // The parallel==serial property is independent of N / n_z, so a modest grid gates it.
    SlimDiskInputs in{};
    in.mass = 1.0; in.spin = 0.9; in.alpha = 0.1; in.r_g = R_G_10MSUN;
    in.r_out = 50.0; in.n_nodes = 12; in.tol = 1e-8;
    in.r_in = 0.5 * slim_detail::isco_prograde(in.mass, in.spin);
    in.mdot = mdot_from_fEdd(in, 1e-3);

    ColumnOpts copt;
    copt.n_z = 96;    // modest column resolution — quick, still column-feasible at f_Edd=1e-3
    const int N = std::max(in.n_nodes, 4);
    const int n = 4 * N + 2;

    G("Base point: a=%.2f  N=%d  n_z=%d  f_Edd=1e-3 (manifold-calibrated seed)\n",
      in.spin, N, copt.n_z);
    std::vector<double> U = build_thin_disk_seed(in, op);
    const int n_cal = calibrate_seed_to_manifold(U, in, op, copt);
    G("  seed T_c calibrated at %d/%d nodes\n\n", n_cal, N);

    // Build the coupled residual with a FRESH (cold) cache at a set thread count; return
    // wall seconds via out_s.  A fresh cold cache makes the serial and parallel builds start
    // from IDENTICAL state (each node's warm-start is its own slot), so any difference is a
    // true parallel/serial disagreement, not a warm-start artifact.
    auto build_R = [&](int threads, std::vector<double>& R, double& out_s) {
        omp_set_num_threads(threads);
        ColumnCache cache; cache.resize(N, copt.n_z);
        bool infeas = false;
        const auto t0 = std::chrono::steady_clock::now();
        slim_coupled_residual(U, in, op, copt, cache, R, infeas);
        out_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return !infeas;
    };
    auto build_J = [&](int threads, std::vector<double>& J, double& out_s) {
        omp_set_num_threads(threads);
        ColumnCache cache; cache.resize(N, copt.n_z);
        const auto t0 = std::chrono::steady_clock::now();
        const bool ok = slim_coupled_reduced_jacobian(U, in, op, copt, cache, J);
        out_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return ok;
    };

    int fails = 0;
    double dt = 0.0;

    // -------- (2) Residual: serial (1) vs parallel (max) --------
    G("### (2) Residual R: serial vs parallel ###\n");
    std::vector<double> R_ser, R_par;
    const bool rs_ok = build_R(1, R_ser, dt);
    const bool rp_ok = build_R(max_threads, R_par, dt);
    if (!rs_ok || !rp_ok) {
        G("  base INFEASIBLE (rs_ok=%d rp_ok=%d) — cannot gate residual.\n", rs_ok, rp_ok);
        ++fails;
    } else {
        double ma, mr; int nm; vec_diff(R_ser, R_par, ma, mr, nm);
        G("  |R| = %d rows;  mismatched entries = %d;  max|abs| = %.3e;  max rel = %.3e\n",
          (int)R_ser.size(), nm, ma, mr);
        const bool pass = (mr <= 1e-12);
        G("  => %s (require max-rel <= 1e-12)\n\n", pass ? "PASS" : "FAIL");
        if (!pass) ++fails;
    }

    // -------- (3) Jacobian: serial (1) vs parallel (max) + TIMING --------
    G("### (3) Jacobian J: serial vs parallel (with timing) ###\n");
    std::vector<double> J_ser, J_par;
    double t_ser = 0.0, t_par = 0.0;
    const bool js_ok = build_J(1, J_ser, t_ser);
    const bool jp_ok = build_J(max_threads, J_par, t_par);
    if (!js_ok || !jp_ok) {
        G("  base INFEASIBLE (js_ok=%d jp_ok=%d) — cannot gate Jacobian.\n", js_ok, jp_ok);
        ++fails;
    } else {
        double ma, mr; int nm; vec_diff(J_ser, J_par, ma, mr, nm);
        G("  J = %dx%d = %d entries;  mismatched = %d;  max|abs| = %.3e;  max rel = %.3e\n",
          n, n, (int)J_ser.size(), nm, ma, mr);
        const bool pass = (mr <= 1e-12);
        G("  => %s (require max-rel <= 1e-12; expect BIT-IDENTICAL / 0)\n\n",
          pass ? "PASS" : "FAIL");
        if (!pass) ++fails;

        // -------- (4) Determinism: parallel vs parallel --------
        G("### (4) Determinism: parallel build twice ###\n");
        std::vector<double> J_par2; double t2 = 0.0;
        build_J(max_threads, J_par2, t2);
        double ma2, mr2; int nm2; vec_diff(J_par, J_par2, ma2, mr2, nm2);
        G("  mismatched = %d;  max|abs| = %.3e;  max rel = %.3e\n", nm2, ma2, mr2);
        const bool detpass = (mr2 <= 1e-12);
        G("  => %s (deterministic)\n\n", detpass ? "PASS" : "FAIL");
        if (!detpass) ++fails;

        // -------- (5) Speedup, COLD cache (from the step-3 timed builds) --------
        G("### (5) Speedup, COLD cache (one Jacobian build) ###\n");
        G("  serial (1 thread)      : %.3f s\n", t_ser);
        G("  parallel (%d threads)  : %.3f s\n", max_threads, t_par);
        G("  speedup                : %.2fx\n\n", (t_par > 0.0) ? t_ser / t_par : 0.0);

        // -------- (6) Speedup, WARM cache (representative of the relax loop) --------
        // The real relax reuses ONE warm ColumnCache across outer iterations, so per-iter
        // each column converges in a few polish steps (more uniform per-node cost) — closer
        // to the steady-state parallel scaling than the cold first solve.  Warm ONE cache,
        // then time serial vs parallel builds that each start from a COPY of that warm state.
        G("### (6) Speedup, WARM cache (representative per-iteration) ###\n");
        ColumnCache warm; warm.resize(N, copt.n_z);
        { std::vector<double> j0; omp_set_num_threads(max_threads);
          slim_coupled_reduced_jacobian(U, in, op, copt, warm, j0); }   // warm-up pass
        std::vector<double> jw;
        omp_set_num_threads(1);
        ColumnCache cw1 = warm;
        auto tw0 = std::chrono::steady_clock::now();
        slim_coupled_reduced_jacobian(U, in, op, copt, cw1, jw);
        const double tw_ser = std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count();
        omp_set_num_threads(max_threads);
        ColumnCache cw2 = warm;
        tw0 = std::chrono::steady_clock::now();
        slim_coupled_reduced_jacobian(U, in, op, copt, cw2, jw);
        const double tw_par = std::chrono::duration<double>(std::chrono::steady_clock::now() - tw0).count();
        G("  serial (1 thread)      : %.3f s\n", tw_ser);
        G("  parallel (%d threads)  : %.3f s\n", max_threads, tw_par);
        G("  speedup                : %.2fx\n\n", (tw_par > 0.0) ? tw_ser / tw_par : 0.0);
    }

    G("=== GATE %s (%d check(s) failed) ===\n", fails == 0 ? "PASS" : "FAIL", fails);
    return fails == 0 ? 0 : 1;
#endif
}
