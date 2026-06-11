// ===========================================================================
// TEMPORARY DIAGNOSTIC PROBE  (NOT a fix; safe to delete)
// ---------------------------------------------------------------------------
// WARM-START Mdot-CONTINUATION SWEEP (fixed spin a=0.9).  Tests whether a
// warm-start f_Edd continuation crosses the Eddington-fraction regularity wall
// where a COLD seed fails (cold converges to f_Edd=0.10, fails f_Edd>=0.20 in
// the sonic-point regularity group with r_sonic FROZEN at 2.274).
//
// PROCEDURE (BYPASSES the auto spin-walk / solve_slim_disk_radial):
//   1. Cold-seed + solve_single_am at (a=0.9, f_Edd=0.10) -> converged anchor U.
//   2. WARM-START continuation UP the f_Edd ladder, carrying U from each
//      converged rung to the next (do NOT cold-seed each point):
//        in.mdot = f_Edd * Mdot_Edd;
//        ok = solve_single_am(in, op, U_warm, /*require_N1=*/false);
//      If a rung FAILS, HALVE the f_Edd step from the last good rung and retry
//      (floor Delta f_Edd >= 0.01) to localize the wall precisely.
//   3. Per rung record: f_Edd, converged?, merit + groups, r_sonic (does the
//      free node TRACK off 2.274?), budget-trip?, wall; on convergence the
//      physics: max H/r, beta range, f_adv range, peak T_c/Sigma.
//
// A per-rung SolveBudget is installed via the file-scope g_budget pointer so the
// tight per-rung wall cap actually trips mid-solve (the cold-seed tool did not
// install it; it only measured wall after the fact). Total ~20-min outer guard.
//
// #includes slim_disk_radial.cpp + opacity.cpp directly to reach internal
// helpers, exactly like tools/slim_coldseed_sweep.cpp.
//
// Build:
//   cmake --build build --config Release --target slim-warmstart-sweep
//   build/Release/slim-warmstart-sweep.exe
// ===========================================================================

#define GRRT_EXPORT
#define GRRT_BUILDING_DLL 1

#include "../src/opacity.cpp"
#include "../src/slim_disk_radial.cpp"

#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>

using namespace grrt;
using namespace grrt::slim_detail;

namespace grrt {
namespace probe {

// Build SlimDiskInputs at fixed spin a and a given f_Edd. Mirrors the cold-seed
// tool's make_inputs exactly so the two sweeps are directly comparable.
static SlimDiskInputs make_inputs(double a, double f_Edd, int N, double wall_s,
                                  double& Mdot_Edd_out) {
    using namespace constants;
    SlimDiskInputs in{};
    in.mass = 1.0;
    in.spin = a;
    in.alpha = 0.1;
    in.r_g = 1.48e6;
    in.r_out = 50.0;
    in.n_nodes = N;
    in.max_iters = 800;
    in.tol = 1e-6;
    in.budget_wall_seconds = wall_s;
    const double r_ph = 2.0 * (1.0 + std::cos((2.0/3.0) * std::acos(-a)));
    in.r_in = r_ph + 0.02;
    const double M_cgs = in.mass * in.r_g * c_cgs * c_cgs / G_cgs;
    const double kappa_es = 0.34;
    const double L_Edd = 4.0 * std::numbers::pi * G_cgs * M_cgs * c_cgs / kappa_es;
    const double Mdot_Edd = 10.0 * L_Edd / (c_cgs * c_cgs);
    Mdot_Edd_out = Mdot_Edd;
    in.mdot = f_Edd * Mdot_Edd;
    return in;
}

static double active_merit(const std::vector<double>& U, const SlimDiskInputs& in,
                           const OpacityLUTs& op) {
    std::vector<double> R;
    slim_radial_residual(U, in, op, R);
    return slim_scaled_residual_norm_active(U, R, in);
}

// Result of a single warm-start rung attempt.
struct RungResult {
    bool ok = false;
    double merit = 0;
    std::string dom;
    GroupMags g{};
    double r_sonic = 0;
    bool budget_trip = false;
    double wall = 0;
    // physics (valid only if ok):
    double maxHr = 0, fadv_lo = 0, fadv_hi = 0, peakTc = 0, peakSig = 0;
    double beta_lo = 0, beta_hi = 0;
};

static const char* dominant_group(const GroupMags& g, double& domv) {
    struct GE { const char* n; double v; };
    GE arr[6] = {{"mass",g.mass},{"ang",g.ang},{"rad",g.rad},
                 {"ene",g.ene},{"bc",g.bc},{"reg",g.reg}};
    const char* dom = arr[0].n; domv = arr[0].v;
    for (int i = 1; i < 6; ++i) if (arr[i].v > domv) { domv = arr[i].v; dom = arr[i].n; }
    return dom;
}

// Run one WARM-START rung at (a, f_Edd) carrying U_warm (in-out). Installs a
// per-rung SolveBudget so the wall cap actually trips mid-solve.
static RungResult rung_warm(const OpacityLUTs& op, double a, double f_Edd, int N,
                            double wall_s, std::vector<double>& U_warm) {
    using namespace constants;
    RungResult rr;
    double Mdot_Edd = 0;
    SlimDiskInputs in = make_inputs(a, f_Edd, N, wall_s, Mdot_Edd);
    const int NN = std::max(in.n_nodes, 4);
    const double r_isco = isco_prograde(in.mass, in.spin);

    std::printf("\n-------------------- WARM rung f_Edd=%.4f  (a=%.3f, N=%d) --------------------\n",
                f_Edd, a, N);
    std::printf("  mdot=%.4e g/s  Mdot_Edd=%.4e  r_in=%.4f  r_isco=%.5f  r_sonic_in=%.5f\n",
                in.mdot, Mdot_Edd, in.r_in, r_isco, U_warm[4*NN+1]);

    // Trial copy: only commit U_warm forward if the rung converges (so a failed
    // rung does not corrupt the warm state used by the halving retry).
    std::vector<double> U = U_warm;

    // Install a per-rung budget so relax_structure trips the wall cap mid-solve.
    SolveBudget budget;
    budget.wall_cap_s = wall_s;
    struct BudgetGuard { ~BudgetGuard() { g_budget = nullptr; } } guard;
    g_budget = &budget;

    auto t0 = std::chrono::steady_clock::now();
    const bool ok = solve_single_am(in, op, U, /*require_N1=*/false);
    auto t1 = std::chrono::steady_clock::now();
    g_budget = nullptr;
    rr.wall = std::chrono::duration<double>(t1 - t0).count();
    rr.budget_trip = budget.tripped || (rr.wall >= wall_s * 0.97);
    rr.ok = ok;

    std::vector<double> R;
    slim_radial_residual(U, in, op, R);
    rr.merit = slim_scaled_residual_norm_active(U, R, in);
    rr.g = slim_group_mags(U, R, in);
    rr.r_sonic = U[4*NN+1];
    double domv = 0;
    rr.dom = dominant_group(rr.g, domv);

    const ValidityResult v = slim_validity_gate(in, op, U, /*require_N1=*/false);

    std::printf("  [SOLVE] ok=%d  wall=%.2fs%s\n", (int)ok, rr.wall,
                rr.budget_trip ? "  <<BUDGET-TRIP>>" : "");
    std::printf("  [MERIT] final=%.4e  dominant=%s(%.4e)\n", rr.merit, rr.dom.c_str(), domv);
    std::printf("  [GROUP] mass=%.4e ang=%.4e rad=%.4e ene=%.4e bc=%.4e reg=%.4e\n",
                rr.g.mass, rr.g.ang, rr.g.rad, rr.g.ene, rr.g.bc, rr.g.reg);
    std::printf("  [GATE]  mass_ok=%d(maxrel=%.4e) sign_ok=%d regD0_ok=%d(%.4e) "
                "rs_ok=%d smooth_ok=%d(jump=%.3f)  r_sonic=%.5f r_isco=%.5f\n",
                (int)v.mass_ok, v.mass_maxrel, (int)v.sign_ok, (int)v.reg_D0_ok,
                v.D0_scaled, (int)v.rs_ok, (int)v.smooth_ok, v.sigma_max_jump,
                v.r_s, v.r_isco);

    if (ok) {
        SlimDiskRadial prof;
        unpack_profile(in, op, U, prof);
        bool first = true;
        for (size_t i = 0; i < prof.r.size(); ++i) {
            const double r = prof.r[i];
            const double Hr = prof.H[i] / (r * in.r_g);
            if (Hr > rr.maxHr) rr.maxHr = Hr;
            if (prof.Tc[i] > rr.peakTc) rr.peakTc = prof.Tc[i];
            if (prof.Sigma[i] > rr.peakSig) rr.peakSig = prof.Sigma[i];
            const OneZoneState oz = one_zone_closure(
                std::max(prof.Sigma[i], kSigmaFloor),
                std::max(prof.Tc[i], kTFloor), r, in, op);
            const double beta = oz.p_gas / std::max(oz.p_mid, 1e-300);
            if (first) {
                rr.fadv_lo = rr.fadv_hi = prof.f_adv[i];
                rr.beta_lo = rr.beta_hi = beta;
                first = false;
            } else {
                rr.fadv_lo = std::min(rr.fadv_lo, prof.f_adv[i]);
                rr.fadv_hi = std::max(rr.fadv_hi, prof.f_adv[i]);
                rr.beta_lo = std::min(rr.beta_lo, beta);
                rr.beta_hi = std::max(rr.beta_hi, beta);
            }
        }
        std::printf("  [PHYS]  maxH/r=%.4f  f_adv=[%.4e,%.4e]  peakTc=%.4e K  "
                    "peakSig=%.4e g/cm2  beta=[%.4e,%.4e]\n",
                    rr.maxHr, rr.fadv_lo, rr.fadv_hi, rr.peakTc, rr.peakSig,
                    rr.beta_lo, rr.beta_hi);
        // Commit converged warm state forward for the next rung.
        U_warm = U;
    }

    std::printf("  [ROW] f_Edd=%.4f a=%.3f ok=%d merit=%.4e dom=%s rsonic=%.5f "
                "risco=%.5f wall=%.2f budget_trip=%d mass=%.3e ang=%.3e rad=%.3e "
                "ene=%.3e bc=%.3e reg=%.3e maxrel=%.3e gate_regD0=%d gate_rs=%d "
                "maxHr=%.4f fadv_lo=%.3e fadv_hi=%.3e peakTc=%.3e peakSig=%.3e "
                "beta_lo=%.3e beta_hi=%.3e\n",
                f_Edd, a, (int)ok, rr.merit, rr.dom.c_str(), rr.r_sonic, r_isco,
                rr.wall, (int)rr.budget_trip, rr.g.mass, rr.g.ang, rr.g.rad,
                rr.g.ene, rr.g.bc, rr.g.reg, v.mass_maxrel, (int)v.reg_D0_ok,
                (int)v.rs_ok, rr.maxHr, rr.fadv_lo, rr.fadv_hi, rr.peakTc,
                rr.peakSig, rr.beta_lo, rr.beta_hi);
    std::fflush(stdout);
    return rr;
}

} // namespace probe
} // namespace grrt

int main(int argc, char** argv) {
    using namespace grrt;
    (void)argc; (void)argv;
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    const int N = 48;
    const double a = 0.9;
    const double wall_s = 300.0;         // generous per-rung wall cap (5 min): let the FD
                                         // re-balance finish instead of cutting it off mid-descent
    const double total_guard_s = 45.0 * 60.0;   // bounded total exploration (45 min)
    const auto t_start = std::chrono::steady_clock::now();
    auto elapsed = [&]() {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
    };

    std::printf("# WARM-START f_Edd CONTINUATION  a=%.3f N=%d wall_cap=%.0fs total_guard=%.0fs\n",
                a, N, wall_s, total_guard_s);

    // ---- Step 1: cold-seed anchor at f_Edd=0.10 ----
    const double f_anchor = 0.10;
    double Mdot_Edd = 0;
    SlimDiskInputs in0 = grrt::probe::make_inputs(a, f_anchor, N, wall_s, Mdot_Edd);
    const int NN = std::max(in0.n_nodes, 4);
    std::vector<double> U = build_thin_disk_seed(in0, op);
    std::printf("\n==================== ANCHOR: COLD-SEED + SOLVE at f_Edd=%.3f ====================\n",
                f_anchor);
    {
        SolveBudget budget;
        budget.wall_cap_s = wall_s;
        g_budget = &budget;
        auto ta = std::chrono::steady_clock::now();
        const bool ok = solve_single_am(in0, op, U, /*require_N1=*/false);
        auto tb = std::chrono::steady_clock::now();
        g_budget = nullptr;
        const double wall = std::chrono::duration<double>(tb - ta).count();
        std::vector<double> R; slim_radial_residual(U, in0, op, R);
        const double merit = slim_scaled_residual_norm_active(U, R, in0);
        const GroupMags g = slim_group_mags(U, R, in0);
        double domv = 0; const char* dom = grrt::probe::dominant_group(g, domv);
        std::printf("  [ANCHOR] ok=%d merit=%.4e dom=%s r_sonic=%.5f wall=%.2fs%s\n",
                    (int)ok, merit, dom, U[4*NN+1], wall,
                    budget.tripped ? "  <<BUDGET-TRIP>>" : "");
        if (!ok) {
            std::printf("\n[probe] ANCHOR FAILED to converge — cannot warm-start. Aborting.\n");
            return 1;
        }
    }

    // ---- Step 2: warm-start continuation ladder ----
    // Fine steps near the start (small Mdot bump = small mass perturbation = fast
    // re-balance, stays in basin), growing as we climb.
    const std::vector<double> ladder = {
        0.105, 0.11, 0.12, 0.13, 0.14, 0.15, 0.17, 0.19, 0.22, 0.25,
        0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90 };

    double f_last_good = f_anchor;
    std::vector<grrt::probe::RungResult> table;
    bool stop = false;

    for (size_t k = 0; k < ladder.size() && !stop; ++k) {
        double f_target = ladder[k];
        // Attempt the target; on failure, bisect toward the last good rung.
        double f_try = f_target;
        bool rung_ok = false;
        while (true) {
            if (elapsed() > total_guard_s) {
                std::printf("\n[probe] TOTAL-GUARD (%.0fs) exceeded — stopping continuation.\n",
                            total_guard_s);
                stop = true; break;
            }
            std::vector<double> U_save = U;   // preserve last-good in case of failure
            grrt::probe::RungResult rr = grrt::probe::rung_warm(op, a, f_try, N, wall_s, U);
            table.push_back(rr);
            if (rr.ok) {
                rung_ok = true;
                f_last_good = f_try;
                if (f_try >= f_target - 1e-9) break;   // reached the ladder target
                // Converged at an intermediate (halved) step — advance toward target.
                f_try = f_target;
                continue;
            }
            // FAILED: restore warm state, halve the step from last good.
            U = U_save;
            const double f_half = 0.5 * (f_last_good + f_try);
            const double df = f_half - f_last_good;
            if (df < 0.0025) {
                std::printf("\n[probe] Step floor (Delta f_Edd=%.4f < 0.01) reached at "
                            "f_try=%.4f after last-good f_Edd=%.4f — WALL LOCALIZED. Stopping.\n",
                            df, f_try, f_last_good);
                stop = true; break;
            }
            std::printf("  [HALVE] rung f_Edd=%.4f failed; retry at f_Edd=%.4f "
                        "(last good=%.4f)\n", f_try, f_half, f_last_good);
            f_try = f_half;
        }
        (void)rung_ok;
    }

    // ---- Summary table ----
    std::printf("\n# ====================== WARM-START CONTINUATION SUMMARY ======================\n");
    std::printf("# f_Edd     ok merit       dom   r_sonic  maxH/r  beta_lo    beta_hi    fadv_lo    fadv_hi    wall   btrip\n");
    // Reconstruct f_Edd per row: walk the same logic is messy; instead re-print
    // from stored rows using their recorded merit/groups. We stored f via order,
    // so print the [ROW] lines already emitted above; here give a compact recap.
    std::printf("# (see per-rung [ROW] lines above for full machine-readable data)\n");
    std::printf("# highest f_Edd that CONVERGED (warm-start) = %.4f\n", f_last_good);
    std::printf("# anchor (cold) converged at f_Edd=%.3f; cold-seed wall was f_Edd>=0.20\n", f_anchor);

    std::printf("\n[probe] done. total wall=%.1fs\n", elapsed());
    return 0;
}
