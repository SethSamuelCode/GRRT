// ===========================================================================
// TEMPORARY DIAGNOSTIC PROBE  (NOT a fix; safe to delete)
// ---------------------------------------------------------------------------
// COLD-SEED EDDINGTON-FRACTION SWEEP (fixed spin a=0.9).  Determines, with
// numbers, at what Eddington fraction f_Edd a FRESH (cold) thin-disk seed stops
// converging — to prove whether (and over what f_Edd range) warm-start
// Mdot-continuation is actually needed at a=0.9, rather than assuming it.
//
// For each f_Edd, BYPASSING the auto spin-walk / Mdot-continuation:
//   U  = build_thin_disk_seed(in@(a=0.9, f_Edd), op)        // fresh cold seed
//   ok = solve_single_am(in@(a=0.9, f_Edd), op, U, false)   // bracket+relax+gate
// Records: ok?, final inner merit + group breakdown, r_sonic, r_isco,
//          wall-time, budget-trip, validity-gate decomposition, and (on
//          convergence) PHYSICS: max H/r, f_adv range, peak T_c, peak Sigma,
//          and midplane beta = p_gas/p_mid range.
//
// #includes slim_disk_radial.cpp + opacity.cpp directly to reach internal
// helpers, exactly like tools/slim_spinwalk_probe.cpp.
//
// Build:
//   cmake --build build --config Release --target slim-coldseed-sweep
//   build/Release/slim-coldseed-sweep.exe                 (default f_Edd sweep)
//   build/Release/slim-coldseed-sweep.exe 0.4 0.6         (bisection override)
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

namespace grrt {
namespace probe {

// Build SlimDiskInputs at fixed spin a and a given f_Edd, small N for speed,
// with a tight per-solve wall budget. Mirrors make_inputs in slim_spinwalk_probe.
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

// Run one cold-seed solve at (a, f_Edd); print one summary line + group/gate/
// physics detail.
static void sweep_one(const OpacityLUTs& op, double a, double f_Edd, int N,
                      double wall_s) {
    using namespace constants;
    double Mdot_Edd = 0;
    SlimDiskInputs in = make_inputs(a, f_Edd, N, wall_s, Mdot_Edd);
    const int NN = std::max(in.n_nodes, 4);
    const double r_isco = isco_prograde(in.mass, in.spin);

    std::printf("\n==================== f_Edd=%.3f  (a=%.3f, N=%d) ====================\n",
                f_Edd, a, N);
    std::printf("  mdot=%.4e g/s  Mdot_Edd=%.4e  r_in=%.4f  r_isco=%.5f\n",
                in.mdot, Mdot_Edd, in.r_in, r_isco);

    // Fresh cold seed.
    std::vector<double> U = build_thin_disk_seed(in, op);
    const double m_seed = active_merit(U, in, op);
    const double rs_seed = U[4*NN+1];
    std::printf("  [SEED] merit=%.4e  r_sonic_seed=%.5f  ell_in_seed=%.6f\n",
                m_seed, rs_seed, U[4*NN+0]);

    // Solve (bracket + relax + validity gate), the same call the continuation rungs use.
    auto t0 = std::chrono::steady_clock::now();
    const bool ok = solve_single_am(in, op, U, /*require_N1=*/false);
    auto t1 = std::chrono::steady_clock::now();
    const double wall = std::chrono::duration<double>(t1 - t0).count();
    const bool budget_trip = (wall >= wall_s * 0.97);

    // Final state diagnostics.
    std::vector<double> R;
    slim_radial_residual(U, in, op, R);
    const double merit = slim_scaled_residual_norm_active(U, R, in);
    const GroupMags g = slim_group_mags(U, R, in);
    const double r_sonic = U[4*NN+1];

    // Validity-gate decomposition (require_N1=false, matching the solve gate).
    const ValidityResult v = slim_validity_gate(in, op, U, /*require_N1=*/false);

    // Dominant group.
    struct GE { const char* n; double v; };
    GE arr[6] = {{"mass",g.mass},{"ang",g.ang},{"rad",g.rad},
                 {"ene",g.ene},{"bc",g.bc},{"reg",g.reg}};
    const char* dom = arr[0].n; double domv = arr[0].v;
    for (int i = 1; i < 6; ++i) if (arr[i].v > domv) { domv = arr[i].v; dom = arr[i].n; }

    std::printf("  [SOLVE] ok=%d  wall=%.2fs%s\n", (int)ok, wall,
                budget_trip ? "  <<BUDGET-TRIP>>" : "");
    std::printf("  [MERIT] final=%.4e  dominant=%s(%.4e)\n", merit, dom, domv);
    std::printf("  [GROUP] mass=%.4e ang=%.4e rad=%.4e ene=%.4e bc=%.4e reg=%.4e\n",
                g.mass, g.ang, g.rad, g.ene, g.bc, g.reg);
    std::printf("  [GATE]  mass_ok=%d(maxrel=%.4e) sign_ok=%d regD0_ok=%d(%.4e) "
                "rs_ok=%d smooth_ok=%d(jump=%.3f)  r_sonic=%.5f r_isco=%.5f\n",
                (int)v.mass_ok, v.mass_maxrel, (int)v.sign_ok, (int)v.reg_D0_ok,
                v.D0_scaled, (int)v.rs_ok, (int)v.smooth_ok, v.sigma_max_jump,
                v.r_s, v.r_isco);

    // PHYSICS extraction on converged points: unpack the profile and report the
    // slim-disk diagnostics. max H/r = H[cm]/(r·r_g); f_adv range; peak T_c, peak
    // Sigma; and midplane beta = p_gas/p_mid (low beta => radiation-pressure dom).
    double maxHr = 0, fadv_min = 0, fadv_max = 0, peakTc = 0, peakSig = 0;
    double beta_min = 0, beta_max = 0;
    if (ok) {
        SlimDiskRadial prof;
        unpack_profile(in, op, U, prof);
        bool first = true;
        for (size_t i = 0; i < prof.r.size(); ++i) {
            const double r = prof.r[i];
            const double Hr = prof.H[i] / (r * in.r_g);
            if (Hr > maxHr) maxHr = Hr;
            if (prof.Tc[i] > peakTc) peakTc = prof.Tc[i];
            if (prof.Sigma[i] > peakSig) peakSig = prof.Sigma[i];
            // beta proxy via one_zone_closure (cheap; reuse solver closure).
            const OneZoneState oz = one_zone_closure(
                std::max(prof.Sigma[i], kSigmaFloor),
                std::max(prof.Tc[i], kTFloor), r, in, op);
            const double beta = oz.p_gas / std::max(oz.p_mid, 1e-300);
            if (first) {
                fadv_min = fadv_max = prof.f_adv[i];
                beta_min = beta_max = beta;
                first = false;
            } else {
                fadv_min = std::min(fadv_min, prof.f_adv[i]);
                fadv_max = std::max(fadv_max, prof.f_adv[i]);
                beta_min = std::min(beta_min, beta);
                beta_max = std::max(beta_max, beta);
            }
        }
        std::printf("  [PHYS]  maxH/r=%.4f  f_adv=[%.4e,%.4e]  peakTc=%.4e K  "
                    "peakSig=%.4e g/cm2  beta=[%.4e,%.4e]\n",
                    maxHr, fadv_min, fadv_max, peakTc, peakSig, beta_min, beta_max);
    }

    // Machine-readable one-liner for the summary table.
    std::printf("  [ROW] f_Edd=%.3f a=%.3f ok=%d merit=%.4e dom=%s rsonic=%.5f "
                "risco=%.5f wall=%.2f budget_trip=%d mass=%.3e ang=%.3e rad=%.3e "
                "ene=%.3e bc=%.3e reg=%.3e maxrel=%.3e gate_mass=%d gate_sign=%d "
                "gate_regD0=%d gate_rs=%d gate_smooth=%d maxHr=%.4f fadv_lo=%.3e "
                "fadv_hi=%.3e peakTc=%.3e peakSig=%.3e beta_lo=%.3e beta_hi=%.3e\n",
                f_Edd, a, (int)ok, merit, dom, r_sonic, r_isco, wall, (int)budget_trip,
                g.mass, g.ang, g.rad, g.ene, g.bc, g.reg, v.mass_maxrel,
                (int)v.mass_ok, (int)v.sign_ok, (int)v.reg_D0_ok, (int)v.rs_ok,
                (int)v.smooth_ok, maxHr, fadv_min, fadv_max, peakTc, peakSig,
                beta_min, beta_max);
    std::fflush(stdout);
}

} // namespace probe
} // namespace grrt

int main(int argc, char** argv) {
    using namespace grrt;
    auto op = build_opacity_luts(1e-14, 1e6, 3000.0, 1e8);

    int N = 48;
    double wall_s = 120.0;
    const double a = 0.9;

    // Default f_Edd sweep; can override via CLI args (e.g. for a bisection step).
    std::vector<double> fedds = {0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0};
    if (argc > 1) {
        fedds.clear();
        for (int i = 1; i < argc; ++i) fedds.push_back(std::atof(argv[i]));
    }

    std::printf("# COLD-SEED f_Edd SWEEP  a=%.3f N=%d wall_cap=%.0fs\n",
                a, N, wall_s);
    for (double f : fedds) grrt::probe::sweep_one(op, a, f, N, wall_s);

    std::printf("\n[probe] done.\n");
    return 0;
}
